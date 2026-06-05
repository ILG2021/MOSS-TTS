import argparse
import functools
import importlib.util
from pathlib import Path
import re
import time
import orjson
import os

import gradio as gr
import numpy as np
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor
import transformers

# Monkeypatch to fix Windows backslash issues in HF repo IDs. Remote processor
# code may round-trip "org/repo" through pathlib.Path and produce "org\repo".
def _patch_repo_id(pretrained_model_name_or_path):
    if isinstance(pretrained_model_name_or_path, (str, os.PathLike)):
        s = str(pretrained_model_name_or_path)
        if "\\" in s and not os.path.exists(s):
            return s.replace("\\", "/")
    return pretrained_model_name_or_path


def _patch_auto_from_pretrained(auto_cls):
    orig_method = auto_cls.from_pretrained
    orig_func = getattr(orig_method, "__func__", orig_method)

    @classmethod
    @functools.wraps(orig_func)
    def patched_method(cls, pretrained_model_name_or_path, *args, **kwargs):
        return orig_func(
            cls,
            _patch_repo_id(pretrained_model_name_or_path),
            *args,
            **kwargs,
        )

    auto_cls.from_pretrained = patched_method


for _auto_cls in (
    transformers.AutoConfig,
    transformers.AutoTokenizer,
    transformers.AutoProcessor,
    transformers.AutoModel,
):
    _patch_auto_from_pretrained(_auto_cls)


_ORIG_TORCHAUDIO_LOAD = torchaudio.load


def _load_audio_with_soundfile_fallback(
    filepath,
    frame_offset=0,
    num_frames=-1,
    normalize=True,
    channels_first=True,
    format=None,
    buffer_size=4096,
    backend=None,
):
    try:
        return _ORIG_TORCHAUDIO_LOAD(
            filepath,
            frame_offset=frame_offset,
            num_frames=num_frames,
            normalize=normalize,
            channels_first=channels_first,
            format=format,
            buffer_size=buffer_size,
            backend=backend,
        )
    except RuntimeError as exc:
        if "Could not load libtorchcodec" not in str(exc):
            raise

        try:
            import soundfile as sf
        except ImportError as sf_exc:
            raise RuntimeError(
                "torchaudio 无法加载 TorchCodec，且当前环境未安装 soundfile。"
                "请安装兼容的 FFmpeg/TorchCodec，或运行：pip install soundfile"
            ) from sf_exc

        start = max(int(frame_offset), 0)
        frames = -1 if num_frames is None else int(num_frames)
        stop = None if frames < 0 else start + frames
        wav, sample_rate = sf.read(
            os.fspath(filepath),
            start=start,
            stop=stop,
            dtype="float32",
            always_2d=True,
        )
        wav = torch.from_numpy(wav.T if channels_first else wav)
        return wav, sample_rate


torchaudio.load = _load_audio_with_soundfile_fallback

# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

MODEL_PATH = "OpenMOSS-Team/MOSS-TTS-v1.5"
DEFAULT_ATTN_IMPLEMENTATION = "auto"
DEFAULT_MAX_NEW_TOKENS = 4096
CONTINUATION_NOTICE = (
    "续写模式已启用。请确认输入文本开头包含参考音频对应的转写文本。"
)

MODE_CLONE = "克隆声音读新文本"
MODE_CONTINUE = "高级：只接续音频上下文"
MODE_CONTINUE_CLONE = "延续上一段语音并保持音色"
ZH_TOKENS_PER_CHAR = 3.098411951313033
EN_TOKENS_PER_CHAR = 0.8673376262755219
AUDIO_TOKENS_PER_SECOND = 12.5
REFERENCE_AUDIO_DIR = Path(__file__).resolve().parent.parent / "assets" / "audio"
EXAMPLE_TEXTS_JSONL_PATH = Path(__file__).resolve().parent.parent / "assets" / "text" / "moss_tts_example_texts.jsonl"
LANGUAGE_TAG_AUTO = "Auto (omit)"
LANGUAGE_TAG_CHOICES = [
    ("自动（省略）", LANGUAGE_TAG_AUTO),
    ("中文", "Chinese"),
    ("粤语", "Cantonese"),
    ("英语", "English"),
    ("阿拉伯语", "Arabic"),
    ("捷克语", "Czech"),
    ("丹麦语", "Danish"),
    ("荷兰语", "Dutch"),
    ("芬兰语", "Finnish"),
    ("法语", "French"),
    ("德语", "German"),
    ("希腊语", "Greek"),
    ("希伯来语", "Hebrew"),
    ("印地语", "Hindi"),
    ("匈牙利语", "Hungarian"),
    ("意大利语", "Italian"),
    ("日语", "Japanese"),
    ("韩语", "Korean"),
    ("马其顿语", "Macedonian"),
    ("马来语", "Malay"),
    ("波斯语", "Persian (Farsi)"),
    ("波兰语", "Polish"),
    ("葡萄牙语", "Portuguese"),
    ("罗马尼亚语", "Romanian"),
    ("俄语", "Russian"),
    ("西班牙语", "Spanish"),
    ("斯瓦希里语", "Swahili"),
    ("瑞典语", "Swedish"),
    ("他加禄语", "Tagalog"),
    ("泰语", "Thai"),
    ("土耳其语", "Turkish"),
    ("越南语", "Vietnamese"),
]
LANGUAGE_TAG_LABELS = {value: label for label, value in LANGUAGE_TAG_CHOICES}


def _parse_example_id(example_id: str) -> tuple[str, int] | None:
    matched = re.fullmatch(r"(zh|en)/(\d+)", (example_id or "").strip())
    if matched is None:
        return None
    return matched.group(1), int(matched.group(2))


def _resolve_reference_audio_path(language: str, index: int) -> Path | None:
    stem_candidates = [f"reference_{language}_{index}"]
    for stem in stem_candidates:
        for ext in (".wav", ".mp3"):
            audio_path = REFERENCE_AUDIO_DIR / f"{stem}{ext}"
            if audio_path.exists():
                return audio_path
    return None


def build_example_rows() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []

    with open(EXAMPLE_TEXTS_JSONL_PATH, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            sample = orjson.loads(line)
            parsed = _parse_example_id(sample.get("id", ""))
            if parsed is None:
                continue

            language, index = parsed
            text = str(sample.get("text", "")).strip()
            audio_path = _resolve_reference_audio_path(language, index)
            if audio_path is None:
                continue

            rows.append((sample['role'], str(audio_path), text))

    return rows


EXAMPLE_ROWS = build_example_rows()


@functools.lru_cache(maxsize=1)
def load_backend(model_path: str, device_str: str, attn_implementation: str):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    resolved_attn_implementation = resolve_attn_implementation(
        requested=attn_implementation,
        device=device,
        dtype=dtype,
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if resolved_attn_implementation:
        model_kwargs["attn_implementation"] = resolved_attn_implementation

    model = AutoModel.from_pretrained(model_path, **model_kwargs).to(device)
    model.eval()

    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
    return model, processor, device, sample_rate


def resolve_attn_implementation(requested: str, device: torch.device, dtype: torch.dtype) -> str | None:
    requested_norm = (requested or "").strip().lower()

    if requested_norm in {"none"}:
        return None

    if requested_norm not in {"", "auto"}:
        return requested

    # Prefer FlashAttention 2 when package + device conditions are met.
    if (
        device.type == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:
            return "flash_attention_2"

    # CUDA fallback: use PyTorch SDPA kernels.
    if device.type == "cuda":
        return "sdpa"

    # CPU fallback.
    return "eager"


def detect_text_language(text: str) -> str:
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_chars = len(re.findall(r"[A-Za-z]", text))
    if zh_chars == 0 and en_chars == 0:
        return "en"
    return "zh" if zh_chars >= en_chars else "en"


def supports_duration_control(mode_with_reference: str) -> bool:
    return mode_with_reference not in {MODE_CONTINUE, MODE_CONTINUE_CLONE}


def estimate_duration_tokens(text: str) -> tuple[str, int, int, int]:
    normalized = text or ""
    effective_len = max(len(normalized), 1)
    language = detect_text_language(normalized)
    factor = ZH_TOKENS_PER_CHAR if language == "zh" else EN_TOKENS_PER_CHAR
    default_tokens = max(1, int(effective_len * factor))
    min_tokens = max(1, int(default_tokens * 0.5))
    max_tokens = max(min_tokens + 1, int(default_tokens * 1.5))
    return language, default_tokens, min_tokens, max_tokens


def audio_tokens_to_seconds(tokens: int) -> float:
    return round(max(tokens, 1) / AUDIO_TOKENS_PER_SECOND, 1)


def seconds_to_audio_tokens(seconds: float | int) -> int:
    return max(1, int(round(float(seconds) * AUDIO_TOKENS_PER_SECOND)))


def update_duration_controls(
    enabled: bool,
    text: str,
    current_seconds: float | int | None,
    mode_with_reference: str,
):
    if not supports_duration_control(mode_with_reference):
        return (
            gr.update(visible=False),
            "续写模式下不可使用时长控制。",
            gr.update(value=False, interactive=False),
        )

    checkbox_update = gr.update(interactive=True)
    if not enabled:
        return gr.update(visible=False), "时长控制未启用。", checkbox_update

    language, default_tokens, min_tokens, max_tokens = estimate_duration_tokens(text)
    default_seconds = audio_tokens_to_seconds(default_tokens)
    min_seconds = max(0.5, audio_tokens_to_seconds(min_tokens))
    max_seconds = max(min_seconds + 0.5, audio_tokens_to_seconds(max_tokens))

    # Slider is initialized with value=1 as a placeholder; treat it as unset
    # so first-time estimation uses the computed default.
    if current_seconds is None or float(current_seconds) == 1.0:
        slider_value = default_seconds
    else:
        slider_value = float(current_seconds)
        slider_value = max(min_seconds, min(max_seconds, slider_value))

    language_label = "中文" if language == "zh" else "英文"
    hint = (
        f"时长控制已启用 | 检测语言：{language_label} | "
        f"建议时长≈{default_seconds:.1f} 秒，范围≈{min_seconds:.1f}-{max_seconds:.1f} 秒"
    )
    return (
        gr.update(
            visible=True,
            minimum=min_seconds,
            maximum=max_seconds,
            value=slider_value,
            step=0.5,
        ),
        hint,
        checkbox_update,
    )


def normalize_language_tag(language_tag: str | None) -> str | None:
    language_tag = (language_tag or "").strip()
    if not language_tag or language_tag == LANGUAGE_TAG_AUTO:
        return None
    return language_tag


def build_conversation(
    text: str,
    reference_audio: str | None,
    mode_with_reference: str,
    expected_tokens: int | None,
    language_tag: str | None,
    processor,
):
    text = (text or "").strip()
    if not text:
        raise ValueError("请输入要合成的文本。")

    user_kwargs = {"text": text}
    normalized_language = normalize_language_tag(language_tag)
    if normalized_language is not None:
        user_kwargs["language"] = normalized_language
    if expected_tokens is not None:
        user_kwargs["tokens"] = int(expected_tokens)

    if not reference_audio:
        conversations = [[processor.build_user_message(**user_kwargs)]]
        return conversations, "generation", "直接生成"

    if mode_with_reference == MODE_CLONE:
        clone_kwargs = dict(user_kwargs)
        clone_kwargs["reference"] = [reference_audio]
        conversations = [[processor.build_user_message(**clone_kwargs)]]
        return conversations, "generation", MODE_CLONE

    if mode_with_reference == MODE_CONTINUE:
        conversations = [
            [
                processor.build_user_message(**user_kwargs),
                processor.build_assistant_message(audio_codes_list=[reference_audio]),
            ]
        ]
        return conversations, "continuation", MODE_CONTINUE

    continue_clone_kwargs = dict(user_kwargs)
    continue_clone_kwargs["reference"] = [reference_audio]
    conversations = [
        [
            processor.build_user_message(**continue_clone_kwargs),
            processor.build_assistant_message(audio_codes_list=[reference_audio]),
        ]
    ]
    return conversations, "continuation", MODE_CONTINUE_CLONE


def render_mode_hint(reference_audio: str | None, mode_with_reference: str):
    if not reference_audio:
        return "当前模式：**直接生成**（未上传参考音频）"
    if mode_with_reference == MODE_CLONE:
        return "当前模式：**克隆声音读新文本**（参考音频只作为声音样本，不会接着它的内容往下说）"
    return f"当前模式：**{mode_with_reference}**  \n> {CONTINUATION_NOTICE}"


def apply_example_selection(
    mode_with_reference: str,
    duration_control_enabled: bool,
    duration_seconds: float,
    evt: gr.SelectData,
):
    if evt is None or evt.index is None:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    if isinstance(evt.index, (tuple, list)):
        row_idx = int(evt.index[0])
    else:
        row_idx = int(evt.index)

    if row_idx < 0 or row_idx >= len(EXAMPLE_ROWS):
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    _, audio_path, example_text = EXAMPLE_ROWS[row_idx]
    duration_slider_update, duration_hint, duration_checkbox_update = update_duration_controls(
        duration_control_enabled,
        example_text,
        duration_seconds,
        mode_with_reference,
    )
    return (
        audio_path,
        example_text,
        render_mode_hint(audio_path, mode_with_reference),
        duration_slider_update,
        duration_hint,
        duration_checkbox_update,
    )


def run_inference(
    text: str,
    reference_audio: str | None,
    mode_with_reference: str,
    duration_control_enabled: bool,
    duration_seconds: float,
    language_tag: str | None,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    model_path: str,
    device: str,
    attn_implementation: str,
    max_new_tokens: int,
):
    started_at = time.monotonic()
    model, processor, torch_device, sample_rate = load_backend(
        model_path=model_path,
        device_str=device,
        attn_implementation=attn_implementation,
    )
    duration_enabled = bool(duration_control_enabled and supports_duration_control(mode_with_reference))
    expected_tokens = seconds_to_audio_tokens(duration_seconds) if duration_enabled else None
    expected_duration = float(duration_seconds) if duration_enabled else None
    conversations, mode, mode_name = build_conversation(
        text=text,
        reference_audio=reference_audio,
        mode_with_reference=mode_with_reference,
        expected_tokens=expected_tokens,
        language_tag=language_tag,
        processor=processor,
    )

    batch = processor(conversations, mode=mode)
    input_ids = batch["input_ids"].to(torch_device)
    attention_mask = batch["attention_mask"].to(torch_device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            audio_temperature=float(temperature),
            audio_top_p=float(top_p),
            audio_top_k=int(top_k),
            audio_repetition_penalty=float(repetition_penalty),
        )

    messages = processor.decode(outputs)
    if not messages or messages[0] is None:
        raise RuntimeError("模型没有返回可解码的音频结果。")

    audio = messages[0].audio_codes_list[0]
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().float().cpu().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)

    if audio_np.ndim > 1:
        audio_np = audio_np.reshape(-1)
    audio_np = audio_np.astype(np.float32, copy=False)

    elapsed = time.monotonic() - started_at
    normalized_language = normalize_language_tag(language_tag)
    language_label = LANGUAGE_TAG_LABELS.get(normalized_language, "自动")
    expected_duration_label = (
        f"{expected_duration:.1f} 秒" if expected_duration is not None else "关闭"
    )
    status = (
        f"完成 | 模式：{mode_name} | 语言：{language_label} | "
        f"耗时：{elapsed:.2f}s | "
        f"最大生成长度={int(max_new_tokens)}, "
        f"预期时长={expected_duration_label}, "
    )
    status += (
        f"温度={float(temperature):.2f}, Top-p={float(top_p):.2f}, "
        f"Top-k={int(top_k)}, 重复惩罚={float(repetition_penalty):.2f}"
    )
    return (sample_rate, audio_np), status


def build_demo(args: argparse.Namespace):
    custom_css = """
    :root {
      --bg: #f6f7f8;
      --panel: #ffffff;
      --ink: #111418;
      --muted: #4d5562;
      --line: #e5e7eb;
      --accent: #0f766e;
    }
    .gradio-container {
      background: linear-gradient(180deg, #f7f8fa 0%, #f3f5f7 100%);
      color: var(--ink);
    }
    .app-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel);
      padding: 14px;
    }
    .app-title {
      font-size: 22px;
      font-weight: 700;
      margin-bottom: 6px;
      letter-spacing: 0.2px;
    }
    .app-subtitle {
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 8px;
    }
    #output_audio {
      padding-bottom: 12px;
      margin-bottom: 8px;
      overflow: hidden !important;
    }
    #output_audio > .wrap {
      overflow: hidden !important;
    }
    #output_audio audio {
      margin-bottom: 6px;
    }
    #run-btn {
      background: var(--accent);
      border: none;
    }
    """

    with gr.Blocks(title="", css=custom_css) as demo:
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="文本",
                    lines=9,
                    placeholder="请输入要合成的文本。使用续写模式时，请在开头写入参考音频的转写文本。",
                )
                reference_audio = gr.Audio(
                    label="参考音频（可选）",
                    type="filepath",
                )
                mode_with_reference = gr.Radio(
                    choices=[MODE_CLONE, MODE_CONTINUE, MODE_CONTINUE_CLONE],
                    value=MODE_CLONE,
                    label="参考音频模式",
                    info="未上传参考音频时，将自动使用直接生成。",
                )
                gr.Markdown(
                    """
                    **不知道选哪个？看这里**

                    - **克隆声音读新文本**：上传一段声音样本，让它用这个声音朗读你输入的新文本。最常用。
                    - **高级：只接续音频上下文**：把参考音频当作前面已经说过的内容，让模型继续往后说；不特别强调声音必须一致。
                    - **延续上一段语音并保持音色**：参考音频是已经说过的前半段，模型会顺着它继续往后说，并尽量保持同一个声音。

                    简单理解：只想换声音读新文本，选第一个；想接着一段音频继续说，并且声音别变，选第三个。
                    """
                )
                mode_hint = gr.Markdown(render_mode_hint(None, MODE_CLONE))
                language_tag = gr.Dropdown(
                    choices=LANGUAGE_TAG_CHOICES,
                    value="Chinese",
                    label="语言标签",
                    info="可选。已知输入语言时建议设置，尤其是中文和英文以外的语言。",
                )
                duration_control_enabled = gr.Checkbox(
                    value=False,
                    label="启用时长控制（按秒估算）",
                )
                duration_seconds = gr.Slider(
                    minimum=1,
                    maximum=2,
                    step=0.5,
                    value=1.0,
                    label="预计音频时长（秒）",
                    visible=False,
                )
                duration_hint = gr.Markdown("时长控制未启用。")

                with gr.Accordion("采样参数（音频）", open=True):
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=3.0,
                        step=0.05,
                        value=1.7,
                        label="温度",
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        step=0.01,
                        value=0.8,
                        label="Top-p",
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=200,
                        step=1,
                        value=25,
                        label="Top-k",
                    )
                    repetition_penalty = gr.Slider(
                        minimum=0.8,
                        maximum=2.0,
                        step=0.05,
                        value=1.0,
                        label="重复惩罚",
                    )
                    max_new_tokens = gr.Slider(
                        minimum=256,
                        maximum=8192,
                        step=128,
                        value=DEFAULT_MAX_NEW_TOKENS,
                        label="最大生成长度（高级）",
                    )

                run_btn = gr.Button("生成语音", variant="primary", elem_id="run-btn")

            with gr.Column(scale=2):
                output_audio = gr.Audio(label="输出音频", type="numpy", elem_id="output_audio")
                status = gr.Textbox(label="状态", lines=4, interactive=False)
                examples_table = gr.Dataframe(
                    headers=["参考音色", "示例文本"],
                    value=[[name, text] for name, _, text in EXAMPLE_ROWS],
                    datatype=["str", "str"],
                    row_count=(len(EXAMPLE_ROWS), "fixed"),
                    col_count=(2, "fixed"),
                    interactive=False,
                    wrap=True,
                    label="示例（点击一行填入输入）",
                )

        reference_audio.change(
            fn=render_mode_hint,
            inputs=[reference_audio, mode_with_reference],
            outputs=[mode_hint],
        )
        mode_with_reference.change(
            fn=render_mode_hint,
            inputs=[reference_audio, mode_with_reference],
            outputs=[mode_hint],
        )
        duration_control_enabled.change(
            fn=update_duration_controls,
            inputs=[duration_control_enabled, text, duration_seconds, mode_with_reference],
            outputs=[duration_seconds, duration_hint, duration_control_enabled],
        )
        text.change(
            fn=update_duration_controls,
            inputs=[duration_control_enabled, text, duration_seconds, mode_with_reference],
            outputs=[duration_seconds, duration_hint, duration_control_enabled],
        )
        mode_with_reference.change(
            fn=update_duration_controls,
            inputs=[duration_control_enabled, text, duration_seconds, mode_with_reference],
            outputs=[duration_seconds, duration_hint, duration_control_enabled],
        )
        examples_table.select(
            fn=apply_example_selection,
            inputs=[mode_with_reference, duration_control_enabled, duration_seconds],
            outputs=[
                reference_audio,
                text,
                mode_hint,
                duration_seconds,
                duration_hint,
                duration_control_enabled,
            ],
        )

        run_btn.click(
            fn=lambda text, reference_audio, mode_with_reference, duration_control_enabled, duration_seconds, language_tag, temperature, top_p, top_k, repetition_penalty, max_new_tokens: run_inference(
                text=text,
                reference_audio=reference_audio,
                mode_with_reference=mode_with_reference,
                duration_control_enabled=duration_control_enabled,
                duration_seconds=duration_seconds,
                language_tag=language_tag,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                model_path=args.model_path,
                device=args.device,
                attn_implementation=args.attn_implementation,
                max_new_tokens=max_new_tokens,
            ),
            inputs=[
                text,
                reference_audio,
                mode_with_reference,
                duration_control_enabled,
                duration_seconds,
                language_tag,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                max_new_tokens,
            ],
            outputs=[output_audio, status],
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description="MossTTS Gradio Demo")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn_implementation", type=str, default=DEFAULT_ATTN_IMPLEMENTATION)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    runtime_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runtime_dtype = torch.bfloat16 if runtime_device.type == "cuda" else torch.float32
    args.attn_implementation = resolve_attn_implementation(
        requested=args.attn_implementation,
        device=runtime_device,
        dtype=runtime_dtype,
    ) or "none"
    print(f"[INFO] Using attn_implementation={args.attn_implementation}", flush=True)

    # Preload model/processor at startup to avoid first-request cold start latency.
    preload_started_at = time.monotonic()
    print(
        f"[Startup] Preloading backend: model={args.model_path}, device={args.device}, attn={args.attn_implementation}",
        flush=True,
    )
    load_backend(
        model_path=args.model_path,
        device_str=args.device,
        attn_implementation=args.attn_implementation,
    )
    print(
        f"[Startup] Backend preload finished in {time.monotonic() - preload_started_at:.2f}s",
        flush=True,
    )

    demo = build_demo(args)
    demo.queue(max_size=16, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        root_path=args.root_path,
        share=args.share,
    )


if __name__ == "__main__":
    main()
