"""Gradio frontend for pwilkin/openmoss' persistent native server.

This module mirrors the UI layout, chunking, ASR transcription, and duration control
logic of clis/moss_tts_app.py, but uses the native C++ openmoss server (moss-tts-server)
for inference.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave

import gradio as gr
import numpy as np
import soundfile as sf

if __package__:
    from .tts_chunking import count_chars, iter_text_chunks, reference_tail, transcribe_reference
else:
    try:
        from tts_chunking import count_chars, iter_text_chunks, reference_tail, transcribe_reference
    except ImportError:
        from clis.tts_chunking import count_chars, iter_text_chunks, reference_tail, transcribe_reference

DEFAULT_MAX_NEW_TOKENS = 4096
CONTINUATION_NOTICE = (
    "参考文本由 faster-whisper large-v3-turbo 自动转录并拼接；只需输入待生成文本。"
)

MODE_CLONE = "克隆"
MODE_CONTINUE = "续写"
MODE_CONTINUE_CLONE = "续写 + 克隆"
ZH_TOKENS_PER_CHAR = 3.098411951313033
EN_TOKENS_PER_CHAR = 0.8673376262755219
LANGUAGE_TAG_AUTO = "自动 (缺省)"

LANGUAGE_TAG_MAP: dict[str, str | None] = {
    "自动 (缺省)": None,
    "中文 (Chinese)": "Chinese",
    "粤语 (Cantonese)": "Cantonese",
    "英语 (English)": "English",
    "日语 (Japanese)": "Japanese",
    "韩语 (Korean)": "Korean",
    "法语 (French)": "French",
    "德语 (German)": "German",
    "西班牙语 (Spanish)": "Spanish",
    "俄语 (Russian)": "Russian",
    "阿拉伯语 (Arabic)": "Arabic",
    "意大利语 (Italian)": "Italian",
    "葡萄牙语 (Portuguese)": "Portuguese",
    "印地语 (Hindi)": "Hindi",
    "泰语 (Thai)": "Thai",
    "越南语 (Vietnamese)": "Vietnamese",
    "土耳其语 (Turkish)": "Turkish",
    "荷兰语 (Dutch)": "Dutch",
    "波兰语 (Polish)": "Polish",
    "丹麦语 (Danish)": "Danish",
    "芬兰语 (Finnish)": "Finnish",
    "捷克语 (Czech)": "Czech",
    "希腊语 (Greek)": "Greek",
    "希伯来语 (Hebrew)": "Hebrew",
    "匈牙利语 (Hungarian)": "Hungarian",
    "马其顿语 (Macedonian)": "Macedonian",
    "马来语 (Malay)": "Malay",
    "波斯语 (Persian)": "Persian (Farsi)",
    "罗马尼亚语 (Romanian)": "Romanian",
    "斯瓦希里语 (Swahili)": "Swahili",
    "瑞典语 (Swedish)": "Swedish",
    "他加禄语 (Tagalog)": "Tagalog",
}
LANGUAGE_TAG_CHOICES = list(LANGUAGE_TAG_MAP.keys())


def detect_text_language(text: str) -> str:
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_chars = len(re.findall(r"[A-Za-z]", text))
    if zh_chars == 0 and en_chars == 0:
        return "en"
    return "zh" if zh_chars >= en_chars else "en"


def supports_duration_control(mode_with_reference: str) -> bool:
    return mode_with_reference not in {MODE_CONTINUE, MODE_CONTINUE_CLONE, "Continuation", "Continuation + Clone"}


def estimate_duration_tokens(text: str) -> tuple[str, int, int, int]:
    normalized = text or ""
    effective_len = max(len(normalized), 1)
    language = detect_text_language(normalized)
    factor = ZH_TOKENS_PER_CHAR if language == "zh" else EN_TOKENS_PER_CHAR
    default_tokens = max(1, int(effective_len * factor))
    min_tokens = max(1, int(default_tokens * 0.5))
    # Gradio Slider requires maximum to be strictly greater than minimum.
    max_tokens = max(min_tokens + 1, int(default_tokens * 1.5))
    return language, default_tokens, min_tokens, max_tokens


def update_duration_controls(
    enabled: bool,
    text: str,
    current_tokens: float | int | None,
    mode_with_reference: str,
    subsequent_mode: str | None = None,
):
    if not any(supports_duration_control(mode) for mode in
               (mode_with_reference, subsequent_mode or mode_with_reference)):
        return (
            gr.update(visible=False),
            "续写模式下不支持时长控制。",
            gr.update(value=False, interactive=False),
        )

    checkbox_update = gr.update(interactive=True)
    if not enabled:
        return gr.update(visible=False), "时长控制已关闭。", checkbox_update

    language, default_tokens, min_tokens, max_tokens = estimate_duration_tokens(text)
    if current_tokens is None or int(current_tokens) == 1:
        slider_value = default_tokens
    else:
        slider_value = int(current_tokens)
        slider_value = max(min_tokens, min(max_tokens, slider_value))

    language_label = "中文" if language == "zh" else "英文"
    hint = (
        f"已开启时长控制 | 识别到语言: {language_label} | "
        f"默认值={default_tokens}，范围=[{min_tokens}, {max_tokens}]"
    )
    return (
        gr.update(
            visible=True,
            minimum=min_tokens,
            maximum=max_tokens,
            value=slider_value,
            step=1,
        ),
        hint,
        checkbox_update,
    )


def normalize_language_tag(language_tag: str | None) -> str | None:
    language_tag = (language_tag or "").strip()
    if not language_tag or language_tag in {LANGUAGE_TAG_AUTO, "Auto (omit)", "Auto"}:
        return None
    if language_tag in LANGUAGE_TAG_MAP:
        return LANGUAGE_TAG_MAP[language_tag]
    return language_tag


def render_mode_hint(reference_audio: str | None, mode_with_reference: str):
    if not reference_audio:
        return "首段：**直接生成**（未上传参考音频）"
    if mode_with_reference in {MODE_CLONE, "Clone"}:
        return "首段：**克隆**（使用上传的参考音频）"
    return f"首段：**续写 + 克隆**  \n> {CONTINUATION_NOTICE}"


def _audio_to_wav_bytes(audio_path: str | Path) -> bytes:
    """Read any audio file and return valid 16-bit PCM RIFF/WAVE bytes."""
    p = Path(audio_path)
    with open(p, "rb") as f:
        head = f.read(12)
    if len(head) >= 12 and head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        with open(p, "rb") as f:
            return f.read()

    data, sr = sf.read(str(p), dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class OpenMossRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.base_url = f"http://{args.openmoss_host}:{args.openmoss_port}"
        self.process: subprocess.Popen | None = None
        self._start_lock = threading.Lock()
        self.adapters = [item.split("=", 1)[0] for item in args.lora]
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def resident(self) -> bool:
        return self._server_info() is not None

    def _server_info(self) -> dict | None:
        try:
            with urllib.request.urlopen(f"{self.base_url}/info", timeout=1) as response:
                info = json.load(response)
            if response.status == 200 and isinstance(info, dict) and "architecture" in info:
                return info
        except (OSError, ValueError, urllib.error.URLError):
            pass
        return None

    def _validate_server(self, info: dict) -> None:
        if info.get("architecture") != "moss_tts_delay":
            raise RuntimeError(
                f"{self.base_url} 已有非 moss_tts_delay 服务：{info.get('architecture')!r}"
            )
        if info.get("codec_cpu") is not self.args.codec_cpu:
            raise RuntimeError(
                f"{self.base_url} 的 codec_cpu={info.get('codec_cpu')!r}，"
                f"但当前 Gradio 请求 codec_cpu={self.args.codec_cpu!r}"
            )
        if self.adapters:
            advertised = info.get("lora_adapters")
            if not isinstance(advertised, list):
                raise RuntimeError(
                    f"{self.base_url} 的 server 不支持多 LoRA 信息；请使用本仓库编译的扩展版"
                )
            missing = sorted(set(self.adapters) - set(advertised))
            if missing:
                raise RuntimeError("server 未加载这些 LoRA：" + ", ".join(missing))

    def ensure_started(self) -> float | None:
        info = self._server_info()
        if info is not None:
            self._validate_server(info)
            return None
        with self._start_lock:
            info = self._server_info()
            if info is not None:
                self._validate_server(info)
                return None

            exe = Path(self.args.openmoss_server).expanduser().resolve()
            model = Path(self.args.model).expanduser().resolve()
            if not exe.is_file():
                raise FileNotFoundError(f"找不到 openmoss server：{exe}")
            if not model.is_file():
                raise FileNotFoundError(f"找不到 GGUF backbone：{model}")
            sidecar = model.with_suffix(".extras.gguf")
            if not sidecar.is_file():
                raise FileNotFoundError(
                    f"找不到匹配的 sidecar：{sidecar}\n"
                    "openmoss 要求 backbone.gguf 与 backbone.extras.gguf 同名并放在一起。"
                )

            command = [
                str(exe), "--model", str(model),
                "--host", self.args.openmoss_host,
                "--port", str(self.args.openmoss_port),
                "--main-gpu", str(self.args.main_gpu),
                "--n-gpu-layers", str(self.args.n_gpu_layers),
                "--n-ctx", str(self.args.n_ctx),
                "--n-batch", str(self.args.n_batch),
                "--cache-type-k", self.args.cache_type_k,
                "--cache-type-v", self.args.cache_type_v,
                "--parallel", str(self.args.parallel),
                "--no-webui",
            ]
            for adapter in self.args.lora:
                name, raw_path = adapter.split("=", 1)
                adapter_path = Path(raw_path).expanduser().resolve()
                if not adapter_path.is_file():
                    raise FileNotFoundError(f"找不到 LoRA '{name}'：{adapter_path}")
                command.extend(["--lora", f"{name}={adapter_path}"])
            if self.args.codec_cpu:
                command.append("--codec-cpu")
            if self.args.no_flash_attn:
                command.append("--no-flash-attn")

            env = os.environ.copy()
            dll_dirs = [
                exe.parent,
                exe.parent / "bin",
                exe.parent.parent / "bin" / "Release",
            ]
            env["PATH"] = os.pathsep.join(str(p) for p in dll_dirs) + os.pathsep + env.get("PATH", "")
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            started = time.monotonic()
            self.process = subprocess.Popen(
                command,
                cwd=str(exe.parent),
                env=env,
                creationflags=creationflags,
            )

            deadline = started + self.args.load_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"moss-tts-server 启动失败，退出码 {self.process.returncode}")
                info = self._server_info()
                if info is not None:
                    try:
                        self._validate_server(info)
                    except Exception:
                        self.stop()
                        raise
                    return time.monotonic() - started
                time.sleep(0.25)
            self.stop()
            raise TimeoutError(f"等待 openmoss 加载模型超过 {self.args.load_timeout} 秒")

    def run_single_chunk(
        self,
        text: str,
        reference_audio: str | None,
        mode: str,
        expected_tokens: int | None,
        language_tag: str | None,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        max_new_tokens: int,
        adapter: str = "Base",
    ) -> tuple[int, np.ndarray, float]:
        self.ensure_started()
        payload = {
            "text": text,
            "response_format": "wav",
            "max_new_tokens": int(max_new_tokens),
            "sampling": {
                "audio_temperature": float(temperature),
                "audio_top_p": float(top_p),
                "audio_top_k": int(top_k),
                "audio_repetition_penalty": float(repetition_penalty),
            },
        }
        normalized_language = normalize_language_tag(language_tag)
        if normalized_language is not None:
            payload["language"] = normalized_language
        if adapter and adapter not in {"Base", "Base (原版)"}:
            payload["lora"] = adapter
        if expected_tokens is not None and expected_tokens > 0:
            payload["token_count"] = int(expected_tokens)

        if reference_audio:
            wav_bytes = _audio_to_wav_bytes(reference_audio)
            payload["reference_wav_b64"] = base64.b64encode(wav_bytes).decode("ascii")
            if mode in {MODE_CONTINUE_CLONE, "Continuation + Clone"}:
                payload["continuation_prefix"] = True
            else:
                payload["continuation_prefix"] = False

        request = urllib.request.Request(
            f"{self.base_url}/tts",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.args.request_timeout) as response:
                audio_bytes = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"openmoss HTTP {exc.code}: {detail}") from exc

        elapsed = time.monotonic() - started
        audio_np, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1)
        audio_np = audio_np.astype(np.float32, copy=False)
        if audio_np.ndim != 1 or not audio_np.size or not np.isfinite(audio_np).all():
            raise RuntimeError("openmoss C++ 后端返回了无效或空的音频。")

        return sample_rate, audio_np, elapsed

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None


def run_inference(
    text: str,
    reference_audio: str | None,
    mode_with_reference: str,
    duration_control_enabled: bool,
    duration_tokens: int,
    language_tag: str | None,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    chunk_chars: int,
    subsequent_mode: str,
    adapter: str,
    runtime: OpenMossRuntime,
    asr_model: str,
    asr_device: str,
):
    if any(mode not in {MODE_CLONE, MODE_CONTINUE_CLONE, "Clone", "Continuation + Clone"}
           for mode in (mode_with_reference, subsequent_mode)):
        raise ValueError("分段模式仅支持克隆和续写+克隆")
    if chunk_chars is None or not float(chunk_chars).is_integer() or chunk_chars < 1:
        raise ValueError("分段字数必须为正整数")
    remaining = text or ""
    if not remaining.strip():
        raise ValueError("请输入待生成文本")

    started = time.monotonic()
    results, details = [], []
    current_reference = reference_audio
    uploaded_transcript = ""
    asr_language = normalize_language_tag(language_tag)
    temp_root = Path(__file__).resolve().parents[1] / "Temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix="moss-openmoss-rolling-", dir=temp_root)
    print(f"[Reference audio] {temporary}", flush=True)

    output_rate = 24000

    while remaining.strip():
        index = len(results)
        mode = mode_with_reference if index == 0 else subsequent_mode
        if index == 0 or mode in {MODE_CLONE, "Clone"}:
            current_reference = reference_audio
        transcript = ""
        continuation_mode = mode in {
            MODE_CONTINUE,
            MODE_CONTINUE_CLONE,
            "Continuation",
            "Continuation + Clone",
        }
        if current_reference and continuation_mode:
            try:
                transcript = transcribe_reference(
                    current_reference,
                    asr_model,
                    asr_device,
                    asr_language,
                )
            except Exception as exc:
                if index == 0:
                    raise
                current_reference = reference_audio
                if current_reference and not uploaded_transcript:
                    uploaded_transcript = transcribe_reference(
                        current_reference,
                        asr_model,
                        asr_device,
                        asr_language,
                    )
                transcript = uploaded_transcript
                fallback = "使用上传的参考音频" if reference_audio else "无参考直接生成"
                notice = f"第{index + 1}段末尾参考转录失败，{fallback}：{exc}"
                print(f"[Reference fallback] {notice}", flush=True)
                details.append(notice)
            if current_reference == reference_audio:
                uploaded_transcript = transcript

        prefix = transcript if transcript and continuation_mode else ""
        reference_chars = count_chars(transcript)
        budget = int(chunk_chars) - reference_chars
        if budget < 1:
            raise ValueError(
                f"第{index + 1}段参考文本有{reference_chars}字，"
                f"已用完分段字数{int(chunk_chars)}，请增大分段字数或缩短参考音频。")

        chunks = iter_text_chunks(remaining, budget)
        for chunk in chunks:
            remaining = remaining[len(chunk):]
            if chunk.strip():
                break

        prompt = prefix + chunk
        total_chars = reference_chars + count_chars(chunk)
        print(f"[Chunk {index + 1}] total_chars={total_chars}, new_chars={count_chars(chunk)}", flush=True)

        duration_enabled = bool(duration_control_enabled and supports_duration_control(mode))
        chunk_expected_tokens = (
            max(1, round(int(duration_tokens) * count_chars(chunk) / max(1, count_chars(text))))
            if duration_enabled else None
        )

        sample_rate, audio, chunk_elapsed = runtime.run_single_chunk(
            text=prompt,
            reference_audio=current_reference,
            mode=mode,
            expected_tokens=chunk_expected_tokens,
            language_tag=language_tag,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            adapter=adapter,
        )

        if results and sample_rate != output_rate:
            raise RuntimeError("分段音频采样率不一致")
        output_rate = sample_rate
        results.append(audio)
        details.append(
            f"第{index + 1}段：模式={mode if current_reference else '直接生成'}；"
            f"耗时={chunk_elapsed:.2f}s；"
            f"总计{total_chars}字（参考{reference_chars}字，新增{count_chars(chunk)}字）\n"
            f"参考音频路径：{current_reference or '无'}\n"
            f"参考文本：{transcript or '无'}\n"
            f"生成文本：{prompt}"
        )

        if remaining.strip() and subsequent_mode in {MODE_CONTINUE_CLONE, "Continuation + Clone"}:
            current_reference = str(Path(temporary) / f"reference-{index}.wav")
            tail = reference_tail(audio, sample_rate)
            with wave.open(current_reference, "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(sample_rate)
                writer.writeframes((np.clip(tail, -1, 1) * 32767).astype("<i2").tobytes())

    concatenated_audio = np.concatenate(results)
    output_filename = f"openmoss-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.wav"
    output_path = runtime.output_dir / output_filename
    with wave.open(str(output_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(output_rate)
        writer.writeframes((np.clip(concatenated_audio, -1, 1) * 32767).astype("<i2").tobytes())

    total_elapsed = time.monotonic() - started
    status_text = (
        f"完成 | {len(results)}段 | 分段字数={int(chunk_chars)} | 总耗时={total_elapsed:.2f}s\n"
        f"输出文件：{output_path}\n"
        f"参考音频目录：{temporary}\n"
        + "\n".join(details)
    )

    gradio_audio = (np.clip(concatenated_audio, -1, 1) * 32767).astype(np.int16)
    return (output_rate, gradio_audio), status_text


def build_demo(runtime: OpenMossRuntime, args: argparse.Namespace) -> gr.Blocks:
    with gr.Blocks(title="MOSS-TTS · openmoss") as demo:
        gr.Markdown(
            """
            支持直接生成、音色克隆、续写 + 克隆、多语言标签以及停顿标记合成。
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                adapter = gr.Dropdown(
                    choices=["Base (原版)", *runtime.adapters],
                    value="Base (原版)",
                    label="音色",
                    visible=bool(runtime.adapters),
                    info="选择已预载的音色。",
                )
                text = gr.Textbox(
                    label="待合成文本",
                    lines=9,
                    placeholder="只输入待生成文本，参考音频文本会自动转录。",
                )
                chunk_chars = gr.Number(
                    label="分段字数上限",
                    value=args.chunk_chars,
                    precision=0,
                    minimum=1,
                    info="包含参考文本与新增文本的总字数上限，标点、空格和换行均计入。每段自动扣除参考字数后优先在标点处分段。",
                )
                gr.Markdown("首段使用上传音频；后续段选择【克隆】使用上传音频，选择【续写 + 克隆】使用上一段末尾不超过10秒的音频接续。")
                reference_audio = gr.Audio(
                    label="参考音频（可选）",
                    type="filepath",
                )
                mode_with_reference = gr.Radio(
                    choices=[MODE_CLONE, MODE_CONTINUE_CLONE],
                    value=MODE_CLONE,
                    label="首段模式",
                    info="两种模式均使用用户上传的参考音频；未上传则直接生成。",
                )
                subsequent_mode = gr.Radio(
                    choices=[MODE_CLONE, MODE_CONTINUE_CLONE],
                    value=MODE_CLONE,
                    label="后续段模式",
                    info="克隆使用上传音频（未上传则直接生成）；续写+克隆使用前一段尾部音频。",
                )
                mode_hint = gr.Markdown(render_mode_hint(None, MODE_CLONE))
                language_tag = gr.Dropdown(
                    choices=LANGUAGE_TAG_CHOICES,
                    value="中文 (Chinese)",
                    label="语言标签",
                    info="指定待生成文本的语种，非中英文时建议显式指定。",
                )
                duration_control_enabled = gr.Checkbox(
                    value=False,
                    label="开启时长控制（期望音频 Token 数，仅作用于克隆段）",
                )
                duration_tokens = gr.Slider(
                    minimum=1,
                    maximum=2,
                    step=1,
                    value=1,
                    label="期望 Token 数 (expected_tokens)",
                    visible=False,
                )
                duration_hint = gr.Markdown("时长控制已关闭。")

                with gr.Accordion("采样参数（音频）", open=True):
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=3.0,
                        step=0.05,
                        value=1.7,
                        label="采样温度 (audio_temperature)",
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        step=0.01,
                        value=0.8,
                        label="Top-P 截断 (audio_top_p)",
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=200,
                        step=1,
                        value=25,
                        label="Top-K 截断 (audio_top_k)",
                    )
                    repetition_penalty = gr.Slider(
                        minimum=0.8,
                        maximum=2.0,
                        step=0.05,
                        value=1.0,
                        label="重复惩罚 (audio_repetition_penalty)",
                    )
                    max_new_tokens = gr.Slider(
                        minimum=256,
                        maximum=8192,
                        step=128,
                        value=DEFAULT_MAX_NEW_TOKENS,
                        label="最大生成 Token 数 (max_new_tokens)",
                    )

            with gr.Column(scale=2):
                output_audio = gr.Audio(label="输出音频", type="numpy", autoplay=True)
                run_btn = gr.Button("开始生成语音", variant="primary")
                status = gr.Textbox(label="运行状态与详情", lines=4, interactive=False)

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
            inputs=[duration_control_enabled, text, duration_tokens, mode_with_reference, subsequent_mode],
            outputs=[duration_tokens, duration_hint, duration_control_enabled],
        )
        text.change(
            fn=update_duration_controls,
            inputs=[duration_control_enabled, text, duration_tokens, mode_with_reference, subsequent_mode],
            outputs=[duration_tokens, duration_hint, duration_control_enabled],
        )
        mode_with_reference.change(
            fn=update_duration_controls,
            inputs=[duration_control_enabled, text, duration_tokens, mode_with_reference, subsequent_mode],
            outputs=[duration_tokens, duration_hint, duration_control_enabled],
        )
        subsequent_mode.change(
            fn=update_duration_controls,
            inputs=[duration_control_enabled, text, duration_tokens, mode_with_reference, subsequent_mode],
            outputs=[duration_tokens, duration_hint, duration_control_enabled],
        )
        run_btn.click(
            fn=lambda text, reference_audio, mode_with_reference, duration_control_enabled, duration_tokens, language_tag, adapter, temperature, top_p, top_k, repetition_penalty, max_new_tokens, chunk_chars, subsequent_mode: run_inference(
                text=text,
                reference_audio=reference_audio,
                mode_with_reference=mode_with_reference,
                duration_control_enabled=duration_control_enabled,
                duration_tokens=duration_tokens,
                language_tag=language_tag,
                adapter=adapter,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                chunk_chars=chunk_chars,
                subsequent_mode=subsequent_mode,
                runtime=runtime,
                asr_model=args.asr_model,
                asr_device=args.asr_device,
            ),
            inputs=[
                text,
                reference_audio,
                mode_with_reference,
                duration_control_enabled,
                duration_tokens,
                language_tag,
                adapter,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                max_new_tokens,
                chunk_chars,
                subsequent_mode,
            ],
            outputs=[output_audio, status],
        )

        chunk_chars.change(
            None,
            inputs=[chunk_chars],
            js="""(v) => {
                try {
                    if (Number.isInteger(v) && v > 0) {
                        localStorage.setItem("tts_params_0", JSON.stringify(v));
                    }
                } catch (_) {}
            }"""
        )
        demo.load(
            None,
            inputs=[chunk_chars],
            outputs=[chunk_chars],
            js="""(current) => {
                try {
                    const saved = JSON.parse(localStorage.getItem("tts_params_0"));
                    if (Number.isInteger(saved) && saved > 0) return saved;
                } catch (_) {}
                return current;
            }"""
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio frontend for pwilkin/openmoss (C++ backend)")
    # Openmoss C++ server configuration
    parser.add_argument(
        "--openmoss-server",
        default=os.environ.get("OPENMOSS_SERVER", "integrations/openmoss/build-cuda/Release/moss-tts-server.exe"),
        help="moss-tts-server.exe path",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENMOSS_MODEL", "integrations/openmoss/weights/moss-tts-base-q4km.gguf"),
        help="Quantized backbone GGUF path",
    )
    parser.add_argument("--openmoss-host", default="127.0.0.1")
    parser.add_argument("--openmoss-port", type=int, default=8080)
    parser.add_argument("--main-gpu", type=int, default=0)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--cache-type-k", choices=["f16", "q8_0", "q4_0"], default="q8_0")
    parser.add_argument("--cache-type-v", choices=["f16", "q8_0", "q4_0"], default="q8_0")
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--lora", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument(
        "--codec-cpu",
        action="store_true",
        help="在 CPU 运行 AudioTokenizer codec；embeddings/LM heads 仍驻留 GPU",
    )
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--load-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--output-dir", default="outputs/openmoss")
    parser.add_argument("--preload", action="store_true")

    # Gradio app & chunking configuration (aligned with moss_tts_app.py)
    parser.add_argument("--chunk-chars", type=int, default=400, help="界面分段字数初始值（默认400，可在界面修改）")
    parser.add_argument("--asr-model", default="large-v3-turbo", help="faster-whisper 模型名或本地 CTranslate2 模型目录")
    parser.add_argument("--asr-device", choices=["cpu", "cuda"], default="cpu", help="默认 CPU int8，避免占用 TTS 显存")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--root-path",
        default=None,
        help="Gradio root path when served behind a reverse proxy, e.g. /moss",
    )
    parser.add_argument("--share", action="store_true")

    args = parser.parse_args()
    if args.parallel < 1:
        parser.error("--parallel must be >= 1")
    if args.chunk_chars < 1:
        parser.error("--chunk-chars 必须大于0")
    if any("=" not in item or not item.split("=", 1)[0] or not item.split("=", 1)[1]
           for item in args.lora):
        parser.error("every --lora must use NAME=PATH")
    adapter_names = [item.split("=", 1)[0] for item in args.lora]
    if len(adapter_names) != len(set(adapter_names)):
        parser.error("--lora adapter names must be unique")
    if args.no_flash_attn and args.cache_type_v != "f16":
        parser.error("quantized V cache requires flash attention; use --cache-type-v f16")

    runtime = OpenMossRuntime(args)
    atexit.register(runtime.stop)
    if args.preload:
        seconds = runtime.ensure_started()
        if seconds is not None:
            print(f"[openmoss] model loaded in {seconds:.2f}s", flush=True)

    demo = build_demo(runtime, args)
    demo.queue(max_size=32, default_concurrency_limit=args.parallel).launch(
        server_name=args.host,
        server_port=args.port,
        root_path=args.root_path,
        share=args.share,
    )


if __name__ == "__main__":
    main()
