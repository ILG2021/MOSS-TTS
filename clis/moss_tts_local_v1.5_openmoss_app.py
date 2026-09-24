"""Gradio frontend for MOSS-TTS Local Transformer v1.5 via openmoss.

This is intentionally a separate entry point from ``moss_tts_openmoss_app.py``:
the local-pattern model emits 12 codebooks per frame, produces 48 kHz stereo
audio, and uses ``ref_text`` for continuation prompting.
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
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
import zipfile

import gradio as gr
import numpy as np
import soundfile as sf

if __package__:
    from . import moss_tts_openmoss_app as shared
    from .tts_chunking import count_chars, iter_text_chunks, transcribe_reference
else:
    try:
        import moss_tts_openmoss_app as shared
        from tts_chunking import count_chars, iter_text_chunks, transcribe_reference
    except ImportError:
        from clis import moss_tts_openmoss_app as shared
        from clis.tts_chunking import count_chars, iter_text_chunks, transcribe_reference


class OpenMossLocalRuntime(shared.OpenMossRuntime):
    """openmoss client with local-pattern validation and stereo preservation."""

    def _validate_server(self, info: dict) -> None:
        if info.get("architecture") != "moss_tts_local":
            raise RuntimeError(
                f"{self.base_url} 已有非 moss_tts_local 服务：{info.get('architecture')!r}"
            )
        if info.get("codec_cpu") is not self.args.codec_cpu:
            raise RuntimeError(
                f"{self.base_url} 的 codec_cpu={info.get('codec_cpu')!r}，"
                f"但当前 Gradio 请求 codec_cpu={self.args.codec_cpu!r}"
            )
        if int(info.get("parallel", 1)) != 1:
            raise RuntimeError("moss_tts_local 当前只支持 --parallel 1")
        if self.adapters:
            advertised = info.get("lora_adapters")
            if not isinstance(advertised, list):
                raise RuntimeError("openmoss server 未提供 LoRA adapter 信息")
            missing = sorted(set(self.adapters) - set(advertised))
            if missing:
                raise RuntimeError("server 未加载这些 LoRA：" + ", ".join(missing))

    def run_single_chunk(
        self,
        text: str,
        reference_audio: str | None,
        ref_text: str | None,
        expected_tokens: int | None,
        language_tag: str | None,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        max_new_tokens: int,
        adapter: str = "Base",
        voice_id: str | None = None,
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
        language = shared.normalize_language_tag(language_tag)
        if language is not None:
            payload["language"] = language
        if adapter and adapter not in {"Base", "Base (原版)"}:
            payload["lora"] = adapter
        if expected_tokens is not None and expected_tokens > 0:
            payload["token_count"] = int(expected_tokens)
        if voice_id:
            payload["voice"] = voice_id
            if ref_text:
                payload["ref_text"] = ref_text
        elif reference_audio:
            wav_bytes = shared._audio_to_wav_bytes(reference_audio)
            payload["reference_wav_b64"] = base64.b64encode(wav_bytes).decode("ascii")
            if ref_text:
                payload["ref_text"] = ref_text

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
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        if audio.shape[1] != 2 or not audio.size or not np.isfinite(audio).all():
            raise RuntimeError(
                f"openmoss local 后端返回了无效音频，期望双声道，实际 shape={audio.shape}"
            )
        return sample_rate, audio.astype(np.float32, copy=False), elapsed

    def register_voice(self, voice_id: str, audio_path: str, transcript: str) -> None:
        self.ensure_started()
        boundary = f"----openmoss-{uuid.uuid4().hex}"
        wav = shared._audio_to_wav_bytes(audio_path)
        parts = []
        for name, value in (("voice_id", voice_id), ("transcript", transcript)):
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"reference.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode() + wav + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        req = urllib.request.Request(f"{self.base_url}/v1/voices", data=b"".join(parts), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.args.request_timeout):
            pass

    def delete_voice(self, voice_id: str) -> None:
        req = urllib.request.Request(f"{self.base_url}/v1/voices/{voice_id}", method="DELETE")
        try:
            with urllib.request.urlopen(req, timeout=30):
                pass
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise


def _stereo_tail(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return at most ten seconds of sample-major stereo audio."""
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate <= 0 or audio.ndim != 2 or audio.shape[1] != 2 or not audio.size:
        raise ValueError("Expected non-empty stereo audio and a positive sample rate")
    start = max(0, audio.shape[0] - round(10 * sample_rate))
    if start == 0:
        return audio.copy()
    window = max(1, round(0.25 * sample_rate))
    end = min(start + round(2 * sample_rate), audio.shape[0] - window)
    probes = list(range(start, end + 1, 512))
    best = min(probes, key=lambda p: float(np.mean(audio[p:p + window].astype(np.float64) ** 2)))
    return audio[best:].copy()


def run_inference(
    text, reference_audio, mode_with_reference, duration_control_enabled,
    duration_tokens, language_tag, temperature, top_p, top_k,
    repetition_penalty, max_new_tokens, chunk_chars, subsequent_mode, adapter,
    runtime, asr_model, asr_device, save_output=True, voice_id=None, initial_ref_text=None,
):
    valid_modes = {
        shared.MODE_CLONE, shared.MODE_CONTINUE_CLONE,
        "Clone", "Continuation + Clone",
    }
    if mode_with_reference not in valid_modes or subsequent_mode not in valid_modes:
        raise ValueError("分段模式仅支持克隆和续写+克隆")
    if chunk_chars is None or not float(chunk_chars).is_integer() or chunk_chars < 1:
        raise ValueError("分段字数必须为正整数")
    remaining = text or ""
    if not remaining.strip():
        raise ValueError("请输入待生成文本")

    started = time.monotonic()
    results: list[np.ndarray] = []
    details: list[str] = []
    current_reference = reference_audio
    uploaded_transcript = ""
    asr_language = shared.normalize_language_tag(language_tag)
    temp_root = Path(__file__).resolve().parents[1] / "Temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temporary_owner = tempfile.TemporaryDirectory(
        prefix="moss-local-openmoss-rolling-", dir=temp_root
    )
    temporary = Path(temporary_owner.name)
    output_rate = 48000

    while remaining.strip():
        index = len(results)
        mode = mode_with_reference if index == 0 else subsequent_mode
        if index == 0 or mode in {shared.MODE_CLONE, "Clone"}:
            current_reference = reference_audio
        continuation = mode in {shared.MODE_CONTINUE_CLONE, "Continuation + Clone"}
        transcript = initial_ref_text if index == 0 and continuation else ""
        if current_reference and continuation and not (index == 0 and initial_ref_text):
            try:
                transcript = transcribe_reference(
                    current_reference, asr_model, asr_device, asr_language
                )
            except Exception as exc:
                if index == 0:
                    raise
                current_reference = reference_audio
                if current_reference and not uploaded_transcript:
                    uploaded_transcript = transcribe_reference(
                        current_reference, asr_model, asr_device, asr_language
                    )
                transcript = uploaded_transcript
                details.append(f"第{index + 1}段末尾参考转录失败，已回退：{exc}")
            if current_reference == reference_audio:
                uploaded_transcript = transcript

        reference_chars = count_chars(transcript)
        budget = int(chunk_chars) - reference_chars
        if budget < 1:
            raise ValueError(
                f"第{index + 1}段参考文本有{reference_chars}字，已用完分段字数"
                f"{int(chunk_chars)}，请增大分段字数或缩短参考音频。"
            )
        for chunk in iter_text_chunks(remaining, budget):
            remaining = remaining[len(chunk):]
            if chunk.strip():
                break

        duration_enabled = bool(
            duration_control_enabled and shared.supports_duration_control(mode)
        )
        expected_tokens = (
            max(1, round(int(duration_tokens) * count_chars(chunk) / max(1, count_chars(text))))
            if duration_enabled else None
        )
        sample_rate, audio, elapsed = runtime.run_single_chunk(
            text=chunk,
            reference_audio=current_reference,
            ref_text=transcript if continuation else None,
            expected_tokens=expected_tokens,
            language_tag=language_tag,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            adapter=adapter,
            voice_id=voice_id if voice_id and (index == 0 or not continuation) else None,
        )
        if results and sample_rate != output_rate:
            raise RuntimeError("分段音频采样率不一致")
        output_rate = sample_rate
        results.append(audio)
        details.append(
            f"第{index + 1}段：模式={mode if current_reference else '直接生成'}；"
            f"耗时={elapsed:.2f}s；参考{reference_chars}字；新增{count_chars(chunk)}字\n"
            f"参考音频：{current_reference or '无'}\n参考文本：{transcript or '无'}\n"
            f"生成文本：{chunk}"
        )

        if remaining.strip() and subsequent_mode in {
            shared.MODE_CONTINUE_CLONE, "Continuation + Clone"
        }:
            current_reference = str(temporary / f"reference-{index}.wav")
            sf.write(current_reference, _stereo_tail(audio, sample_rate), sample_rate, subtype="PCM_16")

    concatenated = np.concatenate(results, axis=0)
    output_path = None
    if save_output:
        output_path = runtime.output_dir / (
            f"openmoss-local-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.wav"
        )
        sf.write(output_path, concatenated, output_rate, subtype="PCM_16")
    total_elapsed = time.monotonic() - started
    status = (
        f"完成 | {len(results)}段 | 48kHz双声道 | 总耗时={total_elapsed:.2f}s\n"
        f"输出文件：{output_path or '由调用方保存'}\n"
        + "\n".join(details)
    )
    temporary_owner.cleanup()
    return (output_rate, (np.clip(concatenated, -1, 1) * 32767).astype(np.int16)), status


def parse_batch_lines(text: str) -> list[tuple[int, str]]:
    """Return ``(display_number, body)`` for every non-empty physical line."""
    rows: list[tuple[int, str]] = []
    for physical_line, raw_line in enumerate((text or "").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)(.*)$", line, flags=re.DOTALL)
        if match:
            number = int(match.group(1))
            body = match.group(2).strip()
            if not body:
                raise ValueError(f"第{physical_line}行只有行号，没有待合成正文")
        else:
            number = len(rows) + 1
            body = line
        rows.append((number, body))
    if not rows:
        raise ValueError("请输入至少一行待合成文本")
    return rows


def sanitize_filename_text(text: str, max_chars: int = 100) -> str:
    """Create a readable Windows-safe filename fragment while retaining CJK."""
    value = unicodedata.normalize("NFKC", text)
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", value)
    value = re.sub(r"\s+", "_", value).strip(" ._")
    value = value[:max_chars].rstrip(" ._")
    if not value:
        return "audio"
    if value.upper() in {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        value = f"_{value}"[:max_chars]
    return value


def run_batch_inference(
    text, reference_audio, mode_with_reference, duration_control_enabled,
    duration_tokens, language_tag, temperature, top_p, top_k,
    repetition_penalty, max_new_tokens, chunk_chars, subsequent_mode, adapter,
    runtime, asr_model, asr_device,
):
    rows = parse_batch_lines(text)
    batch_owner = tempfile.TemporaryDirectory(prefix="batch-", dir=runtime.output_dir)
    batch_dir = Path(batch_owner.name)
    used_names: set[str] = set()
    generated: list[Path] = []
    details: list[str] = []
    started = time.monotonic()

    voice_id = None
    initial_ref_text = None
    if reference_audio:
        if mode_with_reference in {shared.MODE_CONTINUE_CLONE, "Continuation + Clone"}:
            initial_ref_text = transcribe_reference(
                reference_audio,
                asr_model,
                asr_device,
                shared.normalize_language_tag(language_tag),
            )
        voice_id = f"batch_{uuid.uuid4().hex}"
        runtime.register_voice(voice_id, reference_audio, initial_ref_text or "-")
    try:
        for position, (line_number, body) in enumerate(rows, 1):
            audio_result, _ = run_inference(
                text=body,
                reference_audio=reference_audio,
                mode_with_reference=mode_with_reference,
                duration_control_enabled=duration_control_enabled,
                duration_tokens=duration_tokens,
                language_tag=language_tag,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                chunk_chars=chunk_chars,
                subsequent_mode=subsequent_mode,
                adapter=adapter,
                runtime=runtime,
                asr_model=asr_model,
                asr_device=asr_device,
                save_output=False,
                voice_id=voice_id,
                initial_ref_text=initial_ref_text,
            )
            sample_rate, pcm = audio_result
            stem = f"{line_number:04d}_{sanitize_filename_text(body)}"
            candidate = f"{stem}.wav"
            duplicate = 2
            while candidate.casefold() in used_names:
                candidate = f"{stem}_{duplicate}.wav"
                duplicate += 1
            used_names.add(candidate.casefold())
            output_path = batch_dir / candidate
            sf.write(output_path, np.asarray(pcm, dtype=np.int16), sample_rate, subtype="PCM_16")
            generated.append(output_path)
            details.append(f"[{position}/{len(rows)}] {candidate}")
    finally:
        if voice_id:
            runtime.delete_voice(voice_id)

    zip_path = runtime.output_dir / (
        f"openmoss-local-batch-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.zip"
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for audio_path in generated:
            archive.write(audio_path, arcname=audio_path.name)
    elapsed = time.monotonic() - started
    status = (
        f"批量合成完成 | {len(generated)}行 | 总耗时={elapsed:.2f}s\n"
        f"压缩包：{zip_path}\n" + "\n".join(details)
    )
    batch_owner.cleanup()
    return str(zip_path), status


def build_demo(runtime: OpenMossLocalRuntime, args: argparse.Namespace) -> gr.Blocks:
    with gr.Blocks(title="MOSS-TTS Local v1.5 · openmoss") as demo:
        gr.Markdown("支持单条合成、按行批量合成、音色克隆和续写 + 克隆。")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                adapter = gr.Dropdown(
                    choices=["Base (原版)", *runtime.adapters], value="Base (原版)",
                    label="音色", visible=bool(runtime.adapters), info="选择已预载的音色。",
                )
                text = gr.Textbox(
                    label="待合成文本", lines=12,
                    placeholder="单条合成可输入任意文本；批量合成时每个非空行生成一个 WAV。",
                )
                chunk_chars = gr.Number(
                    label="分段字数上限", value=args.chunk_chars, precision=0, minimum=1,
                    info="单行过长时仍按标点自动分段，最后拼成该行对应的一个 WAV。",
                )
                reference_audio = gr.Audio(label="参考音频（可选）", type="filepath")
                mode_with_reference = gr.Radio(
                    choices=[shared.MODE_CLONE, shared.MODE_CONTINUE_CLONE],
                    value=shared.MODE_CLONE, label="首段模式",
                )
                subsequent_mode = gr.Radio(
                    choices=[shared.MODE_CLONE, shared.MODE_CONTINUE_CLONE],
                    value=shared.MODE_CLONE, label="后续段模式",
                )
                mode_hint = gr.Markdown(shared.render_mode_hint(None, shared.MODE_CLONE))
                language_tag = gr.Dropdown(
                    choices=shared.LANGUAGE_TAG_CHOICES, value="中文 (Chinese)", label="语言标签",
                )
                duration_control_enabled = gr.Checkbox(
                    value=False, label="开启时长控制（期望音频 Token 数，仅作用于克隆段）",
                )
                duration_tokens = gr.Slider(
                    minimum=1, maximum=2, step=1, value=1,
                    label="期望 Token 数 (expected_tokens)", visible=False,
                )
                duration_hint = gr.Markdown("时长控制已关闭。")
                with gr.Accordion("采样参数（音频）", open=True):
                    temperature = gr.Slider(0.1, 3.0, value=1.7, step=0.05, label="采样温度")
                    top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.01, label="Top-P")
                    top_k = gr.Slider(1, 200, value=25, step=1, label="Top-K")
                    repetition_penalty = gr.Slider(0.8, 2.0, value=1.0, step=0.05, label="重复惩罚")
                    max_new_tokens = gr.Slider(
                        256, 8192, value=shared.DEFAULT_MAX_NEW_TOKENS, step=128,
                        label="最大生成 Token 数",
                    )
            with gr.Column(scale=2):
                output_audio = gr.Audio(label="输出音频", type="numpy", autoplay=True)
                with gr.Row():
                    run_btn = gr.Button("开始生成语音", variant="primary")
                    batch_btn = gr.Button("批量合成")
                batch_zip = gr.File(label="批量合成 ZIP", interactive=False)
                status = gr.Textbox(label="运行状态与详情", lines=8, interactive=False)

        reference_audio.change(
            shared.render_mode_hint, [reference_audio, mode_with_reference], mode_hint
        )
        mode_with_reference.change(
            shared.render_mode_hint, [reference_audio, mode_with_reference], mode_hint
        )
        duration_inputs = [
            duration_control_enabled, text, duration_tokens,
            mode_with_reference, subsequent_mode,
        ]
        duration_outputs = [duration_tokens, duration_hint, duration_control_enabled]
        for component in (duration_control_enabled, text, mode_with_reference, subsequent_mode):
            component.change(shared.update_duration_controls, duration_inputs, duration_outputs)

        common_inputs = [
            text, reference_audio, mode_with_reference, duration_control_enabled,
            duration_tokens, language_tag, adapter, temperature, top_p, top_k,
            repetition_penalty, max_new_tokens, chunk_chars, subsequent_mode,
        ]
        run_btn.click(
            fn=lambda *values: run_inference(
                *values, runtime=runtime, asr_model=args.asr_model, asr_device=args.asr_device
            ),
            inputs=common_inputs, outputs=[output_audio, status],
        )
        batch_btn.click(
            fn=lambda *values: run_batch_inference(
                *values, runtime=runtime, asr_model=args.asr_model, asr_device=args.asr_device
            ),
            inputs=common_inputs, outputs=[batch_zip, status],
        )
        chunk_chars.change(
            None, [chunk_chars],
            js="""(v) => { if (Number.isInteger(v) && v > 0) localStorage.setItem('tts_params_0', JSON.stringify(v)); }""",
        )
        demo.load(
            None, [chunk_chars], [chunk_chars],
            js="""(v) => { try { const x=JSON.parse(localStorage.getItem('tts_params_0')); return Number.isInteger(x)&&x>0?x:v; } catch (_) { return v; } }""",
        )
    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MOSS-TTS Local Transformer v1.5 frontend (openmoss C++ backend)"
    )
    parser.add_argument(
        "--openmoss-server",
        default=os.environ.get(
            "OPENMOSS_SERVER", "integrations/openmoss/build-cuda/Release/moss-tts-server.exe"
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "OPENMOSS_LOCAL_MODEL", "integrations/openmoss/weights/moss-tts-local.gguf"
        ),
    )
    parser.add_argument("--openmoss-host", default="127.0.0.1")
    parser.add_argument("--openmoss-port", type=int, default=8080)
    parser.add_argument("--main-gpu", type=int, default=0)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--cache-type-k", choices=["f16", "q8_0", "q4_0"], default="q8_0")
    parser.add_argument("--cache-type-v", choices=["f16", "q8_0", "q4_0"], default="q8_0")
    parser.add_argument("--parallel", type=int, default=1, choices=[1])
    parser.add_argument("--lora", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--codec-cpu", action="store_true")
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--load-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--output-dir", default="outputs/openmoss-local-v1.5")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--chunk-chars", type=int, default=400)
    parser.add_argument("--asr-model", default="large-v3-turbo")
    parser.add_argument("--asr-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--root-path", default=None)
    parser.add_argument("--share", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.chunk_chars < 1:
        parser.error("--chunk-chars 必须大于0")
    if any("=" not in item or not all(item.split("=", 1)) for item in args.lora):
        parser.error("every --lora must use NAME=PATH")
    names = [item.split("=", 1)[0] for item in args.lora]
    if len(names) != len(set(names)):
        parser.error("--lora adapter names must be unique")
    if args.no_flash_attn and args.cache_type_v != "f16":
        parser.error("quantized V cache requires flash attention; use --cache-type-v f16")

    runtime = OpenMossLocalRuntime(args)
    atexit.register(runtime.stop)
    if args.preload:
        seconds = runtime.ensure_started()
        if seconds is not None:
            print(f"[openmoss-local] model loaded in {seconds:.2f}s", flush=True)

    demo = build_demo(runtime, args)
    demo.queue(max_size=32, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        root_path=args.root_path,
        share=args.share,
    )


if __name__ == "__main__":
    main()
