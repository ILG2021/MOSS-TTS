"""Gradio frontend for pwilkin/openmoss' persistent native server.

This module does not run the repository's Python/ONNX llama.cpp pipeline.
On the first request it starts ``moss-tts-server``; openmoss then owns the
complete libllama/GGML model and keeps it resident until this app exits.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import gradio as gr


LANGUAGES = [
    "Auto", "zh", "en", "yue", "ja", "ko", "fr", "de", "es", "pt",
    "ru", "ar", "hi", "it", "nl", "pl", "tr", "vi", "th",
]


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
            # Cover both Visual Studio multi-config (build/Release +
            # build/bin/Release) and Ninja single-config (build + build/bin).
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

    def generate(self, payload: dict, reference_path: str | None) -> tuple[str, str]:
        load_seconds = self.ensure_started()
        if reference_path:
            with open(reference_path, "rb") as reference_file:
                payload["reference_wav_b64"] = base64.b64encode(reference_file.read()).decode("ascii")
        request = urllib.request.Request(
            f"{self.base_url}/tts",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.args.request_timeout) as response:
                audio = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"openmoss HTTP {exc.code}: {detail}") from exc

        output = self.output_dir / f"openmoss-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.wav"
        output.write_bytes(audio)
        elapsed = time.monotonic() - started
        cold = f"；首次加载 {load_seconds:.2f}s" if load_seconds is not None else ""
        return str(output), f"完成：推理 {elapsed:.2f}s{cold}；openmoss 模型继续常驻"

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


def build_demo(runtime: OpenMossRuntime) -> gr.Blocks:
    def status():
        return "模型状态：已由 openmoss 加载并常驻" if runtime.resident else "模型状态：未加载（首次生成时启动 openmoss）"

    def generate(text, reference, adapter, language, tokens, max_new_tokens, temperature, top_p, top_k, repetition_penalty):
        text = (text or "").strip()
        if not text:
            raise gr.Error("请输入文本")
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
        if language != "Auto":
            payload["language"] = language
        if adapter != "Base":
            payload["lora"] = adapter
        if tokens and int(tokens) > 0:
            payload["token_count"] = int(tokens)
        try:
            return runtime.generate(payload, reference or None)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="MOSS-TTS · openmoss") as demo:
        gr.Markdown(
            "# MOSS-TTS v1.5 · openmoss/GGML\n"
            "首次生成时启动原生 `moss-tts-server`；完整模型由 openmoss 加载并持续常驻。"
        )
        runtime_status = gr.Textbox(value=status(), label="Runtime", interactive=False)
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(label="Text", lines=8)
                reference = gr.Audio(label="Reference Audio（可选）", type="filepath")
                adapter = gr.Dropdown(["Base", *runtime.adapters], value="Base", label="LoRA voice")
                language = gr.Dropdown(LANGUAGES, value="zh", label="Language")
                tokens = gr.Number(value=0, precision=0, label="Expected tokens（0=自动，约 12.5 token/秒）")
                max_new_tokens = gr.Slider(128, 8192, value=4096, step=128, label="max_new_tokens")
                with gr.Accordion("Sampling", open=True):
                    temperature = gr.Slider(0.1, 3.0, value=1.7, step=0.05, label="audio_temperature")
                    top_p = gr.Slider(0.05, 1.0, value=0.8, step=0.01, label="audio_top_p")
                    top_k = gr.Slider(1, 200, value=25, step=1, label="audio_top_k")
                    repetition_penalty = gr.Slider(0.8, 2.0, value=1.0, step=0.05, label="audio_repetition_penalty")
                button = gr.Button("Generate Speech", variant="primary")
            with gr.Column(scale=2):
                output = gr.Audio(label="Output", type="filepath")
                result = gr.Textbox(label="Status", lines=4, interactive=False)
        button.click(
            generate,
            inputs=[text, reference, adapter, language, tokens, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
            outputs=[output, result],
        ).then(status, outputs=runtime_status)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Lazy Gradio frontend for pwilkin/openmoss")
    parser.add_argument("--openmoss-server", required=True, help="moss-tts-server.exe path")
    parser.add_argument("--model", required=True, help="Quantized backbone GGUF path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
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
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    if args.parallel < 1:
        parser.error("--parallel must be >= 1")
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
    build_demo(runtime).queue(max_size=32, default_concurrency_limit=args.parallel).launch(
        server_name=args.host, server_port=args.port, share=args.share,
    )


if __name__ == "__main__":
    main()
