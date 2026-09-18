#!/usr/bin/env python3
"""Convert a MOSS-TTS PEFT adapter into a Qwen3 llama.cpp LoRA GGUF."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


MOSS_PREFIX = "base_model.model.language_model."
QWEN_PREFIX = "base_model.model."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter", type=Path, help="PEFT adapter directory")
    parser.add_argument("--base-qwen", type=Path, required=True,
                        help="Qwen3 backbone directory retained from openmoss conversion")
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--outtype", choices=["f32", "f16", "bf16"], default="f16")
    parser.add_argument("--llama-cpp-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "third_party" / "llama.cpp")
    args = parser.parse_args()

    adapter = args.adapter.resolve()
    weights = adapter / "adapter_model.safetensors"
    config_path = adapter / "adapter_config.json"
    if not weights.is_file() or not config_path.is_file():
        raise FileNotFoundError("adapter_model.safetensors and adapter_config.json are required")
    if not (args.base_qwen / "config.json").is_file():
        raise FileNotFoundError("--base-qwen must point to the extracted Qwen3 HF directory")

    with tempfile.TemporaryDirectory(prefix="moss-lora-") as tmp:
        normalized = Path(tmp)
        tensors = {}
        # Keep the torch backend: PEFT adapters may contain BF16 tensors, which
        # NumPy cannot represent portably, and llama.cpp's converter uses torch.
        with safe_open(weights, framework="pt", device="cpu") as source:
            for name in source.keys():
                if not name.startswith(MOSS_PREFIX):
                    raise ValueError(
                        f"unexpected adapter tensor {name!r}; only language_model LoRA is supported"
                    )
                tensors[QWEN_PREFIX + name[len(MOSS_PREFIX):]] = source.get_tensor(name)
        save_file(tensors, normalized / "adapter_model.safetensors")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["base_model_name_or_path"] = str(args.base_qwen.resolve())
        (normalized / "adapter_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        converter = args.llama_cpp_dir.resolve() / "convert_lora_to_gguf.py"
        command = [
            sys.executable, str(converter),
            "--base", str(args.base_qwen.resolve()),
            "--outfile", str(args.outfile.resolve()),
            "--outtype", args.outtype,
            str(normalized),
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
