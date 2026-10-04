"""Merge a Delay PEFT adapter into a standalone inference checkpoint."""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _adapter_prompt_template(adapter_path: Path):
    """Return the prompt_template the adapter was trained with (None if unknown)."""
    import json

    for name, key in (("processor_config.json", "prompt_template"),
                      ("finetune_args.json", "resolved_prompt_template")):
        path = adapter_path / name
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8")).get(key)
            except (OSError, ValueError):
                value = None
            if value:
                return value
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--codec-path", default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("--output-dir must be a new or empty directory.")

    import torch
    from peft import PeftModel
    from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel
    from moss_tts_delay.finetuning.sft_lora import copy_support_files, copy_inference_assets

    base = MossTTSDelayModel.from_pretrained(
        args.model_path, torch_dtype=getattr(torch, args.dtype), attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(output, safe_serialization=True)
    copy_support_files(output)
    copy_inference_assets(
        args.model_path, args.codec_path, output,
        prompt_template=_adapter_prompt_template(Path(args.adapter_path)),
    )
    print(f"Merged inference model saved to {output}")


if __name__ == "__main__":
    main()
