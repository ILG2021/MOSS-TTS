"""Clone one reference voice for TXT inputs grouped by physical lines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def group_lines(text: str, lines_per_group: int = 4) -> list[dict]:
    if lines_per_group < 1:
        raise ValueError("lines_per_group must be positive")
    lines = text.splitlines()
    groups = []
    for start in range(0, len(lines), lines_per_group):
        chunk = "\n".join(lines[start:start + lines_per_group]).strip()
        if chunk:
            groups.append({"index": len(groups) + 1, "line_start": start + 1,
                           "line_end": min(start + lines_per_group, len(lines)),
                           "text": chunk})
    return groups


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-txt", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/local_v1_5_batch"))
    parser.add_argument("--model-dir", default="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5")
    parser.add_argument("--codec-dir", default="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2")
    parser.add_argument("--lines-per-group", type=positive_int, default=4)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--language", default="Chinese", help="Use auto to omit the language tag")
    parser.add_argument("--device", default=None, help="Default: CUDA if available, otherwise CPU")
    parser.add_argument("--codec-device", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--attn-implementation", choices=["auto", "sdpa", "eager", "flash_attention_2"], default="auto")
    parser.add_argument("--max-new-tokens", type=positive_int, default=7500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Write grouped texts and manifest without loading models")
    args = parser.parse_args()
    if not args.reference_audio.is_file():
        parser.error(f"Reference audio not found: {args.reference_audio}")
    try:
        groups = group_lines(args.input_txt.read_text(encoding=args.encoding), args.lines_per_group)
    except (OSError, UnicodeError, LookupError) as exc:
        parser.error(str(exc))
    if not groups:
        parser.error("Input TXT contains no non-blank text")
    # An isolated output directory prevents mixing results from different runs.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for group in groups:
        group["audio_file"] = f"{group['index']:04d}.wav"
        group["status"] = "pending"
        (args.output_dir / f"{group['index']:04d}.txt").write_text(group["text"], encoding="utf-8")
    manifest = {"input_txt": str(args.input_txt.resolve()),
                "reference_audio": str(args.reference_audio.resolve()),
                "options": {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()}, "groups": groups}

    def save_manifest():
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    save_manifest()
    print(f"Prepared {len(groups)} groups in {args.output_dir}", flush=True)
    if args.dry_run:
        return

    import torch
    import torchaudio
    from transformers import set_seed

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "moss_tts_local_v1.5"))
    from streaming import load_runtime

    torch.backends.cuda.enable_cudnn_sdp(False)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = args.dtype or ("bf16" if torch.device(device).type == "cuda" else "fp32")
    set_seed(args.seed)
    runtime = load_runtime(model_dir=args.model_dir, codec_dir=args.codec_dir,
                           device=device, codec_device=args.codec_device, dtype=dtype,
                           codec_compute_dtype="fp32" if torch.device(args.codec_device or device).type == "cpu" else dtype,
                           attn_implementation=args.attn_implementation, warmup=False)
    processor = runtime.processor
    language = None if args.language.lower() == "auto" else args.language
    with torch.inference_mode():
        # Encode the shared reference only once, then reuse its discrete codes.
        reference = processor.encode_audios_from_path(str(args.reference_audio.resolve()))
        for start in range(0, len(groups), args.batch_size):
            current = groups[start:start + args.batch_size]
            print(f"Generating groups {start + 1}-{start + len(current)}/{len(groups)}", flush=True)
            try:
                conversations = [[processor.build_user_message(
                    text=group["text"], reference=reference, language=language)] for group in current]
                batch = processor(conversations, mode="generation")
                outputs = runtime.model.generate(
                    input_ids=batch["input_ids"].to(runtime.device),
                    attention_mask=batch["attention_mask"].to(runtime.device),
                    max_new_tokens=args.max_new_tokens, do_sample=True,
                    audio_temperature=1.7, audio_top_p=0.8, audio_top_k=25,
                    audio_repetition_penalty=1.0)
                messages = processor.decode(outputs)
                if len(messages) != len(current):
                    raise RuntimeError("Decoded output count does not match the input batch")
                for group, message in zip(current, messages):
                    if message is None or not message.audio_codes_list:
                        raise RuntimeError(f"No audio generated for group {group['index']}")
                    audio = message.audio_codes_list[0].detach().float().cpu()
                    if audio.numel() == 0:
                        raise RuntimeError(f"Empty audio for group {group['index']}")
                    torchaudio.save(str(args.output_dir / group["audio_file"]), audio, runtime.sample_rate)
                    group.update(status="done", duration_seconds=audio.shape[-1] / runtime.sample_rate)
                    save_manifest()
                del outputs, messages, batch
            except Exception as exc:
                for group in current:
                    if group["status"] != "done":
                        group.update(status="failed", error=str(exc))
                save_manifest()
                raise
    print(f"Done: {len(groups)} WAV files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
