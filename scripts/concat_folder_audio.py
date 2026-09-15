#!/usr/bin/env python3
"""Concatenate audio clips from a folder into a single audio file with silence intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


def natural_sort_key(path: Path) -> list[int | str]:
    """Sort key for human/natural ordering (e.g. 0001, 0002, or 1, 2, 10)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def collect_audio_files(input_dir: Path, output_file: Path, pattern: str = "*.wav") -> list[Path]:
    """Collect audio files using manifest.json if present, otherwise scan directory."""
    audio_files: list[Path] = []
    manifest_path = input_dir / "manifest.json"

    # 1. Try reading manifest.json order if available
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            groups = data.get("groups", [])
            for group in groups:
                fname = group.get("audio_file")
                if fname:
                    fpath = input_dir / fname
                    if fpath.is_file() and fpath.resolve() != output_file.resolve():
                        audio_files.append(fpath)
            if audio_files:
                print(f"[Info] Found {len(audio_files)} audio segments from manifest.json")
                return audio_files
        except Exception as exc:
            print(f"[Warning] Failed to parse manifest.json: {exc}. Falling back to directory scan.")

    # 2. Fallback to scanning matching audio files
    matched = list(input_dir.glob(pattern))
    resolved_out = output_file.resolve()
    for f in matched:
        if f.is_file() and f.resolve() != resolved_out and f.name != "combined.wav":
            audio_files.append(f)

    audio_files.sort(key=natural_sort_key)
    return audio_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        help="Directory containing the audio files to concatenate",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        dest="input_dir_opt",
        help="Alternative flag for specifying the input directory",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        default=None,
        help="Path for the output WAV file (default: <input_dir>/combined.wav)",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=500,
        help="Silence duration in milliseconds inserted between clips (default: 500)",
    )
    parser.add_argument(
        "--pattern",
        default="*.wav",
        help="Glob pattern for audio files when no manifest is used (default: *.wav)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir_opt or args.input_dir
    if not input_dir:
        parser.error("Please specify an input directory.")
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    output_file = args.output_file or (input_dir / "combined.wav")
    output_file = output_file.resolve()

    audio_paths = collect_audio_files(input_dir, output_file, pattern=args.pattern)
    if not audio_paths:
        sys.exit(f"[Error] No audio files found in {input_dir} (pattern: {args.pattern})")

    print(f"[Info] Splicing {len(audio_paths)} audio files with {args.silence_ms}ms intervals...")

    import torch
    import torchaudio

    audios: list[torch.Tensor] = []
    target_sr: int | None = None

    for idx, path in enumerate(audio_paths, start=1):
        waveform, sr = torchaudio.load(str(path))
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if target_sr is None:
            target_sr = sr
        elif sr != target_sr:
            # Resample to target sample rate
            print(f"[Info] Resampling {path.name} from {sr}Hz to {target_sr}Hz...")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        audios.append(waveform)

    # Normalize channel count (e.g., mono vs stereo)
    max_channels = max(a.shape[0] for a in audios)
    normalized: list[torch.Tensor] = []
    for a in audios:
        if a.shape[0] < max_channels:
            a = a.repeat(max_channels, 1)
        normalized.append(a)

    # Insert silence intervals
    silence_samples = int(target_sr * (args.silence_ms / 1000.0))
    if silence_samples > 0 and len(normalized) > 1:
        silence_shape = (*normalized[0].shape[:-1], silence_samples)
        silence = torch.zeros(silence_shape, dtype=normalized[0].dtype, device=normalized[0].device)
        pieces = []
        for i, seg in enumerate(normalized):
            if i > 0:
                pieces.append(silence)
            pieces.append(seg)
        combined = torch.cat(pieces, dim=-1)
    else:
        combined = torch.cat(normalized, dim=-1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_file), combined, target_sr)

    total_duration = combined.shape[-1] / target_sr
    print(
        f"[Done] Successfully saved concatenated audio to:\n  -> {output_file}\n"
        f"  Total clips: {len(audio_paths)}\n"
        f"  Total duration: {total_duration:.2f}s ({total_duration / 60:.2f} min)\n"
        f"  Sample rate: {target_sr}Hz, Channels: {max_channels}"
    )


if __name__ == "__main__":
    main()
