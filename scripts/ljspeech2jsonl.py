#!/usr/bin/env python3
"""
Convert LJSpeech-style metadata into MossTTSLocal fine-tuning JSONL.

Examples:
    python scripts/ljspeech2jsonl.py ^
        --input metadata.txt ^
        --output train_raw.jsonl ^
        --language zh

Input lines are expected to look like:
    2025-01-25/2025-01-25_80.wav|哎，当然呢，

By default the output is plain text/audio pairs:
    {"audio":"2025-01-25/2025-01-25_80.wav","text":"哎，当然呢，","language":"zh"}
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LJSpeech-style metadata to MossTTSLocal JSONL."
    )
    parser.add_argument("--input", required=True, help="Input metadata file.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument("--language", default="zh", help="Language tag written to each record.")
    parser.add_argument(
        "--separator",
        default="|",
        help="Metadata column separator. Defaults to '|'.",
    )
    parser.add_argument(
        "--audio-column",
        type=int,
        default=0,
        help="Zero-based column index for the audio path. Defaults to 0.",
    )
    parser.add_argument(
        "--text-column",
        type=int,
        default=1,
        help="Zero-based column index for text. Defaults to 1.",
    )
    parser.add_argument(
        "--audio-root",
        default=None,
        help=(
            "Root used to resolve relative audio paths. Defaults to the input "
            "metadata file directory."
        ),
    )
    parser.add_argument(
        "--ref-mode",
        choices=("none", "same-dir-random", "same-dir-fixed"),
        default="none",
        help=(
            "Optional ref_audio strategy. 'none' writes plain pairs. "
            "'same-dir-random' picks another item from the same folder. "
            "'same-dir-fixed' uses one stable item per folder."
        ),
    )
    parser.add_argument(
        "--ref-ratio",
        type=float,
        default=1.0,
        help="Fraction of records that get ref_audio when ref-mode is enabled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for same-dir-random and ref-ratio.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip records whose audio file does not exist.",
    )
    return parser.parse_args()


def normalize_path(path_text: str, audio_root: Path) -> str:
    path = Path(path_text.strip())
    if not path.is_absolute():
        path = audio_root / path
    return path.resolve().as_posix()


def read_records(args: argparse.Namespace) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    input_path = Path(args.input)
    audio_root = Path(args.audio_root).resolve() if args.audio_root else input_path.parent.resolve()

    with input_path.open("r", encoding="utf-8-sig") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            columns = line.split(args.separator)
            needed_index = max(args.audio_column, args.text_column)
            if len(columns) <= needed_index:
                raise ValueError(
                    f"Line {line_number} has {len(columns)} columns, "
                    f"but column {needed_index} is required: {line!r}"
                )

            audio = normalize_path(columns[args.audio_column], audio_root)
            text = columns[args.text_column].strip()
            if not audio or not text:
                continue

            if args.skip_missing and not Path(audio).exists():
                continue

            records.append(
                {
                    "audio": audio,
                    "text": text,
                    "language": args.language,
                }
            )

    return records


def group_by_parent(records: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[Path(record["audio"]).parent.as_posix()].append(record)
    return grouped


def attach_reference_audio(records: list[dict[str, str]], args: argparse.Namespace) -> None:
    if args.ref_mode == "none":
        return

    if args.ref_ratio < 0 or args.ref_ratio > 1:
        raise ValueError(f"--ref-ratio must be between 0 and 1, got {args.ref_ratio}.")

    rng = random.Random(args.seed)
    grouped = group_by_parent(records)
    fixed_refs = {
        parent: items[0]["audio"]
        for parent, items in grouped.items()
        if items
    }

    for record in records:
        if rng.random() > args.ref_ratio:
            continue

        siblings = grouped[Path(record["audio"]).parent.as_posix()]
        candidates = [item["audio"] for item in siblings if item["audio"] != record["audio"]]
        if not candidates:
            continue

        if args.ref_mode == "same-dir-fixed":
            ref_audio = fixed_refs[Path(record["audio"]).parent.as_posix()]
            if ref_audio == record["audio"]:
                ref_audio = candidates[0]
        else:
            ref_audio = rng.choice(candidates)

        record["ref_audio"] = ref_audio


def write_jsonl(records: Iterable[dict[str, str]], output: str) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    records = read_records(args)
    attach_reference_audio(records, args)
    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
