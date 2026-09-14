#!/usr/bin/env python3
"""Merge consecutive LJSpeech WAV clips; see merge_ljspeech.md."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Clip:
    path: Path
    text: str
    group: tuple
    number: int | None
    frames: int
    rate: int
    channels: int
    relative_path: Path | None = None


def read_clips(args):
    import soundfile as sf

    clips = []
    seen = set()
    for filename in args.input:
        manifest = Path(filename).resolve()
        root = Path(args.audio_root).resolve() if args.audio_root else manifest.parent / "wavs"
        with manifest.open(encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                columns = line.rstrip("\r\n").split("|")
                if len(columns) <= args.text_column:
                    raise ValueError(f"{manifest}:{line_no}: missing text column")
                name, text = columns[0].strip(), columns[args.text_column].strip()
                if not name or not text:
                    raise ValueError(f"{manifest}:{line_no}: empty path/text")
                path = Path(name.replace("\\", "/"))
                if not path.suffix:
                    path = path.with_suffix(".wav")
                if not path.is_absolute():
                    path = root / path
                path = path.resolve()
                if path in seen:
                    raise ValueError(f"Duplicate input: {path}")
                seen.add(path)
                if path.suffix.lower() != ".wav":
                    raise ValueError(f"Expected WAV: {path}")
                info = sf.info(str(path))
                if info.frames <= 0:
                    raise ValueError(f"Empty audio: {path}")
                match = re.fullmatch(r"(.*?)(\d+)", path.stem)
                prefix = match[1] if match else path.stem
                number = int(match[2]) if match else None
                try:
                    relative_path = path.relative_to(root)
                except ValueError:
                    relative_path = Path(path.parent.name) / path.name
                clips.append(Clip(path, text, (str(manifest), str(path.parent), prefix),
                                  number, info.frames, info.samplerate, info.channels, relative_path))
    if args.order == "natural":
        clips.sort(key=lambda c: (c.group, c.number if c.number is not None else -1))
    return clips


def plan_groups(clips, target, maximum, max_clips):
    """Never cross filename groups, numbering gaps, or audio formats."""
    pending = []
    seconds = 0.0
    for clip in clips:
        duration = clip.frames / clip.rate
        if duration > maximum:
            raise ValueError(f"Single clip exceeds --max-seconds: {clip.path} ({duration:.2f}s)")
        if pending:
            prev = pending[-1]
            contiguous = (clip.group == prev.group and clip.number is not None
                          and prev.number is not None and clip.number == prev.number + 1
                          and (clip.rate, clip.channels) == (prev.rate, prev.channels))
            if (not contiguous or seconds + duration > maximum
                    or (max_clips and len(pending) >= max_clips)):
                yield pending
                pending, seconds = [], 0.0
        pending.append(clip)
        seconds += duration
        if seconds >= target:
            yield pending
            pending, seconds = [], 0.0
    if pending:
        yield pending


def write_group(group, destination):
    import soundfile as sf

    # FLOAT avoids adding PCM quantization; no resampling, fades or silence edits.
    with sf.SoundFile(str(destination), "w", samplerate=group[0].rate,
                      channels=group[0].channels, format="WAV", subtype="FLOAT") as out:
        for clip in group:
            with sf.SoundFile(str(clip.path)) as src:
                if (len(src), src.samplerate, src.channels) != (clip.frames, clip.rate, clip.channels):
                    raise ValueError(f"Source changed after inspection: {clip.path}")
                for block in src.blocks(blocksize=65536, dtype="float32", always_2d=True):
                    out.write(block)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="UTF-8 path|text manifests")
    parser.add_argument("--audio-root", help="Relative-path root; default: manifest directory/wavs")
    parser.add_argument("--output-dir", required=True, help="Must not already exist")
    parser.add_argument("--target-seconds", type=float, default=60)
    parser.add_argument("--min-seconds", type=float, default=30,
                        help="Discard output groups shorter than this; 0: keep all")
    parser.add_argument("--max-seconds", type=float, default=90)
    parser.add_argument("--max-clips", type=int, default=0, help="0: unlimited; 4: at most four clips")
    parser.add_argument("--order", choices=["manifest", "natural"], default="manifest")
    parser.add_argument("--text-column", type=int, default=1, help="1: second column; 2: normalized LJSpeech text")
    parser.add_argument("--text-joiner", default="", help="Default empty for Chinese; use a space for English")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--limit", type=int, default=0, help="Write only first N output groups; 0: all")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and plan without writing")
    args = parser.parse_args(argv)
    if not 0 < args.target_seconds <= args.max_seconds < float("inf"):
        parser.error("Require 0 < target-seconds <= max-seconds < infinity")
    if args.max_clips < 0 or args.limit < 0 or args.text_column < 1:
        parser.error("max-clips/limit must be nonnegative; text-column must be >= 1")
    if not 0 <= args.min_seconds < float("inf"):
        parser.error("Require 0 <= min-seconds < infinity")
    output = Path(args.output_dir).resolve()
    if output.exists() and not args.dry_run:
        parser.error(f"Output already exists; choose a new directory: {output}")
    clips = read_clips(args)
    if not clips:
        parser.error("No input records")
    groups = []
    for group in plan_groups(clips, args.target_seconds, args.max_seconds, args.max_clips):
        duration = sum(c.frames / c.rate for c in group)
        if duration < args.min_seconds:
            print(f"Skipped short group: {group[0].path} -> {group[-1].path.name} "
                  f"({len(group)} clips, {duration:.2f}s < {args.min_seconds:g}s)")
        else:
            groups.append(group)
    if args.limit:
        groups = groups[:args.limit]
    if not groups:
        print("No output groups remain after duration filtering; nothing written.")
        return
    relatives = []
    seen_outputs = set()
    for group in groups:
        source = group[0].relative_path
        relative = Path("wavs") / source.with_name(f"{source.stem}_merge.wav")
        destination = output / relative
        if destination in seen_outputs:
            raise ValueError(f"Output filename collision: {relative}; process inputs separately")
        seen_outputs.add(destination)
        relatives.append(relative.as_posix())
    durations = [sum(c.frames / c.rate for c in g) for g in groups]
    print(f"Selected {sum(map(len, groups))}/{len(clips)} clips -> {len(groups)} outputs; "
          f"duration min/mean/max: {min(durations):.2f}/{sum(durations)/len(durations):.2f}/{max(durations):.2f}s; "
          f"singletons: {sum(len(g) == 1 for g in groups)}")
    if args.dry_run:
        for group, relative in list(zip(groups, relatives))[:10]:
            print(f"{group[0].path.name} -> {group[-1].path.name} ({len(group)} clips) => {relative}")
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "wavs").mkdir()
    with (output / "metadata.txt").open("x", encoding="utf-8") as metadata, \
            (output / "train_raw.jsonl").open("x", encoding="utf-8") as train, \
            (output / "sources.jsonl").open("x", encoding="utf-8") as provenance:
        for index, (group, relative) in enumerate(zip(groups, relatives), 1):
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_group(group, destination)
            text = args.text_joiner.join(c.text for c in group)
            metadata.write(f"{relative}|{text}\n")
            train.write(json.dumps({"audio": destination.as_posix(), "text": text,
                                    "language": args.language}, ensure_ascii=False) + "\n")
            start = 0
            sources = []
            for clip in group:
                sources.append({"audio": clip.path.as_posix(), "text": clip.text,
                                "start_frame": start, "frames": clip.frames})
                start += clip.frames
            provenance.write(json.dumps({"audio": relative, "sample_rate": group[0].rate,
                                         "sources": sources}, ensure_ascii=False) + "\n")
            if index % 100 == 0 or index == len(groups):
                print(f"Written {index}/{len(groups)}")
    print(f"Done: {output}")


if __name__ == "__main__":
    main()
