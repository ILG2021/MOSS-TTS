"""Usage: python scripts/stat_text_chars.py path/to/train_with_codec.jsonl"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clis.tts_chunking import dataset_stats


def audio_duration(file_path):
    """Duration in seconds, read from the audio file header."""
    try:
        import soundfile as sf
        info = sf.info(str(file_path))
        return info.frames / info.samplerate
    except ImportError:
        import torchaudio
        info = torchaudio.info(str(file_path))
        return info.num_frames / info.sample_rate


def duration_stats(path, audio_field):
    base = Path(path).resolve().parent
    durations = []
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                audio = Path(json.loads(line)[audio_field])
                if not audio.is_absolute() and not audio.exists():
                    audio = base / audio
                durations.append(audio_duration(audio))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: cannot read {audio_field}: {exc}") from exc
    if not durations:
        raise ValueError(f"{path}: no records")
    total = sum(durations)
    return dict(total_duration_sec=total, total_duration_hours=total / 3600,
                mean_duration_sec=total / len(durations),
                min_duration_sec=min(durations), max_duration_sec=max(durations))


def main():
    parser = argparse.ArgumentParser(description="统计训练文本字数（含标点、空格和换行）及音频时长")
    parser.add_argument("path", nargs="?", default="train_with_codec.jsonl")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--audio-field", default="audio", help="音频文件路径字段名")
    args = parser.parse_args()
    stats = dataset_stats(args.path, args.text_field)
    stats.update(duration_stats(args.path, args.audio_field))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
