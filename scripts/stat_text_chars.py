"""Usage: python scripts/stat_text_chars.py path/to/train_with_codec.jsonl"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
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


def duration_stats(path, audio_field, text_field="text"):
    """Return (summary dict, per-record rows of audio path, duration, text)."""
    base = Path(path).resolve().parent
    rows = []
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                audio = Path(record[audio_field])
                if not audio.is_absolute() and not audio.exists():
                    audio = base / audio
                rows.append(dict(audio=str(audio), dur=audio_duration(audio),
                                 text=str(record.get(text_field, ""))))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: cannot read {audio_field}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: no records")
    durations = [row["dur"] for row in rows]
    total = sum(durations)
    return dict(total_duration_sec=total, total_duration_hours=total / 3600,
                mean_duration_sec=total / len(durations),
                min_duration_sec=min(durations), max_duration_sec=max(durations)), rows


# ---------------------------------------------------------------------------
# Distribution diagnostics for TTS finetuning data.
# ---------------------------------------------------------------------------
PUNCT = "，。？！；、…,.?!;"
SENTENCE_END = "。？！.?!"
DURATION_BUCKETS = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 75), (75, None)]
_NON_SPOKEN = re.compile(r"[\s" + re.escape(PUNCT) + r"\"'“”‘’（）()《》【】\[\]—\-·:：]")


def spoken_chars(text):
    """Characters that are actually pronounced (no whitespace/punctuation)."""
    return len(_NON_SPOKEN.sub("", text))


def _quantile(sorted_values, fraction):
    return sorted_values[min(len(sorted_values) - 1, int(fraction * len(sorted_values)))]


def print_report(rows, max_list=20):
    for row in rows:
        row["rate"] = spoken_chars(row["text"]) / row["dur"] if row["dur"] > 0 else 0.0
        row["punct"] = sum(row["text"].count(ch) for ch in PUNCT)
        row["punct_per_sec"] = row["punct"] / row["dur"] if row["dur"] > 0 else 0.0
    total_sec = sum(row["dur"] for row in rows)

    print("\n== 时长分布 ==")
    for low, high in DURATION_BUCKETS:
        selected = [r for r in rows if r["dur"] >= low and (high is None or r["dur"] < high)]
        seconds = sum(r["dur"] for r in selected)
        label = f"{low}-{high}s" if high is not None else f"{low}s+"
        print(f"{label:>8}: {len(selected):>5} 条  {seconds / 60:8.1f} 分钟  "
              f"{seconds / total_sec:6.1%}")

    rates = sorted(r["rate"] for r in rows)
    median = _quantile(rates, 0.5)
    print(f"\n== 语速（发音字/秒） ==  p5={_quantile(rates, .05):.2f}  "
          f"p50={median:.2f}  p95={_quantile(rates, .95):.2f}")
    odd = [r for r in rows if r["rate"] < median * 0.6 or r["rate"] > median * 1.5]
    print(f"语速异常（<0.6x 或 >1.5x 中位数，疑似文本/音频不对齐）: {len(odd)} 条")
    for r in sorted(odd, key=lambda r: r["rate"])[:max_list]:
        print(f"  {r['rate']:5.2f} 字/秒  {r['dur']:6.1f}s  {r['audio']}")

    densities = sorted(r["punct_per_sec"] for r in rows)
    sparse = [r for r in rows if r["punct_per_sec"] < 0.15]
    print(f"\n== 标点 ==  密度中位数 {_quantile(densities, 0.5):.2f} 个/秒（正常朗读约 0.3~0.6）")
    print(f"标点过少（<0.15 个/秒，疑似合并丢失句末标点）: {len(sparse)} 条")
    for r in sparse[:max_list]:
        print(f"  {r['dur']:6.1f}s  {r['audio']}  {r['text'][:40]}…")
    counts = Counter(ch for r in rows for ch in r["text"] if ch in PUNCT)
    print("标点计数:", dict(counts.most_common()))
    ends = sum(counts[ch] for ch in SENTENCE_END)
    if ends:
        question = counts["？"] + counts["?"]
        exclaim = counts["！"] + counts["!"]
        print(f"句末类型占比: 陈述 {(ends - question - exclaim) / ends:.1%}  "
              f"问句 {question / ends:.1%}  感叹 {exclaim / ends:.1%}")


def main():
    parser = argparse.ArgumentParser(description="统计训练文本字数（含标点、空格和换行）及音频时长")
    parser.add_argument("path", nargs="?", default="train_with_codec.jsonl")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--audio-field", default="audio", help="音频文件路径字段名")
    parser.add_argument("--no-report", action="store_true",
                        help="只输出 JSON 汇总，不打印时长/语速/标点分布诊断")
    parser.add_argument("--max-list", type=int, default=20, help="每类异常最多列出的条数")
    args = parser.parse_args()
    stats = dataset_stats(args.path, args.text_field)
    duration_summary, rows = duration_stats(args.path, args.audio_field, args.text_field)
    stats.update(duration_summary)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if not args.no_report:
        print_report(rows, args.max_list)


if __name__ == "__main__":
    main()
