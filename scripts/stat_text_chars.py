"""Usage: python scripts/stat_text_chars.py path/to/train_with_codec.jsonl"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clis.tts_chunking import dataset_stats


def main():
    parser = argparse.ArgumentParser(description="统计训练文本字数（含标点、空格和换行）")
    parser.add_argument("path", nargs="?", default="train_with_codec.jsonl")
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args()
    print(json.dumps(dataset_stats(args.path, args.text_field), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
