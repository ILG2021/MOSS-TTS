import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

spec = importlib.util.spec_from_file_location(
    "merge_ljspeech", Path(__file__).resolve().parents[1] / "scripts/merge_ljspeech.py"
)
merge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = merge
spec.loader.exec_module(merge)


class MergeTests(unittest.TestCase):
    def test_waveform_text_gaps_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wavs" / "音频").mkdir(parents=True)
            lines = []
            for number in (1, 2, 4):
                sf.write(root / "wavs" / "音频" / f"切片_{number}.wav",
                         np.full((800, 2), number / 10, dtype=np.float32),
                         8000, subtype="FLOAT")
                lines.append(f"音频\\切片_{number}.wav|第{number}句，")
            manifest = root / "metadata.txt"
            manifest.write_text("\n".join(lines), encoding="utf-8-sig")
            output = root / "merged"
            args = ["--input", str(manifest), "--output-dir", str(output),
                    "--target-seconds", "0.25", "--max-seconds", "0.3", "--min-seconds", "0"]
            merge.main(args + ["--dry-run"])
            self.assertFalse(output.exists())
            merge.main(args)
            rows = [json.loads(line) for line in (output / "train_raw.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(Path(rows[0]["audio"]), output / "wavs/音频/切片_1_merge.wav")
            self.assertEqual(rows[0]["text"], "第1句，第2句，")
            audio, rate = sf.read(rows[0]["audio"], dtype="float32", always_2d=True)
            self.assertEqual((audio.shape, rate), ((1600, 2), 8000))
            np.testing.assert_array_equal(audio[:800], np.full((800, 2), 0.1, np.float32))
            np.testing.assert_array_equal(audio[800:], np.full((800, 2), 0.2, np.float32))
            with self.assertRaises(SystemExit):
                merge.main(args)

    def test_short_groups_filtered_and_exact_minimum_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wavs").mkdir()
            for number, seconds in ((1, 10), (2, 20), (4, 5)):
                sf.write(root / "wavs" / f"clip{number:03d}.wav", np.zeros(seconds * 8000), 8000)
            manifest = root / "metadata.txt"
            manifest.write_text("clip001|a\nclip002|b\nclip004|c\n", encoding="utf-8")
            output = root / "out"
            args = ["--input", str(manifest), "--output-dir", str(output)]
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                merge.main(args)
            self.assertIn("Skipped short group:", log.getvalue())
            self.assertIn("5.00s < 30s", log.getvalue())
            self.assertEqual((output / "metadata.txt").read_text(encoding="utf-8"),
                             "wavs/clip001_merge.wav|ab\n")
            empty_output = root / "empty"
            with contextlib.redirect_stdout(log):
                merge.main(["--input", str(manifest), "--output-dir", str(empty_output),
                            "--min-seconds", "40"])
            self.assertFalse(empty_output.exists())
            self.assertIn("nothing written", log.getvalue())

    def test_duration_format_and_number_boundaries(self):
        def clip(n, frames=20, rate=10, group=("a",)):
            return merge.Clip(Path(str(n)), "text", group, n, frames, rate, 1)
        groups = list(merge.plan_groups([clip(1), clip(2), clip(3), clip(4, rate=20)], 5, 5, 0))
        self.assertEqual([len(g) for g in groups], [2, 1, 1])
        with self.assertRaises(ValueError):
            list(merge.plan_groups([clip(1, frames=100)], 5, 5, 0))


if __name__ == "__main__":
    unittest.main()
