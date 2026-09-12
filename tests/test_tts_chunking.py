import ast
import json
from pathlib import Path
import tempfile
import time
import unittest
import wave

import numpy as np

from clis.tts_chunking import count_chars, dataset_stats, reference_tail, split_text, iter_text_chunks


class ChunkingTests(unittest.TestCase):
    def test_whitespace_counts(self):
        self.assertEqual(count_chars(" a\n中\t。 "), 7)
        self.assertEqual(split_text("a b\nc", 3), ["a ", "b\nc"])

    def test_preserves_text_and_budget(self):
        for text in ["你好，世界。很长的句子没有任何标点也需要被拆开", "Hello world! A very long sentence with spaces.", "甲\n乙  丙。", "abc     "]:
            for budget in [1, 3, 10, 100]:
                chunks = split_text(text, budget)
                self.assertEqual("".join(chunks), text)
                self.assertTrue(all(0 < count_chars(c) <= budget for c in chunks))
        self.assertEqual(split_text("甲乙。丙丁。戊己。", 6), ["甲乙。丙丁。", "戊己。"])

    def test_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_with_codec.jsonl"
            path.write_text('\n'.join(json.dumps({"text": s}) for s in ["你好。", "a b"]) + '\n', encoding="utf-8")
            stats = dataset_stats(path)
            self.assertEqual(stats["mean_chars"], 3)
            self.assertEqual(stats["recommended_chunk_chars"], 3)
            path.write_text('{"audio": "x"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, ':1:'):
                dataset_stats(path)

    def test_quiet_tail(self):
        sr = 1000
        audio = np.ones(15000, dtype=np.float32)
        audio[6024:6400] = 0
        tail = reference_tail(audio, sr)
        self.assertEqual(len(tail), 8976)
        self.assertLessEqual(len(tail), 10 * sr)
        np.testing.assert_array_equal(reference_tail(audio[:3000], sr), audio[:3000])

    def test_rolling_reference_and_prompt(self):
        # Load only the orchestration function, avoiding GPU/model imports.
        path = Path(__file__).resolve().parents[1] / "clis/moss_tts_app.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_inference")
        generated = []
        refs = []
        def infer(**kwargs):
            refs.append(kwargs)
            audio = np.full(12000, 0.1 * len(refs), dtype=np.float32)
            generated.append(audio)
            return (1000, audio), "ok"
        def transcribe(path, *args, **kwargs):
            if path != "upload.wav":
                with wave.open(path, "rb") as reader:
                    sr = reader.getframerate()
                    audio = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2") / 32768
                self.assertLessEqual(len(audio), sr * 10)
                self.assertAlmostEqual(float(audio.mean()), float(generated[-1].mean()), places=4)
            return ["参考。", "参考一。", "参考二。"][len(refs)]
        env = dict(np=np, Path=Path, time=time, tempfile=tempfile, count_chars=count_chars,
                   iter_text_chunks=iter_text_chunks, reference_tail=reference_tail,
                   transcribe_reference=transcribe, _run_single_inference=infer,
                   MODE_CONTINUE="Continuation", MODE_CONTINUE_CLONE="Continuation + Clone")
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), env)
        (sr, audio), status = env["run_inference"](
            "甲乙。丙丁。戊己。", "upload.wav", "Continuation + Clone",
            False, 1, "Chinese", 1, .8, 25, 1, "model", "cpu", "auto", 4096, True,
            chunk_chars=8)
        self.assertEqual(len(refs), 3)
        self.assertEqual(refs[0]["reference_audio"], "upload.wav")
        self.assertEqual([r["text"] for r in refs], ["参考。甲乙。", "参考一。丙丁。", "参考二。戊己。"])
        self.assertTrue(all(len(r["text"]) <= 8 for r in refs))
        np.testing.assert_array_equal(audio, np.concatenate(generated))
        self.assertFalse(Path(refs[1]["reference_audio"]).exists())
        env["transcribe_reference"] = lambda *args: "参考文本实在太长。"
        previous_calls = len(refs)
        with self.assertRaisesRegex(ValueError, "已用完分段总字数"):
            env["run_inference"](
                "甲乙。", "upload.wav", "Continuation + Clone",
                False, 1, "Chinese", 1, .8, 25, 1, "model", "cpu", "auto", 4096, True,
                chunk_chars=8)
        self.assertEqual(len(refs), previous_calls)

        # Clone still reserves reference characters without prepending them.
        for mode, reference in [("Clone", "upload.wav"), ("Continuation + Clone", None)]:
            refs.clear()
            generated.clear()
            asr_calls = []
            def fixed_transcript(*args):
                asr_calls.append(args[0])
                return "参考。"
            env["transcribe_reference"] = fixed_transcript
            env["run_inference"](
                "     甲乙。丙丁。" if reference else "     甲乙。丙丁。戊己。", reference, mode,
                False, 1, "Chinese", 1, .8, 25, 1, "model", "cpu", "auto", 4096, True,
                chunk_chars=6)
            self.assertEqual(refs[0]["text"], "甲乙。" if reference else "甲乙。丙丁。")
            self.assertEqual(refs[1]["text"], "参考。丙丁。" if reference else "参考。戊己。")
            self.assertEqual(len(asr_calls), 2 if reference else 1)

        # A failure after creating a rolling reference must also clean it up.
        refs.clear()
        generated.clear()
        def fail_on_tail(path, *args):
            if path == "upload.wav":
                return "参考。"
            refs.append({"failed_reference": path})
            raise ValueError("ASR failed")
        env["transcribe_reference"] = fail_on_tail
        with self.assertRaisesRegex(ValueError, "ASR failed"):
            env["run_inference"](
                "甲乙。丙丁。", "upload.wav", "Continuation + Clone",
                False, 1, "Chinese", 1, .8, 25, 1, "model", "cpu", "auto", 4096, True,
                chunk_chars=6)
        self.assertFalse(Path(refs[-1]["failed_reference"]).exists())


if __name__ == "__main__":
    unittest.main()
