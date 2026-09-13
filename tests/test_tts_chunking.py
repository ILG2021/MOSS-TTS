import ast
import json
from pathlib import Path
import tempfile
import time
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from clis.tts_chunking import count_chars, dataset_stats, reference_tail, split_text, iter_text_chunks
from clis.tts_chunking import whisper_language, transcribe_reference


class ChunkingTests(unittest.TestCase):
    def test_asr_language_mapping(self):
        for tag, code in [("Chinese", "zh"), ("Cantonese", "yue"),
                          ("Persian (Farsi)", "fa"), ("English", "en"),
                          (" zh ", "zh"), ("Auto (omit)", None), (None, None)]:
            self.assertEqual(whisper_language(tag), code)
            backend = Mock()
            backend.transcribe.return_value = (iter([SimpleNamespace(text="测试。")]), None)
            with patch("clis.tts_chunking.load_asr", return_value=backend):
                self.assertEqual(transcribe_reference("ref.wav", language_tag=tag), "测试。")
            options = backend.transcribe.call_args.kwargs
            self.assertEqual(options["language"], code)
            self.assertEqual(options["initial_prompt"] is not None, code == "zh")
        with self.assertRaises(ValueError):
            whisper_language("not-a-language")
        # Keep the mapping complete when the UI gains another MOSS language.
        path = Path(__file__).resolve().parents[1] / "clis/moss_tts_app.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        choices = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "LANGUAGE_TAG_CHOICES" for t in n.targets))
        for choice in choices.elts:
            if isinstance(choice, ast.Constant):
                self.assertIsNotNone(whisper_language(choice.value))

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
        test_root = tempfile.TemporaryDirectory()
        self.addCleanup(test_root.cleanup)
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
                   __file__=str(Path(test_root.name) / "clis" / "moss_tts_app.py"),
                   iter_text_chunks=iter_text_chunks, reference_tail=reference_tail,
                   transcribe_reference=transcribe, _run_single_inference=infer,
                   MODE_CLONE="Clone", MODE_CONTINUE="Continuation", MODE_CONTINUE_CLONE="Continuation + Clone")
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
        first_reference = Path(refs[1]["reference_audio"])
        self.assertTrue(first_reference.is_file())
        self.assertEqual(first_reference.parent.parent, Path(test_root.name) / "Temp")
        self.assertIn(str(first_reference.parent), status)
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

        # Both segment selectors independently control prompt composition and
        # reference source, including generation without an upload.
        for first_mode in ["Clone", "Continuation + Clone"]:
            for later_mode in ["Clone", "Continuation + Clone"]:
                for reference in ["upload.wav", None]:
                    with self.subTest(first=first_mode, later=later_mode, reference=reference):
                        refs.clear()
                        generated.clear()
                        env["transcribe_reference"] = lambda *args: "参考。"
                        env["run_inference"](
                            "甲乙。丙丁。戊己。庚辛。", reference, first_mode,
                            False, 1, "Chinese", 1, .8, 25, 1, "model", "cpu", "auto", 4096, True,
                            chunk_chars=6, subsequent_mode=later_mode)
                        self.assertGreaterEqual(len(refs), 2)
                        self.assertEqual(refs[0]["reference_audio"], reference)
                        self.assertEqual(refs[0]["mode_with_reference"], first_mode)
                        self.assertEqual(refs[0]["text"].startswith("参考。"),
                                         bool(reference) and first_mode == "Continuation + Clone")
                        for i, call in enumerate(refs[1:], 1):
                            self.assertEqual(call["mode_with_reference"], later_mode)
                            if later_mode == "Clone":
                                self.assertEqual(call["reference_audio"], reference)
                                self.assertFalse(call["text"].startswith("参考。"))
                            else:
                                self.assertEqual(Path(call["reference_audio"]).name, f"reference-{i-1}.wav")
                                self.assertTrue(call["text"].startswith("参考。"))

        # Failed tail ASR falls back without losing text; the following segment
        # tries its new predecessor again. Uploaded transcripts are reused.
        for reference in ["upload.wav", None]:
            refs.clear()
            generated.clear()
            failed_paths, asr_paths = [], []
            def fail_on_tail(path, *args):
                asr_paths.append(path)
                if path != "upload.wav" and len(generated) == 1:
                    failed_paths.append(Path(path))
                    raise ValueError("ASR failed")
                return "参考。"
            env["transcribe_reference"] = fail_on_tail
            (sr, audio), status = env["run_inference"](
                "甲乙。丙丁。戊己。" if reference else "甲乙。丙丁。戊己。庚辛。壬癸。",
                reference, "Continuation + Clone",
                False, 1, "Chinese", 1, .8, 25, 1, "model", "cpu", "auto", 4096, True,
                chunk_chars=6)
            self.assertEqual(len(refs), 3)
            self.assertEqual(refs[1]["reference_audio"], reference)
            self.assertEqual(refs[1]["text"], "参考。丙丁。" if reference else "戊己。庚辛。")
            self.assertEqual(Path(refs[2]["reference_audio"]).name, "reference-1.wav")
            self.assertEqual(refs[2]["text"], "参考。戊己。" if reference else "参考。壬癸。")
            self.assertEqual(asr_paths.count("upload.wav"), 1 if reference else 0)
            self.assertIn("末尾参考转录失败", status)
            self.assertTrue(failed_paths[0].is_file())
            self.assertNotEqual(failed_paths[0].parent, first_reference.parent)
            np.testing.assert_array_equal(audio, np.concatenate(generated))


if __name__ == "__main__":
    unittest.main()
