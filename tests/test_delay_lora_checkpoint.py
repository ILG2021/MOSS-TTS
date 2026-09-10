"""Checkpoint control-flow tests, runnable without the GPU training dependencies.

Run: python -m unittest discover -s tests -p test_delay_lora_checkpoint.py
These tests do not replace an Accelerate/PEFT GPU resume integration test.
"""
import __future__
import argparse
import ast
import hashlib
import json
import math
import tempfile
import time
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock


def load_helpers():
    source = Path(__file__).resolve().parents[1] / "moss_tts_delay/finetuning/sft_lora.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"read_resume_checkpoint", "dataset_fingerprint", "configure_checkpoint_hooks",
             "save_checkpoint", "_training_loop"}
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name in names], type_ignores=[])
    scope = dict(Path=Path, json=json, hashlib=hashlib, math=math, time=time,
                 DistributedType=types.SimpleNamespace(FSDP="FSDP", DEEPSPEED="DEEPSPEED"),
                 broadcast_object_list=lambda items: items,
                 format_timestamp=lambda: "now", copy_support_files=Mock(), copy_inference_assets=Mock())
    exec(compile(module, str(source), "exec", flags=__future__.annotations.compiler_flag), scope)
    return scope


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.scope = load_helpers()

    def test_restore_configuration_and_reject_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = argparse.Namespace(resume_from_checkpoint=folder, lora_resume_adapter=None,
                                      learning_rate=9, output_dir="new", train_jsonl="data", save_steps=10)
            with self.assertRaises(ValueError):
                self.scope["read_resume_checkpoint"](args)
            (root / "training_state").mkdir()
            (root / "trainer_state.json").write_text(json.dumps({"format_version": 1, "global_step": 3}))
            (root / "finetune_args.json").write_text(json.dumps({"learning_rate": 0.0001, "output_dir": "old"}))
            progress = self.scope["read_resume_checkpoint"](args)
            self.assertEqual(progress["global_step"], 3)
            self.assertEqual(args.learning_rate, 0.0001)
            self.assertEqual(args.output_dir, "new")
            args.lora_resume_adapter = "adapter"
            with self.assertRaises(ValueError):
                self.scope["read_resume_checkpoint"](args)

    def test_fingerprint_detects_content_and_order_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "a", Path(folder) / "b"
            a.write_text("a")
            b.write_text("b")
            fingerprint = self.scope["dataset_fingerprint"]
            first = fingerprint([a, b])
            self.assertNotEqual(first, fingerprint([b, a]))
            b.write_text("c")
            self.assertNotEqual(first, fingerprint([a, b]))

    def test_state_hooks_skip_only_replicated_model_weights(self):
        accelerator = Mock(distributed_type="NO")
        self.scope["configure_checkpoint_hooks"](accelerator)
        weights, models = [1], [2]
        accelerator.register_save_state_pre_hook.call_args.args[0](models, weights, "path")
        self.assertEqual(weights, [])
        accelerator.register_load_state_pre_hook.call_args.args[0](models, "path")
        self.assertEqual(models, [])
        sharded = Mock(distributed_type="FSDP")
        self.scope["configure_checkpoint_hooks"](sharded)
        sharded.register_save_state_pre_hook.assert_not_called()

    def test_completion_marker_is_written_only_after_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "checkpoint"
            accelerator = Mock(is_main_process=True, distributed_type="NO")
            accelerator.save_state.side_effect = RuntimeError("disk full")
            with self.assertRaises(RuntimeError):
                self.scope["save_checkpoint"](accelerator, Mock(), "base", "codec", root, {}, {})
            self.assertFalse((root / "trainer_state.json").exists())
            with self.assertRaises(FileExistsError):
                self.scope["save_checkpoint"](accelerator, Mock(), "base", "codec", root, {}, {})
            other = Path(folder) / "complete"
            accelerator.save_state.side_effect = lambda path: Path(path).mkdir()
            self.scope["save_checkpoint"](accelerator, Mock(), "base", "codec", other, {}, {"global_step": 2})
            self.assertEqual(json.loads((other / "trainer_state.json").read_text())["global_step"], 2)

    def run_loop(self, resume=None):
        class Loader(list):
            generator = Mock()
            def set_epoch(self, epoch):
                self.epoch = epoch

        loader = Loader([dict(input_ids=i, attention_mask=1, labels=1) for i in range(4)])
        accelerator = Mock(num_processes=1, distributed_type="NO", optimizer_step_was_skipped=False)
        counter = 0

        @contextmanager
        def accumulate(model):
            nonlocal counter
            counter += 1
            accelerator.sync_gradients = counter % 2 == 0
            yield

        accelerator.accumulate = accumulate
        accelerator.skip_first_batches.side_effect = lambda data, count: Loader(data[count:])
        args = argparse.Namespace(gradient_accumulation_steps=2, num_epochs=2, max_grad_norm=0,
                                  logging_steps=100, save_steps=1, seed=42, model_path="base", codec_path="codec")
        model, optimizer, scheduler, save = Mock(), Mock(), Mock(), Mock()
        self.scope["save_checkpoint"] = save
        self.scope["_training_loop"](accelerator, args, model, loader, optimizer, scheduler,
                                     None, 4, 2, Path("output"), {"dataset_fingerprint": "hash"}, None, resume)
        return model, scheduler, save

    def test_mid_epoch_resume_skips_completed_batches(self):
        model, scheduler, save = self.run_loop({"global_step": 1, "epoch": 0, "next_batch": 2})
        self.assertEqual([call.kwargs["input_ids"] for call in model.call_args_list], [2, 3, 0, 1, 2, 3])
        self.assertEqual(scheduler.step.call_count, 3)
        self.assertEqual(save.call_args.kwargs["progress"]["global_step"], 4)
        self.assertEqual(save.call_args.kwargs["progress"]["epoch"], 2)
        self.assertEqual(save.call_args.kwargs["progress"]["next_batch"], 0)

    def test_finished_checkpoint_does_not_train(self):
        model, scheduler, save = self.run_loop({"global_step": 4, "epoch": 2, "next_batch": 0})
        model.assert_not_called()
        scheduler.step.assert_not_called()
        save.assert_not_called()

    def test_partial_accumulation_checkpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_loop({"global_step": 1, "epoch": 0, "next_batch": 1})


if __name__ == "__main__":
    unittest.main()
