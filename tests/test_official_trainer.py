"""CPU regressions for frame contribution, temporal boundaries, and checkpoint safety."""
import importlib.util
import logging
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import torch
except ImportError:
    torch = None

entry_spec = importlib.util.spec_from_file_location("train_official_entry", str(ROOT / "train_official.py"))
entry = importlib.util.module_from_spec(entry_spec)
entry_spec.loader.exec_module(entry)


if torch is not None:
    from lib.helpers.trainer_helper import Trainer, load_initial_checkpoint

    class TemporalDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.memory_tokens = None
            self.TTM_model = types.SimpleNamespace(simpleDNC=types.SimpleNamespace(current_flag=0, split_memory_tokens=[]))

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
            self.decoder = TemporalDecoder()
            self.observed_phases = []

        def forward(self, images, calibs, sizes, text):
            dnc = self.decoder.TTM_model.simpleDNC
            self.observed_phases.append(dnc.current_flag)
            value = self.weight * images.mean().reshape(1, 1)
            if self.decoder.memory_tokens is not None:
                # This still references the preceding graph unless detached.
                value = value + self.decoder.memory_tokens * 0
            self.decoder.memory_tokens = value
            dnc.current_flag += 1
            dnc.split_memory_tokens = [value]
            return {"value": value}

    class Criterion(torch.nn.Module):
        weight_dict = {"loss_regression": 1.}

        def forward(self, outputs, targets):
            return {"loss_regression": (outputs["value"] - targets[0]["value"]).square().mean()}

    def frame(x, y=0.):
        return {
            "images": torch.full((1, 3, 2, 2), float(x)),
            "calibs": torch.cat((torch.eye(3), torch.zeros(3, 1)), dim=1).unsqueeze(0),
            "image_sizes": torch.tensor([[2., 2.]]), "text": ["car"],
            "target": {"value": torch.tensor([[y]])},
        }


class EntryTests(unittest.TestCase):
    def test_parser_requires_no_torch_import_for_help_and_bounds(self):
        args = entry.parse_args([
            "--config", "cfg.yaml", "--source-project", "source", "--checkpoint", "task.pth",
            "--data-root", "official", "--calibration-root", "calib", "--output", "run",
            "--label-center-reference", "geometric_center", "--max-frames-per-clip", "1", "--lr", ".003",
        ])
        self.assertEqual(args.max_frames_per_clip, 1)
        self.assertEqual(args.lr, .003)
        self.assertEqual(args.trainable, "head")

    def test_explicit_split_rejects_shared_recording_or_frame(self):
        a = types.SimpleNamespace(clips=[{"sequence_id": "0001", "frames": ["a.jpg"]}])
        b = types.SimpleNamespace(clips=[{"sequence_id": "0002", "frames": ["a.jpg"]}])
        with self.assertRaisesRegex(ValueError, "overlap"):
            entry._check_disjoint(a, b)
        b.clips[0]["frames"] = ["b.jpg"]
        entry._check_disjoint(a, b)
        b.clips[0]["sequence_id"] = "0001"
        with self.assertRaisesRegex(ValueError, "overlap"):
            entry._check_disjoint(a, b)


@unittest.skipIf(torch is None, "PyTorch unavailable; run these tests in the training environment")
class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.model = ToyModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.1)

    def trainer(self, train, validation=None, **overrides):
        cfg = {"save_path": self.temporary.name, "max_epoch": 1, "save_frequency": 1,
               "grad_clip_norm": 0, "frozen_modules_eval": True}
        cfg.update(overrides)
        return Trainer(cfg, self.model, self.optimizer, train, validation, None, None,
                       logging.getLogger("trainer_test"), Criterion(), "toy")

    def test_every_frame_contributes_mean_gradient_and_old_graph_is_released(self):
        trainer = self.trainer([[frame(1), frame(2)]])
        self.assertAlmostEqual(trainer.train_one_epoch(0), 2.5)
        # Mean derivative = (2*1**2 + 2*2**2)/2 = 5, so w <- 1 - .1*5.
        self.assertAlmostEqual(self.model.weight.item(), .5, places=6)
        self.assertEqual(trainer.last_train_metrics["frames"], 2)
        self.assertEqual(self.model.observed_phases, [0, 1])
        self.assertIsNone(self.model.decoder.memory_tokens)
        self.assertEqual(self.model.decoder.TTM_model.simpleDNC.current_flag, 0)
        self.assertEqual(self.model.decoder.TTM_model.simpleDNC.split_memory_tokens, [])

    def test_separate_clips_reset_memory_and_phase(self):
        trainer = self.trainer([[frame(1), frame(2)], [frame(3)]])
        trainer.train_one_epoch(0)
        self.assertEqual(self.model.observed_phases, [0, 1, 0])
        self.assertEqual(trainer.last_train_metrics["batches"], 2)

    def test_validation_returns_average_before_clearing_and_preserves_grad_mode(self):
        trainer = self.trainer([], [[frame(1), frame(2)]])
        self.assertTrue(torch.is_grad_enabled())
        self.assertAlmostEqual(trainer.val_one_epoch(0), 2.5)
        self.assertTrue(torch.is_grad_enabled())
        self.assertIsNone(self.model.weight.grad)
        self.assertEqual(trainer.last_val_metrics["frames"], 2)

    def test_frame_mode_smoke_and_checkpoint_can_resume(self):
        trainer = self.trainer([frame(1)], [frame(1)])
        history = trainer.train()
        self.assertEqual(history[0]["epoch"], 1)
        checkpoint = Path(trainer.output_dir) / "checkpoint_latest.pth"
        self.assertTrue(checkpoint.is_file())
        saved_weight = self.model.weight.detach().clone()
        with torch.no_grad():
            self.model.weight.fill_(99.)
        trainer.resume(checkpoint)
        self.assertTrue(torch.equal(self.model.weight, saved_weight))
        self.assertEqual(trainer.epoch, 1)
        self.assertTrue((Path(trainer.output_dir) / "checkpoint_best.pth").is_file())

    def test_no_test_targets_and_no_supervision_cannot_silently_train(self):
        sample = frame(1)
        del sample["target"]
        with self.assertRaisesRegex(ValueError, "without targets"):
            self.trainer([sample]).train_one_epoch(0)
        self.assertIsNone(self.model.decoder.memory_tokens)

    def test_smoke_limits_bound_batches_and_frames(self):
        trainer = self.trainer([[frame(1), frame(2)], [frame(3)]],
                               max_train_batches=1, max_frames_per_clip=1)
        trainer.train_one_epoch(0)
        self.assertEqual(trainer.last_train_metrics["frames"], 1)
        self.assertEqual(trainer.last_train_metrics["batches"], 1)

    def test_legacy_prepare_targets_uses_actual_frame_count(self):
        targets = {"other": [{"mask_2d": torch.tensor([[True, False]]),
                               "depth": torch.tensor([[[2.], [99.]]])} for _ in range(2)]}
        result = Trainer.prepare_targets(targets, 1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0]["depth"].tolist(), [[2.]])

    def test_initialization_only_allows_the_new_head_to_be_absent(self):
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(2, 2)
        model.official_geometry_head = torch.nn.Linear(2, 2)
        path = Path(self.temporary.name) / "initial.pth"
        state = {key: value for key, value in model.state_dict().items() if not key.startswith("official_geometry_head.")}
        torch.save({"model_state": state}, path)
        result = load_initial_checkpoint(model, path, allow_new_geometry_head=True)
        self.assertTrue(result["missing_new_head_keys"])
        del state["backbone.weight"]
        torch.save({"model_state": state}, path)
        with self.assertRaisesRegex(RuntimeError, "backbone.weight"):
            load_initial_checkpoint(model, path, allow_new_geometry_head=True)


if __name__ == "__main__":
    unittest.main()
