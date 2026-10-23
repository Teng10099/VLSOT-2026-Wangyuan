"""CPU checks for official export inputs and tensor-only v2 interface.

These do not claim that an ONNX model has been produced or ORT parity passed.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import export_official_v2_onnx as export

try:
    import torch
except ImportError:
    torch = None


class ExportInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, partition, frames):
        directory = self.root / partition
        (directory / "json").mkdir(parents=True, exist_ok=True)
        (directory / "img").mkdir(exist_ok=True)
        for image in ("a.jpg", "b.jpg"):
            (directory / "img" / image).write_bytes(b"public-image-placeholder")
        record = dict(videoID=12, description="A WHITE CAR", frame0=frames[0], frame1=frames[1])
        (directory / "json" / "000012.json").write_text(json.dumps(record), encoding="utf-8")
        return record

    def test_official_test_strings_are_read_without_target_fields(self):
        self.fixture("test", ["old/a.jpg", r"C:\images\b.jpg"])
        text, paths = export.read_public_clip(self.root, "test", "000012")
        self.assertEqual(text, "a white car")
        self.assertEqual([path.name for path in paths], ["a.jpg", "b.jpg"])

    def test_train_tail_is_ignored_instead_of_used_as_prediction(self):
        # Deliberately nonnumeric tail: input preparation must never inspect it.
        self.fixture("train", [["a.jpg", {"not_a_target": True}], ["b.jpg", "poison"]])
        text, paths = export.read_public_clip(self.root, "train", "000012")
        self.assertEqual([path.name for path in paths], ["a.jpg", "b.jpg"])

    def test_test_annotation_array_is_rejected(self):
        self.fixture("test", [["a.jpg", 99], "b.jpg"])
        with self.assertRaisesRegex(ValueError, "public input image"):
            export.read_public_clip(self.root, "test", "000012")

    def test_requires_two_distinct_images_and_matching_id(self):
        record = self.fixture("test", ["a.jpg", "a.jpg"])
        with self.assertRaisesRegex(ValueError, "different"):
            export.read_public_clip(self.root, "test", "000012")
        record.update(videoID=13, frame1="b.jpg")
        (self.root / "test/json/000012.json").write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "videoID"):
            export.read_public_clip(self.root, "test", "000012")

    def test_comparator_requires_complete_interface_and_restores_legacy_names(self):
        arrays = [np.zeros((1,), dtype=np.float32) for _ in export.OUTPUTS]
        arrays[-1] = np.zeros((1,), dtype=np.int64)
        previous = export.legacy.OUTPUTS
        report = export.compare_outputs(arrays, arrays, np, .001, .001)
        self.assertEqual(set(report), set(export.OUTPUTS))
        self.assertEqual(export.legacy.OUTPUTS, previous)
        with self.assertRaisesRegex(ValueError, "exactly"):
            export.compare_outputs(arrays[:-1], arrays, np, .001, .001)
        self.assertNotIn("pred_angle", export.OUTPUTS)
        self.assertNotIn("target_encoding", export.OUTPUTS)
        self.assertNotIn("calibs", export.INPUTS)

    def test_export_rejects_checkpoint_mask_and_resolution_mismatch(self):
        config = {"model": {"target_encoding": export.ENCODING, "input_size_hw": [544, 960],
                            "official_correct_padding_mask": True}}
        checkpoint = {"target_encoding": export.ENCODING,
                      "model_options": {"input_size_hw": [544, 960], "official_correct_padding_mask": True}}
        contract = export.validate_checkpoint_contract(config, checkpoint)
        self.assertTrue(contract["model_options"]["official_correct_padding_mask"])
        checkpoint["model_options"]["official_correct_padding_mask"] = False
        with self.assertRaisesRegex(ValueError, "padding_mask"):
            export.validate_checkpoint_contract(config, checkpoint)
        checkpoint["model_options"]["official_correct_padding_mask"] = True
        checkpoint["model_options"]["input_size_hw"] = [1080, 1920]
        with self.assertRaisesRegex(ValueError, "resolutions disagree"):
            export.validate_checkpoint_contract(config, checkpoint)

    def test_export_accepts_recorded_old_v2_false_default_but_does_not_invent_resolution(self):
        config = {"model": {"target_encoding": export.ENCODING}}
        checkpoint = {"target_encoding": export.ENCODING}
        with self.assertRaisesRegex(ValueError, "input size is missing"):
            export.validate_checkpoint_contract(config, checkpoint)
        manifest = {"target_encoding": export.ENCODING, "args": {"input_width": 960, "input_height": 544}}
        contract = export.validate_checkpoint_contract(config, checkpoint, [manifest])
        self.assertFalse(contract["model_options"]["official_correct_padding_mask"])


@unittest.skipIf(torch is None, "PyTorch unavailable; CPU-only wrapper check in training environment")
class WrapperTests(unittest.TestCase):
    def test_v2_wrapper_exports_tensors_and_explicit_state_without_angle(self):
        class TTM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = {"model": {"preprocess_mode": "3d", "memory_mode": "TL", "process_unit": "transformer",
                                "memory_tokens_size": 128, "num_blocks": 8, "dim": 256,
                                "summerize_num_tokens": 1, "Read_use_positional_embedding": True,
                                "Write_use_positional_embedding": True, "load_memory_add_noise": True,
                                "load_memory_add_noise_mode": "normal"}}

        class Model(torch.nn.Module):
            target_encoding = export.ENCODING

            def __init__(self):
                super().__init__()
                decoder = torch.nn.Module()
                decoder.TTM_model = TTM()
                decoder.memory_tokens = None
                self.mono3dvg_transformer = torch.nn.Module()
                self.mono3dvg_transformer.decoder = decoder
                self.mono3dvg_transformer.TextGuidedAdapter = types.SimpleNamespace(dynamic_shapes=True)

            def forward(self, images, calibs, sizes, text):
                self.asserted_no_calibs = calibs is None and sizes is None
                decoder = self.mono3dvg_transformer.decoder
                value, decoder.memory_tokens = decoder.TTM_model(images, decoder.memory_tokens)
                return {"pred_logits": value, "pred_boxes": value, "pred_3d_dim": value,
                        "pred_depth": value, "target_encoding": export.ENCODING}

        def step(ttm, inputs, memory, phase, read, write, noise, torch):
            return inputs, memory + noise, (phase + 1) % 8

        model = Model()
        with patch.object(export.legacy, "tensor_ttm_step", step):
            wrapper = export.make_wrapper(model, torch)
            outputs = wrapper(torch.ones(1, 3, 2, 2), torch.zeros(1, 110, dtype=torch.int64),
                              torch.ones(1, 110, dtype=torch.int64), torch.zeros(1, 128, 256),
                              torch.tensor([7]), torch.zeros(1, 17, 256), torch.zeros(1, 50, 256),
                              torch.ones(1, 128, 256))
        self.assertEqual(len(outputs), 6)
        self.assertTrue(all(torch.is_tensor(value) for value in outputs))
        self.assertTrue(model.asserted_no_calibs)
        self.assertEqual(outputs[-1].item(), 0)
        self.assertEqual(outputs[-2].mean().item(), 1.)


if __name__ == "__main__":
    unittest.main()
