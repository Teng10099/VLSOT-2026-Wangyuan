"""CPU contracts for official VLSOT inputs and experimental camera targets."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
SPEC = importlib.util.spec_from_file_location(
    "official_vlsot_dataset", str(PROJECT / "lib/datasets/mono3drefer/official_vlsot_dataset.py"))
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class OfficialVLSOTDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "dataset"
        self.calib = Path(self.tmp.name) / "calib"
        self.R = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
        self.K = np.array([[100., 0., 50.], [0., 80., 40.], [0., 0., 1.]])

    def clip(self, clip_id, sequence, partition="train", center=(20., -2., 1.)):
        folder = self.root / partition
        (folder / "json").mkdir(parents=True, exist_ok=True)
        (folder / "img").mkdir(exist_ok=True)
        obj = {"videoID": clip_id, "sequence_id": sequence, "description": "A White CAR"}
        if partition == "train":
            obj["type"] = "CAR"
        for i in range(30):
            name = "{}_{}.jpg".format(clip_id, i)
            obj["frame{}".format(i)] = [name, 4., 2., 1.5] + list(center) if partition == "train" else name
            Image.new("RGB", (100, 80), (128, 128, 128)).save(folder / "img" / name)
            for kind, document in (
                ("camera_intrinsic", {"cam_K": self.K.reshape(-1).tolist()}),
                ("virtuallidar_to_camera", {"rotation": self.R.tolist(), "translation": [[0.], [0.], [0.]]}),
            ):
                out = self.calib / kind / Path(name).with_suffix(".json")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(document), encoding="utf-8")
        path = folder / "json" / (clip_id + ".json")
        path.write_text(json.dumps(obj), encoding="utf-8")
        return path

    def dataset(self, **kwargs):
        kwargs.setdefault("input_size_wh", (50, 32))
        kwargs.setdefault("label_center_reference", "geometric_center")
        return adapter.OfficialVLSOTDataset(self.root, self.calib, **kwargs)

    def test_official_order_camera_depth_resize_and_no_half_height_shift(self):
        self.clip("000000", "0000")
        dataset = self.dataset()
        sample = dataset[0]
        self.assertEqual(len(dataset), 30)
        self.assertEqual(sample["images"].shape, (3, 32, 50))
        self.assertEqual(sample["images"].dtype, np.float32)
        self.assertTrue(sample["images"].flags.c_contiguous)
        np.testing.assert_allclose(sample["calibs"][:, :3], np.diag([.5, .4, 1.]) @ self.K)
        np.testing.assert_array_equal(sample["calibs"][:, 3], 0.)
        np.testing.assert_array_equal(sample["image_sizes"], [50., 32.])
        target = sample["target"]
        np.testing.assert_allclose(target["size_3d"], [[1.5, 2., 4.]])
        np.testing.assert_allclose(target["native_center_xyz"], [[20., -2., 1.]])
        np.testing.assert_allclose(target["camera_center_xyz"], [[2., -1., 20.]])
        np.testing.assert_allclose(target["depth"], [[20.]])
        np.testing.assert_allclose(target["projected_center"], [[.6, .45]])
        self.assertEqual(sample["text"], "a white car")
        self.assertFalse(sample["label_convention_verified"])
        self.assertFalse(target["has_bbox2d"][0])
        self.assertNotIn("boxes", target)
        self.assertNotIn("heading_bin", target)

    def test_no_silent_clipping_of_outside_projected_center(self):
        self.clip("000000", "0000", center=(20., -200., 1.))
        target = self.dataset()[0]["target"]
        self.assertGreater(target["projected_center"][0, 0], 1.)
        self.assertFalse(target["center_in_input_bounds"][0])

    def test_negative_box_offsets_remain_signed_and_reconstruct_valid_box(self):
        encoded = adapter.signed_lrtb_from_box([.8, .2], [.1, .3, .4, .5])
        self.assertLess(encoded[3], 0.)
        self.assertLess(encoded[4], 0.)
        x, y, left, right, top, bottom = encoded
        np.testing.assert_allclose([x - left, y - top, x + right, y + bottom], [.1, .3, .4, .5], atol=1e-7)

    def test_test_frames_use_only_public_inputs_and_have_no_target(self):
        self.clip("000237", "0000", partition="test")
        sample = self.dataset(split="test")[0]
        self.assertNotIn("target", sample)
        self.assertEqual(sample["video_id"], "000237")
        self.assertEqual(sample["frame_index"], 1)
        self.assertEqual(sample["image_name"], "000237_0.jpg")
        path = self.root / "test/json/000237.json"
        obj = json.loads(path.read_text())
        obj["frame0"] = ["000237_0.jpg", "never interpret a test target tail"]
        path.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "TEST frame must be an image string"):
            self.dataset(split="test")

    def test_clip_mode_is_chronological_and_calibration_is_per_frame(self):
        self.clip("000000", "0000")
        path = self.calib / "camera_intrinsic/000000_1.json"
        K = self.K.copy()
        K[0, 0] *= 2
        path.write_text(json.dumps({"cam_K": K.reshape(-1).tolist()}))
        dataset = self.dataset(sample_mode="clip", frame_indices=(0, 1))
        self.assertEqual(len(dataset), 1)
        clip = dataset[0]
        self.assertEqual([frame["frame_index"] for frame in clip], [1, 2])
        self.assertNotEqual(clip[0]["calibs"][0, 0], clip[1]["calibs"][0, 0])
        with self.assertRaises(IndexError):
            dataset[-1]

    def test_training_only_group_split_and_explicit_val_subset(self):
        self.clip("000000", "0000")
        self.clip("000001", "0000")
        self.clip("000002", "0001")
        # Poison test JSON: grouping must never open it.
        p = self.root / "test/json/000237.json"
        p.parent.mkdir(parents=True)
        p.write_text("not even JSON")
        split = adapter.split_train_by_sequence(self.root, seed=444)
        self.assertEqual(split, adapter.split_train_by_sequence(self.root, seed=444))
        self.assertFalse(set(split["train_sequence_ids"]) & set(split["val_sequence_ids"]))
        self.assertEqual(set(split["train_ids"]) | set(split["val_ids"]), {"000000", "000001", "000002"})
        with self.assertRaisesRegex(ValueError, "explicit training-only"):
            self.dataset(split="val")
        with self.assertRaisesRegex(ValueError, "selected official partition"):
            self.dataset(split="val", clip_ids=["000237"])
        self.assertEqual(len(self.dataset(split="val", clip_ids=split["val_ids"], sample_mode="clip")), len(split["val_ids"]))

    def test_shared_frame_unions_recording_groups_and_single_group_is_rejected(self):
        p = self.clip("000000", "0000")
        q = self.clip("000001", "0001")
        first = json.loads(p.read_text())
        second = json.loads(q.read_text())
        second["frame0"][0] = first["frame0"][0]
        q.write_text(json.dumps(second))
        with self.assertRaisesRegex(ValueError, "At least two"):
            adapter.split_train_by_sequence(self.root)

    def test_missing_calibration_invalid_depth_and_explicit_center_assumption(self):
        self.clip("000000", "0000", center=(-20., -2., 1.))
        with self.assertRaisesRegex(ValueError, "Explicit experimental"):
            adapter.OfficialVLSOTDataset(self.root, self.calib)
        with self.assertRaisesRegex(ValueError, "Z must be positive"):
            self.dataset()[0]
        (self.calib / "camera_intrinsic/000000_0.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing matching-frame"):
            self.dataset()[0]

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable in local CPU runtime")
    def test_pytorch_default_collate_frame_and_clip_contract(self):
        from torch.utils.data import DataLoader
        self.clip("000000", "0000")
        frames = self.dataset(frame_indices=(0, 1))
        batch = next(iter(DataLoader(frames, batch_size=2)))
        self.assertEqual(tuple(batch["images"].shape), (2, 3, 32, 50))
        self.assertEqual(tuple(batch["calibs"].shape), (2, 3, 4))
        self.assertEqual(tuple(batch["target"]["projected_center"].shape), (2, 1, 2))
        self.assertEqual(tuple(batch["target"]["size_3d"].shape), (2, 1, 3))
        clips = self.dataset(sample_mode="clip", frame_indices=(0, 1))
        sequence = next(iter(DataLoader(clips, batch_size=1)))
        self.assertEqual(len(sequence), 2)
        self.assertEqual(tuple(sequence[0]["images"].shape), (1, 3, 32, 50))


if __name__ == "__main__":
    unittest.main()
