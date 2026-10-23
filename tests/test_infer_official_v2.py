"""CPU-only v2 decoding/contract/packaging checks; no model or test GT."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("infer_official_v2", str(TOOLS / "infer_official_v2.py"))
infer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(infer)


class OfficialV2InferenceTests(unittest.TestCase):
    def geometry(self):
        R = np.array([[.001, -1.0002, .02], [-.01, .025, -.9998], [1.0001, .001, .004]])
        t = np.array([.5, -.2, 3.])
        native = np.array([25., 6., -.5])
        camera = R @ native + t
        K = np.array([[1000., 21., 480.], [0., 950., 272.], [0., 0., 1.]])
        size = np.array([960., 544.])
        projected = K @ camera
        xy = (projected[:2] / projected[2]) / size
        predictions = {"target_encoding": infer.ENCODING,
                       "pred_boxes": np.array([[np.r_[xy, [0., 0., 0., 0.]]]]),
                       "pred_depth": np.array([[[camera[2], 0.]]]),
                       "pred_3d_dim": np.array([[[1.5, 2., 4.]]])}
        frame = {"image_sizes": size, "calibs": np.column_stack((K, np.zeros(3))),
                 "lidar_to_camera_R": R, "lidar_to_camera_t": t,
                 # Contradictory target-like content must never be consulted.
                 "target": {"native_center_xyz": [999., 999., 999.]},
                 "virtuallidar_to_world": {"translation": [999., 999., 999.]}}
        return predictions, frame, native, camera

    def test_exact_v2_backprojection_nonorthogonal_inverse_and_hwl_to_lwh(self):
        prediction, frame, native, camera = self.geometry()
        result = infer.decode_prediction(prediction, frame)
        np.testing.assert_allclose(result["values_lwhxyz"], np.r_[[4., 2., 1.5], native], atol=1e-12)
        np.testing.assert_allclose(result["camera_center_xyz"], camera)
        prediction["pred_3d_dim"][0, 0, 0] = 3.
        np.testing.assert_allclose(infer.decode_prediction(prediction, frame)["label_position_native_xyz"], native, atol=1e-12)
        wrong = frame["lidar_to_camera_R"].T @ (camera - frame["lidar_to_camera_t"])
        self.assertGreater(np.linalg.norm(wrong - native), .001)

    def test_outside_center_preserved_and_invalid_predictions_rejected(self):
        prediction, frame, _, _ = self.geometry()
        prediction["pred_boxes"][0, 0, :2] = [1.1, -.1]
        result = infer.decode_prediction(prediction, frame)
        np.testing.assert_allclose(result["normalized_center_xy"], [1.1, -.1])
        prediction["pred_depth"][0, 0, 0] = -1.
        with self.assertRaisesRegex(ValueError, "must be positive"):
            infer.decode_prediction(prediction, frame)
        prediction["pred_depth"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "shape/values"):
            infer.decode_prediction(prediction, frame)

    def test_contract_rejects_legacy_and_resolution_mismatch(self):
        cfg = {"model": {"target_encoding": infer.ENCODING, "input_size_hw": [544, 960]}}
        ckpt = {"target_encoding": infer.ENCODING}
        contract = infer.validate_contract(cfg, ckpt)
        self.assertEqual(contract["input_size_wh"], (960, 544))
        self.assertFalse(contract["label_convention_verified"])
        with self.assertRaisesRegex(ValueError, "legacy"):
            infer.validate_contract(cfg, {})
        with self.assertRaisesRegex(ValueError, "Configuration"):
            infer.validate_contract({"model": {}}, ckpt)
        with self.assertRaisesRegex(ValueError, "resolutions disagree"):
            infer.validate_contract(cfg, ckpt, explicit_size_wh=(1920, 1080))
        manifest = {"target_encoding": infer.ENCODING,
                    "args": {"input_width": 960, "input_height": 544, "seed": 123,
                             "label_center_reference": "geometric_center"}}
        self.assertEqual(infer.validate_contract(cfg, ckpt, [manifest])["seed_per_video"], 123)

    def test_missing_training_size_requires_explicit_resolution(self):
        cfg = {"model": {"target_encoding": infer.ENCODING}}
        ckpt = {"target_encoding": infer.ENCODING}
        with self.assertRaisesRegex(ValueError, "input size is missing"):
            infer.validate_contract(cfg, ckpt)
        self.assertEqual(infer.validate_contract(cfg, ckpt, explicit_size_wh=(960, 544))["input_size_wh"], (960, 544))

    def test_padding_mask_behavior_must_match_checkpoint_despite_identical_state_shapes(self):
        cfg = {"model": {"target_encoding": infer.ENCODING, "input_size_hw": [544, 960]}}
        baseline = {"target_encoding": infer.ENCODING}
        self.assertFalse(infer.validate_contract(cfg, baseline)["model_options"]["official_correct_padding_mask"])
        cfg["model"]["official_correct_padding_mask"] = True
        with self.assertRaisesRegex(ValueError, "padding_mask behavior disagrees"):
            infer.validate_contract(cfg, baseline)
        fixed = {"target_encoding": infer.ENCODING,
                 "model_options": {"official_correct_padding_mask": True, "input_size_hw": [544, 960]}}
        self.assertTrue(infer.validate_contract(cfg, fixed)["model_options"]["official_correct_padding_mask"])
        fixed["model_options"]["input_size_hw"] = [1080, 1920]
        with self.assertRaisesRegex(ValueError, "resolutions disagree"):
            infer.validate_contract(cfg, fixed)

    def test_temporal_residual_requires_matching_checkpoint_behavior_and_preserves_old_false(self):
        cfg = {"model": {"target_encoding": infer.ENCODING, "input_size_hw": [544, 960]}}
        old = {"target_encoding": infer.ENCODING}
        baseline = infer.validate_contract(cfg, old)
        self.assertFalse(baseline["model_options"]["official_temporal_residual"])
        cfg["model"]["official_temporal_residual"] = True
        with self.assertRaisesRegex(ValueError, "official_temporal_residual behavior disagrees"):
            infer.validate_contract(cfg, old)
        residual = {"target_encoding": infer.ENCODING,
                    "model_options": {"official_temporal_residual": True}}
        self.assertTrue(infer.validate_contract(cfg, residual)["model_options"]["official_temporal_residual"])
        cfg["model"]["official_temporal_residual"] = False
        with self.assertRaisesRegex(ValueError, "official_temporal_residual behavior disagrees"):
            infer.validate_contract(cfg, residual)
        cfg["model"].update(official_temporal_residual=True, official_correct_padding_mask=True)
        residual["model_options"]["official_correct_padding_mask"] = True
        matched = infer.validate_contract(cfg, residual)["model_options"]
        self.assertTrue(matched["official_temporal_residual"])
        self.assertTrue(matched["official_correct_padding_mask"])

    def records(self):
        return {clip: [{"frame_index": i, "values_lwhxyz": [4., 2., 1.5, 25., 6., -.5]}
                       for i in range(1, 31)] for clip in infer.OFFICIAL_IDS}

    def test_diagnostic_mode_never_writes_result_zip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = infer.package_predictions(root, {"000237": self.records()["000237"][:1]}, 1)
            self.assertTrue(result["diagnostic_only"])
            self.assertIsNone(result["zip"])
            self.assertEqual(list(root.iterdir()), [])

    def test_full_ten_clip_zip_has_only_30_line_7_column_result_members(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = infer.package_predictions(root, self.records(), 30)
            with zipfile.ZipFile(result["zip"]) as archive:
                self.assertEqual(archive.namelist(), ["result/{}.txt".format(i) for i in infer.OFFICIAL_IDS])
                for member in archive.namelist():
                    lines = archive.read(member).decode().splitlines()
                    self.assertEqual(len(lines), 30)
                    self.assertTrue(all(len(line.split()) == 7 for line in lines))
                    self.assertEqual(lines[0].split(), ["1", "4", "2", "1.5", "25", "6", "-0.5"])
            before = Path(result["zip"]).read_bytes()
            with self.assertRaises(FileExistsError):
                infer.package_predictions(root, self.records(), 30)
            self.assertEqual(Path(result["zip"]).read_bytes(), before)

    def test_full_export_refuses_missing_clip_partial_or_wrong_frame_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = self.records()
            del records["000237"]
            with self.assertRaisesRegex(ValueError, "exactly the ten"):
                infer.package_predictions(root, records, 30)
            records = self.records()
            records["000237"].pop()
            with self.assertRaisesRegex(ValueError, "30 ordered"):
                infer.package_predictions(root, records, 30)
            records = self.records()
            records["000237"][0]["frame_index"] = 2
            with self.assertRaisesRegex(ValueError, "30 ordered"):
                infer.package_predictions(root, records, 30)
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
