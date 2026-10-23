"""CPU-only offline metric and explicit validation-selection contracts."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("evaluate_official_v2", str(TOOLS / "evaluate_official_v2.py"))
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class OfficialV2EvaluationTests(unittest.TestCase):
    def test_exact_native_camera_and_lwh_metrics_frame_weighted(self):
        target = {"native_center_xyz": np.array([[10., 20., -.5]]),
                  "camera_center_xyz": np.array([[1., 2., 50.]]),
                  "dimensions_lwh": np.array([[4., 2., 1.5]])}
        predicted = {"label_position_native_xyz": np.array([13., 24., -.5]),
                     "camera_center_xyz": np.array([1., 2., 62.]),
                     "dimensions_lwh": np.array([4.3, 2.6, 1.8])}
        error = evaluation.frame_errors(predicted, target)
        self.assertEqual(error["native_xyz_error_m"], 5.)
        self.assertEqual(error["camera_xyz_error_m"], 12.)
        summary = evaluation.summarize_errors([error, error])
        self.assertEqual(summary["frame_count"], 2)
        self.assertEqual(summary["native_xyz_mean_error_m"], 5.)
        np.testing.assert_allclose(summary["native_xyz_mae_per_axis_m"], [3., 4., 0.])
        np.testing.assert_allclose(summary["dimensions_mae_lwh_m"], [.3, .6, .3])
        self.assertAlmostEqual(summary["dimensions_mean_absolute_error_m"], .4)

    def test_known_training_overlap_and_other_validation_selection_are_rejected(self):
        manifest = {"train_clip_ids": ["000000"], "val_clip_ids": ["000001", "000002"]}
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluation.check_recorded_partition(["000000"], [manifest])
        with self.assertRaisesRegex(ValueError, "differ"):
            evaluation.check_recorded_partition(["000003"], [manifest])
        result = evaluation.check_recorded_partition(["000001"], [manifest])
        self.assertEqual(result["training_id_overlap_check"], "passed")
        self.assertEqual(evaluation.check_recorded_partition(["000001"], [])["training_id_overlap_check"], "not_recorded")

    def test_val_ids_require_explicit_nonempty_file_preserving_zeroes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "val.txt"
            path.write_text("000001\n000002\n", encoding="utf-8")
            self.assertEqual(evaluation.read_val_ids(path), ["000001", "000002"])
            path.write_text("000001\n000001\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                evaluation.read_val_ids(path)
            path.write_text("1\n")
            with self.assertRaisesRegex(ValueError, "six-digit"):
                evaluation.read_val_ids(path)

    def test_metrics_reject_missing_labels_nonfinite_values_and_empty_evaluation(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluation.summarize_errors([])
        target = {"native_center_xyz": np.zeros((1, 3)), "camera_center_xyz": np.zeros((1, 3)),
                  "dimensions_lwh": np.ones((1, 3))}
        predicted = {"label_position_native_xyz": np.array([np.nan, 0., 0.]),
                     "camera_center_xyz": np.zeros(3), "dimensions_lwh": np.ones(3)}
        with self.assertRaisesRegex(ValueError, "Invalid evaluation"):
            evaluation.frame_errors(predicted, target)


if __name__ == "__main__":
    unittest.main()
