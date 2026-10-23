"""Geometry contracts tested independently of torch/CUDA/model execution."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "tools" / "geometry_decode.py"
SPEC = importlib.util.spec_from_file_location("geometry_decode", str(MODULE))
geometry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(geometry)


class GeometryDecodeTests(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[1000., 0., 640., -100.],
                           [0., 900., 360., 45.], [0., 0., 1., 0.]])
        self.center = np.array([2., -1., 20.])
        point_homogeneous = self.p @ np.append(self.center, 1.)
        self.pixel = point_homogeneous[:2] / point_homogeneous[2]
        self.outputs = {
            "pred_logits": np.array([[[0., 3.]]]),
            "pred_boxes": np.array([[np.r_[self.pixel / [1280., 720.],
                                             [.1, .1, .2, .2]]]]),
            "pred_3d_dim": np.array([[[.2, -.1, .5]]]),
            "pred_depth": np.array([[[20., 0.]]]),
        }
        self.means = np.array([[1., 1., 1.], [1.8, 2.1, 3.5]])

    def decode(self, **kwargs):
        return geometry.decode_native_predictions(
            self.outputs, [1280, 720], self.p, self.means,
            center_encoding="projected_3d_center", **kwargs)

    def test_metric_geometry_class_means_and_center_reference(self):
        original = {key: value.copy() for key, value in self.outputs.items()}
        decoded = self.decode()
        np.testing.assert_allclose(decoded["dimensions_hwl"], [[2., 2., 4.]])
        np.testing.assert_allclose(decoded["dimensions_lwh"], [[4., 2., 2.]])
        np.testing.assert_allclose(decoded["shifted_position_native_xyz"], [self.center])
        np.testing.assert_allclose(decoded["label_position_native_xyz"], [[2., 0., 20.]])
        self.assertEqual(decoded["coordinate_system"], "rectified_camera_assumed")
        self.assertFalse(decoded["world_coordinates_verified"])
        for key, value in original.items():
            np.testing.assert_array_equal(self.outputs[key], value)

    def test_non_matching_aspect_ratio_inverse_affine(self):
        # dataset uses width-derived scale, centering source height, not stretch.
        inverse = geometry.dataset_resize_inverse_affine([1280, 720], [1920, 1200])
        input_point = np.array([self.pixel[0] * 1.5,
                                self.pixel[1] * 1.5 + 60.])
        self.outputs["pred_boxes"][0, 0, :2] = input_point / [1920, 1200]
        decoded = self.decode(input_sizes_wh=[1920, 1200], source_from_input=inverse)
        np.testing.assert_allclose(decoded["shifted_position_native_xyz"], [self.center])
        with self.assertRaisesRegex(ValueError, "inverse affine"):
            self.decode(input_sizes_wh=[1920, 1200])

    def test_query_and_class_chosen_without_ground_truth(self):
        for key in self.outputs:
            self.outputs[key] = np.repeat(self.outputs[key], 2, axis=1)
        self.outputs["pred_logits"][0, 1] = [7., -5.]
        self.outputs["pred_3d_dim"][0, 1] = [1., 1., 3.]
        decoded = self.decode()
        self.assertEqual(decoded["query_index"].item(), 1)
        self.assertEqual(decoded["class_id"].item(), 0)
        np.testing.assert_allclose(decoded["dimensions_hwl"], [[2., 2., 4.]])

    def test_negative_dimensions_not_hidden_by_abs_or_clamping(self):
        self.outputs["pred_3d_dim"][0, 0, 0] = -5.
        with self.assertRaisesRegex(ValueError, "positive"):
            self.decode()

    def test_depth_is_already_metric(self):
        decoded = self.decode()
        self.assertEqual(decoded["shifted_position_native_xyz"][0, 2], 20.)
        self.outputs["pred_depth"][0, 0, 0] = 0.
        with self.assertRaisesRegex(ValueError, "depth"):
            self.decode()

    def test_arbitrary_world_projection_is_rejected(self):
        self.p[2, 0] = .1
        with self.assertRaisesRegex(ValueError, "rectified-camera"):
            self.decode()

    def test_legacy_encoding_is_explicit_and_never_world_coordinates(self):
        self.outputs["pred_boxes"][0, 0, :2] = [2. / 1280., 3. / 720.]
        self.outputs["pred_depth"][0, 0, 0] = -.596588
        decoded = geometry.decode_native_predictions(
            self.outputs, [1280, 720], self.p, self.means,
            center_encoding="legacy_native_xy")
        np.testing.assert_allclose(decoded["shifted_position_native_xyz"], [[2., 3., -.596588]])
        np.testing.assert_allclose(decoded["label_position_native_xyz"], [[2., 4., -.596588]])
        self.assertEqual(decoded["coordinate_system"], "unknown_native_label")
        self.assertNotIn("center_camera_xyz", decoded)
        self.assertFalse(decoded["world_coordinates_verified"])
        with self.assertRaisesRegex(ValueError, "encoding"):
            geometry.decode_native_predictions(
                self.outputs, [1280, 720], self.p, self.means,
                center_encoding="unknown")

    def test_legacy_native_encoding_does_not_require_camera_intrinsics(self):
        decoded = geometry.decode_native_predictions(
            self.outputs, [1280, 720], None, self.means,
            center_encoding="legacy_native_xy")
        self.assertEqual(decoded["coordinate_system"], "unknown_native_label")

    def test_nonfinite_values_fail_before_export(self):
        self.outputs["pred_boxes"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.decode()


if __name__ == "__main__":
    unittest.main()
