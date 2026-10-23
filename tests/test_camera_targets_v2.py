"""CPU contracts for the independent new camera geometric-center encoding."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools" / "camera_targets_v2.py"
SPEC = importlib.util.spec_from_file_location("camera_targets_v2", str(PATH))
geometry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(geometry)


class CameraTargetsV2Tests(unittest.TestCase):
    def setUp(self):
        self.native = np.array([2., -1., 20.])
        self.dims = np.array([4., 2., 1.5])
        self.rotation = np.eye(3)
        self.translation = np.zeros(3)
        self.K = np.array([[1000., 0., 640.], [0., 900., 360.], [0., 0., 1.]])
        self.affine = np.array([[1., 0., 0.], [0., 1., 0.]])
        self.size = np.array([1280., 720.])

    def encode(self):
        return geometry.build_camera_targets(self.native, self.dims, self.rotation,
                                              self.translation, self.K, self.affine, self.size)

    def decode(self, targets):
        return geometry.decode_camera_center(targets["normalized_center_xy"], targets["camera_z"],
                                              self.rotation, self.translation, self.K,
                                              self.affine, self.size)

    def test_known_camera_projection_and_no_implicit_bottom_center_shift(self):
        targets = self.encode()
        np.testing.assert_allclose(targets["camera_geometric_center_xyz"], [2., -1., 20.])
        np.testing.assert_allclose(targets["source_pixel_xy"], [740., 315.])
        np.testing.assert_allclose(targets["normalized_center_xy"], [740. / 1280., 315. / 720.])
        np.testing.assert_allclose(self.decode(targets), self.native)
        self.assertFalse(targets["old_checkpoint_compatible"])
        self.assertEqual(targets["encoding"], "camera_geometric_center_v2")
        self.assertTrue(targets["center_in_input_bounds"])

    def test_dimensions_never_shift_geometric_center(self):
        first = self.encode()
        self.dims = np.array([12., 3., 6.])
        second = self.encode()
        np.testing.assert_array_equal(first["camera_geometric_center_xyz"], second["camera_geometric_center_xyz"])
        np.testing.assert_array_equal(first["normalized_center_xy"], second["normalized_center_xy"])

    def test_nonorthogonal_dair_style_calibration_requires_solve_not_transpose(self):
        # Synthetic point chosen for nonorthogonal-transform roundtrip; not a dataset label.
        self.native = np.array([40., 7., -.5])
        self.rotation = np.array([[.001, -1.0002, .02],
                                  [-.01, .025, -.9998], [1.0001, .001, .004]])
        self.translation = np.array([[.5], [-.2], [3.]])
        targets = self.encode()
        np.testing.assert_allclose(self.decode(targets), self.native, atol=1e-12, rtol=0)
        wrong_inverse = self.rotation.T @ (targets["camera_geometric_center_xyz"] - self.translation[:, 0])
        self.assertGreater(np.linalg.norm(wrong_inverse - self.native), .001)

    def test_arbitrary_invertible_image_affine_and_skewed_intrinsics_roundtrip(self):
        self.affine = np.array([[.7, .13, -20.], [-.04, .9, 15.]])
        self.K[0, 1] = 37.
        targets = self.encode()
        np.testing.assert_allclose(self.decode(targets), self.native, atol=1e-12, rtol=0)
        homogeneous_input = np.r_[targets["normalized_center_xy"] * self.size, 1.]
        np.testing.assert_allclose(self.affine @ homogeneous_input, targets["source_pixel_xy"])

    def test_out_of_bounds_projection_is_flagged_and_never_clamped(self):
        self.native = np.array([100., -3., 1.])
        targets = self.encode()
        self.assertFalse(targets["center_in_input_bounds"])
        self.assertGreater(targets["normalized_center_xy"][0], 1.)
        self.assertLess(targets["normalized_center_xy"][1], 0.)
        np.testing.assert_allclose(self.decode(targets), self.native, atol=1e-12, rtol=0)

    def test_camera_depth_above_old_bins_is_preserved(self):
        self.native[2] = 91.0
        targets = self.encode()
        self.assertEqual(targets["camera_z"], 91.0)
        np.testing.assert_allclose(self.decode(targets), self.native)

    def test_invalid_depth_dimensions_and_singular_calibration_are_rejected(self):
        self.native[2] = 0.
        with self.assertRaisesRegex(ValueError, "Z must be positive"):
            self.encode()
        self.native[2] = 20.
        self.dims[0] = -4.
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self.encode()
        self.dims[0] = 4.
        self.rotation[2] = self.rotation[1]
        with self.assertRaisesRegex(ValueError, "invertible"):
            self.encode()

    def test_singular_affine_nonfinite_and_non_intrinsic_K_are_rejected(self):
        self.affine[1] = self.affine[0]
        with self.assertRaisesRegex(ValueError, "invertible"):
            self.encode()
        self.affine = np.array([[1., 0., 0.], [0., 1., 0.]])
        self.K[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.encode()
        self.K[0, 0] = 1000.
        self.K[2, 0] = .1
        with self.assertRaisesRegex(ValueError, "intrinsic matrix"):
            self.encode()
        self.K = np.column_stack((self.K, np.zeros(3)))
        with self.assertRaisesRegex(ValueError, "shape"):
            self.encode()

    def test_K_only_projection_has_no_extrinsic_or_translation_baked_in(self):
        self.translation = np.array([1., 2., 3.])
        targets = self.encode()
        np.testing.assert_array_equal(targets["projection_K_only"][:, :3], self.K)
        np.testing.assert_array_equal(targets["projection_K_only"][:, 3], np.zeros(3))
        np.testing.assert_array_equal(geometry.projection_matrix_from_intrinsics(self.K),
                                      targets["projection_K_only"])

    def test_world_conversion_requires_explicit_source_and_target(self):
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        transformed = geometry.transform_center(
            self.native, rotation, [100., -20., 3.],
            source_coordinate_system="infrastructure_virtual_lidar",
            target_coordinate_system="virtual_world")
        np.testing.assert_allclose(transformed["point_xyz"], [101., -18., 23.])
        self.assertEqual(transformed["source_coordinate_system"], "infrastructure_virtual_lidar")
        with self.assertRaises(TypeError):
            geometry.transform_center(self.native, rotation, [100., -20., 3.])
        with self.assertRaisesRegex(ValueError, "explicitly specified"):
            geometry.transform_center(self.native, rotation, [100., -20., 3.],
                                      source_coordinate_system="", target_coordinate_system="virtual_world")


if __name__ == "__main__":
    unittest.main()
