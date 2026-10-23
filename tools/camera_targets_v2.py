"""NEW training encoding: calibrated camera GEOMETRIC centers, no h/2 offset.

This independent CPU geometry module does not modify the inherited model,
decoder, training targets, or submission archive. Algebraic correctness does
NOT establish compatibility with any existing checkpoint. Source calibration
provenance and the corresponding training losses must be checked separately.
"""

import argparse
import json
from pathlib import Path

import numpy as np


ENCODING_NAME = "camera_geometric_center_v2"
OFFICIAL_TEST_IDS = {"000237", "001483", "003235", "006543", "024386",
                     "033447", "035729", "036120", "047771", "051657"}


def _finite(value, name):
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError("{} must contain only finite values".format(name))
    return array


def _vector(value, length, name):
    array = _finite(value, name)
    if array.shape == (length, 1):
        array = array[:, 0]
    if array.shape != (length,):
        raise ValueError("{} must be a {}-vector".format(name, length))
    return array


def _invertible_matrix(value, shape, name):
    matrix = _finite(value, name)
    if matrix.shape != shape:
        raise ValueError("{} must have shape {}".format(name, shape))
    try:
        inverse = np.linalg.solve(matrix, np.eye(shape[0]))
    except np.linalg.LinAlgError as error:
        raise ValueError("{} must be invertible".format(name)) from error
    if not np.isfinite(inverse).all():
        raise ValueError("{} has a non-finite inverse".format(name))
    return matrix


def _camera_calibration(rotation, translation, K):
    # DAIR calibration rotations are not assumed exactly orthonormal. Inverse
    # transforms below use solve(R, point-t), NEVER R.T @ (point-t).
    rotation = _invertible_matrix(rotation, (3, 3), "lidar_to_camera_R")
    translation = _vector(translation, 3, "lidar_to_camera_t")
    intrinsics = _invertible_matrix(K, (3, 3), "K")
    if not np.allclose(intrinsics[2], [0., 0., 1.], atol=1e-10, rtol=0):
        raise ValueError("K must be an intrinsic matrix with last row [0,0,1]")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("K focal lengths must be positive")
    return rotation, translation, intrinsics


def _image_geometry(source_from_input, input_size_wh):
    inverse_affine = _finite(source_from_input, "source_from_input")
    if inverse_affine.shape != (2, 3):
        raise ValueError("source_from_input must have shape (2,3)")
    homogeneous = np.vstack((inverse_affine, [0., 0., 1.]))
    _invertible_matrix(homogeneous, (3, 3), "image affine")
    size = _vector(input_size_wh, 2, "input_size_wh")
    if np.any(size <= 0):
        raise ValueError("input image width and height must be positive")
    return inverse_affine, homogeneous, size


def projection_matrix_from_intrinsics(K):
    """Return ONLY [K|0], for points already transformed into camera space."""
    _, _, intrinsics = _camera_calibration(np.eye(3), np.zeros(3), K)
    return np.column_stack((intrinsics, np.zeros(3)))


def build_camera_targets(native_center_xyz, dimensions_lwh, lidar_to_camera_R,
                         lidar_to_camera_t, K, source_from_input, input_size_wh):
    """Encode one native geometric-center object using supplied calibration.

    pc = R @ pn + t; uv_source = project(K @ pc); uv_input = inverse of
    source_from_input applied to uv_source; target_xy = uv_input / input_WH.
    All centers are GEOMETRIC centers. Dimensions do not shift the position.
    Out-of-image centers are returned unchanged and flagged, never clamped.
    """
    native = _vector(native_center_xyz, 3, "native_center_xyz")
    dimensions = _vector(dimensions_lwh, 3, "dimensions_lwh")
    if np.any(dimensions <= 0):
        raise ValueError("Physical l,w,h must be positive")
    rotation, translation, intrinsics = _camera_calibration(
        lidar_to_camera_R, lidar_to_camera_t, K)
    _, affine_homogeneous, input_size = _image_geometry(source_from_input, input_size_wh)
    camera = _finite(rotation @ native + translation, "camera_geometric_center_xyz")
    if camera[2] <= 0:
        raise ValueError("Camera geometric-center Z must be positive")
    projected = intrinsics @ camera
    source_pixel = _finite(projected[:2] / projected[2], "source_pixel_xy")
    input_homogeneous = np.linalg.solve(affine_homogeneous, np.r_[source_pixel, 1.])
    input_pixel = _finite(input_homogeneous[:2], "input_pixel_xy")
    normalized = _finite(input_pixel / input_size, "normalized_center_xy")
    return {
        "encoding": ENCODING_NAME,
        "new_training_encoding": True,
        "old_checkpoint_compatible": False,
        "compatibility_note": "New target definition; old checkpoint compatibility is not established",
        "native_coordinate_system": "infrastructure_virtual_lidar",
        "center_reference": "geometric_center",
        "native_geometric_center_xyz": native,
        "camera_geometric_center_xyz": camera,
        "camera_z": float(camera[2]),
        "source_pixel_xy": source_pixel,
        "input_pixel_xy": input_pixel,
        "normalized_center_xy": normalized,
        "center_in_input_bounds": bool(np.all(normalized >= 0) and np.all(normalized <= 1)),
        "dimensions_lwh": dimensions,
        "projection_K_only": np.column_stack((intrinsics, np.zeros(3))),
        "camera_rotation_condition_number": float(np.linalg.cond(rotation)),
        "depth_policy": "Positive camera Z, no maximum-depth clipping; training bins must be configured separately",
    }


def decode_camera_center(normalized_center_xy, camera_z, lidar_to_camera_R,
                         lidar_to_camera_t, K, source_from_input, input_size_wh):
    """Reverse the NEW encoding to native GEOMETRIC XYZ, with no h/2 shift.

    Camera Z is metric optical-axis depth, not Euclidean range or native height.
    Centers outside [0,1] remain mathematically decodable; no clipping is used.
    """
    normalized = _vector(normalized_center_xy, 2, "normalized_center_xy")
    depth_array = _finite(camera_z, "camera_z")
    if depth_array.shape != () or isinstance(camera_z, (bool, np.bool_)):
        raise ValueError("camera_z must be a positive scalar")
    depth = float(depth_array)
    if depth <= 0:
        raise ValueError("camera_z must be positive")
    rotation, translation, intrinsics = _camera_calibration(
        lidar_to_camera_R, lidar_to_camera_t, K)
    inverse_affine, _, input_size = _image_geometry(source_from_input, input_size_wh)
    source_pixel = inverse_affine @ np.r_[normalized * input_size, 1.]
    ray = np.linalg.solve(intrinsics, np.r_[source_pixel, 1.])
    if not np.isfinite(ray).all() or abs(ray[2]) < 1e-12:
        raise ValueError("Backprojected camera ray is invalid")
    camera = ray * (depth / ray[2])
    native = np.linalg.solve(rotation, camera - translation)
    return _finite(native, "decoded_native_geometric_center_xyz")


def transform_center(point_xyz, rotation, translation, *,
                     source_coordinate_system, target_coordinate_system):
    """Apply an explicitly named source-to-target R@p+t to a geometric center.

    For native->world, callers must supply the corresponding official
    virtuallidar_to_world file's R/t and explicitly name the native source.
    This function never infers a world frame from the numeric point values.
    """
    for name, value in (("source_coordinate_system", source_coordinate_system),
                        ("target_coordinate_system", target_coordinate_system)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("{} must be explicitly specified".format(name))
    point = _vector(point_xyz, 3, "point_xyz")
    matrix = _invertible_matrix(rotation, (3, 3), "source_to_target_R")
    offset = _vector(translation, 3, "source_to_target_t")
    return {
        "point_xyz": _finite(matrix @ point + offset, "transformed_point_xyz"),
        "source_coordinate_system": source_coordinate_system,
        "target_coordinate_system": target_coordinate_system,
        "center_reference": "geometric_center",
        "equation": "target = rotation @ source + translation",
    }


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, help="Explicit training-only geometry JSON")
    parser.add_argument("--output", help="Optional NEW JSON path; otherwise print results")
    args = parser.parse_args(argv)
    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    if fixture.get("split") not in ("train", "training"):
        raise ValueError("CLI requires an explicitly identified training-only fixture")
    if str(fixture.get("video_id", "")) in OFFICIAL_TEST_IDS:
        raise ValueError("Official-test ground-truth fixtures are not permitted here")
    names = ("native_center_xyz", "dimensions_lwh", "lidar_to_camera_R",
             "lidar_to_camera_t", "K", "source_from_input", "input_size_wh")
    encoded = build_camera_targets(*(fixture[name] for name in names))
    decoded = decode_camera_center(
        encoded["normalized_center_xy"], encoded["camera_z"],
        fixture["lidar_to_camera_R"], fixture["lidar_to_camera_t"], fixture["K"],
        fixture["source_from_input"], fixture["input_size_wh"])
    error = float(np.max(np.abs(decoded - np.asarray(fixture["native_center_xyz"]).reshape(3))))
    result = {"split": fixture["split"], "video_id": fixture.get("video_id"),
              "targets": encoded, "decoded_native_geometric_center_xyz": decoded,
              "roundtrip_max_abs_error_m": error,
              "old_checkpoint_compatible": False,
              "roundtrip_is_geometry_check_not_model_accuracy": True}
    if "world_transform" in fixture:
        world = fixture["world_transform"]
        result["world_geometric_center"] = transform_center(
            decoded, world["rotation"], world["translation"],
            source_coordinate_system=world["source_coordinate_system"],
            target_coordinate_system=world["target_coordinate_system"])
    text = json.dumps(_jsonable(result), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        target = Path(args.output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
