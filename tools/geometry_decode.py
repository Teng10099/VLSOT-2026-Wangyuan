"""CPU decoding of the checked-in Mono3DVG heads into native label values.

This module deliberately does not export world-coordinate competition results.
``pred_boxes`` is normalized (cx, cy, left, right, top, bottom), not XYZXYZ.
``pred_3d_dim`` follows the dataset's (h, w, l) minus class mean target.
``pred_depth[..., 0]`` is already fused in the model. The training target is
label position's third component; it is camera-Z only if the label is camera
coordinates. Sample source labels do not establish that assumption.

The current dataset has its center projection commented out. Therefore callers
must explicitly choose a center encoding; the checkpoint's training version is
needed to decide which interpretation is correct. Neither choice validates a
checkpoint or supplies a missing camera-to-world transformation.
"""

import numpy as np


def _array(value, name):
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("{} contains non-finite values".format(name))
    return result


def _broadcast(value, batch, shape, name):
    value = _array(value, name)
    if value.shape == tuple(shape):
        return np.broadcast_to(value, (batch,) + tuple(shape))
    if value.shape != (batch,) + tuple(shape):
        raise ValueError("{} must have shape {} or {}".format(
            name, tuple(shape), (batch,) + tuple(shape)))
    return value


def dataset_resize_inverse_affine(source_size_wh, input_size_wh):
    """Inverse of dataset's zero-rotation, centered, width-based affine.

    This matches get_affine_transform(center=source_size/2,
    scale=source_size, rot=0, output_size=input_size, shift=0). It is NOT
    necessarily an independent horizontal/vertical stretch: image heights with
    a different aspect ratio are centered and cropped/padded.
    """
    source = _array(source_size_wh, "source_size_wh")
    target = _array(input_size_wh, "input_size_wh")
    if source.shape != (2,) or target.shape != (2,):
        raise ValueError("image sizes must each be (width, height)")
    if np.any(source <= 0) or np.any(target <= 0):
        raise ValueError("image sizes must be positive")
    scale = source[0] / target[0]
    return np.array([[scale, 0.0, source[0] / 2 - scale * target[0] / 2],
                     [0.0, scale, source[1] / 2 - scale * target[1] / 2]])


def backproject_rectified_camera(image_xy, camera_z, projection_matrices):
    """Recover rectified-camera XYZ from pixels and camera Z, in native units.

    Only a rectified camera projection with last row [0, 0, 1, 0] is accepted.
    An arbitrary world-to-image matrix does not establish camera coordinates.
    The 2x2 solve also preserves the projection's skew and baseline offsets.
    """
    xy = _array(image_xy, "image_xy")
    depth = _array(camera_z, "camera_z")
    if xy.ndim != 2 or xy.shape[1] != 2 or depth.shape != (xy.shape[0],):
        raise ValueError("image_xy must be (B,2), camera_z must be (B,)")
    if np.any(depth <= 0):
        raise ValueError("camera depth must be positive; do not clamp it")
    matrices = _broadcast(projection_matrices, len(xy), (3, 4), "projection_matrices")
    if not np.allclose(matrices[:, 2, :], [0., 0., 1., 0.], atol=1e-7, rtol=0):
        raise ValueError("projection is not an established rectified-camera P2")
    result = np.empty((len(xy), 3), dtype=np.float64)
    for i, (point, z, projection) in enumerate(zip(xy, depth, matrices)):
        if abs(np.linalg.det(projection[:2, :2])) < 1e-12:
            raise ValueError("projection has singular image intrinsics")
        rhs = point * z - projection[:2, 2] * z - projection[:2, 3]
        result[i, :2] = np.linalg.solve(projection[:2, :2], rhs)
        result[i, 2] = z
    return result


def decode_native_predictions(outputs, image_sizes_wh, projection_matrices,
                              class_mean_size_hwl, *, center_encoding,
                              input_sizes_wh=None, source_from_input=None):
    """Select one top scoring query/class per batch and decode native boxes.

    ``outputs`` contains NumPy arrays from pred_logits (B,Q,C), pred_boxes
    (B,Q,6), pred_3d_dim (B,Q,3), and pred_depth (B,Q,2).

    center_encoding='projected_3d_center' implements the existing decoder's
    intended camera backprojection, assuming camera-coordinate source labels.
    'legacy_native_xy' reverses the current dataset's unprojected native-X/Y
    target. It neither requires nor uses P2, and the third native component may
    be negative. Both modes are diagnostic until the checkpoint training source
    and label coordinate definitions are established.

    The returned label_position_native_xyz adds h/2 to the SECOND component,
    reversing the dataset's explicit subtraction. For unknown source axes this
    is an algebraic label reference, not a claim about a physical bottom center.

    Supply the actual preprocessing input size AND its 2x3 inverse affine if
    images were resized/cropped. With neither, predictions are assumed to be
    normalized directly to the source image dimensions. No ground-truth boxes,
    classes, or positions are read. Inputs are never modified.
    """
    if center_encoding not in ("projected_3d_center", "legacy_native_xy"):
        raise ValueError("center_encoding must be explicitly established")
    logits = _array(outputs["pred_logits"], "pred_logits")
    boxes = _array(outputs["pred_boxes"], "pred_boxes")
    dimensions = _array(outputs["pred_3d_dim"], "pred_3d_dim")
    depths = _array(outputs["pred_depth"], "pred_depth")
    if logits.ndim != 3 or min(logits.shape) < 1:
        raise ValueError("pred_logits must be nonempty (B,Q,C)")
    batch, queries, classes = logits.shape
    for value, width, name in ((boxes, 6, "pred_boxes"),
                               (dimensions, 3, "pred_3d_dim"),
                               (depths, 2, "pred_depth")):
        if value.shape != (batch, queries, width):
            raise ValueError("{} has an unexpected shape".format(name))
    sizes = _broadcast(image_sizes_wh, batch, (2,), "image_sizes_wh")
    input_sizes = sizes if input_sizes_wh is None else _broadcast(
        input_sizes_wh, batch, (2,), "input_sizes_wh")
    if np.any(sizes <= 0) or np.any(input_sizes <= 0):
        raise ValueError("image sizes must be positive")
    if source_from_input is None:
        if not np.allclose(sizes, input_sizes):
            raise ValueError("resized inputs require the actual inverse affine")
        transforms = np.broadcast_to(np.array([[1., 0., 0.], [0., 1., 0.]]),
                                     (batch, 2, 3))
    else:
        transforms = _broadcast(source_from_input, batch, (2, 3), "source_from_input")

    # This is the same ranking as sigmoid(logits).flatten().topk(1).
    flat_index = np.argmax(logits.reshape(batch, -1), axis=1)
    query_index, class_id = flat_index // classes, flat_index % classes
    rows = np.arange(batch)
    selected_boxes = boxes[rows, query_index]
    means = _array(class_mean_size_hwl, "class_mean_size_hwl")
    if means.ndim != 2 or means.shape[1] != 3 or means.shape[0] < classes:
        raise ValueError("class means must cover the model classes in h,w,l order")
    hwl = dimensions[rows, query_index] + means[class_id]
    if np.any(hwl <= 0):
        raise ValueError("decoded h,w,l must be positive; inspect weights/mean shape")
    depth = depths[rows, query_index, 0]
    input_xy = selected_boxes[:, :2] * input_sizes
    source_xy = np.einsum("bij,bj->bi", transforms,
                         np.column_stack((input_xy, np.ones(batch))))
    if center_encoding == "projected_3d_center":
        shifted_xyz = backproject_rectified_camera(source_xy, depth, projection_matrices)
        coordinate_system = "rectified_camera_assumed"
    else:
        shifted_xyz = np.column_stack((source_xy, depth))
        coordinate_system = "unknown_native_label"
    label_position = shifted_xyz.copy()
    label_position[:, 1] += hwl[:, 0] / 2.0
    selected_logits = logits[rows, query_index, class_id]
    class_probability = np.exp(-np.logaddexp(0.0, -selected_logits))
    with np.errstate(over="raise", invalid="raise"):
        try:
            depth_confidence = np.exp(-depths[rows, query_index, 1])
        except FloatingPointError as error:
            raise ValueError("invalid depth uncertainty") from error
    return {
        "dimensions_hwl": hwl,
        "dimensions_lwh": hwl[:, [2, 1, 0]],
        "shifted_position_native_xyz": shifted_xyz,
        "label_position_native_xyz": label_position,
        "center_source_xy": source_xy,
        "query_index": query_index,
        "class_id": class_id,
        "class_probability": class_probability,
        "score": class_probability * depth_confidence,
        "coordinate_system": coordinate_system,
        "center_encoding": center_encoding,
        "world_coordinates_verified": False,
    }
