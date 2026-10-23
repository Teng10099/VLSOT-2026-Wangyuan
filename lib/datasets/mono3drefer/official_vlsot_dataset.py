"""Independent adapter for the official VLSOT train/json,img test/json,img.

Official TRAIN frames are [image, L, W, H, X, Y, Z]; TEST frames are image
strings and never become targets.  L/W/H is supported by TRAIN000000's public
description (4.5/1.7/0.9m versus 4.547617/1.664070/0.949794).  The position
coordinate system and geometric-center interpretation remain an EXPERIMENTAL
contract: their equality to the original source annotation was NOT established.
Callers must explicitly pass label_center_reference='geometric_center'; no h/2
shift is applied, and every sample records label_convention_verified=False.

The adapter uses tools.camera_targets_v2.  No BERT, original loader, torch or
OpenCV import is needed. NumPy arrays are compatible with torch default_collate.
Frame mode returns one dictionary; clip mode returns a chronological list of
frame dictionaries, collated as a list of frame batches. Training-only internal
validation must use explicit clip_ids selected by recording, never official test.
"""

from collections import defaultdict
import json
from pathlib import Path
import random
import re
from typing import Optional

import numpy as np
from PIL import Image

from tools.camera_targets_v2 import ENCODING_NAME, build_camera_targets


CLASS_IDS = {"CAR": 0, "TRUCK": 1, "BUS": 2, "VAN": 3}
FRAME_COUNT = 30


def _read_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Expected exactly one clip object: {}".format(path))
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError("Expected one clip JSON object: {}".format(path))
    return value


def _basename(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Image name must be a nonempty string")
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    if name in ("", ".", ".."):
        raise ValueError("Invalid image basename")
    return name


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{6}", value):
        raise ValueError("Clip IDs must be literal six-digit strings")
    return value


def _sequence(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("sequence_id must be explicit recording metadata")
    text = str(value)
    if not re.fullmatch(r"[0-9]{1,4}", text):
        raise ValueError("sequence_id must be a numeric recording ID")
    return text.zfill(4)


def _read_ids(value, available):
    if value is None:
        return sorted(available)
    if isinstance(value, (str, Path)):
        value = [line.strip() for line in Path(value).read_text(encoding="utf-8-sig").splitlines()
                 if line.strip()]
    ids = [_id(item) for item in value]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("clip_ids must be nonempty and contain no duplicates")
    missing = sorted(set(ids) - set(available))
    if missing:
        raise ValueError("clip_ids are not in the selected official partition: {}".format(missing))
    return ids


def _clip_metadata(path, partition):
    raw = _read_object(path)
    clip_id = _id(path.stem)
    declared = raw.get("videoID")
    if (isinstance(declared, bool) or not isinstance(declared, (str, int))
            or not str(declared).isdigit() or str(declared).zfill(6) != clip_id):
        raise ValueError("videoID disagrees with JSON filename: {}".format(path))
    sequence = _sequence(raw.get("sequence_id"))
    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Missing public language input in {}".format(path))
    frames = []
    for index in range(FRAME_COUNT):
        frame = raw.get("frame{}".format(index))
        if partition == "train":
            if not isinstance(frame, list) or len(frame) != 7:
                raise ValueError("TRAIN frame must be [image,L,W,H,X,Y,Z] in {}".format(path))
            image_name = _basename(frame[0])
        else:
            # Refuse legacy annotation arrays instead of reading test target tails.
            if not isinstance(frame, str):
                raise ValueError("Official TEST frame must be an image string, with no targets")
            image_name = _basename(frame)
        frames.append(image_name)
    if len(set(frames)) != FRAME_COUNT:
        raise ValueError("Duplicate frame images in clip {}".format(clip_id))
    result = {"video_id": clip_id, "sequence_id": sequence,
              "text": description.lower(), "frames": frames, "json_path": path}
    if partition == "train":
        car_type = raw.get("type")
        if not isinstance(car_type, str) or car_type.upper() not in CLASS_IDS:
            raise ValueError("Unsupported TRAIN object class: {!r}".format(car_type))
        result["class_id"] = CLASS_IDS[car_type.upper()]
        # Training numeric values alone are accessed. No official TEST arrays
        # or labels are parsed by this branch.
        values = []
        for index in range(FRAME_COUNT):
            numbers = raw["frame{}".format(index)][1:]
            if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in numbers):
                raise ValueError("TRAIN LWHXYZ must contain six numeric values")
            numbers = np.asarray(numbers, dtype=np.float64)
            if not np.isfinite(numbers).all() or np.any(numbers[:3] <= 0):
                raise ValueError("TRAIN LWH must be positive and LWHXYZ finite")
            values.append(numbers)
        result["lwhxyz"] = values
    return result


def _read_calibration(root, image_name):
    filename = Path(image_name).with_suffix(".json").name
    intrinsic_path = root / "camera_intrinsic" / filename
    extrinsic_path = root / "virtuallidar_to_camera" / filename
    if not intrinsic_path.is_file() or not extrinsic_path.is_file():
        raise ValueError("Missing matching-frame camera calibration for {} under {}".format(image_name, root))
    intrinsic = _read_object(intrinsic_path)
    if "cam_K" in intrinsic:
        K = np.asarray(intrinsic["cam_K"], dtype=np.float64).reshape(3, 3)
    elif "P" in intrinsic:
        projection = np.asarray(intrinsic["P"], dtype=np.float64).reshape(3, 4)
        if not np.allclose(projection[:, 3], 0., atol=1e-10, rtol=0):
            raise ValueError("Intrinsic P must be [K|0]; extrinsics cannot be applied twice")
        K = projection[:, :3]
    else:
        raise ValueError("Camera intrinsic JSON needs cam_K or P=[K|0]")
    extrinsic = _read_object(extrinsic_path)
    if "rotation" not in extrinsic or "translation" not in extrinsic:
        raise ValueError("Virtual-LiDAR-to-camera JSON needs explicit rotation and translation")
    R = np.asarray(extrinsic["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(extrinsic["translation"], dtype=np.float64).reshape(3)
    if (not np.isfinite(K).all() or not np.isfinite(R).all() or not np.isfinite(t).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0., 0., 1.])):
        raise ValueError("Invalid matching-frame camera calibration")
    try:
        np.linalg.solve(K, np.eye(3))
        np.linalg.solve(R, np.eye(3))
    except np.linalg.LinAlgError as error:
        raise ValueError("Camera calibration matrices must be invertible") from error
    return K, R, t


def signed_lrtb_from_box(projected_center_xy, bbox_xyxy):
    """Encode an explicitly supplied box without clamping negative L/R/T/B.

    Both arguments must use the same coordinate units. The projected 3D center
    can lie outside a valid 2D box; signed offsets reconstruct the original box.
    Official six-value labels supply NO 2D box, so the dataset never invents one.
    """
    center = np.asarray(projected_center_xy, dtype=np.float64)
    box = np.asarray(bbox_xyxy, dtype=np.float64)
    if (center.shape != (2,) or box.shape != (4,) or not np.isfinite(center).all()
            or not np.isfinite(box).all() or np.any(box[2:] < box[:2])):
        raise ValueError("Expected finite center[2] and non-inverted bbox[4]")
    return np.array([center[0], center[1], center[0] - box[0], box[2] - center[0],
                     center[1] - box[1], box[3] - center[1]], dtype=np.float32)


class OfficialVLSOTDataset:
    """Map-style dataset, compatible with PyTorch DataLoader default_collate.

    ``sample_mode='frame'`` flattens selected clips and frame_indices;
    ``sample_mode='clip'`` returns list[frame_dict]. ``get_frame(clip_index,
    frame_index)`` streams an individual zero-based frame. No geometric/random
    augmentations are applied. Image resizing and calibration resizing agree.

    ``split='val'`` reads only root/train, requiring explicit clip_ids. Caller
    must keep sequence groups disjoint using split_train_by_sequence(). Test
    has no 'target' field. The old trainer/criterion cannot be used unchanged:
    bbox, angle and dense-depth-map supervision are unavailable.
    """

    def __init__(self, root, calibration_root, split="train", clip_ids=None,
                 input_size_wh=(960, 544), sample_mode="frame", frame_indices=None,
                 label_center_reference: Optional[str] = None):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        if split == "val" and clip_ids is None:
            raise ValueError("Internal val requires explicit training-only clip_ids")
        if split != "test" and label_center_reference != "geometric_center":
            raise ValueError("Explicit experimental label_center_reference='geometric_center' required")
        if sample_mode not in ("frame", "clip"):
            raise ValueError("sample_mode must be frame or clip")
        size = np.asarray(input_size_wh)
        if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0) or np.any(size != size.astype(int)):
            raise ValueError("input_size_wh must contain two positive integers")
        self.input_size = size.astype(np.int32)
        self.partition = "test" if split == "test" else "train"
        self.split, self.sample_mode = split, sample_mode
        self.root = Path(root).expanduser().resolve()
        self.image_root = self.root / self.partition / "img"
        self.calibration_root = Path(calibration_root).expanduser().resolve()
        if not (self.calibration_root / "camera_intrinsic").is_dir() and (self.calibration_root / "calib").is_dir():
            self.calibration_root = self.calibration_root / "calib"
        self.frame_indices = tuple(range(FRAME_COUNT) if frame_indices is None else frame_indices)
        if (not self.frame_indices or len(set(self.frame_indices)) != len(self.frame_indices)
                or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < FRAME_COUNT for i in self.frame_indices)
                or tuple(sorted(self.frame_indices)) != self.frame_indices):
            raise ValueError("frame_indices must be unique chronological zero-based integers in [0,29]")
        json_root = self.root / self.partition / "json"
        available = {path.stem: path for path in sorted(json_root.glob("*.json")) if path.is_file()}
        if not available or not self.image_root.is_dir():
            raise ValueError("Missing official {}/json,img directories".format(self.partition))
        ids = _read_ids(clip_ids, available)
        self.clips = [_clip_metadata(available[clip_id], self.partition) for clip_id in ids]
        self.mean = np.asarray([.485, .456, .406], dtype=np.float32)[:, None, None]
        self.std = np.asarray([.229, .224, .225], dtype=np.float32)[:, None, None]
        self._calibration_cache = {}

    def __len__(self):
        return len(self.clips) if self.sample_mode == "clip" else len(self.clips) * len(self.frame_indices)

    def __getitem__(self, index):
        if not isinstance(index, int) or index < 0 or index >= len(self):
            raise IndexError("Dataset index out of range")
        if self.sample_mode == "clip":
            return [self.get_frame(index, frame) for frame in self.frame_indices]
        clip_index, frame_slot = divmod(index, len(self.frame_indices))
        return self.get_frame(clip_index, self.frame_indices[frame_slot])

    def get_frame(self, clip_index, frame_index):
        """Return one frame; metadata indices here are zero-based, output is 1-based."""
        if not 0 <= clip_index < len(self.clips) or not 0 <= frame_index < FRAME_COUNT:
            raise IndexError("Clip/frame index out of range")
        clip = self.clips[clip_index]
        name = clip["frames"][frame_index]
        image_path = self.image_root / name
        if not image_path.is_file():
            raise ValueError("Missing official input image: {}".format(image_path))
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            source_size = np.asarray(image.size, dtype=np.float64)
            image = image.resize(tuple(self.input_size.tolist()), Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)
            pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.
        pixels = np.ascontiguousarray((pixels - self.mean) / self.std, dtype=np.float32)
        if name not in self._calibration_cache:
            self._calibration_cache[name] = _read_calibration(self.calibration_root, name)
        K, R, t = self._calibration_cache[name]
        resize = np.diag(np.r_[self.input_size / source_size, 1.])
        source_from_input = np.column_stack((np.diag(source_size / self.input_size), np.zeros(2)))
        K_resized = resize @ K
        result = {"images": pixels, "calibs": np.column_stack((K_resized, np.zeros(3))).astype(np.float32),
                  "image_sizes": self.input_size.astype(np.float32),
                  "source_image_sizes": source_size.astype(np.float32),
                  "source_K": K.astype(np.float32), "source_from_input": source_from_input.astype(np.float32),
                  "lidar_to_camera_R": R.astype(np.float32), "lidar_to_camera_t": t.astype(np.float32),
                  "text": clip["text"], "video_id": clip["video_id"], "sequence_id": clip["sequence_id"],
                  "frame_index": np.int64(frame_index + 1), "image_name": name,
                  "encoding": ENCODING_NAME, "label_convention_verified": np.bool_(False)}
        if self.partition == "train":
            values = clip["lwhxyz"][frame_index]
            encoded = build_camera_targets(values[3:], values[:3], R, t, K, source_from_input, self.input_size)
            result["target"] = {
                "labels": np.asarray([clip["class_id"]], dtype=np.int64),
                "projected_center": encoded["normalized_center_xy"][None].astype(np.float32),
                "depth": np.asarray([[encoded["camera_z"]]], dtype=np.float32),
                "size_3d": values[[2, 1, 0]][None].astype(np.float32),
                "dimensions_lwh": values[:3][None].astype(np.float32),
                "native_center_xyz": values[3:][None].astype(np.float32),
                "camera_center_xyz": encoded["camera_geometric_center_xyz"][None].astype(np.float32),
                "center_in_input_bounds": np.asarray([encoded["center_in_input_bounds"]], dtype=bool),
                "has_bbox2d": np.asarray([False]), "has_heading": np.asarray([False]),
            }
        return result


def split_train_by_sequence(root, val_fraction=.2, seed=444):
    """Return reproducible internal TRAIN ID groups; never open official test.

    Recording IDs and any shared input-image basenames are unioned transitively
    before assigning groups. No split files are written. Original official
    competition train/test membership is never altered.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between zero and one")
    paths = sorted((Path(root) / "train" / "json").glob("*.json"))
    if not paths:
        raise ValueError("Missing official training clip JSONs")
    # This helper uses public metadata only, not six-value target tails.
    records = []
    for path in paths:
        raw = _read_object(path)
        records.append((_id(path.stem), _sequence(raw.get("sequence_id")),
                        [_basename(raw["frame{}".format(i)][0]) for i in range(FRAME_COUNT)]))
    parent = {clip: clip for clip, _, _ in records}
    def find(clip):
        while parent[clip] != clip:
            parent[clip] = parent[parent[clip]]
            clip = parent[clip]
        return clip
    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    by_scene, by_image = {}, {}
    for clip, sequence, images in records:
        if sequence in by_scene:
            join(clip, by_scene[sequence])
        by_scene[sequence] = clip
        for name in images:
            if name in by_image:
                join(clip, by_image[name])
            by_image[name] = clip
    groups = defaultdict(list)
    for clip, _, _ in records:
        groups[find(clip)].append(clip)
    group_ids = sorted(groups)
    if len(group_ids) < 2:
        raise ValueError("At least two original scene/shared-image groups are needed for validation")
    random.Random(seed).shuffle(group_ids)
    n_val = max(1, min(len(group_ids) - 1, int(round(len(group_ids) * val_fraction))))
    val_groups = set(group_ids[:n_val])
    train = sorted(clip for group, ids in groups.items() if group not in val_groups for clip in ids)
    val = sorted(clip for group, ids in groups.items() if group in val_groups for clip in ids)
    return {"train_ids": train, "val_ids": val,
            "train_sequence_ids": sorted({sequence for clip, sequence, _ in records if clip in train}),
            "val_sequence_ids": sorted({sequence for clip, sequence, _ in records if clip in val}),
            "group_count": len(group_ids), "method": "original sequence OR shared input image, training partition only"}
