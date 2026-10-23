"""Package smoke_infer predictions using an explicit native-coordinate policy.

This validates the competition FILE FORMAT. It does not establish that the
checkpoint or its inherited native label coordinates meet the competition's
physical coordinate definition. No annotations or dataset root are consulted.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import numpy as np

try:
    from .geometry_decode import decode_native_predictions
except ImportError:
    from geometry_decode import decode_native_predictions


OFFICIAL_IDS = ("000237", "001483", "003235", "006543", "024386",
                "033447", "035729", "036120", "047771", "051657")
HEAD_KEYS = ("pred_logits", "pred_boxes", "pred_3d_dim", "pred_depth")
PRIMARY_COORDINATE_SOURCES = [
    "https://github.com/AIR-THU/DAIR-V2X-Seq/blob/main/dataset/v2x-seq-spd/README.md",
    "https://github.com/AIR-THU/DAIR-V2X/blob/main/tools/dataset_converter/utils.py#L211-L217",
    "https://github.com/AIR-THU/DAIR-V2X/blob/main/v2x/v2x_utils/transformation_utils.py#L163-L184",
]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected(expected_ids, diagnostic_single_id):
    if diagnostic_single_id is not None:
        if expected_ids is not None:
            raise ValueError("Choose official IDs OR diagnostic single ID")
        if not re.fullmatch(r"[0-9]{6}", str(diagnostic_single_id)):
            raise ValueError("Diagnostic ID must preserve six numeric digits")
        return (str(diagnostic_single_id),), False
    if expected_ids != "official":
        raise ValueError("Formal --expected-ids must be 'official' (fixed ten IDs)")
    return OFFICIAL_IDS, True


def _read_world_transforms(path, ids):
    """Read per-image extrinsics and provenance; no annotations are accepted."""
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("world transform manifest requires schema_version=1")
    if (document.get("source_coordinate_system") != "infrastructure_virtual_lidar" or
            document.get("target_coordinate_system") != "virtual_world"):
        raise ValueError("World transforms must map infrastructure_virtual_lidar to virtual_world")
    verification = document.get("source_verification")
    if not isinstance(verification, str) or not verification.strip():
        raise ValueError("World transforms require a source_verification provenance statement")
    sequences = document.get("sequences")
    if not isinstance(sequences, dict) or set(sequences) != set(ids):
        raise ValueError("World transforms must cover exactly the expected sequence IDs")
    validated = {}
    for video_id in ids:
        frames = sequences[video_id]
        if not isinstance(frames, list) or len(frames) != 30:
            raise ValueError("World transforms require 30 frames for {}".format(video_id))
        mapping = {}
        for frame in frames:
            index = frame.get("frame_index")
            if (isinstance(index, bool) or not isinstance(index, int) or
                    index not in range(1, 31) or index in mapping):
                raise ValueError("World transforms have invalid/duplicate frame index")
            image_name = frame.get("image_name")
            calibration_path = frame.get("calibration_path")
            calibration_hash = frame.get("calibration_sha256")
            if not isinstance(image_name, str) or not image_name:
                raise ValueError("Each world transform requires an image_name")
            if not isinstance(calibration_path, str) or not calibration_path:
                raise ValueError("Each world transform requires its calibration_path")
            if (not isinstance(calibration_hash, str) or
                    not re.fullmatch(r"[0-9a-fA-F]{64}", calibration_hash)):
                raise ValueError("Each world transform requires a calibration_sha256")
            calibration_name = calibration_path.replace("\\", "/").rsplit("/", 1)[-1]
            if Path(calibration_name).stem != Path(image_name).stem:
                raise ValueError("World calibration stem differs from image stem")
            rotation = np.asarray(frame.get("rotation"), dtype=np.float64)
            translation = np.asarray(frame.get("translation"), dtype=np.float64)
            if rotation.shape != (3, 3) or translation.shape not in ((3,), (3, 1)):
                raise ValueError("World rotation/translation must be 3x3 and 3 or 3x1")
            if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
                raise ValueError("World extrinsics must be finite")
            if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4, rtol=0) or
                    not np.isclose(np.linalg.det(rotation), 1., atol=1e-4, rtol=0)):
                raise ValueError("World rotation must be a proper orthonormal rotation")
            mapping[index] = {
                "frame_index": index,
                "image_name": image_name,
                "rotation": rotation.tolist(),
                "translation": translation.reshape(3).tolist(),
                "calibration_path": calibration_path,
                "calibration_sha256": calibration_hash.lower(),
            }
        validated[video_id] = mapping
    return validated, {
        "manifest_path": str(path),
        "manifest_sha256": _sha256(path),
        "source_coordinate_system": document["source_coordinate_system"],
        "target_coordinate_system": document["target_coordinate_system"],
        "source_verification": verification,
        "source_verification_is_provided_provenance_not_rechecked_by_exporter": True,
        "official_documentation": PRIMARY_COORDINATE_SOURCES,
    }


def _read_sequence(diagnostic_path, world_transforms=None):
    record = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    video_id = str(record.get("video_id", ""))
    if not re.fullmatch(r"[0-9]{6}", video_id):
        raise ValueError("Invalid video_id in {}".format(diagnostic_path))
    if record.get("state_dict_load") != "strict_success":
        raise ValueError("{} did not record strict task-checkpoint loading".format(video_id))
    frames = record.get("frames")
    if not isinstance(frames, list) or len(frames) != 30:
        raise ValueError("{} requires exactly 30 completed frames".format(video_id))
    frame_map = {}
    for frame in frames:
        index = frame.get("frame_index")
        if isinstance(index, bool) or not isinstance(index, int) or index not in range(1, 31):
            raise ValueError("{} has invalid frame_index".format(video_id))
        if index in frame_map:
            raise ValueError("{} has duplicate frame {}".format(video_id, index))
        frame_map[index] = frame
    if set(frame_map) != set(range(1, 31)):
        raise ValueError("{} must contain frames 1 through 30".format(video_id))

    npz_path = diagnostic_path.with_name("raw_outputs.npz")
    if not npz_path.is_file():
        raise ValueError("Missing paired raw_outputs.npz for {}".format(video_id))
    rows = []
    native_rows = []
    frame_manifest = []
    with np.load(str(npz_path), allow_pickle=False) as raw:
        for index in range(1, 31):
            frame = frame_map[index]
            heads = {}
            for key in HEAD_KEYS:
                array_name = "frame_{:02d}_{}".format(index, key)
                if array_name not in raw.files:
                    raise ValueError("{} missing {}".format(video_id, array_name))
                heads[key] = raw[array_name]
                if key in frame and not np.array_equal(
                        np.asarray(frame[key]), heads[key]):
                    raise ValueError("{} frame {} JSON/NPZ disagree for {}".format(
                        video_id, index, key))
            if heads["pred_logits"].ndim != 3 or heads["pred_logits"].shape[0] != 1:
                raise ValueError("Each frame must contain exactly one inference sample")
            for key in ("source_size_wh", "input_size_wh", "source_from_input"):
                if key not in frame:
                    raise ValueError("{} frame {} lacks {}".format(video_id, index, key))
            classes = heads["pred_logits"].shape[-1]
            # The checked-in config uses meanshape=False. We do not recover
            # means from GT classes or substitute dimensions for invalid heads.
            decoded = decode_native_predictions(
                heads, frame["source_size_wh"], None,
                np.zeros((classes, 3), dtype=np.float64),
                center_encoding="legacy_native_xy",
                input_sizes_wh=frame["input_size_wh"],
                source_from_input=frame["source_from_input"],
            )
            values = np.concatenate((decoded["dimensions_lwh"][0],
                                     decoded["label_position_native_xyz"][0]))
            if not np.isfinite(values).all() or np.any(values[:3] <= 0):
                raise ValueError("{} frame {} has invalid native box".format(video_id, index))
            native_rows.append(_format_row(index, values))
            frame_provenance = {
                "frame_index": index,
                "image_name": frame.get("image_name"),
                "selected_query": int(decoded["query_index"][0]),
                "selected_class": int(decoded["class_id"][0]),
            }
            if world_transforms is not None:
                if video_id not in world_transforms:
                    raise ValueError("No world transforms for {}".format(video_id))
                transform = world_transforms[video_id][index]
                if transform["image_name"] != frame.get("image_name"):
                    raise ValueError("World transform image does not match inference image")
                values = values.copy()
                values[3:] = (np.asarray(transform["rotation"]) @ values[3:] +
                              np.asarray(transform["translation"]))
                frame_provenance["world_transform"] = transform
            rows.append(_format_row(index, values))
            frame_manifest.append(frame_provenance)
    text = "\n".join(rows) + "\n"
    native_text = "\n".join(native_rows) + "\n"
    _validate_text(text, video_id)
    _validate_text(native_text, video_id)
    provenance = {
        "video_id": video_id,
        "diagnostics_path": str(diagnostic_path.resolve()),
        "diagnostics_sha256": _sha256(diagnostic_path),
        "raw_outputs_sha256": _sha256(npz_path),
        "checkpoint": record.get("checkpoint"),
        "checkpoint_epoch": record.get("checkpoint_epoch"),
        "state_dict_load": record["state_dict_load"],
        "frames": frame_manifest,
    }
    return video_id, text, native_text, provenance


def _format_row(index, values):
    if not np.isfinite(values).all() or np.any(values[:3] <= 0):
        raise ValueError("Transformed box must be finite with positive dimensions")
    # Retain small positive dimensions without rounding them down to zero.
    return "{} {}".format(index, " ".join(format(float(x), ".12g") for x in values))


def _validate_text(text, video_id):
    lines = text.splitlines()
    if len(lines) != 30:
        raise ValueError("{} TXT does not have 30 lines".format(video_id))
    for index, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 7 or parts[0] != str(index):
            raise ValueError("{} TXT frame order or columns invalid".format(video_id))
        values = np.asarray([float(value) for value in parts[1:]])
        if not np.isfinite(values).all() or np.any(values[:3] <= 0):
            raise ValueError("{} serialized box is invalid".format(video_id))


def _verify_zip(path, ids):
    expected_names = ["result/{}.txt".format(video_id) for video_id in ids]
    with zipfile.ZipFile(str(path)) as archive:
        if archive.namelist() != expected_names:
            raise ValueError("ZIP must contain only the ordered result/*.txt members")
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
        for video_id, name in zip(ids, expected_names):
            _validate_text(archive.read(name).decode("utf-8"), video_id)


def _write_zip(path, ids, text_by_id):
    with zipfile.ZipFile(str(path), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for video_id in ids:
            info = zipfile.ZipInfo("result/{}.txt".format(video_id), (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, text_by_id[video_id].encode("utf-8"))
    _verify_zip(path, ids)


def export_submission(input_dir, output_dir, coordinate_policy,
                      expected_ids="official", diagnostic_single_id=None,
                      world_transforms=None):
    """Return output paths after validation and atomic, no-overwrite publishing.

    Official mode requires the fixed ten IDs. A diagnostic single-sequence ZIP
    still requires 30 valid frames and is explicitly marked non-official.
    """
    if coordinate_policy not in ("legacy_native_xy", "legacy_native_xy_to_world"):
        raise ValueError("Explicit legacy_native_xy coordinate policy is required")
    ids, official = _expected(expected_ids, diagnostic_single_id)
    to_world = coordinate_policy == "legacy_native_xy_to_world"
    if to_world and world_transforms is None:
        raise ValueError("legacy_native_xy_to_world requires --world-transforms")
    if not to_world and world_transforms is not None:
        raise ValueError("--world-transforms requires legacy_native_xy_to_world policy")
    transforms, transform_provenance = (None, None)
    if to_world:
        transforms, transform_provenance = _read_world_transforms(world_transforms, ids)
    source = Path(input_dir).resolve()
    destination = Path(output_dir).resolve()
    if not source.is_dir():
        raise ValueError("Input directory does not exist")
    if destination.exists():
        raise FileExistsError("Refusing to overwrite output directory: {}".format(destination))
    diagnostic_paths = sorted(source.rglob("diagnostics.json"))
    if not diagnostic_paths:
        raise ValueError("No diagnostics.json found recursively")
    sequences = {}
    for path in diagnostic_paths:
        video_id, text, native_text, provenance = _read_sequence(path, transforms)
        if video_id in sequences:
            raise ValueError("Duplicate video ID {} (multiple inference runs)".format(video_id))
        sequences[video_id] = (text, native_text, provenance)
    if set(sequences) != set(ids):
        raise ValueError("Input IDs differ: missing={}, unexpected={}".format(
            sorted(set(ids) - set(sequences)), sorted(set(sequences) - set(ids))))

    manifest = {
        "status": "format_validated_under_explicit_native_coordinate_assumption",
        "official_ten_sequence_format": official,
        "diagnostic_only": not official,
        "coordinate_policy": coordinate_policy,
        "coordinate_system": "virtual_world" if to_world else "unknown_native_label",
        "native_coordinate_system": "infrastructure_virtual_lidar" if to_world else "unknown_native_label",
        "world_transform_applied": to_world,
        "world_transform_structure_validated": to_world,
        "calibration_file_hashes_recomputed_by_exporter": False,
        "world_coordinates_verified": False,
        "platform_coordinate_interpretation_confirmed_by_scoring": False,
        "model_accuracy_verified": False,
        "ground_truth_used_for_predictions": False,
        "class_mean_policy": "meanshape=False; zero class means, as checked-in config",
        "decode_assumptions": [
            "Current dataset source, not proven historical checkpoint training source",
            "cx,cy times actual input width,height, then saved inverse affine",
            "Second native component adds predicted h/2 to reverse dataset encoding",
            "Third native component is pred_depth[...,0] without an extra transform",
            "Predicted native h,w,l are reordered to l,w,h; no abs, clamp, or GT fallback",
            ("Native supervision traced to infrastructure virtual LiDAR through supplied source verification; "
             "official per-image R@p+t transforms applied; dimensions unchanged; platform interpretation unconfirmed"
             if to_world else "Native physical axes and compatibility with platform world coordinates remain unverified"),
        ],
        "expected_ids": list(ids),
        "archive_members": ["result/{}.txt".format(video_id) for video_id in ids],
        "sequences": [sequences[video_id][2] for video_id in ids],
    }
    if to_world:
        manifest["world_transform_provenance"] = transform_provenance
        manifest["world_transform_equation"] = "p_world = rotation @ p_native + translation"
        manifest["dimensions_transform"] = "Physical box l,w,h retained; dimensions are not rotated as a point"
        manifest["native_backup_archive"] = "native_backup.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".vlsot_export_", dir=str(destination.parent)))
    published = []
    created_destination = False
    try:
        staged_zip = stage / "result.zip"
        _write_zip(staged_zip, ids, {video_id: sequences[video_id][0] for video_id in ids})
        manifest["result_zip_sha256"] = _sha256(staged_zip)
        staged_backup = None
        if to_world:
            staged_backup = stage / "native_backup.zip"
            _write_zip(staged_backup, ids, {video_id: sequences[video_id][1] for video_id in ids})
            manifest["native_backup_zip_sha256"] = _sha256(staged_backup)
        staged_manifest = stage / "manifest.json"
        staged_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
        # Reserve a fresh directory atomically. Hard-link publication is atomic
        # and fails if any destination file already exists, unlike os.replace.
        destination.mkdir(exist_ok=False)
        created_destination = True
        artifacts = [staged_manifest]
        if staged_backup is not None:
            artifacts.append(staged_backup)
        artifacts.append(staged_zip)  # Publish main result only after provenance/backup.
        for artifact in artifacts:
            target = destination / artifact.name
            os.link(str(artifact), str(target))
            published.append(target)
        result = {"zip": str(destination / "result.zip"),
                  "manifest": str(destination / "manifest.json"),
                  "official_ten_sequence_format": official,
                  "coordinate_system": manifest["coordinate_system"],
                  "world_coordinates_verified": False}
        if to_world:
            result["native_backup_zip"] = str(destination / "native_backup.zip")
        return result
    except Exception:
        for artifact in reversed(published):
            artifact.unlink()
        if created_destination:
            try:
                destination.rmdir()  # Never recursively remove someone else's content.
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(str(stage))  # Only our fresh, generated staging directory.


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="Must not already exist")
    parser.add_argument("--coordinate-policy", required=True,
                        choices=["legacy_native_xy", "legacy_native_xy_to_world"])
    parser.add_argument("--world-transforms", help="Per-frame official R/T and provenance JSON")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--expected-ids", choices=["official"],
                      help="Require the fixed ten official IDs")
    mode.add_argument("--diagnostic-single-id", help="One six-digit ID; not an official submission")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        result = export_submission(args.input_dir, args.output_dir,
                                   args.coordinate_policy, args.expected_ids,
                                   args.diagnostic_single_id, args.world_transforms)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        raise SystemExit("Export failed: {}".format(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
