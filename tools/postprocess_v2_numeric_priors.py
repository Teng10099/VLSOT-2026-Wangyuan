"""CPU-only, private numeric-prior candidates from a complete v2 inference run.

Example: python tools/postprocess_v2_numeric_priors.py --source RAW_RUN \
    --output NEW_DIRECTORY --mode initial_range_and_dimensions

No dataset, checkpoint loading, model forward, or target annotations are used.
The initial text range is an experimental constant-radius prior across a clip;
it is not a claim about the object's later motion or an official evaluation.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

try:
    from . import infer_official_v2 as infer
    from . import language_numeric_priors as priors
except ImportError:
    import infer_official_v2 as infer
    import language_numeric_priors as priors


MODES = ("dims_only", "initial_range_and_dimensions")
DECODED_KEYS = ("values_lwhxyz", "camera_center_xyz", "label_position_native_xyz",
                "normalized_center_xy", "camera_z", "dimensions_lwh")
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
CHECKPOINT_DIGEST_SCOPE = (
    "checkpoint_sha256 comes from the completed source run manifest. Each clip's "
    "checkpoint path, epoch, seed, and strict-load status must match that manifest; "
    "an optional per-clip checkpoint_sha256 must also match. Legacy diagnostics "
    "do not independently record a checkpoint digest; no weights are loaded here."
)
EXPERIMENT_NOTE = (
    "Private, unscored candidate. Text dimensions may replace raw LWH. In the range "
    "mode every frame's native XYZ retains its predicted direction and uses the "
    "public initial text range as radius. This may harm moving objects; no motion "
    "extrapolation or azimuth is used. Camera geometry is recalculated with each "
    "frame's recorded R, t and resized K. No h/2 or world transform is applied."
)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite JSON number: {}".format(value))


def reject_target_fields(value, location="root"):
    """Reject annotation-like keys; explicitly allow encoding/no-GT metadata."""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
            if normalized == "ground_truth_used_for_predictions":
                if item is not False:
                    raise ValueError("Source declares ground truth use at {}".format(location))
            elif normalized != "target_encoding" and (
                    normalized in ("target", "targets", "gt", "groundtruth", "ground_truth",
                                   "annotation", "annotations", "label", "labels")
                    or normalized.startswith(("gt_", "target_", "targets_", "ground_truth_"))
                    or normalized.endswith(("_gt", "_target", "_targets"))):
                raise ValueError("Target-like field forbidden: {}.{}".format(location, key))
            reject_target_fields(item, "{}.{}".format(location, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_target_fields(item, "{}[{}]".format(location, index))


def _read_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: {}".format(path))
    reject_target_fields(value, str(path))
    return value


def _digest(value, field):
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
        raise ValueError("Missing/invalid SHA256: {}".format(field))
    return value


def _same_numeric(actual, expected, name):
    array = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(expected, dtype=np.float64)
    if (array.shape != reference.shape or not np.isfinite(array).all()
            or not np.allclose(array, reference, rtol=1e-8, atol=1e-8)):
        raise ValueError("Raw decoded/geometry inconsistency: {}".format(name))


def validate_source(source):
    """Read ten complete raw diagnostics, verify geometry and committed TXT hashes."""
    source = Path(source).resolve()
    manifest_path = source / "inference_manifest.json"
    manifest = _read_json(manifest_path)
    if (manifest.get("schema_version") != 2 or manifest.get("inference_version") != infer.INFERENCE_VERSION
            or manifest.get("status") != "complete_format_valid_quality_unverified"
            or manifest.get("diagnostic_only") is not False
            or manifest.get("official_ten_sequence_format") is not True
            or manifest.get("frames_requested_per_clip") != 30):
        raise ValueError("Source must be a completed official v2 ten-clip/30-frame inference run")
    for key in ("target_encoding", "label_encoding"):
        if manifest.get(key) != infer.ENCODING:
            raise ValueError("Source {} must explicitly declare v2".format(key))
    for key in ("ground_truth_used_for_predictions", "world_transform_applied", "height_offset_applied"):
        if manifest.get(key) is not False:
            raise ValueError("Source {} must explicitly be false".format(key))
    ids = list(infer.OFFICIAL_IDS)
    if manifest.get("video_ids") != ids or [item.get("video_id") for item in manifest.get("clips", [])] != ids:
        raise ValueError("Source manifest requires exactly ten ordered official IDs")
    found = {path.parent.name for path in source.glob("*/diagnostics.json")}
    if found != set(ids):
        raise ValueError("Source diagnostics must contain exactly ten official IDs")
    checkpoint_sha = _digest(manifest.get("checkpoint_sha256"), "checkpoint_sha256")
    _digest(manifest.get("config_sha256"), "config_sha256")
    if not isinstance(manifest.get("checkpoint"), str) or not manifest["checkpoint"]:
        raise ValueError("Source checkpoint path is missing")
    if manifest.get("state_dict_load") != "strict_success":
        raise ValueError("Source checkpoint was not strictly loaded")
    size = infer._size_wh(manifest.get("input_size_wh"), "source input_size_wh")
    seed = manifest.get("seed_per_video")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError("Source seed is invalid")
    result_hashes = manifest.get("result_file_sha256")
    if not isinstance(result_hashes, dict) or set(result_hashes) != set(ids):
        raise ValueError("Source must commit hashes for all ten result TXT files")
    diagnostics, hashes = {}, {}
    for clip_id in ids:
        path = source / clip_id / "diagnostics.json"
        diagnostic = _read_json(path)
        for key in ("checkpoint", "checkpoint_epoch", "seed_per_video", "state_dict_load",
                    "target_encoding", "label_encoding"):
            if key not in diagnostic or diagnostic[key] != manifest.get(key):
                raise ValueError("Clip {} {} disagrees with source manifest".format(clip_id, key))
        if "checkpoint_sha256" in diagnostic and diagnostic["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("Clip {} checkpoint_sha256 disagrees".format(clip_id))
        if diagnostic.get("video_id") != clip_id or not isinstance(diagnostic.get("input_text"), str):
            raise ValueError("Clip ID/public input_text is invalid: {}".format(clip_id))
        if not isinstance(diagnostic.get("sequence_id"), str):
            raise ValueError("Clip sequence_id is missing: {}".format(clip_id))
        frames = diagnostic.get("frames")
        if (not isinstance(frames, list) or len(frames) != 30
                or any(not isinstance(item, dict) or isinstance(item.get("frame_index"), bool) for item in frames)
                or [item.get("frame_index") for item in frames] != list(range(1, 31))):
            raise ValueError("Clip {} must have 30 ordered frames".format(clip_id))
        for frame in frames:
            _same_numeric(frame.get("image_sizes"), size, "image_sizes")
            image = frame.get("image_name")
            if not isinstance(image, str) or Path(image).name != image or "/" in image or "\\" in image:
                raise ValueError("Frame image_name must be a basename")
            heads = {key: frame[key] for key in ("pred_boxes", "pred_depth", "pred_3d_dim")}
            decoded = infer.decode_prediction(dict(heads, target_encoding=infer.ENCODING), frame)
            for key in DECODED_KEYS:
                _same_numeric(frame.get(key), decoded[key], key)
        text = "\n".join(infer._format_row(i, np.asarray(frame["values_lwhxyz"], dtype=np.float64))
                         for i, frame in enumerate(frames, 1)) + "\n"
        expected = _digest(result_hashes.get(clip_id), "result_file_sha256/" + clip_id)
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != expected:
            raise ValueError("Raw diagnostic values disagree with source committed TXT SHA256: " + clip_id)
        diagnostics[clip_id] = diagnostic
        hashes[clip_id] = infer.sha256(path)
    return manifest, diagnostics, hashes


def synchronize_camera(decoded, frame):
    """Recalculate camera XYZ, camera Z and projected xy after native correction."""
    result = copy.deepcopy(decoded)
    native = np.asarray(result["label_position_native_xyz"], dtype=np.float64)
    rotation = np.asarray(frame["lidar_to_camera_R"], dtype=np.float64)
    translation = np.asarray(frame["lidar_to_camera_t"], dtype=np.float64)
    camera = rotation @ native + translation
    if not np.isfinite(camera).all() or camera[2] <= 0:
        raise ValueError("Numeric-prior camera center must remain finite and in front of the camera")
    projection = np.asarray(frame["calibs"], dtype=np.float64)[:, :3] @ camera
    xy = (projection[:2] / projection[2]) / np.asarray(frame["image_sizes"], dtype=np.float64)
    if not np.isfinite(xy).all():
        raise ValueError("Numeric-prior camera projection is nonfinite")
    result.update(camera_center_xyz=camera, camera_z=float(camera[2]), normalized_center_xy=xy,
                  camera_metrics_available=True)
    return result


def postprocess(source, output, mode):
    """Publish a new auditable candidate, retaining the raw decoded values per frame."""
    if mode not in MODES:
        raise ValueError("Unknown numeric-prior mode: {}".format(mode))
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from and outside the source directory")
    if output.exists():
        raise FileExistsError("Refusing to overwrite output directory: {}".format(output))
    source_manifest, diagnostics, hashes = validate_source(source)
    records_by_id, output_diagnostics, summaries = {}, {}, []
    for clip_id in infer.OFFICIAL_IDS:
        diagnostic = diagnostics[clip_id]
        parsed = priors.parse_initial_numeric_priors(diagnostic["input_text"])
        records = []
        for frame in diagnostic["frames"]:
            raw = {key: copy.deepcopy(frame[key]) for key in DECODED_KEYS}
            corrected = priors.apply_numeric_priors(raw, parsed, use_range=mode == MODES[1], use_dimensions=True)
            corrected = synchronize_camera(corrected, frame)
            record = copy.deepcopy(frame)
            record.update(corrected)
            record["raw_decoded"] = raw
            record["raw_model_heads_note"] = "pred_* fields remain unchanged raw model outputs; decoded fields are postprocessed"
            records.append(infer._jsonable(record))
        summary = {"video_id": clip_id, "parse_status": parsed["status"],
                   "range_applied_frames": sum(item["numeric_priors"]["range_applied"] for item in records),
                   "dimensions_applied_frames": sum(item["numeric_priors"]["dimensions_applied"] for item in records)}
        result = copy.deepcopy(diagnostic)
        result.update(frames=records, numeric_prior_mode=mode, numeric_prior_parse_audit=parsed,
                      source_diagnostic_sha256=hashes[clip_id],
                      checkpoint_sha256=source_manifest["checkpoint_sha256"],
                      checkpoint_digest_scope=CHECKPOINT_DIGEST_SCOPE,
                      experiment_note=EXPERIMENT_NOTE, private_candidate=True, officially_scored=False)
        records_by_id[clip_id], output_diagnostics[clip_id] = records, result
        summaries.append(summary)
    # Validate every frame before creating the new directory; never modify raw inputs.
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "postprocess_manifest.json"
    manifest = {"schema_version": 1, "postprocess_version": "numeric_priors_v2/1", "status": "writing",
                "private_candidate": True, "officially_scored": False, "platform_submission_ready": False,
                "strategy": mode, "experiment_note": EXPERIMENT_NOTE,
                "target_encoding": infer.ENCODING, "label_encoding": infer.ENCODING,
                "coordinate_note": source_manifest.get("coordinate_note", infer.COORDINATE_NOTE),
                "label_convention_verified": False, "ground_truth_used_for_predictions": False,
                "world_transform_applied": False, "height_offset_applied": False,
                "source_directory": str(source), "source_manifest": str(source / "inference_manifest.json"),
                "source_manifest_sha256": infer.sha256(source / "inference_manifest.json"),
                "source_diagnostic_sha256": hashes,
                "source_result_txt_hashes_verified": True,
                "source_checkpoint_sha256": source_manifest["checkpoint_sha256"],
                "checkpoint_sha256": source_manifest["checkpoint_sha256"],
                "checkpoint": source_manifest["checkpoint"], "checkpoint_epoch": source_manifest["checkpoint_epoch"],
                "checkpoint_digest_scope": CHECKPOINT_DIGEST_SCOPE,
                "config_sha256": source_manifest["config_sha256"], "model_options": source_manifest.get("model_options"),
                "seed_per_video": source_manifest["seed_per_video"],
                "temporal_mode": source_manifest.get("temporal_mode"),
                "input_size_wh": source_manifest["input_size_wh"], "frames_per_clip": 30,
                "video_ids": list(infer.OFFICIAL_IDS), "frame_count": 300, "clips": summaries,
                "module_sha256": {"language_numeric_priors.py": infer.sha256(Path(priors.__file__)),
                                  "postprocess_v2_numeric_priors.py": infer.sha256(Path(__file__)),
                                  "infer_official_v2.py": infer.sha256(Path(infer.__file__))}}
    infer._write_json(manifest_path, manifest)
    try:
        for clip_id, diagnostic in output_diagnostics.items():
            directory = output / clip_id
            directory.mkdir()
            infer._write_json(directory / "diagnostics.json", diagnostic)
        manifest.update(infer.package_predictions(output, records_by_id, 30))
        manifest["status"] = "private_candidate_complete_format_valid_unscored"
        manifest["output_diagnostic_sha256"] = {
            clip: infer.sha256(output / clip / "diagnostics.json") for clip in infer.OFFICIAL_IDS}
    except Exception as error:
        manifest.update(status="failed", error="{}: {}".format(type(error).__name__, error))
        infer._write_json(manifest_path, manifest)
        raise
    infer._write_json(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Completed raw v2 inference directory")
    parser.add_argument("--output", required=True, help="New directory outside the source")
    parser.add_argument("--mode", required=True, choices=MODES)
    args = parser.parse_args(argv)
    result = postprocess(args.source, args.output, args.mode)
    print(json.dumps({key: result[key] for key in ("status", "strategy", "zip", "zip_sha256", "frame_count")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
