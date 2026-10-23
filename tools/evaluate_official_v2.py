r"""Evaluate ONE explicitly supplied v2 checkpoint on official TRAIN val IDs.

python tools/evaluate_official_v2.py --source-project SOURCE --checkpoint RUN/CHECKPOINT.pth \
    --config RUN/resolved_config.yaml --data-root OFFICIAL_DATASET --calibration-root SOURCE/calib \
    --val-ids VAL_IDS.txt --output NEW_EVALUATION.json --gpu 0

Every selected clip is evaluated for 30 ordered frames by default. --frames
and --max-clips produce explicitly bounded diagnostics. This command never
selects checkpoints, creates an optimizer, modifies weights, or opens official
TEST. Numeric XYZ error uses the official TRAIN labels directly; the physical
native/geometric-center interpretation remains an explicit unverified contract.
"""

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np

try:
    from .infer_official_v2 import (ENCODING, COORDINATE_NOTE, INFERENCE_VERSION,
                                    decode_prediction, load_v2_runtime, sha256)
except ImportError:
    from infer_official_v2 import (ENCODING, COORDINATE_NOTE, INFERENCE_VERSION,
                                   decode_prediction, load_v2_runtime, sha256)


def read_val_ids(path):
    """Require literal six-digit IDs from a user-selected, nonempty file."""
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not ids or any(not re.fullmatch(r"[0-9]{6}", item) for item in ids):
        raise ValueError("val ID file must contain nonempty six-digit ID lines")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in explicit val ID file")
    return ids


def check_recorded_partition(ids, manifests):
    """Refuse known training IDs; do not infer a split or choose a checkpoint."""
    checked = 0
    for record in manifests:
        train = record.get("train_clip_ids")
        val = record.get("val_clip_ids")
        if train is not None:
            if not isinstance(train, list) or set(ids) & set(train):
                raise ValueError("Requested validation IDs overlap the recorded training selection")
            checked += 1
        if val:
            if not isinstance(val, list) or not set(ids) <= set(val):
                raise ValueError("Requested validation IDs differ from the recorded internal validation selection")
    return {"training_id_overlap_check": "passed" if checked else "not_recorded",
            "recorded_training_partitions_checked": checked}


def frame_errors(decoded, target):
    """Compare predictions with ONLY the selected official TRAIN frame labels."""
    native = np.asarray(target["native_center_xyz"], dtype=np.float64)
    camera = np.asarray(target["camera_center_xyz"], dtype=np.float64)
    dimensions = np.asarray(target["dimensions_lwh"], dtype=np.float64)
    predicted_native = np.asarray(decoded["label_position_native_xyz"], dtype=np.float64)
    predicted_camera = np.asarray(decoded["camera_center_xyz"], dtype=np.float64)
    predicted_dims = np.asarray(decoded["dimensions_lwh"], dtype=np.float64)
    if (native.shape != (1, 3) or camera.shape != (1, 3) or dimensions.shape != (1, 3)
            or any(value.shape != (3,) for value in (predicted_native, predicted_camera, predicted_dims))
            or not all(np.isfinite(value).all() for value in
                       (native, camera, dimensions, predicted_native, predicted_camera, predicted_dims))):
        raise ValueError("Invalid evaluation labels/predictions")
    return {"native_xyz_error_m": float(np.linalg.norm(predicted_native - native[0])),
            "camera_xyz_error_m": float(np.linalg.norm(predicted_camera - camera[0])),
            "native_xyz_abs_error_per_axis_m": np.abs(predicted_native - native[0]),
            "camera_xyz_abs_error_per_axis_m": np.abs(predicted_camera - camera[0]),
            "dimensions_abs_error_lwh_m": np.abs(predicted_dims - dimensions[0])}


def summarize_errors(records):
    """Aggregate per frame, with explicit counts; these are not platform scores."""
    if not records:
        raise ValueError("Cannot summarize an empty evaluation")
    native = np.asarray([record["native_xyz_error_m"] for record in records], dtype=np.float64)
    camera = np.asarray([record["camera_xyz_error_m"] for record in records], dtype=np.float64)
    native_axis = np.stack([record["native_xyz_abs_error_per_axis_m"] for record in records])
    camera_axis = np.stack([record["camera_xyz_abs_error_per_axis_m"] for record in records])
    dims = np.stack([record["dimensions_abs_error_lwh_m"] for record in records])
    if not all(np.isfinite(value).all() for value in (native, camera, native_axis, camera_axis, dims)):
        raise ValueError("Nonfinite evaluation errors")
    return {"frame_count": len(records), "native_xyz_mean_error_m": float(native.mean()),
            "native_xyz_median_error_m": float(np.median(native)),
            "camera_xyz_mean_error_m": float(camera.mean()),
            "native_xyz_mae_per_axis_m": native_axis.mean(axis=0).tolist(),
            "camera_xyz_mae_per_axis_m": camera_axis.mean(axis=0).tolist(),
            "dimensions_mae_lwh_m": dims.mean(axis=0).tolist(),
            "dimensions_mean_absolute_error_m": float(dims.mean())}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ("source-project", "checkpoint", "config", "data-root", "calibration-root", "val-ids", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--frames", type=int, choices=range(1, 31), default=30)
    parser.add_argument("--audit-numeric-priors", action="store_true",
                        help="Report experimental public-caption priors alongside unchanged raw metrics")
    parser.add_argument("--max-clips", type=int, help="Evaluate a prefix of the explicit val list; diagnostic only")
    parser.add_argument("--input-width", type=int)
    parser.add_argument("--input-height", type=int)
    args = parser.parse_args(argv)
    if args.max_clips is not None and args.max_clips <= 0:
        parser.error("--max-clips must be positive")
    if (args.input_width is None) != (args.input_height is None):
        parser.error("Provide both --input-width and --input-height")
    return args


def run(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite evaluation output: {}".format(output))
    requested_ids = read_val_ids(args.val_ids)
    missing = [clip for clip in requested_ids
               if not (Path(args.data_root) / "train" / "json" / (clip + ".json")).is_file()]
    if missing:
        raise ValueError("Validation IDs are not present in official TRAIN: {}".format(missing))
    manifests = []
    for path in sorted({Path(args.config).resolve().parent / "run_manifest.json",
                        Path(args.checkpoint).resolve().parent / "run_manifest.json"}):
        if path.is_file():
            manifests.append(json.loads(path.read_text(encoding="utf-8-sig")))
    partition_check = check_recorded_partition(requested_ids, manifests)
    explicit = None if args.input_width is None else (args.input_width, args.input_height)
    model, contract, provenance = load_v2_runtime(
        args.source_project, args.checkpoint, args.config, args.gpu, explicit)
    # load_v2_runtime configures CUDA visibility before any torch/model import.
    import torch
    from lib.datasets.mono3drefer.official_vlsot_dataset import OfficialVLSOTDataset
    from lib.helpers.trainer_helper import reset_temporal_state
    ids = requested_ids if args.max_clips is None else requested_ids[:args.max_clips]
    dataset = OfficialVLSOTDataset(args.data_root, args.calibration_root, split="val", clip_ids=ids,
                                   input_size_wh=contract["input_size_wh"], sample_mode="frame",
                                   label_center_reference="geometric_center")
    seed = contract["seed_per_video"]
    all_errors, per_clip = [], []
    audit_priors = bool(getattr(args, "audit_numeric_priors", False))
    prior_totals = {"dimensions_only": [], "initial_range_and_dimensions": []}
    if audit_priors:
        from tools.language_numeric_priors import parse_initial_numeric_priors, apply_numeric_priors
    started = time.perf_counter()
    with torch.inference_mode():
        for clip_index, clip in enumerate(dataset.clips):
            reset_temporal_state(model)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            errors = []
            prior_errors = {name: [] for name in prior_totals}
            parsed_prior = None
            clip_start = time.perf_counter()
            for index in range(args.frames):
                frame = dataset.get_frame(clip_index, index)
                if "target" not in frame:
                    raise ValueError("Only official TRAIN frames with validation targets may be evaluated")
                prediction = model(torch.from_numpy(frame["images"]).unsqueeze(0).cuda(),
                                   torch.from_numpy(frame["calibs"]).unsqueeze(0).cuda(),
                                   torch.from_numpy(frame["image_sizes"]).unsqueeze(0).cuda(), [frame["text"]])
                heads = {key: prediction[key].detach().float().cpu().numpy()
                         for key in ("pred_boxes", "pred_depth", "pred_3d_dim")}
                heads["target_encoding"] = prediction.get("target_encoding")
                decoded = decode_prediction(heads, frame)
                errors.append(frame_errors(decoded, frame["target"]))
                if audit_priors:
                    if parsed_prior is None:
                        parsed_prior = parse_initial_numeric_priors(frame["text"])
                    for name in prior_errors:
                        alternative = apply_numeric_priors(
                            decoded, parsed_prior,
                            use_range=name == "initial_range_and_dimensions", use_dimensions=True)
                        native = np.asarray(alternative["label_position_native_xyz"], dtype=np.float64)
                        alternative["camera_center_xyz"] = (
                            np.asarray(frame["lidar_to_camera_R"], dtype=np.float64) @ native
                            + np.asarray(frame["lidar_to_camera_t"], dtype=np.float64).reshape(3))
                        prior_errors[name].append(frame_errors(alternative, frame["target"]))
            summary = {"video_id": clip["video_id"], "sequence_id": clip["sequence_id"],
                       "elapsed_seconds": time.perf_counter() - clip_start}
            summary.update(summarize_errors(errors))
            all_errors.extend(errors)
            if audit_priors:
                summary["experimental_numeric_prior_metrics"] = {
                    name: summarize_errors(values) for name, values in prior_errors.items()}
                for name, values in prior_errors.items():
                    prior_totals[name].extend(values)
            per_clip.append(summary)
            print("EVALUATED_CLIP " + json.dumps(summary), flush=True)
    complete = args.frames == 30 and args.max_clips is None
    report = {"schema_version": 1, "evaluation_version": INFERENCE_VERSION,
              "status": "complete_selected_validation" if complete else "bounded_validation_diagnostic",
              "diagnostic_only": not complete, "label_encoding": ENCODING,
              "label_convention_verified": False, "coordinate_note": COORDINATE_NOTE,
              "scope": "explicit internal validation IDs from official TRAIN only; not an official platform score",
              "official_test_accessed": False, "optimizer_created": False, "checkpoint_modified": False,
              "model_eval_mode": True, "inference_mode": True,
              "data_root": str(Path(args.data_root).resolve()),
              "calibration_root": str(Path(args.calibration_root).resolve()),
              "calibration_source": "Existing matching-image source calibration; official ZIP contains none",
              "val_id_file": str(Path(args.val_ids).resolve()), "val_id_file_sha256": sha256(args.val_ids),
              "requested_clip_count": len(requested_ids), "evaluated_clip_count": len(ids),
              "evaluated_clip_ids": ids, "frames_per_clip": args.frames,
              "expected_frame_count": len(ids) * args.frames,
              "input_size_wh": list(contract["input_size_wh"]), "seed_per_video": seed,
              "temporal_mode": "reset memory/phase per clip; carry within ordered selected frames",
              "elapsed_seconds": time.perf_counter() - started, "clips": per_clip,
              "evaluation_script_sha256": sha256(__file__), "partition_check": partition_check}
    report.update(provenance)
    report["metrics"] = summarize_errors(all_errors)
    if audit_priors:
        report["experimental_numeric_prior_metrics"] = {
            name: summarize_errors(values) for name, values in prior_totals.items()}
        report["numeric_prior_audit_policy"] = (
            "Raw predictions and metrics remain unchanged. Alternatives use public caption initial "
            "range/dimensions as constants over the clip; no target-dependent adjustment. Native "
            "direction is preserved; camera metrics are recomputed using the supplied calibration. "
            "Motion can invalidate constant initial range. This audit does not enable TEST postprocessing.")
    if report["metrics"]["frame_count"] != report["expected_frame_count"]:
        raise RuntimeError("Evaluation frame count does not match explicit selection")
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "status": report["status"], "metrics": report["metrics"]}), flush=True)
    return report


if __name__ == "__main__":
    run(parse_args())
