r"""Infer official TEST inputs with an explicitly trained camera-v2 checkpoint.

python tools/infer_official_v2.py --source-project SOURCE --checkpoint RUN/checkpoint_best.pth \
    --config RUN/resolved_config.yaml --data-root OFFICIAL_DATASET --calibration-root SOURCE/calib \
    --output NEW_OUTPUT --gpu 0

--frames 1..29 produces diagnostic files ONLY. A result ZIP requires exactly
the ten official clips and 30 ordered frames each. No test targets, h/2 offset,
or virtual-LiDAR-to-world transformation are used. XYZ is decoded to the same
assumed native coordinate representation used by the official TRAIN labels;
the geometric-center/native interpretation remains an unverified assumption.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

try:
    from .export_vlsot import OFFICIAL_IDS, _format_row, _validate_text, _write_zip
except ImportError:
    from export_vlsot import OFFICIAL_IDS, _format_row, _validate_text, _write_zip


ENCODING = "camera_geometric_center_v2"
INFERENCE_VERSION = "official_v2/1"
COORDINATE_NOTE = (
    "XYZ reproduces the official TRAIN numeric native-coordinate representation under the "
    "experimental geometric-center contract. camera=R@native+t; native=solve(R,camera-t). "
    "No h/2 shift and no virtual-LiDAR-to-world transform. Native frame/center semantics "
    "are not independently verified; format validity does not establish model quality."
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _size_wh(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)):
        raise ValueError("{} must contain two positive integer dimensions".format(name))
    return tuple(value)


def validate_contract(config, checkpoint, run_manifests=(), explicit_size_wh=None):
    """Reject legacy weights/configuration and resolve recorded training resolution."""
    model = config.get("model", {})
    if not isinstance(model, dict) or model.get("target_encoding") != ENCODING:
        raise ValueError("Configuration must explicitly declare camera_geometric_center_v2")
    if not isinstance(checkpoint, dict) or checkpoint.get("target_encoding") != ENCODING:
        raise ValueError("Checkpoint must explicitly declare camera_geometric_center_v2; legacy is forbidden")
    options = checkpoint.get("model_options", {})
    if not isinstance(options, dict):
        raise ValueError("Checkpoint model_options must be a mapping")
    behavior_flags = {}
    for name in ("official_correct_padding_mask", "official_temporal_residual"):
        saved = options.get(name, False)
        configured = bool(model.get(name, False))
        if not isinstance(saved, bool) or saved != configured:
            raise ValueError("Checkpoint/config {} behavior disagrees".format(name))
        behavior_flags[name] = configured
    resolutions = []
    if "input_size_hw" in model:
        h, w = _size_wh(model["input_size_hw"], "model.input_size_hw")
        resolutions.append((w, h))
    if "input_size_hw" in options:
        h, w = _size_wh(options["input_size_hw"], "checkpoint model_options.input_size_hw")
        resolutions.append((w, h))
    seed = config.get("random_seed", 444)
    recorded_seeds = []
    for manifest in run_manifests:
        if manifest.get("target_encoding") != ENCODING:
            raise ValueError("Neighboring run_manifest target encoding disagrees with v2")
        args = manifest.get("args", {})
        if "input_width" in args or "input_height" in args:
            resolutions.append(_size_wh([args.get("input_width"), args.get("input_height")], "run_manifest input WH"))
        if "label_center_reference" in args and args["label_center_reference"] != "geometric_center":
            raise ValueError("Run manifest uses another label-center contract")
        if "seed" in args:
            recorded_seeds.append(args["seed"])
    if explicit_size_wh is not None:
        resolutions.append(_size_wh(list(explicit_size_wh), "explicit input WH"))
    if not resolutions:
        raise ValueError("Training input size is missing; provide recorded config/run_manifest or explicit input width and height")
    if len(set(resolutions)) != 1:
        raise ValueError("Training/configuration/explicit input resolutions disagree")
    if recorded_seeds:
        if len(set(recorded_seeds)) != 1:
            raise ValueError("Run manifests disagree about the training/inference seed")
        seed = recorded_seeds[0]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError("Seed must be an integer in [0,2**32)")
    return {"input_size_wh": resolutions[0], "seed_per_video": seed,
            "target_encoding": ENCODING, "label_convention_verified": False,
            "model_options": dict(behavior_flags,
                                  input_size_hw=[resolutions[0][1], resolutions[0][0]])}


def decode_prediction(predictions, frame):
    """Decode ONE referred object via resized K; never access target annotations."""
    if predictions.get("target_encoding") != ENCODING:
        raise ValueError("Model output does not declare the v2 target encoding")
    arrays = {}
    for key, shape in (("pred_boxes", (1, 1, 6)), ("pred_depth", (1, 1, 2)),
                       ("pred_3d_dim", (1, 1, 3))):
        value = np.asarray(predictions[key], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("Invalid v2 prediction shape/values: {}".format(key))
        arrays[key] = value
    normalized_xy = arrays["pred_boxes"][0, 0, :2]
    depth = arrays["pred_depth"][0, 0, 0]
    hwl = arrays["pred_3d_dim"][0, 0]
    if depth <= 0 or np.any(hwl <= 0):
        raise ValueError("Predicted camera depth and dimensions must be positive")
    size = np.asarray(frame["image_sizes"], dtype=np.float64)
    projection = np.asarray(frame["calibs"], dtype=np.float64)
    rotation = np.asarray(frame["lidar_to_camera_R"], dtype=np.float64)
    translation = np.asarray(frame["lidar_to_camera_t"], dtype=np.float64)
    if (size.shape != (2,) or np.any(size <= 0) or projection.shape != (3, 4)
            or rotation.shape != (3, 3) or translation.shape != (3,)
            or not all(np.isfinite(value).all() for value in (size, projection, rotation, translation))):
        raise ValueError("Invalid frame geometry metadata")
    K = projection[:, :3]
    if not np.allclose(projection[:, 3], 0., atol=1e-10, rtol=0):
        raise ValueError("Resized projection must be [K|0], not a world/extrinsic projection")
    if not np.allclose(K[2], [0., 0., 1.], atol=1e-7, rtol=0) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid resized camera intrinsics")
    try:
        ray = np.linalg.solve(K, np.r_[normalized_xy * size, 1.])
        if not np.isfinite(ray).all() or abs(ray[2]) < 1e-12:
            raise ValueError("Invalid camera ray")
        camera = ray * (depth / ray[2])
        native = np.linalg.solve(rotation, camera - translation)
    except np.linalg.LinAlgError as error:
        raise ValueError("Camera/native transform must be invertible") from error
    values = np.r_[hwl[[2, 1, 0]], native]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite decoded official-label XYZ")
    return {"values_lwhxyz": values, "camera_center_xyz": camera,
            "label_position_native_xyz": native, "normalized_center_xy": normalized_xy,
            "camera_z": float(depth), "dimensions_lwh": hwl[[2, 1, 0]]}


def package_predictions(output, records_by_id, frames_requested):
    """Write result/*.txt and a validated ZIP only for the complete official run."""
    output = Path(output)
    if frames_requested != 30:
        return {"diagnostic_only": True, "zip": None, "official_ten_sequence_format": False}
    if set(records_by_id) != set(OFFICIAL_IDS):
        raise ValueError("A full result ZIP requires exactly the ten official video IDs")
    texts = {}
    for clip_id in OFFICIAL_IDS:
        frames = records_by_id[clip_id]
        if (len(frames) != 30 or any(isinstance(item.get("frame_index"), bool) for item in frames)
                or [item.get("frame_index") for item in frames] != list(range(1, 31))):
            raise ValueError("Full result requires 30 ordered frames for {}".format(clip_id))
        texts[clip_id] = "\n".join(_format_row(index, np.asarray(item["values_lwhxyz"], dtype=np.float64))
                                   for index, item in enumerate(frames, 1)) + "\n"
        _validate_text(texts[clip_id], clip_id)
    result = output / "result"
    zip_path = output / "result.zip"
    temporary = output / "result.zip.tmp"
    if result.exists() or zip_path.exists() or temporary.exists():
        raise FileExistsError("Refusing to overwrite result files")
    result.mkdir()
    for clip_id in OFFICIAL_IDS:
        with (result / (clip_id + ".txt")).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(texts[clip_id])
    _write_zip(temporary, OFFICIAL_IDS, texts)
    temporary.replace(zip_path)
    return {"diagnostic_only": False, "zip": str(zip_path.resolve()),
            "zip_sha256": sha256(zip_path), "official_ten_sequence_format": True,
            "result_file_sha256": {clip: sha256(result / (clip + ".txt")) for clip in OFFICIAL_IDS}}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(_jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ("source-project", "checkpoint", "config", "data-root", "calibration-root", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--gpu", default="0", help="Physical CUDA GPU exposed as cuda:0")
    parser.add_argument("--frames", type=int, default=30, choices=range(1, 31),
                        help="Frames per clip; fewer than 30 prohibits result files/ZIP")
    parser.add_argument("--input-width", type=int, help="Must agree with any recorded training input size")
    parser.add_argument("--input-height", type=int)
    args = parser.parse_args(argv)
    if (args.input_width is None) != (args.input_height is None):
        parser.error("Provide both --input-width and --input-height")
    return args


def load_v2_runtime(source_project, checkpoint_path, config_path, gpu="0", explicit_size_wh=None):
    """Shared strict v2 loader; imports CUDA model code only when explicitly called."""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false")
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    import torch
    import yaml
    from lib.helpers.model_helper import build_model
    config_path, checkpoint_path = Path(config_path).resolve(), Path(checkpoint_path).resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("The inherited model requires CUDA")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True, mmap=True)
    manifests = []
    manifest_paths = sorted({config_path.parent / "run_manifest.json", checkpoint_path.parent / "run_manifest.json"})
    for path in manifest_paths:
        if path.is_file():
            manifests.append(json.loads(path.read_text(encoding="utf-8-sig")))
    contract = validate_contract(config, checkpoint, manifests, explicit_size_wh)
    width, height = contract["input_size_wh"]
    config["model"].update(input_size_hw=[height, width], target_encoding=ENCODING, device="cpu",
                           text_encoder_path=str(Path(source_project).resolve() / "roberta-base"),
                           text_encoder_from_config=True, swin_use_pretrained_backbone=False)
    config["model"].pop("pretrained_backbone_path", None)
    state = checkpoint.get("model_state")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint lacks a complete model_state mapping")
    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    if not any(key.startswith("official_geometry_head.") for key in state):
        raise ValueError("Checkpoint lacks the trained official_geometry_head state")
    seed = contract["seed_per_video"]
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    model, criterion = build_model(config["model"])
    del criterion
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    torch.cuda.manual_seed_all(seed)
    provenance = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
                  "checkpoint_epoch": checkpoint.get("epoch"), "state_dict_load": "strict_success",
                  "state_dict_keys": len(state), "config": str(config_path), "config_sha256": sha256(config_path),
                  "run_manifest_paths": [str(path) for path in manifest_paths if path.is_file()],
                  "model_options": contract["model_options"],
                  "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0)}
    return model, contract, provenance


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Output directory already exists: {}".format(output))
    explicit = None if args.input_width is None else (args.input_width, args.input_height)
    model, contract, provenance = load_v2_runtime(
        args.source_project, args.checkpoint, args.config, args.gpu, explicit)
    import torch
    from lib.datasets.mono3drefer.official_vlsot_dataset import OfficialVLSOTDataset
    from lib.helpers.trainer_helper import reset_temporal_state
    project = Path(__file__).resolve().parents[1]
    width, height = contract["input_size_wh"]
    seed = contract["seed_per_video"]
    dataset = OfficialVLSOTDataset(args.data_root, args.calibration_root, split="test",
                                   input_size_wh=(width, height), sample_mode="frame")
    ids = [clip["video_id"] for clip in dataset.clips]
    if not ids or not set(ids) <= set(OFFICIAL_IDS):
        raise ValueError("Official TEST inputs contain unexpected video IDs")
    if args.frames == 30 and set(ids) != set(OFFICIAL_IDS):
        raise ValueError("Full inference requires exactly the ten official TEST clips")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "inference_manifest.json"
    manifest = {"schema_version": 2, "inference_version": INFERENCE_VERSION, "status": "running",
                "target_encoding": ENCODING, "label_encoding": ENCODING, "coordinate_note": COORDINATE_NOTE,
                "label_convention_verified": False, "ground_truth_used_for_predictions": False,
                "world_transform_applied": False, "height_offset_applied": False,
                "input_size_wh": [width, height], "seed_per_video": seed,
                "temporal_mode": "reset memory/phase per clip; carry memory between ordered frames",
                "frames_requested_per_clip": args.frames, "video_ids": ids,
                "calibration_root": str(Path(args.calibration_root).resolve()),
                "calibration_note": "Matching-image source calibration; not supplied by the official competition ZIP",
                "code_sha256": {name: sha256(project / name) for name in (
                    "tools/infer_official_v2.py", "lib/losses/official_geometry.py",
                    "lib/datasets/mono3drefer/official_vlsot_dataset.py", "lib/models/mono3dvg/mono3dvg.py")},
                "diagnostic_only": args.frames != 30, "platform_submission_ready": False, "clips": []}
    manifest.update(provenance)
    _write_json(manifest_path, manifest)
    all_records = {}
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for clip_index, clip in enumerate(dataset.clips):
                reset_temporal_state(model)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                torch.cuda.reset_peak_memory_stats()
                clip_start = time.perf_counter()
                clip_dir = output / clip["video_id"]
                clip_dir.mkdir()
                records, raw_arrays = [], {}
                for index in range(args.frames):
                    frame = dataset.get_frame(clip_index, index)
                    if "target" in frame:
                        raise ValueError("Official TEST inference cannot receive target annotations")
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    prediction = model(torch.from_numpy(frame["images"]).unsqueeze(0).cuda(),
                                       torch.from_numpy(frame["calibs"]).unsqueeze(0).cuda(),
                                       torch.from_numpy(frame["image_sizes"]).unsqueeze(0).cuda(), [frame["text"]])
                    torch.cuda.synchronize()
                    heads = {key: prediction[key].detach().float().cpu().numpy()
                             for key in ("pred_boxes", "pred_depth", "pred_3d_dim")}
                    heads["target_encoding"] = prediction.get("target_encoding")
                    decoded = decode_prediction(heads, frame)
                    record = {key: frame[key] for key in ("frame_index", "image_name", "image_sizes",
                              "source_image_sizes", "source_K", "source_from_input", "calibs",
                              "lidar_to_camera_R", "lidar_to_camera_t")}
                    record.update(decoded)
                    record["forward_seconds"] = time.perf_counter() - before
                    for key in ("pred_boxes", "pred_depth", "pred_3d_dim"):
                        record[key] = heads[key]
                        raw_arrays["frame_{:02d}_{}".format(index + 1, key)] = heads[key]
                    records.append(_jsonable(record))
                np.savez_compressed(clip_dir / "raw_outputs.npz", **raw_arrays)
                diagnostic = {"video_id": clip["video_id"], "sequence_id": clip["sequence_id"],
                              "input_text": clip["text"], "target_encoding": ENCODING,
                              "label_encoding": ENCODING, "coordinate_note": COORDINATE_NOTE,
                              "checkpoint": provenance["checkpoint"], "checkpoint_epoch": provenance["checkpoint_epoch"],
                              "state_dict_load": "strict_success", "frames": records,
                              "seed_per_video": seed, "elapsed_seconds": time.perf_counter() - clip_start,
                              "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / (1024 ** 2)}
                _write_json(clip_dir / "diagnostics.json", diagnostic)
                all_records[clip["video_id"]] = records
                summary = {key: diagnostic[key] for key in ("video_id", "elapsed_seconds", "peak_cuda_memory_mib")}
                manifest["clips"].append(summary)
                _write_json(manifest_path, manifest)
                print("CLIP_COMPLETE " + json.dumps(summary), flush=True)
        manifest.update(package_predictions(output, all_records, args.frames))
        manifest.update(status="diagnostic_complete" if args.frames != 30 else "complete_format_valid_quality_unverified",
                        elapsed_seconds=time.perf_counter() - started)
        _write_json(manifest_path, manifest)
    except Exception as error:
        manifest.update(status="failed", error="{}: {}".format(type(error).__name__, error),
                        elapsed_seconds=time.perf_counter() - started)
        _write_json(manifest_path, manifest)
        raise
    print(json.dumps({"output": str(output), "status": manifest["status"], "zip": manifest.get("zip"),
                      "label_encoding": ENCODING, "label_convention_verified": False}, ensure_ascii=False), flush=True)
    return manifest


if __name__ == "__main__":
    run(parse_args())
