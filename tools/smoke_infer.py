"""Run existing task weights on a few frames, without using target annotations.

Raw model outputs are diagnostic artifacts, NOT platform-ready 3D predictions.
The inherited label coordinate convention must be established before exporting.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-project", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence-json", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--calib-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", default="1", help="Physical GPU index, exposed as cuda:0")
    parser.add_argument("--frames", type=int, default=1, choices=range(1, 31))
    return parser.parse_args()


def read_sequence(path):
    """Read public text/frame paths only; target boxes and labels are not returned."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(document, list):
        if not document:
            raise ValueError("Sequence JSON is empty")
        document = document[0]
    text = document.get("description")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Expected a nonempty language description")
    frames = []
    for index in range(30):
        frame = document.get("frame{}".format(index))
        if not isinstance(frame, list) or len(frame) < 3:
            raise ValueError("Missing frame paths at frame {}".format(index))
        # The remaining numeric fields contain labels. Do not expose them to
        # inference or use them as fallback predictions.
        basename = lambda value: str(value).replace("\\", "/").rsplit("/", 1)[-1]
        frames.append({
            "frame_index": index + 1,
            "image_name": basename(frame[2]),
            "calibration_name": basename(frame[1]),
        })
    video_id = str(document.get("videoID", Path(path).stem))
    filename_id = Path(path).stem
    if filename_id.isdigit() and video_id.isdigit() and int(filename_id) == int(video_id):
        video_id = filename_id  # Preserve leading zeros from the actual file.
    return video_id, text.lower(), frames


def load_projection(path, np):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if "P" in document:
        projection = np.asarray(document["P"], dtype=np.float32).reshape(3, 4)
    elif "cam_K" in document:
        projection = np.column_stack([
            np.asarray(document["cam_K"], dtype=np.float32).reshape(3, 3),
            np.zeros(3, dtype=np.float32),
        ])
    else:
        raise ValueError("Calibration has neither P nor cam_K: {}".format(path))
    if not np.isfinite(projection).all():
        raise ValueError("Calibration contains nonfinite values")
    return projection


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))

    import numpy as np
    import torch
    import yaml
    from PIL import Image
    from lib.datasets.mono3drefer.mono3drefer_utils import get_affine_transform
    from lib.helpers.model_helper import build_model

    if not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    output_dir = Path(args.output).resolve()
    # Refuse overwrites so a failed reproduction cannot replace previous results.
    output_dir.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source = Path(args.source_project).resolve()
    cfg["model"]["text_encoder_path"] = str(source / "roberta-base")
    cfg["model"]["text_encoder_from_config"] = True
    # The full checkpoint must provide every backbone parameter. Initialization
    # does not need another model.safetensors file or an upstream download.
    cfg["model"]["swin_use_pretrained_backbone"] = False
    cfg["model"].pop("pretrained_backbone_path", None)
    cfg["model"]["device"] = "cpu"
    torch.manual_seed(int(cfg.get("random_seed", 444)))
    torch.set_num_threads(4)
    np.random.seed(int(cfg.get("random_seed", 444)))

    started = time.perf_counter()
    model, criterion = build_model(cfg["model"])
    del criterion
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    state = checkpoint.get("model_state", checkpoint)
    state = {key.removeprefix("module.") if hasattr(key, "removeprefix") else
             (key[7:] if key.startswith("module.") else key): value
             for key, value in state.items()}
    # A partially initialized head could produce plausible-looking garbage.
    # Exact task-weight compatibility is required for this reproduction.
    model.load_state_dict(state, strict=True)
    model.to("cuda:0").eval()
    torch.cuda.reset_peak_memory_stats()
    print("Full checkpoint loaded strictly; model ready", flush=True)

    video_id, description, frames = read_sequence(args.sequence_json)
    summary = {
        "status": "diagnostic_only",
        "platform_submission_ready": False,
        "coordinate_system": "not_yet_established",
        "video_id": video_id,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_result_unverified": str(checkpoint.get("best_result")),
        "state_dict_keys": len(state),
        "state_dict_load": "strict_success",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "physical_gpu_index": args.gpu,
        "input_text": description,
        "frames": [],
    }
    arrays = {}
    resolution = np.asarray([1920, 1080], dtype=np.int32)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    with torch.inference_mode():
        for frame in frames[:args.frames]:
            image_path = Path(args.image_dir) / frame["image_name"]
            calib_path = Path(args.calib_dir) / frame["calibration_name"]
            projection = load_projection(calib_path, np)
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_size = np.asarray(image.size, dtype=np.float32)
                transform, inverse = get_affine_transform(
                    image_size / 2, image_size, 0, resolution, inv=1
                )
                # Match original validation preprocessing exactly.
                image = image.transform(
                    tuple(resolution.tolist()), Image.AFFINE,
                    tuple(inverse.reshape(-1).tolist()), Image.BILINEAR,
                )
                pixels = np.asarray(image, dtype=np.float32) / 255.0
            pixels = np.ascontiguousarray(((pixels - mean) / std).transpose(2, 0, 1))
            inputs = torch.from_numpy(pixels).unsqueeze(0).to("cuda:0")
            calibs = torch.from_numpy(projection).unsqueeze(0).to("cuda:0")
            sizes = torch.from_numpy(image_size).unsqueeze(0).to("cuda:0")
            torch.cuda.synchronize()
            before = time.perf_counter()
            prediction = model(inputs, calibs, sizes, [description])
            torch.cuda.synchronize()
            frame_record = dict(frame)
            frame_record.update({
                "image_path": str(image_path),
                "source_size_wh": image_size.tolist(),
                "input_size_wh": resolution.tolist(),
                "source_from_input": inverse.tolist(),
                "projection_matrix": projection.tolist(),
                "forward_seconds": time.perf_counter() - before,
            })
            for key in ["pred_logits", "pred_boxes", "pred_3d_dim", "pred_depth", "pred_angle"]:
                tensor = prediction[key].detach().float().cpu().numpy()
                if not np.isfinite(tensor).all():
                    raise RuntimeError("Nonfinite model output: {}".format(key))
                arrays["frame_{:02d}_{}".format(frame["frame_index"], key)] = tensor
                frame_record[key] = tensor.tolist()
            summary["frames"].append(frame_record)
            print("Frame {} forward completed in {:.2f}s".format(
                frame["frame_index"], frame_record["forward_seconds"]), flush=True)
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["peak_cuda_memory_mib"] = torch.cuda.max_memory_allocated() / 1024 ** 2
    summary["raw_dimensions_all_positive"] = bool(all(
        (value > 0).all() for key, value in arrays.items() if key.endswith("pred_3d_dim")
    ))
    summary["dimension_check_note"] = (
        "Current meanshape=False uses zero class means; positive h,w,l are necessary. "
        "This check does not establish position coordinates or model accuracy."
    )
    np.savez_compressed(output_dir / "raw_outputs.npz", **arrays)
    (output_dir / "diagnostics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in [
        "status", "video_id", "state_dict_load", "elapsed_seconds", "peak_cuda_memory_mib",
        "raw_dimensions_all_positive"
    ]}, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
