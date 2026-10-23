"""Export the official camera-geometry v2 head, then require recurrent ORT parity.

This is a separate interface from export_onnx.py's legacy five-head graph.
Batch=1, image width=960/height=544, text length=110 are intentionally fixed.
The eight inputs are pixels, token IDs/mask, memory/phase and three RNG draws.
Camera calibration is required by host-side native-coordinate decoding, but is
not a neural graph input: the v2 forward does not consume calibs or img_sizes.
The non-tensor target_encoding metadata stays in the manifest, never ONNX.

Example:
  python tools/export_official_v2_onnx.py export --config TRAIN_RUN/resolved_config.yaml \
    --source-project SOURCE --checkpoint TRAIN_RUN/checkpoint_best.pth \
    --data-root OFFICIAL --partition test --clip-id 000237 --output-dir NEW_EXPORT --gpu 0

Only successful checker, CUDA/reference comparison and two ORT steps produce a
validated manifest. Export success alone does not establish competition accuracy
or verify the experimental official-label geometric-center interpretation.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import types
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from tools import export_onnx as legacy

ENCODING = "camera_geometric_center_v2"
WIDTH, HEIGHT, TEXT_LENGTH = 960, 544, 110
HEADS = ("pred_logits", "pred_boxes", "pred_3d_dim", "pred_depth")
INPUTS = ("images", "input_ids", "attention_mask", "memory_tokens", "memory_phase",
          "read_noise", "write_noise", "memory_noise")
OUTPUTS = HEADS + ("next_memory_tokens", "next_memory_phase")


def read_public_clip(root, partition, clip_id):
    """Read only language and image names; TRAIN numeric tails are never used."""
    if partition not in ("train", "test") or not re.fullmatch(r"[0-9]{6}", clip_id):
        raise ValueError("Expected train/test partition and a literal six-digit clip ID")
    directory = Path(root).resolve() / partition
    source = directory / "json" / (clip_id + ".json")
    value = json.loads(source.read_text(encoding="utf-8-sig"))
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Expected one clip record")
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError("Clip JSON must contain an object")
    declared = value.get("videoID")
    if (isinstance(declared, bool) or not isinstance(declared, (int, str))
            or not str(declared).isdigit() or str(declared).zfill(6) != clip_id):
        raise ValueError("Public videoID disagrees with the selected filename")
    text = value.get("description")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nonempty public description is required")
    frames = []
    for index in range(2):
        item = value.get("frame{}".format(index))
        if partition == "train":
            if not isinstance(item, list) or not item:
                raise ValueError("TRAIN frame must begin with an input image path")
            item = item[0]
        if not isinstance(item, str) or not item.strip():
            raise ValueError("Expected public input image path")
        name = item.replace("\\", "/").rsplit("/", 1)[-1]
        if name in ("", ".", ".."):
            raise ValueError("Invalid image basename")
        image = directory / "img" / name
        if not image.is_file():
            raise FileNotFoundError(str(image))
        frames.append(image)
    if frames[0] == frames[1]:
        raise ValueError("Two different consecutive images are needed for parity")
    return text.lower(), frames


def prepare_image(path, np, torch, device):
    """Match official_vlsot_dataset.py's direct resize and ImageNet normalization."""
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.
    mean = np.asarray([.485, .456, .406], dtype=np.float32)[:, None, None]
    std = np.asarray([.229, .224, .225], dtype=np.float32)[:, None, None]
    return torch.from_numpy(np.ascontiguousarray((pixels - mean) / std)).unsqueeze(0).to(device)


def make_wrapper(model, torch):
    """Bind the same independently checked explicit TTM step as legacy export."""
    if getattr(model, "target_encoding", None) != ENCODING:
        raise ValueError("Only an official camera-geometry v2 model may use this wrapper")
    adapter = model.mono3dvg_transformer.TextGuidedAdapter
    if not getattr(adapter, "dynamic_shapes", False):
        raise ValueError("960x544 requires TextGuidedAdapter.dynamic_shapes=True")
    ttm = model.mono3dvg_transformer.decoder.TTM_model
    legacy.check_ttm_configuration(ttm)

    def explicit_forward(this, inputs, memory_tokens):
        output, updated, phase = legacy.tensor_ttm_step(
            this, inputs, memory_tokens, this._export_phase, this._export_read_noise,
            this._export_write_noise, this._export_memory_noise, torch)
        this._export_next_phase = phase
        return output, updated

    ttm.forward = types.MethodType(explicit_forward, ttm)

    class StateExplicitV2(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.model = wrapped

        def forward(self, images, input_ids, attention_mask, memory_tokens,
                    memory_phase, read_noise, write_noise, memory_noise):
            decoder = self.model.mono3dvg_transformer.decoder
            ttm = decoder.TTM_model
            decoder.memory_tokens = memory_tokens
            ttm._export_phase, ttm._export_read_noise = memory_phase, read_noise
            ttm._export_write_noise, ttm._export_memory_noise = write_noise, memory_noise
            result = self.model(images, None, None, {
                "input_ids": input_ids, "attention_mask": attention_mask,
            })
            return tuple(result[key] for key in HEADS) + (decoder.memory_tokens, ttm._export_next_phase)

    return StateExplicitV2(model).eval()


def compare_outputs(actual, expected, np, rtol, atol):
    if len(actual) != len(OUTPUTS) or len(expected) != len(OUTPUTS):
        raise ValueError("Expected exactly {} v2 output tensors".format(len(OUTPUTS)))
    # Reuse the numerical comparator with an explicit process-local interface.
    # The legacy module and its CLI retain their original interface on exit.
    with patch.object(legacy, "OUTPUTS", OUTPUTS):
        return legacy.compare_outputs(actual, expected, np, rtol, atol)


def standard_onnx_report(path, onnx):
    with patch.object(legacy, "INPUTS", INPUTS), patch.object(legacy, "OUTPUTS", OUTPUTS):
        return legacy.standard_onnx_report(path, onnx)


def validate_checkpoint_contract(config, checkpoint, run_manifests=()):
    """Require recorded mask semantics/resolution before building a CUDA model."""
    from tools.infer_official_v2 import validate_contract
    contract = validate_contract(config, checkpoint, run_manifests)
    if tuple(contract["input_size_wh"]) != (WIDTH, HEIGHT):
        raise ValueError("This fixed v2 export supports only trained input width=960,height=544")
    return contract


def export_model(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    missing = [name for name, present in legacy.dependencies().items() if not present]
    if missing:
        raise RuntimeError("Missing export/verification dependencies: {}".format(missing))
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import yaml
    from lib.helpers.model_helper import build_model

    if not torch.cuda.is_available():
        raise RuntimeError("The inherited TTM/attention model requires CUDA to export")
    if args.opset < 16:
        raise ValueError("GridSample requires ONNX opset >= 16")
    description, frame_paths = read_public_clip(args.data_root, args.partition, args.clip_id)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "export_manifest.json"
    manifest = {
        "status": "started", "target_encoding": ENCODING, "onnx_runtime_verified": False,
        "platform_prediction_coordinates_verified": False, "label_convention_verified": False,
        "fixed_batch": 1, "fixed_image_wh": [WIDTH, HEIGHT], "fixed_text_length": TEXT_LENGTH,
        "inputs": list(INPUTS), "outputs": list(OUTPUTS), "opset": args.opset,
        "seed": args.seed, "example_partition": args.partition, "example_clip_id": args.clip_id,
        "example_images": [str(path) for path in frame_paths],
        "ground_truth_used_for_graph_or_parity": False,
        "host_calibration_note": "Matching-frame K and virtual-LiDAR-to-camera R/t are required by external v2 native XYZ decoding; the neural graph does not consume calibration or image-size tensors.",
        "raw_head_meaning": {
            "pred_boxes": "[1,1,6]: first two are unbounded normalized projected camera center; last four are placeholders, not LRTB extents",
            "pred_depth": "[1,1,2]: first value is positive camera Z in meters; second is a placeholder",
            "pred_3d_dim": "[1,1,3]: positive H,W,L in meters",
            "pred_logits": "Inherited unsupervised class head; not used to select the single referred object",
        },
    }
    start = time.perf_counter()
    try:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if cfg["model"].get("dec_layers") != 1 or cfg["model"].get("num_queries") != 1:
            raise ValueError("State adapter requires dec_layers=1 and num_queries=1")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        run_manifests = []
        for path in sorted({Path(args.config).resolve().parent / "run_manifest.json",
                            Path(args.checkpoint).resolve().parent / "run_manifest.json"}):
            if path.is_file():
                run_manifests.append(json.loads(path.read_text(encoding="utf-8-sig")))
        contract = validate_checkpoint_contract(cfg, checkpoint, run_manifests)
        manifest["model_options"] = contract["model_options"]
        # Keep the recorded mask flag unchanged. Loading the same parameter
        # tensors under another masking behavior is not checkpoint parity.
        cfg["model"].update(text_encoder_path=str(Path(args.source_project).resolve() / "roberta-base"),
                             text_encoder_from_config=True, swin_use_pretrained_backbone=False, device="cpu")
        cfg["model"].pop("pretrained_backbone_path", None)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_num_threads(args.threads)
        model, criterion = build_model(cfg["model"])
        del criterion
        state = checkpoint["model_state"]
        state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
        if not any(key.startswith("official_geometry_head.") for key in state):
            raise ValueError("Checkpoint has no v2 geometry head")
        model.load_state_dict(state, strict=True)
        manifest.update(state_dict_load="strict_success", state_dict_keys=len(state),
                        checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_epoch=checkpoint.get("epoch"),
                        torch=str(torch.__version__), onnx=onnx.__version__, onnxruntime=ort.__version__)
        del checkpoint, state
        device = torch.device("cuda:0")
        model.to(device).eval()
        for module in model.modules():
            if hasattr(module, "fused_attn"):
                module.fused_attn = False
        with legacy.export_math_context(torch) as contexts:
            manifest["rank_safe_maxpool_reference_checks"] = legacy.install_rank_safe_pooling(model, torch, np, device)
            manifest["ttm_original_reference_checks"] = legacy.verify_ttm_reference(
                model.mono3dvg_transformer.decoder.TTM_model, torch, np, device, args.rtol, args.atol)
            wrapper = make_wrapper(model, torch)
            tokens = model.tokenizer.batch_encode_plus([description], max_length=TEXT_LENGTH,
                                                       padding="max_length", truncation=True, return_tensors="pt")
            ids, mask = tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
            rng = np.random.RandomState(args.seed)

            def noise(shape):
                return torch.from_numpy(rng.standard_normal(shape).astype("float32")).to(device)

            def inputs_for(path, memory, phase):
                return (prepare_image(path, np, torch, device), ids, mask, memory, phase,
                        noise((1, 17, 256)), noise((1, 50, 256)), noise((1, 128, 256)))

            initial = inputs_for(frame_paths[0], torch.zeros((1, 128, 256), device=device),
                                 torch.zeros((1,), dtype=torch.int64, device=device))
            legacy.enable_reference_attention(model, False)
            cuda_outputs = legacy.tensor_outputs(wrapper, initial, torch)
            manifest["autocast_disabled_for_export"] = bool(args.disable_autocast)
            if args.disable_autocast:
                amp_outputs = cuda_outputs
                contexts.enter_context(legacy.disabled_autocast_context(torch))
                cuda_outputs = legacy.tensor_outputs(wrapper, initial, torch)
                manifest["native_amp_vs_fp32"] = compare_outputs(cuda_outputs, amp_outputs, np, args.rtol, args.atol)
            manifest["deformable_attention_module_count"] = legacy.enable_reference_attention(model, True)
            expected_initial = legacy.tensor_outputs(wrapper, initial, torch)
            manifest["cuda_vs_grid_reference"] = compare_outputs(expected_initial, cuda_outputs, np, args.rtol, args.atol)
            print("TTM reference, rank-safe pooling and CUDA/GridSample parity passed", flush=True)
            model_path = output / "model_official_v2.onnx"
            with torch.no_grad():
                torch.onnx.export(wrapper, initial, str(model_path), input_names=list(INPUTS),
                                  output_names=list(OUTPUTS), opset_version=args.opset,
                                  do_constant_folding=True, export_params=True,
                                  training=torch.onnx.TrainingMode.EVAL,
                                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX)
            manifest["graph"] = standard_onnx_report(model_path, onnx)
            manifest["model_bytes"] = model_path.stat().st_size
            session = legacy.create_session(model_path, ort, args.ort_provider, args.threads)
            manifest["ort_provider"] = session.get_providers()
            actual_memory = np.zeros((1, 128, 256), dtype=np.float32)
            actual_phase = np.zeros((1,), dtype=np.int64)
            expected_memory, expected_phase = expected_initial[-2:]
            comparisons = []
            for step in range(2):
                if step == 0:
                    tensors, expected = initial, expected_initial
                else:
                    tensors = inputs_for(frame_paths[step], torch.from_numpy(expected_memory).to(device),
                                         torch.from_numpy(expected_phase).to(device))
                    expected = legacy.tensor_outputs(wrapper, tensors, torch)
                feed = {name: tensor.detach().cpu().numpy() for name, tensor in zip(INPUTS, tensors)}
                feed["memory_tokens"], feed["memory_phase"] = actual_memory, actual_phase
                np.savez_compressed(output / "verification_step{}_inputs.npz".format(step), **feed)
                actual = session.run(list(OUTPUTS), feed)
                comparisons.append(compare_outputs(actual, expected, np, args.rtol, args.atol))
                np.savez_compressed(output / "verification_step{}_outputs.npz".format(step), **dict(zip(OUTPUTS, actual)))
                actual_memory, actual_phase = actual[-2:]
                expected_memory, expected_phase = expected[-2:]
                print("ORT/Torch recurrent v2 step {} parity passed".format(step), flush=True)
            manifest.update(status="validated", onnx_runtime_verified=True, two_step_parity=comparisons,
                            parity_rtol=args.rtol, parity_atol=args.atol,
                            state_protocol={"new_video": "memory=zeros[1,128,256], phase=zeros[1] int64",
                                            "next_frame": "feed next_memory_tokens and next_memory_phase back",
                                            "noise": "NumPy RandomState(seed), standard_normal read/write/memory order each frame",
                                            "read_write_noise_scale_inside_graph": .02,
                                            "memory_noise_scale_inside_graph": .2,
                                            "tokenizer": str(Path(args.source_project).resolve() / "roberta-base"),
                                            "text": "lowercase; max_length=110, padding=max_length, truncation=True"})
    except Exception as error:
        manifest.update(status="failed", error="{}: {}".format(type(error).__name__, error))
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - start
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "model": str(output / "model_official_v2.onnx"),
                      "manifest": str(manifest_path)}, indent=2), flush=True)


def parser():
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("check-dependencies")
    export = commands.add_parser("export")
    for name in ("config", "source-project", "checkpoint", "data-root", "output-dir", "clip-id"):
        export.add_argument("--" + name, required=True)
    export.add_argument("--partition", choices=("train", "test"), default="test")
    export.add_argument("--gpu", default="0")
    export.add_argument("--opset", type=int, default=17)
    export.add_argument("--seed", type=int, default=444)
    export.add_argument("--threads", type=int, default=4)
    export.add_argument("--ort-provider", default="CPUExecutionProvider")
    export.add_argument("--rtol", type=float, default=3e-3)
    export.add_argument("--atol", type=float, default=3e-4)
    export.add_argument("--disable-autocast", action="store_true")
    run = commands.add_parser("run", help="Torch-free inference from one saved tensor fixture")
    for name in ("model", "input-npz", "output-npz"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--ort-provider", default="CPUExecutionProvider")
    run.add_argument("--threads", type=int, default=4)
    return root


def main():
    args = parser().parse_args()
    if args.command == "check-dependencies":
        print(json.dumps(legacy.dependencies(), indent=2))
    elif args.command == "export":
        export_model(args)
    else:
        legacy.run_onnx(args)


if __name__ == "__main__":
    main()
