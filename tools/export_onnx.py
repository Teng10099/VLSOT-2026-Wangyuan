"""Export an executable, state-explicit Mono3DVLT ONNX and verify two steps.

Export example (from the work project):
  python tools/export_onnx.py export --config configs/mono3dvg.yaml \
      --source-project /path/to/original/project --checkpoint /path/task.pth \
      --sequence-json /path/033447.json --image-dir /path/img \
      --calib-dir /path/calib --output-dir /path/new_export --gpu 1

Torch-free runtime example, using an input fixture saved by export:
  python tools/export_onnx.py run --model /path/new_export/model.onnx \
      --input-npz /path/new_export/verification_step0_inputs.npz \
      --output-npz /path/new_outputs.npz

The ONNX contains raw neural heads, not a claimed world-coordinate decoder.
Image preprocessing and the local RoBERTa tokenizer remain outside the graph.
Batch=1, image=1920x1080 and text length=110 are intentionally fixed. TTM memory,
phase and three standard-normal draws are explicit inputs: no frozen Python
counter, hidden recurrent state, unimplemented custom op, or traced RNG draw.
"""

import argparse
from contextlib import contextmanager, ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
from unittest.mock import patch


HEADS = ("pred_logits", "pred_boxes", "pred_3d_dim", "pred_depth", "pred_angle")
INPUTS = ("images", "calibs", "img_sizes", "input_ids", "attention_mask",
          "memory_tokens", "memory_phase", "read_noise", "write_noise", "memory_noise")
OUTPUTS = HEADS + ("next_memory_tokens", "next_memory_phase")


def dependencies():
    """Report availability without importing torch or starting CUDA."""
    return {name: importlib.util.find_spec(name) is not None
            for name in ("torch", "onnx", "onnxruntime", "yaml", "numpy", "PIL")}


def check_ttm_configuration(ttm):
    model = ttm.config["model"]
    required = {
        "preprocess_mode": "3d", "memory_mode": "TL", "process_unit": "transformer",
        "memory_tokens_size": 128, "num_blocks": 8, "dim": 256,
        "summerize_num_tokens": 1, "Read_use_positional_embedding": True,
        "Write_use_positional_embedding": True, "load_memory_add_noise": True,
        "load_memory_add_noise_mode": "normal",
    }
    mismatches = {key: (model.get(key), value) for key, value in required.items()
                  if model.get(key) != value}
    if mismatches:
        raise ValueError("TTM export adapter does not match this configuration: {}".format(mismatches))


def tensor_ttm_step(ttm, inputs, memory_tokens, phase, read_noise,
                    write_noise, memory_noise, torch):
    """Algebraic equivalent of the checked-in one-step TTM, with explicit RNG/state.

    The original Conv3d receives a four-dimensional unbatched tensor. Explicitly
    add/remove that batch dimension so ONNX Conv has the valid five-dimensional
    input while preserving the original B=1 computation.
    """
    value = ttm.pre1.relu(ttm.pre1.conv(inputs.unsqueeze(0))).squeeze(0)
    value = value.flatten(3).permute(0, 2, 3, 1).permute(0, 1, 3, 2)
    input_tokens = value[:, 0, :, :]
    memory_blocks = memory_tokens.reshape(1, 8, 16, 256)
    current_index = torch.remainder(phase, 8)
    previous_index = torch.remainder(current_index - 1, 8)
    following_index = torch.remainder(current_index + 1, 8)
    current = torch.index_select(memory_blocks, 1, current_index).squeeze(1)
    previous = torch.index_select(memory_blocks, 1, previous_index).squeeze(1)
    following = torch.index_select(memory_blocks, 1, following_index).squeeze(1)
    unit = ttm.tokenTuringMachineUnit
    position = read_noise * 0.02
    current_all = torch.cat((current, input_tokens), dim=1) + position
    previous_all = torch.cat((previous, input_tokens), dim=1) + position
    following_all = torch.cat((following, input_tokens), dim=1) + position
    current_all = unit.tokenLearner1(current_all)
    previous_all = unit.tokenLearner1(previous_all)
    following_all = unit.tokenLearner1(following_all)
    output = unit.tokenLearner3(torch.cat((current_all, previous_all, following_all), dim=1))
    for _ in range(unit.num_layers):
        output = unit.transformerBlock(output)
    memory_input = torch.cat((current, previous, following, input_tokens, output), dim=1)
    write_block = unit.tokenLearner2(memory_input + write_noise * 0.02)
    # Original ReadFromDNC increments current_flag before WriteToDNC chooses
    # its write location, so the following block is replaced, not current.
    write_mask = (torch.arange(8, device=phase.device) == following_index).reshape(1, 8, 1, 1)
    updated = torch.where(write_mask, write_block.unsqueeze(1), memory_blocks)
    updated = updated.reshape(1, 128, 256) + memory_noise * 0.2
    return output, updated, following_index


def verify_ttm_reference(ttm, torch, np, device, rtol, atol):
    """Inject the exact same draws into original TTM and tensor adapter.

    Checks a normal phase, an interior phase and ring-buffer wraparound. This
    comparison runs the original forward, not another copy of the adapter.
    """
    check_ttm_configuration(ttm)
    generator = np.random.RandomState(704)
    report = []
    original_flag = ttm.simpleDNC.current_flag
    try:
        for index in (0, 3, 7):
            def random_tensor(shape):
                return torch.from_numpy(generator.standard_normal(shape).astype("float32")).to(device)
            inputs = random_tensor((1, 1, 1, 256))
            memory = random_tensor((1, 128, 256))
            phase = torch.tensor([index], dtype=torch.int64, device=device)
            read_noise = random_tensor((1, 17, 256))
            write_noise = random_tensor((1, 50, 256))
            memory_noise = random_tensor((1, 128, 256))
            calls = {"position": 0, "memory": 0}

            def fixed_normal(tensor, mean=0.0, std=1.0, generator=None):
                noise = (read_noise, write_noise)[calls["position"]]
                calls["position"] += 1
                if tuple(tensor.shape) != tuple(noise.shape):
                    raise AssertionError("Original TTM positional-noise shape changed")
                with torch.no_grad():
                    tensor.copy_(noise * std + mean)
                return tensor

            def fixed_randn_like(tensor, *args, **kwargs):
                calls["memory"] += 1
                if tuple(tensor.shape) != tuple(memory_noise.shape):
                    raise AssertionError("Original TTM memory-noise shape changed")
                return memory_noise.clone()

            ttm.simpleDNC.current_flag = index
            with torch.no_grad(), patch("torch.nn.init.normal_", fixed_normal), \
                    patch("torch.randn_like", fixed_randn_like):
                expected_output, expected_memory = ttm(inputs, memory.clone())
            with torch.no_grad():
                actual_output, actual_memory, actual_phase = tensor_ttm_step(
                    ttm, inputs, memory, phase, read_noise, write_noise, memory_noise, torch)
            if calls != {"position": 2, "memory": 1}:
                raise AssertionError("Unexpected original TTM RNG calls: {}".format(calls))
            np.testing.assert_allclose(actual_output.cpu().numpy(), expected_output.cpu().numpy(),
                                       rtol=rtol, atol=atol)
            np.testing.assert_allclose(actual_memory.cpu().numpy(), expected_memory.cpu().numpy(),
                                       rtol=rtol, atol=atol)
            if int(actual_phase.item()) != (index + 1) % 8:
                raise AssertionError("TTM phase update mismatch")
            report.append({
                "phase": index, "passed": True,
                "output_max_abs_error": float((actual_output - expected_output).abs().max().item()),
                "memory_max_abs_error": float((actual_memory - expected_memory).abs().max().item()),
            })
    finally:
        ttm.simpleDNC.current_flag = original_flag
    return report


def make_wrapper(model, torch):
    """Bind the functional TTM only to this export process, not source defaults."""
    decoder = model.mono3dvg_transformer.decoder
    ttm = decoder.TTM_model
    check_ttm_configuration(ttm)

    def explicit_forward(this, inputs, memory_tokens):
        output, updated, phase = tensor_ttm_step(
            this, inputs, memory_tokens, this._export_phase, this._export_read_noise,
            this._export_write_noise, this._export_memory_noise, torch)
        this._export_next_phase = phase
        return output, updated

    ttm.forward = types.MethodType(explicit_forward, ttm)

    class StateExplicitModel(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.model = wrapped

        def forward(self, images, calibs, img_sizes, input_ids, attention_mask,
                    memory_tokens, memory_phase, read_noise, write_noise, memory_noise):
            decoder = self.model.mono3dvg_transformer.decoder
            ttm = decoder.TTM_model
            decoder.memory_tokens = memory_tokens
            ttm._export_phase = memory_phase
            ttm._export_read_noise = read_noise
            ttm._export_write_noise = write_noise
            ttm._export_memory_noise = memory_noise
            result = self.model(images, calibs, img_sizes, {
                "input_ids": input_ids, "attention_mask": attention_mask,
            })
            return tuple(result[key] for key in HEADS) + (
                decoder.memory_tokens, ttm._export_next_phase)

    return StateExplicitModel(model).eval()


def enable_reference_attention(model, enabled):
    count = 0
    for module in model.modules():
        if module.__class__.__name__ == "MSDeformAttn":
            module.onnx_export_mode = enabled
            count += 1
    if not count:
        raise ValueError("No MSDeformAttn modules found; refusing an unverified export")
    return count


def install_rank_safe_pooling(model, torch, np, device):
    """Preserve unbatched MaxPool2d math while exporting a valid NCHW node.

    TextGuidedAdapter pools [B,H,W] verification scores. PyTorch treats that as
    an unbatched [C,H,W] input, but the legacy ONNX exporter emits a three-rank
    MaxPool with two-dimensional attributes. Explicitly adding/removing N=1
    avoids that malformed graph. This changes only this export process.
    """
    module = model.mono3dvg_transformer.TextGuidedAdapter.downsample
    if not isinstance(module, torch.nn.MaxPool2d) or module.return_indices:
        raise ValueError("Unexpected verification-score pooling module")
    original_forward = module.forward

    def rank_safe_forward(this, inputs):
        unbatched = inputs.dim() == 3
        if unbatched:
            inputs = inputs.unsqueeze(0)
        result = torch.nn.functional.max_pool2d(
            inputs, kernel_size=this.kernel_size, stride=this.stride,
            padding=this.padding, dilation=this.dilation,
            ceil_mode=this.ceil_mode, return_indices=False)
        return result.squeeze(0) if unbatched else result

    reports = []
    with torch.no_grad():
        for shape in ((1, 5, 7), (1, 68, 120), (1, 34, 60)):
            # Deterministic positive and negative inputs also verify padding
            # semantics, and cover both odd and even spatial dimensions.
            size = int(np.prod(shape))
            values = torch.linspace(-1.0, 1.0, size, device=device).reshape(shape)
            expected = original_forward(values)
            actual = rank_safe_forward(module, values)
            np.testing.assert_array_equal(actual.cpu().numpy(), expected.cpu().numpy())
            reports.append({"input_shape": list(shape), "output_shape": list(actual.shape),
                            "max_abs_error": 0.0, "passed": True})
    module.forward = types.MethodType(rank_safe_forward, module)
    return reports


@contextmanager
def export_math_context(torch):
    """Avoid opaque native eval kernels and unnecessary gradient checkpoints."""
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)

    def direct_checkpoint(function, *args, **kwargs):
        for key in ("use_reentrant", "preserve_rng_state", "context_fn", "determinism_check", "debug"):
            kwargs.pop(key, None)
        return function(*args, **kwargs)

    try:
        with ExitStack() as contexts:
            contexts.enter_context(patch("torch.utils.checkpoint.checkpoint", direct_checkpoint))
            yield contexts
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


@contextmanager
def disabled_autocast_context(torch):
    """Optional process-local FP32 fallback when CPU ORT lacks FP16 kernels."""
    def disabled_autocast(*args, **kwargs):
        return torch.autocast(device_type="cuda", enabled=False)

    with ExitStack() as contexts:
        # Swin uses the amp module; depth transformer cached the class itself.
        contexts.enter_context(patch("torch.cuda.amp.autocast", disabled_autocast))
        contexts.enter_context(patch(
            "lib.models.mono3dvg.depth_predictor.transformer.autocast", disabled_autocast))
        yield


def prepare_image(image_path, calibration_path, device, torch, np):
    from PIL import Image
    from lib.datasets.mono3drefer.mono3drefer_utils import get_affine_transform
    from tools.smoke_infer import load_projection
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        size = np.asarray(image.size, dtype=np.float32)
        _, inverse = get_affine_transform(size / 2, size, 0, np.asarray([1920, 1080]), inv=1)
        image = image.transform((1920, 1080), Image.AFFINE,
                                tuple(inverse.reshape(-1).tolist()), Image.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], dtype=np.float32)
    return (
        torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))).unsqueeze(0).to(device),
        torch.from_numpy(load_projection(calibration_path, np)).unsqueeze(0).to(device),
        torch.from_numpy(size).unsqueeze(0).to(device),
    )


def tensor_outputs(wrapper, inputs, torch):
    with torch.no_grad():
        return tuple(value.detach().cpu().numpy() for value in wrapper(*inputs))


def compare_outputs(actual, expected, np, rtol, atol):
    comparisons = {}
    for name, value, reference in zip(OUTPUTS, actual, expected):
        if value.shape != reference.shape or not np.isfinite(value).all():
            raise ValueError("Invalid output {}: shape={} expected={}".format(
                name, value.shape, reference.shape))
        if name == "next_memory_phase":
            np.testing.assert_array_equal(value, reference)
        else:
            np.testing.assert_allclose(value, reference, rtol=rtol, atol=atol,
                                       err_msg="Output parity failed for {}".format(name))
        comparisons[name] = {"shape": list(value.shape),
                             "max_abs_error": float(np.max(np.abs(value - reference)))}
    return comparisons


def standard_onnx_report(path, onnx):
    """Require checker-valid standard-domain nodes, including nested graphs."""
    onnx.checker.check_model(str(path), full_check=False)
    model = onnx.load(str(path), load_external_data=False)
    counts = {}

    def inspect(graph):
        for node in graph.node:
            if node.domain not in ("", "ai.onnx"):
                raise ValueError("Unsupported custom-domain operator {}::{}".format(
                    node.domain, node.op_type))
            if node.op_type in ("ATen", "PythonOp", "MSDeformAttn", "MultiScaleDeformableAttention"):
                raise ValueError("Unimplemented operator leaked into ONNX: {}".format(node.op_type))
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    inspect(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph in attribute.graphs:
                        inspect(graph)

    inspect(model.graph)
    if counts.get("GridSample", 0) < 1:
        raise ValueError("Expected standard GridSample nodes were not exported")
    declared_inputs = [value.name for value in model.graph.input]
    declared_outputs = [value.name for value in model.graph.output]
    if declared_inputs != list(INPUTS) or declared_outputs != list(OUTPUTS):
        raise ValueError("ONNX tensor interface changed: {} / {}".format(
            declared_inputs, declared_outputs))
    return {"checker": "passed", "operator_domains": ["ai.onnx"],
            "node_counts": counts, "inputs": declared_inputs, "outputs": declared_outputs}


def create_session(path, ort, provider, threads):
    if provider not in ort.get_available_providers():
        raise ValueError("ORT provider {} unavailable; installed={}".format(
            provider, ort.get_available_providers()))
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    # Avoid optimizer peak memory when loading the large Swin/RoBERTa graph.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(path), sess_options=options, providers=[provider])


def export_model(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    available = dependencies()
    missing = [name for name, present in available.items() if not present]
    if missing:
        raise RuntimeError("Missing export/verification dependencies: {}".format(missing))
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import yaml
    from lib.helpers.model_helper import build_model
    from tools.smoke_infer import read_sequence

    if not torch.cuda.is_available():
        raise RuntimeError("The checked-in TTM builds CUDA tensors; export requires CUDA")
    if args.opset < 16:
        raise ValueError("GridSample export requires opset >= 16")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "export_manifest.json"
    manifest = {"status": "started", "onnx_runtime_verified": False,
                "platform_prediction_coordinates_verified": False,
                "opset": args.opset, "seed": args.seed, "fixed_batch": 1,
                "fixed_image_wh": [1920, 1080], "fixed_text_length": 110,
                "inputs": list(INPUTS), "outputs": list(OUTPUTS)}
    started = time.perf_counter()
    try:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if cfg["model"].get("dec_layers") != 1 or cfg["model"].get("num_queries") != 1:
            raise ValueError("State adapter currently requires dec_layers=1 and num_queries=1")
        cfg["model"]["text_encoder_path"] = str(Path(args.source_project).resolve() / "roberta-base")
        cfg["model"]["text_encoder_from_config"] = True
        cfg["model"]["swin_use_pretrained_backbone"] = False
        cfg["model"].pop("pretrained_backbone_path", None)
        cfg["model"]["device"] = "cpu"
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_num_threads(args.threads)
        model, criterion = build_model(cfg["model"])
        del criterion
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint.get("model_state", checkpoint)
        state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        manifest.update({"state_dict_load": "strict_success", "state_dict_keys": len(state),
                         "checkpoint": str(Path(args.checkpoint).resolve()),
                         "checkpoint_epoch": checkpoint.get("epoch"), "torch": torch.__version__,
                         "onnx": onnx.__version__, "onnxruntime": ort.__version__})
        del state, checkpoint
        device = torch.device("cuda:0")
        model.to(device).eval()
        # timm's fused SDPA is an implementation choice, not learned state.
        for module in model.modules():
            if hasattr(module, "fused_attn"):
                module.fused_attn = False
        with export_math_context(torch) as export_contexts:
            manifest["rank_safe_maxpool_reference_checks"] = install_rank_safe_pooling(
                model, torch, np, device)
            print("Unbatched MaxPool2d vs explicit NCHW reference: passed", flush=True)
            ttm = model.mono3dvg_transformer.decoder.TTM_model
            manifest["ttm_original_reference_checks"] = verify_ttm_reference(
                ttm, torch, np, device, args.rtol, args.atol)
            print("Original TTM vs explicit-state adapter: phases 0/3/7 passed", flush=True)
            wrapper = make_wrapper(model, torch)
            video_id, description, frames = read_sequence(args.sequence_json)
            manifest["example_video_id"] = video_id
            tokens = model.tokenizer.batch_encode_plus(
                [description], max_length=110, padding="max_length", truncation=True, return_tensors="pt")
            ids, mask = tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
            random = np.random.RandomState(args.seed)

            def noise(shape):
                return torch.from_numpy(random.standard_normal(shape).astype("float32")).to(device)

            def inputs_for(frame, memory, phase):
                image, calibration, size = prepare_image(
                    Path(args.image_dir) / frame["image_name"],
                    Path(args.calib_dir) / frame["calibration_name"], device, torch, np)
                return (image, calibration, size, ids, mask, memory, phase,
                        noise((1, 17, 256)), noise((1, 50, 256)), noise((1, 128, 256)))

            zero_memory = torch.zeros((1, 128, 256), dtype=torch.float32, device=device)
            zero_phase = torch.zeros((1,), dtype=torch.int64, device=device)
            first_inputs = inputs_for(frames[0], zero_memory, zero_phase)
            enable_reference_attention(model, False)
            cuda_outputs = tensor_outputs(wrapper, first_inputs, torch)
            manifest["autocast_disabled_for_export"] = args.disable_autocast
            if args.disable_autocast:
                amp_outputs = cuda_outputs
                export_contexts.enter_context(disabled_autocast_context(torch))
                cuda_outputs = tensor_outputs(wrapper, first_inputs, torch)
                manifest["native_amp_vs_fp32"] = compare_outputs(
                    cuda_outputs, amp_outputs, np, args.rtol, args.atol)
                print("Native AMP vs FP32 export precision parity: passed", flush=True)
            manifest["deformable_attention_module_count"] = enable_reference_attention(model, True)
            reference_outputs = tensor_outputs(wrapper, first_inputs, torch)
            manifest["cuda_vs_grid_reference"] = compare_outputs(
                reference_outputs, cuda_outputs, np, args.rtol, args.atol)
            print("CUDA deformable attention vs GridSample reference: passed", flush=True)
            model_path = output / "model.onnx"
            with torch.no_grad():
                torch.onnx.export(
                    wrapper, first_inputs, str(model_path), input_names=list(INPUTS),
                    output_names=list(OUTPUTS), opset_version=args.opset,
                    do_constant_folding=True, export_params=True,
                    training=torch.onnx.TrainingMode.EVAL,
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                )
            manifest["graph"] = standard_onnx_report(model_path, onnx)
            manifest["model_bytes"] = model_path.stat().st_size
            print("ONNX checker and standard-operator audit: passed", flush=True)
            session = create_session(model_path, ort, args.ort_provider, args.threads)
            manifest["ort_provider"] = session.get_providers()
            reference_memory, reference_phase = reference_outputs[-2:]
            actual_memory = np.zeros((1, 128, 256), dtype=np.float32)
            actual_phase = np.zeros((1,), dtype=np.int64)
            checks = []
            for step in range(2):
                if step == 0:
                    tensors = first_inputs
                    expected = reference_outputs
                else:
                    tensors = inputs_for(frames[1], torch.from_numpy(reference_memory).to(device),
                                         torch.from_numpy(reference_phase).to(device))
                    expected = tensor_outputs(wrapper, tensors, torch)
                feed = {name: tensor.detach().cpu().numpy() for name, tensor in zip(INPUTS, tensors)}
                # Feed ORT's own previous outputs back, so validation covers
                # real recurrence rather than always resetting to Torch state.
                feed["memory_tokens"] = actual_memory
                feed["memory_phase"] = actual_phase
                np.savez_compressed(output / "verification_step{}_inputs.npz".format(step), **feed)
                actual = session.run(list(OUTPUTS), feed)
                np.savez_compressed(output / "verification_step{}_outputs.npz".format(step),
                                    **dict(zip(OUTPUTS, actual)))
                checks.append(compare_outputs(actual, expected, np, args.rtol, args.atol))
                actual_memory, actual_phase = actual[-2:]
                reference_memory, reference_phase = expected[-2:]
                print("ORT/Torch recurrent step {} parity: passed".format(step), flush=True)
            manifest["two_step_parity"] = checks
            manifest["parity_rtol"] = args.rtol
            manifest["parity_atol"] = args.atol
            manifest["onnx_runtime_verified"] = True
            manifest["status"] = "validated"
            manifest["state_protocol"] = {
                "new_video": "zero memory_tokens[1,128,256] and memory_phase[1]=0",
                "next_frame": "feed next_memory_tokens and next_memory_phase back as inputs",
                "noise": "NumPy RandomState(seed).standard_normal draws in read/write/memory order per frame",
                "read_noise_scale_inside_graph": 0.02,
                "write_noise_scale_inside_graph": 0.02,
                "memory_noise_scale_inside_graph": 0.2,
                "tokenizer": str(Path(args.source_project).resolve() / "roberta-base"),
                "text": "lowercase; RoBERTa max_length=110 padding=max_length truncation=True",
            }
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = "{}: {}".format(type(exc).__name__, exc)
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "validated", "model": str(output / "model.onnx"),
                      "manifest": str(manifest_path)}, indent=2), flush=True)


def run_onnx(args):
    """Run an exported graph without torch, source code, CUDA extension, or GT."""
    import numpy as np
    import onnxruntime as ort
    target = Path(args.output_npz)
    if target.exists():
        raise FileExistsError("Refusing to overwrite {}".format(target))
    session = create_session(Path(args.model), ort, args.ort_provider, args.threads)
    with np.load(args.input_npz, allow_pickle=False) as source:
        required = [item.name for item in session.get_inputs()]
        if set(source.files) != set(required):
            raise ValueError("Input NPZ must contain exactly {}".format(required))
        feed = {name: source[name] for name in required}
    result = session.run(None, feed)
    output_names = [item.name for item in session.get_outputs()]
    if any(not np.isfinite(value).all() for value in result):
        raise ValueError("ONNX runtime produced nonfinite outputs")
    with target.open("xb") as stream:
        np.savez_compressed(stream, **dict(zip(output_names, result)))
    print(json.dumps({"output": str(target), "providers": session.get_providers(),
                      "outputs": {name: list(value.shape) for name, value in zip(output_names, result)}}, indent=2))


def parser():
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("check-dependencies", help="report without importing torch or starting CUDA")
    export = commands.add_parser("export", help="export and require checker+ORT parity success")
    for name in ("config", "source-project", "checkpoint", "sequence-json", "image-dir", "calib-dir", "output-dir"):
        export.add_argument("--" + name, required=True)
    export.add_argument("--gpu", default="1")
    export.add_argument("--opset", type=int, default=17)
    export.add_argument("--seed", type=int, default=444)
    export.add_argument("--threads", type=int, default=4)
    export.add_argument("--ort-provider", default="CPUExecutionProvider")
    export.add_argument("--rtol", type=float, default=3e-3)
    export.add_argument("--atol", type=float, default=3e-4)
    export.add_argument("--disable-autocast", action="store_true",
                        help="FP32 fallback for unsupported CPU ORT FP16 kernels; increases GPU memory and checks AMP parity")
    run = commands.add_parser("run", help="Torch-free inference from a saved tensor NPZ")
    run.add_argument("--model", required=True)
    run.add_argument("--input-npz", required=True)
    run.add_argument("--output-npz", required=True)
    run.add_argument("--ort-provider", default="CPUExecutionProvider")
    run.add_argument("--threads", type=int, default=4)
    return root


def main():
    args = parser().parse_args()
    if args.command == "check-dependencies":
        print(json.dumps(dependencies(), indent=2))
    elif args.command == "export":
        export_model(args)
    else:
        run_onnx(args)


if __name__ == "__main__":
    main()
