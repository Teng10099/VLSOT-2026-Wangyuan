"""Controlled TRAIN-input perturbations; diagnostic only, no labels or scoring."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-project", "checkpoint", "config", "data-root", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--clip-ids", help="Two comma-separated TRAIN IDs; default selects different images/captions")
    a = p.parse_args()
    from tools.infer_official_v2 import load_v2_runtime
    from tools.export_official_v2_onnx import read_public_clip
    model, contract, provenance = load_v2_runtime(a.source_project, a.checkpoint, a.config, a.gpu)
    import numpy as np
    import torch
    from PIL import Image
    from lib.helpers.trainer_helper import reset_temporal_state
    output = Path(a.output).resolve()
    if output.exists():
        raise FileExistsError(str(output))
    ids = a.clip_ids.split(",") if a.clip_ids else [x.stem for x in sorted((Path(a.data_root) / "train/json").glob("*.json"))]
    chosen = []
    for clip_id in ids:
        text, frames = read_public_clip(a.data_root, "train", clip_id)
        if not chosen or (frames[0] != chosen[0][2] and text != chosen[0][1]):
            chosen.append((clip_id, text, frames[0]))
        if len(chosen) == 2:
            break
    if len(chosen) != 2:
        raise ValueError("Need two TRAIN clips with different first images and captions")
    width, height = contract["input_size_wh"]
    def pixels(path):
        with Image.open(path) as im:
            im = im.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
            x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.
        x = (x - np.asarray([.485,.456,.406], dtype=np.float32)[:,None,None]) / np.asarray([.229,.224,.225], dtype=np.float32)[:,None,None]
        return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).cuda()
    image_a, image_b = pixels(chosen[0][2]), pixels(chosen[1][2])
    decoder = model.mono3dvg_transformer.decoder
    snapshots = {}
    hooks = []
    def capture(name):
        def hook(module, inputs, result):
            values = list(result) if isinstance(result, (tuple, list)) else [result]
            for index, value in enumerate(values):
                if torch.is_tensor(value):
                    key = name if len(values) == 1 else name + "/" + str(index)
                    snapshots.setdefault(key, []).append(value.detach().float().cpu().numpy().copy())
        return hook
    for name, module in (("decoder_layer0", decoder.layers[0]), ("ttm_pre1", decoder.TTM_model.pre1),
                         ("ttm_output", decoder.TTM_model), ("ttm_tokenLearner3", decoder.TTM_model.tokenTuringMachineUnit.tokenLearner3)):
        hooks.append(module.register_forward_hook(capture(name)))
    cases = [("baseline", image_a, chosen[0][1]), ("repeat", image_a, chosen[0][1]),
             ("zero_image", torch.zeros_like(image_a), chosen[0][1]),
             ("other_image", image_b, chosen[0][1]), ("other_caption", image_a, chosen[1][1])]
    arrays, records = {}, {}
    try:
        with torch.inference_mode():
            for name, image, text in cases:
                reset_temporal_state(model)
                seed = contract["seed_per_video"]
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
                snapshots = {}
                prediction = model(image, None, None, [text])
                values = {"center": prediction["pred_boxes"][..., :2].detach().float().cpu().numpy(),
                          "depth": prediction["pred_depth"][..., :1].detach().float().cpu().numpy(),
                          "dims": prediction["pred_3d_dim"].detach().float().cpu().numpy()}
                for key, value in snapshots.items():
                    for index, tensor in enumerate(value):
                        values[key + "/call" + str(index)] = tensor
                arrays[name] = values
                records[name] = {key: {"shape": list(value.shape), "min": float(value.min()), "max": float(value.max()),
                                       "mean": float(value.mean()), "std": float(value.std()), "l2": float(np.linalg.norm(value))}
                                 for key, value in values.items()}
                print("CASE_COMPLETE " + name, flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        reset_temporal_state(model)
    differences = {name: {key: float(np.max(np.abs(value - arrays["baseline"][key])))
                           for key, value in values.items()} for name, values in arrays.items() if name != "baseline"}
    report = {"status": "diagnostic_complete", "ground_truth_used": False, "provenance": provenance,
              "cases": records, "max_abs_difference_from_baseline": differences,
              "clips": [{"video_id": item[0], "text": item[1], "image": str(item[2])} for item in chosen],
              "mask_flags": {name: bool(getattr(module, "correct_padding_mask", False)) for name, module in model.named_modules()
                             if module.__class__.__name__ == "MultiheadDiffAttn"},
              "control": "reset TTM and reset torch/cuda/numpy RNG before every case; zero_image means zero normalized tensor",
              "input_size_wh": [width, height]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "differences": differences}, indent=2), flush=True)


if __name__ == "__main__":
    main()
