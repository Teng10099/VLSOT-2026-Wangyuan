"""Fine-tune the explicit camera-geometry head on official TRAIN clips only.

Example bounded smoke (paths are deployment-specific):
  python train_official.py --config configs/mono3dvg.yaml --source-project SOURCE \
    --checkpoint TASK.pth --data-root OFFICIAL --calibration-root SOURCE/calib \
    --output RUN --label-center-reference geometric_center --gpu 0 \
    --epochs 2 --max-train-batches 1 --max-val-batches 1 --max-frames-per-clip 2

The full inherited model uses CUDA internally. CPU regression tests exercise
the loader/trainer with small dummy models, without loading this CUDA model.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import random
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ("config", "source-project", "checkpoint", "data-root", "calibration-root", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--label-center-reference", choices=("geometric_center",), required=True,
                        help="Explicit experimental TRAIN XYZ convention; not independently verified")
    parser.add_argument("--gpu", default="0", help="One physical CUDA GPU exposed as cuda:0")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument("--input-width", type=int, default=960)
    parser.add_argument("--input-height", type=int, default=544)
    parser.add_argument("--sample-mode", choices=("clip", "frame"), default="clip")
    parser.add_argument("--trainable", choices=("head", "decoder", "fusion", "all"), default="head")
    parser.add_argument("--head-lr", type=float, help="Optional separate learning rate for the new geometry head")
    parser.add_argument("--head-init-std", type=float, default=0.001)
    parser.add_argument("--temporal-residual", action="store_true",
                        help="Keep pre-TTM spatial features through an additive temporal residual")
    parser.add_argument("--train-ids", help="Comma-separated six-digit IDs or a newline-delimited file")
    parser.add_argument("--val-ids", help="Training-only internal validation IDs; never official TEST")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--no-validation", action="store_true", help="Train-only smoke; never reports a validation result")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-frames-per-clip", type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--correct-padding-mask", action="store_true",
                        help="Screen padded text tokens instead of legacy Boolean +1 behavior")
    parser.add_argument("--resume", help="Resume this entrypoint's complete training checkpoint")
    args = parser.parse_args(argv)
    for key in ("epochs", "input_width", "input_height"):
        if getattr(args, key) <= 0:
            parser.error("--{} must be positive".format(key.replace("_", "-")))
    for key in ("max_train_batches", "max_val_batches", "max_frames_per_clip"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            parser.error("--{} must be positive".format(key.replace("_", "-")))
    if args.max_frames_per_clip is not None and args.max_frames_per_clip > 30:
        parser.error("--max-frames-per-clip cannot exceed 30")
    if args.workers < 0 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("Invalid workers, learning rate, or weight decay")
    if args.head_init_std <= 0 or (args.head_lr is not None and args.head_lr <= 0):
        parser.error("Head initialization scale and learning rate must be positive")
    if (args.train_ids is None) != (args.val_ids is None) and not args.no_validation:
        parser.error("Supply both --train-ids and --val-ids, or allow automatic sequence splitting")
    if args.no_validation and args.val_ids:
        parser.error("--no-validation cannot consume --val-ids")
    return args


def _ids(value):
    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        return [item.strip() for item in path.read_text(encoding="utf-8-sig").splitlines() if item.strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def configure_trainable(model, mode):
    """Default head-only smoke; decoder mode includes the temporal adapter."""
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = (mode == "all" or name.startswith("official_geometry_head.")
                   or (mode == "decoder" and name.startswith("mono3dvg_transformer.decoder."))
                   or (mode == "fusion" and name.startswith(("mono3dvg_transformer.", "input_proj.", "resizer."))))
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    if not trainable or not any(name.startswith("official_geometry_head.") for name in trainable):
        raise ValueError("Expected an initialized official_geometry_head for this entrypoint")
    return trainable


def _check_disjoint(train_set, val_set):
    """Explicit ID selections must meet the same recording/frame isolation."""
    if val_set is None:
        return
    training_scenes = {clip["sequence_id"] for clip in train_set.clips}
    validation_scenes = {clip["sequence_id"] for clip in val_set.clips}
    training_frames = {frame for clip in train_set.clips for frame in clip["frames"]}
    validation_frames = {frame for clip in val_set.clips for frame in clip["frames"]}
    if training_scenes & validation_scenes or training_frames & validation_frames:
        raise ValueError("Internal train/validation recording IDs or image frames overlap")


def run(args):
    # Set visibility before importing torch or the CUDA-bound model modules.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.device == "cuda" else ""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    project = Path(__file__).resolve().parent
    sys.path.insert(0, str(project))
    import numpy as np
    import torch
    import yaml
    from lib.datasets.mono3drefer.official_vlsot_dataset import OfficialVLSOTDataset, split_train_by_sequence
    from lib.helpers.dataloader_helper import build_official_dataloader
    from lib.helpers.trainer_helper import Trainer, load_initial_checkpoint

    if args.device != "cuda":
        raise RuntimeError("The inherited model has CUDA-only TTM/attention code. Run CPU trainer regression tests instead.")
    if not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA GPU is unavailable")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg["model"].update(
        target_encoding="camera_geometric_center_v2",
        text_encoder_path=str(Path(args.source_project).resolve() / "roberta-base"),
        text_encoder_from_config=True, swin_use_pretrained_backbone=False,
        device="cpu", freeze_text_encoder=args.trainable != "all",
        freeze_backbone=args.trainable != "all",
        input_size_hw=[args.input_height, args.input_width],
        official_correct_padding_mask=args.correct_padding_mask,
        official_temporal_residual=args.temporal_residual,
        official_head_init_std=args.head_init_std,
    )
    cfg["model"].pop("pretrained_backbone_path", None)
    train_ids, val_ids = _ids(args.train_ids), _ids(args.val_ids)
    split_record = None
    if not args.no_validation and train_ids is None:
        split_record = split_train_by_sequence(args.data_root, val_fraction=args.val_fraction, seed=args.seed)
        train_ids, val_ids = split_record["train_ids"], split_record["val_ids"]
    frames = None if args.max_frames_per_clip is None else list(range(args.max_frames_per_clip))
    dataset_kwargs = dict(root=args.data_root, calibration_root=args.calibration_root,
                          input_size_wh=(args.input_width, args.input_height),
                          sample_mode=args.sample_mode, frame_indices=frames,
                          label_center_reference=args.label_center_reference)
    train_set = OfficialVLSOTDataset(split="train", clip_ids=train_ids, **dataset_kwargs)
    val_set = None if args.no_validation else OfficialVLSOTDataset(split="val", clip_ids=val_ids, **dataset_kwargs)
    _check_disjoint(train_set, val_set)
    train_loader = build_official_dataloader(train_set, workers=args.workers, shuffle=True, seed=args.seed, pin_memory=True)
    val_loader = None if val_set is None else build_official_dataloader(val_set, workers=args.workers, seed=args.seed, pin_memory=True)

    # No output directory is created until data selection and leakage checks pass.
    output = Path(args.output).expanduser().resolve()
    if args.resume:
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger("official_vlsot_train")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(str(output / "train.log"), encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    from lib.helpers.model_helper import build_model
    model, criterion = build_model(cfg["model"])
    initialization = load_initial_checkpoint(model, args.checkpoint, allow_new_geometry_head=True)
    trainable = configure_trainable(model, args.trainable)
    model = model.to("cuda:0")
    criterion = criterion.to("cuda:0")
    head_parameters, other_parameters = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (head_parameters if name.startswith("official_geometry_head.") else other_parameters).append(parameter)
    groups = [{"params": head_parameters, "lr": args.lr if args.head_lr is None else args.head_lr}]
    if other_parameters:
        groups.append({"params": other_parameters, "lr": args.lr})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs), gamma=0.5)
    trainer_cfg = {
        "save_path": str(output.parent), "max_epoch": args.epochs, "save_frequency": 1,
        "max_train_batches": args.max_train_batches, "max_val_batches": args.max_val_batches,
        "max_frames_per_clip": args.max_frames_per_clip, "amp": args.amp,
        "grad_clip_norm": 0.1, "frozen_modules_eval": True,
        "target_encoding": "camera_geometric_center_v2", "resume_model": args.resume,
        "model_options": {"official_correct_padding_mask": args.correct_padding_mask,
                          "official_temporal_residual": args.temporal_residual,
                          "input_size_hw": [args.input_height, args.input_width]},
    }
    manifest = {
        "schema_version": 1, "status": "training_configured", "args": vars(args),
        "target_encoding": "camera_geometric_center_v2",
        "label_convention_verified": False,
        "label_convention_note": "Experimental geometric-center/native-coordinate interpretation; diagnostic training only until source alignment is independently established.",
        "source_checkpoint": str(Path(args.checkpoint).resolve()), "initialization": initialization,
        "train_clip_ids": [clip["video_id"] for clip in train_set.clips],
        "val_clip_ids": [] if val_set is None else [clip["video_id"] for clip in val_set.clips],
        "split_record": split_record, "sample_mode": args.sample_mode,
        "temporal_backpropagation": "per-frame loss mean, detached memory within clip, reset memory and phase between clips",
        "trainable_parameters": trainable,
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
        "criterion": type(criterion).__name__, "supervision": ["projected_center", "camera_z", "dimensions_hwl"],
        "no_bbox_yaw_or_dense_depth_ground_truth": True,
        "validation_scope": "none" if val_set is None else "recording/frame-disjoint subset of official TRAIN",
        "official_test_used_for_training_or_selection": False,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    logger.info("Initialization: %s", initialization)
    logger.info("Trainable parameters=%d; train=%d samples; validation=%s", manifest["trainable_parameter_count"], len(train_set), len(val_set) if val_set is not None else "disabled")
    trainer = Trainer(trainer_cfg, model, optimizer, train_loader, val_loader, scheduler, None, logger, criterion, output.name)
    torch.cuda.reset_peak_memory_stats()
    history = trainer.train()
    manifest.update(status="bounded_training_complete" if args.max_train_batches else "training_complete",
                    final_train_metrics=trainer.last_train_metrics, final_val_metrics=trainer.last_val_metrics,
                    completed_epoch=trainer.epoch, best_epoch=trainer.best_epoch,
                    peak_cuda_memory_mib=torch.cuda.max_memory_allocated() / 1024 ** 2)
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "epochs": trainer.epoch, "history": history,
                      "peak_cuda_memory_mib": manifest["peak_cuda_memory_mib"]}, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
