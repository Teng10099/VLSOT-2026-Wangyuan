"""Train on every frame with one-frame truncated BPTT and clip-boundary reset.

Each frame contributes loss / frames_in_clip. TTM memory values carry within a
clip but past graphs are detached after each backward. This is deliberately not
full-clip backpropagation, and independent videos never share memory or phase.
"""
from collections import defaultdict
from contextlib import nullcontext
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _detach(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, list):
        return [_detach(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)
    if isinstance(value, dict):
        return {key: _detach(item) for key, item in value.items()}
    return value


def reset_temporal_state(model):
    """Reset runtime caches only, never learned parameters or buffers."""
    model = _unwrap(model)
    for module in model.modules():
        if hasattr(module, "TTM_model"):
            module.memory_tokens = None
            dnc = getattr(module.TTM_model, "simpleDNC", None)
            if dnc is not None:
                dnc.current_flag = 0
                dnc.split_memory_tokens = []
    if hasattr(model, "reset_temporal_state"):
        model.reset_temporal_state()


def detach_temporal_state(model):
    for module in _unwrap(model).modules():
        if hasattr(module, "TTM_model"):
            module.memory_tokens = _detach(getattr(module, "memory_tokens", None))
            dnc = getattr(module.TTM_model, "simpleDNC", None)
            if dnc is not None and hasattr(dnc, "split_memory_tokens"):
                dnc.split_memory_tokens = _detach(dnc.split_memory_tokens)


def _load(path, device="cpu"):
    return torch.load(str(path), map_location=device, weights_only=True)


def load_initial_checkpoint(model, path, allow_new_geometry_head=False):
    """Allow missing ONLY official_geometry_head.*; all legacy keys must match."""
    checkpoint = _load(path)
    state = checkpoint.get("model_state", checkpoint)
    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    model = _unwrap(model)
    expected, supplied = set(model.state_dict()), set(state)
    missing, unexpected = expected - supplied, supplied - expected
    permitted = {key for key in expected if key.startswith("official_geometry_head.")}
    if unexpected or (missing and not (allow_new_geometry_head and permitted and missing <= permitted)):
        raise RuntimeError("Checkpoint key mismatch: missing={}, unexpected={}".format(sorted(missing), sorted(unexpected)))
    model.load_state_dict(state, strict=not bool(missing))
    return {"missing_new_head_keys": sorted(missing), "checkpoint_epoch": checkpoint.get("epoch")}


def _target_list(target, batch_size):
    if isinstance(target, list) and len(target) == batch_size and all(isinstance(item, dict) for item in target):
        return target
    if not isinstance(target, dict):
        raise TypeError("Expected a collated target dict or one target dict per image")
    result = []
    for batch_index in range(batch_size):
        current = {}
        for key, value in target.items():
            if torch.is_tensor(value) or isinstance(value, np.ndarray):
                if value.ndim == 0 or value.shape[0] != batch_size:
                    raise ValueError("Target {!r} lacks its batch dimension".format(key))
                current[key] = value[batch_index]
            elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                current[key] = value[batch_index]
            else:
                current[key] = value
        result.append(current)
    return result


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, device=device)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


class Trainer:
    def __init__(self, cfg, model, optimizer, train_loader, val_loader,
                 lr_scheduler, warmup_lr_scheduler, logger, loss, model_name):
        self.cfg, self.model, self.optimizer = cfg, model, optimizer
        self.train_loader, self.val_loader = train_loader, val_loader
        self.lr_scheduler, self.warmup_lr_scheduler = lr_scheduler, warmup_lr_scheduler
        self.logger = logger or logging.getLogger(__name__)
        self.mono3dvg_loss, self.model_name = loss, model_name
        self.device = next(model.parameters()).device
        self.output_dir = Path(cfg["save_path"]) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epoch, self.best_epoch = 0, 0
        self.best_result, self.best_val_loss = 0.0, float("inf")
        self.last_train_metrics, self.last_val_metrics = {}, {}
        self.tester = None
        self.amp = bool(cfg.get("amp", False)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        if cfg.get("detr_model"):
            raise ValueError("Legacy load_detr did not load any weights. Use an explicit complete checkpoint instead.")
        if cfg.get("pretrain_model"):
            load_initial_checkpoint(self.model, cfg["pretrain_model"], bool(cfg.get("allow_new_geometry_head", False)))
        if cfg.get("resume_model"):
            path = cfg["resume_model"]
            self.resume(self.output_dir / "checkpoint_latest.pth" if path is True else path)

    def resume(self, path):
        checkpoint = _load(path, self.device)
        if checkpoint.get("target_encoding", "legacy") != self.cfg.get("target_encoding", "legacy"):
            raise ValueError("Resume target encoding disagrees with the current model")
        previous = checkpoint.get("model_options", {})
        current = self.cfg.get("model_options", {})
        if bool(previous.get("official_correct_padding_mask", False)) != bool(current.get("official_correct_padding_mask", False)):
            raise ValueError("Resume padding-mask behavior disagrees with the current model")
        if bool(previous.get("official_temporal_residual", False)) != bool(current.get("official_temporal_residual", False)):
            raise ValueError("Resume temporal-residual behavior disagrees with the current model")
        if previous.get("input_size_hw") is not None and previous["input_size_hw"] != current.get("input_size_hw"):
            raise ValueError("Resume input resolution disagrees with the checkpoint")
        _unwrap(self.model).load_state_dict(checkpoint["model_state"], strict=True)
        if checkpoint.get("optimizer_state") is None:
            raise ValueError("Resume requires optimizer state; use initialization for model-only weights")
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        for name, instance in (("lr_scheduler", self.lr_scheduler), ("warmup_lr_scheduler", self.warmup_lr_scheduler), ("scaler", self.scaler)):
            if instance is not None and checkpoint.get(name) is not None:
                instance.load_state_dict(checkpoint[name])
        self.epoch = int(checkpoint.get("epoch", 0))
        self.best_epoch = int(checkpoint.get("best_epoch", 0))
        self.best_result = float(checkpoint.get("best_result", 0))
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        self.logger.info("Resumed %s at epoch %d", path, self.epoch)

    def _save(self, name):
        checkpoint = {
            "epoch": self.epoch, "model_state": _unwrap(self.model).state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_epoch": self.best_epoch, "best_result": self.best_result,
            "best_val_loss": self.best_val_loss,
            "target_encoding": self.cfg.get("target_encoding", "legacy"),
            "model_options": self.cfg.get("model_options", {}),
            "train_metrics": self.last_train_metrics, "val_metrics": self.last_val_metrics,
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "warmup_lr_scheduler": self.warmup_lr_scheduler.state_dict() if self.warmup_lr_scheduler is not None else None,
            "scaler": self.scaler.state_dict(),
        }
        destination = self.output_dir / (name + ".pth")
        temporary = destination.with_suffix(".pth.tmp")
        torch.save(checkpoint, str(temporary))
        temporary.replace(destination)

    def train(self):
        if self.train_loader is None:
            raise ValueError("No training loader was supplied")
        history = []
        for epoch in range(self.epoch, int(self.cfg["max_epoch"])):
            self.train_one_epoch(epoch)
            self.epoch = epoch + 1
            warmup_epochs = int(self.cfg.get("warmup_epochs", 5))
            scheduler = self.warmup_lr_scheduler if self.warmup_lr_scheduler is not None and epoch < warmup_epochs else self.lr_scheduler
            if scheduler is not None:
                scheduler.step()
            if self.epoch % int(self.cfg.get("save_frequency", 1)) == 0:
                if self.tester is not None:
                    raise ValueError("The legacy Tester misinterprets encoded boxes; use verified evaluation or validation loss")
                if self.val_loader is not None:
                    current = self.val_one_epoch(self.epoch)
                    if current < self.best_val_loss:
                        self.best_val_loss, self.best_epoch = current, self.epoch
                        self._save("checkpoint_best")
                self._save("checkpoint_latest")
            record = {"epoch": self.epoch, "train": self.last_train_metrics, "validation": self.last_val_metrics}
            history.append(record)
            with (self.output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        self._save("checkpoint_latest")
        return history

    def _frames(self, batch):
        if isinstance(batch, dict):
            return [batch]
        if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], dict) and "images" in batch[0]:
            return list(batch)
        if not isinstance(batch, (list, tuple)) or len(batch) not in (3, 4):
            raise ValueError("Expected official frame/clip dicts or a legacy 3/4-tuple")
        inputs, calibs, targets = batch[:3]
        if torch.is_tensor(inputs):
            if inputs.ndim != 5:
                raise ValueError("Legacy clip tensor must have shape [B,T,C,H,W]")
            images = [inputs[:, index] for index in range(inputs.shape[1])]
        else:
            images = list(inputs)
        prepared = self.prepare_targets(targets, images[0].shape[0])
        captions = targets["text"]["text"]
        return [{"images": image, "calibs": calibs[:, index] if calibs.ndim == 4 else calibs,
                 "image_sizes": targets["other"][index]["img_size"], "text": captions,
                 "target": prepared[index]} for index, image in enumerate(images)]

    def _prepare_frame(self, frame):
        images = _to_device(frame["images"], self.device)
        if images.ndim != 4:
            raise ValueError("Batched images must have shape [B,C,H,W]")
        if "target" not in frame:
            raise ValueError("Cannot train/validate on official test samples without targets")
        targets = _to_device(_target_list(frame["target"], images.shape[0]), self.device)
        captions = frame["text"]
        captions = [captions] if isinstance(captions, str) else list(captions)
        if len(captions) != images.shape[0]:
            raise ValueError("One caption per image is required")
        return images, _to_device(frame["calibs"], self.device), _to_device(frame["image_sizes"], self.device), captions, targets

    def _diagnostics(self, outputs, targets, sizes, calibs, frame=None):
        """Camera-frame error is diagnostic, not a platform ranking score."""
        if not targets or "camera_center_xyz" not in targets[0] or not all(key in outputs for key in ("pred_boxes", "pred_depth", "pred_3d_dim")):
            return {}
        with torch.no_grad():
            centers = outputs["pred_boxes"][:, 0, :2].float() * sizes.float()
            rays = torch.linalg.solve(calibs[:, :, :3].float(), torch.cat((centers, torch.ones_like(centers[:, :1])), dim=-1).unsqueeze(-1)).squeeze(-1)
            predicted = rays / rays[:, 2:].clamp_min(1e-8) * outputs["pred_depth"][:, 0, :1].float()
            truth = torch.stack([target["camera_center_xyz"].reshape(-1, 3)[0] for target in targets]).float()
            dimensions = torch.stack([target["size_3d"].reshape(-1, 3)[0] for target in targets]).float()
            metrics = {
                "camera_center_error_m": float(torch.linalg.vector_norm(predicted - truth, dim=-1).mean()),
                "dimension_mae_m": float((outputs["pred_3d_dim"][:, 0].float() - dimensions).abs().mean()),
            }
            if frame is not None and "lidar_to_camera_R" in frame and "native_center_xyz" in targets[0]:
                rotation = _to_device(frame["lidar_to_camera_R"], self.device).float()
                translation = _to_device(frame["lidar_to_camera_t"], self.device).float()
                # Invert the exact supplied calibration, without an h/2 offset.
                native = torch.linalg.solve(rotation, (predicted - translation).unsqueeze(-1)).squeeze(-1)
                native_truth = torch.stack([target["native_center_xyz"].reshape(-1, 3)[0] for target in targets]).float()
                metrics["official_center_error_m"] = float(torch.linalg.vector_norm(native - native_truth, dim=-1).mean())
            return metrics

    def _run_epoch(self, loader, training, epoch):
        if loader is None:
            raise ValueError("Requested loader is unavailable")
        self.model.train(training)
        self.mono3dvg_loss.train(training)
        if training and self.cfg.get("frozen_modules_eval", True):
            for module in self.model.modules():
                parameters = list(module.parameters())
                if parameters and not any(parameter.requires_grad for parameter in parameters):
                    module.eval()
        totals, frame_count, batch_count = defaultdict(float), 0, 0
        limit = self.cfg.get("max_train_batches" if training else "max_val_batches")
        for batch_index, batch in enumerate(loader):
            if limit is not None and batch_index >= int(limit):
                break
            frames = self._frames(batch)
            max_frames = self.cfg.get("max_frames_per_clip")
            if max_frames is not None:
                frames = frames[:int(max_frames)]
            if not frames:
                raise ValueError("No selected frames in clip")
            reset_temporal_state(self.model)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            try:
                for frame in frames:
                    images, calibs, sizes, captions, targets = self._prepare_frame(frame)
                    with torch.set_grad_enabled(training):
                        autocast = torch.autocast("cuda", dtype=torch.float16) if self.amp else nullcontext()
                        with autocast:
                            outputs = self.model(images, calibs, sizes, captions)
                            losses = self.mono3dvg_loss(outputs, targets)
                            weighted = {key: value * self.mono3dvg_loss.weight_dict[key] for key, value in losses.items() if key in self.mono3dvg_loss.weight_dict}
                            if not weighted:
                                raise ValueError("Criterion produced no weighted supervision")
                            loss = sum(weighted.values())
                        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                            raise FloatingPointError("Nonfinite/nonscalar loss at epoch {}, batch {}, frame {}".format(epoch, batch_index, frame_count))
                        if training:
                            self.scaler.scale(loss / len(frames)).backward()
                    totals["loss"] += float(loss.detach())
                    for key, value in losses.items():
                        if torch.is_tensor(value) and value.numel() == 1:
                            totals[key] += float(value.detach())
                    for key, value in self._diagnostics(outputs, targets, sizes, calibs, frame).items():
                        totals[key] += value
                    frame_count += 1
                    detach_temporal_state(self.model)
                    del outputs, losses, weighted, loss, images, calibs, sizes, targets
                if training:
                    self.scaler.unscale_(self.optimizer)
                    parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad and parameter.grad is not None]
                    if not parameters:
                        raise RuntimeError("No trainable parameter received a gradient")
                    if not all(bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
                        raise FloatingPointError("Nonfinite training gradient")
                    max_norm = float(self.cfg.get("grad_clip_norm", 0.1))
                    if max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(parameters, max_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                batch_count += 1
                self.logger.info("%s epoch=%d batch=%d frames=%d running_loss=%.6f", "train" if training else "validation", epoch, batch_count, frame_count, totals["loss"] / frame_count)
            finally:
                reset_temporal_state(self.model)
        if frame_count == 0:
            raise ValueError("No batches were processed; check data selection and smoke limits")
        metrics = {key: value / frame_count for key, value in totals.items()}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Nonfinite epoch metrics")
        metrics.update(frames=frame_count, batches=batch_count)
        return metrics

    def train_one_epoch(self, epoch):
        self.last_train_metrics = self._run_epoch(self.train_loader, True, epoch)
        return self.last_train_metrics["loss"]

    def val_one_epoch(self, epoch):
        self.last_val_metrics = self._run_epoch(self.val_loader, False, epoch)
        return self.last_val_metrics["loss"]

    @staticmethod
    def prepare_targets(targets, batch_size):
        allowed = {"labels", "boxes", "calibs", "depth", "size_3d", "heading_bin", "heading_res", "boxes_3d"}
        result = []
        for frame in targets["other"]:
            per_image = []
            for batch_index in range(batch_size):
                mask = frame["mask_2d"][batch_index].bool()
                per_image.append({key: value[batch_index][mask] for key, value in frame.items() if key in allowed})
            result.append(per_image)
        return result
