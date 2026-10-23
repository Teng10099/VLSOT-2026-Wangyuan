"""Single referred-object supervision using only official LWH/XYZ labels.

The camera encoding is an explicit new training contract. It never uses
unprovided 2D boxes, classes, yaw, or dense depth as ground truth.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


ENCODING = "camera_geometric_center_v2"


class OfficialGeometryHead(nn.Module):
    def __init__(self, hidden_dim, depth_scale=50.0, init_std=0.001):
        super().__init__()
        if not math.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError("depth_scale must be finite and positive")
        self.depth_scale = float(depth_scale)
        self.layers = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, 6))
        # Small nonzero weights allow gradients to reach decoder features.
        if not math.isfinite(init_std) or init_std <= 0:
            raise ValueError("init_std must be finite and positive")
        nn.init.normal_(self.layers[-1].weight, std=init_std)
        with torch.no_grad():
            self.layers[-1].bias.copy_(torch.tensor([
                0.5, 0.5, math.log(math.expm1(1.0)),
                math.log(math.expm1(1.8)), math.log(math.expm1(1.8)),
                math.log(math.expm1(4.5))]))

    def forward(self, features):
        raw = self.layers(features)
        center = raw[..., :2]  # Projection may lie outside the image.
        depth = F.softplus(raw[..., 2:3]) * self.depth_scale + 1e-4
        dimensions = F.softplus(raw[..., 3:6]) + 1e-4  # H,W,L in meters
        return {"pred_boxes": torch.cat((center, raw.new_zeros(raw.shape[:-1] + (4,))), -1),
                "pred_depth": torch.cat((depth, torch.zeros_like(depth)), -1),
                "pred_3d_dim": dimensions}


class OfficialGeometryCriterion(nn.Module):
    def __init__(self, center_weight=10.0, depth_weight=1.0, dim_weight=1.0):
        super().__init__()
        self.weight_dict = {"loss_center": center_weight,
                            "loss_depth": depth_weight, "loss_dim": dim_weight}

    def forward(self, outputs, targets):
        center = outputs["pred_boxes"][..., :2]
        depth = outputs["pred_depth"][..., :1]
        dimensions = outputs["pred_3d_dim"]
        if center.ndim != 3 or center.shape[1:] != (1, 2):
            raise ValueError("Official single-object loss requires [B,1,2] centers")
        if len(targets) != center.shape[0]:
            raise ValueError("Target count differs from batch size")
        values = {}
        for name, width in (("projected_center", 2), ("depth", 1), ("size_3d", 3)):
            parts = []
            for item in targets:
                value = item[name].to(device=center.device, dtype=center.dtype)
                if value.shape != (1, width) or not torch.isfinite(value).all():
                    raise ValueError("Invalid official target: " + name)
                if name != "projected_center" and not (value > 0).all():
                    raise ValueError("Metric depth and dimensions must be positive")
                parts.append(value)
            values[name] = torch.stack(parts)
        if not (depth > 0).all() or not (dimensions > 0).all():
            raise ValueError("Use the positive official geometry head")
        return {
            "loss_center": F.l1_loss(center, values["projected_center"]),
            "loss_depth": F.smooth_l1_loss(depth.log(), values["depth"].log()),
            "loss_dim": F.smooth_l1_loss(dimensions.log(), values["size_3d"].log()),
            "center_abs_error_normalized": F.l1_loss(center.detach(), values["projected_center"]),
            "depth_abs_error_m": F.l1_loss(depth.detach(), values["depth"]),
            "dims_abs_error_m": F.l1_loss(dimensions.detach(), values["size_3d"]),
        }
