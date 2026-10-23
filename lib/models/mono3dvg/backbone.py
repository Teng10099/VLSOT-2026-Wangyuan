# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers) # IntermediateLayerGetter的两个参数是backbone和return_layers，代表了backbone的输出层

    def forward(self, images):
        xs = self.body(images)
        out = {}
        for name, x in xs.items():
            m = torch.zeros(x.shape[0], x.shape[2], x.shape[3]).to(torch.bool).to(x.device)
            out[name] = NestedTensor(x, m)
        return out


class Backbone(BackboneBase):   #   ('resnet50', True, True, False)   
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)     # 4层
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)#  backbone  pos 传给sequential 
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, images,masks):
        # self[0](images)意思是调用Sequential的第一个模块，也就是backbone
        # all_tfeatures0 = []
        # for t in range(images.size(1)):
        #     timage = images[:, t, :, :, :]  # 选择第t个图像
        #     timage = F.interpolate(timage, scale_factor=0.5, mode='bilinear', align_corners=False) # bilinear是双线性插值，这行代码的目的是将图片的尺寸缩小一半，align_corners是对齐角点
        #     all_tfeatures0.append(timage)
        # all_tfeatures0 = torch.stack(all_tfeatures0)
        # all_tfeatures0 = all_tfeatures0.transpose(0,1)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # conv3d = nn.Conv3d(in_channels=30, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1)).to(device) # Swin3d
        # images = conv3d(all_tfeatures0)
        # images = torch.squeeze(images, dim=0)

        xs = self[0](images)    #里面有3层特征 每层有mask ([1, 48, 160]) 和 tensors  [1, 512, 48, 160])  ch.Size([1, 1024, 24, 80]
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))  #传入掩码和tensor 到pos embed

        return out, pos   #和上面一样的xs


def build_backbone(cfg):
    from utils.parse import dict_to_namespace
    config = dict_to_namespace(cfg)
    backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type
    if backbone_model_type.startswith("swin"):
        #
        from .swim_vision_backbone import VisionBackboneModel
        from . import build_vision_backbone
        position_embedding = build_position_encoding(cfg)
        vision_backbone = build_vision_backbone(cfg)
        # 冻结 vision_backbone 的权重
        for param in vision_backbone.parameters():
            param.requires_grad = False
        model = VisionBackboneModel(vision_backbone,position_embedding)  # 需要pixel_values, pixel_mask 输出out, pos 
        
    elif backbone_model_type.startswith("resnet"):
        position_embedding = build_position_encoding(cfg)
        return_interm_layers = cfg['masks'] or cfg['num_feature_levels'] > 1
        backbone = Backbone(cfg['backbone'], cfg['train_backbone'], return_interm_layers, cfg['dilation'])  #('resnet50', True, True, False)
        model = Joiner(backbone, position_embedding)    #return out 和 pos  需要images

    else:
        raise ValueError(f"Not supported {backbone_model_type}")
    
    
    

    return model


