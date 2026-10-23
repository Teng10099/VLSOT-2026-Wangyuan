import torch
from torch import nn
import torch.cuda.amp as amp
from transformers.utils import requires_backends, logging, is_timm_available
from transformers.models.auto import AutoBackbone

logger = logging.get_logger(__name__)

if is_timm_available():
    from timm import create_model

logger = logging.get_logger(__name__)


class SwinVisionBackbone(nn.Module):
    """
    Swin Vision backbone, using either the AutoBackbone API or one from the timm library.
    """
    
    def __init__(self, config):
        super().__init__()

        self.config = config
        input_size_hw = tuple(getattr(config, "input_size_hw", (1080, 1920)))
        if len(input_size_hw) != 2 or any(not isinstance(v, int) or v <= 0 for v in input_size_hw):
            raise ValueError("input_size_hw must contain positive integer height and width")

        if config.use_timm_backbone:
            requires_backends(self, ["timm"])
            
            if config.backbone.startswith("swinv2"):
                # swin V2 never pad grid size to divide by window size, 
                # but it intorduced Log-Spaced Continuous relative position bias (CPB),
                # which allow the adpated window_size from different window_size of pretrained model.
                kwargs = {
                    'img_size': input_size_hw,
                    'pretrained_window_sizes': (12, 12, 12, 6), # from 12 window_size pretrained model
                }
            elif config.backbone.startswith("swin"):
                kwargs = {'img_size': input_size_hw}
            
            # load local pretrained model or download to .cache/huggingface from timm
            if hasattr(config, 'pretrained_backbone_path') and config.pretrained_backbone_path is not None:
                kwargs['pretrained_cfg_overlay'] = dict(file=config.pretrained_backbone_path)
            
            backbone = create_model(
                config.backbone,
                pretrained=config.swin_use_pretrained_backbone,
                features_only=True,
                out_indices=(1, 2, 3) if config.num_feature_levels > 1 else (3,),
                in_chans=config.num_channels,
                **kwargs,
            )
        else:
            backbone = AutoBackbone.from_config(config.backbone_config)
        
        self.model = backbone
        self.intermediate_channel_sizes = (
            self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        )
        
        backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type
        if "swin" in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if config.use_timm_backbone:
                    if config.freeze_backbone or "layers_1" not in name and "layers_2" not in name and "layers_3" not in name:
                        parameter.requires_grad_(False)
                else:
                    if config.freeze_backbone or "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                        parameter.requires_grad_(False)
    
    # def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor=None):
    #     # with amp.autocast():
    #     # send pixel_values through the model to get list of feature maps
    #     features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps
    #     out = []


    #     for feature_map in features:
    #         # convert feature_map from [B, H, W, C] to [B, C, H, W]
    #         feature_map = feature_map.permute(0, 3, 1, 2).contiguous()
    #         mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
    #         out.append((feature_map, mask))
    #     return out

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor = None):

        out = []

        # 使用 autocast 处理需要半精度运算的部分
        with amp.autocast():
            # 通过模型处理 pixel_values 来得到特征图
            features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps

            for feature_map in features:
                # 将 feature_map 从 [B, H, W, C] 转换为 [B, C, H, W]
                feature_map = feature_map.permute(0, 3, 1, 2).contiguous()

                # 插值操作通常是浮点运算，放在 autocast 内部
                mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]

                # 将特征图和 mask 结果添加到输出列表中
                out.append((feature_map, mask))

        return out
