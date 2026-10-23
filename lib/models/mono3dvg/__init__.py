from .mono3dvg import build
from .swin_backbone import SwinVisionBackbone
from utils.parse import dict_to_namespace
def build_mono3dvg(cfg):
    return build(cfg)







def build_vision_backbone(config):#config明天要改一下
    config = dict_to_namespace(config)
    backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type
    

    if backbone_model_type.startswith("swin"):
        vision_backbone = SwinVisionBackbone(config)
    elif backbone_model_type.startswith("resnet"):
        pass

    else:
        raise ValueError(f"Not supported {backbone_model_type}")

    return vision_backbone