"""
Mono3DVG (https://github.com/ZhanYang-nwpu/Mono3DVG)
@ Paper: https://arxiv.org/pdf/2312.08022.pdf
@ Dataset: https://drive.google.com/drive/folders/1ICBv0SRbRIUnl_z8DVuH8lz7KQt580EI?usp=drive_link
"""

import torch
import torch.nn.functional as F
from torch import nn
import math
import copy

from utils import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                            accuracy, get_world_size, interpolate,
                            is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .mono3dvg_transformer import build_mono3dvg_trans, MultiheadDiffAttn
from .depth_predictor import DepthPredictor
from .depth_predictor.ddn_loss import DDNLoss
from lib.losses.focal_loss import sigmoid_focal_loss
from lib.losses.official_geometry import OfficialGeometryHead, OfficialGeometryCriterion, ENCODING

from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

def convert_to_full_precision(data):
    """
    递归地将传入列表中的所有tensor转换为float32（全精度）。
    如果传入的是tensor，直接转换为float32。
    如果是列表或元组，则递归处理。
    """
    if isinstance(data, torch.Tensor):
        # 如果是tensor，将其转换为float32
        if data.dtype == torch.float16:
            return data.float()
        else:
            return data
        # return data.float()
    elif isinstance(data, list):
        # 如果是列表，递归地转换其中的tensor
        return [convert_to_full_precision(item) for item in data]
    elif isinstance(data, tuple):
        # 如果是元组，递归地转换其中的tensor
        return tuple(convert_to_full_precision(item) for item in data)
    else:
        # 如果是其他数据类型，直接返回
        return data
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Mono3DVG(nn.Module):
    """ Thi  s is the Mono3DVG-TR network that performs Monocular 3D Visual Grounding """

    def __init__(self, backbone, mono3dvg_transformer, depth_predictor, num_classes,
                 num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, init_box=False,
                 text_encoder_type="roberta-base",
                 freeze_text_encoder=False,cfg = None):
                
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            mono3dvg_transformer: depth-aware transformer architecture. See depth_aware_transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For KITTI, we recommend 50 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
        """
        super().__init__()
 
        self.num_queries = num_queries
        self.mono3dvg_transformer = mono3dvg_transformer
        self.depth_predictor = depth_predictor
        hidden_dim = mono3dvg_transformer.d_model
        self.num_feature_levels = num_feature_levels

        # grounding heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3) #6个dim  3 层
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2)  # depth and deviation

        if init_box == True:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if num_feature_levels > 1:
            if cfg['backbone'].startswith ('swin'):
                num_backbone_outs = len(backbone.backbone.intermediate_channel_sizes)
            else:
                num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.backbone.intermediate_channel_sizes[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.target_encoding = cfg.get("target_encoding", "legacy")
        temporal_residual = cfg.get("official_temporal_residual", False)
        if not isinstance(temporal_residual, bool):
            raise ValueError("official_temporal_residual must be a boolean")
        if temporal_residual and self.target_encoding != ENCODING:
            raise ValueError("official_temporal_residual is supported only by camera_geometric_center_v2")
        self.mono3dvg_transformer.decoder.official_temporal_residual = temporal_residual
        if temporal_residual and not self.mono3dvg_transformer.decoder.return_intermediate:
            raise ValueError("official_temporal_residual requires return_intermediate_dec=True")
        if self.target_encoding == ENCODING:
            self.mono3dvg_transformer.TextGuidedAdapter.dynamic_shapes = True
            if cfg.get("official_correct_padding_mask", False):
                for module in self.mono3dvg_transformer.modules():
                    if isinstance(module, MultiheadDiffAttn):
                        module.correct_padding_mask = True
            self.official_geometry_head = OfficialGeometryHead(
                hidden_dim, cfg.get("official_depth_scale", 50.0),
                init_std=cfg.get("official_head_init_std", 0.001))
        elif self.target_encoding != "legacy":
            raise ValueError("Unknown target_encoding: " + self.target_encoding)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = mono3dvg_transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.mono3dvg_transformer.decoder.bbox_embed = self.bbox_embed
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.mono3dvg_transformer.decoder.dim_embed = self.dim_embed_3d
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.depth_embed = _get_clones(self.depth_embed, num_pred)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = nn.ModuleList([self.angle_embed for _ in range(num_pred)])
            self.depth_embed = nn.ModuleList([self.depth_embed for _ in range(num_pred)])
            self.mono3dvg_transformer.decoder.bbox_embed = None
#################################################################################        # 
        # Use the existing local text encoder, so reproduction never downloads
        # a different upstream revision or depends on a network connection.
        text_encoder_type = cfg.get("text_encoder_path", text_encoder_type)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            text_encoder_type, local_files_only=True
        )
        if cfg.get("text_encoder_from_config", False):
            # Inference tools strictly load the complete task state afterwards,
            # including this encoder. Avoid bundling duplicate base weights.
            self.text_encoder = RobertaModel(RobertaConfig.from_pretrained(
                text_encoder_type, local_files_only=True))
        else:
            self.text_encoder = RobertaModel.from_pretrained(
                text_encoder_type, local_files_only=True
            )
# #######################################################################################
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        self.Is_Swin = cfg['backbone'].startswith('swin')
        self.expander_dropout = 0.1
        config = self.text_encoder.config
        self.resizer = FeatureResizer(
            input_feat_size=config.hidden_size,
            output_feat_size=hidden_dim,
            dropout=self.expander_dropout,
        )

    def forward(self, images, calibs,  img_sizes, text):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        # aa = torch.randn(3,3,128,128)
        # a = self.backbone.backbone.model.patch_embed(aa)  # torch.Size([1, 3, 384, 1280])
        batch_size, num_channels, height, width = images.shape
        
        # if pixel_mask is None:
        pixel_mask = torch.zeros(((batch_size, height, width)), dtype=torch.bool, device=images.device)
        # scaler = torch.cuda.amp.GradScaler()

        # with torch.cuda.amp.autocast():
        features, pos = self.backbone(images, pixel_mask)
        # features, pos = self.backbone(images,pixel_mask)   #   torch.Size([1, 3, 384, 1280])
        features = convert_to_full_precision(features)
        pos = convert_to_full_precision(pos)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            if self.Is_Swin:
            # src, mask = feat.decompose()
                src, mask = feat
            else:
                src, mask = feat.decompose()
                
              
            srcs.append(self.input_proj[l](src.float()))  #512->256    #   统一到256dim
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    if not self.Is_Swin:
                        src = self.input_proj[l](features[-1].tensors)
                    else:
                        src = self.input_proj[l](features[-1][0])
                else:
                    src = self.input_proj[l](srcs[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                # pos_l = self.backbone.position_embedding(NestedTensor(src, mask)).to(src.dtype)    #vacjbone是pos信息
                pos_l = self.backbone.position_embedding(src, mask).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        device = srcs[0].device
        if isinstance(text, dict):
            # Tensor-only text entry for ONNX. Tokenization remains in Python;
            # these are the same RoBERTa IDs/mask used by the string branch.
            encoded_text = self.text_encoder(
                input_ids=text["input_ids"], attention_mask=text["attention_mask"])
            text_memory = encoded_text.last_hidden_state.transpose(0, 1)
            text_attention_mask = text["attention_mask"].ne(1).bool()
            text_memory_resized = self.resizer(text_memory)
            tokenized = text
        elif isinstance(text[0], str):      #text就是描述的文本
            # Encode the text
            tokenized = self.tokenizer.batch_encode_plus(text,max_length=110, padding="max_length",truncation=True, return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)#  #baseModelOutputWithPoolingAndCrossAttentions(last_hidden_stat

            # Transpose memory because pytorch's attention expects sequence first
            text_memory = encoded_text.last_hidden_state.transpose(0, 1)#[110, 1, 768])     NOTE  110是最大文本长度 包括填充的  768是hidden dim
            # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
            text_attention_mask = tokenized.attention_mask.ne(1).bool()#[1, 110])

            # Resize the encoder hidden states to be of the same d_model as the decoder
            text_memory_resized = self.resizer(text_memory)#[110, 1, 256])  就是一个fc层   同意维度 方便后续
        else:
            # The text is already encoded, use as is.
            text_attention_mask, text_memory_resized, tokenized = text

        # permute LenxBxDim to BxLenxDim
        
        #到这一段不用看  可以换任意huggingface上面的模型 
        text_memory_resized = text_memory_resized.permute(1, 0, 2)    # [1, 110, 256])    NOTE  上面就是把文本 一句话   [1，] -> [1,max_lenth,768]  -> [1,ma_lenth,256]  后续处理

        pred_depth_map_logits, depth_pos_embed, weighted_depth = self.depth_predictor(srcs, masks[1], # depth_predictor占显存
                                            pos[1],text_memory_resized, text_attention_mask)
#传入的特征 srcs   masks   pos    pred_depth_map_logits,Torch.Size([1, 81, 24, 80]),  depth_pos_embed,1,256,24,80 , weighted_depth  [1, 24, 80])
        query_embeds = self.query_embed.weight  #1,256 

        hs, init_reference, inter_references, inter_references_dim, enc_outputs_class, enc_outputs_coord_unact = self.mono3dvg_transformer(
            srcs, masks, pos, query_embeds, depth_pos_embed,text_memory_resized, text_attention_mask)

        # Average Pooling
        hs = hs.mean(dim=2).unsqueeze(2)    # hs是 [1, 1, 1, 256]) 
        init_reference = init_reference.mean(dim=1).unsqueeze(1) #[1, 1, 2])
        inter_references = inter_references.mean(dim=2).unsqueeze(2)#  ize([1, 1, 1, 6])
        inter_references_dim = inter_references_dim.mean(dim=2).unsqueeze(2)#e([1, 1, 1, 3])
        if self.target_encoding == ENCODING:
            # The official labels do not contain 2D extent or yaw supervision.
            # Use calibrated camera center/depth and positive metric HWL,
            # without the inherited camera-depth/native-Z contradiction.
            out = self.official_geometry_head(hs[-1])
            out["pred_logits"] = self.class_embed[-1](hs[-1])
            out["target_encoding"] = ENCODING
            return out
#hs 里面是最终处理好的特征？？？？
        outputs_coords = []
        outputs_classes = []
        outputs_3d_dims = []
        outputs_depths = []
        outputs_angles = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference    # 1  1  2 #初始？
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)#

            tmp = self.bbox_embed[lvl](hs[lvl])  #256 -> 6 
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference       #对xy进行调整

            # 3d center + 2d box #  1   1   6   3dbox 
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)  #  1   1   6

            # classes    1   1   9 classs
            outputs_class = self.class_embed[lvl](hs[lvl])  #256 -> 9
            outputs_classes.append(outputs_class)

            # 3D sizes
            size3d = inter_references_dim[lvl]
            outputs_3d_dims.append(size3d)
           #深度信息暂时不看
            # depth_geo
            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1: 2], min=1.0)
            depth_geo = size3d[:, :, 0] / box2d_height * calibs[:, 0, 0].unsqueeze(1)

            # depth_reg
            depth_reg = self.depth_embed[lvl](hs[lvl])

            # depth_map
            outputs_center3d = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2).detach()
            depth_map = F.grid_sample(
                weighted_depth.unsqueeze(1),
                outputs_center3d,
                mode='bilinear',
                align_corners=True).squeeze(1)


            # depth average + sigma
            depth_ave = torch.cat([((1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.) + depth_geo.unsqueeze(-1) + depth_map) / 3,
                                    depth_reg[:, :, 1: 2]], -1)
            outputs_depths.append(depth_ave)

            # angles
            outputs_angle = self.angle_embed[lvl](hs[lvl])
            outputs_angles.append(outputs_angle)

        outputs_coord = torch.stack(outputs_coords) # 1 1 6
        outputs_class = torch.stack(outputs_classes)# 9 
        outputs_3d_dim = torch.stack(outputs_3d_dims) # 3 
        outputs_depth = torch.stack(outputs_depths)#  1 1 1 2
        outputs_angle = torch.stack(outputs_angles)# 1 1 1 24
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}  #取最后一层？状态
        out['pred_3d_dim'] = outputs_3d_dim[-1]
        out['pred_depth'] = outputs_depth[-1]
        out['pred_angle'] = outputs_angle[-1]
        out['pred_depth_map_logits'] = pred_depth_map_logits



        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 
                 'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e}
                for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1],
                                         outputs_3d_dim[:-1], outputs_angle[:-1], outputs_depth[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for Mono3DVG-TR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss()  # for depth map

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes_o = target_classes_o.cuda()
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_3dcenter(self, outputs, targets, indices, num_boxes):
        
        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = outputs['pred_boxes'][:, :, 0: 2][idx]
        target_3dcenter = torch.cat([t['boxes'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # target_cboxes = torch.cat([t['boxes'][:, 0: 4][i] for t, (_, i) in zip(targets, indices)], dim=0)  # [xmin, ymin, xmax, ymax]
        # target_3dcenter = torch.stack([(target_cboxes[:, 0] + target_cboxes[:, 2]) / 2,   # (xmin + xmax) / 2
        #                            (target_cboxes[:, 1] + target_cboxes[:, 3]) / 2],  # (ymin + ymax) / 2
        #                           dim=1)

        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')
        losses = {}
        losses['loss_center'] = loss_3dcenter.sum() / num_boxes
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_2dboxes = outputs['pred_boxes'][:, :, 2: 6][idx]
        target_2dboxes = torch.cat([t['boxes_3d'][:, 2: 6][i] for t, (_, i) in zip(targets, indices)], dim=0) # t['boxes_3d'][:, 2: 6][i]
        target_2dboxes = target_2dboxes.cuda()

        # l1
        loss_bbox = F.l1_loss(src_2dboxes, target_2dboxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # giou
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_boxes = target_boxes.cuda()
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcylrtb_to_xyxy(src_boxes),
            box_ops.box_cxcylrtb_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_depths(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
        src_depths = outputs['pred_depth'][idx]
        target_depths = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()

        depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1] 
        depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + depth_log_variance  
        losses = {}
        losses['loss_depth'] = depth_loss.sum() / num_boxes 
        return losses  
    
    def loss_dims(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
        src_dims = outputs['pred_3d_dim'][idx]
        target_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_dims = target_dims.cuda()

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses['loss_dim'] = dim_loss.sum() / num_boxes
        return losses

    def loss_angles(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
        heading_input = outputs['pred_angle'][idx]
        target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_heading_cls = target_heading_cls.cuda()
        target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_heading_res = target_heading_res.cuda()

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')
        
        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / num_boxes 
        return losses

    def loss_depth_map(self, outputs, targets, indices, num_boxes):
        depth_map_logits = outputs['pred_depth_map_logits']

        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * torch.tensor([80, 24, 80, 24], device='cuda')
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)

        losses = dict()
        losses["loss_depth_map"] = self.ddn_loss(
            depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth)
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'depths': self.loss_depths,
            'dims': self.loss_dims,
            'angles': self.loss_angles,
            'center': self.loss_3dcenter,
            'depth_map': self.loss_depth_map,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'depth_map':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses



class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(cfg):
    # backbone
    backbone = build_backbone(cfg)   #  这个backbone 
    
    #TEST
    
    # backbone = build_swim_backbone(cfg)   #   得到swin
    
    # depth prediction and depth encoder module
    depth_predictor = DepthPredictor(cfg) #   得到depth_predictor

    # visual encoder and text-guided adapter and grounding encoder
    # 翻译：视觉编码器和文本引导适配器和接地编码器
    mono3dvg_trans = build_mono3dvg_trans(cfg)  #   得到主体

    # Mono3DVG
    model = Mono3DVG(
        backbone,
        mono3dvg_trans,
        depth_predictor,
        num_classes=cfg['num_classes'],
        num_queries=cfg['num_queries'],
        aux_loss=cfg['aux_loss'],
        num_feature_levels=cfg['num_feature_levels'],
        with_box_refine=cfg['with_box_refine'],
        init_box=cfg['init_box'],
        freeze_text_encoder=cfg['freeze_text_encoder'],
        cfg=cfg,
    )

    if cfg.get("target_encoding", "legacy") == ENCODING:
        criterion = OfficialGeometryCriterion(
            cfg.get("3dcenter_loss_coef", 10.0), cfg.get("depth_loss_coef", 1.0),
            cfg.get("dim_loss_coef", 1.0)).to(torch.device(cfg["device"]))
        return model, criterion

    # matcher
    matcher = build_matcher(cfg)

    # loss
    weight_dict = {'loss_ce': cfg['cls_loss_coef'], 'loss_bbox': cfg['bbox_loss_coef']}
    weight_dict['loss_giou'] = cfg['giou_loss_coef']
    weight_dict['loss_dim'] = cfg['dim_loss_coef']
    weight_dict['loss_angle'] = cfg['angle_loss_coef']
    weight_dict['loss_depth'] = cfg['depth_loss_coef']
    weight_dict['loss_center'] = cfg['3dcenter_loss_coef']
    weight_dict['loss_depth_map'] = cfg['depth_map_loss_coef']

    # TODO this is a hack
    if cfg['aux_loss']:
        aux_weight_dict = {}
        for i in range(cfg['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'depth_map']
    
    criterion = SetCriterion(
        cfg['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=cfg['focal_alpha'],
        losses=losses)
    device = torch.device(cfg['device'])
    criterion.to(device)
    
    return model, criterion
