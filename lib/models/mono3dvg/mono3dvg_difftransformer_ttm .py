import math
import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from utils.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn
from .LMTTM import TokenTuringMachineEncoder

class TextGuidedAdapter(nn.Module):   #文本引导图像 通过加权文本对应的图像特征   原论文图五
    def __init__(self, d_model=256, 
                 dropout=0.1,
                 n_levels=4,
                 n_heads=8,
                 n_points=4,):
        super().__init__()

        # img2text: Cross attention
        self.img2text_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        
        self.adapt_proj = MLP(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1)
        self.orig_proj = MLP(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1)

        self.tf_pow = 2.0
        self.tf_scale = nn.Parameter(torch.Tensor([1.0]))
        self.tf_sigma = nn.Parameter(torch.Tensor([0.5]))

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, return_indices=False, ceil_mode=False)

        # img2img: Multi-Scale Deformable Attention
        self.img2img_msdeform_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

        self.norm_text_cond_img = nn.LayerNorm(d_model)
        self.norm_img = nn.LayerNorm(d_model)

        # depth2text: Cross attention
        self.depth2textcross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # depth2depth:  attention
        self.depth2depth_attn =  nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.norm_text_cond_depth = nn.LayerNorm(d_model)
        self.norm_depth = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, img_feat_src, masks, img_pos_embeds,
                reference_points, spatial_shapes,
                level_start_index,
                src_valid_ratios,
                word_feat, word_key_padding_mask,
                depth_pos_embed, mask_depth,im_name, instanceID, ann_id,
                word_pos=None):
        orig_multiscale_img_feat = img_feat_src
        orig_multiscale_masks = masks
        orig_multiscale_img_pos_embeds = img_pos_embeds   #前3个 

        # split four level multi-scale img_feat/masks/img_pos_embeds
        bs, sum, dim = img_feat_src.shape
        img_feat_src_list = img_feat_src.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1) #在dim=1上面来切分 也就是seqense序列上切分
        masks_list = masks.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1)
        img_pos_embeds_list = img_pos_embeds.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1)
        #分成对应层级大小的特征
        # For second level img_feat/masks/img_pos_embeds to compute score   #用第二个层级的特征计算分数
        img_feat_src = img_feat_src_list[1]
        masks = masks_list[1]
        img_pos_embeds = img_pos_embeds_list[1]

                    
        q = self.with_pos_embed(img_feat_src, img_pos_embeds)  # q是图像特征
        k = self.with_pos_embed(word_feat, word_pos)   #k是文本特征 加了位置罢了
        imgfeat_adapt = self.img2text_attn(query=q.transpose(0, 1),    
                                  key=k.transpose(0, 1),
                                  value=word_feat.transpose(0, 1),
                                  key_padding_mask=word_key_padding_mask)[0].transpose(0, 1) #做query 计算atten    q  tokens batch dim ,k tokens batch,dim

        imgfeat_adapt_embed = self.adapt_proj(imgfeat_adapt)  #  文本融合过的   [bs, 1920, 256]  #映射到元图  根据文本信息处理得到的图像
        imgfeat_orig_embed = self.orig_proj(img_feat_src)  #原本就有的

        verify_score = (F.normalize(imgfeat_orig_embed, p=2, dim=-1) *
                        F.normalize(imgfeat_adapt_embed, p=2, dim=-1)).sum(dim=-1, keepdim=True)   #L2范数进行归一化 点乘[1,1920,1]
        verify_score = self.tf_scale * \
                       torch.exp( - (1 - verify_score).pow(self.tf_pow) \
                        / (2 * self.tf_sigma**2))   # [12, 1920, 1]
       #[1, 1920, 1]
        # For score of map-16 to upsample and downsample  #对16层级的分数进行上采样和下采样 
        verify_score_16 = verify_score.reshape(bs, spatial_shapes[1][0], spatial_shapes[1][1], 1).squeeze(-1)#verify_score ->[1, 24, 80]) 
        verify_score_8 = self.upsample(verify_score_16.unsqueeze(1)).squeeze(1)
        verify_score_8 = verify_score_8[:,:-1,:]
        verify_score_32 = self.downsample(verify_score_16)
        verify_score_64 = self.downsample(verify_score_32)
        verify_score_list = [verify_score_8.flatten(1), verify_score_16.flatten(1),verify_score_32.flatten(1), verify_score_64.flatten(1)]
        verify_score = torch.cat(verify_score_list, dim=1).unsqueeze(-1) #这不就是对图像mutil level 做融合拼接吗

        q = k = img_feat_src + imgfeat_adapt   # second image feature   #第二层原有的+融入文本信息的处理后的图像
        # concat multi-scale image feature
        src = torch.cat([img_feat_src_list[0],q ,img_feat_src_list[2],img_feat_src_list[3]], 1)  #拼接多尺度的图像特征 未加入文本特征的

        reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]# 1 1 4 2 ,

        text_cond_img_ctx = self.img2img_msdeform_attn(
                            self.with_pos_embed(src, orig_multiscale_img_pos_embeds),
                            reference_points_input, orig_multiscale_img_feat, spatial_shapes,
                                  level_start_index, orig_multiscale_masks)#NOTE 这里没看懂,是得到了增强后的文本特征吗

        # adapted image feature
        adapt_img_feat = (self.norm_img(orig_multiscale_img_feat) + self.norm_text_cond_img(text_cond_img_ctx)) * verify_score
#adapt_img_feat 可能包含了更关注文本语义的图像特征表达，这对于后续的目标检测或其他视觉任务可能具有重要的影响
        # text-guided depth encoder
        depthfeat_adapt = self.depth2textcross_attn(
            query=depth_pos_embed,
            key=self.with_pos_embed(word_feat, word_pos).transpose(0, 1),
            value=word_feat.transpose(0, 1), key_padding_mask=word_key_padding_mask)[0]

        q = k = depth_pos_embed + depthfeat_adapt   # depth feature of second image
        text_cond_depth = self.depth2depth_attn(query=q, key=k, value=depth_pos_embed, key_padding_mask=mask_depth)[0]

        # adapted depth feature
        adapt_depth_feat = (self.norm_depth(depth_pos_embed.transpose(0, 1)) + self.norm_text_cond_depth(text_cond_depth.transpose(0, 1))) * verify_score_16.flatten(1).unsqueeze(-1)
        adapt_depth_feat = adapt_depth_feat.transpose(0, 1)
        return torch.cat([orig_multiscale_img_feat, adapt_img_feat], dim=-1), torch.cat([depth_pos_embed, adapt_depth_feat], dim=-1)


class Mono3DVGTransformer(nn.Module):
    def __init__(
            self,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            return_intermediate_dec=False,
            num_feature_levels=4,
            dec_n_points=4,
            enc_n_points=4,
    ):

        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = VisualEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points)   # d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
        self.encoder = VisualEncoder(encoder_layer, num_encoder_layers)

        self.TextGuidedAdapter = TextGuidedAdapter()

        decoder_layer = Mono3DVGDecoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points)
        self.decoder = Mono3DVGDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            
            lr = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            tb = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            wh = torch.cat((lr, tb), -1)

            proposal = torch.cat((grid, wh), -1).view(N_, -1, 6)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, depth_pos_embed=None,
                text_memory=None, text_mask=None,  im_name=None, instanceID=None, ann_id=None):
        assert query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)    #拼接 1  c  h  w  -->  1  h*w  c
            pos_embed = pos_embed.flatten(2).transpose(1, 2)# bs h*w c  [1, 7680, 256])
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)   #pos为毛再加一个level_embed

            mask = mask.flatten(1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)   #  1  a  dim  1  b  dim  abcd拼起来得到合起来的
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory, text_memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,text_memory, text_mask, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for adapter and decoder
        bs, _, c = memory.shape
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)

        # Initial target query
        tgt = torch.zeros_like(query_embed)

        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points

        depth_pos_embed = depth_pos_embed.flatten(2).permute(2, 0, 1)  # ([1, 256, 24, 80]) -->  ([1920, 1, 256])
        mask_depth = masks[1].flatten(1)

        # prepare for adapter  对编码后的图像特征、深度信息、文本信息进行融合和调整，得到适配后的图像特征和深度特征。
        img_feat_orig2adapt, depth_feat_orig2adapt = self.TextGuidedAdapter(memory, mask_flatten, lvl_pos_embed_flatten,
                reference_points, spatial_shapes,
                level_start_index,
                valid_ratios,
                text_memory, text_mask, depth_pos_embed, mask_depth, im_name, instanceID, ann_id)
#([1, 10200, 512])    ,,,,  rch.Size([1920, 1, 512])
        img_feat_srcs = img_feat_orig2adapt.chunk(2, dim=-1)#最后一个维度均分
        memory_adapt_k = img_feat_srcs[1]
        depth_feat_srcs = depth_feat_orig2adapt.chunk(2, dim=-1)
        depth_adapt_k = depth_feat_srcs[1]

        # decoder
        hs, inter_references, inter_references_dim = self.decoder(
            tgt, # A Learneable Query 
            reference_points,
            memory_adapt_k,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_embed,
            mask_flatten,
            depth_pos_embed,
            depth_adapt_k,
            mask_depth,
            text_memory, text_mask,)

        inter_references_out = inter_references
        inter_references_out_dim = inter_references_dim

        return hs, init_reference_out, inter_references_out, inter_references_out_dim, None, None


class VisualEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # Multi-Scale Deformable Attention MSDA
        self.msdeform_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        depth=1
        # cross attention MHCA
        self.cross_attn = MultiheadDiffAttn(d_model, n_heads, depth)
        #self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src
############Q做 图像特征 K做文本特征 V 是文本特征？？？？   不应该文本指导   文本做query  图像做key value吗
    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index ,text_memory, text_mask, padding_mask=None):
        # Multi-Scale Deformable Attention
        src2 = self.msdeform_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)#残差

        # cross attention
        src3 = self.cross_attn( self.with_pos_embed(src, pos).transpose(0, 1),
                                self.with_pos_embed(text_memory, torch.zeros_like(text_memory)).transpose(0, 1),
                                text_memory.transpose(0, 1),text_mask
                               )[0].transpose(0, 1)
        src3.unsqueeze(0)
        src = src + self.dropout4(src3)
        src = self.norm3(src)

        # ffn
        src = self.forward_ffn(src)
        return src, text_memory


class VisualEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))  #生成网课点的坐标  # torch.Size([24, 80])
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)  
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)      # xy
            reference_points_list.append(ref)   #torch.Size([1, 7680, 2])
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios,text_memory, text_mask,
                pos=None, padding_mask=None):
        output = src   #已经拼接好的特征
        text_output = text_memory
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output, text_output = layer(output, pos, reference_points, spatial_shapes, level_start_index,text_output, text_mask, padding_mask)

        return output, text_output   #


class Mono3DVGDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # Deformable attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        depth =1
        # text cross attention
        self.cross_attn_text = MultiheadDiffAttn(d_model, n_heads, depth)
        #self.cross_attn_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_text = nn.Dropout(dropout)
        self.norm_text = nn.LayerNorm(d_model)
        
        # depth cross attention
        self.cross_attn_depth = MultiheadDiffAttn(d_model, n_heads, depth)
        #self.cross_attn_depth = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_depth = nn.Dropout(dropout)
        self.norm_depth = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self,
                tgt,
                query_pos,
                reference_points,
                src,
                src_spatial_shapes,
                level_start_index,
                src_padding_mask,
                depth_pos_embed,
                depth_adapt_k,
                mask_depth,
                text_memory, text_mask):

        # Gather depth feats based on the text info q = tgt.transpose(0, 1) k = depth_adapt_k  v = depth_pos_embed
        tgt3 = self.cross_attn_depth(
                                    tgt.transpose(0, 1),
                                    depth_adapt_k,
                                    depth_pos_embed,
                                    key_padding_mask=mask_depth)[0].transpose(0, 1)   #????????????????????????????????????????????????????
        tgt2 = self.dropout_depth(tgt3).transpose(0, 1)
        tgt2 = self.norm_depth(tgt2)
        # Aggregate text info about the object
        tgt_text = self.cross_attn_text(self.with_pos_embed(tgt2, query_pos).transpose(0, 1),
                                        text_memory.transpose(0, 1),
                                        text_memory.transpose(0, 1),
                                        key_padding_mask=text_mask)[0].transpose(0, 1)
        tgt2 = self.dropout_text(tgt_text).transpose(0, 1)
        tgt2 = self.norm_text(tgt2)
        # Gather visual feats based on the linguistic info
        tgt3 = self.cross_attn(self.with_pos_embed(tgt2, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt2 = tgt + self.dropout1(tgt3)#1  1 256
        tgt = self.norm1(tgt2)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class Mono3DVGDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.dim_embed = None
        self.class_embed = None
        self.memory_tokens = None      
        config ={
                "batch_size": 1,
                "dataset_name": "fracturemnist3d",
                "all_dataset_name1": "organmnist3d(11), nodulemnist3d(2), fracturemnist3d(3)",
                "all_dataset_name2": "adrenalmnist3d(2), vesselmnist3d(2), synapsemnist3d(2), hmdb_dataset0/1/2",
                "root": "./datasets_data/medmnist",
                "all_root": "./datasets_data/medmnist  ./datasets_data/hmdb/dataset0 1 2",
                "log_dir": "./logs",
                "checkpoint_dir": "./checkpoints",
                "model": {
                    "model": "lmttm",
                    "_all_model_": "ttm, lmttm",
                    "drop_r": 0.2,
                    "preprocess_mode": "3d",
                    "all_preprocess_mode": "3d, 3dBN, resnet18",
                    "process_unit": "transformer",
                    "all_process_unit": "transformer, mixer,transformer",
                    "memory_mode": "TL",
                    "all_memory_mode": "TL-MHA, TL-AddErase, TL",
                    "in_channels": 1,
                    "dim": 256,
                    "memory_tokens_size": 128,
                    "num_blocks": 8,
                    "summerize_num_tokens": 1,
                    "step": 1,
                    "out_class_num": 3,
                    "patch_size": 1,
                    "Read_use_positional_embedding": True,
                    "Write_use_positional_embedding": True,
                    "load_memory_add_noise": True,
                    "load_memory_add_noise_mode": "normal",
                    "all_load_memory_add_noise_mode": "None, uniform, laplace, normal, exp, gamma, poisson"
                },
                "train": {
                    "gpu": "1",
                    "name": "test008",
                    "epoch": 300,
                    "optimizer": "Adam",
                    "lr": 0.0001,
                    "weight_decay": 0.0001,
                    "load_memory_tokens": False,
                    "input_H": 28,
                    "input_W": 28,
                    "val_gap": 100
                }
            }          
        self.TTM_model = TokenTuringMachineEncoder(config).cuda()

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, depth_pos_embed=None, depth_adapt_k=None, mask_depth=None,text_memory=None, text_mask=None,):
        output = tgt
        
        intermediate = []
        intermediate_reference_points = []
        intermediate_reference_dims = []

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 6:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]    

            output = layer(output,
                           query_pos,
                           reference_points_input,
                           src,
                           src_spatial_shapes,
                           src_level_start_index,
                           src_padding_mask,
                           depth_pos_embed,
                           depth_adapt_k,
                           mask_depth,
                           text_memory, text_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 6:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)    #   d对   xy 处理
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.dim_embed is not None:
                reference_dims = self.dim_embed[lid](output) #维度到3

            if self.return_intermediate:
                output = output.unsqueeze(2)
                output = output.cuda()
                output, self.memory_tokens = self.TTM_model(output, self.memory_tokens)
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_reference_dims.append(reference_dims)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(intermediate_reference_dims)


        return output, reference_points

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        if num_layers > 0:
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = []

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_mono3dvg_trans(cfg):
    return Mono3DVGTransformer(
        d_model=cfg['hidden_dim'],
        dropout=cfg['dropout'],
        activation="relu",
        nhead=cfg['nheads'],
        dim_feedforward=cfg['dim_feedforward'],
        num_encoder_layers=cfg['enc_layers'],
        num_decoder_layers=cfg['dec_layers'],
        return_intermediate_dec=cfg['return_intermediate_dec'],
        num_feature_levels=cfg['num_feature_levels'],
        dec_n_points=cfg['dec_n_points'],
        enc_n_points=cfg['enc_n_points'],
)

#  试验

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
    

# Copyright (c) 2023, Tri Dao.

from typing import Optional, Union

import torch

# import triton
# import triton.language as tl


# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 2}),
#         triton.Config({"BLOCK_M": 4}),
#         triton.Config({"BLOCK_M": 8}),
#         triton.Config({"BLOCK_M": 16}),
#     ],
#     key=["CACHE_KEY_SEQLEN", "BLOCK_K", "INTERLEAVED"],
# )
# @triton.jit
# def rotary_kernel(
#     OUT,  # Pointers to matrices
#     X,
#     COS,
#     SIN,
#     CU_SEQLENS,
#     SEQLEN_OFFSETS,  # this could be int or a pointer
#     # Matrix dimensions
#     seqlen,
#     nheads,
#     rotary_dim,
#     seqlen_ro,
#     CACHE_KEY_SEQLEN,
#     # strides
#     stride_out_batch,
#     stride_out_seqlen,
#     stride_out_nheads,
#     stride_out_headdim,
#     stride_x_batch,
#     stride_x_seqlen,
#     stride_x_nheads,
#     stride_x_headdim,
#     # Meta-parameters
#     BLOCK_K: tl.constexpr,
#     IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
#     IS_VARLEN: tl.constexpr,
#     INTERLEAVED: tl.constexpr,
#     CONJUGATE: tl.constexpr,
#     BLOCK_M: tl.constexpr,
# ):
#     pid_m = tl.program_id(axis=0)
#     pid_batch = tl.program_id(axis=1)
#     pid_head = tl.program_id(axis=2)
#     rotary_dim_half = rotary_dim // 2

#     if not IS_VARLEN:
#         X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
#         OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
#     else:
#         start_idx = tl.load(CU_SEQLENS + pid_batch)
#         seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
#         X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
#         OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

#     if pid_m * BLOCK_M >= seqlen:
#         return
#     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     if not IS_SEQLEN_OFFSETS_TENSOR:
#         rm_cs = rm + SEQLEN_OFFSETS
#     else:
#         rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
#     rk = tl.arange(0, BLOCK_K)
#     rk_half = tl.arange(0, BLOCK_K // 2)

#     if not INTERLEAVED:
#         # Load the 1st and 2nd halves of X, do calculation, then store to 1st and 2nd halves of OUT
#         X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
#         COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
#         SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
#         cos = tl.load(
#             COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0
#         ).to(tl.float32)
#         sin = tl.load(
#             SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0
#         ).to(tl.float32)
#         x0 = tl.load(
#             X, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0
#         ).to(tl.float32)
#         x1 = tl.load(
#             X + rotary_dim_half * stride_x_headdim,
#             mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
#             other=0.0,
#         ).to(tl.float32)
#         if CONJUGATE:
#             sin = -sin
#         o0 = x0 * cos - x1 * sin
#         o1 = x0 * sin + x1 * cos
#         # write back result
#         OUT = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim)
#         tl.store(OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
#         tl.store(
#             OUT + rotary_dim_half * stride_out_headdim,
#             o1,
#             mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
#         )
#     else:
#         # We don't want to load X[0, 2, 4, ...] and X[1, 3, 5, ...] separately since both are slow.
#         # Instead, we load x0 = X[0, 1, 2, 3, ...] and x1 = X[1, 0, 3, 2, ...].
#         # Loading x0 will be fast but x1 will be slow.
#         # Then we load cos = COS[0, 0, 1, 1, ...] and sin = SIN[0, 0, 1, 1, ...].
#         # Then we do the calculation and use tl.where to pick put the right outputs for the even
#         # and for the odd indices.
#         rk_swap = rk + ((rk + 1) % 2) * 2 - 1  # 1, 0, 3, 2, 5, 4, ...
#         rk_repeat = tl.arange(0, BLOCK_K) // 2
#         X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
#         X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
#         COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
#         SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
#         cos = tl.load(
#             COS,
#             mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
#             other=1.0,
#         ).to(tl.float32)
#         sin = tl.load(
#             SIN,
#             mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
#             other=0.0,
#         ).to(tl.float32)
#         x0 = tl.load(X0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0).to(
#             tl.float32
#         )
#         x1 = tl.load(
#             X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0
#         ).to(tl.float32)
#         if CONJUGATE:
#             sin = -sin
#         x0_cos = x0 * cos
#         x1_sin = x1 * sin
#         out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
#         OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
#         tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))


# def apply_rotary(
#     x: torch.Tensor,
#     cos: torch.Tensor,
#     sin: torch.Tensor,
#     seqlen_offsets: Union[int, torch.Tensor] = 0,
#     cu_seqlens: Optional[torch.Tensor] = None,
#     max_seqlen: Optional[int] = None,
#     interleaved=False,
#     inplace=False,
#     conjugate=False,
# ) -> torch.Tensor:
#     """
#     Arguments:
#         x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
#             else (total_seqlen, nheads, headdim).
#         cos: (seqlen_ro, rotary_dim / 2)
#         sin: (seqlen_ro, rotary_dim / 2)
#         seqlen_offsets: integer or integer tensor of size (batch,)
#         cu_seqlens: (batch + 1,) or None
#         max_seqlen: int
#     Returns:
#         y: (batch, seqlen, nheads, headdim)
#     """
#     is_varlen = cu_seqlens is not None
#     if not is_varlen:
#         batch, seqlen, nheads, headdim = x.shape
#     else:
#         assert max_seqlen is not None, "If cu_seqlens is passed in, then max_seqlen must be passed"
#         total_seqlen, nheads, headdim = x.shape
#         batch_p_1 = cu_seqlens.shape[0]
#         batch = batch_p_1 - 1
#         seqlen = max_seqlen
#     seqlen_ro, rotary_dim = cos.shape
#     assert sin.shape == cos.shape
#     rotary_dim *= 2
#     assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
#     assert headdim <= 256, "Only support headdim <= 256"
#     assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

#     assert (
#         cos.dtype == sin.dtype
#     ), f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
#     assert (
#         x.dtype == cos.dtype
#     ), f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"

#     cos, sin = cos.contiguous(), sin.contiguous()
#     if isinstance(seqlen_offsets, torch.Tensor):
#         assert seqlen_offsets.shape == (batch,)
#         assert seqlen_offsets.dtype in [torch.int32, torch.int64]
#         seqlen_offsets = seqlen_offsets.contiguous()
#     else:
#         assert seqlen_offsets + seqlen <= seqlen_ro

#     output = torch.empty_like(x) if not inplace else x
#     if rotary_dim < headdim and not inplace:
#         output[..., rotary_dim:].copy_(x[..., rotary_dim:])

#     BLOCK_K = (
#         32
#         if rotary_dim <= 32
#         else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
#     )
#     grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)  # noqa
#     BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)

#     # Need this, otherwise Triton tries to launch from cuda:0 and we get
#     # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
#     with torch.cuda.device(x.device.index):
#         rotary_kernel[grid](
#             output,  # data ptrs
#             x,
#             cos,
#             sin,
#             cu_seqlens,
#             seqlen_offsets,
#             seqlen,  # shapes
#             nheads,
#             rotary_dim,
#             seqlen_ro,
#             seqlen // 128,  # key for triton cache (limit number of compilations)
#             output.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
#             output.stride(-3),  # seqlen_stride or total_seqlen_stride
#             output.stride(-2),  # nheads_stride
#             output.stride(-1),  # headdim_stride
#             x.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
#             x.stride(-3),  # seqlen stride or total_seqlen_stride
#             x.stride(-2),  # nheads stride
#             x.stride(-1),  # headdim stride
#             BLOCK_K,
#             isinstance(seqlen_offsets, torch.Tensor),
#             is_varlen,
#             interleaved,
#             conjugate,
#             BLOCK_M,
#         )
#     return output

# class ApplyRotaryEmb(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         x,
#         cos,
#         sin,
#         interleaved=False,
#         inplace=False,
#         seqlen_offsets: Union[int, torch.Tensor] = 0,
#         cu_seqlens: Optional[torch.Tensor] = None,
#         max_seqlen: Optional[int] = None,
#     ):
#         out = apply_rotary(
#             x,
#             cos,
#             sin,
#             seqlen_offsets=seqlen_offsets,
#             cu_seqlens=cu_seqlens,
#             max_seqlen=max_seqlen,
#             interleaved=interleaved,
#             inplace=inplace,
#         )
#         if isinstance(seqlen_offsets, int):
#             # Can't save int with save_for_backward
#             ctx.save_for_backward(cos, sin, cu_seqlens)
#             ctx.seqlen_offsets = seqlen_offsets
#         else:
#             ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
#             ctx.seqlen_offsets = None
#         ctx.interleaved = interleaved
#         ctx.inplace = inplace
#         ctx.max_seqlen = max_seqlen
#         return out if not inplace else x

#     @staticmethod
#     def backward(ctx, do):
#         seqlen_offsets = ctx.seqlen_offsets
#         if seqlen_offsets is None:
#             cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
#         else:
#             cos, sin, cu_seqlens = ctx.saved_tensors
#         # TD [2023-09-02]: For some reason Triton (2.0.0.post1) errors with
#         # "[CUDA]: invalid device context", and cloning makes it work. Idk why. Triton 2.1.0 works.
#         if not ctx.interleaved and not ctx.inplace:
#             do = do.clone()
#         dx = apply_rotary(
#             do,
#             cos,
#             sin,
#             seqlen_offsets=seqlen_offsets,
#             cu_seqlens=cu_seqlens,
#             max_seqlen=ctx.max_seqlen,
#             interleaved=ctx.interleaved,
#             inplace=ctx.inplace,
#             conjugate=True,
#         )
#         return dx, None, None, None, None, None, None, None


# def apply_rotary_emb(
#     x,
#     cos,
#     sin,
#     interleaved=False,
#     inplace=False,
#     seqlen_offsets: Union[int, torch.Tensor] = 0,
#     cu_seqlens: Optional[torch.Tensor] = None,
#     max_seqlen: Optional[int] = None,
# ):
#     """
#     Arguments:
#         x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
#             else (total_seqlen, nheads, headdim)
#         cos, sin: (seqlen_rotary, rotary_dim / 2)
#         interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
#             of 1st half and 2nd half (GPT-NeoX style).
#         inplace: if True, apply rotary embedding in-place.
#         seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
#             Most commonly used in inference when we have KV cache.
#         cu_seqlens: (batch + 1,) or None
#         max_seqlen: int
#     Return:
#         out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
#             else (total_seqlen, nheads, headdim)
#     rotary_dim must be <= headdim
#     Apply rotary embedding to the first rotary_dim of x.
#     """
#     return ApplyRotaryEmb.apply(
#         x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
#     )

import math
import torch
import torch.nn.functional as F
from torch import nn

# from .kernel.rotary import apply_rotary_emb
# try:
#     from apex.normalization import FusedRMSNorm as RMSNorm 
# except ModuleNotFoundError:
#     print("No fused RMSNorm")
#     from .rms_norm import RMSNorm


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        depth   
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        depth = 1
        # arg num_heads set to half of Transformer's num_heads
        self.num_heads = num_heads
        
        # arg decoder_kv_attention_heads set to half of Transformer's num_kv_heads if use GQA
        # set to same as num_heads if use normal MHA
        self.num_kv_heads = num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
         
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
    def forward(
        self,
        q,
        k,
        v,
        key_padding_mask,
    ):
        bsz, tgt_len, embed_dim = q.size()
        src_len = tgt_len
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        src_len = k.size(0)   
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        # q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        # k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if key_padding_mask is None:
            key_padding_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        # src_len = k.size(1)
        # key_padding_mask = (
        #         torch.zeros_like(key_padding_mask, dtype=torch.float64)
        #         .masked_fill_(key_padding_mask, float("-inf"))
        #     )
        # key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
        #     expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += key_padding_mask   
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        return attn

