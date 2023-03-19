# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.utils.checkpoint as checkpoint
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from ...util.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn

from .vlfusion import VLFuse, BertEncoderLayer
from transformers.models.bert.modeling_bert import BertConfig
from .fuse_helper import BiMultiHeadAttention
from einops import repeat

def agg_lang_feat(features, mask, pool_type="average"):
    """average pooling of language features"""
    # feat: (bs, seq_len, C)
    # mask: (bs, seq_len)
    if pool_type == "average":
        embedded = features * mask.unsqueeze(-1).float() # use mask to zero out invalid token features
        aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())
    elif pool_type == "max":
        out = []
        for i in range(len(features)):
            pool_feat, _ = torch.max(features[i][mask[i]], 0) # (L, C) -> (C, )
            out.append(pool_feat)
        aggregate = torch.stack(out, dim=0) # (bs, C)
    else:
        raise ValueError("pool_type should be average or max")
    return aggregate


# DeformableTransformerVL is designed for language-guided object detection 
# Try formulation in DAB-DETR (anchor -> pos_embed)
# Try DINO architecture (denoising mechanism)
class DeformableTransformerVLDINO(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, 
                 look_forward_twice=False, mixed_selection=False,
                 use_checkpoint=False, cfg=None):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        # for language-guided object detection, vl_fusion layer & language-encoder layer are added.
        vl_fusion_layer = VLFuse(cfg) if cfg.MODEL.USE_EARLY_FUSION else nn.Identity()
        if cfg.MODEL.USE_ADDITIONAL_BERT:
            lang_cfg = BertConfig.from_pretrained("projects/UNINEXT/bert-base-uncased")
            lang_encoder_layer = BertEncoderLayer(lang_cfg, \
                clamp_min_for_underflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW, # True
                clamp_max_for_overflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW)
        else: 
            lang_encoder_layer = nn.Identity()
        num_vl_layers = getattr(cfg.MODEL.DDETRS, "NUM_VL_LAYERS", None)
        self.encoder = DeformableTransformerEncoderVL(vl_fusion_layer, encoder_layer, lang_encoder_layer, num_encoder_layers, use_checkpoint, num_vl_layers=num_vl_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(d_model, decoder_layer, num_decoder_layers, return_intermediate_dec, look_forward_twice, use_checkpoint)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.tgt_embed = nn.Embedding(self.two_stage_num_proposals, d_model)
        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
        else:
            self.reference_points = nn.Linear(d_model, 2)
        
        self.mixed_selection = mixed_selection
        self._reset_parameters()
        self.resizer = FeatureResizer(
                input_feat_size=768,
                output_feat_size=d_model, # 256
                dropout=0.1
            )
        # decoupled designs
        self.decouple_tgt = cfg.MODEL.DECOUPLE_TGT
        self.still_tgt_for_both = cfg.MODEL.STILL_TGT_FOR_BOTH

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
            if isinstance(m, BiMultiHeadAttention):
                m._reset_parameters()
        if not self.two_stage:
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
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
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

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, srcs, masks, pos_embeds, query_embed=None, mask_on=False, language_dict_features=None, task=None, 
    attn_masks=None, return_src_info=False):
        """ref_feat: reference features for visual grounding, SOT and VOS.
        ref_feat: (bs, C) """
        assert language_dict_features is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2) # (bs, C, H, W) -> (bs, HW, C)
            mask = mask.flatten(1) # (bs, H, W)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1) # (bs, sigma(HW), C)
        mask_flatten = torch.cat(mask_flatten, 1) # (bs, sigma(HW))
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder & VL early fusion
        vl_feats_dict = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, language_dict_features, task=task)
        memory, language_dict_features = vl_feats_dict["visual"], vl_feats_dict["lang"]

        # use language feature after early fusion as ref_feat
        lang_feat_pool = agg_lang_feat(language_dict_features["hidden"], language_dict_features["masks"]) # (bs, 768)
        ref_feat = self.resizer(lang_feat_pool) # (bs, 256)
        two_stage_num_proposals = self.two_stage_num_proposals
        ref_feat = ref_feat.unsqueeze(1) # (bs, 1, 256)

        # prepare input for decoder
        bs, _, c = memory.shape
        assert self.two_stage
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory, lang_feat_pool.unsqueeze(1))
        enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        topk = two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
        topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        reference_points = topk_coords_unact.sigmoid()
        # special processing for DN
        if query_embed[1] is not None:
            reference_points = torch.cat([query_embed[1].sigmoid(), reference_points], 1)
        init_reference_out = reference_points
        # mixed query selection
        tgt = self.tgt_embed.weight[None].repeat(bs, 1, 1)
        # special design for DN
        if query_embed[0] is not None:
            tgt = torch.cat([query_embed[0], tgt], 1)
        # VL Fusion (query-level)
        if ref_feat is not None:
            if self.decouple_tgt:
                if self.still_tgt_for_both:
                    tgt_new = tgt + 0.0 * ref_feat # both use the original tgt
                else:
                    if task == "detection":
                        tgt_new = tgt + 0.0 * ref_feat # "+ 0.0 *" is to avoid unused parameters
                    elif task == "grounding":
                        tgt_new = ref_feat + 0.0 * tgt # "+ 0.0 *" is to avoid unused parameters
                    else:
                        raise ValueError("task should be detection or grounding")
            else:
                if query_embed[0] is None:
                    tgt_new = ref_feat.repeat(1, self.two_stage_num_proposals, 1)
                else:
                    tgt_new = torch.cat([query_embed[0], ref_feat.repeat(1, self.two_stage_num_proposals, 1)], 1)
                # avoid unused parameters
                tgt_new += 0.0 * torch.sum(self.tgt_embed.weight)

        # decoder
        # for DAB, query_pos should be None
        hs, inter_references = self.decoder(tgt_new, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, 
                                            query_pos=None, src_padding_mask=mask_flatten,
                                            attn_masks=attn_masks)

        inter_references_out = inter_references

        if mask_on:
            if return_src_info:
                src_info_dict = {
                    "src": memory.detach(),
                    "src_spatial_shapes": spatial_shapes,
                    "src_level_start_index": level_start_index,
                    "src_valid_ratios": valid_ratios,
                    "src_padding_mask": mask_flatten}
                return hs, memory, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact, language_dict_features, src_info_dict
            else:
                return hs, memory, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact, language_dict_features
        else:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact, language_dict_features

# DeformableTransformerEncoderVL is designed for language-guided object detection 
class DeformableTransformerEncoderVL(nn.Module):
    def __init__(self, vl_fusion_layer, encoder_layer, lang_encoder_layer, num_layers, use_checkpoint=False, num_vl_layers=None):
        super().__init__()
        num_vl_layers = num_layers if num_vl_layers is None else num_vl_layers
        self.vl_layers = _get_clones_advanced(vl_fusion_layer, num_layers, num_vl_layers)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.lang_layers = _get_clones(lang_encoder_layer, num_layers)
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None, language_dict_features=None, task=None):
        # output = src
        output = {"visual": src, "lang": language_dict_features}
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, (vl_layer, layer, lang_layer) in enumerate(zip(self.vl_layers, self.layers, self.lang_layers)):
            if self.use_checkpoint:
                raise ValueError("NOT supported for now")
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    padding_mask,
                )
            else:
                if isinstance(vl_layer, nn.Identity):
                    output = vl_layer(output)
                else:
                    output = vl_layer(output, task=task)
                output["visual"] = layer(output["visual"], pos, reference_points, spatial_shapes, level_start_index, padding_mask)
                output = lang_layer(output)
        return output


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

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

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

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

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, 
    src_padding_mask=None, attn_masks=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=attn_masks)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, embed_dim, decoder_layer, num_layers, return_intermediate=False, look_forward_twice=False, use_checkpoint=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate # True
        self.look_forward_twice = look_forward_twice
        self.use_checkpoint = use_checkpoint
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)
        self.bbox_embed = None
        self.class_embed = None

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, attn_masks=None):
        output = tgt
        bs, num_queries, _ = output.size()
        if reference_points.dim() == 2:
            reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)  # bs, num_queries, 4
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            if self.use_checkpoint:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    query_pos,
                    reference_points_input,
                    src,
                    src_spatial_shapes,
                    src_level_start_index,
                    src_padding_mask,
                    attn_masks
                )
            else:
                output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, attn_masks)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach() # new reference points in the next decoder layer

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(
                    new_reference_points
                    if self.look_forward_twice
                    else reference_points
                )

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points

# simplify DeformableTransformerDecoder as ReID head
class DeformableReidHead(nn.Module):
    def __init__(self, embed_dim, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, attn_masks=None):
        output = tgt
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                raise ValueError("reference_points.shape[-1] should be 4")

            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, attn_masks)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_clones_advanced(module, N, N_valid):
    assert N_valid <= N
    layers = []
    for i in range(N):
        if i < N_valid:
            layers.append(copy.deepcopy(module))
        else:
            layers.append(nn.Identity())
    return nn.ModuleList(layers)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


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
    """The implementation of simple multi-layer perceptron layer
    without dropout and identity connection.

    The feature process order follows `Linear -> ReLU -> Linear -> ReLU -> ...`.

    Args:
        input_dim (int): The input feature dimension.
        hidden_dim (int): The hidden dimension of MLPs.
        output_dim (int): the output feature dimension of MLPs.
        num_layer (int): The number of FC layer used in MLPs.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> torch.Tensor:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        """Forward function of `MLP`.

        Args:
            x (torch.Tensor): the input tensor used in `MLP` layers.

        Returns:
            torch.Tensor: the forward results of `MLP` layer
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
) -> torch.Tensor:
    """generate sine position embedding from a position tensor

    Args:
        pos_tensor (torch.Tensor): Shape as `(None, n)`.
        num_pos_feats (int): projected shape for each float in the tensor. Default: 128
        temperature (int): The temperature used for scaling
            the position embedding. Default: 10000.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is `[x, y]`, the results will  # noqa 
            be `[pos(y), pos(x)]`. Defaults: True.

    Returns:
        torch.Tensor: Returned position embedding  # noqa 
        with shape `(None, n * num_pos_feats)`.
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[:, :, 0::2].sin(), sin_x[:, :, 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=2)
    return pos_res
