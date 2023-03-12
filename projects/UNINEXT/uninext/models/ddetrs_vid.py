# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ..util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list, inverse_sigmoid
from .deformable_detr.deformable_transformer import agg_lang_feat
from .ddetrs import DDETRSegmUni, MLP
from .pos_neg_select import select_pos_neg
import math

# Deformable DETR + Segmentaion (CondInst) + reid head for Video
class DDETRSegmUniVID(DDETRSegmUni):
    def __init__(self, detr, rel_coord=True, ota=False, new_mask_head=False, use_raft=False, mask_out_stride=4, \
        template_sz=256, extra_backbone_for_template=False, search_area_factor=2,
        ref_feat_sz=8, sot_feat_fusion=False, use_iou_branch=False, decouple_tgt=False, cfg=None):
        super().__init__(detr, rel_coord=rel_coord, ota=ota, new_mask_head=new_mask_head, use_raft=use_raft, mask_out_stride=mask_out_stride, 
        use_iou_branch=use_iou_branch, decouple_tgt=decouple_tgt)
        hidden_dim = detr.transformer.d_model
        self.reid_embed_head = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        # adjust dim
        self.adjust_layer = nn.Linear(hidden_dim, 768)
        self.template_sz = template_sz
        self.extra_backbone_for_template = extra_backbone_for_template # use an extra backbone extracting features for the template
        self.search_area_factor = search_area_factor
        self.ref_feat_sz = ref_feat_sz
        self.sot_feat_fusion = sot_feat_fusion
        if self.sot_feat_fusion:
            self.sot_fuser = FeatureFuser(hidden_dim, hidden_dim)

    def get_template(self, img, mask, bbox):
        """img: (1, 3, H, W), mask: (1, 1, H, W), bbox: (1, 4)"""
        assert len(bbox) == 1
        x, y, w, h = bbox[0].tolist()

        crop_sz = math.ceil(math.sqrt(w * h) * self.search_area_factor)
        x1 = round(x + 0.5 * w - crop_sz * 0.5)
        x2 = x1 + crop_sz
        y1 = round(y + 0.5 * h - crop_sz * 0.5)
        y2 = y1 + crop_sz

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - img.shape[-1] + 1, 0)
        y1_pad = max(0, -y1)
        y2_pad = max(y2 - img.shape[-2] + 1, 0)

        # Crop target
        im_crop = img[:, :, y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]
        mask_crop = mask[:, :, y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

        # Pad
        im_crop_padded = F.pad(im_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=1).float()

        # resize
        im_crop_padded = F.interpolate(im_crop_padded, (self.template_sz, self.template_sz), mode='bilinear', align_corners=False)
        mask_crop_padded = F.interpolate(mask_crop_padded, (self.template_sz, self.template_sz), mode='bilinear', align_corners=False)[0].bool()
        return im_crop_padded, mask_crop_padded

    def get_template_4c(self, img, pad_mask, bbox, gt_mask=None):
        """img: (1, 3, H, W), mask: (1, 1, H, W), bbox: (1, 4)"""
        """get 4-channel template"""
        assert len(bbox) == 1
        x, y, w, h = bbox[0].tolist()

        crop_sz = math.ceil(math.sqrt(w * h) * self.search_area_factor)
        x1 = round(x + 0.5 * w - crop_sz * 0.5)
        x2 = x1 + crop_sz
        y1 = round(y + 0.5 * h - crop_sz * 0.5)
        y2 = y1 + crop_sz

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - img.shape[-1] + 1, 0)
        y1_pad = max(0, -y1)
        y2_pad = max(y2 - img.shape[-2] + 1, 0)

        # Crop target
        im_crop = img[:, :, y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]
        mask_crop = pad_mask[:, :, y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]
        if gt_mask is None:
            mask_crop_gt = torch.zeros_like(mask_crop)
        else:
            mask_crop_gt = gt_mask[:, :, y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]
        # Pad
        im_crop_padded = F.pad(im_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=1).float()
        mask_crop_padded_gt = F.pad(mask_crop_gt, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0).float()

        if gt_mask is None:
            # target box coordinates on the cropped image patch
            x1_t = round(x - x1)
            x2_t = round(x1_t + w)
            y1_t = round(y - y1)
            y2_t = round(y1_t + h)
            # set regions inside the target box as 1
            mask_crop_padded_gt[:, :, max(0, y1_t):y2_t, max(0, x1_t):x2_t] = 1

        # resize
        im_crop_padded = F.interpolate(im_crop_padded, (self.template_sz, self.template_sz), mode='bilinear', align_corners=False)
        mask_crop_padded = F.interpolate(mask_crop_padded, (self.template_sz, self.template_sz), mode='bilinear', align_corners=False)[0].bool()
        mask_crop_padded_gt = F.interpolate(mask_crop_padded_gt, (self.template_sz, self.template_sz), mode='bilinear', align_corners=False)
        im_crop_padded_4c = torch.cat([im_crop_padded, mask_crop_padded_gt], dim=1)
        
        return im_crop_padded_4c, mask_crop_padded

    def debug_template(self, samples):
        import numpy as np
        import cv2
        import torch.distributed as dist
        mean = np.array([123.675, 116.280, 103.530])
        std = np.array([58.395, 57.120, 57.375])
        for i in range(len(samples.tensors)):
            image = samples.tensors[i].permute((1, 2, 0)).cpu().numpy() * std + mean # (H, W, 3)
            input_mask = samples.mask[i].float().cpu().numpy() * 255 # (H, W)
            image = np.ascontiguousarray(image[:, :, ::-1]).clip(0, 255)
            cv2.imwrite("rank_%02d_batch_%d_template_img.jpg"%(dist.get_rank(), i), image)
            cv2.imwrite("rank_%02d_batch_%d_template_mask.jpg"%(dist.get_rank(), i), input_mask)
        import sys
        sys.exit(0)

    def debug_template_4c(self, samples):
        import numpy as np
        import cv2
        import torch.distributed as dist
        mean = np.array([123.675, 116.280, 103.530])
        std = np.array([58.395, 57.120, 57.375])
        for i in range(len(samples.tensors)):
            image_mask = samples.tensors[i].permute((1, 2, 0)).cpu().numpy() # (H, W, 4)
            image = image_mask[:, :, :3]
            image = image * std + mean # (H, W, 3)
            gt_mask = image_mask[:, :, -1] # (H, W)
            input_mask = samples.mask[i].float().cpu().numpy() * 255 # (H, W)
            image = np.ascontiguousarray(image[:, :, ::-1]).clip(0, 255)
            image[:, :, -1] = np.clip(image[:, :, -1] + 100 * gt_mask, 0, 255)
            cv2.imwrite("rank_%02d_batch_%d_template_img.jpg"%(dist.get_rank(), i), image)
            cv2.imwrite("rank_%02d_batch_%d_template_mask.jpg"%(dist.get_rank(), i), input_mask)
        import sys
        sys.exit(0)

    def forward_backbone(self, samples, template_branch=False):
        if template_branch and self.extra_backbone_for_template:
            features, pos = self.detr.ref_backbone(samples)
        else:
            features, pos = self.detr.backbone(samples)
        
        srcs = []
        masks = []
        poses = []
        spatial_shapes = []

        for l, feat in enumerate(features):
            # src: [N, _C, Hi, Wi],
            # mask: [N, Hi, Wi],
            # pos: [N, C, H_p, W_p]
            src, mask = feat.decompose() 
            src_proj_l = self.detr.input_proj[l](src)    # src_proj_l: [N, C, Hi, Wi]
            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos[l])
            n, c, h, w = src_proj_l.shape
            spatial_shapes.append((h, w))

        if self.detr.num_feature_levels > len(features):
            _len_srcs = len(features)
            for l in range(_len_srcs, self.detr.num_feature_levels):
                if l == _len_srcs:
                    src = self.detr.input_proj[l](features[-1].tensors)
                else:
                    src = self.detr.input_proj[l](srcs[-1])
                m = masks[0]   # [N, H, W]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)
                n, c, h, w = src.shape
                spatial_shapes.append((h, w))

        return srcs, masks, poses, spatial_shapes


    def coco_forward_sot(self, samples, det_targets, ref_targets, criterion, train=False, task=None):
        # language_dict_features = {"hidden": (bs, L, 768), "masks": (bs, L)}
        image_sizes = samples.image_sizes
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=32)
        # samples.tensors: (2bs, 3, H, W)
        # samples.mask: (2bs, H, W)

        # parse reference and key
        bz = samples.tensors.shape[0]//2
        key_ids = list(range(0, bz*2, 2))
        ref_ids = list(range(1, bz*2, 2))

        # get key-frame features
        samples_key = NestedTensor(samples.tensors[key_ids], samples.mask[key_ids])
        srcs_key, masks_key, poses_key, spatial_shapes = self.forward_backbone(samples_key)

        # forward template to get language_dict_features_key  
        tensors_ref = samples.tensors[ref_ids] # (bs, 3, H, W)
        mask_ref = samples.mask[ref_ids] # (bs, H, W)
        ref_bboxes = [x["bboxes_unorm"] for x in ref_targets] # List (len is bs)
        ref_tensor_list, ref_mask_list = [], []
        for batch_idx in range(bz):
            img = tensors_ref[batch_idx:batch_idx+1] # (1, 3, H, W)
            mask = mask_ref[batch_idx:batch_idx+1][None] # (1, H, W) -> (1, 1, H, W)
            if "masks" in ref_targets[batch_idx]:
                ref_gt_mask = ref_targets[batch_idx]["masks"][None] # (1, 1, H, W)
            else:
                ref_gt_mask = None
            bbox = ref_bboxes[batch_idx] # (1, 4)
            bbox[:, 2:] = bbox[:, 2:] - bbox[:, :2]
            if self.extra_backbone_for_template:
                template_tensor, template_mask = self.get_template_4c(img, mask, bbox, gt_mask=ref_gt_mask)
            else:
                template_tensor, template_mask = self.get_template(img, mask, bbox)
            ref_tensor_list.append(template_tensor)
            ref_mask_list.append(template_mask)
        ref_tensor = torch.cat(ref_tensor_list, dim=0)
        ref_mask = torch.cat(ref_mask_list, dim=0)
        samples_ref = NestedTensor(ref_tensor, ref_mask)
        srcs_ref, masks_ref, _, _ = self.forward_backbone(samples_ref, template_branch=True)

        language_dict_features_key = {}
        if self.sot_feat_fusion:
            ref_feats = self.sot_fuser(srcs_ref).flatten(-2).permute(0, 2, 1) # (bs, L, C)
            ref_masks = masks_ref[0].flatten(-2) # (bs, L)
        else:
            ref_feats, ref_masks = [], []
            for n_l in range(self.detr.num_feature_levels):
                ref_feat_l = F.interpolate(srcs_ref[n_l], size=(self.ref_feat_sz, self.ref_feat_sz)) # (bs, C, 8, 8)
                ref_feats.append(ref_feat_l.flatten(-2)) # (bs, C, 64)
                ref_mask_l = F.interpolate(masks_ref[n_l][None].float(), size=(self.ref_feat_sz, self.ref_feat_sz))[0].bool() # (bs, 8, 8)
                ref_masks.append(ref_mask_l.flatten(-2)) # (bs, 64)
            ref_feats = torch.cat(ref_feats, dim=-1).permute(0, 2, 1) # (bs, C, L) L=256 --> (bs, L, C)
            ref_masks = torch.cat(ref_masks, dim=-1) # (bs, L)
        # original masks need to be reversed as text_masks!
        language_dict_features_key["hidden"], language_dict_features_key["masks"] = self.adjust_layer(ref_feats), ~ref_masks

        if not self.detr.two_stage or self.decouple_tgt:
            query_embeds = self.detr.query_embed.weight
        else:
            query_embeds = None

        image_sizes = [image_sizes[_i] for _i in key_ids]


        # ref frames are not used for detection
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, language_dict_features_key = self.detr.transformer(srcs_key, masks_key, poses_key, query_embeds, mask_on=True, language_dict_features=language_dict_features_key, task=task)

        if task == "grounding" or task == "sot":
            lang_feat_pool_key = agg_lang_feat(language_dict_features_key["hidden"], language_dict_features_key["masks"]).unsqueeze(1) # (bs, 1, 768)
            # lang_feat_pool_ref = agg_lang_feat(language_dict_features_ref["hidden"], language_dict_features_ref["masks"]).unsqueeze(1) # (bs, 1, 768)
        elif task == "detection":
            pass
        else:
            raise ValueError("task must be detection or grounding")
        # memory: [N, \sigma(HiWi), C]
        # hs: [num_encoders, N, num_querries, C]

        outputs = {}
        outputs_classes = []
        outputs_coords = []
        outputs_masks = []
        indices_list = []
        if self.use_iou_branch:
            outputs_ious = []
        enc_lay_num = hs.shape[0]
        
        for lvl in range(enc_lay_num):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            if task == "grounding" or task == "sot":
                outputs_class = self.detr.class_embed[lvl](hs[lvl], lang_feat_pool_key)
            elif task == "detection":
                outputs_class = self.detr.class_embed[lvl](hs[lvl], language_dict_features_key["hidden"])
            else:
                raise ValueError("task must be detection, grounding or sot")
            tmp = self.detr.bbox_embed[lvl](hs[lvl])
            if self.use_iou_branch:
                pred_iou = self.detr.iou_head[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            if self.use_iou_branch:
                outputs_ious.append(pred_iou)
            outputs_layer = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
            dynamic_mask_head_params = self.controller(hs[lvl])    # [bs, num_quries, num_params]

            # for training & log evaluation loss
            if self.ota:
                indices, matched_ids = criterion.matcher.forward_ota(outputs_layer, det_targets)
            else:
                indices = criterion.matcher.forward(outputs_layer, det_targets)
            indices_list.append(indices)
            reference_points, mask_head_params, num_insts = [], [], []
            for i, indice in enumerate(indices):
                pred_i, tgt_j = indice
                if self.ota:
                    num_insts.append(len(pred_i))
                else:
                    num_insts.append(len(pred_i))
                mask_head_params.append(dynamic_mask_head_params[i, pred_i].unsqueeze(0))

                # This is the image size after data augmentation (so as the gt boxes & masks)
                
                orig_h, orig_w = image_sizes[i]
                orig_h = torch.as_tensor(orig_h).to(reference)
                orig_w = torch.as_tensor(orig_w).to(reference)
                scale_f = torch.stack([orig_w, orig_h], dim=0)
                
                ref_cur_f = reference[i].sigmoid()
                ref_cur_f = ref_cur_f[..., :2]
                ref_cur_f = ref_cur_f * scale_f[None, :] 
                reference_points.append(ref_cur_f[pred_i].unsqueeze(0))
            # reference_points: [1, nf,  \sum{selected_insts}, 2]
            # mask_head_params: [1, \sum{selected_insts}, num_params]
            reference_points = torch.cat(reference_points, dim=1)
            mask_head_params = torch.cat(mask_head_params, dim=1)
 
            # mask prediction
            has_mask_list = ["masks" in x.keys() for x in det_targets]
            assert len(set(has_mask_list)) == 1 # must be "all True" or "all False"
            if has_mask_list[0]:
                outputs_layer = self.forward_mask_head_train(outputs_layer, memory, spatial_shapes, 
                                                            reference_points, mask_head_params, num_insts)
            else:
                # avoid unused parameters
                dummy_output = torch.sum(mask_head_params)
                for p in self.mask_head.parameters():
                    dummy_output += p.sum()
                outputs_layer['pred_masks'] = 0.0 * dummy_output
            outputs_masks.append(outputs_layer['pred_masks'])
            
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_mask = outputs_masks


        # ref_cls = self.detr.class_embed[enc_lay_num-1](hs_ref[-1], language_dict_features_ref["hidden"]).sigmoid() # NOTE: self.detr.class_embed[-1] may cause conflict with two-stage !
        # # Only compute Embedding Loss on the last layer of decoder
        # contrast_items = select_pos_neg(inter_references_ref[-1], matched_ids, ref_targets, det_targets, self.reid_embed_head, hs[-1], hs_ref[-1], ref_cls)

        # outputs['pred_samples'] = inter_samples[-1]
        
        outputs['pred_logits'] = outputs_class[-1]
        outputs['pred_boxes'] = outputs_coord[-1]
        outputs['pred_masks'] = outputs_mask[-1]
        if self.use_iou_branch:
            outputs_iou = torch.stack(outputs_ious)
            outputs['pred_boxious'] = outputs_iou[-1]
        # outputs['pred_qd'] = contrast_items
        # reid_params = 0
        # for p in self.reid_embed_head.parameters():
        #     reid_params += 0.0 * p.sum() # avoid unused parameters
        # outputs['reid_params'] = reid_params
        if task == "grounding" or "sot":
            bs, device = language_dict_features_key["masks"].size(0), language_dict_features_key["masks"].device
            text_masks = torch.ones((bs, 1), dtype=torch.bool, device=device)
        elif task == "detection":
            text_masks = language_dict_features_key["masks"]
        else:
            raise ValueError("task must be detection or grounding")
        outputs["text_masks"] = text_masks
        if self.detr.aux_loss:
            if self.use_iou_branch:
                outputs['aux_outputs'] = self._set_aux_loss_with_iou(outputs_class, outputs_coord, outputs_mask, outputs_iou)
            else:
                outputs['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_mask)
            for x in outputs['aux_outputs']:
                x["text_masks"] = text_masks
        if self.detr.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            outputs['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord, "text_masks": text_masks}

        # skip reid loss for simplicity
        outputs['pred_qd'] = []
        reid_params = 0
        for p in self.reid_embed_head.parameters():
            reid_params += 0.0 * p.sum() # avoid unused parameters
        outputs['reid_params'] = reid_params

        loss_dict = criterion(outputs, det_targets, indices_list)
        # fix unused parameters of the mask head
        no_valid_obj = False
        if isinstance(outputs['pred_masks'], list):
            if len(outputs['pred_masks']) == 0:
                no_valid_obj = True
        if no_valid_obj:
            loss_mask = loss_dict["loss_mask"]
            for n, p in self.mask_head.named_parameters():
                loss_mask += 0.0 * torch.sum(p)
            for n, p in self.controller.named_parameters():
                loss_mask += 0.0 * torch.sum(p)
            loss_dict["loss_mask"] = loss_mask
            loss_dict["loss_dice"] = loss_mask
            for i in range(enc_lay_num-1):
                loss_dict["loss_mask_%d"%i] = loss_mask
                loss_dict["loss_dice_%d"%i] = loss_mask
        return outputs, loss_dict

    def debug(self, samples, gt_targets):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=32)
        self.debug_data(samples, gt_targets)

    # for SOT (initialize on the 1st frame)
    def coco_inference_ref(self, samples, gt_targets):
        if not isinstance(samples, NestedTensor):
            size_divisibility = getattr(self.detr.backbone[0].backbone, "size_divisibility", 32)
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=size_divisibility)

        bz = samples.tensors.shape[0]
        assert bz == 1
        # forward template to get language_dict_features_key  
        tensors_ref = samples.tensors # (bs, 3, H, W)
        mask_ref = samples.mask # (bs, H, W)
        ref_bboxes = [x["bboxes_unorm"] for x in gt_targets] # List (len is bs)
        ref_tensor_list, ref_mask_list = [], []
        for batch_idx in range(bz):
            img = tensors_ref[batch_idx:batch_idx+1] # (1, 3, H, W)
            mask = mask_ref[batch_idx:batch_idx+1][None] # (1, H, W) -> (1, 1, H, W)
            bbox = ref_bboxes[batch_idx].clone() # (1, 4)
            bbox[:, 2:] = bbox[:, 2:] - bbox[:, :2]
            if "masks" in gt_targets[batch_idx]:
                ref_gt_mask = gt_targets[batch_idx]["masks"][None] # (1, 1, H, W)
            else:
                ref_gt_mask = None
            if self.extra_backbone_for_template:
                template_tensor, template_mask = self.get_template_4c(img, mask, bbox, gt_mask=ref_gt_mask)
            else:
                template_tensor, template_mask = self.get_template(img, mask, bbox)
            ref_tensor_list.append(template_tensor)
            ref_mask_list.append(template_mask)
        ref_tensor = torch.cat(ref_tensor_list, dim=0)
        ref_mask = torch.cat(ref_mask_list, dim=0)
        samples_ref = NestedTensor(ref_tensor, ref_mask)
        srcs_ref, masks_ref, _, _ = self.forward_backbone(samples_ref, template_branch=True)
        language_dict_features_key = {}
        if self.sot_feat_fusion:
            ref_feats = self.sot_fuser(srcs_ref).flatten(-2).permute(0, 2, 1) # (bs, L, C)
            ref_masks = masks_ref[0].flatten(-2) # (bs, L)
        else:
            ref_feats, ref_masks = [], []
            for n_l in range(self.detr.num_feature_levels):
                ref_feat_l = F.interpolate(srcs_ref[n_l], size=(self.ref_feat_sz, self.ref_feat_sz)) # (bs, C, 8, 8)
                ref_feats.append(ref_feat_l.flatten(-2)) # (bs, C, 64)
                ref_mask_l = F.interpolate(masks_ref[n_l][None].float(), size=(self.ref_feat_sz, self.ref_feat_sz))[0].bool() # (bs, 8, 8)
                ref_masks.append(ref_mask_l.flatten(-2)) # (bs, 64)
            ref_feats = torch.cat(ref_feats, dim=-1).permute(0, 2, 1) # (bs, C, L) L=256 --> (bs, L, C)
            ref_masks = torch.cat(ref_masks, dim=-1) # (bs, L)
        # original masks need to be reversed as text_masks!
        language_dict_features_key["hidden"], language_dict_features_key["masks"] = self.adjust_layer(ref_feats), ~ref_masks
        return language_dict_features_key, samples_ref


    # for VOS (initialize on the 1st appearing frame)
    def coco_inference_ref_vos(self, samples, ref_bboxes, ref_gt_masks):
        """ref_bboxes: List (len is bs)"""
        if not isinstance(samples, NestedTensor):
            size_divisibility = getattr(self.detr.backbone[0].backbone, "size_divisibility", 32)
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=size_divisibility)

        bz = samples.tensors.shape[0]
        assert bz == 1
        # forward template to get language_dict_features_key  
        tensors_ref = samples.tensors # (bs, 3, H, W)
        mask_ref = samples.mask # (bs, H, W)
        ref_tensor_list, ref_mask_list = [], []
        for batch_idx in range(bz):
            img = tensors_ref[batch_idx:batch_idx+1] # (1, 3, H, W)
            mask = mask_ref[batch_idx:batch_idx+1][None] # (1, H, W) -> (1, 1, H, W)
            bbox = ref_bboxes[batch_idx].clone() # (1, 4)
            bbox[:, 2:] = bbox[:, 2:] - bbox[:, :2]
            ref_gt_mask = ref_gt_masks[batch_idx][None] # (1, 1, H, W)
            if self.extra_backbone_for_template:
                template_tensor, template_mask = self.get_template_4c(img, mask, bbox, gt_mask=ref_gt_mask)
            else:
                template_tensor, template_mask = self.get_template(img, mask, bbox)
            ref_tensor_list.append(template_tensor)
            ref_mask_list.append(template_mask)
        ref_tensor = torch.cat(ref_tensor_list, dim=0)
        ref_mask = torch.cat(ref_mask_list, dim=0)
        samples_ref = NestedTensor(ref_tensor, ref_mask)
        srcs_ref, masks_ref, _, _ = self.forward_backbone(samples_ref, template_branch=True)
        language_dict_features_key = {}
        if self.sot_feat_fusion:
            ref_feats = self.sot_fuser(srcs_ref).flatten(-2).permute(0, 2, 1) # (bs, L, C)
            ref_masks = masks_ref[0].flatten(-2) # (bs, L)
        else:
            ref_feats, ref_masks = [], []
            for n_l in range(self.detr.num_feature_levels):
                ref_feat_l = F.interpolate(srcs_ref[n_l], size=(self.ref_feat_sz, self.ref_feat_sz)) # (bs, C, 8, 8)
                ref_feats.append(ref_feat_l.flatten(-2)) # (bs, C, 64)
                ref_mask_l = F.interpolate(masks_ref[n_l][None].float(), size=(self.ref_feat_sz, self.ref_feat_sz))[0].bool() # (bs, 8, 8)
                ref_masks.append(ref_mask_l.flatten(-2)) # (bs, 64)
            ref_feats = torch.cat(ref_feats, dim=-1).permute(0, 2, 1) # (bs, C, L) L=256 --> (bs, L, C)
            ref_masks = torch.cat(ref_masks, dim=-1) # (bs, L)
        # original masks need to be reversed as text_masks!
        language_dict_features_key["hidden"], language_dict_features_key["masks"] = self.adjust_layer(ref_feats), ~ref_masks
        return language_dict_features_key, samples_ref

    def coco_forward_vis(self, samples, det_targets, ref_targets, criterion, train=False, language_dict_features=None, task=None):
        image_sizes = samples.image_sizes
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=32)

        features, pos = self.detr.backbone(samples)
        
        srcs = []
        masks = []
        poses = []
        spatial_shapes = []

        for l, feat in enumerate(features):
            # src: [N, _C, Hi, Wi],
            # mask: [N, Hi, Wi],
            # pos: [N, C, H_p, W_p]
            src, mask = feat.decompose() 
            src_proj_l = self.detr.input_proj[l](src)    # src_proj_l: [N, C, Hi, Wi]
            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos[l])
            n, c, h, w = src_proj_l.shape
            spatial_shapes.append((h, w))

        if self.detr.num_feature_levels > len(features):
            _len_srcs = len(features)
            for l in range(_len_srcs, self.detr.num_feature_levels):
                if l == _len_srcs:
                    src = self.detr.input_proj[l](features[-1].tensors)
                else:
                    src = self.detr.input_proj[l](srcs[-1])
                m = masks[0]   # [N, H, W]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)
                n, c, h, w = src.shape
                spatial_shapes.append((h, w))

        if not self.detr.two_stage or self.decouple_tgt:
            query_embeds = self.detr.query_embed.weight
        else:
            query_embeds = None

        srcs_key = []
        masks_key = []
        poses_key = []

        srcs_reference = []
        masks_reference = []
        poses_reference = []

        bz = samples.tensors.shape[0]//2
        key_ids = list(range(0, bz*2, 2))
        ref_ids = list(range(1, bz*2, 2))

        for n_l in range(self.detr.num_feature_levels):
            srcs_key.append(srcs[n_l][key_ids])
            srcs_reference.append(srcs[n_l][ref_ids])
            masks_key.append(masks[n_l][key_ids])
            masks_reference.append(masks[n_l][ref_ids])
            poses_key.append(poses[n_l][key_ids])
            poses_reference.append(poses[n_l][ref_ids])
        image_sizes = [image_sizes[_i] for _i in key_ids]
        # det_targets = [gt_targets[_i] for _i in key_ids]
        # ref_targets = [gt_targets[_i] for _i in ref_ids]
        language_dict_features_key = {}
        language_dict_features_ref = {}
        for k, v in language_dict_features.items():
            language_dict_features_key[k] = v[key_ids].clone()
            language_dict_features_ref[k] = v[ref_ids].clone()

        # ref frames are not used for detection
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, language_dict_features_key = self.detr.transformer(srcs_key, masks_key, poses_key, query_embeds, mask_on=True, language_dict_features=language_dict_features_key, task=task)
        hs_ref, _, _, inter_references_ref, _un1, _un2, language_dict_features_ref = self.detr.transformer(srcs_reference, masks_reference, poses_reference, query_embeds, mask_on=True, language_dict_features=language_dict_features_ref, task=task)

        if task == "grounding":
            lang_feat_pool_key = agg_lang_feat(language_dict_features_key["hidden"], language_dict_features_key["masks"]).unsqueeze(1) # (bs, 1, 768)
            lang_feat_pool_ref = agg_lang_feat(language_dict_features_ref["hidden"], language_dict_features_ref["masks"]).unsqueeze(1) # (bs, 1, 768)
        elif task == "detection":
            pass
        else:
            raise ValueError("task must be detection or grounding")
        # memory: [N, \sigma(HiWi), C]
        # hs: [num_encoders, N, num_querries, C]

        outputs = {}
        outputs_classes = []
        outputs_coords = []
        outputs_masks = []
        indices_list = []
        if self.use_iou_branch:
            outputs_ious = []
        enc_lay_num = hs.shape[0]
        
        for lvl in range(enc_lay_num):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            if task == "grounding":
                outputs_class = self.detr.class_embed[lvl](hs[lvl], lang_feat_pool_key)
            elif task == "detection":
                outputs_class = self.detr.class_embed[lvl](hs[lvl], language_dict_features_key["hidden"])
            else:
                raise ValueError("task must be detection or grounding")
            tmp = self.detr.bbox_embed[lvl](hs[lvl])
            if self.use_iou_branch:
                pred_iou = self.detr.iou_head[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            if self.use_iou_branch:
                outputs_ious.append(pred_iou)
            outputs_layer = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
            dynamic_mask_head_params = self.controller(hs[lvl])    # [bs, num_quries, num_params]

            # for training & log evaluation loss
            if self.ota:
                indices, matched_ids = criterion.matcher.forward_ota(outputs_layer, det_targets)
            else:
                indices = criterion.matcher.forward(outputs_layer, det_targets)
            indices_list.append(indices)
            reference_points, mask_head_params, num_insts = [], [], []
            for i, indice in enumerate(indices):
                pred_i, tgt_j = indice
                if self.ota:
                    num_insts.append(len(pred_i))
                else:
                    num_insts.append(len(pred_i))
                mask_head_params.append(dynamic_mask_head_params[i, pred_i].unsqueeze(0))

                # This is the image size after data augmentation (so as the gt boxes & masks)
                
                orig_h, orig_w = image_sizes[i]
                orig_h = torch.as_tensor(orig_h).to(reference)
                orig_w = torch.as_tensor(orig_w).to(reference)
                scale_f = torch.stack([orig_w, orig_h], dim=0)
                
                ref_cur_f = reference[i].sigmoid()
                ref_cur_f = ref_cur_f[..., :2]
                ref_cur_f = ref_cur_f * scale_f[None, :] 
                reference_points.append(ref_cur_f[pred_i].unsqueeze(0))
            # reference_points: [1, nf,  \sum{selected_insts}, 2]
            # mask_head_params: [1, \sum{selected_insts}, num_params]
            reference_points = torch.cat(reference_points, dim=1)
            mask_head_params = torch.cat(mask_head_params, dim=1)
 
            # mask prediction
            has_mask_list = ["masks" in x.keys() for x in det_targets]
            assert len(set(has_mask_list)) == 1 # must be "all True" or "all False"
            if has_mask_list[0]:
                outputs_layer = self.forward_mask_head_train(outputs_layer, memory, spatial_shapes, 
                                                            reference_points, mask_head_params, num_insts)
            else:
                # avoid unused parameters
                dummy_output = torch.sum(mask_head_params)
                for p in self.mask_head.parameters():
                    dummy_output += p.sum()
                outputs_layer['pred_masks'] = 0.0 * dummy_output
            outputs_masks.append(outputs_layer['pred_masks'])
            
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_mask = outputs_masks

        if task == "grounding":
            ref_cls = self.detr.class_embed[enc_lay_num-1](hs_ref[-1], lang_feat_pool_ref).sigmoid()
        else:
            ref_cls = self.detr.class_embed[enc_lay_num-1](hs_ref[-1], language_dict_features_ref["hidden"]).sigmoid() # NOTE: self.detr.class_embed[-1] may cause conflict with two-stage !
        # Only compute Embedding Loss on the last layer of decoder
        contrast_items = select_pos_neg(inter_references_ref[-1], matched_ids, ref_targets, det_targets, self.reid_embed_head, hs[-1], hs_ref[-1], ref_cls)

        # outputs['pred_samples'] = inter_samples[-1]
        
        outputs['pred_logits'] = outputs_class[-1]
        outputs['pred_boxes'] = outputs_coord[-1]
        outputs['pred_masks'] = outputs_mask[-1]
        outputs['pred_qd'] = contrast_items
        if self.use_iou_branch:
            outputs_iou = torch.stack(outputs_ious)
            outputs['pred_boxious'] = outputs_iou[-1]
        reid_params = 0
        for p in self.reid_embed_head.parameters():
            reid_params += 0.0 * p.sum() # avoid unused parameters
        outputs['reid_params'] = reid_params
        if task == "grounding":
            bs, device = language_dict_features_key["masks"].size(0), language_dict_features_key["masks"].device
            text_masks = torch.ones((bs, 1), dtype=torch.bool, device=device)
        elif task == "detection":
            text_masks = language_dict_features_key["masks"]
        else:
            raise ValueError("task must be detection or grounding")
        outputs["text_masks"] = text_masks
        if self.detr.aux_loss:
            if self.use_iou_branch:
                outputs['aux_outputs'] = self._set_aux_loss_with_iou(outputs_class, outputs_coord, outputs_mask, outputs_iou)
            else:
                outputs['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_mask)
            for x in outputs['aux_outputs']:
                x["text_masks"] = text_masks
        if self.detr.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            outputs['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord, "text_masks": text_masks}
        loss_dict = criterion(outputs, det_targets, indices_list)
        # fix unused parameters of the mask head
        no_valid_obj = False
        if isinstance(outputs['pred_masks'], list):
            if len(outputs['pred_masks']) == 0:
                no_valid_obj = True
        if no_valid_obj:
            loss_mask = loss_dict["loss_mask"]
            for n, p in self.mask_head.named_parameters():
                loss_mask += 0.0 * torch.sum(p)
            for n, p in self.controller.named_parameters():
                loss_mask += 0.0 * torch.sum(p)
            loss_dict["loss_mask"] = loss_mask
            loss_dict["loss_dice"] = loss_mask
            for i in range(enc_lay_num-1):
                loss_dict["loss_mask_%d"%i] = loss_mask
                loss_dict["loss_dice_%d"%i] = loss_mask
        return outputs, loss_dict


class FeatureFuser(nn.Module):
    """
    Feature Fuser for SOT (inspired by CondInst)
    """
    def __init__(self, in_channels, channels=256):
        super().__init__()

        self.refine = nn.ModuleList()
        for _ in range(4):
            self.refine.append(nn.Conv2d(in_channels, channels, 3, padding=1))

    def forward(self, features):
        # -4, -3, -2, -1 corresponds to P3, P4, P5, P6
        for i, f in enumerate([-4, -3, -2, -1]):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])
                target_h, target_w = x.size()[2:]
                h, w = x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p
        return x

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]