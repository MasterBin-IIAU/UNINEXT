# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from typing import Dict, List
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from .backbone.masked_backbone import MaskedBackbone
from .models.deformable_detr.backbone import Joiner
from .models.deformable_detr.deformable_detr import DeformableDETR, SetCriterion, DeformableDETRDINO, DINOCriterion
from .models.deformable_detr.matcher import HungarianMatcherVL
from .models.deformable_detr.position_encoding import PositionEmbeddingSine
from .models.deformable_detr.deformable_transformer import DeformableTransformerVL
from .models.deformable_detr.deformable_transformer_dino import DeformableTransformerVLDINO
from .models.ddetrs_vid import DDETRSegmUniVID
from .models.ddetrs_vid_dn import DDETRSegmUniVIDDN
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import NestedTensor
import torchvision.ops as ops
# Language-guided detection
from transformers import AutoTokenizer
from .models.deformable_detr.bert_model import BertEncoder

from collections import OrderedDict
from einops import repeat
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)
from detectron2.structures import BoxMode
import cv2
from .models.tracker import IDOL_Tracker, QuasiDenseEmbedTracker
import copy
from collections import defaultdict
from PIL import Image
import time
from detectron2.layers import ShapeSpec
import pycocotools.mask as mask_util
from scipy.optimize import linear_sum_assignment
import vot_tool

__all__ = ["UNINEXT_VOTS"]


@META_ARCH_REGISTRY.register()
class UNINEXT_VOTS(nn.Module):
    """
    Unified model for video-level tasks (SOT, VOS, R-VOS, MOT, MOTS, VIS)
    """

    def __init__(self, cfg):
        super().__init__()
        self.debug_only = False
        self.cfg = cfg
        self.use_amp = cfg.SOLVER.AMP.ENABLED
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.mask_stride = cfg.MODEL.DDETRS.MASK_STRIDE
        self.mask_on = cfg.MODEL.MASK_ON
        self.ota = cfg.MODEL.OTA
        self.mask_thres = cfg.MODEL.DDETRS.MASK_THRES
        self.new_mask_head = cfg.MODEL.DDETRS.NEW_MASK_HEAD
        self.use_raft = cfg.MODEL.DDETRS.USE_RAFT
        self.use_rel_coord = cfg.MODEL.DDETRS.USE_REL_COORD
        self.num_queries = cfg.MODEL.DDETRS.NUM_OBJECT_QUERIES

        ### inference setting
        self.merge_on_cpu = cfg.MODEL.IDOL.MERGE_ON_CPU
        self.is_multi_cls = cfg.MODEL.IDOL.MULTI_CLS_ON
        self.apply_cls_thres = cfg.MODEL.IDOL.APPLY_CLS_THRES
        self.temporal_score_type = cfg.MODEL.IDOL.TEMPORAL_SCORE_TYPE
        self.inference_select_thres = cfg.MODEL.IDOL.INFERENCE_SELECT_THRES
        self.inference_fw = cfg.MODEL.IDOL.INFERENCE_FW
        self.inference_tw = cfg.MODEL.IDOL.INFERENCE_TW
        self.memory_len = cfg.MODEL.IDOL.MEMORY_LEN
        self.batch_infer_len = cfg.MODEL.IDOL.BATCH_INFER_LEN # 10
        self.merge_device = "cpu" if self.merge_on_cpu else self.device
        self.save_path_prefix = os.path.join(cfg.OUTPUT_DIR, "Annotations")

        # Transformer parameters:
        hidden_dim = cfg.MODEL.DDETRS.HIDDEN_DIM
        nheads = cfg.MODEL.DDETRS.NHEADS
        dim_feedforward = cfg.MODEL.DDETRS.DIM_FEEDFORWARD
        dec_layers = cfg.MODEL.DDETRS.DEC_LAYERS

        num_feature_levels = cfg.MODEL.DDETRS.NUM_FEATURE_LEVELS
        two_stage = cfg.MODEL.DDETRS.TWO_STAGE
        two_stage_num_proposals = cfg.MODEL.DDETRS.TWO_STAGE_NUM_PROPOSALS

        # Loss parameters:
        mask_weight = cfg.MODEL.DDETRS.MASK_WEIGHT
        dice_weight = cfg.MODEL.DDETRS.DICE_WEIGHT
        giou_weight = cfg.MODEL.DDETRS.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DDETRS.L1_WEIGHT
        class_weight = cfg.MODEL.DDETRS.CLASS_WEIGHT
        reid_weight = cfg.MODEL.DDETRS.REID_WEIGHT
        deep_supervision = cfg.MODEL.DDETRS.DEEP_SUPERVISION
        focal_alpha = cfg.MODEL.DDETRS.FOCAL_ALPHA
        # Cost parameters (for label assignment):
        set_cost_class = cfg.MODEL.DDETRS.SET_COST_CLASS
        set_cost_bbox = cfg.MODEL.DDETRS.SET_COST_BOX
        set_cost_giou = cfg.MODEL.DDETRS.SET_COST_GIOU

        # Backbone
        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels  # only take [c3 c4 c5] from resnet and gengrate c6 later
        backbone.strides = d2_backbone.feature_strides

        # Transformer & Early Fusion
        if cfg.MODEL.DDETRS.USE_DINO:
            transformer_class = DeformableTransformerVLDINO
        else:
            transformer_class = DeformableTransformerVL
        transformer = transformer_class(
        d_model= hidden_dim,
        nhead=nheads,
        num_encoder_layers=cfg.MODEL.DDETRS.ENC_LAYERS,
        num_decoder_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        dropout=cfg.MODEL.DDETRS.DROPOUT,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=num_feature_levels,
        dec_n_points=cfg.MODEL.DDETRS.DEC_N_POINTS,
        enc_n_points=cfg.MODEL.DDETRS.ENC_N_POINTS,
        two_stage=two_stage,
        two_stage_num_proposals=two_stage_num_proposals,
        use_checkpoint=cfg.MODEL.DDETRS.USE_CHECKPOINT,
        look_forward_twice=cfg.MODEL.DDETRS.LOOK_FORWARD_TWICE,
        mixed_selection=cfg.MODEL.DDETRS.MIXED_SELECTION,
        cfg=cfg)
        
        if cfg.MODEL.DDETRS.USE_DINO:
            detr_class = DeformableDETRDINO
        else:
            detr_class = DeformableDETR
        model = detr_class(
        backbone,
        transformer,
        num_queries=self.num_queries,
        num_feature_levels=num_feature_levels,
        aux_loss=deep_supervision,
        with_box_refine=True,
        two_stage=two_stage,
        mixed_selection=cfg.MODEL.DDETRS.MIXED_SELECTION,
        cfg=cfg)
        
        # backbone for the template branch (for SOT and VOS)
        if cfg.SOT.EXTRA_BACKBONE_FOR_TEMPLATE:
            cfg.defrost()
            cfg.MODEL.CONVNEXT.USE_CHECKPOINT = False
            d2_backbone_ref = MaskedBackbone(cfg, input_shape=ShapeSpec(channels=4))
            ref_backbone = Joiner(d2_backbone_ref, PositionEmbeddingSine(N_steps, normalize=True))
            ref_backbone.num_channels = d2_backbone.num_channels  # only take [c3 c4 c5] from resnet and gengrate c6 later
            ref_backbone.strides = d2_backbone.feature_strides
            model.ref_backbone = ref_backbone
        # DETR + Segmentation (CondInst)
        if cfg.MODEL.DDETRS.USE_DINO:
            model_class = DDETRSegmUniVIDDN
        else:
            model_class = DDETRSegmUniVID
        self.detr = model_class(model, rel_coord=self.use_rel_coord, ota=self.ota, 
        new_mask_head=self.new_mask_head, use_raft=self.use_raft, mask_out_stride=self.mask_stride, 
        template_sz=cfg.SOT.TEMPLATE_SZ, 
        extra_backbone_for_template=cfg.SOT.EXTRA_BACKBONE_FOR_TEMPLATE, search_area_factor=cfg.SOT.SEARCH_AREA_FACTOR,
        ref_feat_sz=cfg.SOT.REF_FEAT_SZ, sot_feat_fusion=cfg.SOT.FEAT_FUSE,
        use_iou_branch=cfg.MODEL.USE_IOU_BRANCH, decouple_tgt=cfg.MODEL.DECOUPLE_TGT, cfg=cfg)

        self.detr.to(self.device)

        # building criterion
        matcher = HungarianMatcherVL(
            cost_class=set_cost_class,
            cost_bbox=set_cost_bbox,
            cost_giou=set_cost_giou)

        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou":giou_weight, \
            "loss_mask": mask_weight, "loss_dice": dice_weight, "loss_reid": reid_weight, "loss_reid_aux":reid_weight*1.5}

        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        if cfg.MODEL.DDETRS.USE_DINO:
            weight_dict_dn = {"loss_ce_dn": class_weight, "loss_bbox_dn": l1_weight, "loss_giou_dn": giou_weight}
            aux_weight_dict_dn = {}
            for i in range(dec_layers - 1):
                aux_weight_dict_dn.update({k + f"_{i}": v for k, v in weight_dict_dn.items()})
            weight_dict_dn.update(aux_weight_dict_dn)
            weight_dict.update(weight_dict_dn)

        losses = ['labelsVL', 'boxes', 'masks', 'reid']
        if cfg.MODEL.DDETRS.USE_DINO:
            criterion_class = DINOCriterion
        else:
            criterion_class = SetCriterion
        self.criterion = criterion_class(matcher, weight_dict, losses, focal_alpha=focal_alpha, ota=self.ota, 
        still_cls_for_encoder=cfg.MODEL.STILL_CLS_FOR_ENCODER, cfg=cfg)
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        self.use_lsj = cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj"
        # tracker params
        self.init_score_thr = cfg.TRACK.INIT_SCORE_THR # score threshold to start a new track
        self.obj_score_thr = cfg.TRACK.OBJ_SCORE_THR # score threshold to continue a track
        # SOT inference params
        self.online_update = cfg.SOT.ONLINE_UPDATE
        self.update_interval = cfg.SOT.UPDATE_INTERVAL
        self.update_thr = cfg.SOT.UPDATE_THR
        # for SOT and VOS
        self.extra_backbone_for_template = cfg.SOT.EXTRA_BACKBONE_FOR_TEMPLATE
        self.inference_on_3f = cfg.SOT.INFERENCE_ON_3F
        self.inst_thr_vos = cfg.SOT.INST_THR_VOS


    def forward(self, batched_inputs, frame_idx, obj_idx, mask_anno=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        task = "sot"
        # inference for SOT and VOS datasets
        positive_map_label_to_token = {1: [0]}
        num_classes = len(positive_map_label_to_token) # num_classes during testing
        # batchsize = 1 during inference
        height = batched_inputs[0]['height']
        width = batched_inputs[0]['width']
        dataset_name = "vots2023"
        height = batched_inputs[0]["height"]
        width = batched_inputs[0]["width"]
        clip_inputs = [{'image':batched_inputs[0]['image']}]
        images = self.preprocess_video(clip_inputs)
        if self.debug_only:
            save_dir = "/scratch/binyan/UNINEXT-VOTS/debug"
            vid_name = "cur_vid"
            save_img_dir = os.path.join(save_dir, vid_name, obj_idx)
            if not os.path.exists(save_img_dir):
                os.makedirs(save_img_dir)
        if frame_idx == 0:
            assert mask_anno is not None
            x1_c, y1_c, w_c, h_c = bounding_box(mask_anno) # current bounding box
            cur_ref_bboxes = torch.tensor([x1_c, y1_c, x1_c+w_c, y1_c+h_c]).view(1, 4)
            cur_ref_bboxes = [cur_ref_bboxes.to(self.device)] # List (1, 4)
            size_divisibility = getattr(self.detr.detr.backbone[0].backbone, "size_divisibility", 32)
            if size_divisibility != 0:
                mask_h, mask_w = mask_anno.shape
                mask_h_new = (mask_h + (size_divisibility - 1)) // size_divisibility * size_divisibility
                mask_w_new = (mask_w + (size_divisibility - 1)) // size_divisibility * size_divisibility
                mask_anno_new = np.zeros((mask_h_new, mask_w_new), dtype=np.uint8)
                mask_anno_new[:mask_h, :mask_w] = mask_anno
                cur_ref_masks = [torch.from_numpy(mask_anno_new[None]).to(self.device)]
            else:
                cur_ref_masks = [torch.from_numpy(mask_anno[None]).to(self.device)]
            self.language_dict_features, template = self.detr.coco_inference_ref_vos(images, cur_ref_bboxes, cur_ref_masks)
            self.language_dict_features_prev = copy.deepcopy(self.language_dict_features)
            if self.debug_only:
                self.debug_template_4c(template, vid_name, obj_idx, frame_idx, save_img_dir)
            return
        assert mask_anno is None
        if self.online_update:
            language_dict_features1 = copy.deepcopy(self.language_dict_features) # Important
            language_dict_features2 = copy.deepcopy(self.language_dict_features_prev) # Important
            language_dict_features_cur = {}
            language_dict_features_cur["hidden"] = torch.cat([language_dict_features1["hidden"], language_dict_features2["hidden"]], dim=1)
            language_dict_features_cur["masks"] = torch.cat([language_dict_features1["masks"], language_dict_features2["masks"]], dim=1)
        else:
            language_dict_features_cur = copy.deepcopy(self.language_dict_features) # Important
        output, _ = self.detr.coco_inference(images, None, None, language_dict_features=language_dict_features_cur, task=task)
        box_cls = output["pred_logits"]
        box_pred = output["pred_boxes"]
        mask_pred = output["pred_masks"] if self.mask_on else [None] * len(batched_inputs)
        if self.detr.use_iou_branch:
            iou_pred = output["pred_boxious"]
        else:
            iou_pred = [None]
        results = self.inference(box_cls, box_pred, mask_pred, images.image_sizes, positive_map_label_to_token, num_classes, task=task, iou_pred=iou_pred)
        for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            scale_x, scale_y = (
                width / results_per_image.image_size[1],
                height / results_per_image.image_size[0],
            )
            results_per_image.pred_boxes.scale(scale_x, scale_y)
            results_per_image.pred_boxes.clip((height, width))
            x1, y1, x2, y2 = results_per_image.pred_boxes.tensor.tolist()[0]
            # mask
            if results_per_image.scores.item() < self.inst_thr_vos:
                final_mask = vot_tool.Empty()
            else:
                final_mask = F.interpolate(results_per_image.pred_masks[:,:,:image_size[0],:image_size[1]].float(), size=(height, width), mode="bilinear", align_corners=False)
                final_mask = final_mask[0, 0].cpu().numpy().astype(np.uint8) # (H, W)
                if self.debug_only:
                    # debug
                    ori_img = F.interpolate(batched_inputs[0]['image'][0].unsqueeze(0)[:, :, :images.image_sizes[0][0], :images.image_sizes[0][1]], size=(height, width))
                    img_arr = ori_img[0].permute((1, 2, 0)).cpu().numpy()
                    img_arr = np.ascontiguousarray(img_arr[:, :, ::-1]).clip(0, 255)
                    img_arr_det = copy.deepcopy(img_arr)
                    cv2.rectangle(img_arr_det, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,0,255), thickness=2)
                    img_arr_det[:, :, -1] = np.clip(img_arr_det[:, :, -1] + 128 * final_mask, 0, 255)
                    cv2.imwrite(os.path.join(save_img_dir, "%05d.jpg"%frame_idx), img_arr_det)
        if self.online_update and (frame_idx % self.update_interval == 0) and (results[0].scores > self.update_thr):
            # update the template
            bboxes_unorm = torch.tensor([[x1, y1, x2, y2]]) / torch.tensor([scale_x, scale_y, scale_x, scale_y])
            self.language_dict_features_prev, new_template = self.detr.coco_inference_ref_vos(images, [bboxes_unorm.to(self.device)], [results_per_image.pred_masks.float()[0]])
            if self.debug_only:
                self.debug_template_4c(new_template, vid_name, obj_idx, frame_idx, save_img_dir)
        
        return final_mask


    def debug_template_4c(self, samples, vid_name, obj_id, frame_idx, save_img_dir):
        import numpy as np
        import cv2
        mean = np.array([123.675, 116.280, 103.530])
        std = np.array([58.395, 57.120, 57.375])
        assert len(samples.tensors) == 1
        i = 0
        image_mask = samples.tensors[i].permute((1, 2, 0)).cpu().numpy()
        image = image_mask[:, :, :3]
        gt_mask = image_mask[:, :, -1]
        image = image * std + mean # (H, W, 3)
        input_mask = samples.mask[i].float().cpu().numpy() * 255 # (H, W)
        image = np.ascontiguousarray(image[:, :, ::-1]).clip(0, 255)
        image[:, :, -1] = np.clip(image[:, :, -1] + 100 * gt_mask, 0, 255)
        cv2.imwrite("%s/%05d_img.jpg"%(save_img_dir, frame_idx), image)
        cv2.imwrite("%s/%05d_mask.jpg"%(save_img_dir, frame_idx), input_mask)


    def inference(self, box_cls, box_pred, mask_pred, image_sizes, positive_map_label_to_token, num_classes, score_thres=0.0, task=None, binary_mask=True, iou_pred=None):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        if task == "detection":
            max_num_inst = 100
        elif task == "grounding" or task == "sot":
            max_num_inst = 1
        else:
            raise ValueError("task must be detection or grounding")
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)
        for (logits_per_image, box_pred_per_image, mask_pred_per_image, image_size, iou_per_image) in zip(
            box_cls, box_pred, mask_pred, image_sizes, iou_pred):

            if self.ota:
                # NMS
                logits_per_image = convert_grounding_to_od_logits(logits_per_image.unsqueeze(0), num_classes, positive_map_label_to_token)
                logits_per_image = logits_per_image[0] # (num_query, C)
                prob = logits_per_image.sigmoid()
                # cls x iou
                if iou_per_image is not None:
                    prob = torch.sqrt(prob * iou_per_image.sigmoid()) # (num_query, C)
                # filter low-score predictions
                if score_thres > 0.0:
                    valid_mask = (prob > score_thres)
                    num_valid = torch.sum(valid_mask).item()
                    num_inst = min(num_valid, max_num_inst)
                    prob[~valid_mask] = -1.0 # set to invalid status
                else:
                    num_inst = max_num_inst
                nms_scores,idxs = torch.max(prob,1)
                boxes_before_nms = box_cxcywh_to_xyxy(box_pred_per_image)
                keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.7)  
                prob = prob[keep_indices]
                box_pred_per_image = box_pred_per_image[keep_indices]
                if mask_pred_per_image is not None:
                    mask_pred_per_image = mask_pred_per_image[keep_indices]

                topk_values, topk_indexes = torch.topk(prob.view(-1), num_inst, dim=0)
                scores = topk_values
                topk_boxes = torch.div(topk_indexes, logits_per_image.shape[1], rounding_mode='floor')
                # topk_boxes = topk_indexes // logits_per_image.shape[1]
                labels = topk_indexes % logits_per_image.shape[1]
                scores_per_image = scores
                labels_per_image = labels

                box_pred_per_image = box_pred_per_image[topk_boxes]
                if mask_pred_per_image is not None:
                    mask_pred_per_image = mask_pred_per_image[topk_boxes]
            else:
                logits_per_image = convert_grounding_to_od_logits(logits_per_image.unsqueeze(0), num_classes, positive_map_label_to_token)
                logits_per_image = logits_per_image[0] # (num_query, C)
                prob = logits_per_image.sigmoid()
                # cls x iou
                if iou_per_image is not None:
                    prob = torch.sqrt(prob * iou_per_image.sigmoid()) # (num_query, C)
                # filter low-score predictions
                if score_thres > 0.0:
                    valid_mask = (prob > score_thres)
                    num_valid = torch.sum(valid_mask).item()
                    num_inst = min(num_valid, max_num_inst)
                    prob[~valid_mask] = -1.0 # set to invalid status
                else:
                    num_inst = max_num_inst
                topk_values, topk_indexes = torch.topk(prob.view(-1), num_inst, dim=0)
                scores = topk_values
                topk_boxes = torch.div(topk_indexes, logits_per_image.shape[1], rounding_mode='floor')
                # topk_boxes = topk_indexes // logits_per_image.shape[1]
                labels = topk_indexes % logits_per_image.shape[1]

                scores_per_image = scores
                labels_per_image = labels

                box_pred_per_image = box_pred_per_image[topk_boxes]
                if mask_pred_per_image is not None:
                    mask_pred_per_image = mask_pred_per_image[topk_boxes]
            
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                N, C, H, W = mask_pred_per_image.shape
                mask = F.interpolate(mask_pred_per_image, size=(H*self.mask_stride, W*self.mask_stride), mode='bilinear', align_corners=False)
                mask = mask.sigmoid()
                if binary_mask:
                    mask = mask > self.mask_thres
                # mask = mask[:,:,:image_size[0],:image_size[1]]
                result.pred_masks = mask
                
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def preprocess_video(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(self.normalizer(frame.to(self.device)))
        images = ImageList.from_tensors(images)
        return images


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout):
        super().__init__()
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        x = self.layer_norm(x)
        output = self.dropout(x)
        return output

def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2-x1), int(y2-y1)] # (x1, y1, w, h) 

def convert_grounding_to_od_logits(logits, num_classes, positive_map, score_agg="MEAN"):
    """
    logits: (bs, num_query, max_seq_len)
    num_classes: 80 for COCO
    """
    assert logits.ndim == 3
    assert positive_map is not None
    scores = torch.zeros(logits.shape[0], logits.shape[1], num_classes).to(logits.device)
    # 256 -> 80, average for each class
    # score aggregation method
    if score_agg == "MEAN": # True
        for label_j in positive_map:
            scores[:, :, label_j - 1] = logits[:, :, torch.LongTensor(positive_map[label_j])].mean(-1)
    else:
        raise NotImplementedError
    return scores

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

