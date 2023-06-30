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
from .models.ddetrs import DDETRSegmUni, segmentation_postprocess
from .models.ddetrs_dn import DDETRSegmUniDN
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
from skimage import color
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
__all__ = ["UNINEXT_IMG"]


@META_ARCH_REGISTRY.register()
class UNINEXT_IMG(nn.Module):
    """
    Unified model for image-level tasks (OD, IS, REC, RES)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.demo_only = False
        self.num_frames = 1
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
        
        # DETR
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

        # Language (text encoder and tokenizer)
        self.parallel_det = cfg.MODEL.PARALLEL_DET
        # Here we use BERT as the text encoder in a hard-code way
        self.tokenizer = AutoTokenizer.from_pretrained("projects/UNINEXT/bert-base-uncased")
        if self.parallel_det:
            self.text_encoder = BertEncoder(cfg)
        else:
            self.text_encoder = nn.Sequential(OrderedDict([("body", BertEncoder(cfg))]))
        if cfg.MODEL.FREEZE_TEXT_ENCODER:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        # DETR + Segmentation (CondInst)
        if cfg.MODEL.DDETRS.USE_DINO:
            model_class = DDETRSegmUniDN
        else:
            model_class = DDETRSegmUni
        self.detr = model_class(model, rel_coord=self.use_rel_coord, ota=self.ota, 
        new_mask_head=self.new_mask_head, use_raft=self.use_raft, mask_out_stride=self.mask_stride, 
        decouple_tgt=cfg.MODEL.DECOUPLE_TGT, cls_pool_type=cfg.MODEL.CLS_POOL_TYPE,
        use_iou_branch=cfg.MODEL.USE_IOU_BRANCH, cfg=cfg)

        self.detr.to(self.device)

        # building criterion
        matcher = HungarianMatcherVL(
            cost_class=set_cost_class,
            cost_bbox=set_cost_bbox,
            cost_giou=set_cost_giou)

        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight, \
            "loss_mask": mask_weight, "loss_dice": dice_weight}

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

        if cfg.MODEL.BOXINST.ENABLED:
            losses = ['labelsVL', 'boxes', 'masks_boxinst']
        else:
            losses = ['labelsVL', 'boxes', 'masks']

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
        
        # BoxInst
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH

        # Loss weights for different tasks
        self.loss_weight_det = cfg.SOLVER.LOSS_WEIGHT_DET
        self.loss_weight_grd = cfg.SOLVER.LOSS_WEIGHT_GRD

    def forward(self, batched_inputs, do_postprocess=True):
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

        # images = self.preprocess_image(batched_inputs)
        # output = self.detr(images)
        task_list = [x["task"] for x in batched_inputs]
        assert len(set(task_list)) == 1
        task = task_list[0]
        if self.training:
            if self.boxinst_enabled:
                images, targets = self.prepare_image_targets_boxinst(batched_inputs)
            else:
                images = self.preprocess_image(batched_inputs)
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances)
            # captions: list[str]
            captions = [x["expressions"] for x in batched_inputs]
            if self.parallel_det:
                language_dict_features = self.forward_text(captions, device="cuda", task=task)
            else:
                language_dict_features = self.forward_text(captions, device="cuda")
            output, loss_dict = self.detr.coco_forward(images, targets, self.criterion, train=True, language_dict_features=language_dict_features, task=task)
            # loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    if task == "detection":
                        loss_dict[k] *= (weight_dict[k] * self.loss_weight_det)
                    elif task == "grounding":
                        loss_dict[k] *= (weight_dict[k] * self.loss_weight_grd)
                    else:
                        raise ValueError("task should be detection or grounding")
            return loss_dict
        else:
            images = self.preprocess_image(batched_inputs)
            # gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            # targets = self.prepare_targets(gt_instances)
            # captions: list[str]
            captions = [x["expressions"] for x in batched_inputs]
            if task == "grounding":
                positive_map_label_to_token = {1: [0]}
            elif task == "detection":
                positive_map_label_to_token = batched_inputs[0]["positive_map_label_to_token"] # defaultdict(<class 'list'>, {1: [1], 2: [3], 3: [5], 4: [7], 5: [9], 6: [11], 7: [13], 8: [15], 9: [17], 10: [19, 20], 11: [22, 23, 24], 12: [26, 27], 13: [29, 30], 14: [32], 15: [34], 16: [36], 17: [38], 18: [40], 19: [42], 20: [44], 21: [46], 22: [48], 23: [50], 24: [52, 53, 54], 25: [56], 26: [58], 27: [60, 61], 28: [63], 29: [65], 30: [67, 68, 69], 31: [71, 72], 32: [74, 75], 33: [77, 78], 34: [80], 35: [82, 83], 36: [85, 86], 37: [88, 89], 38: [91, 92], 39: [94, 95, 96], 40: [98], 41: [100, 101], 42: [103], 43: [105], 44: [107], 45: [109], 46: [111], 47: [113], 48: [115], 49: [117], 50: [119], 51: [121, 122, 123], 52: [125], 53: [127, 128], 54: [130], 55: [132, 133], 56: [135], 57: [137], 58: [139], 59: [141, 142, 143], 60: [145], 61: [147, 148], 62: [150], 63: [152], 64: [154], 65: [156], 66: [158], 67: [160], 68: [162, 163], 69: [165], 70: [167], 71: [169, 170], 72: [172], 73: [174], 74: [176], 75: [178], 76: [180], 77: [182], 78: [184, 185], 79: [187, 188, 189], 80: [191, 192]})
            else:
                raise ValueError("task must be detection or grounding")
            num_classes = len(positive_map_label_to_token) # num_classes during testing

            if self.parallel_det:
                language_dict_features = self.forward_text(captions, device="cuda", task=task)
            else:
                language_dict_features = self.forward_text(captions, device="cuda")
            output, loss_dict = self.detr.coco_inference(images, None, self.criterion, train=False, language_dict_features=language_dict_features, task=task)
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"] if self.mask_on else None
            if self.detr.use_iou_branch:
                iou_pred = output["pred_boxious"]
            else:
                iou_pred = [None]
            # mask_pred = mask_pred[:,:,0]
            results = self.inference(box_cls, box_pred, mask_pred, images.image_sizes, positive_map_label_to_token, num_classes, task=task, iou_pred=iou_pred)
            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = segmentation_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                    # # visualization (assume single gpu and batch=1)
                    # caption = captions[0]
                    # ori_images = [cv2.imread(x["file_name"]) for x in batched_inputs][0]
                    # boxes = r.pred_boxes.tensor.cpu().numpy()
                    # boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
                    # boxes = boxes.tolist()
                    # masks = r.pred_masks.cpu().numpy().astype(np.float32)
                    # save_images = ori_images.astype(np.float32)
                    # classes = r.pred_classes.cpu().numpy()
                    # # cv2.putText(save_images, caption, (100, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
                    # for (box, mask, class_idx) in zip(boxes, masks, classes):
                    #     color = COCO_CATEGORIES[int(class_idx)]["color"]
                    #     color_mask = np.array(color) * mask[:, :, None] * 0.3
                    #     save_images += color_mask
                    #     x1, y1, w, h = box
                    #     cv2.rectangle(save_images, (int(x1), int(y1)), (int(x1+w), int(y1+h)), tuple(color), thickness=2)
                    # save_images = save_images.clip(0, 255)
                    # save_path = batched_inputs[0]["file_name"].split("/")[-1]
                    # cv2.imwrite(save_path, save_images)
                    # # print(caption)
                return processed_results
            else:
                return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            # for language-guided detection, classification loss is computed based on the positive map
            positive_map = targets_per_image.positive_map # (N, 256) or (1, 1). N is number of objects per image
            if self.use_amp:
                gt_boxes = gt_boxes.half()
                image_size_xyxy = image_size_xyxy.half()
            if hasattr(targets_per_image, "gt_masks"):
                if self.use_lsj:
                    gt_masks = targets_per_image.gt_masks
                else:
                    gt_masks = targets_per_image.gt_masks.tensor
                if self.use_amp:
                    gt_masks = gt_masks.half()
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks, "image_size": image_size_xyxy, 
                "positive_map": positive_map})
            else:
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "image_size": image_size_xyxy, 
                "positive_map": positive_map})
        return new_targets

    def prepare_targets_boxinst(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            # for language-guided detection, classification loss is computed based on the positive map
            positive_map = targets_per_image.positive_map # (N, 256) or (1, 1). N is number of objects per image
            if self.use_amp:
                gt_boxes = gt_boxes.half()
                image_size_xyxy = image_size_xyxy.half()
            if self.use_lsj:
                raise NotImplementedError
            else:
                gt_masks = targets_per_image.gt_bitmasks_full
            if self.use_amp:
                gt_masks = gt_masks.half()
            image_color_similarity = targets_per_image.image_color_similarity
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks, "image_size": image_size_xyxy, 
            "positive_map": positive_map, "image_color_similarity": image_color_similarity})
        return new_targets

    def inference(self, box_cls, box_pred, mask_pred, image_sizes, positive_map_label_to_token, num_classes, score_thres=0.0, task=None, iou_pred=None):
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
        elif task == "grounding":
            max_num_inst = 1
        else:
            raise ValueError("task must be detection or grounding")
        assert len(box_cls) == len(image_sizes)
        results = []

        for i, (logits_per_image, box_pred_per_image, image_size, iou_per_image) in enumerate(zip(
            box_cls, box_pred, image_sizes, iou_pred
        )):

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
                
                # pre-NMS for duplicate removal
                nms_scores,idxs = torch.max(prob,1)
                boxes_before_nms = box_cxcywh_to_xyxy(box_pred_per_image)
                keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.7)  
                prob = prob[keep_indices]
                box_pred_per_image = box_pred_per_image[keep_indices]
                if mask_pred is not None:
                    mask_pred_i = mask_pred[i][keep_indices]
                
                if not self.demo_only:
                    # from the remaining queries (N' x C), picking up topk
                    num_inst = min(num_inst, len(prob.view(-1)))
                    topk_values, topk_indexes = torch.topk(prob.view(-1), num_inst, dim=0)
                    scores = topk_values
                    topk_boxes = torch.div(topk_indexes, logits_per_image.shape[1], rounding_mode='floor')
                    # topk_boxes = topk_indexes // logits_per_image.shape[1]
                    labels = topk_indexes % logits_per_image.shape[1]
                    scores_per_image = scores
                    labels_per_image = labels

                    box_pred_per_image = box_pred_per_image[topk_boxes]
                    if mask_pred is not None:
                        mask_pred_i = mask_pred_i[topk_boxes]
                else:
                    # Demo Only
                    scores_per_image = nms_scores[keep_indices]
                    labels_per_image = idxs[keep_indices]
                    valid_indices = scores_per_image > score_thres
                    box_pred_per_image = box_pred_per_image[valid_indices]
                    scores_per_image = scores_per_image[valid_indices]
                    labels_per_image = labels_per_image[valid_indices]
                    mask_pred_i = mask_pred_i[valid_indices]
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
                if mask_pred is not None:
                    mask_pred_i = mask_pred[i][topk_boxes]
            
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            # import pdb;pdb.set_trace()
            if self.mask_on:
                N, C, H, W = mask_pred_i.shape
                mask = F.interpolate(mask_pred_i, size=(H*self.mask_stride, W*self.mask_stride), mode='bilinear', align_corners=False)
                mask = mask.sigmoid() > self.mask_thres
                # import pdb;pdb.set_trace()
                mask = mask[:,:,:image_size[0],:image_size[1]]
                result.pred_masks = mask
                
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        if self.use_lsj and self.training:
            image_sizes = [x["instances"].image_size for x in batched_inputs]
            input_masks = [x["padding_mask"].to(self.device) for x in batched_inputs]
            H, W = images[0].size()[-2:]
            images_new = torch.zeros((len(images), 3, H, W), device=self.device)
            for i in range(len(images)):
                h, w = image_sizes[i]
                images_new[i, :, :h, :w] = images[i][:, :h, :w]
            outputs = NestedTensor(images_new, torch.stack(input_masks, dim=0))
            outputs.image_sizes = image_sizes
            return outputs
        else:
            images = ImageList.from_tensors(images)
            return images

    def forward_text(self, captions, device, task=None):
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions,
                                                        max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, # 256
                                                        padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest", # max_length
                                                        return_special_tokens_mask=True,
                                                        return_tensors='pt',
                                                        truncation=True).to(device)

            tokenizer_input = {"input_ids": tokenized.input_ids,
                            "attention_mask": tokenized.attention_mask}
            if self.parallel_det:
                language_dict_features = self.text_encoder(tokenizer_input, task=task) # dict with keys: ['aggregate', 'embedded', 'masks', 'hidden']
            else:
                language_dict_features = self.text_encoder(tokenizer_input) # dict with keys: ['aggregate', 'embedded', 'masks', 'hidden']
            # language_dict_features["masks"] is equal to tokenizer_input["attention_mask"]
            # aggregate: (bs, 768), embedded: (bs, L, 768), masks: (bs, 768), hidden: (bs, L, 768) L=256 here
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return language_dict_features


    def prepare_image_targets_boxinst(self, batched_inputs, size_divisibility=32):
        original_images = [x["image"].to(self.device) for x in batched_inputs] # [tensor((3,H,W))] len=bs
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        # normalize images
        images_norm = [self.normalizer(x) for x in original_images]
        images_norm = ImageList.from_tensors(images_norm)

        original_image_masks = [torch.ones_like(x[0], dtype=torch.float32) for x in original_images] # [torch.ones(H, W),...] len=bs

        # mask out the bottom area where the COCO dataset probably has wrong annotations
        for i in range(len(original_image_masks)):
            im_h = batched_inputs[i]["height"]
            pixels_removed = int(
                self.bottom_pixels_removed *
                float(original_images[i].size(1)) / float(im_h)
            )
            if pixels_removed > 0:
                original_image_masks[i][-pixels_removed:, :] = 0

        original_images = ImageList.from_tensors(original_images, size_divisibility)
        original_image_masks = ImageList.from_tensors(
            original_image_masks, size_divisibility, pad_value=0.0
        ) # (bs, H, W) image=1, padding=0
        self.add_bitmasks_from_boxes(
            gt_instances, original_images.tensor, original_image_masks.tensor,
            original_images.tensor.size(-2), original_images.tensor.size(-1)
        )
        
        new_targets = self.prepare_targets_boxinst(gt_instances)
        
        return images_norm, new_targets


    def add_bitmasks_from_boxes(self, instances, images, image_masks, im_h, im_w):
        stride = self.mask_stride
        start = int(stride // 2)

        assert images.size(2) % stride == 0
        assert images.size(3) % stride == 0

        downsampled_images = F.avg_pool2d(
            images.float(), kernel_size=stride,
            stride=stride, padding=0
        )[:, [2, 1, 0]] # RGB-format original images (with padding) (bs, 3, H//4, W//4)
        image_masks = image_masks[:, start::stride, start::stride] # (bs, H//4, W//4)

        for im_i, per_im_gt_inst in enumerate(instances):
            images_lab = color.rgb2lab(downsampled_images[im_i].byte().permute(1, 2, 0).cpu().numpy()) # (H, W, 3)
            images_lab = torch.as_tensor(images_lab, device=downsampled_images.device, dtype=torch.float32)
            images_lab = images_lab.permute(2, 0, 1)[None] # (1, 3, H//4, W//4)
            images_color_similarity = get_images_color_similarity(
                images_lab, image_masks[im_i],
                self.pairwise_size, self.pairwise_dilation
            ) # (1, 8, H//4, W//4)

            per_im_boxes = per_im_gt_inst.gt_boxes.tensor
            per_im_bitmasks_full = []
            for per_box in per_im_boxes:
                bitmask_full = torch.zeros((im_h, im_w), device=self.device).float()
                bitmask_full[int(per_box[1]):int(per_box[3] + 1), int(per_box[0]):int(per_box[2] + 1)] = 1.0
                per_im_bitmasks_full.append(bitmask_full)

            per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0) # (N, H, W)
            per_im_gt_inst.image_color_similarity = torch.cat([
                images_color_similarity for _ in range(len(per_im_gt_inst)) # (N, 8, H//4, W//4)
            ], dim=0)


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


def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x


def get_images_color_similarity(images, image_masks, kernel_size, dilation):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    unfolded_weights = unfold_wo_center(
        image_masks[None, None], kernel_size=kernel_size,
        dilation=dilation
    )
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]

    return similarity * unfolded_weights