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

        # Language (text encoder and tokenizer)
        # Here we use BERT as the text encoder in a hard-code way
        self.tokenizer = AutoTokenizer.from_pretrained("projects/UNINEXT/bert-base-uncased")
        self.text_encoder = nn.Sequential(OrderedDict([("body", BertEncoder(cfg))]))
        if cfg.MODEL.FREEZE_TEXT_ENCODER:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        
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


    def forward(self, batched_inputs, frame_idx):
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
        video_len = len(batched_inputs[0]["image"])
        gt_instances = [batched_inputs[0]["instances"][0].to(self.device)]
        gt_targets = self.prepare_targets_test(gt_instances)
        init_file_name = batched_inputs[0]["file_names"][0]
        dataset_name = init_file_name.split("/")[1]
        if "video" in batched_inputs[0]:
            vid_name = batched_inputs[0]["video"]
        else:
            if dataset_name in ["LaSOT", "LaSOT_extension_subset", "TNL-2K"]:
                vid_name = init_file_name.split("/")[-3]
            elif dataset_name in ["GOT10K", "TrackingNet", "ytbvos18", "DAVIS"]:
                vid_name = init_file_name.split("/")[-2]
            elif dataset_name == "nfs":
                vid_name = "nfs_" + init_file_name.split("/")[-2]
            else:
                raise NotImplementedError
        # switch to VOS mode
        if dataset_name in ["ytbvos18", "DAVIS"]:
            if self.inference_on_3f:
                self.inference_ytbvos_3f(batched_inputs, task)
            else:
                self.inference_ytbvos(batched_inputs, task)
            return 

        height = batched_inputs[0]["height"]
        width = batched_inputs[0]["width"]
        clip_inputs = [{'image':batched_inputs[0]['image'][0:1]}]
        images = self.preprocess_video(clip_inputs)
        if frame_idx == 0:
            self.language_dict_features, _ = self.detr.coco_inference_ref(images, gt_targets)
            scale_x, scale_y = (
                width / images.image_sizes[0][1],
                height / images.image_sizes[0][0],
            )
            gt_instances[0].gt_boxes.scale(scale_x, scale_y)
            gt_instances[0].gt_boxes.clip((height, width))
            x1, y1, x2, y2 = gt_instances[0].gt_boxes.tensor.tolist()[0]
            self.language_dict_features_prev = copy.deepcopy(self.language_dict_features)
            return
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
            final_mask = F.interpolate(results_per_image.pred_masks.float(), size=(height, width), mode="bilinear", align_corners=False)
            final_mask = final_mask[0, 0].cpu().numpy() # (H, W)
            if self.debug_only:
                # debug
                ori_img = F.interpolate(batched_inputs[0]['image'][frame_idx].unsqueeze(0)[:, :, :images.image_sizes[0][0], :images.image_sizes[0][1]], size=(height, width))
                img_arr = ori_img[0].permute((1, 2, 0)).cpu().numpy()
                img_arr = np.ascontiguousarray(img_arr[:, :, ::-1]).clip(0, 255)
                img_arr_det = copy.deepcopy(img_arr)
                cv2.rectangle(img_arr_det, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,0,255), thickness=2)
                img_arr_det[:, :, -1] = np.clip(img_arr_det[:, :, -1] + 128 * final_mask, 0, 255)
                save_img_dir = os.path.join(save_dir, vid_name)
                if not os.path.exists(save_img_dir):
                    os.makedirs(save_img_dir)
                cv2.imwrite(os.path.join(save_img_dir, "%05d.jpg"%frame_idx), img_arr_det)
        if self.online_update and (frame_idx % self.update_interval == 0) and (results[0].scores > self.update_thr):
            # update the template
            bboxes_unorm = torch.tensor([[x1, y1, x2, y2]]) / torch.tensor([scale_x, scale_y, scale_x, scale_y])
            cur_targets = [{"bboxes_unorm": bboxes_unorm.to(self.device)}]
            self.language_dict_features_prev, new_template = self.detr.coco_inference_ref(images, cur_targets)
            if self.debug_only:
                self.debug_template_4c(new_template, vid_name, 1, frame_idx)
        
        return final_mask


    def preprocess_clip_image(self, batched_inputs, clip_idx=None):
        """
        Normalize, pad and batch the input images.
        """
        if clip_idx is None:
            images = []
            for video in batched_inputs:
                for frame in video["image"]:
                    images.append(self.normalizer(frame.to(self.device)))
            images = ImageList.from_tensors(images)
        else:
            images = []
            for video in batched_inputs:
                for idx in clip_idx:
                    images.append(self.normalizer(video["image"][idx].to(self.device)))
            images = ImageList.from_tensors(images)
        return images

    def inference_ytbvos(self, batched_inputs, task):
        init_file_name = batched_inputs[0]["file_names"][0]
        dataset_name = init_file_name.split("/")[1]
        if dataset_name == "ytbvos18":
            palette_img = "datasets/ytbvos18/val/Annotations/0a49f5265b/00000.png"
        elif dataset_name == "DAVIS":
            palette_img = "datasets/DAVIS/Annotations/480p/bear/00000.png"
        else:
            raise ValueError
        palette = Image.open(palette_img).getpalette()
        # inference on Youtube-VOS datasets
        positive_map_label_to_token = {1: [0]}
        num_classes = len(positive_map_label_to_token) # num_classes during testing
        # batchsize = 1 during inference
        height = batched_inputs[0]['height']
        width = batched_inputs[0]['width']
        video_len = len(batched_inputs[0]["image"])
        gt_instances = batched_inputs[0]["instances"]
        gt_instances = [x.to(self.device) for x in gt_instances]
        gt_targets = self.prepare_targets_test(gt_instances)
        assert len(gt_targets) == video_len
        file_names = batched_inputs[0]["file_names"]
        if "video" in batched_inputs[0]:
            vid_name = batched_inputs[0]["video"]
        else:
            if dataset_name == "LaSOT":
                vid_name = init_file_name.split("/")[-3]
            elif dataset_name in ["GOT10K", "TrackingNet", "ytbvos18", "DAVIS"]:
                vid_name = init_file_name.split("/")[-2]
            elif dataset_name == "nfs":
                vid_name = "nfs_" + init_file_name.split("/")[-2]
            else:
                raise NotImplementedError
        save_dir = os.path.join(self.cfg.OUTPUT_DIR, "inference", dataset_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        track_results = []
        height = batched_inputs[0]["height"]
        width = batched_inputs[0]["width"]
        language_dict_features_dict = {}
        mask_file_names = [x.split("/")[-1].replace(".jpg", ".png") for x in file_names]
        for frame_idx in range(video_len):
            clip_inputs = [{'image':batched_inputs[0]['image'][frame_idx:frame_idx+1]}]
            images = self.preprocess_video(clip_inputs)
            cur_gt_instances = gt_instances[frame_idx]
            if len(cur_gt_instances) > 0:
                # there are new objects appearing in this frame, initialize templates for them
                ref_bboxes = gt_targets[frame_idx]["bboxes_unorm"] # Tensor with size (N, 4)
                ref_masks = gt_targets[frame_idx]["masks"]
                cur_obj_ids = cur_gt_instances.ori_id
                assert len(ref_bboxes) == len(cur_obj_ids)
                num_new_obj = len(cur_obj_ids)
                for obj_idx in range(num_new_obj):
                    cur_obj_id = cur_obj_ids[obj_idx]
                    cur_ref_bboxes = [ref_bboxes[obj_idx:obj_idx+1]] # List (1, 4)
                    cur_ref_masks = [ref_masks[obj_idx:obj_idx+1]]
                    assert cur_obj_id not in language_dict_features_dict
                    language_dict_features_dict[cur_obj_id], new_template = self.detr.coco_inference_ref_vos(images, cur_ref_bboxes, cur_ref_masks)
                    if self.debug_only:
                        if self.extra_backbone_for_template:
                            self.debug_template_4c(new_template, int(cur_obj_id))
                        else:
                            self.debug_template(new_template, int(cur_obj_id))
            # track
            mask_dict = {} # store mask results of different obj_id on the current frame
            for obj_id, language_dict_features in language_dict_features_dict.items():
                language_dict_features_cur = copy.deepcopy(language_dict_features) # Important
                output, _ = self.detr.coco_inference(images, None, None, language_dict_features=language_dict_features_cur, task=task)
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                mask_pred = output["pred_masks"] if self.mask_on else [None] * len(batched_inputs)
                if self.detr.use_iou_branch:
                    iou_pred = output["pred_boxious"]
                else:
                    iou_pred = [None]
                results = self.inference(box_cls, box_pred, mask_pred, images.image_sizes, positive_map_label_to_token, num_classes, task=task, binary_mask=False, iou_pred=iou_pred)
                for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                    scale_x, scale_y = (
                        width / results_per_image.image_size[1],
                        height / results_per_image.image_size[0],
                    )
                    results_per_image.pred_boxes.scale(scale_x, scale_y)
                    results_per_image.pred_boxes.clip((height, width))
                    x1, y1, x2, y2 = results_per_image.pred_boxes.tensor.tolist()[0]
                    track_results.append([x1, y1, x2-x1, y2-y1]) # (x1, y1, w, h) format
                    # mask
                    final_mask = F.interpolate(results_per_image.pred_masks.float(), size=(height, width), mode="bilinear", align_corners=False)
                    mask_dict[obj_id] = final_mask[0, 0].cpu().numpy() # (H, W)
            # deal with objects appearing in the current frame
            if len(cur_gt_instances) > 0:
                # there are new objects appearing in this frame, replace predicted mask results with gts
                cur_obj_ids = cur_gt_instances.ori_id
                num_new_obj = len(cur_obj_ids)
                gt_masks = gt_targets[frame_idx]["masks"][:, :image_size[0], :image_size[1]]
                gt_masks = F.interpolate(gt_masks[None].float(), size=(height, width), mode="bilinear", align_corners=False).cpu().numpy()[0]
                for obj_idx in range(num_new_obj):
                    cur_obj_id = cur_obj_ids[obj_idx]
                    mask_dict[cur_obj_id] = gt_masks[obj_idx]
            # post-processing (soft-aggregation)
            cur_obj_ids = list(language_dict_features_dict.keys())
            cur_obj_ids_int = [int(x) for x in cur_obj_ids] # 1, 2, 3...
            if len(cur_obj_ids_int) != 0:
                mask_merge = np.zeros((height, width, max(cur_obj_ids_int)+1)) # (H, W, N+1)
            else:
                mask_merge = np.zeros((height, width, 1))
            tmp_list = []
            for cur_id in cur_obj_ids:
                mask_merge[:, :, int(cur_id)] = mask_dict[cur_id]
                tmp_list.append(mask_dict[cur_id])
            if len(tmp_list) != 0:
                back_prob = np.prod(1 - np.stack(tmp_list, axis=-1), axis=-1, keepdims=False)
                mask_merge[:, :, 0] = back_prob
            mask_merge_final = np.argmax(mask_merge, axis=-1).astype(np.uint8) # (H, W)
            mask_merge_final = Image.fromarray(mask_merge_final).convert('P')
            mask_merge_final.putpalette(palette)
            save_img_dir = os.path.join(save_dir, vid_name)
            if not os.path.exists(save_img_dir):
                os.makedirs(save_img_dir)
            mask_merge_final.save(os.path.join(save_img_dir, mask_file_names[frame_idx]))
        print("%s done."%vid_name)

    # inference based on three frames: the 1st frame, frame T-1, frame T   
    def inference_ytbvos_3f(self, batched_inputs, task):
        init_file_name = batched_inputs[0]["file_names"][0]
        dataset_name = init_file_name.split("/")[1]
        if dataset_name == "ytbvos18":
            palette_img = "datasets/ytbvos18/val/Annotations/0a49f5265b/00000.png"
        elif dataset_name == "DAVIS":
            palette_img = "datasets/DAVIS/Annotations/480p/bear/00000.png"
        else:
            raise ValueError
        palette = Image.open(palette_img).getpalette()
        # inference on Youtube-VOS datasets
        positive_map_label_to_token = {1: [0]}
        num_classes = len(positive_map_label_to_token) # num_classes during testing
        # batchsize = 1 during inference
        height = batched_inputs[0]['height']
        width = batched_inputs[0]['width']
        video_len = len(batched_inputs[0]["image"])
        gt_instances = batched_inputs[0]["instances"]
        gt_instances = [x.to(self.device) for x in gt_instances]
        gt_targets = self.prepare_targets_test(gt_instances)
        assert len(gt_targets) == video_len
        file_names = batched_inputs[0]["file_names"]
        if "video" in batched_inputs[0]:
            vid_name = batched_inputs[0]["video"]
        else:
            if dataset_name == "LaSOT":
                vid_name = init_file_name.split("/")[-3]
            elif dataset_name in ["GOT10K", "TrackingNet", "ytbvos18", "DAVIS"]:
                vid_name = init_file_name.split("/")[-2]
            elif dataset_name == "nfs":
                vid_name = "nfs_" + init_file_name.split("/")[-2]
            else:
                raise NotImplementedError
        save_dir = os.path.join(self.cfg.OUTPUT_DIR, "inference", dataset_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        track_results = []
        height = batched_inputs[0]["height"]
        width = batched_inputs[0]["width"]
        language_dict_features_dict_init = {}
        language_dict_features_dict_prev = {}
        mask_file_names = [x.split("/")[-1].replace(".jpg", ".png") for x in file_names]
        for frame_idx in range(video_len):
            clip_inputs = [{'image':batched_inputs[0]['image'][frame_idx:frame_idx+1]}]
            images = self.preprocess_video(clip_inputs)
            cur_gt_instances = gt_instances[frame_idx]
            cur_new_obj_ids = []
            score_dict = {}
            if len(cur_gt_instances) > 0:
                # there are new objects appearing in this frame, initialize templates for them
                ref_bboxes = gt_targets[frame_idx]["bboxes_unorm"] # Tensor with size (N, 4)
                ref_masks = gt_targets[frame_idx]["masks"]
                cur_obj_ids = cur_gt_instances.ori_id
                assert len(ref_bboxes) == len(cur_obj_ids)
                num_new_obj = len(cur_obj_ids)
                for obj_idx in range(num_new_obj):
                    cur_obj_id = cur_obj_ids[obj_idx]
                    cur_new_obj_ids.append(cur_obj_id)
                    cur_ref_bboxes = [ref_bboxes[obj_idx:obj_idx+1]] # List (1, 4)
                    cur_ref_masks = [ref_masks[obj_idx:obj_idx+1]]
                    assert cur_obj_id not in language_dict_features_dict_init
                    language_dict_features_dict_init[cur_obj_id], new_template = self.detr.coco_inference_ref_vos(images, cur_ref_bboxes, cur_ref_masks)
                    # copy to language_dict_features_dict_prev
                    language_dict_features_dict_prev[cur_obj_id] = copy.deepcopy(language_dict_features_dict_init[cur_obj_id])
                    if self.debug_only:
                        if self.extra_backbone_for_template:
                            self.debug_template_4c(new_template, vid_name, cur_obj_id, frame_idx)
                        else:
                            self.debug_template(new_template, int(cur_obj_id))
                    score_dict[cur_obj_id] = 1.0
            # track
            mask_dict = {} # store mask results of different obj_id on the current frame
            for obj_id in language_dict_features_dict_init.keys():
                language_dict_features_init = copy.deepcopy(language_dict_features_dict_init[obj_id]) # Important
                language_dict_features_prev = copy.deepcopy(language_dict_features_dict_prev[obj_id]) # Important
                language_dict_features_cur = {}
                language_dict_features_cur["hidden"] = torch.cat([language_dict_features_init["hidden"], language_dict_features_prev["hidden"]], dim=1)
                language_dict_features_cur["masks"] = torch.cat([language_dict_features_init["masks"], language_dict_features_prev["masks"]], dim=1)
                output, _ = self.detr.coco_inference(images, None, None, language_dict_features=language_dict_features_cur, task=task)
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                mask_pred = output["pred_masks"] if self.mask_on else [None] * len(batched_inputs)
                if self.detr.use_iou_branch:
                    iou_pred = output["pred_boxious"]
                else:
                    iou_pred = [None]
                results = self.inference(box_cls, box_pred, mask_pred, images.image_sizes, positive_map_label_to_token, num_classes, task=task, binary_mask=False, iou_pred=iou_pred)
                for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                    scale_x, scale_y = (
                        width / results_per_image.image_size[1],
                        height / results_per_image.image_size[0],
                    )
                    results_per_image.pred_boxes.scale(scale_x, scale_y)
                    results_per_image.pred_boxes.clip((height, width))
                    x1, y1, x2, y2 = results_per_image.pred_boxes.tensor.tolist()[0]
                    track_results.append([x1, y1, x2-x1, y2-y1]) # (x1, y1, w, h) format
                    # mask
                    if dataset_name == "ytbvos18" and results_per_image.scores.item() < self.inst_thr_vos:
                        mask_dict[obj_id] = np.zeros((height, width))
                    else:
                        final_mask = F.interpolate(results_per_image.pred_masks.float(), size=(height, width), mode="bilinear", align_corners=False)
                        mask_dict[obj_id] = final_mask[0, 0].cpu().numpy() # (H, W)
                    score_dict[obj_id] = results_per_image.scores.item()
            # deal with objects appearing in the current frame
            if len(cur_gt_instances) > 0:
                # there are new objects appearing in this frame, replace predicted mask results with gts
                cur_obj_ids = cur_gt_instances.ori_id
                num_new_obj = len(cur_obj_ids)
                gt_masks = gt_targets[frame_idx]["masks"][:, :image_size[0], :image_size[1]]
                gt_masks = F.interpolate(gt_masks[None].float(), size=(height, width), mode="bilinear", align_corners=False).cpu().numpy()[0]
                for obj_idx in range(num_new_obj):
                    cur_obj_id = cur_obj_ids[obj_idx]
                    mask_dict[cur_obj_id] = gt_masks[obj_idx]
                    score_dict[cur_obj_id] = 1.0
            # post-processing (soft-aggregation)
            cur_obj_ids = list(language_dict_features_dict_init.keys())
            cur_obj_ids_int = [int(x) for x in cur_obj_ids] # 1, 2, 3...
            if len(cur_obj_ids_int) != 0:
                mask_merge = np.zeros((height, width, max(cur_obj_ids_int)+1)) # (H, W, N+1)
            else:
                mask_merge = np.zeros((height, width, 1))
            tmp_list = []
            for cur_id in cur_obj_ids:
                mask_merge[:, :, int(cur_id)] = mask_dict[cur_id]
                tmp_list.append(mask_dict[cur_id])
            if len(tmp_list) != 0:
                back_prob = np.prod(1 - np.stack(tmp_list, axis=-1), axis=-1, keepdims=False)
                mask_merge[:, :, 0] = back_prob
            mask_merge = np.argmax(mask_merge, axis=-1).astype(np.uint8) # (H, W)
            mask_merge_final = Image.fromarray(mask_merge).convert('P')
            mask_merge_final.putpalette(palette)
            save_img_dir = os.path.join(save_dir, vid_name)
            if not os.path.exists(save_img_dir):
                os.makedirs(save_img_dir)
            mask_merge_final.save(os.path.join(save_img_dir, mask_file_names[frame_idx]))
            # update language_dict_features_dict_prev
            for cur_id in cur_obj_ids:
                if cur_id in cur_new_obj_ids:
                    continue
                if score_dict[cur_id] < self.update_thr:
                    continue
                cur_mask = (mask_merge == int(cur_id))
                try:
                    x1_c, y1_c, w_c, h_c = bounding_box(cur_mask) # current bounding box
                    cur_ref_bboxes = torch.tensor([x1_c, y1_c, x1_c+w_c, y1_c+h_c]).view(1, 4) / torch.tensor([scale_x, scale_y, scale_x, scale_y])
                    cur_ref_bboxes = [cur_ref_bboxes.to(self.device)] # List (1, 4)
                    cur_mask_tensor = torch.from_numpy(cur_mask).to(self.device)[None] # (1, H, W)
                    cur_mask_tensor_rsz = F.interpolate(cur_mask_tensor[None].float(), size=(image_size[0], image_size[1]))[0]
                    cur_mask_tensor_final = torch.zeros((1, images[0].size(-2), images[0].size(-1)), device=self.device)
                    cur_mask_tensor_final[:, :image_size[0], :image_size[1]] = cur_mask_tensor_rsz
                    cur_ref_masks = [cur_mask_tensor_final]
                    language_dict_features_dict_prev[cur_id], new_template = self.detr.coco_inference_ref_vos(images, cur_ref_bboxes, cur_ref_masks)
                    if self.debug_only:
                        self.debug_template_4c(new_template, vid_name, cur_id, frame_idx)
                except:
                    continue
        print("%s done."%vid_name)

    def debug_template_4c(self, samples, vid_name, obj_id, frame_idx):
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
        cv2.imwrite("%s_frame_%05d_obj_%s_img.jpg"%(vid_name, frame_idx, obj_id), image)
        cv2.imwrite("%s_frame_%05d_obj_%s_mask.jpg"%(vid_name, frame_idx, obj_id), input_mask)

    def prepare_targets(self, targets):
        new_targets = []
        # padding gt_masks to max size over a batch (This is important for training with img pairs)
        if hasattr(targets[0], "gt_masks"):
            # mask size: (n_inst, hm, wm)
            gt_masks_list = [x.gt_masks if self.use_lsj else x.gt_masks.tensor for x in targets]
            max_size = _max_by_axis([list(m.shape[1:]) for m in gt_masks_list])
            stride = 32 # size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size[-2] = (max_size[-2] + (stride - 1)) // stride * stride
            max_size[-1] = (max_size[-1] + (stride - 1)) // stride * stride
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            inst_ids = targets_per_image.gt_ids
            valid_id = inst_ids!=-1  # if a object is disappeared，its gt_ids is -1
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
                # add padding to masks
                n_inst, hm, wm = gt_masks.size()
                gt_masks_pad = torch.zeros((n_inst, max_size[0], max_size[1]), device=gt_masks.device, dtype=gt_masks.dtype)
                gt_masks_pad[:, :hm, :wm].copy_(gt_masks)
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks_pad, "image_size": image_size_xyxy, 
                "positive_map": positive_map, 'inst_id':inst_ids, "valid": valid_id, "bboxes_unorm": targets_per_image.gt_boxes.tensor})
            else:
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "image_size": image_size_xyxy, 
                "positive_map": positive_map, 'inst_id':inst_ids, "valid": valid_id, "bboxes_unorm": targets_per_image.gt_boxes.tensor})
        bz = len(new_targets) // 2
        key_ids = list(range(0, len(new_targets), 2))
        ref_ids = list(range(1, len(new_targets), 2))
        det_targets = [new_targets[_i] for _i in key_ids] # targets on key frames
        ref_targets = [new_targets[_i] for _i in ref_ids] # targets on ref frames
        for i in range(bz):  # fliter empety object in key frame. Note that det_loss is only computed on the key frame !
            det_target = det_targets[i]
            ref_target = ref_targets[i]
            if False in det_target['valid']:
                valid_i = det_target['valid'].clone()
                for k,v in det_target.items():
                    if k != "image_size":
                        det_target[k] = v[valid_i]
                for k,v in ref_target.items():
                    if k != "image_size":
                        ref_target[k] = v[valid_i]


        return det_targets,ref_targets

    def prepare_targets_test(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            # for language-guided detection, classification loss is computed based on the positive map
            positive_map = torch.ones((len(targets_per_image), 1), dtype=torch.bool, device=self.device) # (N, 256) or (1, 1). N is number of objects per image
            if hasattr(targets_per_image, "gt_masks"):
                # padding gt_masks to max size over a batch (This is important for training with img pairs)
                # mask size: (n_inst, hm, wm)
                gt_masks_list = [targets_per_image.gt_masks if self.use_lsj else targets_per_image.gt_masks.tensor]
                max_size = _max_by_axis([list(m.shape[1:]) for m in gt_masks_list])
                stride = 32 # size_divisibility
                # the last two dims are H,W, both subject to divisibility requirement
                max_size[-2] = (max_size[-2] + (stride - 1)) // stride * stride
                max_size[-1] = (max_size[-1] + (stride - 1)) // stride * stride
                if self.use_lsj:
                    gt_masks = targets_per_image.gt_masks
                else:
                    gt_masks = targets_per_image.gt_masks.tensor
                if self.use_amp:
                    gt_masks = gt_masks.half()
                # add padding to masks
                n_inst, hm, wm = gt_masks.size()
                gt_masks_pad = torch.zeros((n_inst, max_size[0], max_size[1]), device=gt_masks.device, dtype=gt_masks.dtype)
                gt_masks_pad[:, :hm, :wm].copy_(gt_masks)
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks_pad, "image_size": image_size_xyxy, 
                "positive_map": positive_map, "bboxes_unorm": targets_per_image.gt_boxes.tensor})
            else:
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "image_size": image_size_xyxy, 
                "positive_map": positive_map, "bboxes_unorm": targets_per_image.gt_boxes.tensor})

        return new_targets

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

    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions,
                                                        max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, # 256
                                                        padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest", # max_length
                                                        return_special_tokens_mask=True,
                                                        return_tensors='pt',
                                                        truncation=True).to(device)

            tokenizer_input = {"input_ids": tokenized.input_ids,
                            "attention_mask": tokenized.attention_mask}
            language_dict_features = self.text_encoder(tokenizer_input) # dict with keys: ['aggregate', 'embedded', 'masks', 'hidden']
            # language_dict_features["masks"] is equal to tokenizer_input["attention_mask"]
            # aggregate: (bs, 768), embedded: (bs, L, 768), masks: (bs, 768), hidden: (bs, L, 768) L=256 here
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return language_dict_features


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

def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]

def track2result(bboxes, labels, ids, num_classes):
    valid_inds = ids > -1
    bboxes = bboxes[valid_inds]
    labels = labels[valid_inds]
    ids = ids[valid_inds]

    if bboxes.shape[0] == 0:
        return [np.zeros((0, 6), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
            labels = labels.cpu().numpy()
            ids = ids.cpu().numpy()
        return [
            np.concatenate((ids[labels == i, None], bboxes[labels == i, :]),
                           axis=1) for i in range(num_classes)
        ]

def segtrack2result(bboxes, labels, segms, ids):
    valid_inds = ids > -1
    bboxes = bboxes[valid_inds].cpu().numpy()
    labels = labels[valid_inds].cpu().numpy()
    segms = [segms[i] for i in range(len(segms)) if valid_inds[i] == True]
    ids = ids[valid_inds].cpu().numpy()

    outputs = defaultdict(list)
    for bbox, label, segm, id in zip(bboxes, labels, segms, ids):
        outputs[id] = dict(bbox=bbox, label=label, segm=segm)
    return outputs

def encode_track_results(track_results):
    """Encode bitmap mask to RLE code.

    Args:
        track_results (list | tuple[list]): track results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).

    Returns:
        list | tuple: RLE encoded mask.
    """
    for id, roi in track_results.items():
        roi['segm'] = mask_util.encode(
            np.array(roi['segm'][:, :, np.newaxis], order='F',
                     dtype='uint8'))[0]  # encoded with RLE
    return track_results