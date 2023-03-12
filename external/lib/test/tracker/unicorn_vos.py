# Copyright (c) 2022 ByteDance. All Rights Reserved.
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.test.tracker.basetracker import BaseTracker
from unicorn.utils.boxes import postprocess_inst
from unicorn.exp import get_exp
import copy
"""Inference code for VOS datasets"""

class UnicornVOSTrack(BaseTracker):
    def __init__(self, params, dataset_name) -> None:
        super(UnicornVOSTrack, self).__init__(params)
        self.soft_aggregate = True
        """ model config """
        self.num_classes = 1
        exp_file_name = "exps/default/%s"%params.exp_name
        exp = get_exp(exp_file_name, None)
        self.normalize = exp.normalize
        self.exp_name = params.exp_name
        self.input_size = exp.test_size
        self.confthre = 0.001 # lead to High recall
        self.nmsthre = 0.65
        self.max_inst = 1
        self.mask_thres = 0.30
        self.d_rate = exp.d_rate
        self.use_raft = exp.use_raft
        """ build network and load state dict"""
        self.model = exp.get_model(load_pretrain=False)
        print('Loading weights:', params.checkpoint)
        self.model.load_state_dict(torch.load(params.checkpoint, map_location="cpu")["model"])
        self.model.cuda()
        self.device = "cuda"
        self.model.eval()
        """ Others """
        self.preprocessor = PreprocessorX(normalize=self.normalize) # use normalization
        self.state = None
        # for debug
        self.frame_id = 0

    def initialize(self, image, info: dict):
        self.frame_id = 0
        # process init_info
        self.init_object_ids = info["init_object_ids"]
        self.sequence_object_ids = info['sequence_object_ids']
        # assert self.init_object_ids == self.sequence_object_ids
        # forward the reference frame once
        """resize the original image and transform the coordinates"""
        self.H, self.W, _ = image.shape
        ref_frame_t, r = self.preprocessor.process(image, self.input_size)
        """forward the network"""
        with torch.no_grad():
            _, self.out_dict_pre = self.model(imgs=ref_frame_t, mode="backbone")  # backbone output (previous frame) (b, 3, H, W)
        self.dh, self.dw = self.out_dict_pre["h"] * 2, self.out_dict_pre["w"] * 2  # STRIDE = 8
        """get initial label mask (K, H/8*W/8)"""
        self.lbs_pre_dict = {}
        self.state_pre_dict = {}
        for obj_id in self.init_object_ids:
            self.state_pre_dict[obj_id] = info["init_bbox"][obj_id]
            init_box = torch.tensor(info["init_bbox"][obj_id]).view(-1)
            init_box[2:] += init_box[:2] # (x1, y1, x2, y2)
            init_box_rsz = init_box * r # coordinates on the resized image
            self.lbs_pre_dict[obj_id] = F.interpolate(get_label_map(init_box_rsz, self.input_size[0], self.input_size[1]) \
                , scale_factor=1/8, mode="bilinear", align_corners=False)[0].flatten(-2).to(self.device) # (1, H/8*W/8)
        """deal with new-incoming instances"""
        self.out_dict_pre_new = [] # a list containing out_dict for new in-coming instances
        self.obj_ids_new = []
    
    def track(self, image, info: dict = None, bboxes=None, scores=None, gt_box=None):
        self.frame_id += 1
        """resize the original image and transform the coordinates"""
        cur_frame_t, r = self.preprocessor.process(image, self.input_size)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                fpn_outs_cur, out_dict_cur = self.model(imgs=cur_frame_t, mode="backbone")  # backbone output (current frame)
        # deal with instances from the first frame
        final_mask_dict, inst_scores = self.get_mask_results(fpn_outs_cur, out_dict_cur, self.out_dict_pre, r, self.init_object_ids)
        # deal with instances from the intermediate frames
        for (out_dict_pre, init_object_ids) in zip(self.out_dict_pre_new, self.obj_ids_new):
            final_mask_dict_tmp, inst_scores_tmp = self.get_mask_results(fpn_outs_cur, out_dict_cur, out_dict_pre, r, init_object_ids)
            final_mask_dict.update(final_mask_dict_tmp)
            inst_scores = np.concatenate([inst_scores, inst_scores_tmp])
        # deal with instances from the current frame"""
        if "init_object_ids" in info.keys():
            self.out_dict_pre_new.append(out_dict_cur)
            self.obj_ids_new.append(info["init_object_ids"])
            inst_scores_tmp = np.ones((len(info["init_object_ids"]),))
            inst_scores = np.concatenate([inst_scores, inst_scores_tmp])
            for obj_id in info["init_object_ids"]:
                self.state_pre_dict[obj_id] = info["init_bbox"][obj_id]
                init_box = torch.tensor(info["init_bbox"][obj_id]).view(-1)
                init_box[2:] += init_box[:2] # (x1, y1, x2, y2)
                init_box_rsz = init_box * r # coordinates on the resized image
                self.lbs_pre_dict[obj_id] = F.interpolate(get_label_map(init_box_rsz, self.input_size[0], self.input_size[1]) \
                    , scale_factor=1/8, mode="bilinear", align_corners=False)[0].flatten(-2).to(self.device) # (1, H/8*W/8)
                final_mask_dict[obj_id] = (info["init_mask"] == int(obj_id))
        # Deal with overlapped masks
        cur_obj_ids = copy.deepcopy(self.init_object_ids)
        for obj_ids_inter in self.obj_ids_new:
            cur_obj_ids += obj_ids_inter
        if "init_object_ids" in info.keys():
            cur_obj_ids += info["init_object_ids"]
        # soft aggregation
        cur_obj_ids_int = [int(x) for x in cur_obj_ids]
        mask_merge = np.zeros((self.H, self.W, max(cur_obj_ids_int)+1)) # (H, W, N+1)
        tmp_list = []
        for cur_id in cur_obj_ids:
            mask_merge[:, :, int(cur_id)] = final_mask_dict[cur_id]
            tmp_list.append(final_mask_dict[cur_id])
        back_prob = np.prod(1 - np.stack(tmp_list, axis=-1), axis=-1, keepdims=False)
        mask_merge[:, :, 0] = back_prob
        mask_merge_final = np.argmax(mask_merge, axis=-1) # (H, W)
        for cur_id in cur_obj_ids:
            final_mask_dict[cur_id] = (mask_merge_final == int(cur_id))
        """get the final result"""
        final_mask = np.zeros((self.H, self.W), dtype=np.uint8)
        for obj_id in cur_obj_ids:
            final_mask[final_mask_dict[obj_id]==1] = int(obj_id)
        return {"segmentation": final_mask}
    
    def get_mask_results(self, fpn_outs_cur, out_dict_cur, out_dict_pre, r, init_object_ids):
        """get detection results"""
        output_dict, output_mask_dict = self.get_det_results(fpn_outs_cur, out_dict_cur, out_dict_pre, init_object_ids)
        final_mask_dict = {}
        inst_scores = np.zeros((len(init_object_ids),)) # (N, )
        for i, obj_id in enumerate(init_object_ids):
            output = output_dict[obj_id]
            # assert output is not None
            """map the coordinates back to the original image"""
            if output is not None:
                output[:, 0:4:2] = output[:, 0:4:2].clamp(min=0, max=self.input_size[1])
                output[:, 1:4:2] = output[:, 1:4:2].clamp(min=0, max=self.input_size[0])      
                output = output.cpu().numpy()
                if len(output) > self.max_inst:
                    output = output[:self.max_inst]
                bboxes = output[:, 0:4]
                scores = output[:, 4] * output[:, 5]
                best_idx = 0
                bboxes /= r
                bboxes_xywh = xyxy2xywh_np(bboxes).astype(np.int)
                self.state_pre_dict[obj_id] = list(bboxes_xywh[best_idx])
                inst_scores[i] = scores[best_idx]
                """deal with masks"""
                output_mask = output_mask_dict[obj_id]
                # soft aggregation
                masks = F.interpolate(output_mask, scale_factor=1/r, mode="bilinear", align_corners=False)\
            [:, 0, :self.H, :self.W] # (N, height, width)
                final_mask_dict[obj_id] = np.zeros((self.H, self.W), dtype=np.float32)
                h_m, w_m = masks.size()[1:]
                final_mask_dict[obj_id][:h_m, :w_m] = masks[best_idx].cpu().numpy()
            else:
                final_mask_dict[obj_id] = np.zeros((self.H, self.W), dtype=np.uint8)
        return final_mask_dict, inst_scores

    def get_det_results(self, fpn_outs_cur, out_dict_cur, out_dict_pre, init_object_ids):
        """forward the network"""
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                """ feature interaction """
                new_feat_pre, new_feat_cur = self.model(seq_dict0=out_dict_pre, seq_dict1=out_dict_cur, mode="interaction")
                """ up-sampling"""
                embed_map_pre = self.model(feat=new_feat_pre, mode="upsample")
                embed_map_cur = self.model(feat=new_feat_cur, mode="upsample")
                embed_pre = embed_map_pre.flatten(-2).squeeze()  # (1, C, H/8*W/8) --> (C, H/8*W/8)
                embed_cur = embed_map_cur.flatten(-2).squeeze()  # (1, C, H/8*W/8) --> (C, H/8*W/8)
                keys = embed_pre
                """Compute Correspondence"""
                # use FP16 for propagation (speed up inference)
                keys = keys.half()
                embed_cur = embed_cur.half()
                simi_mat = torch.mm(keys.transpose(1, 0), embed_cur)  # (N*H/8*W/8, H/8*W/8)
                trans_mat = torch.softmax(simi_mat, dim=0)  # transfer matrix # (N*H/8*W/8, H/8*W/8)
                """propagate masks"""
                output_dict = {}
                output_mask_dict = {}
                for obj_id in init_object_ids:
                    values = self.lbs_pre_dict[obj_id]
                    values = values.half()
                    cur_pred = values @ trans_mat  # (K, H/8*W/8)
                    coarse_m = cur_pred.view(1, -1, self.dh, self.dw) # (1, K, H/8, W/8)
                    coarse_m = coarse_m.float()
                    coarse_m_ms = (coarse_m, 
                    F.interpolate(coarse_m, scale_factor=1/2, mode="bilinear", align_corners=False),
                    F.interpolate(coarse_m, scale_factor=1/4, mode="bilinear", align_corners=False)) # [8, 16, 32]
                    head = self.model.head
                    if self.use_raft:
                        outputs, locations, dynamic_params, fpn_levels, mask_feats, up_masks = head(fpn_outs_cur, coarse_m_ms, mode="sot")
                        up_mask_b = up_masks[0:1]
                        outputs, outputs_mask = postprocess_inst(
                            outputs, locations, dynamic_params, fpn_levels, mask_feats, head.mask_head,
                            self.num_classes, self.confthre, self.nmsthre, class_agnostic=False, d_rate=self.d_rate, up_masks=up_mask_b)
                    else:
                        outputs, locations, dynamic_params, fpn_levels, mask_feats = head(fpn_outs_cur, coarse_m_ms, mode="sot")
                        outputs, outputs_mask = postprocess_inst(
                            outputs, locations, dynamic_params, fpn_levels, mask_feats, head.mask_head,
                            self.num_classes, self.confthre, self.nmsthre, class_agnostic=False, d_rate=self.d_rate)
                    output_dict[obj_id] = outputs[0] # (N, 6)
                    output_mask_dict[obj_id] = outputs_mask[0] # (N, 1, H, W)
        return output_dict, output_mask_dict


class PreprocessorX(object):
    def __init__(self, normalize=False):
        self.normalize = normalize
    def process(self, img_arr: np.ndarray, input_size: tuple):
        # resize and padding
        height, width = img_arr.shape[:2]
        r = min(input_size[0] / height, input_size[1] / width)
        # RGB2BGR, to tensor, swap dims
        img_arr_rsz = cv2.resize(cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR), (int(width * r), int(height * r)), interpolation=cv2.INTER_LINEAR)
        img_tensor_rsz = torch.tensor(img_arr_rsz, device="cuda", dtype=torch.float).permute((2, 0, 1)).unsqueeze(0) # (1, 3, H, W)
        padded_img_tensor = torch.full((1, 3, input_size[0], input_size[1]), 114, device="cuda", dtype=torch.float)
        padded_img_tensor[:, :, :int(height * r), :int(width * r)] = img_tensor_rsz
        return padded_img_tensor, r

def get_tracker_class():
    return UnicornVOSTrack

def get_label_map(boxes, H, W):
    """target: (4, )"""
    # boxes are in xyxy format
    labels = torch.zeros((1, 1, H, W), dtype=torch.float32).cuda()
    x1, y1, x2, y2 = torch.round(boxes).int().tolist()
    # The following processing is very important!
    x1 = max(0, min(x1, W))
    y1 = max(0, min(y1, H))
    x2 = max(0, min(x2, W))
    y2 = max(0, min(y2, H))
    labels[0, 0, y1:y2, x1:x2] = 1.0
    return labels # (1, 1, H, W)

def xyxy2xywh_np(bboxes: np.array):
    bboxes_new = copy.deepcopy(bboxes)
    bboxes_new[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes_new[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes_new
