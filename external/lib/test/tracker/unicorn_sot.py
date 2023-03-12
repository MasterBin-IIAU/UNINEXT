# Copyright (c) 2022 ByteDance. All Rights Reserved.
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.test.tracker.basetracker import BaseTracker
# from unicorn.utils.boxes import postprocess
# from unicorn.exp import get_exp
import copy
"""Inference code for SOT datasets"""

class UnicornSOTTrack(BaseTracker):
    def __init__(self, params, dataset_name) -> None:
        super(UnicornSOTTrack, self).__init__(params)
        """ model config """
        self.num_classes = 1
        exp_file_name = "exps/default/%s"%params.exp_name
        exp = get_exp(exp_file_name, None)
        self.normalize = exp.normalize
        self.exp_name = params.exp_name
        self.input_size = exp.test_size
        self.confthre = 0.001 # lead to High recall
        self.nmsthre = 0.65
        self.max_inst = 3
        """ build network and load state dict"""
        self.model = exp.get_model(load_pretrain=False)
        print('Loading weights:', params.checkpoint)
        self.model.load_state_dict(torch.load(params.checkpoint, map_location="cpu")["model"], strict=False)
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
        # forward the reference frame once
        """resize the original image and transform the coordinates"""
        ref_frame_t, r = self.preprocessor.process(image, self.input_size)
        init_box = torch.tensor(info["init_bbox"]).view(-1)
        init_box[2:] += init_box[:2] # (x1, y1, x2, y2)
        init_box_rsz = init_box * r # coordinates on the resized image
        """forward the network"""
        with torch.no_grad():
            _, self.out_dict_pre = self.model(imgs=ref_frame_t, mode="backbone")  # backbone output (previous frame) (b, 3, H, W)
        self.dh, self.dw = self.out_dict_pre["h"] * 2, self.out_dict_pre["w"] * 2  # STRIDE = 8
        """get initial label mask (K, H/8*W/8)"""
        self.lbs_pre = F.interpolate(get_label_map(init_box_rsz, self.input_size[0], self.input_size[1]) \
            , scale_factor=1/8, mode="bilinear", align_corners=False)[0].flatten(-2).to(self.device) # (1, H/8*W/8)
        # save states
        self.state = info['init_bbox']
    
    def track(self, image, info: dict = None, bboxes=None, scores=None, gt_box=None):
        self.frame_id += 1
        """resize the original image and transform the coordinates"""
        cur_frame_t, r = self.preprocessor.process(image, self.input_size)
        """get detection results"""
        output = self.get_det_results(cur_frame_t)
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
            self.state = list(bboxes_xywh[best_idx])
        return {"target_bbox": self.state}
    
    def get_det_results(self, cur_frame_t):
        """forward the network"""
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                fpn_outs_cur, out_dict_cur = self.model(imgs=cur_frame_t, mode="backbone")  # backbone output (current frame)  # backbone output (current frame)
                """ feature interaction """
                new_feat_pre, new_feat_cur = self.model(seq_dict0=self.out_dict_pre, seq_dict1=out_dict_cur, mode="interaction")
                """ up-sampling"""
                embed_map_pre = self.model(feat=new_feat_pre, mode="upsample")
                embed_map_cur = self.model(feat=new_feat_cur, mode="upsample")
                embed_pre = embed_map_pre.flatten(-2).squeeze()  # (1, C, H/8*W/8) --> (C, H/8*W/8)
                embed_cur = embed_map_cur.flatten(-2).squeeze()  # (1, C, H/8*W/8) --> (C, H/8*W/8)
                # memory management
                keys = embed_pre
                values = self.lbs_pre
                """Compute Correspondence and propagate masks"""
                # use FP16 for propagation (speed up inference)
                keys = keys.half()
                embed_cur = embed_cur.half()
                values = values.half()
                simi_mat = torch.mm(keys.transpose(1, 0), embed_cur)  # (N*H/8*W/8, H/8*W/8)
                trans_mat = torch.softmax(simi_mat, dim=0)  # transfer matrix # (N*H/8*W/8, H/8*W/8)
                cur_pred = values @ trans_mat  # (K, H/8*W/8)
                coarse_m = cur_pred.view(1, -1, self.dh, self.dw) # (1, K, H/8, W/8)
                coarse_m = coarse_m.float()
                coarse_m_ms = (coarse_m, 
                F.interpolate(coarse_m, scale_factor=1/2, mode="bilinear", align_corners=False),
                F.interpolate(coarse_m, scale_factor=1/4, mode="bilinear", align_corners=False)) # [8, 16, 32]
                outputs = self.model.head(fpn_outs_cur, coarse_m_ms, mode="sot")
                """finish NMS and get the final detection predictions"""
                output = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)[0]
        return output

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
    return UnicornSOTTrack

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
