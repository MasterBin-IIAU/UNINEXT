# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
import torchvision.ops as ops
from ...util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, generalized_multi_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 cost_mask: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_mask != 0, "all costs cant be 0"

    def forward_ota(self, outputs, targets, nf=1):
        """ simOTA for detr
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            out_prob = outputs["pred_logits"].sigmoid()
            out_bbox = outputs["pred_boxes"]
            indices = []
            matched_ids = []
            for batch_idx in range(bs):
                cost, pair_wise_ious, bz_gtboxs = self.compute_cost(batch_idx, out_bbox, out_prob, targets, nf)
                if bz_gtboxs.shape[0]>0:
                    indices_batchi, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs.shape[0])
                else:
                    # non_valid = torch.zeros(pair_wise_ious.shape[0]).to(pair_wise_ious)>0
                    # indices_batchi = (non_valid,torch.arange(0,0).to(pair_wise_ious))
                    indices_i = torch.tensor([], dtype=torch.int64).to(out_prob.device)
                    indices_j = torch.tensor([], dtype=torch.int64).to(out_prob.device)
                    indices_batchi = (indices_i, indices_j)
                    matched_qidx = []
                # import pdb;pdb.set_trace()
                indices.append(indices_batchi)
                matched_ids.append(matched_qidx)

        
        return indices, matched_ids

    def compute_cost(self, batch_idx, out_bbox, out_prob, targets, nf):
        bz_boxes = out_bbox[batch_idx] #[300,4]
        bz_out_prob = out_prob[batch_idx] 
        bz_tgt_ids = targets[batch_idx]["labels"]
        num_insts = len(bz_tgt_ids)
        bz_gtboxs = targets[batch_idx]['boxes'].reshape(num_insts,nf,4)[:,0] #[num_gt, 4]
        # import pdb;pdb.set_trace()
        # 这里的strides 在ddetr上是FPN输出的最小分辨率stride，可能需要调整. 得到的只是center的先验
        fg_mask, is_in_boxes_and_center  = \
            self.get_in_boxes_info(bz_boxes,bz_gtboxs,expanded_strides=32)
        # bboxes_preds_per_image = bz_boxes[fg_mask]
        # cls_preds_ = out_prob[batch_idx][fg_mask]
        # num_in_boxes_anchor = bboxes_preds_per_image.shape[0]
        pair_wise_ious = ops.box_iou(box_cxcywh_to_xyxy(bz_boxes), box_cxcywh_to_xyxy(bz_gtboxs))
        # pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(bz_boxes),  box_cxcywh_to_xyxy(bz_gtboxs))

        cost = ( cost_class + 3.0 * cost_giou + 100.0 * (~is_in_boxes_and_center) )  #[num_query,num_gt]
        # import pdb;pdb.set_trace()
        cost[~fg_mask] = cost[~fg_mask] + 10000.0
        return cost, pair_wise_ious, bz_gtboxs

    def get_in_boxes_info(self, boxes, target_gts, expanded_strides):
        # size (h,w) 
        # size = size[[1,0]].repeat(2) # (w,h,w,h)

        # import pdb;pdb.set_trace()
        # ori_gt_boxes = target_gts*size
        xy_target_gts = box_cxcywh_to_xyxy(target_gts) #x1y1x2y2
        
        anchor_center_x = boxes[:,0].unsqueeze(1)
        anchor_center_y = boxes[:,1].unsqueeze(1)

        # 判断每个anchor 的中心是否在某个gt box 内部
        b_l = anchor_center_x > xy_target_gts[:,0].unsqueeze(0)  # x1 满足要求
        b_r = anchor_center_x < xy_target_gts[:,2].unsqueeze(0)  # x2 满足要求
        b_t = anchor_center_y > xy_target_gts[:,1].unsqueeze(0)
        b_b = anchor_center_y < xy_target_gts[:,3].unsqueeze(0)
        # (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4 [300,num_gt] , 等于4表示四个条件都满足，所以只要每个query 对应的num_gt 个里面有一个4则说明该query 有效
        is_in_boxes = ( (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4)
        is_in_boxes_all = is_in_boxes.sum(1)>0  # [num_query]

        #每个gt的cx与cy向外扩展2.5*expanded_strides距离得到left_b,right_b,top_b,bottom_b，
        # 与anchor进行比较，计算anchor中心点是否包含在left_b,right_b,top_b,bottom_b中，得到 is_in_centers_all (shape:[num_anchors])
        # in fixed center
        center_radius = 2.5
        b_l = anchor_center_x > (target_gts[:,0]-(1*center_radius/expanded_strides)).unsqueeze(0)  # x1 满足要求
        b_r = anchor_center_x < (target_gts[:,0]+(1*center_radius/expanded_strides)).unsqueeze(0)  # x2 满足要求
        b_t = anchor_center_y > (target_gts[:,1]-(1*center_radius/expanded_strides)).unsqueeze(0)
        b_b = anchor_center_y < (target_gts[:,1]+(1*center_radius/expanded_strides)).unsqueeze(0)
        is_in_centers = ( (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4)
        is_in_centers_all = is_in_centers.sum(1)>0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all      # 上述两个条件只要满足一个，就成为候选区域。注意！！！这里是“|”求或

        # ！！！shape:[num_gt, num_in_boxes_anchor]，注意：这里是每一个gt与每一个候选区域的关系
        # 这里一个anchor可能与多个gt存在候选关系
        # is_in_boxes_anchor 里两者都满足的anchor
        # import pdb;pdb.set_trace()
        # is_in_boxes_and_center = (is_in_boxes[is_in_boxes_anchor, :] & is_in_centers[is_in_boxes_anchor,:] )       

        is_in_boxes_and_center = (is_in_boxes & is_in_centers)   

        return is_in_boxes_anchor,is_in_boxes_and_center
    
    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt):
        matching_matrix = torch.zeros_like(cost) # [300,num_gt] matching_matrix表示映关系，必须保证只能1gt对多query，且每个gt 有query
        ious_in_boxes_matrix = pair_wise_ious
        n_query = len(ious_in_boxes_matrix)
        n_candidate_k = min(n_query, 10)
        
        # 取预测值与gt拥有最大iou前10名的iou总和作为dynamic_k
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)
        # min=1,即把dynamic_ks限制最小为1，保证一个gt至少有一个正样本
        # 刚开始训练时候，由于预测基本不准，导致dynamic_k基本上都是1
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
        # 取cost排名最小的前dynamic_k个anchor作为postive
            _, pos_idx = torch.topk(cost[:,gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:,gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1)
        
        if (anchor_matching_gt > 1).sum() > 0: # 如果有一个query 匹配上了多个gt
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1) #  对这些query找cost 最小的gt
            matching_matrix[anchor_matching_gt > 1] *= 0 # 清零这些query的映射关系
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1 # 只保留cost 最小的gt 

        # import pdb;pdb.set_trace()
        ## 注意！此时删除了一些映射关系可能导致有些gt 没匹配上任何query, 此时通过while 循环重复上述过程直至所有gt 都匹配上
        while (matching_matrix.sum(0)==0).any(): # 只要有gt 没匹配上
            num_zero_gt = (matching_matrix.sum(0)==0).sum()
            matched_query_id = matching_matrix.sum(1)>0
            cost[matched_query_id] += 100000.0 
            unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:,gt_idx])
                matching_matrix[:,gt_idx][pos_idx] = 1.0
            # 继续判断是否有 多gt 对1 query的情况
            if (matching_matrix.sum(1) > 1).sum() > 0: # 如果有一个query 匹配上了多个gt
                _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1) #  对这些query找cost 最小的gt
                matching_matrix[anchor_matching_gt > 1] *= 0 # 清零这些query的映射关系
                matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1 # 只保留cost 最小的gt 

        assert not (matching_matrix.sum(0)==0).any() # 所有gt 都匹配上query
        # 此时 matching_matrix 是一个[300,num_gt] 的0-1矩阵
        # 
        selected_query = matching_matrix.sum(1)>0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        assert selected_query.sum() == len(gt_indices)

        # import pdb;pdb.set_trace()# 此处需要额外返回，跟每个gt 最匹配的query idx
        # matched_cost = ((matching_matrix==0)+cost) # 找出所有没匹配上的cost，加上极大值，找到每个gt最小的cost
        # matched_query_id  = torch.min(matched_cost,dim=0)[1]
        cost[matching_matrix==0] = cost[matching_matrix==0] + float('inf')
        matched_query_id = torch.min(cost,dim=0)[1]

        # in this version, the selected_query is [300,] with true/false
        # we convert it to value indices
        matched_anchor_inds = torch.arange(len(matching_matrix)).to(gt_indices)
        selected_query = matched_anchor_inds[selected_query]     # [num_pos,]

        return (selected_query,gt_indices) , matched_query_id

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

"""Hungarian Matcher for language-guided object detection"""
class HungarianMatcherVL(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 cost_mask: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_mask != 0, "all costs cant be 0"

    def forward_ota(self, outputs, targets, nf=1):
        """ simOTA for detr
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            out_prob = outputs["pred_logits"].sigmoid()
            out_bbox = outputs["pred_boxes"]
            indices = []
            matched_ids = []
            for batch_idx in range(bs):
                cost, pair_wise_ious, bz_gtboxs = self.compute_cost(batch_idx, out_bbox, out_prob, targets, nf)
                if bz_gtboxs.shape[0]>0:
                    indices_batchi, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs.shape[0])
                else:
                    # non_valid = torch.zeros(pair_wise_ious.shape[0]).to(pair_wise_ious)>0
                    # indices_batchi = (non_valid,torch.arange(0,0).to(pair_wise_ious))
                    indices_i = torch.tensor([], dtype=torch.int64).to(out_prob.device)
                    indices_j = torch.tensor([], dtype=torch.int64).to(out_prob.device)
                    indices_batchi = (indices_i, indices_j)
                    matched_qidx = []
                # import pdb;pdb.set_trace()
                indices.append(indices_batchi)
                matched_ids.append(matched_qidx)

        
        return indices, matched_ids

    def compute_cost(self, batch_idx, out_bbox, out_prob, targets, nf):
        bz_boxes = out_bbox[batch_idx] #[300,4]
        bz_out_prob = out_prob[batch_idx] 
        bz_tgt_ids = targets[batch_idx]["positive_map"] # (N_obj, 256) or (N_obj, 1)
        num_insts = len(bz_tgt_ids)
        bz_gtboxs = targets[batch_idx]['boxes'].reshape(num_insts,nf,4)[:,0] #[num_gt, 4]
        # import pdb;pdb.set_trace()
        # 这里的strides 在ddetr上是FPN输出的最小分辨率stride，可能需要调整. 得到的只是center的先验
        fg_mask, is_in_boxes_and_center  = \
            self.get_in_boxes_info(bz_boxes,bz_gtboxs,expanded_strides=32)
        # bboxes_preds_per_image = bz_boxes[fg_mask]
        # cls_preds_ = out_prob[batch_idx][fg_mask]
        # num_in_boxes_anchor = bboxes_preds_per_image.shape[0]
        pair_wise_ious = ops.box_iou(box_cxcywh_to_xyxy(bz_boxes), box_cxcywh_to_xyxy(bz_gtboxs))
        # pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log())
        cost_class = torch.zeros((bz_out_prob.size(0), bz_tgt_ids.size(0)), device=out_prob.device)
        for idx in range(bz_tgt_ids.size(0)):
            cost_class[:, idx] = (pos_cost_class[:, bz_tgt_ids[idx]] - neg_cost_class[:, bz_tgt_ids[idx]]).mean(-1) # mean(-1) is to deal with situations where one class name is divided into multiple tokens.
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(bz_boxes),  box_cxcywh_to_xyxy(bz_gtboxs))

        cost = ( cost_class + 3.0 * cost_giou + 100.0 * (~is_in_boxes_and_center) )  #[num_query,num_gt]
        # import pdb;pdb.set_trace()
        cost[~fg_mask] = cost[~fg_mask] + 10000.0
        return cost, pair_wise_ious, bz_gtboxs

    def get_in_boxes_info(self, boxes, target_gts, expanded_strides):
        # size (h,w) 
        # size = size[[1,0]].repeat(2) # (w,h,w,h)

        # import pdb;pdb.set_trace()
        # ori_gt_boxes = target_gts*size
        xy_target_gts = box_cxcywh_to_xyxy(target_gts) #x1y1x2y2
        
        anchor_center_x = boxes[:,0].unsqueeze(1)
        anchor_center_y = boxes[:,1].unsqueeze(1)

        # 判断每个anchor 的中心是否在某个gt box 内部
        b_l = anchor_center_x > xy_target_gts[:,0].unsqueeze(0)  # x1 满足要求
        b_r = anchor_center_x < xy_target_gts[:,2].unsqueeze(0)  # x2 满足要求
        b_t = anchor_center_y > xy_target_gts[:,1].unsqueeze(0)
        b_b = anchor_center_y < xy_target_gts[:,3].unsqueeze(0)
        # (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4 [300,num_gt] , 等于4表示四个条件都满足，所以只要每个query 对应的num_gt 个里面有一个4则说明该query 有效
        is_in_boxes = ( (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4)
        is_in_boxes_all = is_in_boxes.sum(1)>0  # [num_query]

        #每个gt的cx与cy向外扩展2.5*expanded_strides距离得到left_b,right_b,top_b,bottom_b，
        # 与anchor进行比较，计算anchor中心点是否包含在left_b,right_b,top_b,bottom_b中，得到 is_in_centers_all (shape:[num_anchors])
        # in fixed center
        center_radius = 2.5
        b_l = anchor_center_x > (target_gts[:,0]-(1*center_radius/expanded_strides)).unsqueeze(0)  # x1 满足要求
        b_r = anchor_center_x < (target_gts[:,0]+(1*center_radius/expanded_strides)).unsqueeze(0)  # x2 满足要求
        b_t = anchor_center_y > (target_gts[:,1]-(1*center_radius/expanded_strides)).unsqueeze(0)
        b_b = anchor_center_y < (target_gts[:,1]+(1*center_radius/expanded_strides)).unsqueeze(0)
        is_in_centers = ( (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4)
        is_in_centers_all = is_in_centers.sum(1)>0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all      # 上述两个条件只要满足一个，就成为候选区域。注意！！！这里是“|”求或

        # ！！！shape:[num_gt, num_in_boxes_anchor]，注意：这里是每一个gt与每一个候选区域的关系
        # 这里一个anchor可能与多个gt存在候选关系
        # is_in_boxes_anchor 里两者都满足的anchor
        # import pdb;pdb.set_trace()
        # is_in_boxes_and_center = (is_in_boxes[is_in_boxes_anchor, :] & is_in_centers[is_in_boxes_anchor,:] )       

        is_in_boxes_and_center = (is_in_boxes & is_in_centers)   

        return is_in_boxes_anchor,is_in_boxes_and_center
    
    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt):
        matching_matrix = torch.zeros_like(cost) # [300,num_gt] matching_matrix表示映关系，必须保证只能1gt对多query，且每个gt 有query
        ious_in_boxes_matrix = pair_wise_ious
        n_query = len(ious_in_boxes_matrix)
        n_candidate_k = min(n_query, 10)
        
        # 取预测值与gt拥有最大iou前10名的iou总和作为dynamic_k
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)
        # min=1,即把dynamic_ks限制最小为1，保证一个gt至少有一个正样本
        # 刚开始训练时候，由于预测基本不准，导致dynamic_k基本上都是1
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
        # 取cost排名最小的前dynamic_k个anchor作为postive
            _, pos_idx = torch.topk(cost[:,gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:,gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1)
        
        if (anchor_matching_gt > 1).sum() > 0: # 如果有一个query 匹配上了多个gt
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1) #  对这些query找cost 最小的gt
            matching_matrix[anchor_matching_gt > 1] *= 0 # 清零这些query的映射关系
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1 # 只保留cost 最小的gt 

        # import pdb;pdb.set_trace()
        ## 注意！此时删除了一些映射关系可能导致有些gt 没匹配上任何query, 此时通过while 循环重复上述过程直至所有gt 都匹配上
        while (matching_matrix.sum(0)==0).any(): # 只要有gt 没匹配上
            num_zero_gt = (matching_matrix.sum(0)==0).sum()
            matched_query_id = matching_matrix.sum(1)>0
            cost[matched_query_id] += 100000.0 
            unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:,gt_idx])
                matching_matrix[:,gt_idx][pos_idx] = 1.0
            # 继续判断是否有 多gt 对1 query的情况
            if (matching_matrix.sum(1) > 1).sum() > 0: # 如果有一个query 匹配上了多个gt
                _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1) #  对这些query找cost 最小的gt
                matching_matrix[anchor_matching_gt > 1] *= 0 # 清零这些query的映射关系
                matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1 # 只保留cost 最小的gt 

        assert not (matching_matrix.sum(0)==0).any() # 所有gt 都匹配上query
        # 此时 matching_matrix 是一个[300,num_gt] 的0-1矩阵
        # 
        selected_query = matching_matrix.sum(1)>0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        assert selected_query.sum() == len(gt_indices)

        # import pdb;pdb.set_trace()# 此处需要额外返回，跟每个gt 最匹配的query idx
        # matched_cost = ((matching_matrix==0)+cost) # 找出所有没匹配上的cost，加上极大值，找到每个gt最小的cost
        # matched_query_id  = torch.min(matched_cost,dim=0)[1]
        cost[matching_matrix==0] = cost[matching_matrix==0] + float('inf')
        matched_query_id = torch.min(cost,dim=0)[1]

        # in this version, the selected_query is [300,] with true/false
        # we convert it to value indices
        matched_anchor_inds = torch.arange(len(matching_matrix)).to(gt_indices)
        selected_query = matched_anchor_inds[selected_query]     # [num_pos,]

        return (selected_query,gt_indices) , matched_query_id

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid() # (bs * num_queries, 256). 256 is max_seq_len
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            # tgt_ids = torch.cat([v["labels"] for v in targets]) # (N_obj, )
            tgt_ids = torch.cat([v["positive_map"] for v in targets]) # (N_obj, 256) or (N_obj, 1)
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = torch.zeros((out_prob.size(0), tgt_ids.size(0)), device=out_prob.device)
            for idx in range(tgt_ids.size(0)):
                cost_class[:, idx] = (pos_cost_class[:, tgt_ids[idx]] - neg_cost_class[:, tgt_ids[idx]]).mean(-1) # mean(-1) is to deal with situations where one class name is divided into multiple tokens.

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou # (bs * num_queries, N_obj)
            C = C.view(bs, num_queries, -1).cpu() # (bs, num_query, N_obj)

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] # List of length batchsize. Each element is a tuple of two arrays. 
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices] # List of length batchsize. Each element is a tuple of two tensors. 


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)
