# ------------------------------------------------------------------------
# IDOL: In Defense of Online Models for Video Instance Segmentation
# Copyright (c) 2022 ByteDance. All Rights Reserved.
# ------------------------------------------------------------------------


import torch
import torch.nn as nn
import torchvision
from ..util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
import random
import torchvision.ops as ops


def select_pos_neg(ref_box, all_indices, targets, det_targets, embed_head, hs_key, hs_ref, ref_cls, detach_reid=False,
use_deformable_reid_head=False, src_info_key=None, src_info_ref=None):
    """
    ref_box: (bs, num_query, 4)
    all_indices: the most matched query id to the GTs of the key frames. List of len is bs
    targets: GTs of the reference frames. List of len bs.
    det_targets: GTs of the key frames. List of len bs.
    ref_cls: (bs, num_query, 1)
    """
    if use_deformable_reid_head:
        assert (src_info_key is not None) and (src_info_ref is not None)
        assert detach_reid
        ref_embeds = embed_head[1](embed_head[0](
            hs_ref.detach(), src_info_ref["reference_points"], src_info_ref["src"], 
            src_info_ref["src_spatial_shapes"], src_info_ref["src_level_start_index"], 
            src_info_ref["src_valid_ratios"], src_info_ref["src_padding_mask"]))
        key_embedds = embed_head[1](embed_head[0](
            hs_key.detach(), src_info_key["reference_points"], src_info_key["src"], 
            src_info_key["src_spatial_shapes"], src_info_key["src_level_start_index"], 
            src_info_key["src_valid_ratios"], src_info_key["src_padding_mask"]))
    else:
        if detach_reid:
            ref_embeds = embed_head(hs_ref.detach())
            key_embedds = embed_head(hs_key.detach())
        else:
            ref_embeds = embed_head(hs_ref)
            key_embedds = embed_head(hs_key)
    one = torch.tensor(1).to(ref_embeds)
    zero = torch.tensor(0).to(ref_embeds)
    contrast_items = []
    assert len(targets) == len(all_indices)
    # l2_items = []
    # loop over images in a batchsize
    for bz_i,(v,detv, indices) in enumerate(zip(targets,det_targets,all_indices)):
        num_insts = len(v["labels"]) 
        # tgt_valid = v["valid"].reshape(num_insts)
        tgt_bbox = v["boxes"].reshape(num_insts,4) 
        tgt_labels = v["positive_map"]
        # tgt_valid = tgt_valid[:,1]    
        ref_box_bz = ref_box[bz_i] # (num_query, 4)
        ref_cls_bz = ref_cls[bz_i] # (num_query, 1)
        tgt_valid = v["valid"]
               
        contrastive_pos = get_pos_idx(ref_box_bz,ref_cls_bz,tgt_bbox,tgt_labels, tgt_valid)

        # loop over gts in one image
        for inst_i, (valid,matched_query_id) in enumerate(zip(tgt_valid,indices)):
            
            if not valid:  
                continue
            gt_box = tgt_bbox[inst_i].unsqueeze(0)
            key_embed_i = key_embedds[bz_i,matched_query_id].unsqueeze(0)

            pos_embed = ref_embeds[bz_i][contrastive_pos[0][inst_i]]
            neg_embed = ref_embeds[bz_i][~contrastive_pos[1][inst_i]] # remove querys which may have overlap with the given gt. The rest querys are negative embed
            contrastive_embed = torch.cat([pos_embed,neg_embed],dim=0)
            contrastive_label = torch.cat([one.repeat(len(pos_embed)),zero.repeat(len(neg_embed))],dim=0) 

            contrast = torch.einsum('nc,kc->nk',[contrastive_embed,key_embed_i]) # (N_pos+N_neg, N_gt)

            # auxiliary loss for cosine similarity
            if len(pos_embed) ==0 :
                num_sample_neg = 10
            elif len(pos_embed)*10 >= len(neg_embed):
                num_sample_neg = len(neg_embed)
            else:
                num_sample_neg = len(pos_embed)*10 # control num_neg (max is 10 * num_pos)

            sample_ids = random.sample(list(range(0, len(neg_embed))), num_sample_neg)

            aux_contrastive_embed = torch.cat([pos_embed,neg_embed[sample_ids]],dim=0)
            aux_contrastive_label = torch.cat([one.repeat(len(pos_embed)),zero.repeat(num_sample_neg)],dim=0) 
            aux_contrastive_embed=nn.functional.normalize(aux_contrastive_embed.float(),dim=1)
            key_embed_i=nn.functional.normalize(key_embed_i.float(),dim=1)    
            cosine = torch.einsum('nc,kc->nk',[aux_contrastive_embed,key_embed_i])


            contrast_items.append({'contrast':contrast,'label':contrastive_label, 'aux_consin':cosine,'aux_label':aux_contrastive_label})

    return contrast_items



# The following functions are almost the same as those in matcher.py 
def get_pos_idx(bz_boxes,bz_out_prob,bz_gtboxs,bz_tgt_ids,valid):
    with torch.no_grad():  
        if False in valid: 
            bz_gtboxs = bz_gtboxs[valid]
            bz_tgt_ids = bz_tgt_ids[valid]

        fg_mask, is_in_boxes_and_center  = \
            get_in_boxes_info(bz_boxes,bz_gtboxs,expanded_strides=32)
        pair_wise_ious = ops.box_iou(box_cxcywh_to_xyxy(bz_boxes), box_cxcywh_to_xyxy(bz_gtboxs))
        # pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        
        # Compute the classification cost. (as same as that in matcher.py)
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log())
        cost_class = torch.zeros((bz_out_prob.size(0), bz_tgt_ids.size(0)), device=pos_cost_class.device)
        for idx in range(bz_tgt_ids.size(0)):
            cost_class[:, idx] = (pos_cost_class[:, bz_tgt_ids[idx]] - neg_cost_class[:, bz_tgt_ids[idx]]).mean(-1) # mean(-1) is to deal with situations where one class name is divided into multiple tokens.
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(bz_boxes),  box_cxcywh_to_xyxy(bz_gtboxs))

        cost = ( cost_class + 3.0 * cost_giou + 100.0 * (~is_in_boxes_and_center) )

        cost[~fg_mask] = cost[~fg_mask] + 10000.0

        
       
        if False in valid:
            indices_batchi_pos = []
            indices_batchi_neg = []
            if valid.sum()>0:
                indices_batchi_pos_s = dynamic_k_matching(cost, pair_wise_ious, int(valid.sum()),10)
                indices_batchi_neg_s = dynamic_k_matching(cost, pair_wise_ious, int(valid.sum()),100)
            valid_idx = 0
            valid_list = valid.tolist()
            for istrue in valid_list:
                if istrue:
                    indices_batchi_pos.append(indices_batchi_pos_s[valid_idx])
                    indices_batchi_neg.append(indices_batchi_neg_s[valid_idx])
                    valid_idx = valid_idx+1
                else:
                    indices_batchi_pos.append(None)
                    indices_batchi_neg.append(None)
            
        else:
            if valid.sum()>0:
                indices_batchi_pos = dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs.shape[0],10)
                indices_batchi_neg = dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs.shape[0],100)
            else:
                indices_batchi_pos = [None]
                indices_batchi_neg = [None]
                # print('empty object in pos_neg select')

    
    return (indices_batchi_pos, indices_batchi_neg)

def get_in_boxes_info(boxes, target_gts, expanded_strides):
    # size (h,w) 
    # size = size[[1,0]].repeat(2) # (w,h,w,h)

    # ori_gt_boxes = target_gts*size
    xy_target_gts = box_cxcywh_to_xyxy(target_gts) #x1y1x2y2
    
    anchor_center_x = boxes[:,0].unsqueeze(1)
    anchor_center_y = boxes[:,1].unsqueeze(1)

    b_l = anchor_center_x > xy_target_gts[:,0].unsqueeze(0)  
    b_r = anchor_center_x < xy_target_gts[:,2].unsqueeze(0) 
    b_t = anchor_center_y > xy_target_gts[:,1].unsqueeze(0)
    b_b = anchor_center_y < xy_target_gts[:,3].unsqueeze(0)
    is_in_boxes = ( (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4)
    is_in_boxes_all = is_in_boxes.sum(1)>0  # [num_query]

    # in fixed center
    center_radius = 2.5
    b_l = anchor_center_x > (target_gts[:,0]-(1*center_radius/expanded_strides)).unsqueeze(0)  
    b_r = anchor_center_x < (target_gts[:,0]+(1*center_radius/expanded_strides)).unsqueeze(0)  
    b_t = anchor_center_y > (target_gts[:,1]-(1*center_radius/expanded_strides)).unsqueeze(0)
    b_b = anchor_center_y < (target_gts[:,1]+(1*center_radius/expanded_strides)).unsqueeze(0)
    is_in_centers = ( (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4)
    is_in_centers_all = is_in_centers.sum(1)>0

    is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all    

    is_in_boxes_and_center = (is_in_boxes & is_in_centers)   

    return is_in_boxes_anchor,is_in_boxes_and_center

def dynamic_k_matching(cost, pair_wise_ious, num_gt, n_candidate_k):
    matching_matrix = torch.zeros_like(cost) 
    ious_in_boxes_matrix = pair_wise_ious
    # n_candidate_k = 10
    
    topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(cost[:,gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
        matching_matrix[:,gt_idx][pos_idx] = 1.0

    del topk_ious, dynamic_ks, pos_idx

    anchor_matching_gt = matching_matrix.sum(1)
    
    if (anchor_matching_gt > 1).sum() > 0: 
        _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1) 
        matching_matrix[anchor_matching_gt > 1] *= 0
        matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1 

    while (matching_matrix.sum(0)==0).any(): 
        num_zero_gt = (matching_matrix.sum(0)==0).sum()
        matched_query_id = matching_matrix.sum(1)>0
        cost[matched_query_id] += 100000.0 
        unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
        for gt_idx in unmatch_id:
            pos_idx = torch.argmin(cost[:,gt_idx])
            matching_matrix[:,gt_idx][pos_idx] = 1.0
        if (matching_matrix.sum(1) > 1).sum() > 0: 
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1) 
            matching_matrix[anchor_matching_gt > 1] *= 0 
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1

    assert not (matching_matrix.sum(0)==0).any() 
 
    matched_pos = []
    for gt_idx in range(num_gt):
        matched_pos.append(matching_matrix[:,gt_idx]>0)        

    return matched_pos


