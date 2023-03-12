# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area
# from torchvision.ops import generalized_box_iou_loss

def box_cxcywh_to_xyxy(x):
    # print('box:\n', x)

    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def multi_box_iou(boxes1, boxes2):
    area1 = box_area(boxes1.flatten(0,1)).reshape(boxes1.shape[0], boxes1.shape[1])
    area2 = box_area(boxes2.flatten(0,1)).reshape(boxes2.shape[0], boxes2.shape[1])

    lt = torch.max(boxes1[:, :, None, :2], boxes2[:, None, :, :2])  # [nf,N,M,2]
    rb = torch.min(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])  # [nf,N,M,2]

    wh = (rb - lt).clamp(min=0)  # [nf,N,M,2]
    inter = wh[:, :, :, 0] * wh[:, :, :, 1]  # [nf,N,M]

    union = area1[:, :, None] + area2[:, None, :] - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area+1e-7)


def generalized_multi_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format
    boxes1.shape = [nf, N, 4]
    boxes2.shape = [nf, M, 4]
    Returns a [nf, N, M] pairwise matrix, where N = boxes1.shape[1]
    and M = boxes2.shape[1]
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    assert (boxes1[:, :, 2:] >= boxes1[:, :, :2]).all()
    assert (boxes2[:, :, 2:] >= boxes2[:, :, :2]).all()
    iou, union = multi_box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, :, None, :2], boxes2[:, None, :, :2])
    rb = torch.max(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])

    wh = (rb - lt).clamp(min=0)  # [nf,N,M,2]
    area = wh[:, :, :, 0] * wh[:, :, :, 1]
    

    return iou - (area - union) / (area + 1e-7)


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)



