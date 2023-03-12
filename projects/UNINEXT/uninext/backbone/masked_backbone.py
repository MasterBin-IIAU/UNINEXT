import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict
from detectron2.modeling import build_backbone
from ..util.misc import NestedTensor


class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg, input_shape=None):
        super().__init__()
        self.backbone = build_backbone(cfg, input_shape=input_shape)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]

        self.num_channels = [backbone_shape[f].channels for f in backbone_shape.keys()]

    def forward(self, tensor_list):
        xs = self.backbone(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks