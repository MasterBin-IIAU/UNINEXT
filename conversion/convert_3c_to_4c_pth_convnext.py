import pickle
import numpy as np
import copy
import torch
from collections import OrderedDict


if __name__ == "__main__":
    src_path = "image_joint_convnext_large/model_final.pth"
    des_path = "image_joint_convnext_large/model_final_4c.pth"
    assert src_path.endswith(".pth")
    data = torch.load(src_path, map_location="cpu")
    data_type = "torch"
    data_new = copy.deepcopy(data)
    assert isinstance(data_new["model"], OrderedDict)
    for k, v in data["model"].items():
        if "backbone.0.backbone." in k:
            k_new = k.replace("backbone.0.backbone.", "ref_backbone.0.backbone.")
        else:
            continue
        if "downsample_layers.0.0.weight" in k:
            new_value = torch.zeros((192, 4, 4, 4), dtype=torch.float32)
            new_value[:, :-1, :, :] = v
            data_new["model"][k_new] = new_value
        else:
            data_new["model"][k_new] = v
    assert des_path.endswith(".pth")
    torch.save(data_new, des_path)
