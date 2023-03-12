import pickle
import numpy as np
import copy
import torch
from collections import OrderedDict


if __name__ == "__main__":
    src_path = "image_joint_vit_huge_dino_32g/model_final.pth"
    des_path = "image_joint_vit_huge_dino_32g/model_final_4c.pth"
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
        if "patch_embed.proj.weight" in k:
            new_value = torch.zeros((1280, 4, 16, 16), dtype=torch.float32)
            new_value[:, :-1, :, :] = v
            data_new["model"][k_new] = new_value
        else:
            data_new["model"][k_new] = v
    assert des_path.endswith(".pth")
    torch.save(data_new, des_path)
