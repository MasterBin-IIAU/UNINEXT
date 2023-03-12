import torch



if __name__ == "__main__":
    path1 = "image_joint_r50_dino_load_obj365/model_final_4c.pth"
    path2 = "DEBUG/model_0000019.pth"
    ckpt1 = torch.load(path1, map_location="cpu")["model"]
    ckpt2 = torch.load(path2, map_location="cpu")["model"]
    for k in ckpt1.keys():
        v1 = ckpt1[k]
        v2 = ckpt2[k]
        if torch.sum(torch.abs(v1-v2)) != 0:
            print(k)
