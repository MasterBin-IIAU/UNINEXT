import torch

if __name__ == "__main__":
    ckpt_path = "video_joint_r50_dino/model_final.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")["model"]
    torch.save(ckpt, "model.pth")
    ckpt_txt_enc = {}
    ckpt_backbone = {}
    ckpt_ref_backbone = {}
    ckpt_enc = {}
    ckpt_dec = {}
    ckpt_other = {}
    for k in ckpt.keys():
        assert (k.startswith("detr") or k.startswith("text_encoder"))
        if k.startswith("text_encoder"):
            ckpt_txt_enc[k] = ckpt[k]
        elif k.startswith("detr.detr.backbone.0.backbone."):
            ckpt_backbone[k] = ckpt[k]
        elif k.startswith("detr.detr.ref_backbone.0.backbone."):
            ckpt_ref_backbone[k] = ckpt[k]
        elif k.startswith("detr.detr.transformer.encoder."):
            ckpt_enc[k] = ckpt[k]
        elif k.startswith("detr.detr.transformer.decoder."):
            ckpt_dec[k] = ckpt[k]
        else:
            # print(k)
            ckpt_other[k] = ckpt[k]
    torch.save(ckpt_txt_enc, "model_txt_enc.pth")
    torch.save(ckpt_backbone, "model_backbone.pth")
    torch.save(ckpt_ref_backbone, "model_ref_backbone.pth")
    torch.save(ckpt_enc, "model_enc.pth")
    torch.save(ckpt_dec, "model_dec.pth")
    torch.save(ckpt_other, "model_other.pth")