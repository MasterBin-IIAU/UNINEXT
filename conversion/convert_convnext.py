import torch
import argparse

# For D2, we need to convert nn.Parameters to nn.Embedding
# this script is used to convert the original ckpt to the needed form

def parse_args():
    parser = argparse.ArgumentParser("ConvNeXt converter")

    parser.add_argument("--source_model", default="weights/convnext_tiny_1k_224_ema.pth", type=str, help="Path or url to the DETR model to convert")
    parser.add_argument("--output_model", default="weights/convnext_tiny_1k_224_ema_new.pth", type=str, help="Path where to save the converted model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt = torch.load(args.source_model, map_location="cpu")
    state_dict = ckpt["model"]
    change_list = ["downsample_layers.0.1.bias", "downsample_layers.0.1.weight",
    "downsample_layers.1.0.bias", "downsample_layers.1.0.weight",
    "downsample_layers.2.0.bias", "downsample_layers.2.0.weight",
    "downsample_layers.3.0.bias", "downsample_layers.3.0.weight"]
    # new ckpt
    ckpt_new = {"model":{}}
    for k in state_dict.keys():
        if ("gamma" in k) or ("norm" in k) or k in change_list:
            ckpt_new["model"][k+".weight"] = state_dict[k].unsqueeze(0)
        else:
            ckpt_new["model"][k] = state_dict[k]
    torch.save(ckpt_new, args.output_model)