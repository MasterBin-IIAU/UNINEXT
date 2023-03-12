import torch
import torch.nn.functional as NNF


def merge_backbone_output(inp_list):
    """ merge outputs from backbone
        feat: HWxBxC, pos: HWxBxC, mask: BxHW
    """
    nf = len(inp_list)
    seq_dict = {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
                "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
                "pos": torch.cat([x["pos"] for x in inp_list], dim=0),
                "h": inp_list[0]["h"], "w": inp_list[0]["w"], "nf": nf}
    return seq_dict


def convert_to_onehot(x, dim):
    """convert to one-hot"""
    out_lbs = torch.zeros_like(x, device=x.device)
    pos_index = torch.argmax(x, dim=dim, keepdim=True)
    out_lbs.scatter_(dim, pos_index, 1)
    return out_lbs

def adjust_labels_sz(inp_lbs, dh, dw):
    """ inp_lbs: PyTorch tensor (F, K, H, W) --> (F, K, H*s, W*s)
    dh: destination h, dw: destination w"""
    assert len(inp_lbs.size()) == 4
    """interpolation"""
    x = NNF.interpolate(inp_lbs, size=(dh, dw), mode="bilinear", align_corners=False)
    """convert to one-hot"""
    out_lbs = convert_to_onehot(x, dim=1)
    return out_lbs
