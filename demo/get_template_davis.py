import os
import numpy as np
import cv2
from PIL import Image
import math
import torch.nn.functional as F
import torch

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
          [0.494, 0.000, 0.556], [0.494, 0.000, 0.000], [0.000, 0.745, 0.000],
          [0.700, 0.300, 0.600],[0.000, 0.447, 0.741], [0.850, 0.325, 0.098]]

def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)] # (x1, y1, x2, y2) 

def get_template(img, bbox):
    """img: (1, 3, H, W), mask: (1, 1, H, W), bbox: (4, )"""
    search_area_factor = 2
    template_sz = 256

    x, y, w, h = bbox

    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz
    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - img.shape[-1] + 1, 0)
    y1_pad = max(0, -y1)
    y2_pad = max(y2 - img.shape[-2] + 1, 0)

    # Crop target
    im_crop = img[:, :, y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = F.pad(im_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    # resize
    im_crop_padded = F.interpolate(im_crop_padded, (template_sz, template_sz), mode='bilinear', align_corners=False)
    
    return im_crop_padded

if __name__ == "__main__":
    data_dir = "datasets/DAVIS"
    mask_dir = "outputs/video_joint_vit_huge/inference/DAVIS"
    output_dir = "outputs_DAVIS_template"
    vids = ["camel", "breakdance", "gold-fish", "motocross-jump", "pigs", "soapbox"]
    palette_img = "datasets/DAVIS/Annotations/480p/bear/00000.png"
    palette = Image.open(palette_img).getpalette()
    for vid_name in vids:
        output_dir_vid = os.path.join(output_dir, vid_name)
        if not os.path.exists(output_dir_vid):
            os.makedirs(output_dir_vid)
        # load images
        image_dir = os.path.join(data_dir, "JPEGImages/480p", vid_name)
        image_names = sorted(os.listdir(image_dir))
        num_frame = 1
        image_paths = [os.path.join(image_dir, x) for x in image_names]
        imgs = [cv2.imread(x) for x in image_paths]
        # load masks
        anno_dir = os.path.join(mask_dir, vid_name)
        anno_paths = [os.path.join(anno_dir, x) for x in sorted(os.listdir(anno_dir))]
        num_anno = len(anno_paths)
        # bboxs
        for f_id in range(num_frame): # only use the first frame
            anno_path = anno_paths[f_id]
            # add box
            mask = Image.open(anno_path).convert('P')
            mask = np.array(mask)
            values = np.unique(mask)
            for v in values:
                if v != 0:
                    cur_mask = (mask == v)
                    x1, y1, x2, y2 = bounding_box(cur_mask)
                    color = palette[3*v:3*(v+1)]
                    color.reverse()
                    mask_c = cur_mask[:, :, None] * np.array(color)
                    img = imgs[f_id].astype(np.float32) + 1.0 * mask_c.astype(np.float32)
                    # rescale to 255
                    img = img/img.max()*255
                    # to tensor
                    img_tensor = torch.from_numpy(img).permute((2, 0, 1)).unsqueeze(0)
                    img_tensor = get_template(img_tensor, [x1, y1, x2-x1, y2-y1])
                    img_tensor = img_tensor.squeeze(0).permute((1,2,0))
                    img = np.array(img_tensor)
                    save_path = os.path.join(output_dir_vid, "%05d_%d.jpg"%(f_id, v))
                    cv2.imwrite(save_path, img)