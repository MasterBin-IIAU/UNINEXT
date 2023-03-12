import os
import numpy as np
import cv2
from PIL import Image

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

if __name__ == "__main__":
    data_dir = "datasets/DAVIS"
    mask_dir = "outputs/video_joint_vit_huge/inference/DAVIS"
    output_dir = "outputs_DAVIS"
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
        num_frame = len(image_names)
        image_paths = [os.path.join(image_dir, x) for x in image_names]
        imgs = [cv2.imread(x) for x in image_paths]
        # load masks
        anno_dir = os.path.join(mask_dir, vid_name)
        anno_paths = [os.path.join(anno_dir, x) for x in sorted(os.listdir(anno_dir))]
        num_anno = len(anno_paths)
        assert num_frame == num_anno
        # bboxs
        bboxs = {}
        for f_id in range(num_frame):
            anno_path = anno_paths[f_id]
            # add color mask
            mask_c = cv2.imread(anno_path)
            img = imgs[f_id].astype(np.float32) + 1.0 * mask_c.astype(np.float32)
            # rescale to 255
            imgs[f_id] = img/img.max()*255
            # add box
            mask = Image.open(anno_path).convert('P')
            mask = np.array(mask)
            values = np.unique(mask)
            for v in values:
                if v != 0:
                    cur_mask = (mask == v)
                    bbox = bounding_box(cur_mask)
                    if v not in bboxs:
                        bboxs[v] = []
                    bboxs[v].append(bbox)
        # save
        for i in range(num_frame):
            save_path = os.path.join(output_dir_vid, "%05d.jpg"%i)
            img = imgs[i]
            img = img/img.max() * 255 # rescale to [0, 255]
            # add bbox
            for f_id in bboxs.keys():
                x1, y1, x2, y2 = bboxs[f_id][i]
                color = palette[3*f_id:3*(f_id+1)]
                color.reverse()
                color = tuple(color)
                cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)
            cv2.imwrite(save_path, img)