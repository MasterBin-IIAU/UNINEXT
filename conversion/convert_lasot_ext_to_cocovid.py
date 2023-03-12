import json
import os
import cv2
import numpy as np
import torch
import csv


if __name__ == "__main__":
    data_root = "datasets/LaSOT_extension_subset"
    des_dataset = {'videos':[], 'categories':[{"supercategory": "object", "id": 1, "name": "object"}], 'annotations':[]}
    data_dir = data_root
    # with open(os.path.join(split_dir, "lasot_%s_split.txt"%split), "r") as f:
    #     vids = [x.strip("\n") for x in f.readlines()]
    macro_vids = sorted(os.listdir(data_dir))
    vids = []
    for name in macro_vids:
        vids.extend([name+"-%d"%i for i in range(1, 11)])
    vid_id = 0
    for vid in vids:
        vid_dict = {}
        vid_id += 1
        cls_name = vid.split("-")[0] 
        vid_dir = os.path.join(data_dir, cls_name, vid)
        vid_dict["file_names"] = []
        vid_dict["id"] = vid_id
        img_dir = os.path.join(vid_dir, "img")
        for x in sorted(os.listdir(img_dir)):
            if x.endswith(".jpg"):
                vid_dict["file_names"].append(os.path.join(cls_name, vid, "img", x))
        vid_dict["height"], vid_dict["width"] = cv2.imread(os.path.join(data_dir, vid_dict["file_names"][0])).shape[:-1]
        vid_dict["length"] = len(vid_dict["file_names"])
        # load gts
        anno_dict = {}
        anno_dict["iscrowd"], anno_dict["category_id"], anno_dict["id"] = 0, 1, vid_id
        anno_dict["video_id"] = vid_id # one trajectory per sequence
        anno_dict["bboxes"] = []
        gts = np.loadtxt(os.path.join(vid_dir, "groundtruth.txt"), delimiter=",") # (x1, y1, w, h)
        areas = list(gts[:, 2] * gts[:, 3])
        for i in range(len(gts)):
            anno_dict["bboxes"].append(list(gts[i]))
            anno_dict["areas"] = areas
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(vid_dir, "full_occlusion.txt")
        out_of_view_file = os.path.join(vid_dir, "out_of_view.txt")
        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        with open(out_of_view_file, 'r') as f:
            out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        target_visible = ~occlusion & ~out_of_view
        anno_dict["valid"] = target_visible.bool().tolist()
        # merge
        des_dataset["videos"].append(vid_dict)
        des_dataset["annotations"].append(anno_dict)
        print("%s done"%vid)
    # save
    with open(os.path.join(data_root, "test.json"), "w") as f:
        json.dump(des_dataset, f)