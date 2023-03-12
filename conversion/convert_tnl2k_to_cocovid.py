import json
import os
import cv2
import numpy as np
import torch
import csv


if __name__ == "__main__":
    data_root = "datasets/TNL-2K"
    des_dataset = {'videos':[], 'categories':[{"supercategory": "object", "id": 1, "name": "object"}], 'annotations':[]}
    data_dir = data_root
    vids = sorted(os.listdir(data_dir))
    assert len(vids) == 700
    vid_id = 0
    for vid_idx, vid in enumerate(vids):
        vid_dict = {}
        vid_id += 1
        vid_dir = os.path.join(data_dir, vid)
        vid_dict["file_names"] = []
        vid_dict["id"] = vid_id
        img_dir = os.path.join(vid_dir, "imgs")
        for x in sorted(os.listdir(img_dir)):
            if x.endswith(".jpg") or x.endswith(".png"):
                vid_dict["file_names"].append(os.path.join(vid, "imgs", x))
        vid_dict["height"], vid_dict["width"] = cv2.imread(os.path.join(data_dir, vid_dict["file_names"][0])).shape[:-1]
        vid_dict["length"] = len(vid_dict["file_names"])
        # load gts
        anno_dict = {}
        anno_dict["iscrowd"], anno_dict["category_id"], anno_dict["id"] = 0, 1, vid_id
        anno_dict["video_id"] = vid_id # one trajectory per sequence
        anno_dict["bboxes"] = []
        gts = np.loadtxt(os.path.join(vid_dir, "groundtruth.txt"), delimiter=",") # (x1, y1, w, h)
        assert vid_dict["length"] == len(gts)
        areas = list(gts[:, 2] * gts[:, 3])
        for i in range(len(gts)):
            anno_dict["bboxes"].append(list(gts[i]))
            anno_dict["areas"] = areas
        # merge
        des_dataset["videos"].append(vid_dict)
        des_dataset["annotations"].append(anno_dict)
        print("%d/%d %s done"%(vid_idx+1, len(vids), vid))
    # save
    with open(os.path.join(data_root, "test.json"), "w") as f:
        json.dump(des_dataset, f)