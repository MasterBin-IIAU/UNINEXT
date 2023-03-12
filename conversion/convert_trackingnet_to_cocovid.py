import json
import os
import cv2
import numpy as np
import torch
import csv


if __name__ == "__main__":
    data_root = "datasets/TrackingNet"
    splits = ["TRAIN_0", "TRAIN_1", "TRAIN_2", "TRAIN_3", "TEST"]
    for split in splits:
        print("converting %s split..." %split)
        des_dataset = {'videos':[], 'categories':[{"supercategory": "object", "id": 1, "name": "object"}], 'annotations':[]}
        data_dir = os.path.join(data_root, split)
        anno_dir = os.path.join(data_dir, "anno")
        frame_dir = os.path.join(data_dir, "frames")
        anno_files = sorted(os.listdir(anno_dir))
        vid_names = sorted(os.listdir(frame_dir))
        assert len(anno_files) == len(vid_names)
        vid_id = 0
        for vid in vid_names:
            vid_dict = {}
            vid_id += 1
            vid_dir = os.path.join(frame_dir, vid)
            vid_dict["file_names"] = []
            vid_dict["id"] = vid_id
            num_frames = len(os.listdir(vid_dir))
            for idx in range(num_frames):
                vid_dict["file_names"].append(os.path.join(split, "frames", vid, "%d.jpg"%idx))
            vid_dict["height"], vid_dict["width"] = cv2.imread(os.path.join(data_root, vid_dict["file_names"][0])).shape[:-1]
            vid_dict["length"] = len(vid_dict["file_names"])
            # load gts
            anno_dict = {}
            anno_dict["iscrowd"], anno_dict["category_id"], anno_dict["id"] = 0, 1, vid_id
            anno_dict["video_id"] = vid_id # one trajectory per sequence
            anno_dict["bboxes"] = []
            gts = np.loadtxt(os.path.join(anno_dir, "%s.txt"%vid), delimiter=",") # (x1, y1, w, h)
            if gts.ndim == 1:
                # if only the anno in the 1st frame is avaliable, we repeat it by the sequence length
                gts = gts[None]
                gts = np.tile(gts, (vid_dict["length"], 1))
            areas = list(gts[:, 2] * gts[:, 3])
            for i in range(len(gts)):
                anno_dict["bboxes"].append(list(gts[i]))
                anno_dict["areas"] = areas
            # merge
            des_dataset["videos"].append(vid_dict)
            des_dataset["annotations"].append(anno_dict)
        # save
        with open(os.path.join(data_root, "%s.json"%split), "w") as f:
            json.dump(des_dataset, f)