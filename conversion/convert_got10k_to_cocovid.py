import json
import os
import cv2
import numpy as np
import torch
import csv


if __name__ == "__main__":
    data_root = "datasets/GOT10K"
    splits = ["train", "val", "test"]
    for split in splits:
        print("converting %s split..." %split)
        des_dataset = {'videos':[], 'categories':[{"supercategory": "object", "id": 1, "name": "object"}], 'annotations':[]}
        data_dir = os.path.join(data_root, split)
        with open(os.path.join(data_dir, "list.txt"), "r") as f:
            vids = [x.strip("\n") for x in f.readlines()]
        vid_id = 0
        for vid in vids:
            vid_dict = {}
            vid_id += 1
            vid_dir = os.path.join(data_dir, vid)
            vid_dict["file_names"] = []
            vid_dict["id"] = vid_id
            for x in sorted(os.listdir(vid_dir)):
                if x.endswith(".jpg"):
                    vid_dict["file_names"].append(os.path.join(vid, x))
            vid_dict["height"], vid_dict["width"] = cv2.imread(os.path.join(data_dir, vid_dict["file_names"][0])).shape[:-1]
            vid_dict["length"] = len(vid_dict["file_names"])
            # load gts
            anno_dict = {}
            anno_dict["iscrowd"], anno_dict["category_id"], anno_dict["id"] = 0, 1, vid_id
            anno_dict["video_id"] = vid_id # one trajectory per sequence
            anno_dict["bboxes"] = []
            gts = np.loadtxt(os.path.join(vid_dir, "groundtruth.txt"), delimiter=",") # (x1, y1, w, h)
            if gts.ndim == 1:
                # if only the anno in the 1st frame is avaliable, we repeat it by the sequence length
                gts = gts[None]
                gts = np.tile(gts, (vid_dict["length"], 1))
            areas = list(gts[:, 2] * gts[:, 3])
            for i in range(len(gts)):
                anno_dict["bboxes"].append(list(gts[i]))
                anno_dict["areas"] = areas
            if split != "test":
                # Read full occlusion and out_of_view
                occlusion_file = os.path.join(vid_dir, "absence.label")
                cover_file = os.path.join(vid_dir, "cover.label")
                with open(occlusion_file, 'r', newline='') as f:
                    occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
                with open(cover_file, 'r', newline='') as f:
                    cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
                target_visible = ~occlusion & (cover>0).byte()
                anno_dict["valid"] = target_visible.bool().tolist()
            # merge
            des_dataset["videos"].append(vid_dict)
            des_dataset["annotations"].append(anno_dict)
        # save
        with open(os.path.join(data_root, "%s.json"%split), "w") as f:
            json.dump(des_dataset, f)