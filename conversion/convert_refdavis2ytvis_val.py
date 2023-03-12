"""
There are 2 steps for converting ref-davis to ytvis. (we only convert the val split for evaluation without finetune)
1. convert_refdavis2refytvos.py.
2. convert_refdavis2ytvis_val.py.

There are 4 annotations for each obj, we split it into 4 json files.
Each video is a sample, there may be multiple expressions.
"""
import json
import argparse
import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser("json converter")
    parser.add_argument("--data_dir", default="datasets/ref-davis", type=str, help="directory of ref-davis")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    split = "valid"
    img_folder = os.path.join(data_dir, split)
    # read the video list
    with open(os.path.join(data_dir, "meta_expressions", split, "meta_expressions.json"), 'r') as f:
        data = json.load(f)['videos']
        valid_test_videos = set(data.keys())
    valid_videos = valid_test_videos
    video_list = sorted([video for video in valid_videos])
    assert len(video_list) == 30, 'error: incorrect number of validation videos'

    # there are 2 annotators, and each of them gives the first and full video annotations.
    for anno_id in range(4):
        new_data = {"videos": [], "categories": [{"supercategory": "object","id": 1,"name": "object"}]}
        video_idx = 0
        # 1. For each video
        for video in tqdm(video_list):
            expressions = data[video]["expressions"]   
            expression_list = list(expressions.keys()) # "0", "1", ...
            num_expressions = len(expression_list)
            video_len = len(data[video]["frames"])
            frames = [os.path.join(video, x+".jpg") for x in data[video]["frames"]]
            H, W = cv2.imread(os.path.join(img_folder, "JPEGImages", frames[0])).shape[:-1]

            video_idx += 1
            meta = {"height": H, "width": W, "length": video_len, "file_names": frames, "id": video_idx}
            meta["video"] = video
            # read all the anno meta
            num_obj = num_expressions // 4
            tmp_expressions = []
            for i in range(num_obj):
                tmp_expressions.append(expressions[expression_list[i*4+anno_id]]["exp"])
            # [["exp1", "exp2", ...]], for being judged as refdavis in ytvis_dataset_mapper
            # for refcoco and refytvos, only has one string in the "expressions": ["exp1"]
            meta["expressions"] = [tmp_expressions]  
            new_data["videos"].append(meta)

        output_json = os.path.join(data_dir, f"{split}_{anno_id}.json")
        json.dump(new_data, open(output_json, 'w'))
