import json
import argparse
import os
from PIL import Image
import numpy as np
import cv2
import pycocotools.mask as maskUtils
from detectron2.structures import PolygonMasks
import copy

def parse_args():
    parser = argparse.ArgumentParser("json converter")
    parser.add_argument("--data_dir", default="datasets/ref-youtube-vos", type=str, help="directory of ref-youtube-vos")
    parser.add_argument("--mask_format", default="rle", choices=["polygon", "rle"], type=str)
    return parser.parse_args()

def compute_area(segmentation):
    if isinstance(segmentation, list):
        polygons = PolygonMasks([segmentation])
        area = polygons.area()[0].item()
    elif isinstance(segmentation, dict):  # RLE
        area = maskUtils.area(segmentation).item()
    else:
        raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
    return area

def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2-x1), int(y2-y1)] # (x1, y1, w, h) 

def mask2polygon(input_mask):
    contours, hierarchy = cv2.findContours(input_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        result.append(segmentation)
    return result

def mask2rle(input_mask):
    rle = maskUtils.encode(np.array(input_mask, order="F", dtype="uint8"))
    if not isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle

if __name__ == "__main__":
    min_vid_len = 2 # there must be at least 2 frames in a video. Or it will be invalid.
    args = parse_args()
    data_dir = args.data_dir
    splits = ["train"]
    for split in splits:
        assert split == "train"
        new_data = {"videos": [], "annotations": [], "categories": [{"supercategory": "object","id": 1,"name": "object"}]}
        inst_idx = 0
        # read object information
        img_folder = os.path.join(data_dir, split)
        with open(os.path.join(img_folder, 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']
        # read expression data
        ann_file = os.path.join(data_dir, "meta_expressions/%s/meta_expressions.json"%split)
        with open(ann_file, 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        assert len(subset_metas_by_video) == len(subset_expressions_by_video)
        videos = list(subset_expressions_by_video.keys())
        num_vids = len(videos)
        images_dir = os.path.join(data_dir, split, "JPEGImages")
        masks_dir = os.path.join(data_dir, split, "Annotations")
        for vid_idx, vid in enumerate(videos):
            # parse video information
            vid_dict = {"height": None, "width": None, "length": None, "file_names": None, "id": None}
            vid_img_dir = os.path.join(images_dir, vid)
            vid_mask_dir = os.path.join(masks_dir, vid)
            frames = sorted(os.listdir(vid_img_dir))
            masks = sorted(os.listdir(vid_mask_dir))
            assert len(frames) == len(masks)
            init_frame_path = os.path.join(images_dir, vid, frames[0])
            H, W, _ = cv2.imread(init_frame_path).shape
            vid_dict["height"], vid_dict["width"] = H, W
            # parse expressions in a video
            data_dict = {}
            for _, exp_data in subset_expressions_by_video[vid]["expressions"].items():
                exp, obj_id = exp_data["exp"], exp_data["obj_id"]
                if obj_id not in data_dict:
                    data_dict[obj_id] = {"exp": [], "frames": None}
                data_dict[obj_id]["exp"].append(exp)
            metas_vid = subset_metas_by_video[vid]["objects"]
            # save expressions to vid_dict
            vid_obj_dict = {}
            vid_info_dict = {}
            for obj_id in metas_vid.keys():
                valid_len = len(metas_vid[obj_id]["frames"])
                vid_obj_dict[obj_id] = {"video_id": None, "id": None, "iscrowd": 0, "category_id": 1, 
                    "bboxes": [], "segmentations": [], "areas": []}
                vid_dict_cur = copy.deepcopy(vid_dict)
                vid_dict_cur["file_names"] = []
                vid_dict_cur["expressions"] = data_dict[obj_id]["exp"]
                vid_dict_cur["id"] = None
                vid_info_dict[obj_id] = vid_dict_cur
            # parse mask information in the current video
            for frame_idx in range(len(frames)):
                mask_path = os.path.join(vid_mask_dir, masks[frame_idx])
                mask = Image.open(mask_path).convert('P')
                mask = np.array(mask)
                H, W = mask.shape
                # loop over obj_id in a video
                for obj_id in metas_vid.keys():
                    # get annos
                    mask_cur = (mask==int(obj_id)).astype(np.uint8) # 0,1 binary
                    # some frame didn't contain the instance
                    if (mask_cur > 0).any():
                        box = bounding_box(mask_cur)
                        area = int(box[-2] * box[-1])
                        vid_obj_dict[obj_id]["bboxes"].append(box)
                        if args.mask_format == "polygon":
                            vid_obj_dict[obj_id]["segmentations"].append(mask2polygon(mask_cur))
                        elif args.mask_format == "rle":
                            vid_obj_dict[obj_id]["segmentations"].append(mask2rle(mask_cur))
                        else:
                            raise ValueError("Unsupported mask format")
                        vid_obj_dict[obj_id]["areas"].append(area)
                        vid_info_dict[obj_id]["file_names"].append(os.path.join(vid, masks[frame_idx].replace(".png", ".jpg")))
                        # save to annotations
            for obj_id in metas_vid.keys():
                vid_info_dict[obj_id]["length"] = len(vid_obj_dict[obj_id]["bboxes"])
                assert len(vid_info_dict[obj_id]["file_names"]) == vid_info_dict[obj_id]["length"]
                if len(vid_obj_dict[obj_id]["bboxes"]) >= min_vid_len:
                    # accumulate
                    inst_idx += 1
                    vid_obj_dict[obj_id]["video_id"] = inst_idx
                    vid_obj_dict[obj_id]["id"] = inst_idx
                    vid_info_dict[obj_id]["id"] = inst_idx
                    # save
                    new_data["annotations"].append(vid_obj_dict[obj_id])
                    new_data["videos"].append(vid_info_dict[obj_id])
            print("%05d/%05d done."%(vid_idx+1, num_vids))
        output_json = os.path.join(data_dir, "%s.json"%split)
        json.dump(new_data, open(output_json, 'w'))