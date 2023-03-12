import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser("image to video converter")
    parser.add_argument("--src_json", default="datasets/bdd/labels/box_track_20/box_track_val_cocofmt.json", type=str, help="")
    parser.add_argument("--des_json", default="datasets/bdd/labels/box_track_20/box_track_val_cocofmt_uni.json", type=str, help="")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    src_dataset = json.load(open(args.src_json, 'r')) 
    des_dataset = {'videos':[], 'categories':[], 'annotations':[]}
    des_dataset["categories"] = src_dataset["categories"]
    
    # videos
    videos_dict = {}
    imgid2vidid = {} # map frame id to video id
    imgid2frameid = {} # map image id to frame id (0-based)
    for img_dict in src_dataset["images"]:
        video_id = img_dict["video_id"]
        if video_id not in videos_dict:
            videos_dict[video_id] = {"length": 0, "file_names": [], \
                "width":img_dict["width"], "height":img_dict["height"], "id": img_dict["video_id"]}
            frame_id = 0
        videos_dict[video_id]["length"] += 1
        assert frame_id == img_dict["frame_id"]
        frame_id += 1
        videos_dict[video_id]["file_names"].append(img_dict["file_name"])
        imgid2vidid[img_dict["id"]] = img_dict["video_id"]
        imgid2frameid[img_dict["id"]] = img_dict["frame_id"]
    for k in sorted(videos_dict.keys()):
        des_dataset["videos"].append(videos_dict[k])

    # annotations
    annotations_dict = {}
    """
    NOTE:
    if Object A does not appear in some frames, their annotations on these frames should be None!
    id should be global id in the whole dataset rather than the current video!
    """
    inst_id_dict = {}
    last_video_id = 0
    inst_id_base = 0
    for anno_dict in src_dataset["annotations"]:
        img_id = anno_dict["image_id"]
        video_id = imgid2vidid[img_id]
        if video_id not in annotations_dict:
            assert video_id == (last_video_id + 1) # assume the video_id increases one be one
            last_video_id = video_id
            annotations_dict[video_id] = {}
            inst_id_base += len(inst_id_dict) # add number of instances in the last video
            inst_id_dict = {} # reset inst_id_dict when dealing with a new video
        inst_id = inst_id_base + anno_dict["instance_id"]
        if inst_id not in inst_id_dict:
            inst_id_dict[inst_id] = None # add inst_id to dict
        if inst_id not in annotations_dict[video_id]:
            video_len = videos_dict[video_id]["length"]
            annotations_dict[video_id][inst_id] = {"iscrowd": anno_dict["iscrowd"], \
                "category_id": anno_dict["category_id"], "id": inst_id, "video_id": video_id, \
                    "bboxes": [None] * video_len, "areas": [None] * video_len}
            if "segmentation" in anno_dict:
                annotations_dict[video_id][inst_id]["segmentations"] = [None] * video_len
        # assert anno_dict["iscrowd"] != 1
        frame_id = imgid2frameid[img_id]
        annotations_dict[video_id][inst_id]["bboxes"][frame_id] = anno_dict["bbox"]
        annotations_dict[video_id][inst_id]["areas"][frame_id] = anno_dict["area"]
        if "segmentation" in anno_dict:
            annotations_dict[video_id][inst_id]["segmentations"][frame_id] = anno_dict["segmentation"]
    for video_id in sorted(annotations_dict.keys()):
        for inst_id in sorted(annotations_dict[video_id].keys()):
            des_dataset["annotations"].append(annotations_dict[video_id][inst_id])
    # save
    with open(args.des_json, "w") as f:
        json.dump(des_dataset, f)