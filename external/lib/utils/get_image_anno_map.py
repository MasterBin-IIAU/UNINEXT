import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="basic")
    parser.add_argument("--coco_root", type=str, default="/data/sdb/coco_2017", help="path to the coco root dir", required=True)
    parser.add_argument("--set", type=str, default="train", choices=["train", "val"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    """Build a mapping between image_id and instances"""
    args = parse_args()
    coco_root = args.coco_root
    json_path = os.path.join(coco_root, 'annotations/instances_%s2017.json' % args.set)
    with open(json_path) as f:
        data = json.load(f)
    images_list = data['images']  # length 118287
    anno_list = data['annotations']  # length 860001
    result = {}
    for ann in anno_list:
        img_id = ann["image_id"]
        if img_id not in result:
            result[img_id] = [ann]
        else:
            result[img_id].append(ann)
    """deal with images without instances belonging to the specific 80 classes"""
    """there are 1021 images without instances"""
    for img in images_list:
        id = img["id"]
        if id not in result:
            result[id] = []
    """save results"""
    result_path = os.path.join(coco_root, 'annotations/instances_%s2017_image_anno.json' % args.set)
    with open(result_path, "w") as f_w:
        json.dump(result, f_w)
