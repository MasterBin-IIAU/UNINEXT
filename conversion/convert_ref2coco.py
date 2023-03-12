import json
import argparse
import pycocotools.mask as maskUtils
from detectron2.structures import PolygonMasks

def parse_args():
    parser = argparse.ArgumentParser("json converter")
    parser.add_argument("--src_json", default="datasets/annotations/refcoco-unc/instances.json", type=str, help="the original json file")
    parser.add_argument("--des_json", default="datasets/annotations/refcoco-unc/instances.json", type=str, help="the processed json file")
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

if __name__ == "__main__":
    args = parse_args()
    data = json.load(open(args.src_json, 'r'))
    inst_idx = 0 # index of the instance
    for split in data.keys():
        new_data = {"images": [], "annotations": [], "categories": [{"supercategory": "object","id": 1,"name": "object"}]}
        for cur_data in data[split]:
            inst_idx += 1
            image = {"file_name": "COCO_train2014_%012d.jpg"%cur_data["image_id"], "height": cur_data["height"], "width": cur_data["width"], \
                "id": inst_idx, "expressions": cur_data["expressions"]}
            area = compute_area(cur_data["mask"])
            anno = {"bbox":cur_data["bbox"], "segmentation":cur_data["mask"], "image_id":inst_idx, \
                "iscrowd":0, "category_id":1, "id":inst_idx, "area": area}
            new_data["images"].append(image)
            new_data["annotations"].append(anno)
        assert len(new_data["images"]) == len(data[split])
        assert len(new_data["annotations"]) == len(data[split])
        output_json = args.des_json.replace(".json", "_%s.json"%split)
        json.dump(new_data, open(output_json, 'w'))