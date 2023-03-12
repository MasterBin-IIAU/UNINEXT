import os
import json

if __name__ == "__main__":
    for dataset in ["refcoco-unc", "refcocog-umd", "refcocoplus-unc"]:
        json_path = "datasets/annotations/%s/instances.json" % dataset
        os.system("python3 conversion/convert_ref2coco.py --src_json %s --des_json %s" %(json_path, json_path))
    # merge train split
    merged_dir = "datasets/annotations/refcoco-mixed"
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    merged_json = "datasets/annotations/refcoco-mixed/instances_train.json"
    inst_idx = 0 # index of the instance
    new_data = {"images": [], "annotations": [], "categories": [{"supercategory": "object","id": 1,"name": "object"}]}
    for dataset in ["refcoco-unc", "refcocog-umd", "refcocoplus-unc"]:
        json_path = "datasets/annotations/%s/instances_train.json" % dataset
        data = json.load(open(json_path, 'r'))
        # for split in data.keys():
        for (img, anno) in zip(data["images"], data["annotations"]):
            inst_idx += 1
            img["id"] = inst_idx
            anno["image_id"] = inst_idx
            anno["id"] = inst_idx
            new_data["images"].append(img)
            new_data["annotations"].append(anno)
    json.dump(new_data, open(merged_json, 'w')) # 126908 referred objects