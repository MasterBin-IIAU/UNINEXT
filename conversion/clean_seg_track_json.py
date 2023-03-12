import os
import json


if __name__ == "__main__":
    data_root = "datasets/bdd/labels/seg_track_20/polygons/train"
    file_list = os.listdir(data_root)
    for fname in file_list:
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(data_root, fname), "r") as f:
            data = json.load(f)
        if "/" in data[0]["name"]:
            print(fname)
            for d in data:
                d["name"] = d["name"].split("/")[-1]
            with open(os.path.join(data_root, fname), "w") as f:
                json.dump(data, f)
        else:
            continue
