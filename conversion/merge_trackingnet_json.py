import json
import os


if __name__ == "__main__":
    data_root = "datasets/TrackingNet"
    splits = ["TRAIN_0", "TRAIN_1", "TRAIN_2", "TRAIN_3"]
    des_dataset = {'videos':[], 'categories':[{"supercategory": "object", "id": 1, "name": "object"}], 'annotations':[]}
    vid_id = 0
    for split in splits:
        print("merge %s split into TRAIN.json..." %split)
        cur_json_file = os.path.join(data_root, "%s.json"%split)
        dataset = json.load(open(cur_json_file, 'r'))
        # merge
        vids, annos = dataset["videos"], dataset["annotations"]
        assert len(vids) == len(annos)
        for vid, anno in zip(vids, annos):
            vid_id += 1
            vid["id"], anno["id"], anno["video_id"] = vid_id, vid_id, vid_id
            des_dataset["videos"].append(vid)
            des_dataset["annotations"].append(anno)
    # save
    with open(os.path.join(data_root, "TRAIN.json"), "w") as f:
        json.dump(des_dataset, f)