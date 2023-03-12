import argparse
import os
import pickle
from detectron2.projects.uninext.data.datasets.coco_video_dataset import CocoVideoDataset


def parse_args():
    parser = argparse.ArgumentParser(description='qdtrack test model')
    parser.add_argument("result", type=str, help="result file path")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # load gts
    dataset = CocoVideoDataset("datasets/bdd/labels/box_track_20/box_track_val_cocofmt.json")
    # load tracking results
    outputs = pickle.load(open(args.result, "rb"))
    # eval
    with open("eval_log.txt", "a") as f:
        print(args.result, file=f)
    print(dataset.evaluate(outputs, metric=["track"]))
