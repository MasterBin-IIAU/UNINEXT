import argparse
import os

import mmcv
from mmcv import Config, DictAction
import pickle
from detectron2.projects.uninext.data.datasets.coco_video_dataset import CocoVideoDataset
from tools_bin.to_bdd100k import preds2bdd100k


def parse_args():
    parser = argparse.ArgumentParser(description='qdtrack test model')
    parser.add_argument('--res', help='output result file')
    parser.add_argument(
        '--bdd-dir',
        type=str,
        help='path to the folder that will contain files in bdd100k format')
    parser.add_argument(
        '--task',
        type=str,
        nargs='+',
        help='task types',
        choices=['det', 'ins_seg', 'box_track', 'seg_track'])
    parser.add_argument(
        '--nproc',
        type=int,
        help='number of process for mask merging')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not os.path.isfile(args.res):
        raise ValueError('The result file does not exist.')

    dataset = CocoVideoDataset("datasets/bdd/labels/seg_track_20/seg_track_val_cocoformat.json")

    print(f'\nLoading results from {args.res}')

    # load tracking results
    results = pickle.load(open(args.res, "rb"))

    print("converting results to bdd100k...")
    preds2bdd100k(
        dataset, results, args.task, out_base=args.bdd_dir, nproc=args.nproc)
    print("Conversion is done.")

if __name__ == '__main__':
    main()
