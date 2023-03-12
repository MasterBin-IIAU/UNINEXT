#!/usr/bin/env bash

python3 conversion/convert_coco_to_video.py \
--src_json datasets/bdd/labels/det_20/det_train_cocofmt.json \
--des_json datasets/bdd/labels/det_20/det_train_cocofmt_uni.json
echo "BDD object detection train split is done"

python3 conversion/convert_coco_to_video.py \
--src_json datasets/bdd/labels/det_20/det_val_cocofmt.json \
--des_json datasets/bdd/labels/det_20/det_val_cocofmt_uni.json
echo "BDD object detection val split is done"

python3 conversion/convert_coco_to_video.py \
--src_json datasets/bdd/labels/ins_seg/polygons/ins_seg_train_cocoformat.json \
--des_json datasets/bdd/labels/ins_seg/polygons/ins_seg_train_cocoformat_uni.json
echo "BDD instance segmentation train split is done"

python3 conversion/convert_coco_to_video.py \
--src_json datasets/bdd/labels/ins_seg/polygons/ins_seg_val_cocoformat.json \
--des_json datasets/bdd/labels/ins_seg/polygons/ins_seg_val_cocoformat_uni.json
echo "BDD instance segmentation val split is done"

python3 conversion/convert_bddvid_to_cocovid.py \
--src_json datasets/bdd/labels/seg_track_20/seg_track_train_cocoformat.json \
--des_json datasets/bdd/labels/seg_track_20/seg_track_train_cocoformat_uni.json
echo "BDD seg tracking train split is done"

python3 conversion/convert_bddvid_to_cocovid.py \
--src_json datasets/bdd/labels/box_track_20/box_track_train_cocofmt.json \
--des_json datasets/bdd/labels/box_track_20/box_track_train_cocofmt_uni.json
echo "BDD box tracking train split is done"

python3 conversion/convert_bddvid_to_cocovid.py \
--src_json datasets/bdd/labels/seg_track_20/seg_track_val_cocoformat.json \
--des_json datasets/bdd/labels/seg_track_20/seg_track_val_cocoformat_uni.json
echo "BDD seg tracking val split is done"

python3 conversion/convert_bddvid_to_cocovid.py \
--src_json datasets/bdd/labels/box_track_20/box_track_val_cocofmt.json \
--des_json datasets/bdd/labels/box_track_20/box_track_val_cocofmt_uni.json
echo "BDD box tracking val split is done"


