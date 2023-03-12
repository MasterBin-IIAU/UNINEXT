#!/usr/bin/env bash

mkdir -p datasets
mv BDD100K/bdd100k datasets/bdd

pip3 install -U numpy

{
    python3 -m bdd100k.label.to_coco -m det -i datasets/bdd/labels/det_20/det_train.json -o datasets/bdd/labels/det_20/det_train_cocofmt.json
} &
{
    python3 -m bdd100k.label.to_coco -m det -i datasets/bdd/labels/det_20/det_val.json -o datasets/bdd/labels/det_20/det_val_cocofmt.json
} &
{
    python3 -m bdd100k.label.to_coco -m box_track -i datasets/bdd/labels/box_track_20/train -o datasets/bdd/labels/box_track_20/box_track_train_cocofmt.json
} &
{
    python3 -m bdd100k.label.to_coco -m box_track -i datasets/bdd/labels/box_track_20/val -o datasets/bdd/labels/box_track_20/box_track_val_cocofmt.json
} &
{
    python3 -m bdd100k.label.to_coco -m ins_seg -i datasets/bdd/labels/ins_seg/polygons/ins_seg_train.json -o datasets/bdd/labels/ins_seg/polygons/ins_seg_train_cocoformat.json -mb datasets/bdd/labels/ins_seg/bitmasks/train
} &
{
    python3 -m bdd100k.label.to_coco -m ins_seg -i datasets/bdd/labels/ins_seg/polygons/ins_seg_val.json -o datasets/bdd/labels/ins_seg/polygons/ins_seg_val_cocoformat.json -mb datasets/bdd/labels/ins_seg/bitmasks/val
} &
{
    python3 conversion/clean_seg_track_json.py
    python3 -m bdd100k.label.to_coco -m seg_track -i datasets/bdd/labels/seg_track_20/polygons/train -o datasets/bdd/labels/seg_track_20/seg_track_train_cocoformat.json -mb datasets/bdd/labels/seg_track_20/bitmasks/train
} &
{
    python3 -m bdd100k.label.to_coco -m seg_track -i datasets/bdd/labels/seg_track_20/polygons/val -o datasets/bdd/labels/seg_track_20/seg_track_val_cocoformat.json -mb datasets/bdd/labels/seg_track_20/bitmasks/val
} &
wait