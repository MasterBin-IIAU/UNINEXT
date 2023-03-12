# -*- coding: utf-8 -*-

import os

from .refcoco import (
    register_refcoco,
    _get_refcoco_meta,
)
from .flicker import register_flicker, _get_flicker_meta
from detectron2.data.datasets.register_coco import register_coco_instances

# ==== Predefined splits for REFCOCO datasets ===========
_PREDEFINED_SPLITS_REFCOCO = {
    # refcoco
    "refcoco-unc-train": ("coco/train2014", "annotations/refcoco-unc/instances_train.json"),
    "refcoco-unc-val": ("coco/train2014", "annotations/refcoco-unc/instances_val.json"),
    "refcoco-unc-testA": ("coco/train2014", "annotations/refcoco-unc/instances_testA.json"),
    "refcoco-unc-testB": ("coco/train2014", "annotations/refcoco-unc/instances_testB.json"),
    # refcocog
    "refcocog-umd-train": ("coco/train2014", "annotations/refcocog-umd/instances_train.json"),
    "refcocog-umd-val": ("coco/train2014", "annotations/refcocog-umd/instances_val.json"),
    "refcocog-umd-test": ("coco/train2014", "annotations/refcocog-umd/instances_test.json"),
    "refcocog-google-val": ("coco/train2014", "annotations/refcocog-google/instances_val.json"),
    # refcoco+
    "refcocoplus-unc-train": ("coco/train2014", "annotations/refcocoplus-unc/instances_train.json"),
    "refcocoplus-unc-val": ("coco/train2014", "annotations/refcocoplus-unc/instances_val.json"),
    "refcocoplus-unc-testA": ("coco/train2014", "annotations/refcocoplus-unc/instances_testA.json"),
    "refcocoplus-unc-testB": ("coco/train2014", "annotations/refcocoplus-unc/instances_testB.json"),
    # mixed
    "refcoco-mixed": ("coco/train2014", "annotations/refcoco-mixed/instances_train.json"),
    "refcoco-mixed-filter": ("coco/train2014", "annotations/refcoco-mixed/instances_train_filter.json"),
}


def register_all_refcoco(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_REFCOCO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_refcoco(
            key,
            _get_refcoco_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

# ==== Predefined splits for flicker datasets ===========
_PREDEFINED_SPLITS_FLICKER = {
    # flicker-30k
    "flicker-train": ("flickr30k-images", "OpenSource/final_flickr_separateGT_train.json"),
}


def register_all_flicker(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_FLICKER.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_flicker(
            key,
            _get_flicker_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
    _get_ovis_instances_meta,
    _get_coco_video_instances_meta
)

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/annotations/instances_train_sub.json"),
    "ytvis_2019_val": ("ytvis_2019/val/JPEGImages",
                       "ytvis_2019/annotations/instances_val_sub.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
    "ytvis_2019_dev": ("ytvis_2019/train/JPEGImages",
                       "ytvis_2019/instances_train_sub.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/annotations/instances_train_sub.json"),
    "ytvis_2021_val": ("ytvis_2021/val/JPEGImages",
                       "ytvis_2021/annotations/instances_val_sub.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
    "ytvis_2021_dev": ("ytvis_2021/train/JPEGImages",
                       "ytvis_2021/instances_train_sub.json"),
    "ytvis_2022_val_full": ("ytvis_2022/val/JPEGImages",
                        "ytvis_2022/instances.json"),
    "ytvis_2022_val_sub": ("ytvis_2022/val/JPEGImages",
                       "ytvis_2022/instances_sub.json"),
}


_PREDEFINED_SPLITS_OVIS = {
    "ytvis_ovis_train": ("ovis/train",
                         "ovis/annotations_train.json"),
    "ytvis_ovis_val": ("ovis/valid",
                       "ovis/annotations_valid.json"),
    "ytvis_ovis_train_sub": ("ovis/train",
                         "ovis/ovis_sub_train.json"),
    "ytvis_ovis_val_sub": ("ovis/train",
                       "ovis/ovis_sub_val.json"),
}


_PREDEFINED_SPLITS_COCO_VIDS = {
    "coco_2017_train_video": ("coco/train2017", "coco/annotations/instances_train2017_video.json"),
    "coco_2017_val_video": ("coco/val2017", "coco/annotations/instances_val2017_video.json"),
}

_PREDEFINED_SPLITS_REFYTBVOS = {
    "rvos-refcoco-mixed": ("coco/train2014", "annotations/refcoco-mixed/instances_train_video.json"),
    "rvos-refytb-train": ("ref-youtube-vos/train/JPEGImages", "ref-youtube-vos/train.json"),
    "rvos-refytb-val": ("ref-youtube-vos/valid/JPEGImages", "ref-youtube-vos/valid.json"),
    "rvos-refdavis-val-0": ("ref-davis/valid/JPEGImages", "ref-davis/valid_0.json"),
    "rvos-refdavis-val-1": ("ref-davis/valid/JPEGImages", "ref-davis/valid_1.json"),
    "rvos-refdavis-val-2": ("ref-davis/valid/JPEGImages", "ref-davis/valid_2.json"),
    "rvos-refdavis-val-3": ("ref-davis/valid/JPEGImages", "ref-davis/valid_3.json"),

}


def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_coco_videos(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO_VIDS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_coco_video_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_refytbvos_videos(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_REFYTBVOS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_refcoco_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_expression=True
        )

from .bdd100k import (
    _get_bdd_obj_det_meta,
    _get_bdd_inst_seg_meta,
    _get_bdd_obj_track_meta
)

# ==== Predefined splits for BDD100K object detection ===========
_PREDEFINED_SPLITS_BDD_OBJ_DET = {
    "bdd_det_train": ("bdd/images/100k/train", "bdd/labels/det_20/det_train_cocofmt_uni.json"),
    "bdd_det_val": ("bdd/images/100k/val", "bdd/labels/det_20/det_val_cocofmt_uni.json"),
}

# ==== Predefined splits for BDD100K instance segmentation ===========
_PREDEFINED_SPLITS_BDD_INST_SEG = {
    "bdd_inst_train": ("bdd/images/10k/train", "bdd/labels/ins_seg/polygons/ins_seg_train_cocoformat_uni.json"),
    "bdd_inst_val": ("bdd/images/10k/val", "bdd/labels/ins_seg/polygons/ins_seg_val_cocoformat_uni.json"),
}

# ==== Predefined splits for BDD100K box tracking ===========
_PREDEFINED_SPLITS_BDD_BOX_TRACK = {
    "bdd_box_track_train": ("bdd/images/track/train", "bdd/labels/box_track_20/box_track_train_cocofmt_uni.json"),
    "bdd_box_track_val": ("bdd/images/track/val", "bdd/labels/box_track_20/box_track_val_cocofmt_uni.json"),
}

# ==== Predefined splits for BDD100K seg tracking ===========
_PREDEFINED_SPLITS_BDD_SEG_TRACK = {
    "bdd_seg_track_train": ("bdd/images/seg_track_20/train", "bdd/labels/seg_track_20/seg_track_train_cocoformat_uni.json"),
    "bdd_seg_track_val": ("bdd/images/seg_track_20/val", "bdd/labels/seg_track_20/seg_track_val_cocoformat_uni.json"),
}

# ==== Predefined splits for BDD100K mixed detection & tracking ===========
_PREDEFINED_SPLITS_BDD_DET_TRK_MIXED = {
    "bdd_det_trk_mixed_train": ("bdd/images", "bdd/labels/det_trk_mix.json"),
}

def register_all_bdd_obj_det(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BDD_OBJ_DET.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_bdd_obj_det_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_mask=False
        )


def register_all_bdd_inst_seg(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BDD_INST_SEG.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_bdd_inst_seg_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_mask=True
        )


def register_all_bdd_box_track(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BDD_BOX_TRACK.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_bdd_obj_track_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_mask=False
        )

def register_all_bdd_seg_track(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BDD_SEG_TRACK.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_bdd_obj_track_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_mask=True
        )

def register_all_bdd_det_trk_mix(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BDD_DET_TRK_MIXED.items():
        register_coco_instances(
            key,
            _get_bdd_obj_det_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_PREDEFINED_SPLITS_SOT = {
    "sot_got10k_train": ("GOT10K/train", "GOT10K/train.json"),
    "sot_got10k_val": ("GOT10K/val", "GOT10K/val.json"),
    "sot_got10k_test": ("GOT10K/test", "GOT10K/test.json"),
    "sot_lasot_train": ("LaSOT", "LaSOT/train.json"),
    "sot_lasot_test": ("LaSOT", "LaSOT/test.json"),
    "sot_lasot_ext_test": ("LaSOT_extension_subset", "LaSOT_extension_subset/test.json"),
    "sot_trackingnet_train": ("TrackingNet", "TrackingNet/TRAIN.json"),
    "sot_trackingnet_test": ("TrackingNet", "TrackingNet/TEST.json"),
    "sot_coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017_video_sot.json"),
    "sot_coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017_video_sot.json"),
    "sot_ytbvos18_train": ("ytbvos18/train/JPEGImages", "ytbvos18/train/train.json"),
    "sot_ytbvos18_val": ("ytbvos18/val/JPEGImages", "ytbvos18/val/val.json"),
    "sot_davis17_val": ("DAVIS/JPEGImages/480p", "DAVIS/2017_val.json"),
    "sot_nfs": ("nfs/sequences", "nfs/nfs.json"),
    "sot_uav123": ("UAV123/data_seq/UAV123", "UAV123/UAV123.json"),
    "sot_tnl2k_test": ("TNL-2K", "TNL-2K/test.json")
}

SOT_CATEGORIES = [{"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "object"}] # only one class for visual grounding

def _get_sot_meta():
    thing_ids = [k["id"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 1, len(thing_ids)

    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def register_all_sot(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_SOT.items():
        has_mask = ("coco" in key) or ("vos" in key) or ("davis" in key)
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_sot_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_mask=has_mask,
            sot=True
        )

if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    # refcoco/g/+
    register_all_refcoco(_root)
    # VIS
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_ovis(_root)
    register_all_coco_videos(_root)
    # BDD100K MOT & MOTS
    register_all_bdd_obj_det(_root)
    register_all_bdd_inst_seg(_root)
    register_all_bdd_box_track(_root)
    register_all_bdd_seg_track(_root)
    register_all_bdd_det_trk_mix(_root)
    # R-VOS
    register_all_refytbvos_videos(_root)
    # SOT
    register_all_sot(_root)
    # Phrase Grounding
    register_all_flicker(_root)