import copy
import logging
import random
import numpy as np
from typing import List, Union
import torch

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .augmentation import build_augmentation
from .coco_dataset_mapper_uni import filter_empty_instances_soft

__all__ = ["SOTDatasetMapper"]


def _get_dummy_anno(num_classes=-1, has_mask=True):
    anno = {
        "iscrowd": 0,
        "category_id": num_classes,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
    }
    if has_mask:
        anno["segmentation"] = [np.array([0.0] * 6)]
    return anno


def ytvis_annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes", "gt_ids",
            "gt_masks", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    ids = [int(obj["id"]) for obj in annos]
    ids = torch.tensor(ids, dtype=torch.int64)
    target.gt_ids = ids

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            masks.append(segm)
        # torch.from_numpy does not support array with negative stride.
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        )
        target.gt_masks = masks

    return target


class SOTDatasetMapper:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        augmentations_nocrop = None,
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_interval: int = 1,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
        cfg=None,
        test_categories=None,
        multidataset=False
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.multidataset           = multidataset
        if not self.multidataset:
            self.augmentations          = T.AugmentationList(augmentations)
            if augmentations_nocrop is not None:
                self.augmentations_nocrop   = T.AugmentationList(augmentations_nocrop)
            else:
                self.augmentations_nocrop   = None
        else:
            self.augmentations = [T.AugmentationList(x) for x in augmentations]
            self.augmentations_nocrop = [T.AugmentationList(x) if x is not None else None for x in augmentations_nocrop]
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_interval      = sampling_interval
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
        # language-guided detection
        self.lang_guide_det = cfg.MODEL.LANG_GUIDE_DET

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, test_categories=None):
        # augs = build_augmentation(cfg, is_train)
        if cfg.DATALOADER.SAMPLER_TRAIN == "MultiDatasetSampler" and is_train:
            multidataset = True
            assert len(cfg.INPUT.MIN_SIZE_TRAIN_MULTI) == len(cfg.INPUT.MAX_SIZE_TRAIN_MULTI)
            augs_nocrop, augs = [], []
            for (min_size_train, max_size_train) in zip(cfg.INPUT.MIN_SIZE_TRAIN_MULTI, cfg.INPUT.MAX_SIZE_TRAIN_MULTI):
                if cfg.INPUT.CROP.ENABLED and is_train:
                    augs_nocrop_cur, augs_cur = build_augmentation(cfg, is_train, min_size_train, max_size_train)
                else:
                    augs_cur = build_augmentation(cfg, is_train, min_size_train, max_size_train)
                    augs_nocrop_cur = None
                augs_nocrop.append(augs_nocrop_cur)
                augs.append(augs_cur)
        else:
            multidataset = False
            if cfg.INPUT.CROP.ENABLED and is_train:
                augs_nocrop, augs = build_augmentation(cfg, is_train, cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN)
            else:
                augs = build_augmentation(cfg, is_train, cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN)
                augs_nocrop = None
        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE
        sampling_interval = cfg.INPUT.SAMPLING_INTERVAL

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "augmentations_nocrop": augs_nocrop,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_interval": sampling_interval,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.DDETRS.NUM_CLASSES,
            "cfg": cfg,
            "test_categories": test_categories,
            "multidataset": multidataset
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        video_length = dataset_dict["length"]
        if self.is_train:
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame-self.sampling_frame_range)
            start_interval = max(0, ref_frame-self.sampling_interval+1)
            end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)
            end_interval = min(video_length, ref_frame+self.sampling_interval )
            
            selected_idx = np.random.choice(
                np.array(list(range(start_idx, start_interval)) + list(range(end_interval, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)
        # selected_idx is a List of length self.sampling_frame_num
        video_annos = dataset_dict.pop("annotations", None) # List
        file_names = dataset_dict.pop("file_names", None) # List

        if self.is_train:
            _ids = set()
            for frame_idx in selected_idx:
                _ids.update([anno["id"] for anno in video_annos[frame_idx]])
            ids = dict()
            for i, _id in enumerate(_ids):
                ids[_id] = i # original instance id -> zero-based

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []
        
        if self.augmentations_nocrop is not None and self.is_train:
            if np.random.rand() > 0.5:
                selected_augmentations = self.augmentations_nocrop
            else:
                selected_augmentations = self.augmentations
        else:
            selected_augmentations = self.augmentations
        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])

            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            try:
                utils.check_image_size(dataset_dict, image)
            except:
                # there are some videos with inconsistent resolutions...
                # eg. GOT10K/val/GOT-10k_Val_000137
                return None

            aug_input = T.AugInput(image)
            if self.multidataset and self.is_train:
                transforms = selected_augmentations[dataset_dict['dataset_source']](aug_input)
            else:
                transforms = selected_augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            # for SOT and VOS, we need the box anno in the 1st frame during inference
            # if (video_annos is None) or (not self.is_train):
            #     continue
            if not self.is_train:
                # NOTE copy() is to prevent annotations getting changed from applying augmentations
                _frame_annos = []
                for anno in video_annos[frame_idx]:
                    _anno = {}
                    for k, v in anno.items():
                        _anno[k] = copy.deepcopy(v)
                    _frame_annos.append(_anno)

                has_mask = dataset_dict["has_mask"]
                # USER: Implement additional transformations if you have other types of data
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in _frame_annos
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
                if instances.has("gt_masks"):
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                # add ori_id for VOS inference
                ori_id_list = [x["ori_id"] if "ori_id" in x else None for x in annos]
                instances.ori_id = ori_id_list
                dataset_dict["instances"].append(instances)
                continue

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []
            for anno in video_annos[frame_idx]:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _frame_annos.append(_anno)

            has_mask = dataset_dict["has_mask"]
            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _frame_annos
                if obj.get("iscrowd", 0) == 0
            ]
            sorted_annos = [_get_dummy_anno(has_mask=has_mask) for _ in range(len(ids))]

            for _anno in annos:
                idx = ids[_anno["id"]]
                sorted_annos[idx] = _anno
            _gt_ids = [_anno["id"] for _anno in sorted_annos]

            instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
            instances.gt_ids = torch.tensor(_gt_ids)
            instances_tmp = utils.filter_empty_instances(copy.deepcopy(instances))
            if len(instances_tmp) == 0:
                return None 
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = filter_empty_instances_soft(instances)
            # else:
            #     instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            if torch.sum(instances.gt_ids != -1) == 0:
                return None
            dataset_dict["instances"].append(instances)
        if not self.is_train:
            return dataset_dict
        # only keep one instance for SOT during training
        key_instances = dataset_dict["instances"][0]
        ref_instances = dataset_dict["instances"][1]
        key_ids = key_instances.gt_ids.tolist()
        ref_ids = ref_instances.gt_ids.tolist()
        valid_key_ids = [x for x in key_ids if x != -1]
        valid_ref_ids = [x for x in ref_ids if x != -1]
        valid_ids_both = []
        for index in valid_key_ids:
            if index in valid_ref_ids:
                valid_ids_both.append(index)
        if len(valid_ids_both) == 0:
            return None
        else:
            pick_id = random.choice(valid_ids_both)
            new_instances = []
            for _ in range(len(key_instances)):
                if key_ids[_] == pick_id:
                    new_instances.append(key_instances[_])
                    break
            for _ in range(len(ref_instances)):
                if ref_ids[_] == pick_id:
                    new_instances.append(ref_instances[_])
                    break
            dataset_dict["instances"] = new_instances
        # add positive_map
        for instances in dataset_dict["instances"]:
            instances.positive_map = torch.ones((1, 1), dtype=torch.bool) # 1 instance, 1 (pooled) token.
        return dataset_dict

