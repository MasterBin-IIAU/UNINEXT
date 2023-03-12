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

__all__ = ["CocoDatasetMapper"]


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances


class CocoDatasetMapper:
    """
    A callable which takes a COCO image which converts into multiple frames,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 1,
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
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        # fmt: on
        # import pdb;pdb.set_trace()
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # import pdb;pdb.set_trace()
        img_annos = dataset_dict.pop("annotations", None)
        # file_name = dataset_dict.pop("file_name", None)
        file_name = dataset_dict['file_name']
        original_image = utils.read_image(file_name, format=self.image_format)

        # dataset_dict["image"] = []
        # dataset_dict["instances"] = []
        # dataset_dict["file_names"] = [file_name] * self.sampling_frame_num


        utils.check_image_size(dataset_dict, original_image)
        aug_input = T.AugInput(original_image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if (img_annos is None) or (not self.is_train):
            dataset_dict.pop("annotations", None)
            return dataset_dict

        # import pdb;pdb.set_trace()
        _img_annos = []
        for anno in img_annos:
            _anno = {}
            for k, v in anno.items():
                _anno[k] = copy.deepcopy(v)
            _img_annos.append(_anno)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in _img_annos
            if obj.get("iscrowd", 0) == 0
        ]
        _gt_ids = list(range(len(annos)))
        for idx in range(len(annos)):
            if len(annos[idx]["segmentation"]) == 0:
                annos[idx]["segmentation"] = [np.array([0.0] * 6)]

        instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
        instances.gt_ids = torch.tensor(_gt_ids)
        if instances.has("gt_masks"):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = filter_empty_instances(instances)
            if len(instances) == 0:
                return None 
        else:
            instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
        dataset_dict["instances"]=instances
        # import pdb;pdb.set_trace()


        

        return dataset_dict
