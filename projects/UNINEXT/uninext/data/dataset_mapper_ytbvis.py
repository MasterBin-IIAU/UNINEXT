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
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from .datasets.ytvis import YTVIS_CATEGORIES_2019, YTVIS_CATEGORIES_2021, OVIS_CATEGORIES
from .datasets.bdd100k import BDD_DET_CATEGORIES, BDD_INST_CATEGORIES, BDD_TRACK_CATEGORIES
from .coco_dataset_mapper_uni import cat2ind, RobertaTokenizerFast, AutoTokenizer, ConvertCocoPolysToMask, create_queries_and_maps, \
    check_for_positive_overflow, convert_object_detection_to_grounding_optimized_for_od, filter_empty_instances_soft
import re
from fvcore.transforms.transform import HFlipTransform
__all__ = ["YTVISDatasetMapper"]


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


class YTVISDatasetMapper:
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
        if self.lang_guide_det:
            self.ind_to_class_dict = {}
            datasets = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
            for dataset_name in datasets:
                if dataset_name.startswith("ytvis_2019"):
                    self.ind_to_class_dict["vis19"] = cat2ind(YTVIS_CATEGORIES_2019)
                elif dataset_name.startswith("ytvis_2021"):
                    self.ind_to_class_dict["vis21"] = cat2ind(YTVIS_CATEGORIES_2021)
                elif dataset_name.startswith("ytvis_ovis"):
                    self.ind_to_class_dict["ovis"] = cat2ind(OVIS_CATEGORIES)
                elif dataset_name.startswith("coco"):
                    self.ind_to_class_dict["coco"] = cat2ind(COCO_CATEGORIES)
                elif dataset_name.startswith("bdd_det"):
                    self.ind_to_class_dict["bdd_det"] = cat2ind(BDD_DET_CATEGORIES)
                elif dataset_name.startswith("bdd_inst"):
                    self.ind_to_class_dict["bdd_inst"] = cat2ind(BDD_INST_CATEGORIES)
                elif dataset_name.startswith("bdd_box_track") or dataset_name.startswith("bdd_seg_track"):
                    self.ind_to_class_dict["bdd_track"] = cat2ind(BDD_TRACK_CATEGORIES)
                elif dataset_name.startswith("refytb-val") or dataset_name.startswith("rvos"):
                    pass
                else:
                    raise ValueError("Unsupported dataset_name: %s"%dataset_name)
            use_roberta = cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "roberta-base" and cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "roberta-base"
            if use_roberta:
                self.tokenizer = RobertaTokenizerFast.from_pretrained('projects/UNINEXT/roberta-base')
            else:
                self.tokenizer = AutoTokenizer.from_pretrained('projects/UNINEXT/bert-base-uncased') # align with GLIP
            self.max_query_len = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN
            self.prepare = ConvertCocoPolysToMask(
                return_tokens=True,
                tokenizer=self.tokenizer,
                max_query_len=self.max_query_len
            )
            self.prompt_test_dict = {}
            self.positive_map_label_to_token_dict = {}
            if test_categories is not None:
                prompt_test, positive_map_label_to_token = create_queries_and_maps(test_categories, self.tokenizer) # for example, test_categories = [{"name": "person"}]
            else:
                for dataset_name in datasets:
                    if dataset_name.startswith("ytvis_2019"):
                        prompt_test, positive_map_label_to_token = create_queries_and_maps(YTVIS_CATEGORIES_2019, self.tokenizer)
                        self.prompt_test_dict["vis19"] = prompt_test
                        self.positive_map_label_to_token_dict["vis19"] = positive_map_label_to_token
                    elif dataset_name.startswith("ytvis_2021"):
                        prompt_test, positive_map_label_to_token = create_queries_and_maps(YTVIS_CATEGORIES_2021, self.tokenizer)
                        self.prompt_test_dict["vis21"] = prompt_test
                        self.positive_map_label_to_token_dict["vis21"] = positive_map_label_to_token
                    elif dataset_name.startswith("ytvis_ovis"):
                        prompt_test, positive_map_label_to_token = create_queries_and_maps(OVIS_CATEGORIES, self.tokenizer)
                        self.prompt_test_dict["ovis"] = prompt_test
                        self.positive_map_label_to_token_dict["ovis"] = positive_map_label_to_token
                    elif dataset_name.startswith("coco"):
                        prompt_test, positive_map_label_to_token = create_queries_and_maps(COCO_CATEGORIES, self.tokenizer)
                        self.prompt_test_dict["coco"] = prompt_test
                        self.positive_map_label_to_token_dict["coco"] = positive_map_label_to_token
                    elif dataset_name.startswith("bdd_det"):
                        prompt_test, positive_map_label_to_token = create_queries_and_maps(BDD_DET_CATEGORIES, self.tokenizer)
                        self.prompt_test_dict["bdd_det"] = prompt_test
                        self.positive_map_label_to_token_dict["bdd_det"] = positive_map_label_to_token
                    elif dataset_name.startswith("bdd_inst"):
                        prompt_test, positive_map_label_to_token = create_queries_and_maps(BDD_INST_CATEGORIES, self.tokenizer)
                        self.prompt_test_dict["bdd_inst"] = prompt_test
                        self.positive_map_label_to_token_dict["bdd_inst"] = positive_map_label_to_token
                    elif dataset_name.startswith("bdd_box_track") or dataset_name.startswith("bdd_seg_track"):
                        prompt_test, positive_map_label_to_token = create_queries_and_maps(BDD_TRACK_CATEGORIES, self.tokenizer)
                        self.prompt_test_dict["bdd_track"] = prompt_test
                        self.positive_map_label_to_token_dict["bdd_track"] = positive_map_label_to_token
                    elif dataset_name.startswith("refytb-val") or dataset_name.startswith("rvos"):
                        pass
                    else:
                        raise ValueError("Unsupported dataset_name: %s"%dataset_name)

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
        if "expressions" not in dataset_dict:
            dataset_dict["expressions"] = []
        task = dataset_dict["task"] if "task" in dataset_dict else None
        
        if self.augmentations_nocrop is not None and self.is_train:
            if np.random.rand() > 0.5:
                selected_augmentations = self.augmentations_nocrop
            else:
                selected_augmentations = self.augmentations
        else:
            selected_augmentations = self.augmentations
        if task == "grounding":
            dataset_dict["expressions_ground"] = []
        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])

            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            if self.multidataset and self.is_train:
                transforms = selected_augmentations[dataset_dict['dataset_source']](aug_input)
            else:
                transforms = selected_augmentations(aug_input)
            if task == "grounding":
                dataset_dict["expressions_ground"].append(self.transform_expressions(dataset_dict["expressions"], transforms))
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (video_annos is None) or (not self.is_train):
                if self.lang_guide_det and task == "detection":
                    dataset_dict["expressions"].append(self.prompt_test_dict[dataset_dict["dataset_name"]])
                    if frame_idx == 0:
                        dataset_dict["positive_map_label_to_token"] = self.positive_map_label_to_token_dict[dataset_dict["dataset_name"]]
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

            if self.lang_guide_det and task == "detection":
                ind_to_class = self.ind_to_class_dict[dataset_dict["dataset_name"]]
                original_box_num = len(instances)
                instances, positive_caption_length = check_for_positive_overflow(instances, ind_to_class, self.tokenizer, self.max_query_len-2)
                if len(instances) < original_box_num:
                    print("WARNING: removed {} boxes due to positive caption overflow".format(original_box_num - len(instances)))
                annotations, caption, label_to_positions = convert_object_detection_to_grounding_optimized_for_od(
                    instances=instances, ind_to_class=ind_to_class,
                    positive_caption_length=positive_caption_length,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.max_query_len-2
                )
                anno = {"annotations": annotations, "caption": caption, "label_to_positions": label_to_positions}
                anno = self.prepare(anno)
                instances.positive_map = anno["positive_map"].bool() # (N, max_seq_len). N is num of objects. bool() -> 0 or 1
                expressions_new = anno["caption"] # "expressions" are shared between detection and grounding
                # postive_map must be consistent with expressions！
                dataset_dict["expressions"].append(expressions_new) # expression for the key frame
            elif self.lang_guide_det and task == "grounding":
                if len(instances) != 0:
                    instances.positive_map = torch.ones((1, 1), dtype=torch.bool) # 1 instance, 1 (pooled) token.
                else:
                    print("invalid instance is found:", instances, annos, dataset_dict["dataset_name"], file_names[frame_idx], dataset_dict["expressions_ground"])
            else:
                raise ValueError()
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
        if task == "grounding":
            dataset_dict["expressions"] = dataset_dict["expressions_ground"]
        return dataset_dict

    def transform_expressions(self, expressions, transforms):
        # pick one expression if there are multiple expressions
        if self.is_train:
            expression = expressions[np.random.choice(len(expressions))]
            expression = clean_string(expression)
        else:
            if isinstance(expressions[0], list):
                # for refdavis, the json has been preprocessed
                # so "expressions": [["exp1", "exp2", ...]]
                expression = [clean_string(e) for e in expressions[0]]  # list
            else:
                # for refcoco and refytvos, the json has been preprocessed
                # so only one "expressions": ["exp1"]
                expression = expressions[0]
                expression = clean_string(expression)                   # str
        # deal with hflip for expression
        hflip_flag = False
        for x in transforms:
            if isinstance(x, HFlipTransform):
                hflip_flag = True
                break
        if hflip_flag:
            if isinstance(expression, list):
                expression = [e.replace('left', '@').replace('right', 'left').replace('@', 'right') for e in expression]
            else:
                expression = expression.replace('left', '@').replace('right', 'left').replace('@', 'right')
        return expression

def clean_string(expression):
    return re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ')