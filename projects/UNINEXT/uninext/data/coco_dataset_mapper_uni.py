# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import re

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
import random
from transformers import AutoTokenizer
from collections import defaultdict
from transformers import RobertaTokenizerFast
from fvcore.transforms.transform import HFlipTransform
from .objects365_v2 import categories as OBJECTS365V2_CATEGORIES
from .seginw import CATEGORIES as SEGINW_CATEGORIES
import os

__all__ = ["DetrDatasetMapper"]


def cat2ind(categories):
    ind_to_class = {0: '__background__'}
    index = 1
    for x in categories:
        isthing = x["isthing"] if "isthing" in x else 1
        if isthing == 1:
            ind_to_class[index] = x["name"]
            index += 1
    return ind_to_class


def create_queries_and_maps(categories, tokenizer, separation_tokens=". "):
    label_list = []
    for x in categories:
        isthing = x["isthing"] if "isthing" in x else 1
        if isthing == 1:
            label_list.append(x["name"])
    labels = list(range(1, len(label_list) + 1)) # [1, 2, ..., 80]

    # Clean label list
    label_list = [clean_name(i) for i in label_list]
    # Form the query and get the mapping
    tokens_positive = []
    start_i = 0
    end_i = 0
    objects_query = ""

    # sep between tokens, follow training
    separation_tokens = ". "
    
    for _index, label in enumerate(label_list):
        
        start_i = len(objects_query)

        objects_query += label
        
        end_i = len(objects_query)
        tokens_positive.append([(start_i, end_i)])  # Every label has a [(start, end)]

        if _index != len(label_list) - 1:
            objects_query += separation_tokens

    # print(objects_query) # 'person. bicycle. car. motorcycle. airplane. bus. train. truck. boat. traffic light. fire hydrant. stop sign. parking meter. bench. bird. cat. dog. horse. sheep. cow. elephant. bear. zebra. giraffe. backpack. umbrella. handbag. tie. suitcase. frisbee. skis. snowboard. sports ball. kite. baseball bat. baseball glove. skateboard. surfboard. tennis racket. bottle. wine glass. cup. fork. knife. spoon. bowl. banana. apple. sandwich. orange. broccoli. carrot. hot dog. pizza. donut. cake. chair. couch. potted plant. bed. dining table. toilet. tv. laptop. mouse. remote. keyboard. cell phone. microwave. oven. toaster. sink. refrigerator. book. clock. vase. scissors. teddy bear. hair drier. toothbrush'

    tokenized = tokenizer(objects_query, return_tensors="pt")

    # Create the mapping between tokenized sentence and the original label
    positive_map_token_to_label, positive_map_label_to_token = create_positive_dict(tokenized, tokens_positive, labels=labels)  # from token position to original label
    return objects_query, positive_map_label_to_token


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens

# Unified DataMapper for image-level tasks
class DetrDatasetMapperUni:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DETR.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True, test_categories=None):
        # test_categories: categories to detect during testing
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.mask_on = cfg.MODEL.MASK_ON
        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        
        # language-guided detection
        self.lang_guide_det = cfg.MODEL.LANG_GUIDE_DET
        if self.lang_guide_det:
            self.ind_to_class_dict = {}
            self.ind_to_class_dict["coco"] = cat2ind(COCO_CATEGORIES)
            self.ind_to_class_dict["obj365v2"] = cat2ind(OBJECTS365V2_CATEGORIES)
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
                for dataset_name in cfg.DATASETS.TRAIN:
                    if dataset_name.startswith("coco"):
                        prompt_test, positive_map_label_to_token = create_queries_and_maps(COCO_CATEGORIES, self.tokenizer)
                        self.prompt_test_dict["coco"] = prompt_test
                        self.positive_map_label_to_token_dict["coco"] = positive_map_label_to_token
                    elif dataset_name.startswith("objects365_v2"):
                        prompt_test, positive_map_label_to_token = create_queries_and_maps(OBJECTS365V2_CATEGORIES, self.tokenizer)
                        self.prompt_test_dict["obj365v2"] = prompt_test
                        self.positive_map_label_to_token_dict["obj365v2"] = positive_map_label_to_token
                if cfg.DATASETS.TEST[0].startswith("seginw"):
                    for dataset_name in cfg.DATASETS.TEST:
                        if dataset_name.startswith("seginw"):
                            prompt_test, positive_map_label_to_token = create_queries_and_maps(SEGINW_CATEGORIES[dataset_name], self.tokenizer)
                            self.prompt_test_dict[dataset_name] = prompt_test
                            self.positive_map_label_to_token_dict[dataset_name] = positive_map_label_to_token
        # oridinal numbers
        self.ordinal_nums = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
    
    def transform_img(self, image, disable_crop=False):
        if self.crop_gen is None or disable_crop:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        return image, image_shape, transforms
    
    def transform_expressions(self, expressions, transforms):
        # pick one expression if there are multiple expressions
        expression = expressions[np.random.choice(len(expressions))]
        expression = clean_string(expression)
        # deal with hflip for expression
        hflip_flag = False
        for x in transforms:
            if isinstance(x, HFlipTransform):
                hflip_flag = True
                break
        if hflip_flag:
            expression = expression.replace('left', '@').replace('right', 'left').replace('@', 'right')
        return expression

    def transform_annos(self, annotations_ori, transforms, image_shape, dataset_dict):
        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in annotations_ori
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
        
        # language-guided detection
        task = dataset_dict["task"] if "task" in dataset_dict else None
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
        elif self.lang_guide_det and task == "grounding":
            instances.positive_map = torch.ones((1, 1), dtype=torch.bool) # 1 instance, 1 (pooled) token.
            expressions_new = dataset_dict["expressions"]
        elif self.lang_guide_det and task == "phrase_grounding":
            expressions_new = dataset_dict["expressions"]
            anno = {"annotations": dataset_dict["annotations"], "caption": expressions_new}
            anno = self.prepare(anno)
            instances.positive_map = anno["positive_map"].bool() # (N, max_seq_len). N is num of objects. bool() -> 0 or 1
            expressions_new = anno["caption"] # "expressions" are shared between detection and grounding
        else:
            raise ValueError("task must be detection or grounding")
        if hasattr(instances, "gt_masks"):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        
        return instances, expressions_new

    def has_ordinal_num(self, expressions_list):
        flag = False
        for expression in expressions_list:
            expression_low = expression.lower()
            for word in self.ordinal_nums:
                if word in expression_low:
                    flag = True
                    break
            if flag == True:
                break
        return flag

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.mask_on:
                anno.pop("segmentation", None)
            anno.pop("keypoints", None)
        # if there are ordinal numbers in expressions, disable crop
        disable_crop = self.has_ordinal_num(dataset_dict["expressions"]) if "expressions" in dataset_dict else False
        dataset_dict["image"], image_shape, transforms = self.transform_img(image, disable_crop=disable_crop)
        if "expressions" in dataset_dict and dataset_dict["task"] == "grounding":
            dataset_dict["expressions"] = self.transform_expressions(dataset_dict["expressions"], transforms)
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            # language-guided detection
            task = dataset_dict["task"] if "task" in dataset_dict else None
            if self.lang_guide_det and task == "detection":
                if dataset_dict["dataset_name"] == "seginw":
                    name_list = dataset_dict["file_name"].split("/")[1:3]
                    dataset_name = os.path.join(*name_list)
                    dataset_dict["expressions"] = self.prompt_test_dict[dataset_name]
                    dataset_dict["positive_map_label_to_token"] = self.positive_map_label_to_token_dict[dataset_name]
                else:
                    dataset_dict["expressions"] = self.prompt_test_dict[dataset_dict["dataset_name"]]
                    dataset_dict["positive_map_label_to_token"] = self.positive_map_label_to_token_dict[dataset_dict["dataset_name"]]
            return dataset_dict

        if "annotations" in dataset_dict:
            instances, expressions_new = self.transform_annos(dataset_dict["annotations"], transforms, image_shape, dataset_dict)
            # add "expressions" for detection data
            dataset_dict["expressions"] = expressions_new
            instances = utils.filter_empty_instances(instances)
            if len(instances) == 0:
                return None 
            dataset_dict["instances"] = instances
        if dataset_dict["task"] == "phrase_grounding":
            dataset_dict["task"] = "detection"
        return dataset_dict

# generate image pairs (reference-key) based on still images
# This mapper is used only for training
class DetrDatasetMapperUniCLIP(DetrDatasetMapperUni):
    def __init__(self, cfg, is_train=True, test_categories=None):
        super().__init__(cfg, is_train=is_train, test_categories=test_categories)
        assert self.is_train
    
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image_key = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image_key)

        # USER: Modify this if you want to keep them for some reason.
        dataset_dict["image"] = []
        for anno in dataset_dict["annotations"]:
            if not self.mask_on:
                anno.pop("segmentation", None)
            anno.pop("keypoints", None)
        annotations_key = dataset_dict.pop("annotations")
        annotations_ref = copy.deepcopy(annotations_key)
        image_ref = copy.deepcopy(image_key)
    
        image_key, image_shape_key, transforms_key = self.transform_img(image_key)
        dataset_dict["image"].append(image_key)
        image_ref, image_shape_ref, transforms_ref = self.transform_img(image_ref)
        dataset_dict["image"].append(image_ref)
        assert "expressions" not in dataset_dict

        dataset_dict["expressions"] = []
        instances_key, expressions_new_key = self.transform_annos(annotations_key, transforms_key, image_shape_key, dataset_dict)
        instances_ref, expressions_new_ref = self.transform_annos(annotations_ref, transforms_ref, image_shape_ref, dataset_dict)
        # add "expressions" for detection data
        dataset_dict["expressions"].append(expressions_new_key)
        dataset_dict["expressions"].append(expressions_new_ref)


        instances_key_tmp = utils.filter_empty_instances(copy.deepcopy(instances_key))
        instances_ref_tmp = utils.filter_empty_instances(copy.deepcopy(instances_ref))
        if len(instances_key_tmp) == 0 or len(instances_ref_tmp) == 0:
            return None 
        _gt_ids = list(range(1,1+len(instances_ref)))
        instances_key.gt_ids = torch.tensor(_gt_ids)
        instances_ref.gt_ids = torch.tensor(_gt_ids)
        dataset_dict["instances"] = [filter_empty_instances_soft(instances_key),  filter_empty_instances_soft(instances_ref)] # instances of two frames
        # for key/ref frame， we don't remove empty instances，but mark them with gt_ids=-1, and process them in idol.py
        # gt_ids has no practical meaning, we just use it as a flag to indicate whether an instance exists, 
        # idx indicates the object correspondence between key&reference frame

        return dataset_dict

def filter_empty_instances_soft(instances, by_box=True, by_mask=True, box_threshold=1e-5):
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

    instances.gt_ids[~m] = -1 # invalid instances are marked with -1
    return instances

def clean_string(expression):
    return re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ')

def check_for_positive_overflow(instances, ind_to_class, tokenizer, max_seq_length=256):
    # NOTE: Only call this function for OD data; DO NOT USE IT FOR GROUNDING DATA
    # NOTE: called only in coco_dt

    # Check if we have too many positive labels
    # generate a caption by appending the positive labels
    positive_label_set = set()
    for i in range(len(instances)):
        label_i = instances.gt_classes[i].item() + 1 # "+1" for mapping 0~79 to 1~80
        positive_label_set.add(label_i)
    positive_label_list = list(positive_label_set)

    # random shuffule so we can sample different annotations at different epochs
    random.shuffle(positive_label_list)

    kept_lables = []
    length = 0

    for index, label in enumerate(positive_label_list):

        label_text = clean_name(ind_to_class[label]) + ". " # "dog. "

        tokenized = tokenizer.tokenize(label_text)

        length += len(tokenized)

        if length > max_seq_length: # there could not be overflow for COCO dataset
            break
        else:
            kept_lables.append(label)
    
    ## filter boxes
    keep_box_index = []
    for i in range(len(instances)):
        label_i = instances.gt_classes[i].item() + 1 # "+1" for mapping 0~79 to 1~80
        if label_i in kept_lables:
            keep_box_index.append(i)
    
    # keep_box_index = torch.LongTensor(keep_box_index)
    instances = instances[keep_box_index] ## filter boxes

    return instances, length

def clean_name(name):
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    return name

def convert_object_detection_to_grounding_optimized_for_od(
        instances,
        ind_to_class,
        disable_shuffle=False,
        add_detection_prompt=False,
        add_detection_prompt_advanced=False,
        random_sample_negative=85,
        control_probabilities=(0.0, 0.0, 0.5, 0.0),
        restricted_negative_list=None,
        separation_tokens=". ",
        max_num_labels=-1,
        max_seq_length=256,
        tokenizer=None,
        positive_caption_length=0
):
    '''
    ind_to_class: {0: "__background__", 1 : "person" ...}
    instances:

    restricted_negative_list : for datasets with restricted negatives, sample only the negatives

    Convert object detection data into grounding data format, on the fly.

    Control options:
        1. add_detection_prompt: add "object detection : " to the front of the prompt
        2. num_negatives: randomly sampled negative classes
        3. num_positives: how many positives to keep (-1 means do not cut any)

    Probabilities to generate the control options:

        a. probability_one_negative: only give one negative class to mimic evaluation
        b. probability_one_positive: only give one positive class to mimic evaluation
        c. probability_full: add both all positive and all negatives
        d. other:
            randomly sample some negatives and some positives
            The below control options are independent of each other:
            - probability_random_negative: probability of randomly sample X negatives
            - probability_random_positive: probability of randomly sample some positives
    '''
    if restricted_negative_list is None: # True
        valid_negative_indexes = list(ind_to_class.keys()) # [0, 1, 2, ... 80]
    else:
        valid_negative_indexes = restricted_negative_list
    # import ipdb; ipdb.set_trace()
    def generate_senetence_given_labels(
            positive_label_list,
            negative_label_list,
            prompt_engineer_version="v2",
            disable_shuffle=False):

        '''
        v3: with simple prompt such as "there are", "are there?"
        v4: try to merge some are there / there are together, to avoid sequence being too long
        '''

        label_to_positions = {}

        assert (prompt_engineer_version == "v2")
        num_negatives = len(negative_label_list)
        num_positives = len(positive_label_list)
        label_list = negative_label_list + positive_label_list
        if not disable_shuffle: # True
            random.shuffle(label_list)

        if add_detection_prompt: # False
            if add_detection_prompt_advanced and (num_negatives == 0 or num_positives == 0) and not disable_shuffle:
                pheso_caption = "object detection query : "
            else:
                pheso_caption = "object detection : "
        else:
            pheso_caption = ""

        for index, label in enumerate(label_list):

            start_index = len(pheso_caption)

            pheso_caption += clean_name(ind_to_class[label])  # NOTE: slight change...
            end_index = len(pheso_caption)

            # e.g.: pheso_caption = "cat dog", where cat is label 4, and dog is label 17
            # label_to_positions: {4: (0, 3), 17: (4, 7)}
            label_to_positions[label] = [start_index, end_index]

            if index != len(label_list) - 1:
                pheso_caption += separation_tokens # += ". "

        return label_to_positions, pheso_caption

    if disable_shuffle: # False
        label_list = list(sorted(ind_to_class.keys()))[1:]  # do not include the background
        label_to_positions, pheso_caption = generate_senetence_given_labels(
            positive_label_list=label_list,
            negative_label_list=[],
            disable_shuffle=True)
        # print(label_to_positions, pheso_caption)
    else:
        positive_label_set = set()
        for i in range(len(instances)):
            label_i = instances.gt_classes[i].item() + 1
            positive_label_set.add(label_i)

        full_positive = len(positive_label_set) # num classes containing in the current image
        if max_num_labels <= 0: # -1
            full_negative = random_sample_negative # 85
        else:
            full_negative = max(min(max_num_labels-full_positive, random_sample_negative), 0)

        if full_negative > len(valid_negative_indexes): # True (85 > 81)
            full_negative = len(valid_negative_indexes) # 81

        num_negatives, num_positives = generate_control_options_given_probabilities(
            control_probabilities=control_probabilities, # (0.0, 0.0, 0.5, 0.0)
            full_positive=full_positive,
            full_negative=full_negative)
        # num_positives not used
        

        # Keep some negatives
        negative_label_list = set()
        if num_negatives != -1:
            if num_negatives > len(valid_negative_indexes):
                num_negatives = len(valid_negative_indexes)
            for i in np.random.choice(valid_negative_indexes, size=num_negatives, replace=False):
                # label_sets.add(i)
                if i not in positive_label_set:
                    negative_label_list.add(i)

        # Keep all positives; ignoring num_positives
        positive_label_list = list(positive_label_set)
        random.shuffle(positive_label_list)

        negative_label_list = list(negative_label_list)  # e.g.: [17, 1, 13] where each number is the class name
        random.shuffle(negative_label_list)

        # Do a pre-screen. If we cannot afford this many negatives, we will sample less
        negative_max_length = max_seq_length - positive_caption_length
        screened_negative_label_list = []
        for negative_label in negative_label_list:
            label_text = clean_name(ind_to_class[negative_label]) + ". " # "dog. "

            tokenized = tokenizer.tokenize(label_text)
            
            negative_max_length -= len(tokenized)

            if negative_max_length > 0: 
                screened_negative_label_list.append(negative_label) # keep this negative
            else:
                break
        negative_label_list = screened_negative_label_list

        label_to_positions, pheso_caption = generate_senetence_given_labels(
            positive_label_list=positive_label_list,
            negative_label_list=negative_label_list)
    new_target = []
    # label_to_positions: dict
    # key: class index (range from 0-80)
    # value: their (char-level) positions in the caption
    for i in range(len(instances)):
        new_target_i = {}
        label_i = instances.gt_classes[i].item() + 1
        if label_i in label_to_positions:  # NOTE: Only add those that actually appear in the final caption
            new_target_i["tokens_positive"] = [label_to_positions[label_i]]
            new_target.append(new_target_i)
    return new_target, pheso_caption, label_to_positions


def generate_control_options_given_probabilities(
        control_probabilities,
        full_positive,
        full_negative):
    
    # The function was originally designed to perform data augmentation by randomly dropping negative and positive classes. Later, we decided to only consider dropping negative classes. So the returned 'num_positives' by this function will be ignored.
    # 0828 use all positive classes. prob 0.5 -> use all negative classes; prob 0.5 -> random number of negative classes
    outer_prob = random.random()

    probability_one_negative = control_probabilities[0]
    probability_one_positive = control_probabilities[1]
    probability_full = control_probabilities[2] # 0.5
    probability_drop_positive = control_probabilities[3]

    assert(probability_drop_positive == 0)

    if outer_prob < probability_one_negative:
        # a. probability_one_negative: only give one negative class to mimic evaluation (10%)
        num_negatives = 1
        num_positives = 0
    elif outer_prob < probability_one_positive + probability_one_negative:
        # b. probability_one_positive: only give one positive class to mimic evaluation (10%)
        num_negatives = 0
        num_positives = 1
    elif outer_prob < probability_full + probability_one_positive + probability_one_negative: # prob 0.5
        # c. probability_full: add both all positive and all negatives (20%)
        num_negatives = full_negative
        num_positives = full_positive
    else: # prob 0.5
        if random.random() < 1.0:  # - probability_random_negative: probability of randomly sample X negatives (100%)
            num_negatives = np.random.choice(max(1, full_negative)) + 1  # mininum 1
        else:
            num_negatives = full_negative  # Full

        if random.random() < probability_drop_positive:  # False
            num_positives = np.random.choice(max(1, full_positive)) + 1
        else:
            num_positives = full_positive  # Full

    return num_negatives, num_positives

class ConvertCocoPolysToMask(object):
    def __init__(self, return_tokens=False, tokenizer=None, max_query_len=256):
        self.return_tokens = return_tokens # True
        self.tokenizer = tokenizer # AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_query_len = max_query_len

    def __call__(self, target):

        anno = target["annotations"]
        caption = target["caption"] if "caption" in target else None
        tokens_positive = [obj["tokens_positive"] for obj in anno]

        target = {}
        if caption is not None:
            target["caption"] = caption

        if tokens_positive is not None:
            target["tokens_positive"] = tokens_positive

        if self.return_tokens and self.tokenizer is not None: # True
            tokenized = self.tokenizer(caption, return_tensors="pt",
                max_length=self.max_query_len,
                truncation=True)
            target["positive_map"] = create_positive_map(tokenized, target["tokens_positive"]) # (N, 256) N is num of objects. value > 0 where positive class
            # if a class name is tokenized into M tokens, value is 1/M. For example, if a class name is divided into 3 tokens, value is 1/3

        return target

def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)

    for j, tok_list in enumerate(tokens_positive): # loop over each object
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos: end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)

def create_positive_dict(tokenized, tokens_positive, labels):
    """construct a dictionary such that positive_map[i] = j, iff token i is mapped to j label"""
    positive_map = defaultdict(int)

    # Additionally, have positive_map_label_to_tokens
    positive_map_label_to_token = {}

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map_label_to_token[labels[j]] = [] 
            for i in range(beg_pos, end_pos + 1):
                positive_map[i] = labels[j]  # because the labels starts from 1
                positive_map_label_to_token[labels[j]].append(i)
            # positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map, positive_map_label_to_token  # / (positive_map.sum(-1)[:, None] + 1e-6)
