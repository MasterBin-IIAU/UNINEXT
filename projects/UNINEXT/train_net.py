#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
UNINEXT Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import sys
import itertools
import time
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators, LVISEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping

from detectron2.projects.uninext import build_detection_train_loader, build_detection_test_loader
from detectron2.projects.uninext import add_uninext_config
from detectron2.projects.uninext.data import (
    get_detection_dataset_dicts, DetrDatasetMapperUni, YTVISDatasetMapper, YTVISEvaluator, SOTDatasetMapper, UniVidDatasetMapper
)
import logging
from collections import OrderedDict

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass
import logging
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer, TrainerBase
from torch.nn.parallel import DistributedDataParallel
from detectron2.utils.env import TORCH_VERSION
# Unification
from detectron2.projects.uninext.data import build_custom_train_loader



class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        assert cfg.UNI == True
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        evaluator_list = []
        if dataset_name.startswith("seginw"):
            evaluator_type = "coco"
        else:
            evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "lvis":
            evaluator_list.append(LVISEvaluator(dataset_name, cfg, True, output_folder))
        elif evaluator_type == "coco":
            force_tasks = {"bbox"} if "objects365" in cfg.DATASETS.TRAIN[0] else None
            if "refcoco" in dataset_name:
                evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder, force_tasks=force_tasks, refcoco=True))
            elif "coco" in dataset_name or "objects365" in dataset_name or "seginw" in dataset_name:
                evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder, force_tasks=force_tasks, refcoco=False))
        elif evaluator_type == "ytvis":
            evaluator_list.append(YTVISEvaluator(dataset_name, cfg, True, output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        assert cfg.UNI == True
        if cfg.DATALOADER.SAMPLER_TRAIN == "MultiDatasetSampler":
            # multiple datasets (for example, detection & grounding)
            if cfg.UNI_VID:
                mapper = UniVidDatasetMapper(cfg, is_train=True)
            else:
                # early version: different mappers for different tasks
                if cfg.DATASETS.TRAIN[0].startswith("ytvis") or cfg.DATASETS.TRAIN[0].startswith("rvos") or \
                    (cfg.DATASETS.TRAIN[0].startswith("bdd") and cfg.DATASETS.TRAIN[0] != "bdd_det_trk_mixed_train"):
                    # joint training of multiple VIS/MOT datasets
                    mapper = YTVISDatasetMapper(cfg, is_train=True)
                elif cfg.DATASETS.TRAIN[0].startswith("sot"):
                    # joint training of multiple SOT datasets
                    mapper = SOTDatasetMapper(cfg, is_train=True)
                else:
                    # joint training of detection and grounding datasets
                    mapper = DetrDatasetMapperUni(cfg, is_train=True)
            data_loader = build_custom_train_loader(cfg, mapper=mapper)
            return data_loader
        else:
            assert len(cfg.DATASETS.TRAIN) == 1
            dataset_name = cfg.DATASETS.TRAIN[0]
            dataset_dict = get_detection_dataset_dicts(
                dataset_name,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            )
            if dataset_name.startswith('ytvis') or dataset_name.startswith("rvos") or \
                (dataset_name.startswith('bdd') and dataset_name != "bdd_det_trk_mixed_train"):
                mapper = YTVISDatasetMapper(cfg, is_train=True)
            elif dataset_name.startswith("sot"):
                mapper = SOTDatasetMapper(cfg, is_train=True)
            else:
                # detection only or grounding only
                mapper = DetrDatasetMapperUni(cfg, is_train=True)

            return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dict)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if len(cfg.DATASETS.TEST) > 1:
            assert cfg.UNI == True
        else:
            dataset_name = cfg.DATASETS.TEST[0]
        if dataset_name.startswith("coco") or dataset_name.startswith("refcoco") or dataset_name.startswith("objects365_v2") or dataset_name.startswith("seginw"):
            mapper = DetrDatasetMapperUni(cfg, is_train=False)
        elif dataset_name.startswith('ytvis') or dataset_name.startswith('bdd') or dataset_name.startswith("refytb") or dataset_name.startswith("rvos"):
            mapper = YTVISDatasetMapper(cfg, is_train=False)
        elif dataset_name.startswith("sot"):
            mapper = SOTDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            elif "sampling_offsets" in key or "reference_points" in key:
                lr = lr * cfg.SOLVER.LINEAR_PROJ_MULTIPLIER
            elif "text_encoder" in key or "lang_layers" in key:
                lr = cfg.SOLVER.LANG_LR
            elif "vl_layers" in key:
                lr = cfg.SOLVER.VL_LR
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_uninext_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()
    return 


def new_argument_parser():
    parser = default_argument_parser()
    parser.add_argument("--uni", type=int, default=1, help="whether to use a unified model for multiple tasks")
    return parser


if __name__ == "__main__":
    args = new_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
