# Copyright (c) 2022 ByteDance. All Rights Reserved.
from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings

""" Parameters for Unicorn-VOS """

def parameters(exp_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    
    # search region setting
    params.exp_name = exp_name
    # Network checkpoint path
    params.checkpoint = os.path.join(prj_dir, "Unicorn_outputs/%s/latest_ckpt.pth"%exp_name)

    return params