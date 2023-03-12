#!/usr/bin/env bash

python3 evaluation_method.py --task unsupervised --results_path ../../video_joint_r50/inference/rvos-refdavis-val-0 --davis_path ../../datasets/ref-davis/DAVIS
python3 evaluation_method.py --task unsupervised --results_path ../../video_joint_r50/inference/rvos-refdavis-val-1 --davis_path ../../datasets/ref-davis/DAVIS
python3 evaluation_method.py --task unsupervised --results_path ../../video_joint_r50/inference/rvos-refdavis-val-2 --davis_path ../../datasets/ref-davis/DAVIS
python3 evaluation_method.py --task unsupervised --results_path ../../video_joint_r50/inference/rvos-refdavis-val-3 --davis_path ../../datasets/ref-davis/DAVIS