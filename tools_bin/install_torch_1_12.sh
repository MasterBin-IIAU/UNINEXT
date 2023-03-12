#!/usr/bin/env bash


# upgrade pytorch
sudo pip3 uninstall -y torch torchvision
pip3 uninstall -y torch torchvision
pip3 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# re-make Deformable Attention
cd projects/UNINEXT/uninext/models/deformable_detr/ops
rm -rf build dist MultiScaleDeformableAttention.egg-info
bash make.sh
cd ../../../..