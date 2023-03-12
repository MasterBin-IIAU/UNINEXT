# Install
## Requirements
We test the codes in the following environments, other versions may also be compatible but Pytorch vision should be >= 1.7

- CUDA 11.3
- Python 3.7
- Pytorch 1.10.0
- Torchvison 0.11.1

## Install environment for UNINEXT

```
pip3 install -e . --user
pip3 install --user shapely==1.7.1
pip3 install --user git+https://github.com/XD7479/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
pip3 install --user git+https://github.com/lvis-dataset/lvis-api.git
pip3 install --user jpeg4py visdom easydict scikit-image
pip3 install --user transformers tikzplotlib motmetrics

# compile Deformable DETR
cd projects/UNINEXT/uninext/models/deformable_detr/ops
bash make.sh
```