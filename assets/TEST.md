# Tutorial for Testing

We provide three one-key running scripts, [infer.sh](infer.sh), [infer_large.sh](infer_large.sh), and [infer_huge.sh](infer_huge.sh) for inference of **UNINEXT-50**, **UNINEXT-L**, and **UNINEXT-H**. Users can either run ```bash assets/xxx.sh``` or run part of commands. Some datasets may need further process to get the metrics. We list them below.

## Instance Segmentaion
Please refer to [upload tutorial](https://cocodataset.org/#upload) to prepare the result file then upload it to [COCO Server](https://codalab.lisn.upsaclay.fr/competitions/7383#).

## VIS
**VIS-2019**. Submit VIS19.zip to [VIS19 Server](https://codalab.lisn.upsaclay.fr/competitions/6064#).

**OVIS**. Submit OVIS.zip to [OVIS Server](https://codalab.lisn.upsaclay.fr/competitions/4763#).

## RVOS
**Ref-Youtube-VOS**. Submit RVOS.zip to [Ref-Youtube-VOS Server](https://codalab.lisn.upsaclay.fr/competitions/3282#).

**Ref-DAVIS**. Run the following commands and average the metrics as the final results.
```
cd external/davis2017-evaluation
python3 evaluation_method.py --task unsupervised --results_path ../../outputs/${EXP_NAME}/inference/rvos-refdavis-val-0 --davis_path ../../datasets/ref-davis/DAVIS
python3 evaluation_method.py --task unsupervised --results_path ../../outputs/${EXP_NAME}/inference/rvos-refdavis-val-1 --davis_path ../../datasets/ref-davis/DAVIS
python3 evaluation_method.py --task unsupervised --results_path ../../outputs/${EXP_NAME}/inference/rvos-refdavis-val-2 --davis_path ../../datasets/ref-davis/DAVIS
python3 evaluation_method.py --task unsupervised --results_path ../../outputs/${EXP_NAME}/inference/rvos-refdavis-val-3 --davis_path ../../datasets/ref-davis/DAVIS
```

## VOS
**Youtube-VOS**. Submit VOS.zip to [Youtube-VOS-2018 Server](https://codalab.lisn.upsaclay.fr/competitions/7685#).

**DAVIS**. Run the following commands.
```
cd external/davis2017-evaluation
python3 evaluation_method.py --task semi-supervised --results_path ../../outputs/${EXP_NAME}/inference/DAVIS --davis_path ../../datasets/DAVIS
```

## SOT
**TrackingNet**. Run ```python3 tools_bin/transform_trackingnet.py --exp_name ${EXP_NAME}``` then submit TrackingNet_submit.zip to [TrackingNet Server](https://eval.ai/web/challenges/challenge-page/1805/overview).

**LaSOT, LaSOT-ext, TNL-2K**. Copy the original result files to another directory.
```
mkdir -p UNINEXT/${EXP_NAME}
cp outputs/${EXP_NAME}/inference/LaSOT/* UNINEXT/${EXP_NAME}
cp outputs/${EXP_NAME}/inference/LaSOT_extension_subset/* UNINEXT/${EXP_NAME}
cp outputs/${EXP_NAME}/inference/TNL-2K/* UNINEXT/${EXP_NAME}
```
Run ```python3 tools_bin/analysis_results.py --exp_name ${EXP_NAME}```. Change Line 28 to corresponding datasets.

## MOTS
Run the following commands.
```
# Install extra packages
git clone https://github.com/bdd100k/bdd100k.git
cd bdd100k
python3 setup.py develop --user
pip3 uninstall -y scalabel
pip3 install --user git+https://github.com/scalabel/scalabel.git
pip3 install -U numpy
cd ..
# convert to BDD100K format (bitmask)
python3 tools_bin/to_bdd100k.py --res outputs/${EXP_NAME}/inference/instances_predictions_init_0.40_obj_0.30.pkl --task seg_track --bdd-dir . --nproc 32
# evaluate
bash tools_bin/eval_bdd_submit.sh
```