# Data Preparation

## Pretrained Weights
Language Model (BERT-base)
```
mkdir -p projects/UNINEXT/bert-base-uncased
cd projects/UNINEXT/bert-base-uncased
wget -c https://huggingface.co/bert-base-uncased/resolve/main/config.json
wget -c https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt
wget -c https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
cd ../../..
```

Visual Backbones
```
mkdir -p weights
cd weights
# R50
wget -c https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl
# ConvNeXt-Large
wget -c https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth
# Convert ConvNeXt-Large
cd ..
python3 conversion/convert_convnext.py --source_model weights/convnext_large_22k_1k_384.pth --output_model weights/convnext_large_22k_1k_384_new.pth
python3 projects/UNINEXT/convert_pth2pkl.py weights/convnext_large_22k_1k_384_new.pth weights/convnext_large_22k_1k_384_new.pkl
# ViT-Huge
wget -c https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MAE/mae_pretrain_vit_huge_p14to16.pth
```

Other pretrained models can be found in [MODEL_ZOO.md](MODEL_ZOO.md)

## Data
For users who are only interested in part of tasks, there is no need of downloading all datasets. The following lines list the datasets needed for different tasks. Datasets in the brackets are only used during the inference.

- **Pretrain**: Objects365
- **Object detection & instance segmentation**: COCO2017
- **REC & RES**: RefCOCO, RefCOCOg, RefCOCO+
- **SOT**: COCO, LaSOT, GOT-10K, TrackingNet, (LaSOT-ext, TNL-2k)
- **VOS**: Youtube-VOS 2018, COCO, (DAVIS)
- **MOT & MOTS**: COCO, BDD100K
- **VIS**: COCO, Youtube-VIS 2019, OVIS
- **R-VOS**: RefCOCO, RefCOCOg, RefCOCO+, Ref-Youtube-VOS, (Ref-DAVIS)



### Pretraining
Pretraining on Objects365 requires many training resources. For UNINEXT-50, Objects365 pretraining needs 3~4 days on 32 A100 GPUs. Thus we suggest users directly loading provided weights instead of re-running this step. If users still want to use this dataset, we provide a script for automatically downloading images of Objects365 V2.
```
python3 conversion/download_obj365_v2.py
```
Following DINO, we select the first 5,000 out of 80,000 validation images as our
validation set and add the others to training. We put the processed json files on [OneDrive](https://maildluteducn-my.sharepoint.com/:u:/g/personal/yan_bin_mail_dlut_edu_cn/ETscaOUVpeVBmjXUKYHfYvMB5wSxfb9A9Ag4KKe5lL3Xwg?e=3e41N8), which can be directly downloaded.
We expect that the data is organized as below.
```
${UNINEXT_ROOT}
    -- datasets
        -- Objects365
            -- annotations
                -- zhiyuan_objv2_train_new.json
                -- zhiyuan_objv2_val_new.json
            -- images
```

### Object Detection & Instance Segmentation
Please download [COCO](https://cocodataset.org/#home) from the offical website. We use [train2017.zip](http://images.cocodataset.org/zips/train2017.zip), [val2017.zip](http://images.cocodataset.org/zips/val2017.zip), [test2017.zip](http://images.cocodataset.org/zips/test2017.zip) & [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), [image_info_test2017.zip](http://images.cocodataset.org/annotations/image_info_test2017.zip). We expect that the data is organized as below.
```
${UNINEXT_ROOT}
    -- datasets
        -- coco
            -- annotations
            -- train2017
            -- val2017
            -- test2017
```

### REC & RES
Please download processed json files by [SeqTR](https://github.com/sean-zhuh/SeqTR) from [Google Drive](https://drive.google.com/drive/folders/1IXnSieVr5CHF2pVJpj0DlwC6R3SbfolU). We need three folders: refcoco-unc, refcocog-umd, and refcocoplus-unc. These folders should be organized as below.
```
${UNINEXT_ROOT}
    -- datasets
        -- annotations
            -- refcoco-unc
            -- refcocog-umd
            -- refcocoplus-unc
```
Then run ```python3 conversion/convert_mix_ref.py``` to convert the original jsons to COCO format and merge them into one dataset ```refcoco-mixed```. Besides, please download images of [COCO2014 train](http://images.cocodataset.org/zips/train2014.zip) and put ```train2014``` folder under ```datasets/coco```. 

### SOT & VOS
To train UNINEXT for SOT&VOS, please download [LaSOT](http://vision.cs.stonybrook.edu/~lasot/download.html), [GOT-10K](http://got-10k.aitestunion.com/downloads), [TrackingNet](https://tracking-net.org/), and [Youtube-VOS 2018](https://youtube-vos.org/dataset/). Since TrackingNet is very large and hard to download, we only use the first 4 splits (TRAIN_0.zip, TRAIN_1.zip, TRAIN_2.zip, TRAIN_3.zip) rather than the complete 12 splits for the training set. The original TrackingNet zips (put under `datasets`) can be unzipped by ```python3 conversion/unzip_trackingnet.py```.

To infer on the test sets of SOT&VOS, please download [LaSOT-ext](http://vision.cs.stonybrook.edu/~lasot/download.html), [TNL-2K](https://sites.google.com/view/langtrackbenchmark/), [DAVIS-17](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip). We expect that the data is organized as below.
```
${UNINEXT_ROOT}
    -- datasets
        -- GOT10K
            -- train
                -- GOT-10k_Train_000001
                -- ...
        -- LaSOT
            -- airplane
            -- basketball
            -- ...
        -- TrackingNet
            -- TEST
            -- TRAIN_0
            -- TRAIN_1
            -- TRAIN_2
            -- TRAIN_3
        -- ytbvos18
            -- train
            -- val
        -- LaSOT_extension_subset
            -- atv
            -- badminton
            -- ...
        -- TNL-2K
            -- advSamp_Baseball_game_002-Done
            -- advSamp_Baseball_video_01-Done
            -- ...
        -- DAVIS
            -- Annotations
            -- ImageSets
            -- JPEGImages
```
After downloading the original data, please run the following commands to get json files of [Video COCO format](https://codalab.lisn.upsaclay.fr/competitions/6064#participate).
```
python3 conversion/convert_got10k_to_cocovid.py
python3 conversion/convert_lasot_to_cocovid.py
python3 conversion/convert_trackingnet_to_cocovid.py
python3 conversion/convert_ytbvos2cocovid.py
python3 conversion/convert_coco_to_sot.py
# for inference only
python3 conversion/convert_tnl2k_to_cocovid.py
python3 conversion/convert_lasot_ext_to_cocovid.py
python3 conversion/convert_ytbvos2cocovid_val.py
python3 conversion/convert_davis2cocovid.py
```

### MOT & MOTS
We need to download the `detection` set, `tracking` set, `instance seg` set and `tracking & seg` set for training and validation.
For more details about the dataset, please refer to the [offial documentation](https://doc.bdd100k.com/download.html).

We provide the following commands to download and process BDD100K datasets.
```
python3 conversion/download_bdd.py
bash conversion/prepare_bdd.sh
bash conversion/convert_bdd.sh
```
We expect that the data is organized as below
```
${UNINEXT_ROOT}
    -- datasets
        -- bdd
            -- images
                -- 10k
                -- 100k
                -- seg_track_20
                -- track
            -- labels
                -- box_track_20
                -- det_20
                -- ins_seg
                -- seg_track_20
```

### VIS
Download [YouTube-VIS 2019](https://codalab.lisn.upsaclay.fr/competitions/6064#participate-get_data), [OVIS](https://codalab.lisn.upsaclay.fr/competitions/4763#participate) and COCO 2017 datasets. Then, convert the original coco json to [Video COCO format](https://codalab.lisn.upsaclay.fr/competitions/6064#participate) format by 
```
python3 conversion/convert_coco_to_video.py --src_json datasets/coco/annotations/instances_train2017.json --des_json datasets/coco/annotations/instances_train2017_video.json
```
We expect the directory structure to be the following:
```
${UNINEXT_ROOT}
    -- datasets
        -- ytvis_2019
            -- train
            -- val
            -- annotations
                -- instances_train_sub.json
                -- instances_val_sub.json
        -- ovis
            -- train
            -- val
            -- annotations_train.json
            -- annotations_valid.json
        -- coco
            -- annotations
                -- instances_train2017_video.json
                ...
            -- train
            -- val
```

### R-VOS
Follow [REC&RES](DATA.md#rec--res) to prepare RefCOCO/g/+ datasets. Then convert mixed RefCOCO/g/+ to [Video COCO format](https://codalab.lisn.upsaclay.fr/competitions/6064#participate) format by 

```
python3 conversion/convert_refcoco_to_video.py
```
Download [Ref-Youtube-VOS](https://codalab.lisn.upsaclay.fr/competitions/3282#participate-submit_results) and put them under ```datasets/ref-youtube-vos```.
Then run the following commands
```
cd datasets/ref-youtube-vos
unzip -qq train.zip
unzip -qq valid.zip
unzip meta_expressions_test.zip
rm train.zip valid.zip test.zip meta_expressions_test.zip
cd ../..
python3 conversion/convert_refytb2cocovid.py
python3 conversion/convert_refytvos2ytvis_val.py
```
Download ```DAVIS-2017-Unsupervised-trainval-480p.zip``` and ```DAVIS-2017_semantics-480p.zip``` from [DAVIS website](https://davischallenge.org/davis2017/code.html). Download the text annotations ```davis_text_annotations.zip``` from the [website](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/video-segmentation/video-object-segmentation-with-language-referring-expressions). Put the zip files under ```datasets/ref-davis```. Run the following commands.
```
unzip -qq datasets/ref-davis/davis_text_annotations.zip -d datasets/ref-davis
unzip -qq datasets/ref-davis/DAVIS-2017_semantics-480p.zip -d datasets/ref-davis
unzip -qq datasets/ref-davis/DAVIS-2017-Unsupervised-trainval-480p.zip -d datasets/ref-davis
rm datasets/ref-davis/*.zip
cp -r datasets/ref-davis/valid/Annotations/* datasets/ref-davis/DAVIS/Annotations_unsupervised/480p/
cp -r datasets/ref-davis/valid/JPEGImages/* datasets/ref-davis/DAVIS/JPEGImages/480p/
cp -r datasets/ref-davis/train/Annotations/* datasets/ref-davis/DAVIS/Annotations_unsupervised/480p/
cp -r datasets/ref-davis/train/JPEGImages/* datasets/ref-davis/DAVIS/JPEGImages/480p/
python3 conversion/convert_refdavis2refytvos.py
python3 conversion/convert_refdavis2ytvis_val.py
```
The data should be organized as below.
```
${UNINEXT_ROOT}
    -- datasets
        -- ref-youtube-vos
            -- meta_expressions
            -- train
            -- valid
            -- train.json
            -- valid.json
        -- ref-davis
            -- DAVIS
            -- davis_text_annotations
            -- meta_expressions
            -- train
            -- valid
            -- valid_0.json
            -- valid_1.json
            -- valid_2.json
            -- valid_3.json
```
