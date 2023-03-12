import os
import os.path as osp

from scalabel.label.io import save
from scalabel.label.transforms import bbox_to_box2d
from scalabel.label.typing import Frame, Label
from tqdm import tqdm

# from ..evaluation import xyxy2xywh
from .utils import mask_merge_parallel

def xyxy2xywh(bbox):
    return [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    ]

CATEGORIES = [
    '', 'pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle', 'traffic light', 'traffic sign'
]


def det_to_bdd100k(dataset, results, out_base, nproc):
    bdd100k = []
    ann_id = 0
    print(f'\nStart converting to BDD100K detection format')
    if 'bbox_results' in results:
        results = results['bbox_results']
    for idx, bboxes_list in tqdm(enumerate(results)):
        img_name = dataset.data_infos[idx]['file_name']
        frame = Frame(name=img_name, labels=[])

        for cls_, bboxes in enumerate(bboxes_list):
            for bbox in bboxes:
                ann_id += 1
                label = Label(
                    id=ann_id,
                    score=bbox[-1],
                    box2d=bbox_to_box2d(xyxy2xywh(bbox)),
                    category=CATEGORIES[cls_ + 1])
                frame.labels.append(label)
        bdd100k.append(frame)

    print(f'\nWriting the converted json')
    out_path = osp.join(out_base, "det.json")
    save(out_path, bdd100k)


def ins_seg_to_bdd100k(dataset, results, out_base, nproc=4):
    bdd100k = []
    bitmask_base = osp.join(out_base, "ins_seg")
    if not osp.exists(bitmask_base):
        os.makedirs(bitmask_base)

    if 'bbox_results' in results and 'segm_results' in results:
        results = [[bbox, segm] for bbox, segm in zip(results['bbox_results'],
                                                      results['segm_results'])]

    track_dicts = []
    img_names = [
        dataset.data_infos[idx]['file_name'] for idx in range(len(results))
    ]

    print(f'\nStart converting to BDD100K instance segmentation format')
    ann_id = 0
    for idx, [bboxes_list, segms_list] in enumerate(results):
        index = 0
        frame = Frame(name=img_names[idx], labels=[])
        track_dict = {}
        for cls_, (bboxes, segms) in enumerate(zip(bboxes_list, segms_list)):
            for bbox, segm in zip(bboxes, segms):
                ann_id += 1
                index += 1
                label = Label(id=str(ann_id), index=index, score=bbox[-1])
                frame.labels.append(label)
                instance = {'bbox': bbox, 'segm': segm, 'label': cls_}
                track_dict[index] = instance

        bdd100k.append(frame)
        track_dicts.append(track_dict)

    print(f'\nWriting the converted json')
    out_path = osp.join(out_base, 'ins_seg.json')
    save(out_path, bdd100k)

    mask_merge_parallel(track_dicts, img_names, bitmask_base, nproc)


def box_track_to_bdd100k(dataset, results, out_base, nproc):
    bdd100k = []
    track_base = osp.join(out_base, "box_track")
    if not osp.exists(track_base):
        os.makedirs(track_base)

    print(f'\nStart converting to BDD100K box tracking format')
    for idx, track_dict in tqdm(enumerate(results['track_results'])):
        img_name = dataset.data_infos[idx]['file_name']
        frame_index = dataset.data_infos[idx]['frame_id']
        vid_name = os.path.split(img_name)[0]
        frame = Frame(
            name=img_name,
            video_name=vid_name,
            frame_index=frame_index,
            labels=[])

        for id_, instance in track_dict.items():
            bbox = instance['bbox']
            cls_ = instance['label']
            label = Label(
                id=id_,
                score=bbox[-1],
                box2d=bbox_to_box2d(xyxy2xywh(bbox)),
                category=CATEGORIES[cls_ + 1])
            frame.labels.append(label)
        bdd100k.append(frame)

    print(f'\nWriting the converted json')
    out_path = osp.join(out_base, "box_track.json")
    save(out_path, bdd100k)


def seg_track_to_bdd100k(dataset, results, out_base, nproc=4):
    bitmask_base = osp.join(out_base, "seg_track")
    if not osp.exists(bitmask_base):
        os.makedirs(bitmask_base)

    print(f'\nStart converting to BDD100K seg tracking format')
    img_names = [
        dataset.data_infos[idx]['file_name']
        for idx in range(len(results['track_result']))
    ]
    mask_merge_parallel(results['track_result'], img_names, bitmask_base,
                        nproc)


def preds2bdd100k(dataset, results, tasks, out_base, *args, **kwargs):
    metric2func = dict(
        det=det_to_bdd100k,
        ins_seg=ins_seg_to_bdd100k,
        box_track=box_track_to_bdd100k,
        seg_track=seg_track_to_bdd100k)

    for task in tasks:
        metric2func[task](dataset, results, out_base, *args, **kwargs)
