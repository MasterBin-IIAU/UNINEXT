import time
from multiprocessing import Pool
import motmetrics as mm
import numpy as np
import pandas as pd
import torch
from motmetrics.lap import linear_sum_assignment
from motmetrics.math_util import quiet_divide

METRIC_MAPS = {
    'idf1': 'IDF1',
    'mota': 'MOTA',
    'motp': 'MOTP',
    'num_false_positives': 'FP',
    'num_misses': 'FN',
    'num_switches': 'IDSw',
    'recall': 'Rcll',
    'precision': 'Prcn',
    'mostly_tracked': 'MT',
    'partially_tracked': 'PT',
    'mostly_lost': 'ML',
    'num_fragmentations': 'FM'
}


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]

def bbox_overlaps(bboxes1,
                  bboxes2,
                  mode='iou',
                  eps=1e-6,
                  use_legacy_coordinate=False):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1 (ndarray): Shape (n, 4)
        bboxes2 (ndarray): Shape (k, 4)
        mode (str): IOU (intersection over union) or IOF (intersection
            over foreground)
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.

    Returns:
        ious (ndarray): Shape (n, k)
    """

    assert mode in ['iou', 'iof']
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
        bboxes1[:, 3] - bboxes1[:, 1] + extra_length)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
        bboxes2[:, 3] - bboxes2[:, 1] + extra_length)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(
            y_end - y_start + extra_length, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious

def track2result(bboxes, labels, ids, num_classes):
    valid_inds = ids > -1
    bboxes = bboxes[valid_inds]
    labels = labels[valid_inds]
    ids = ids[valid_inds]

    if bboxes.shape[0] == 0:
        return [np.zeros((0, 6), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
            labels = labels.cpu().numpy()
            ids = ids.cpu().numpy()
        return [
            np.concatenate((ids[labels == i, None], bboxes[labels == i, :]),
                           axis=1) for i in range(num_classes)
        ]

def bbox_distances(bboxes1, bboxes2, iou_thr=0.5):
    """Calculate the IoU distances of two sets of boxes."""
    ious = bbox_overlaps(bboxes1, bboxes2, mode='iou')
    distances = 1 - ious
    distances = np.where(distances > iou_thr, np.nan, distances)
    return distances


def acc_single_video(results,
                     gts,
                     iou_thr=0.5,
                     ignore_iof_thr=0.5,
                     ignore_by_classes=False):
    """Accumulate results in a single video."""
    num_classes = len(results[0])
    accumulators = [
        mm.MOTAccumulator(auto_id=True) for i in range(num_classes)
    ]
    for result, gt in zip(results, gts):
        if ignore_by_classes:
            gt_ignore = bbox2result(gt['bboxes_ignore'], gt['labels_ignore'],
                                    num_classes)
        else:
            gt_ignore = [gt['bboxes_ignore'] for i in range(num_classes)]
        gt = track2result(gt['bboxes'], gt['labels'], gt['instance_ids'],
                          num_classes)
        for i in range(num_classes):
            gt_ids, gt_bboxes = gt[i][:, 0].astype(np.int), gt[i][:, 1:]
            pred_ids, pred_bboxes = result[i][:, 0].astype(
                np.int), result[i][:, 1:-1]
            dist = bbox_distances(gt_bboxes, pred_bboxes, iou_thr)
            if gt_ignore[i].shape[0] > 0:
                # 1. assign gt and preds
                fps = np.ones(pred_bboxes.shape[0]).astype(np.bool)
                row, col = linear_sum_assignment(dist)
                for m, n in zip(row, col):
                    if not np.isfinite(dist[m, n]):
                        continue
                    fps[n] = False
                # 2. ignore by iof
                iofs = bbox_overlaps(pred_bboxes, gt_ignore[i], mode='iof')
                ignores = (iofs > ignore_iof_thr).any(axis=1)
                # 3. filter preds
                valid_inds = ~(fps & ignores)
                pred_ids = pred_ids[valid_inds]
                dist = dist[:, valid_inds]
            if dist.shape != (0, 0):
                accumulators[i].update(gt_ids, pred_ids, dist)
    return accumulators


def aggregate_accs(accumulators, classes):
    """Aggregate results from each class."""
    # accs for each class
    items = list(classes)
    names, accs = [[] for c in classes], [[] for c in classes]
    for video_ind, _accs in enumerate(accumulators):
        for cls_ind, acc in enumerate(_accs):
            if len(acc._events['Type']) == 0:
                continue
            name = f'{classes[cls_ind]}_{video_ind}'
            names[cls_ind].append(name)
            accs[cls_ind].append(acc)

    # overall
    items.append('OVERALL')
    names.append([n for name in names for n in name])
    accs.append([a for acc in accs for a in acc])

    return names, accs, items


def eval_single_class(names, accs):
    """Evaluate CLEAR MOT results for each class."""
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs, names=names, metrics=METRIC_MAPS.keys(), generate_overall=True)
    results = [v['OVERALL'] for k, v in summary.to_dict().items()]
    motp_ind = list(METRIC_MAPS).index('motp')
    if np.isnan(results[motp_ind]):
        num_dets = mh.compute_many(
            accs,
            names=names,
            metrics=['num_detections'],
            generate_overall=True)
        sum_motp = (summary['motp'] * num_dets['num_detections']).sum()
        motp = quiet_divide(sum_motp, num_dets['num_detections']['OVERALL'])
        results[motp_ind] = float(1 - motp)
    else:
        results[motp_ind] = 1 - results[motp_ind]
    return results


def eval_mot(results,
             annotations,
             logger=None,
             classes=None,
             iou_thr=0.5,
             ignore_iof_thr=0.5,
             ignore_by_classes=False,
             nproc=4):
    """Evaluation CLEAR MOT metrics.

    Args:
        results (list[list[list[ndarray]]]): The first list indicates videos,
            The second list indicates images. The third list indicates
            categories. The ndarray indicates the tracking results.
        annotations (list[list[dict]]): The first list indicates videos,
            The second list indicates images. The third list indicates
            the annotations of each video. Keys of annotations are

            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `instance_ids`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        logger (logging.Logger | str | None, optional): The way to print the
            evaluation results. Defaults to None.
        classes (list, optional): Classes in the dataset. Defaults to None.
        iou_thr (float, optional): IoU threshold for evaluation.
            Defaults to 0.5.
        ignore_iof_thr (float, optional): Iof threshold to ignore results.
            Defaults to 0.5.
        ignore_by_classes (bool, optional): Whether ignore the results by
            classes or not. Defaults to False.
        nproc (int, optional): Number of the processes. Defaults to 4.

    Returns:
        dict[str, float]: Evaluation results.
    """
    print('---CLEAR MOT Evaluation---')
    t = time.time()
    gts = annotations.copy()
    if classes is None:
        classes = [i + 1 for i in range(len(results[0]))]
    assert len(results) == len(gts)
    metrics = METRIC_MAPS.keys()

    print('Accumulating...')

    pool = Pool(nproc)
    accs = pool.starmap(
        acc_single_video,
        zip(results, gts, [iou_thr for _ in range(len(gts))],
            [ignore_iof_thr for _ in range(len(gts))],
            [ignore_by_classes for _ in range(len(gts))]))
    names, accs, items = aggregate_accs(accs, classes)
    print('Evaluating...')
    eval_results = pd.DataFrame(columns=metrics)
    summaries = pool.starmap(eval_single_class, zip(names, accs))
    pool.close()

    # category and overall results
    for i, item in enumerate(items):
        eval_results.loc[item] = summaries[i]

    dtypes = {m: type(d) for m, d in zip(metrics, summaries[0])}
    # average results
    avg_results = []
    for i, m in enumerate(metrics):
        v = np.array([s[i] for s in summaries[:len(classes)]])
        v = np.nan_to_num(v, nan=0)
        if dtypes[m] == int:
            avg_results.append(int(v.sum()))
        elif dtypes[m] == float:
            avg_results.append(float(v.mean()))
        else:
            raise TypeError()
    eval_results.loc['AVERAGE'] = avg_results
    eval_results = eval_results.astype(dtypes)

    print('Rendering...')
    strsummary = mm.io.render_summary(
        eval_results,
        formatters=mm.metrics.create().formatters,
        namemap=METRIC_MAPS)
    print('\n' + strsummary)
    with open("eval_log.txt", "a") as f:
        print('\n' + strsummary + '\n', file=f)
    print(f'Evaluation finishes with {(time.time() - t):.2f} s.')

    eval_results = eval_results.to_dict()
    out = {METRIC_MAPS[k]: v['OVERALL'] for k, v in eval_results.items()}
    for k, v in out.items():
        out[k] = float(f'{(v):.3f}') if isinstance(v, float) else int(f'{v}')
    for m in ['OVERALL', 'AVERAGE']:
        out[f'track_{m}_copypaste'] = ''
        for k in METRIC_MAPS.keys():
            v = eval_results[k][m]
            v = f'{(v):.3f} ' if isinstance(v, float) else f'{v} '
            out[f'track_{m}_copypaste'] += v

    return out
