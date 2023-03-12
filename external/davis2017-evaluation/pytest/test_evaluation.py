import os
import sys
import numpy as np
import pandas
from time import time
from collections import defaultdict

from davis2017.evaluation import DAVISEvaluation
from davis2017 import utils
from davis2017.metrics import db_eval_boundary, db_eval_iou


davis_root = 'input_dir/ref'
methods_root = 'examples'


def test_task(task, gt_set, res_path, J_target=None, F_target=None, metric=('J', 'F')):
    dataset_eval = DAVISEvaluation(davis_root=davis_root, gt_set=gt_set, task=task, codalab=True)
    metrics_res = dataset_eval.evaluate(res_path, debug=False, metric=metric)

    num_seq = len(list(dataset_eval.dataset.get_sequences()))
    J = metrics_res['J'] if 'J' in metric else {'M': np.zeros(num_seq), 'R': np.zeros(num_seq), 'D': np.zeros(num_seq)}
    F = metrics_res['F'] if 'F' in metric else {'M': np.zeros(num_seq), 'R': np.zeros(num_seq), 'D': np.zeros(num_seq)}

    if gt_set == "val" or gt_set == "train" or gt_set == "test-dev":
        sys.stdout.write("----------------Global results in CSV---------------\n")
        g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
        final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2. if 'J' in metric and 'F' in metric else 0
        g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]), np.mean(F["D"])])
        table_g = pandas.DataFrame(data=np.reshape(g_res, [1, len(g_res)]), columns=g_measures)
        table_g.to_csv(sys.stdout, index=False, float_format="%0.3f")
    if J_target is not None:
        assert check_results_similarity(J, J_target), f'J {print_error(J, J_target)}'
    if F_target is not None:
        assert check_results_similarity(F, F_target), f'F {print_error(F, F_target)}'
    return J, F


def check_results_similarity(target, result):
    return np.isclose(np.mean(target['M']) - result[0], 0, atol=0.001) & \
           np.isclose(np.mean(target['R']) - result[1], 0, atol=0.001) & \
           np.isclose(np.mean(target['D']) - result[2], 0, atol=0.001)


def print_error(target, result):
    return f'M:{np.mean(target["M"])} = {result[0]}\t' + \
           f'R:{np.mean(target["R"])} = {result[1]}\t' + \
           f'D:{np.mean(target["D"])} = {result[2]}'


def test_semisupervised_premvos():
    method_path = os.path.join(methods_root, 'premvos')
    print('Evaluating PREMVOS val')
    J_val = [0.739, 0.831, 0.162]
    F_val = [0.818, 0.889, 0.195]
    test_task('semi-supervised', 'val', method_path, J_val, F_val)
    print('Evaluating PREMVOS test-dev')
    J_test_dev = [0.675, 0.768, 0.217]
    F_test_dev = [0.758, 0.843, 0.206]
    test_task('semi-supervised', 'test-dev', method_path, J_test_dev, F_test_dev)
    print('\n')


def test_semisupervised_onavos():
    method_path = os.path.join(methods_root, 'onavos')
    print('Evaluating OnAVOS val')
    J_val = [0.616, 0.674, 0.279]
    F_val = [0.691, 0.754, 0.266]
    test_task('semi-supervised', 'val', method_path, J_val, F_val)
    print('Evaluating OnAVOS test-dev')
    J_test_dev = [0.499, 0.543, 0.230]
    F_test_dev = [0.557, 0.603, 0.234]
    test_task('semi-supervised', 'test-dev', method_path, J_test_dev, F_test_dev)
    print('\n')


def test_semisupervised_osvos():
    method_path = os.path.join(methods_root, 'osvos')
    print('Evaluating OSVOS val')
    J_val = [0.566, 0.638, 0.261]
    F_val = [0.639, 0.738, 0.270]
    test_task('semi-supervised', 'val', method_path, J_val, F_val)
    print('Evaluating OSVOS test-dev')
    J_test_dev = [0.470, 0.521, 0.192]
    F_test_dev = [0.548, 0.597, 0.198]
    test_task('semi-supervised', 'test-dev', method_path, J_test_dev, F_test_dev)
    print('\n')


def test_unsupervised_flip_gt():
    print('Evaluating Unsupervised Permute GT')
    method_path = os.path.join(methods_root, 'swap_gt')
    if not os.path.isdir(method_path):
        utils.generate_random_permutation_gt_obj_proposals(davis_root, 'val', method_path)
        # utils.generate_random_permutation_gt_obj_proposals('test-dev', method_path)
    J_val = [1, 1, 0]
    F_val= [1, 1, 0]
    test_task('unsupervised', 'val', method_path, J_val, F_val)
    # test_task('unsupervised', 'test-dev', method_path, J_val, F_val)


def test_unsupervised_rvos():
    print('Evaluating RVOS')
    method_path = os.path.join(methods_root, 'rvos')
    test_task('unsupervised', 'val', method_path)
    # test_task('unsupervised', 'test-dev', method_path)


def test_unsupervsied_multiple_proposals(num_proposals=20, metric=('J', 'F')):
    print('Evaluating Multiple Proposals')
    method_path = os.path.join(methods_root,  f'generated_proposals_{num_proposals}')
    utils.generate_obj_proposals(davis_root, 'val', num_proposals, method_path)
    # utils.generate_obj_proposals('test-dev', num_proposals, method_path)
    test_task('unsupervised', 'val', method_path, metric=metric)
    # test_task('unsupervised', 'test-dev', method_path, metric=metric)


def test_void_masks():
    gt = np.zeros((2, 200, 200))
    mask = np.zeros((2, 200, 200))
    void = np.zeros((2, 200, 200))

    gt[:, 100:150, 100:150] = 1
    void[:, 50:100, 100:150] = 1
    mask[:, 50:150, 100:150] = 1

    assert np.mean(db_eval_iou(gt, mask, void)) == 1
    assert np.mean(db_eval_boundary(gt, mask, void)) == 1


def benchmark_number_proposals():
    number_proposals = [10, 15, 20, 30]
    timing_results = defaultdict(dict)
    for n in number_proposals:
        time_start = time()
        test_unsupervsied_multiple_proposals(n, 'J')
        timing_results['J'][n] = time() - time_start

    for n in number_proposals:
        time_start = time()
        test_unsupervsied_multiple_proposals(n)
        timing_results['J_F'][n] = time() - time_start

    print(f'Using J {timing_results["J"]}')
    print(f'Using J&F {timing_results["J_F"]}')

    # Using J {10: 156.45335865020752, 15: 217.91797709465027, 20: 282.0747673511505, 30: 427.6770250797272}
    # Using J & F {10: 574.3529748916626, 15: 849.7542386054993, 20: 1123.4619634151459, 30: 1663.6704666614532}
    # Codalab
    # Using J & F {10: 971.196366071701, 15: 1473.9757001399994, 20: 1918.787559747696, 30: 3007.116141319275}


if __name__ == '__main__':
    # Test void masks
    test_void_masks()

    # Test semi-supervised methods
    test_semisupervised_premvos()
    test_semisupervised_onavos()
    test_semisupervised_osvos()

    # Test unsupervised methods
    test_unsupervised_flip_gt()
    # test_unsupervised_rvos()
    test_unsupervsied_multiple_proposals()
