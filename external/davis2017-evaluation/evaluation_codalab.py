#!/usr/bin/env python
import sys
import os.path
from time import time

import numpy as np
import pandas
from davis2017.evaluation import DAVISEvaluation

task = 'semi-supervised'
gt_set = 'test-dev'

time_start = time()
# as per the metadata file, input and output directories are the arguments
if len(sys.argv) < 3:
    input_dir = "input_dir"
    output_dir = "output_dir"
    debug = True
else:
    [_, input_dir, output_dir] = sys.argv
    debug = False

# unzipped submission data is always in the 'res' subdirectory
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
submission_path = os.path.join(input_dir, 'res')
if not os.path.exists(submission_path):
    sys.exit('Could not find submission file {0}'.format(submission_path))

# unzipped reference data is always in the 'ref' subdirectory
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
gt_path = os.path.join(input_dir, 'ref')
if not os.path.exists(gt_path):
    sys.exit('Could not find GT file {0}'.format(gt_path))


# Create dataset
dataset_eval = DAVISEvaluation(davis_root=gt_path, gt_set=gt_set, task=task, codalab=True)

# Check directory structure
res_subfolders = os.listdir(submission_path)
if len(res_subfolders) == 1:
    sys.stdout.write(
        "Incorrect folder structure, the folders of the sequences have to be placed directly inside the "
        "zip.\nInside every folder of the sequences there must be an indexed PNG file for every frame.\n"
        "The indexes have to match with the initial frame.\n")
    sys.exit()

# Check that all sequences are there
missing = False
for seq in dataset_eval.dataset.get_sequences():
    if seq not in res_subfolders:
        sys.stdout.write(seq + " sequence is missing.\n")
        missing = True
if missing:
    sys.stdout.write(
        "Verify also the folder structure, the folders of the sequences have to be placed directly inside "
        "the zip.\nInside every folder of the sequences there must be an indexed PNG file for every frame.\n"
        "The indexes have to match with the initial frame.\n")
    sys.exit()

metrics_res = dataset_eval.evaluate(submission_path, debug=debug)
J, F = metrics_res['J'], metrics_res['F']

# Generate output to the stdout
seq_names = list(J['M_per_object'].keys())
if gt_set == "val" or gt_set == "train" or gt_set == "test-dev":
    sys.stdout.write("----------------Global results in CSV---------------\n")
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    table_g = pandas.DataFrame(data=np.reshape(g_res, [1, len(g_res)]), columns=g_measures)
    table_g.to_csv(sys.stdout, index=False, float_format="%0.3f")

    sys.stdout.write("\n\n------------Per sequence results in CSV-------------\n")
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pandas.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
    table_seq.to_csv(sys.stdout, index=False, float_format="%0.3f")

# Write scores to a file named "scores.txt"
with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    output_file.write("GlobalMean: %f\n" % final_mean)
    output_file.write("JMean: %f\n" % np.mean(J["M"]))
    output_file.write("JRecall: %f\n" % np.mean(J["R"]))
    output_file.write("JDecay: %f\n" % np.mean(J["D"]))
    output_file.write("FMean: %f\n" % np.mean(F["M"]))
    output_file.write("FRecall: %f\n" % np.mean(F["R"]))
    output_file.write("FDecay: %f\n" % np.mean(F["D"]))
total_time = time() - time_start
sys.stdout.write('\nTotal time:' + str(total_time))
