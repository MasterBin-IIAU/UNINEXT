import numpy as np
import os
import shutil
import argparse


def transform_trackingnet(exp_name):
    src_dir = os.path.join("outputs", exp_name, "inference", "TrackingNet/")
    dest_dir = os.path.join("outputs", exp_name, "inference", "TrackingNet_submit/")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    items = os.listdir(src_dir)
    for item in items:
        if "all" in item:
            continue
        if "time" not in item:
            src_path = os.path.join(src_dir, item)
            dest_path = os.path.join(dest_dir, item)
            bbox_arr = np.loadtxt(src_path, dtype=np.int, delimiter='\t')
            np.savetxt(dest_path, bbox_arr, fmt='%d', delimiter=',')
    # make zip archive
    shutil.make_archive(src_dir, "zip", src_dir)
    shutil.make_archive(dest_dir, "zip", dest_dir)
    # Remove the original files
    shutil.rmtree(src_dir)
    shutil.rmtree(dest_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='transform trackingnet results.')
    parser.add_argument('--exp_name', type=str, help='Name of exp.')

    args = parser.parse_args()
    transform_trackingnet(args.exp_name)