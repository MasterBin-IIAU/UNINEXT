import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Launcher')
    parser.add_argument('--config_name', type=str, default='video_joint_r50.yaml')
    parser.add_argument('--exp_dir_name', type=str, default='video_joint_r50')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_cmd = "python3 launch.py --nn 1 --eval-only \
                --uni 1 --config-file projects/UNINEXT/configs/%s \
                --resume OUTPUT_DIR outputs/%s MODEL.USE_IOU_BRANCH False \
                " %(args.config_name, args.exp_dir_name)
    for obj_thr in [0.3]: # np.arange(0.2, 0.7, 0.1)
        for init_thr in [0.4]: # np.arange(0.2, 0.7, 0.1)
            if init_thr <= obj_thr:
                continue
            cmd = base_cmd + "TRACK.INIT_SCORE_THR %.2f TRACK.OBJ_SCORE_THR %.2f"%(init_thr, obj_thr)
            print(cmd)
            save_path = os.path.join("outputs/%s/inference/instances_predictions_init_%.2f_obj_%.2f.pkl"%(args.exp_dir_name, init_thr, obj_thr))
            if os.path.exists(save_path):
                print("%s exists. Skip." % save_path)
            else:
                os.system(cmd)
            # eval
            eval_cmd = "python3 tools_bin/eval_bdd.py %s" % save_path
            os.system(eval_cmd)
            
