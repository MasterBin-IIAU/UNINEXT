'''
wjf5203
'''
import argparse
import datetime
import json
from cv2 import threshold


import numpy as np
import torch
import os
import json
import pycocotools.mask as mask_util
import sys
import cv2



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # dataset parameters
    parser.add_argument('--img_path', default='datasets/ovis/valid')
    parser.add_argument('--ann_path', default='datasets/ovis/annotations_valid.json')
    parser.add_argument('--output_dir', default='output_ytvos',
                        help='path where to save, empty for no saving')
    parser.add_argument('--vis_json', default='outputs/video_joint_vit_huge/inference/results.json')

    return parser

def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)] # (x1, y1, x2, y2) 

CLASSES=['person','giant_panda','lizard','parrot','skateboard','sedan','ape',
         'dog','snake','monkey','hand','rabbit','duck','cat','cow','fish',
         'train','horse','turtle','bear','motorbike','giraffe','leopard',
         'fox','deer','owl','surfboard','airplane','truck','zebra','tiger',
         'elephant','snowboard','boat','shark','mouse','frog','eagle','earless_seal',
         'tennis_racket']
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
          [0.494, 0.000, 0.556], [0.494, 0.000, 0.000], [0.000, 0.745, 0.000],
          [0.700, 0.300, 0.600],[0.000, 0.447, 0.741], [0.850, 0.325, 0.098]]


# COLORS = [ [0.850, 0.325, 0.098], [0.000, 0.447, 0.741],
# [0.850, 0.325, 0.098], [0.000, 0.447, 0.741], 
# [0.850, 0.325, 0.098], [0.000, 0.447, 0.741], 
# [0.850, 0.325, 0.098], [0.000, 0.447, 0.741], 
# [0.850, 0.325, 0.098], [0.000, 0.447, 0.741], ]



def main(args):

    

    with torch.no_grad():

        folder = args.img_path
        videos = json.load(open(args.ann_path,'rb'))['videos']#[:5]
        # videos = [videos[1],videos[8],videos[22],videos[34]]
        vis_num = len(videos)
        # postprocess = PostProcessSegm_ifc()
        result = [] 
        
        threshold = 0.4
        json_results = json.load(open(args.vis_json,'rb'))
        # import pdb;pdb.set_trace()
        all_result={}
        for inst in json_results:
            if inst['video_id'] in all_result.keys():
                if inst['score']>threshold:
                    all_result[inst['video_id']].append(inst['segmentations'])
            else:
                all_result[inst['video_id']] = []
                if inst['score']>threshold:
                    all_result[inst['video_id']].append(inst['segmentations'])
        # mask_util.decode(json_results[0]['segmentations'][frame_i])
        # import pdb;pdb.set_trace()
        
        # for i in range(293,294):
        # for i in range(vis_num):
        # video_list = [28,48,124]
        video_list = [2, 3, 20, 38, 55, 70, 89, 123]
        for i in video_list:
            print("Process video: ",i)
            id_ = videos[i]['id']
            vid_len = videos[i]['length']
            file_names = videos[i]['file_names']

            ori_img_list = [cv2.imread(os.path.join(folder,file_names[n]),-1) for n in range(vid_len)]
            zero_masks = [np.zeros_like(ig) for ig in ori_img_list]
            # import pdb;pdb.set_trace()
            num_inst = len(all_result[id_])
            bbox_list = []
            for nn, segmentations in enumerate(all_result[id_]):
                bbox_list.append([])
                # import pdb;pdb.set_trace()
                # print('score',)
                for n in range(vid_len):            
                    mask_i = segmentations[n]
                    if mask_i == None:
                        bbox_list[nn].append(None)
                    else:
                        # import pdb;pdb.set_trace()
                        mask = mask_util.decode(mask_i)
                        if (mask > 0).any():
                            bbox = bounding_box(mask)
                        else:
                            bbox = None
                        bbox_list[nn].append(bbox)
                        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
                        lar = np.concatenate((mask*COLORS[nn%12][0], mask*COLORS[nn%12][1], mask*COLORS[nn%12][2]), axis = 2)
                        zero_masks[n] = zero_masks[n]+ lar
    
              
                   
            output_dir = os.path.join('visualization',args.output_dir,'vid_{}'.format(str(i)))
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            for n in range(len(ori_img_list)):
                img_n = ori_img_list[n]+np.clip(zero_masks[n],0,1)*255
                for nn in range(num_inst):
                    bbox = bbox_list[nn][n]
                    if bbox is not None:
                        color = (int(COLORS[nn%12][0]*255), int(COLORS[nn%12][1]*255), int(COLORS[nn%12][2]*255))
                        cv2.rectangle(img_n, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
                max_p = img_n.max()
                ret = 255*img_n/max_p
                img_path = os.path.join(output_dir, file_names[n].split('/')[-1])
                cv2.imwrite(img_path, ret)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(' inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
