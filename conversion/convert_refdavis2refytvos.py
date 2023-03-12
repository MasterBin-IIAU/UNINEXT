"""
There are 2 steps for converting ref-davis to ytvis. (we only convert the val split for evaluation without finetune)
1. convert_refdavis2refytvos.py.
2. convert_refdavis2ytvis_val.py.
"""

import os
import json

"""
ytvos structure
- train
    - Annotations
        - video1
        - video2
    - JPEGImages
        - video1 
        -video2
    meta.json
- valid
    - Annotations
    - JPEGImages
    meta.json
- meta_expressions
    - train
        meta_expressions.json
    - valid
        meta_expressions.json
"""

def read_split_set(data_root='data/ref-davis'):
    set_split_path = os.path.join(data_root, "DAVIS/ImageSets/2017")
    # train set
    with open(os.path.join(set_split_path, "train.txt"), "r") as f:
        train_set = f.readlines()
    train_set = [x.strip() for x in train_set] # 60 videos
    # val set
    with open(os.path.join(set_split_path, "val.txt"), "r") as f:
        val_set = f.readlines()
    val_set = [x.strip() for x in val_set] # 30 videos
    return train_set, val_set # List


def mv_images_to_folder(data_root='data/ref-davis', output_root='data/ref-davis'):
    train_img_path = os.path.join(output_root, "train/JPEGImages")
    train_anno_path = os.path.join(output_root, "train/Annotations")
    val_img_path = os.path.join(output_root, "valid/JPEGImages")
    val_anno_path = os.path.join(output_root, "valid/Annotations")
    meta_train_path = os.path.join(output_root, "meta_expressions/train")
    meta_val_path = os.path.join(output_root, "meta_expressions/valid")
    paths = [train_img_path, train_anno_path, val_img_path, val_anno_path,
             meta_train_path, meta_val_path]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    # 1. read the train/val split
    train_set, val_set = read_split_set(data_root)

    # 2. move images and annotations
    # train set
    for video in train_set:
        # move images
        base_img_path = os.path.join(data_root, "DAVIS/JPEGImages/480p", video)
        mv_cmd = f"mv {base_img_path} {train_img_path}"
        os.system(mv_cmd)
        # move annotations
        base_anno_path = os.path.join(data_root, "DAVIS/Annotations_unsupervised/480p", video)
        mv_cmd = f"mv {base_anno_path} {train_anno_path}"
        os.system(mv_cmd)
    # val set
    for video in val_set:
        # move images
        base_img_path = os.path.join(data_root, "DAVIS/JPEGImages/480p", video)
        mv_cmd = f"mv {base_img_path} {val_img_path}"
        os.system(mv_cmd)
        # move annotations
        base_anno_path = os.path.join(data_root, "DAVIS/Annotations_unsupervised/480p", video)
        mv_cmd = f"mv {base_anno_path} {val_anno_path}"
        os.system(mv_cmd)

def create_meta_expressions(data_root='data/ref-davis', output_root='data/ref-davis'):
    """
    NOTE: expressions odd: first anno, even: full anno
    meta_expression.json format
    {
        "videos": {
            "video1: {
                "expressions": {
                    "0": {
                        "exp": "xxxxx",
                        "obj_id": "1" (start from 1)
                    }
                    "1": {
                        "exp": "xxxxx",
                        "obj_id": "1"
                    }
                }
                "frames": [
                    "00000",
                    "00001",
                    ...
                ]
            }
        }
    }
    """
    train_img_path = os.path.join(output_root, "train/JPEGImages")
    val_img_path = os.path.join(output_root, "valid/JPEGImages")
    meta_train_path = os.path.join(output_root, "meta_expressions/train")
    meta_val_path = os.path.join(output_root, "meta_expressions/valid")

    # 1. read the train/val split
    train_set, val_set = read_split_set(data_root)

    # 2. create meta_expression.json
    # NOTE: there are two annotators, and each annotator have first anno and full anno, respectively
    def read_expressions_from_txt(file_path, encoding='utf-8'):
        """
        videos["video1"] = [
            {"obj_id": 1, "exp": "xxxxx"},
            {"obj_id": 2, "exp": "xxxxx"},
            {"obj_id": 3, "exp": "xxxxx"},
        ]
        """
        videos = {}
        with open(file_path, "r", encoding=encoding) as f:
            for idx, line in enumerate(f.readlines()):
                line = line.strip()
                video_name, obj_id = line.split()[:2]
                exp = ' '.join(line.split()[2:])[1:-1]
                # handle bad case
                if video_name == "clasic-car":
                    video_name = "classic-car"
                elif video_name == "dog-scale":
                    video_name = "dogs-scale"
                elif video_name == "motor-bike":
                    video_name = "motorbike"

                
                if not video_name in videos.keys():
                    videos[video_name] = []
                exp_dict = {
                    "exp": exp,
                    "obj_id": obj_id
                }
                videos[video_name].append(exp_dict)

        # sort the order of expressions in each video
        for key, value in videos.items():
            value = sorted(value, key = lambda e:e.__getitem__('obj_id'))
            videos[key] = value
        return videos

    anno1_first_path = os.path.join(data_root, "davis_text_annotations/Davis17_annot1.txt")
    anno1_full_path = os.path.join(data_root, "davis_text_annotations/Davis17_annot1_full_video.txt")
    anno2_first_path = os.path.join(data_root, "davis_text_annotations/Davis17_annot2.txt")
    anno2_full_path = os.path.join(data_root, "davis_text_annotations/Davis17_annot2_full_video.txt")
    # all videos information
    anno1_first = read_expressions_from_txt(anno1_first_path, encoding='utf-8')
    anno1_full = read_expressions_from_txt(anno1_full_path, encoding='utf-8')
    anno2_first = read_expressions_from_txt(anno2_first_path, encoding='latin-1')
    anno2_full = read_expressions_from_txt(anno2_full_path, encoding='latin-1')

    # 2(1). train
    train_videos = {}  # {"video1": {}, "video2": {}, ...}, the final results to dump
    for video in train_set: # 60 videos
        video_dict = {} # for each video

        # store the information of video
        expressions = {}
        exp_id = 0 # start from 0
        for anno1_first_video, anno1_full_video, anno2_first_video, anno2_full_video in zip(
                                anno1_first[video], anno1_full[video], anno2_first[video], anno2_full[video]):
            expressions[str(exp_id)] = anno1_first_video
            exp_id += 1
            expressions[str(exp_id)] = anno1_full_video
            exp_id += 1
            expressions[str(exp_id)] = anno2_first_video
            exp_id += 1
            expressions[str(exp_id)] = anno2_full_video
            exp_id += 1
        video_dict["expressions"] = expressions
        # read frame names for each video
        video_frames = os.listdir(os.path.join(train_img_path, video))
        video_frames = [x.split(".")[0] for x in video_frames] # remove ".jpg"
        video_frames.sort()
        video_dict["frames"] = video_frames

        train_videos[video] = video_dict
    
    # 2(2). val
    val_videos = {}
    for video in val_set:
        video_dict = {} # for each video

        # store the information of video
        expressions = {}
        exp_id = 0 # start from 0
        for anno1_first_video, anno1_full_video, anno2_first_video, anno2_full_video in zip(
                                anno1_first[video], anno1_full[video], anno2_first[video], anno2_full[video]):
            expressions[str(exp_id)] = anno1_first_video
            exp_id += 1
            expressions[str(exp_id)] = anno1_full_video
            exp_id += 1
            expressions[str(exp_id)] = anno2_first_video
            exp_id += 1
            expressions[str(exp_id)] = anno2_full_video
            exp_id += 1
        video_dict["expressions"] = expressions
        # read frame names for each video
        video_frames = os.listdir(os.path.join(val_img_path, video))
        video_frames = [x.split(".")[0] for x in video_frames] # remove ".jpg"
        video_frames.sort()
        video_dict["frames"] = video_frames

        val_videos[video] = video_dict

    # 3. store the meta_expressions.json
    # train
    train_meta = {"videos": train_videos}
    with open(os.path.join(meta_train_path, "meta_expressions.json"), "w") as out:
        json.dump(train_meta, out)
    # val 
    val_meta = {"videos": val_videos}
    with open(os.path.join(meta_val_path, "meta_expressions.json"), "w") as out:
        json.dump(val_meta, out)

def create_meta_annotaions(data_root='data/ref-davis', output_root='data/ref-davis'):
    """
    NOTE: frame names are not stored compared with ytvos
    meta.json format
    {
        "videos": {
            "video1: {
                "objects": {
                    "1": {"category": "bike"},
                    "2": {"category": "person"}
                }
            }
        }
    }
    """
    out_train_path = os.path.join(output_root, "train")
    out_val_path = os.path.join(output_root, "valid")

    # read the semantic information
    with open(os.path.join(data_root, "DAVIS/davis_semantics.json")) as f:
        davis_semantics = json.load(f)

    # 1. read the train/val split
    train_set, val_set = read_split_set(data_root)

    # 2. create meta.json
    # train
    train_videos = {}
    for video in train_set:
        video_dict = {} # for each video
        video_dict["objects"] = {}
        num_obj = len(davis_semantics[video].keys())
        for obj_id in range(1, num_obj+1): # start from 1
            video_dict["objects"][str(obj_id)] = {"category": davis_semantics[video][str(obj_id)]}
        train_videos[video] = video_dict

    # val
    val_videos = {}
    for video in val_set:
        video_dict = {}
        video_dict["objects"] = {}
        num_obj = len(davis_semantics[video].keys())
        for obj_id in range(1, num_obj+1): # start from 1
            video_dict["objects"][str(obj_id)] = {"category": davis_semantics[video][str(obj_id)]}
        val_videos[video] = video_dict
    
    # store the meta.json file
    train_meta = {"videos": train_videos}
    with open(os.path.join(out_train_path, "meta.json"), "w") as out:
        json.dump(train_meta, out)
    val_meta = {"videos": val_videos}
    with open(os.path.join(out_val_path, "meta.json"), "w") as out:
        json.dump(val_meta, out)

if __name__ == '__main__':
    data_root = "datasets/ref-davis"
    output_root = "datasets/ref-davis"
    print("Converting ref-davis to ref-youtube-vos format....")
    mv_images_to_folder(data_root, output_root)
    create_meta_expressions(data_root, output_root)
    create_meta_annotaions(data_root, output_root)

