import argparse
from detectron2.config import get_cfg
from detectron2.projects.uninext import add_uninext_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
import torch
from vot_tool import VOT
import cv2
import sys
import os
import numpy as np

prj_root = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

def get_parser():
    parser = argparse.ArgumentParser(description="UNINEXT for builtin configs")
    parser.add_argument(
        "--config-file",
        default=os.path.join(prj_root, "projects/UNINEXT/configs/eval-vid/video_joint_vit_huge_eval_vots.yaml"),
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[
            "MODEL.WEIGHTS", os.path.join(prj_root, "outputs/video_joint_vit_huge/model_final.pth"), \
            "SOT.ONLINE_UPDATE", "True"
            ],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--gpu_id", default=0)
    return parser

def setup_cfg(args):
    cfg = get_cfg()
    add_uninext_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg



class UNINEXTPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self):
        args = get_parser().parse_args()
        cfg = setup_cfg(args)
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, frame_idx, obj_idx, original_mask=None):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            transform = self.aug.get_transform(original_image)
            image = transform.apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": [image], "height": height, "width": width}
            if original_mask is not None:
                mask = make_full_size(original_mask, (width, height))
                mask = transform.apply_segmentation(mask) # (H, W)
                # print(mask.shape, image.shape)
                prediction = self.model([inputs], frame_idx, obj_idx, mask)
            else:
                prediction = self.model([inputs], frame_idx, obj_idx)
            return prediction


def run_vot_exp(gpu_id=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    handle = VOT("mask", multiobject=True)
    objects = handle.objects() # List of masks
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)

    image = cv2.imread(imagefile)
    frame_idx = 0
    trackers = [UNINEXTPredictor() for _ in objects]
    for obj_idx, (tracker, cur_object) in enumerate(zip(trackers, objects)):
        tracker(image, frame_idx, str(obj_idx), cur_object)

    while True:
        imagefile = handle.frame()
        frame_idx += 1
        if not imagefile:
            break
        image = cv2.imread(imagefile)
        res = []
        for obj_idx, tracker in enumerate(trackers):
            res.append(tracker(image, frame_idx, str(obj_idx)))
        handle.report(res)

def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)

if __name__ == "__main__":
    run_vot_exp()