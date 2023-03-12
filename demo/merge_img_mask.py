import os
import numpy as np
import cv2

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
          [0.494, 0.000, 0.556], [0.494, 0.000, 0.000], [0.000, 0.745, 0.000],
          [0.700, 0.300, 0.600],[0.000, 0.447, 0.741], [0.850, 0.325, 0.098]]

def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)] # (x1, y1, x2, y2) 

if __name__ == "__main__":
    data_dir = "results"
    output_dir = "outputs_res"
    vids = os.listdir(data_dir)
    for vid_name in vids:
        output_dir_vid = os.path.join(output_dir, vid_name)
        if not os.path.exists(output_dir_vid):
            os.makedirs(output_dir_vid)
        # load images
        image_dir = os.path.join(data_dir, vid_name, "images")
        image_names = sorted(os.listdir(image_dir))
        num_frame = len(image_names)
        image_paths = [os.path.join(image_dir, x) for x in image_names]
        imgs = [cv2.imread(x) for x in image_paths]
        # load masks
        folders = sorted(os.listdir(os.path.join(data_dir, vid_name)))
        folders.remove("images")
        num_folder = len(folders)
        # bboxs
        bboxs = []
        for folder in folders:
            bboxs.append([])
            anno_dir = os.path.join(data_dir, vid_name, folder)
            anno_paths = [os.path.join(anno_dir, x) for x in sorted(os.listdir(anno_dir))]
            num_anno = len(anno_paths)
            assert num_frame == num_anno
            for i in range(num_frame):
                img = imgs[i].astype(np.float32)
                mask = cv2.imread(anno_paths[i]) # [0, 255]
                bbox = bounding_box(mask)
                bboxs[int(folder)].append(bbox)
                img += mask * np.array(COLORS[int(folder)])
                imgs[i] = img
        # save
        for i in range(num_frame):
            save_path = os.path.join(output_dir_vid, "%05d.jpg"%i)
            img = imgs[i]
            img = img/img.max() * 255 # rescale to [0, 255]
            # add bbox
            for f_id in range(num_folder):
                x1, y1, x2, y2 = bboxs[f_id][i]
                color = (int(COLORS[f_id][0]*255), int(COLORS[f_id][1]*255), int(COLORS[f_id][2]*255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)
            cv2.imwrite(save_path, img)