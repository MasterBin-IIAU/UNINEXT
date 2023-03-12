import os
import cv2
import numpy as np

fps = 24 # 12
size = (1280, 720)
# DATA = ["0782a6df7e", "3e03f623bb", "4f6662e4e0", 
#         "68dab8f80c", "7a72130f21", "b3b92781d9"]
# DATA = ["vid_2", "vid_3", "vid_20", "vid_38", "vid_55", "vid_70", "vid_89", "vid_123"]
DATA = ["camel", "breakdance", "gold-fish", "motocross-jump", "pigs", "soapbox"]
for vid_name in DATA:
    video = cv2.VideoWriter(f"{vid_name}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    path = f"outputs_DAVIS/{vid_name}"
    filelist = sorted(os.listdir(path))
    for item in filelist:
        if item.endswith('.jpg'): 
            item = os.path.join(path, item)
            img = cv2.imread(item)
            # resize
            img = cv2.resize(img, (1280, 720))
            video.write(img)

    video.release()