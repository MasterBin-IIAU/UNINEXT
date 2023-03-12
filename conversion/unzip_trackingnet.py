import os

"""
This script is used to process the original trackingnet zip files
TRAIN_0.zip ~ TRAIN_3.zip should be put under "datasets" dir
"""
if __name__ == "__main__":
    data_root = "datasets"
    set_ids = list(range(4))
    """unzip"""
    for i in set_ids:
        ori_zip_file = os.path.join(data_root, "TRAIN_%d.zip"%i)
        unzip_dir = os.path.join(data_root, "TRAIN_%d"%i)
        if not os.path.exists(unzip_dir):
            os.makedirs(unzip_dir)
        print("unzipping %s"%ori_zip_file)
        os.system("unzip -qq %s -d %s" %(ori_zip_file, unzip_dir))
        print("%s is done."%ori_zip_file)
        frame_root = os.path.join(unzip_dir, "frames")
        os.makedirs(frame_root)
        zip_dir = os.path.join(unzip_dir, "zips")
        sub_zips = os.listdir(zip_dir)
        for sub_zip in sub_zips:
            seq_name = sub_zip.replace(".zip", "")
            unzip_frame_dir = os.path.join(frame_root, seq_name)
            if not os.path.exists(unzip_frame_dir):
                os.makedirs(unzip_frame_dir)
            src_path = os.path.join(zip_dir, sub_zip)
            des_dir = unzip_frame_dir
            os.system("unzip -qq %s -d %s" %(src_path, des_dir))
            print("%s is done." %sub_zip)
        os.system("rm -rf %s"%zip_dir)
    os.chdir(data_root)
    os.makedirs("TrackingNet")
    os.system("mv TRAIN_* TrackingNet")