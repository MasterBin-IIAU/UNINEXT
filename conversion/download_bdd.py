import os
import multiprocessing

def download(url):
    os.system("wget -c %s" % url)
    if url.endswiths(".zip"):
        filename = os.path.split(url)[-1]
        os.system("unzip -qq %s" % filename)
        os.system("rm %s"%filename)

if __name__ == "__main__":
    save_dir = "BDD100K"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    url_list = [
        "http://dl.yf.io/bdd100k/mots20/bdd100k_seg_track_20_images.zip",
        "http://dl.yf.io/bdd100k/mots20/bdd100k_seg_track_20_images.zip.md5",
        "http://dl.yf.io/bdd100k/mot20/images20-track-test-1.zip",
        "http://dl.yf.io/bdd100k/mot20/images20-track-test-1.zip.md5",
        "http://dl.yf.io/bdd100k/mot20/images20-track-test-2.zip",
        "http://dl.yf.io/bdd100k/mot20/images20-track-test-2.zip.md5",
        "http://dl.yf.io/bdd100k/mot20/images20-track-train-1.zip",
        "http://dl.yf.io/bdd100k/mot20/images20-track-train-1.zip.md5",
        "http://dl.yf.io/bdd100k/mot20/images20-track-train-2.zip",
        "http://dl.yf.io/bdd100k/mot20/images20-track-train-2.zip.md5",
        "http://dl.yf.io/bdd100k/mot20/images20-track-train-3.zip",
        "http://dl.yf.io/bdd100k/mot20/images20-track-train-3.zip.md5",
        "http://dl.yf.io/bdd100k/mot20/images20-track-train-4.zip",
        "http://dl.yf.io/bdd100k/mot20/images20-track-train-4.zip.md5",
        "http://dl.yf.io/bdd100k/mot20/images20-track-train-5.zip",
        "http://dl.yf.io/bdd100k/mot20/images20-track-train-5.zip.md5",
        "http://dl.yf.io/bdd100k/mot20/images20-track-train-6.zip",
        "http://dl.yf.io/bdd100k/mot20/images20-track-train-6.zip.md5",
        "http://dl.yf.io/bdd100k/mot20/images20-track-train-7.zip",
        "http://dl.yf.io/bdd100k/mot20/images20-track-train-7.zip.md5",
        "http://dl.yf.io/bdd100k/mot20/images20-track-val-1.zip",
        "http://dl.yf.io/bdd100k/mot20/images20-track-val-1.zip.md5",
        "https://bdd-data-storage-release.s3.us-west-2.amazonaws.com/bdd100k/2021/bdd100k_box_track_20_labels_trainval.zip",
        "https://bdd-data-storage-release.s3.us-west-2.amazonaws.com/bdd100k/2021/bdd100k_seg_track_20_labels_trainval.zip",
        "https://bdd-data-storage-release.s3.us-west-2.amazonaws.com/bdd100k/2021/bdd100k_images_100k.zip",
        "https://bdd-data-storage-release.s3.us-west-2.amazonaws.com/bdd100k/2021/bdd100k_det_20_labels_trainval.zip",
        # instance segmentation
        "https://bdd-data-storage-release.s3.us-west-2.amazonaws.com/bdd100k/2021/bdd100k_images_10k.zip",
        "https://bdd-data-storage-release.s3.us-west-2.amazonaws.com/bdd100k/2021/bdd100k_ins_seg_labels_trainval.zip",
    ]
    os.chdir(save_dir)
    for url in url_list:
        download(url)