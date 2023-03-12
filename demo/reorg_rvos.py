import os

DATA = ["0782a6df7e", "3e03f623bb", "4f6662e4e0", 
        "68dab8f80c", "7a72130f21", "b3b92781d9"]

if __name__ == "__main__":
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    for data in DATA:
        vid_name = data
        img_dir = os.path.join("datasets/ref-youtube-vos/valid/JPEGImages", vid_name)
        res_dir = os.path.join("outputs/video_joint_vit_huge/Annotations", vid_name)
        os.system(f"cp -r {res_dir} {output_dir}/{vid_name}")
        os.system(f"cp -r {img_dir} {output_dir}/{vid_name}/images")
        # manually remove folders for the same instances
        # then rename to 0, 1, 2, ...

