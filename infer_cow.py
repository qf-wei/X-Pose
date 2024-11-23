import os
from inference_on_a_image import *
import json

base_path = "/nas/Main/Hutzper/COW_dataset"

sub_folders = ["Chiba_0911", "Kagoshima_0925", "Kagoshima_1008", "Obihiro_0912", "Obihiro_1016"]

images_paths = []

for sub_folder in sub_folders:
    path = os.path.join(base_path, sub_folder)
    print(path)
    for sub_sub_folder in os.listdir(path):
        sub_path = os.path.join(path, sub_sub_folder)
        images_paths_in_subfolder = [os.path.join(sub_path, image) for image in os.listdir(sub_path) if image.endswith("jpg")]
        images_paths.extend(images_paths_in_subfolder)

print("Total images: ", len(images_paths))


output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "json"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)


# load model
config_file = "config_model/UniPose_SwinT.py"
checkpoint_path = "unipose_swint.pth"
model = load_model(config_file, checkpoint_path, cpu_only=False)

instance_text_prompt = "cow"
keypoint_dict = globals()["animal"]
keypoint_text_prompt = keypoint_dict.get("keypoints")
keypoint_skeleton = keypoint_dict.get("skeleton")

box_threshold = 0.05
iou_threshold = 0.9


# run model
for image_path in images_paths:
    # load image
    image_pil, image = load_image(image_path)

    boxes_filt,keypoints_filt = get_unipose_output(
        model, image, instance_text_prompt, keypoint_text_prompt, box_threshold, iou_threshold, cpu_only=False
    )

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": np.array(boxes_filt).tolist(),
        "keypoints": np.array(keypoints_filt).tolist(),
        "size": [size[1], size[0]]
    }
    # import ipdb; ipdb.set_trace()
    save_image_name = image_path.split("/")[-3] + "_" + image_path.split("/")[-2] + "_" + image_path.split("/")[-1]
    with open(os.path.join(output_dir, "json", os.path.basename(save_image_name).replace("jpg", "json")), "w") as f:
        json.dump(pred_dict, f, indent=4)

    plot_on_image(image_pil, pred_dict,keypoint_skeleton,keypoint_text_prompt, os.path.join(output_dir, "images", os.path.basename(save_image_name)))
        