"""
Convert a NeRF synthetic dataset to a Nerfstudio dataset.
"""

import tyro
from pathlib import Path
import json
import mediapy
import numpy as np
import cv2
import glob

import torch
from nerfiller.nerf.dataset_utils import create_nerfstudio_frame
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap


def main(
    input_folder: Path = Path("data/blender/chair"),
    output_folder: Path = Path("data/nerfstudio/chair"),
    num_frames: int = 40,
):
    """
    This function converts NeRF synethetic data to a Nerfstudio formatted dataset.
    """

    f = open(input_folder / "transforms_test.json")
    transforms_train = json.load(f)
    f.close()

    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / "images").mkdir(parents=True, exist_ok=True)
    (output_folder / "original_images").mkdir(parents=True, exist_ok=True)
    (output_folder / "masks").mkdir(parents=True, exist_ok=True)
    (output_folder / "original_depth").mkdir(parents=True, exist_ok=True)
    (output_folder / "depth").mkdir(parents=True, exist_ok=True)

    num_images = len(transforms_train["frames"])
    file_paths = []
    mask_file_paths = []
    depth_file_paths = []
    poses = []

    target_height, target_width = 512, 512

    # indices = torch.linspace(0, num_images - 1, num_frames).int().tolist()
    indices = torch.arange(num_images).int().tolist()

    for idx, i in enumerate(indices):
        image_filename = input_folder / (transforms_train["frames"][i]["file_path"] + ".png")
        image_4 = mediapy.read_image(image_filename)
        # white background
        image = image_4[:, :, :3]
        alpha = image_4[:, :, 3:4] / 255.0
        image = (image * alpha + (1 - alpha) * 255.0).astype("uint8")
        height, width = image.shape[:2]  # original shape
        image = cv2.resize(image, (target_width, target_height), cv2.INTER_LINEAR)
        # image = mediapy.resize_image(image, shape=(target_height, target_width))
        # image[image_4[:, :, 3] == 0] = 255

        file_path = f"images/image_{idx:06d}.png"
        file_paths.append(file_path)
        mediapy.write_image(output_folder / file_path, image)

        mask_file_path = f"masks/mask_{idx:06d}.png"
        mask_file_paths.append(mask_file_path)
        mask = 1 - alpha[..., 0]
        mask = cv2.resize(mask, (target_width, target_height), cv2.INTER_NEAREST)
        mask[:] = 1
        # mask[64:448, 192:320] = 0
        # mask[128:384, 192:320] = 0
        mask[128:384, 128:384] = 0
        mediapy.write_image(output_folder / mask_file_path, mask)

        mediapy.write_image(
            output_folder / f"images/image_{idx:06d}.png",
            image * mask[..., None].astype(image.dtype),
        )
        mediapy.write_image(output_folder / f"original_images/image_{idx:06d}.png", image)

        depth_filename = glob.glob(str(input_folder / (transforms_train["frames"][i]["file_path"] + "_depth_*.png")))[0]
        depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (target_width, target_height), cv2.INTER_LINEAR)
        depth = torch.from_numpy(depth).float()[..., 0]
        depth = torch.where(
            depth != 0.0, (1.0 - (depth / 255.0)) * 8.0, 0.0
        )  # conversion from Blender params used in original NeRF paper

        # original depth
        depth_file_path = f"original_depth/depth_{idx:06d}.npy"
        # depth_file_paths.append(depth_file_path)
        np.save(output_folder / depth_file_path, depth.numpy())
        depth_with_colormap = apply_colormap(depth[..., None], ColormapOptions(normalize=True)).detach().cpu().numpy()
        mediapy.write_image(output_folder / f"original_depth/depth_{i:06d}.png", depth_with_colormap)

        # masked depth
        depth_file_path = f"depth/depth_{idx:06d}.npy"
        depth_file_paths.append(depth_file_path)
        np.save(output_folder / depth_file_path, depth.numpy() * mask)
        depth_with_colormap = (
            apply_colormap(
                depth[..., None] * torch.from_numpy(mask).to(depth)[..., None],
                ColormapOptions(normalize=True),
            )
            .detach()
            .cpu()
            .numpy()
        )
        mediapy.write_image(output_folder / f"depth/depth_{i:06d}.png", depth_with_colormap)

        mediapy.write_image(
            output_folder / f"original_depth/image_{idx:06d}.png",
            image * mask[..., None].astype(image.dtype),
        )

        poses.append(transforms_train["frames"][i]["transform_matrix"])

    template = {
        "camera_model": "OPENCV",
        "orientation_override": "none",
        "frames": [],
    }

    camera_angle_x = float(transforms_train["camera_angle_x"])
    focal_length = 0.5 * width / np.tan(0.5 * camera_angle_x)
    focal_length = focal_length * target_width / width
    fx = focal_length
    fy = focal_length
    cx = target_width / 2.0
    cy = target_height / 2.0

    frames = []
    for i in range(len(poses)):
        frame = create_nerfstudio_frame(
            fl_x=fx,
            fl_y=fy,
            cx=cx,
            cy=cy,
            w=target_width,
            h=target_height,
            pose=poses[i],
            file_path=file_paths[i],
            mask_file_path=mask_file_paths[i],
            depth_file_path=depth_file_paths[i],
        )
        frames.append(frame)
    template["frames"] = frames
    with open(output_folder / "transforms.json", "w") as f:
        json.dump(template, f, indent=4)


if __name__ == "__main__":
    tyro.cli(main)
