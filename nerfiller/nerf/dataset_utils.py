"""
Utils for creating a Nerfstudio dataset.
"""

from __future__ import annotations

from pathlib import Path

import mediapy
import torch
from rich.progress import Console
import numpy as np

from nerfiller.utils.typing import *
from nerfstudio.cameras.cameras import Cameras
from nerfiller.utils.camera_utils import rescale_intrinsics

CONSOLE = Console(width=120)


def parse_nerfstudio_frame(
    transforms: Dict,
    data_path: Path,
    idx: int,
    depth_max: int = None,
    device: str = "cpu",
    size: Optional[Tuple[int, int]] = None,
    dtype=torch.float32,
):
    """Parses a Nerfstudio frame, where idx == 0 is the first image sorted by filename.
    The frames are not normally sorted, but we sort them before doing any operations.
    We return processed information where we load images, depth maps, and masks, useful for inpainting this dataset.
    Size will resize the image to (height, width).
    """
    sorted_frames = sorted(transforms["frames"], key=lambda x: x["file_path"])
    imf = data_path / Path(sorted_frames[idx]["file_path"])
    image = torch.from_numpy(mediapy.read_image(imf) / 255.0).permute(2, 0, 1)[None].to(dtype).to(device)
    if "mask_path" in sorted_frames[idx]:
        maf = data_path / Path(sorted_frames[idx]["mask_path"])
        mask = 1 - torch.from_numpy(mediapy.read_image(maf) / 255.0)[None, None].to(dtype).to(device)
    else:
        mask = torch.zeros_like(image[:, :1])
    if "depth_file_path" in sorted_frames[idx]:
        daf = data_path / Path(sorted_frames[idx]["depth_file_path"])
        depth = torch.from_numpy(np.load(daf))[None, None].to(dtype).to(device)
    else:
        depth = torch.zeros_like(image[:, :1])
    # image *= 1 - mask
    # depth *= 1 - mask
    if depth_max:
        depth[depth > depth_max] = 0.0
    # check if the values are stored per frame
    if "fl_x" in sorted_frames[idx]:
        fx = sorted_frames[idx]["fl_x"]
        fy = sorted_frames[idx]["fl_y"]
        cx = sorted_frames[idx]["cx"]
        cy = sorted_frames[idx]["cy"]
    else:
        fx = transforms["fl_x"]
        fy = transforms["fl_y"]
        cx = transforms["cx"]
        cy = transforms["cy"]
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)
    c2wh = torch.tensor(sorted_frames[idx]["transform_matrix"]).to(torch.float32).to(device)
    c2w = c2wh[:3]
    w2ch = torch.inverse(c2wh)
    w2c = w2ch[:3]
    K = K[None]
    c2w = c2w[None]

    if size:
        scale_factor_x = size[1] / image.shape[-1]
        scale_factor_y = size[0] / image.shape[-2]
        image = torch.nn.functional.interpolate(image, size=size, mode="bilinear")
        depth = torch.nn.functional.interpolate(depth, size=size, mode="bilinear")
        mask = torch.nn.functional.interpolate(mask, size=size, mode="nearest")
        K = rescale_intrinsics(K, scale_factor_x, scale_factor_y)

    return image, depth, mask, c2w, K


def create_nerfstudio_frame(
    fl_x,
    fl_y,
    cx,
    cy,
    w,
    h,
    file_path,
    pose: Float[Tensor, "3 4"],
    mask_file_path: Optional[Path] = None,
    depth_file_path: Optional[Path] = None,
):
    """Get a frame in the Nerfstudio DataParser format.

    Args:
        poses: A 4x4 matrix.

    Returns:
        A dictionary a frame/image in the dataset.
    """
    frame = {
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "file_path": file_path,
    }
    if mask_file_path:
        frame["mask_path"] = mask_file_path
    if depth_file_path:
        frame["depth_file_path"] = depth_file_path
    transform_matrix = [[float(pose[i][j]) for j in range(4)] for i in range(4)]
    frame["transform_matrix"] = transform_matrix
    return frame


def random_train_pose(
    size: int,
    resolution: int,
    device: Union[torch.device, str],
    radius_mean: float = 1.0,
    radius_std: float = 0.1,
    central_rotation_range: Tuple[float, float] = (0, 360),
    vertical_rotation_range: Tuple[float, float] = (-90, 0),
    focal_range: Tuple[float, float] = (0.75, 1.35),
    jitter_std: float = 0.01,
    center: Tuple[float, float, float] = (0, 0, 0),
):
    """generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius_mean: mean radius of the orbit camera.
        radius_std: standard deviation of the radius of the orbit camera.
        central_rotation_range: amount that we rotate around the center of the object
        vertical_rotation_range: amount that we allow the cameras to pan up and down from horizontal
        focal_range: focal length range
        jitter_std: standard deviation of the jitter added to the camera position
        center: center of the object
    Return:
        poses: [size, 4, 4]
    """

    assert (
        vertical_rotation_range[0] >= -90 and vertical_rotation_range[1] <= 90
    ), "vertical_rotation_range must be in [-90, 90]"

    vertical_rotation_range = [
        vertical_rotation_range[0] + 90,
        vertical_rotation_range[1] + 90,
    ]
    # This is the uniform sample on the part of the sphere we care about where 0 = 0 degrees and 1 = 360 degrees
    sampled_uniform = (
        torch.rand(size) * (vertical_rotation_range[1] - vertical_rotation_range[0]) + vertical_rotation_range[0]
    ) / 180
    vertical_rotation = torch.arccos(1 - 2 * sampled_uniform)
    central_rotation = torch.deg2rad(
        torch.rand(size) * (central_rotation_range[1] - central_rotation_range[0]) + central_rotation_range[0]
    )

    c_cos = torch.cos(central_rotation)
    c_sin = torch.sin(central_rotation)
    v_cos = torch.cos(vertical_rotation)
    v_sin = torch.sin(vertical_rotation)
    zeros = torch.zeros_like(central_rotation)
    ones = torch.ones_like(central_rotation)

    rot_z = torch.stack(
        [
            torch.stack([c_cos, -c_sin, zeros], dim=-1),
            torch.stack([c_sin, c_cos, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )

    rot_x = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, v_cos, -v_sin], dim=-1),
            torch.stack([zeros, v_sin, v_cos], dim=-1),
        ],
        dim=-2,
    )

    # Default directions are facing in the -z direction, so origins should face opposite way
    origins = torch.stack([torch.tensor([0, 0, 1])] * size, dim=0)
    origins = (origins * radius_mean) + (origins * (torch.randn((origins.shape)) * radius_std))
    R = torch.bmm(rot_z, rot_x)  # Want to have Rx @ Ry @ origin
    t = (
        torch.bmm(R, origins.unsqueeze(-1))
        + torch.randn((size, 3, 1)) * jitter_std
        + torch.tensor(center)[None, :, None]
    )
    camera_to_worlds = torch.cat([R, t], dim=-1)

    focals = torch.rand(size) * (focal_range[1] - focal_range[0]) + focal_range[0]

    cameras = Cameras(
        camera_to_worlds=camera_to_worlds,
        fx=focals * resolution,
        fy=focals * resolution,
        cx=resolution / 2,
        cy=resolution / 2,
    ).to(device)

    return cameras, torch.rad2deg(vertical_rotation), torch.rad2deg(central_rotation)
