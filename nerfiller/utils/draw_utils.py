"""
Utils for drawing.
"""

import cv2
import numpy as np
import torch

from nerfiller.utils.typing import *


def draw_lines_on_image(image, lines, colors, thickness: int = 2):
    """
    Image of shape [h,w,3] in range [0,255].
    Lines as list of [(x,y),(x,y)] coordinates as pixels.
    Colors in range [0,255].
    """
    new_image = np.ascontiguousarray(image, dtype=np.uint8)
    assert len(lines) == len(colors)
    for (p0, p1), color in zip(lines, colors):
        cv2.line(new_image, p0, p1, color, thickness)
    return new_image


def draw_keypoints_on_image(image, keypoints, colors, radius=3, thickness: int = -1):
    """
    Image of shape [h,w,3] in range [0,255].
    Keypoints as (x,y) coordinates as pixels.
    Colors in range [0,255].
    """
    new_image = np.ascontiguousarray(image, dtype=np.uint8)
    assert len(keypoints) == len(colors)
    for keypoint, color in zip(keypoints, colors):
        cv2.circle(new_image, keypoint, radius, color, thickness)
    return new_image


def get_images_with_keypoints(
    images: Float[Tensor, "B 3 H W"],
    keypoints: Float[Tensor, "B N 2"],
    colors: Optional[Float[Tensor, "B N 3"]] = None,
    keypoint_size: int = 10,
    thickness: int = -1,
):
    """Returns the batch of images with keypoints drawn in the colors.
    Images in range [0, 1].
    Keypoints are (x,y) coordinates in range [-1,1].
    Colors are RGB in range (0, 1).
    """
    device = images.device
    b, _, h, w = images.shape
    _, N, _ = keypoints.shape

    if colors is None:
        colors = torch.rand((b, N, 3), device=device)

    new_images = []
    for idx in range(b):
        im = np.ascontiguousarray(
            (images[idx].permute(1, 2, 0).detach().clone().cpu().numpy() * 255.0).astype("uint8")
        ).astype("uint8")

        ke = ((keypoints[idx] * 0.5 + 0.5) * torch.tensor([w, h], device=device)) - 0.5
        ke = [(int(x), int(y)) for x, y in ke]
        co = (colors[idx] * 255.0).detach().clone().cpu().numpy().astype("uint8")
        co = [(int(r), int(g), int(b)) for r, g, b in co]
        im = draw_keypoints_on_image(im, ke, co, radius=keypoint_size, thickness=thickness)

        new_images.append(im)
    new_images = np.stack(new_images, axis=0)
    new_images = torch.tensor(new_images).permute(0, 3, 1, 2).float().to(device) / 255.0
    return new_images


def get_images_with_lines(
    images: Float[Tensor, "B 3 H W"],
    lines: Float[Tensor, "B N 2 2"],
    colors: Optional[Float[Tensor, "B N 3"]] = None,
    line_width: int = 2,
):
    """Returns the batch of images with lines drawn in the colors.
    Images in range [0, 1].
    Lines are [(x,y), (x,y)] coordinates in range [-1,1].
    Colors are RGB in range (0, 1).
    """
    device = images.device
    b, _, h, w = images.shape
    _, N, _, _ = lines.shape

    if colors is None:
        colors = torch.rand((b, N, 3), device=device)

    new_images = []
    for idx in range(b):
        im = np.ascontiguousarray(
            (images[idx].permute(1, 2, 0).detach().clone().cpu().numpy() * 255.0).astype("uint8")
        ).astype("uint8")

        li = ((lines[idx] * 0.5 + 0.5) * torch.tensor([w, h], device=device)) - 0.5
        co = (colors[idx] * 255.0).detach().clone().cpu().numpy().astype("uint8")
        co = [(int(r), int(g), int(b)) for r, g, b in co]
        li = [((int(p[0, 0]), int(p[0, 1])), (int(p[1, 0]), int(p[1, 1]))) for p in li]
        im = draw_lines_on_image(im, li, co, thickness=line_width)

        new_images.append(im)
    new_images = np.stack(new_images, axis=0)
    new_images = torch.tensor(new_images).permute(0, 3, 1, 2).float().to(device) / 255.0
    return new_images
