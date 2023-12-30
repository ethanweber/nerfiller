import math

import torch
import numpy as np

from nerfiller.utils.typing import *


def c2wh_from_c2w(c2w):
    c2wh = torch.cat([c2w, torch.zeros_like(c2w[:1])])
    c2wh[-1, -1] = 1
    return c2wh


def rot_x(theta: float):
    """
    theta in radians
    """
    return [
        [1, 0, 0],
        [0, math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta), math.cos(theta)],
    ]


def rot_y(theta: float):
    """
    theta in radians
    """
    return [
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)],
    ]


def rot_z(theta: float):
    """
    theta in radians
    """
    return [
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1],
    ]


def get_perspective_directions(height, width, theta_fov, phi_fov, device="cpu"):
    """Return a grid of directions for the perspective image. The directions are normalized."""
    y_offset = 1.0 / width  # half a pixel
    z_offset = 1.0 / height  # half a pixel
    y = -1 * torch.linspace(-1 + y_offset, 1 - y_offset, width, device=device)
    z = -1 * torch.linspace(-1 + z_offset, 1 - z_offset, height, device=device)
    zs, ys = torch.meshgrid([z, y], indexing="ij")
    ys_zs = torch.stack([ys, zs])
    y = ys_zs[0:1]
    z = ys_zs[1:2]
    direction = torch.cat(
        [
            torch.ones_like(y),  # x = forward
            y * math.tan(theta_fov / 2),  # y = left
            z * math.tan(phi_fov / 2),  # z = up
        ],
        dim=0,
    )
    direction = direction / torch.norm(direction, dim=0, keepdim=True)
    return direction


def get_equirectangular_directions(height, width, theta_fov, phi_fov, device="cpu"):
    """Return a grid of directions for the equirectangular image. The directions are normalized."""
    # http://corysimon.github.io/articles/uniformdistn-on-sphere/
    theta_offset = (theta_fov / width) / 2.0  # half a pixel
    phi_offset = (phi_fov / height) / 2.0  # half a pixel
    theta = -1 * torch.linspace(
        -theta_fov / 2 + theta_offset,
        theta_fov / 2 - theta_offset,
        width,
        device=device,
    )  # [theta_fov/2, -theta_fov/2]
    phi = torch.linspace(
        torch.pi / 2 - phi_fov / 2 + phi_offset,
        torch.pi / 2 + phi_fov / 2 - phi_offset,
        height,
        device=device,
    )  # [pi/2 - phi_fov/2, pi/2 + phi_fov/2]
    phis, thetas = torch.meshgrid([phi, theta], indexing="ij")
    theta_phi = torch.stack([thetas, phis])
    theta = theta_phi[0:1]
    phi = theta_phi[1:2]
    direction = torch.cat(
        [
            torch.sin(phi) * torch.cos(theta),  # x = forward
            torch.sin(phi) * torch.sin(theta),  # y = left
            torch.cos(phi),  # z = up
        ],
        dim=0,
    )
    direction = direction / torch.norm(direction, dim=0, keepdim=True)
    return direction


def get_theta_and_phi_from_directions(directions):
    # convert direction to equirectangular coordinates
    x = directions[:, 0]
    y = directions[:, 1]
    z = directions[:, 2]
    theta = torch.atan2(y, x)
    phi = torch.acos(z) - (torch.pi / 2)
    return theta, phi


def get_fov_from_focal_len(height_or_width, focal_len):
    """Returns fov in degrees."""
    fov_rad = 2.0 * np.arctan(height_or_width / (2.0 * focal_len))
    fov_deg = fov_rad * 180.0 / np.pi
    return fov_deg


def get_focal_len_from_fov(height_or_width, fov_in_degrees):
    """Returns the focal length."""
    fov_rad = fov_in_degrees * np.pi / 180.0
    focal_len = 0.5 * height_or_width / np.tan(0.5 * fov_rad)
    return focal_len


def get_pinhole_intrinsics_from_fov(H, W, fov_in_degrees=55.0):
    px, py = (W - 1) / 2.0, (H - 1) / 2.0
    fx = fy = get_focal_len_from_fov(W, fov_in_degrees)
    k_ref = np.array(
        [
            [fx, 0.0, px, 0.0],
            [0.0, fy, py, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    k_ref = torch.tensor(k_ref)  # K is [4,4]

    return k_ref


def rescale_intrinsics(Ks: Float[Tensor, "B 3 3 3"], scale_factor_x: float, scale_factor_y: float):
    Ks_new = Ks.clone()
    Ks_new[:, 0:1] *= scale_factor_x
    Ks_new[:, 1:2] *= scale_factor_y
    return Ks_new


def get_projection_matrix(K: Float[Tensor, "B 3 3"], c2w: Float[Tensor, "B 3 4"]) -> Float[Tensor, "B 3 4"]:
    batch_size = K.shape[0]
    device = K.device

    row = torch.tensor([[[0, 0, 0, 1]]], device=device).repeat(batch_size, 1, 1)
    c2wh = torch.cat([c2w, row], dim=1)
    P = torch.bmm(K, torch.inverse(c2wh)[:, :3])
    return P
