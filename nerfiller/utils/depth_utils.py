import math

import torch

from kornia.geometry.conversions import (
    convert_points_to_homogeneous,
)
from kornia.geometry.epipolar import (
    triangulate_points,
)
from nerfiller.utils.typing import *


def depth_to_distance(depth: Float[Tensor, "B 1 H W"], fov_x: float, fov_y: float):
    device = depth.device
    batch_size, _, height, width = depth.shape
    fov_x_rad = fov_x * torch.pi / 180.0
    fov_y_rad = fov_y * torch.pi / 180.0
    fx = 1 / math.tan(fov_x_rad / 2)
    fy = 1 / math.tan(fov_y_rad / 2)
    x = torch.linspace(-1, 1, width, device=device) / fx
    y = torch.linspace(-1, 1, height, device=device) / fy
    ys, xs = torch.meshgrid([y, x], indexing="ij")
    xy = torch.stack([xs, ys]).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    xy *= depth
    z = depth
    xyz = torch.cat([xy, z], dim=1)
    distance = torch.sqrt((xyz**2).sum(1, keepdim=True))
    return distance


def distance_to_depth(distance: Float[Tensor, "B 1 H W"], fov_x: float, fov_y: float):
    device = distance.device
    batch_size, _, height, width = distance.shape
    fov_x_rad = fov_x * torch.pi / 180.0
    fov_y_rad = fov_y * torch.pi / 180.0
    fx = 1 / math.tan(fov_x_rad / 2)
    fy = 1 / math.tan(fov_y_rad / 2)
    x = torch.linspace(-1, 1, width, device=device) / fx
    y = torch.linspace(-1, 1, height, device=device) / fy
    ys, xs = torch.meshgrid([y, x], indexing="ij")
    xy = torch.stack([xs, ys]).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    xyz = torch.cat([xy, torch.ones_like(xy[:, 0:1])], dim=1)
    xyz = xyz / torch.norm(xyz, dim=1, keepdim=True)
    xyz = xyz * distance
    depth = xyz[:, 2:3]
    return depth


def get_scale_aligned(
    predicted_quantity: Float[Tensor, "B N"],
    fixed_quantity: Float[Tensor, "B N"],
    mask: Float[Tensor, "B N"],
    fuse=True,
):
    """Solves for scale alignment in the overlapping region specified by mask.
    Correspondences are valid where mask==1.
    """
    device = predicted_quantity.device

    b, n = predicted_quantity.shape
    p = predicted_quantity.view(b, -1, 1)
    f = fixed_quantity.view(b, -1, 1)
    m = mask.view(b, -1)

    scale = []
    for i in range(b):
        # solve for A, where ||SX-B|. A is the scale
        X = p[i, m[i] == 1]
        B = f[i, m[i] == 1]
        S = torch.linalg.lstsq(X, B)[0]
        scale.append(S[0, 0])
    scale = torch.stack(scale).to(device)

    return S.view(b, 1) * predicted_quantity


def get_aligned_depth_or_distance(
    predicted_depth_or_distance: Float[Tensor, "B 1 H W"],
    fixed_depth_or_distance: Float[Tensor, "B 1 H W"],
    overlapping_region_mask: Optional[Float[Tensor, "B 1 H W"]] = None,
    fuse=True,
    use_inverse=False,
    mode: str = "scale",
) -> Float[Tensor, "B 1 H W"]:
    """
    Optimize scale and shift parameters in the least squares sense, such that rendered depth_or_distance and predicted depth_or_distance match.
    Overlapping region mask is where the depth aligns.
    """

    batch_size, _, height, width = fixed_depth_or_distance.shape
    assert batch_size == 1

    if overlapping_region_mask is None or overlapping_region_mask.sum() == 0:
        return predicted_depth_or_distance

    p = predicted_depth_or_distance.view(batch_size, -1)
    f = fixed_depth_or_distance.view(batch_size, -1)
    m = overlapping_region_mask.view(batch_size, -1)

    if use_inverse:
        raise NotImplementedError("Inverse is not implemented.")

    assert mode == "scale", "Only scale is implemented."
    a = get_scale_aligned(p, f, m).view(batch_size, -1, height, width)

    if fuse:
        final = torch.where(fixed_depth_or_distance == 0, a, fixed_depth_or_distance)
        return final
    else:
        return a


def depth_from_matches(
    P1: Float[Tensor, "B 3 4"],
    P2: Float[Tensor, "B 3 4"],
    points1: Float[Tensor, "B H W 2"],
    points2: Float[Tensor, "B H W 2"],
):
    """Returns the depth in camera 1.
    points1 is the camera pixel coordinates xy grid of image 1 and
    points2 is the correspondences in image 2.
    """
    B, H, W, _ = points1.shape
    points3d = triangulate_points(P1, P2, points1.view(1, -1, 2), points2.view(1, -1, 2))
    p = torch.bmm(P1, convert_points_to_homogeneous(points3d).permute(0, 2, 1)).permute(0, 2, 1)
    depth = -p.view(1, H, W, 3)[:, :, :, 2]
    return depth


def tv_loss(grids: Float[Tensor, "grids feature_dim row column"], squared: bool = True) -> Float[Tensor, ""]:
    """
    https://github.com/apchenstu/TensoRF/blob/4ec894dc1341a2201fe13ae428631b58458f105d/utils.py#L139

    Args:
        grids: stacks of explicit feature grids (stacked at dim 0)
    Returns:
        average total variation loss for neighbor rows and columns.
    """
    # number_of_grids = grids.shape[0]
    # h_tv_count = grids[:, :, 1:, :].shape[1] * grids[:, :, 1:, :].shape[2] * grids[:, :, 1:, :].shape[3]
    # w_tv_count = grids[:, :, :, 1:].shape[1] * grids[:, :, :, 1:].shape[2] * grids[:, :, :, 1:].shape[3]
    if squared:
        h_tv = torch.pow((grids[:, :, 1:, :] - grids[:, :, :-1, :]), 2)
        w_tv = torch.pow((grids[:, :, :, 1:] - grids[:, :, :, :-1]), 2)
    else:
        h_tv = torch.abs((grids[:, :, 1:, :] - grids[:, :, :-1, :]))
        w_tv = torch.abs((grids[:, :, :, 1:] - grids[:, :, :, :-1]))

    result = torch.zeros_like(grids)
    result[:, :, :-1, :] += h_tv
    result[:, :, :, :-1] += w_tv
    # print(result.shape)
    # print(h_tv.shape)
    # print(w_tv.shape)
    return result.sum(1)

    # return 2 * (h_tv / h_tv_count + w_tv / w_tv_count) / number_of_grids


def reproject(from_l: Float[Tensor, "b H W 2"], depth, from_K, from_c2w, to_K, to_c2w):
    """Reproject from camera 2 into camera 1."""

    device = from_K.device
    BS, H, W, _ = from_l.shape

    K2 = from_K
    c2w2 = from_c2w
    K1 = to_K
    c2w1 = to_c2w

    size = torch.tensor([W, H], device=device)
    pts = ((from_l * 0.5 + 0.5) * size).permute(0, 3, 1, 2)

    d = depth
    ptsh = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1) * d

    Kinv1 = torch.inverse(K1)
    Kinv2 = torch.inverse(K2)
    c2wh1 = torch.cat([c2w1, torch.tensor([[[0, 0, 0, 1]]], device=device)], dim=1)
    c2wh2 = torch.cat([c2w2, torch.tensor([[[0, 0, 0, 1]]], device=device)], dim=1)

    # TODO(ethan): avoid needing to do this
    c2wh1[:, :3, 1:3] *= -1
    c2wh2[:, :3, 1:3] *= -1

    w2ch1 = torch.inverse(c2wh1)[:, :3]
    w2ch2 = torch.inverse(c2wh2)[:, :3]
    w2ch1 = torch.cat([w2ch1, torch.tensor([[[0, 0, 0, 1]]], device=device)], dim=1)
    w2ch2 = torch.cat([w2ch2, torch.tensor([[[0, 0, 0, 1]]], device=device)], dim=1)

    ptsw = torch.bmm(Kinv2, ptsh.view(BS, 3, -1)).view(BS, 3, H, W)
    ptsw = torch.cat([ptsw, torch.ones_like(ptsw[:, :1])], dim=1)
    ptsw = torch.bmm(c2wh2, ptsw.view(BS, 4, -1)).view(BS, 4, H, W)
    ptsw = torch.bmm(w2ch1, ptsw.view(BS, 4, -1)).view(BS, 4, H, W)
    ptsc = torch.bmm(K1, ptsw.view(BS, 4, -1)[:, :3]).view(BS, 3, H, W)

    # non-continuous version
    z = ptsc[0, 2]
    x = (ptsc[0, 0] / z).long()
    y = (ptsc[0, 1] / z).long()
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    valid_depth = torch.zeros_like(depth[0, 0])
    valid_depth[y[valid], x[valid]] = z[valid]

    depth_out = ptsc[:, 2:]
    ptsc = ptsc[:, :2] / depth_out

    ptsc = (ptsc.permute(0, 2, 3, 1) / size) * 2 - 1

    return ptsc, depth_out, valid_depth
