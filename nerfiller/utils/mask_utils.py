"""
Code for generating masks to inpaint.
"""

import torch

from nerfiller.utils.mesh_utils import (
    dilate,
    get_mesh_from_perspective_images,
    project_mesh_into_perspective_image,
)
from nerfiller.utils.typing import *
from nerfstudio.cameras.camera_utils import viewmatrix


def create_depth_aware_mask(
    image: Float[Tensor, "B 3 H W"],
    distance: Float[Tensor, "B 1 H W"],
    fov: float,
    max_distance: float,
    resolution: int,
    scale_factor: int = 4,
):
    """

    Args:
        image: Image to create a mask for
        distance: Distance to use to create the mesh
        fov: Camera field of view in degrees.
        max_distance: Maximum translation distance that the newx view could be rendered from.
        resolution: The resolution to render the image with.
        scale_factor: How much higher in resolution the rendered image should be to avoid issues with too few faces being rendered.
    """

    device = image.device

    # backproject to create a mesh
    vertices, vertex_colors, faces, faces_mask = get_mesh_from_perspective_images(
        image,
        distance,
        fov * torch.pi / 180.0,
        fov * torch.pi / 180.0,
    )

    # create a random camera pose to render from
    tr = torch.rand((3,)) * 2 - 1
    tr /= torch.sqrt((tr**2).sum())
    pos = tr * torch.rand(1) * max_distance
    c2w = viewmatrix(
        lookat=torch.tensor([0, 0, 1]).float(),
        up=torch.tensor([0, 1, 0]).float(),
        pos=pos,
    ).float()

    rendered_image, _, p2f = project_mesh_into_perspective_image(
        vertices=vertices,
        colors=vertex_colors,
        faces=faces,
        fov=fov,
        image_size=resolution * scale_factor,
        faces_per_pixel=1,
        c2w=c2w,
    )

    rendered_image = torch.nn.functional.interpolate(
        rendered_image.permute(2, 0, 1)[None], scale_factor=1.0 / scale_factor
    )[0].permute(1, 2, 0)

    # NOTE(ethan): we make some assumptions here about the face layout
    num_faces = (resolution - 1) * (resolution - 1) * 2
    faces_all = torch.arange(num_faces).reshape(2, resolution - 1, resolution - 1).to(device)
    valid_faces_ul = torch.isin(faces_all[0], p2f[..., 0])
    valid_faces_lr = torch.isin(faces_all[1], p2f[..., 0])

    valid_mask = torch.zeros((resolution, resolution)).float().to(device)
    valid_mask[: resolution - 1, : resolution - 1] = valid_faces_ul
    valid_mask[1:, 1:] = (valid_mask[1:, 1:] + valid_faces_lr).clamp(0, 1)

    mask = 1 - valid_mask
    return mask, rendered_image


def random_mask_custom(im_shape, ratio=1, mask_full_image=False, max_np=10, min_np=2):
    num_points = int(torch.rand(1) * (max_np - min_np)) + min_np
    size = torch.randint(low=0, high=im_shape[0] // 4, size=(num_points,))
    y = torch.randint(low=0, high=im_shape[0], size=(num_points,))
    x = torch.randint(low=0, high=im_shape[0], size=(num_points,))
    mask = torch.zeros(im_shape)
    for i in range(num_points):
        m = torch.zeros(im_shape)
        m[y[i], x[i]] = 1.0
        for i in range(size[i]):
            m = dilate(m[None, None])[0, 0]
        mask = (mask + m).clamp(0, 1)
    return mask


def downscale_mask(mask, size=None, scale_factor=None, dilate_iters=0, dilate_kernel_size=3):
    """
    Downscale the mask in a conservative way. 1s are where to inpaint, 0 where to not inpaint.
    Inpaints extra pixels to prevent leakage under the mask.
    """
    assert size or scale_factor
    if size:
        assert scale_factor is None
    if scale_factor:
        assert size is None
    for _ in range(dilate_iters):
        mask = dilate(mask, kernel_size=dilate_kernel_size)
    mask = torch.nn.functional.interpolate(mask, size=size, scale_factor=scale_factor, mode="bilinear")
    mask = (mask != 0.0).float()  # expands the mask slightly for no leakage of pixels
    return mask
