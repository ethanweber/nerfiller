"""
A class for perspective cropping from an equirectangular photo.
"""

import math
from dataclasses import dataclass

import mediapy
import torch
from jaxtyping import Float, Int
from torch import Tensor

from nerfiller.utils.camera_utils import (
    rot_y,
    rot_z,
    get_perspective_directions,
    get_equirectangular_directions,
    get_theta_and_phi_from_directions,
)

from nerfiller.utils.typing import *


@dataclass
class EquiImageOutput:
    image: Float[Tensor, "bs C H W"]
    """Image output."""
    image_mask: Optional[Float[Tensor, "bs 1 H W"]] = None
    """Image mask output."""
    distance: Optional[Float[Tensor, "bs 1 H W"]] = None
    """Distance output."""
    distance_mask: Optional[Float[Tensor, "bs 1 H W"]] = None
    """Distance mask output."""
    grid: Optional[Float[Tensor, "bs H H 2"]] = None
    """Grid used to query from 360 photo into perspective crop."""
    inverse_grid: Optional[Float[Tensor, "bs H_equi W_equi 2"]] = None
    """Grid used to query from perspective crop into 360 photo."""


class EquiImage(torch.nn.Module):
    """Class to represent the scene as a set of equirectangular images."""

    def __init__(
        self,
        num_images: int = 10,
        width: int = 612,
        height: int = 306,
        image_dim: int = 3,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.num_images = num_images
        self.width = width
        self.height = height
        self.image_dim = image_dim
        self.device = device
        self.configure()

    def configure(self):
        self.image = torch.nn.Parameter(
            torch.zeros(
                (self.num_images, self.image_dim, self.height, self.width),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.register_buffer(
            "image_mask",
            torch.zeros(
                (self.num_images, 1, self.height, self.width),
                dtype=torch.float32,
                device=self.device,
            ),
        )
        self.register_buffer(
            "distance",
            torch.zeros(
                (self.num_images, 1, self.height, self.width),
                dtype=torch.float32,
                device=self.device,
            ),
        )
        self.register_buffer(
            "distance_mask",
            torch.zeros(
                (self.num_images, 1, self.height, self.width),
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def __len__(self) -> int:
        return self.image.shape[0]

    def mask_from_grid(self, grid: Float[Tensor, "bs H W 2"]) -> Float[Tensor, "bs 1 H_orig W_orig"]:
        """Returns the mask where the grid sampled."""
        bs = grid.shape[0]
        H_orig, W_orig = self.mask.shape[-2:]
        mask = torch.zeros((bs, 1, H_orig, W_orig), device=grid.device)
        for b in range(bs):
            x = ((0.5 * grid[b, :, :, 0].flatten() + 0.5) * (W_orig - 1)).long().clamp(0, W_orig - 1)
            y = ((0.5 * grid[b, :, :, 1].flatten() + 0.5) * (H_orig - 1)).long().clamp(0, H_orig - 1)
            mask[b, 0, y, x] = 1.0
        return mask

    def insert_into_image(
        self,
        image_indices: Int[Tensor, "bs"],
        inverse_grid: Float[Tensor, "bs H_equi W_equi 2"],
        image: Float[Tensor, "bs 3 H W"],
        mask: Optional[Float[Tensor, "bs 1 H W"]] = None,
    ):
        """Inserts image values into the equirectangular image."""
        sampled_image = torch.nn.functional.grid_sample(
            image,
            inverse_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        if mask is None:
            mask = torch.ones_like(image[:, 0:1])
        sampled_mask = torch.nn.functional.grid_sample(
            mask,
            inverse_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        self.image.data[image_indices] = (
            self.image.data[image_indices] * (1 - sampled_mask) + sampled_mask * sampled_image
        )
        self.image_mask[image_indices] = self.image_mask[image_indices] * (1 - sampled_mask) + sampled_mask

    def insert_into_distance(
        self,
        image_indices: Int[Tensor, "bs"],
        inverse_grid: Float[Tensor, "bs H_equi W_equi 2"],
        distance: Float[Tensor, "bs 1 H W"],
        mask: Optional[Float[Tensor, "bs 1 H W"]] = None,
    ):
        """Inserts distance values into the equirectangular image."""
        sampled_distance = torch.nn.functional.grid_sample(
            distance,
            inverse_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        if mask is None:
            mask = torch.ones_like(image[:, 0:1])
        sampled_mask = torch.nn.functional.grid_sample(
            mask,
            inverse_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        self.distance[image_indices] = (
            self.distance[image_indices] * (1 - sampled_mask) + sampled_mask * sampled_distance
        )
        self.distance_mask[image_indices] = self.distance_mask[image_indices] * (1 - sampled_mask) + sampled_mask

    def scene_outputs_from_grid_sample(self, image_indices: Int[Tensor, "bs"], grid: Float[Tensor, "bs H W 2"]):
        im = self.image[image_indices]  # size: (bs, 3, H, W)
        im_ma = self.image_mask[image_indices]  # size: (bs, 1, H, W)
        di = self.distance[image_indices]  # size: (bs, 1, H, W)
        di_ma = self.distance_mask[image_indices]  # size: (bs, 1, H, W)
        grid = grid.to(im)
        image_out = torch.nn.functional.grid_sample(
            im,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        image_mask_out = torch.nn.functional.grid_sample(
            im_ma,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        distance_out = torch.nn.functional.grid_sample(
            di,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        distance_mask_out = torch.nn.functional.grid_sample(
            di_ma,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return EquiImageOutput(
            image=image_out,
            image_mask=image_mask_out,
            distance=distance_out,
            distance_mask=distance_mask_out,
            grid=grid,
        )

    def show(self, display_height=50, roll=False, columns=None):
        """Show the scene if using a notebook."""
        mediapy.show_images(
            self.image.permute(0, 2, 3, 1).detach().cpu(),
            height=display_height,
            columns=columns,
        )
        mediapy.show_images(
            self.distance.permute(0, 2, 3, 1).detach().cpu(),
            height=display_height,
            columns=columns,
        )
        if roll:
            mediapy.show_images(
                torch.roll(self.distance, width // 2, dims=-1).permute(0, 2, 3, 1).detach().cpu(),
                height=display_height,
                columns=columns,
            )
            mediapy.show_images(
                torch.roll(self.distance, width // 2, dims=-1).permute(0, 2, 3, 1).detach().cpu(),
                height=display_height,
                columns=columns,
            )

    def forward(
        self,
        image_indices: Int[Tensor, "bs"],
        yaws: Float[Tensor, "bs"],
        pitches: Float[Tensor, "bs"],
        fov_x: int,
        fov_y: int,
        width: int,
        height: int,
        invalid_number: float = 2.0,
    ) -> EquiImageOutput:
        """
        Args:
            image_indices: The image indices to sample from.
            yaws: The yaw rotation in degrees.
            pitches: The pitch rotation (elevation) in degrees.
            fov_x: Field of view of the horizontal axis in degrees.
            fov_y: Field of view of the vertical axis in degrees.
            width: Width resolution of the crop.
            height: Height resolution of the crop.

        Returns:
            The scene outputs.
        """
        num_images, c, h_equi, w_equi = self.image.shape
        h_equi = torch.tensor(h_equi, device=self.device)
        w_equi = torch.tensor(w_equi, device=self.device)
        bs = image_indices.shape[0]

        # field of view in radians
        fov_x_rad = fov_x * torch.pi / 180.0
        fov_y_rad = fov_y * torch.pi / 180.0

        # rotations
        yaws_rad = yaws * torch.pi / 180.0
        pitches_rad = -1 * pitches * torch.pi / 180.0
        rot_yaw = torch.tensor(
            [rot_z(y) for y in yaws_rad],
            device=self.device,
        )
        rot_pitch = torch.tensor(
            [rot_y(p) for p in pitches_rad],
            device=self.device,
        )
        rot = torch.einsum("bij,bjk->bik", rot_yaw, rot_pitch)
        rot_inv = rot.permute(0, 2, 1)

        # crop equirectangular image
        directions = (
            get_perspective_directions(
                height,
                width,
                theta_fov=fov_x_rad,
                phi_fov=fov_y_rad,
                device=self.device,
            )
            .unsqueeze(0)
            .repeat(bs, 1, 1, 1)
        )
        # apply rotation
        directions = torch.einsum("bij,bjhw->bihw", rot, directions)
        theta, phi = get_theta_and_phi_from_directions(directions)

        u = -theta / torch.pi
        v = 2 * phi / torch.pi
        grid = torch.stack((u, v), dim=3)  # [batch_size, H, W, 2]

        scene_outputs = self.scene_outputs_from_grid_sample(image_indices, grid)

        #########################################
        # go on to compute the inverse as well

        # full equirectangular image
        directions = (
            get_equirectangular_directions(
                h_equi,
                w_equi,
                theta_fov=2 * torch.pi,
                phi_fov=torch.pi,
                device=self.device,
            )
            .unsqueeze(0)
            .repeat(bs, 1, 1, 1)
        )
        # apply opposite rotation
        directions = torch.einsum("bij,bjhw->bihw", rot_inv, directions)

        fx = 1 / math.tan(fov_x_rad / 2)
        fy = 1 / math.tan(fov_y_rad / 2)
        x = directions[:, 0]
        y = directions[:, 1]
        z = directions[:, 2]
        u = -(y / x) * fy
        v = -(z / x) * fx

        # remove projections on the back side of the perspective camera
        # by setting the value to an arbitrary value of 2, outside the range [-1, 1]
        u[x < 0] = invalid_number
        v[x < 0] = invalid_number

        # coordinate grid for sampling
        grid = torch.stack((u, v), dim=3)  # [batch_size, h_equi, w_equi, 2]
        scene_outputs.inverse_grid = grid

        scene_outputs.inverse_grid = torch.nan_to_num(
            scene_outputs.inverse_grid,
            nan=invalid_number,
            posinf=invalid_number,
            neginf=invalid_number,
        )

        return scene_outputs
