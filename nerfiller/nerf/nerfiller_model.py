# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model for NeRFiller
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import torch
from nerfstudio.model_components.losses import L1Loss, MSELoss, interlevel_loss, tv_loss
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.model_components.losses import (
    depth_ranking_loss,
)

import mediapy
import wandb
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio


@dataclass
class NeRFillerModelConfig(NerfactoModelConfig):
    """Configuration for the NeRFillerModel."""

    _target: Type = field(default_factory=lambda: NeRFillerModel)
    use_lpips: bool = True
    """Whether to use LPIPS loss"""
    use_tv: bool = False
    """Whether to use total variation loss"""
    use_l1: bool = True
    """Whether to use L1 loss"""
    patch_size: int = 32
    """Patch size to use for LPIPS loss."""
    lpips_loss_mult: float = 1.0
    """Multiplier for LPIPS loss."""
    use_depth_ranking: bool = False
    """Whether to use depth ranking loss"""
    depth_ranking_loss_mult: float = 0.1
    """Multiplier for depth ranking loss"""
    depth_ranking_pixel_dist: int = 1
    """Distance between pixels that are compared with each other."""
    tv_loss_mult: float = 0.01
    """Multiplier for tv loss"""
    fix_density: bool = False
    """Whether to fix the density."""
    rgb_only_on_original: bool = False
    """Whether to do rgb loss only on the original valid regions."""
    lpips_only_on_original: bool = False
    """Whether to do lpips loss only on the original valid regions."""
    start_depth_loss: int = 0
    replace_white_with_max_depth: bool = False


class NeRFillerModel(NerfactoModel):
    """Model for NeRFiller."""

    config: NeRFillerModelConfig

    def populate_modules(self):
        """Required to use L1 Loss."""
        super().populate_modules()

        if self.config.use_l1:
            self.rgb_loss = L1Loss()
        else:
            self.rgb_loss = MSELoss()

        self.step = None
        self.start_step = None
        self.trainer_base_dir = None  # to be populated by the pipeline

        if self.config.fix_density:
            self.field._pass_density_gradients = False
            for network in self.proposal_networks:
                network._pass_density_gradients = False

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        depth = batch["depth_image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=batch["image"].to(self.device),
        )

        if self.config.replace_white_with_max_depth:
            white_mask = ((gt_rgb == 1.0).sum(-1, keepdim=True) == 3.0).float()
            depth = torch.where(white_mask == 1.0, torch.ones_like(depth) * depth.max(), depth)

        fake = batch["original_mask"][:, 0].view(-1, self.config.patch_size * self.config.patch_size).eq(0).sum(-1) > 0
        real = ~fake

        original_mask_patches = batch["original_mask"][:, 0].view(-1, 1, self.config.patch_size, self.config.patch_size)
        out_patches = (pred_rgb.view(-1, self.config.patch_size, self.config.patch_size, 3).permute(0, 3, 1, 2)).clamp(
            0, 1
        )
        gt_patches = (gt_rgb.view(-1, self.config.patch_size, self.config.patch_size, 3).permute(0, 3, 1, 2)).clamp(
            0, 1
        )

        if self.config.rgb_only_on_original:
            loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb * batch["original_mask"], pred_rgb * batch["original_mask"])

        if self.config.use_lpips:
            if self.config.lpips_only_on_original:
                real_mask = real[:, None, None, None].float()
                img1 = out_patches * real_mask
                img2 = gt_patches * real_mask
            else:
                img1 = out_patches
                img2 = gt_patches
            loss_dict["lpips_loss"] = self.config.lpips_loss_mult * self.lpips(img1, img2)

        if (
            self.config.use_depth_ranking
            and fake.sum() > 0
            and (self.step - self.start_step >= self.config.start_depth_loss)
        ):
            assert (
                self.config.patch_size % 2 == 0
            ), "Patch size must be even because of how this loss works on adjacent pixels."
            r = self.config.depth_ranking_pixel_dist
            pred_d = outputs["expected_depth"].view(-1, self.config.patch_size, self.config.patch_size, 1)[
                fake, ::r, ::r, :
            ]
            gt_d = depth.view(-1, self.config.patch_size, self.config.patch_size, 1)[fake, ::r, ::r, :]
            dr_x = depth_ranking_loss(pred_d.view(-1, 1), gt_d.view(-1, 1))
            dr_y = depth_ranking_loss(
                pred_d.permute(0, 2, 1, 3).reshape(-1, 1),
                gt_d.permute(0, 2, 1, 3).reshape(-1, 1),
            )
            loss_dict["depth_ranking"] = self.config.depth_ranking_loss_mult * (dr_x + dr_y)

        if self.config.use_tv:
            loss_dict["tv_loss"] = self.config.tv_loss_mult * tv_loss(
                outputs["expected_depth"]
                .view(-1, self.config.patch_size, self.config.patch_size, 1)
                .permute(0, 3, 1, 2)
            )

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )

        return loss_dict
