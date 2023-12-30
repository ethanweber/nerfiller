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
Instruct-NeRF2NeRF Datamanager.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.progress import Console
import torch
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfiller.utils.typing import *

CONSOLE = Console(width=120)


@dataclass
class NeRFillerDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for the InstructNeRF2NeRFDataManager."""

    _target: Type = field(default_factory=lambda: NeRFillerDataManager)
    patch_size: int = 32
    """Size of patch to sample from. If >1, patch-based sampling will be used."""
    sample_everywhere: bool = False
    """If true, sample any pixel regardless of masking."""


class NeRFillerDataManager(VanillaDataManager[DepthDataset]):
    """Data manager for InstructNeRF2NeRF."""

    config: NeRFillerDataManagerConfig

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=[],
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))

        # pre-fetch the image batch (how images are replaced in dataset)
        self.image_batch = next(self.iter_train_image_dataloader)

        # sort the image batch based on self.image_batch['image_idx']
        i_sorted = torch.sort(self.image_batch["image_idx"]).indices.cpu()
        self.image_batch["image_idx"] = self.image_batch["image_idx"][i_sorted]
        self.image_batch["image"] = self.image_batch["image"][i_sorted]
        self.image_batch["mask"] = self.image_batch["mask"][i_sorted]
        self.image_batch["depth_image"] = self.image_batch["depth_image"][i_sorted]

        # keep a copy of the original image batch
        self.original_image_batch = {}
        self.original_image_batch["image_idx"] = self.image_batch["image_idx"].clone()
        self.original_image_batch["image"] = self.image_batch["image"].clone()
        self.original_image_batch["mask"] = self.image_batch["mask"].clone().float()
        self.original_image_batch["depth_image"] = self.image_batch["depth_image"].clone()

        if self.config.sample_everywhere:
            self.image_batch["mask"] = torch.ones_like(self.image_batch["mask"])

        # if set, sample from only these image indices
        self.sample_image_indices = None

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        assert self.train_pixel_sampler is not None

        # possibly choose a subset of images to sample from
        temp_batch = {}
        if self.sample_image_indices:
            for key in self.image_batch.keys():
                temp_batch[key] = self.image_batch[key][self.sample_image_indices]
                temp_batch["original_mask"] = self.original_image_batch["mask"][self.sample_image_indices]
        else:
            temp_batch = self.image_batch
            temp_batch["original_mask"] = self.original_image_batch["mask"]

        batch = self.train_pixel_sampler.sample(temp_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        return ray_bundle, batch
