"""
Code for epipolar guidance.
"""

import mediapy
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfiller.utils.typing import *

from nerfstudio.model_components.losses import L1Loss, MSELoss

from dreamsim import dreamsim


import torch
import random
import mediapy


def get_random_patches(images: Float[Tensor, "B C H W"], N: int, patch_size: int) -> Float[Tensor, "B N C Hp Wp"]:
    """Gets random square patches from images.
    N is the number of patches.
    patch_size is the height and width.
    """

    batch_size, channels, H, W = images.shape

    patches = torch.zeros((batch_size, N, channels, patch_size, patch_size)).to(images)

    for i in range(batch_size):
        for j in range(N):
            x = random.randint(0, W - patch_size)
            y = random.randint(0, H - patch_size)
            patches[i, j] = images[i, :, y : y + patch_size, x : x + patch_size]

    return patches


class SimilarityMetric(torch.nn.Module):
    """
    Feature extractor.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

    def forward(self, image: Float[Tensor, "B C H W"], target: Float[Tensor, "B C H W"]):
        pass


class ReconstructionMetric(SimilarityMetric):
    """
    Computes the similarity per-pixel.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(device=device)
        self.l1loss = L1Loss()
        self.l2loss = MSELoss()

    def forward(self, image: Float[Tensor, "B C H W"], target: Float[Tensor, "B C H W"]):
        loss = self.l2loss(image, target)
        return loss


class LPIPSMetric(SimilarityMetric):
    """
    Computes the similarity with LPIPS.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(device=device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)

    def forward(self, image: Float[Tensor, "B C H W"], target: Float[Tensor, "B C H W"]):
        B, C, H, W = image.shape
        num_patches = 32
        patch_size = 64
        combined = torch.cat([image, target], dim=1)
        combined_patches = get_random_patches(combined, num_patches, patch_size)
        image_patches, target_patches = combined_patches.chunk(2, dim=2)
        image_patches = image_patches.view(B * num_patches, C, patch_size, patch_size)
        target_patches = target_patches.view(B * num_patches, C, patch_size, patch_size)
        all_patches = torch.cat(
            [
                torch.cat(list(image_patches.permute(0, 2, 3, 1)), dim=1).detach().cpu(),
                torch.cat(list(target_patches.permute(0, 2, 3, 1)), dim=1).detach().cpu(),
            ],
            dim=0,
        )
        mediapy.write_image("all_patches.png", all_patches)
        return self.lpips(image_patches, target_patches)


class DreamSimMetric(SimilarityMetric):
    """
    Computes the similarity with DreamSim.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(device=device)
        self.model, preprocess = dreamsim(pretrained=True, device=device)
        self.img_size = 224

    def forward(self, image: Float[Tensor, "B C H W"], target: Float[Tensor, "B C H W"]):
        image_resized = torch.nn.functional.interpolate(image, size=(self.img_size, self.img_size), mode="bilinear")
        target_resized = torch.nn.functional.interpolate(target, size=(self.img_size, self.img_size), mode="bilinear")
        distance = self.model(image_resized, target_resized)
        return distance
