"""
Depth-aware dataset code with masks based on occlusion boundaries.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torchvision import transforms
from torchvision.transforms import v2

from nerfiller.equi_image import EquiImage
from nerfiller.utils.depth_utils import distance_to_depth
from nerfiller.utils.mask_utils import create_depth_aware_mask
from nerfiller.utils.diff_utils import tokenize_prompt
from nerfiller.utils.io_utils import (
    image_distance_from_dataset,
)
from nerfiller.utils.typing import *
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap
from nerfiller.utils.mask_utils import random_mask_custom
import glob
import mediapy
import random
from nerfiller.inpaint.depth_inpainter import DepthInpainter
from torchvision.transforms import InterpolationMode


class NerfbustersDataset(Dataset):
    """
    A dataset that operates on a folder of images.
    """

    def __init__(
        self,
        instance_prompt: str,
        tokenizer,
        dataset_type: str,
        dataset_name: Path,
        fov: float = 90.0,
        path_prefix: Path = Path("data"),
        tile_images: bool = False,
        tile_images_percentage: float = 0.5,
        length: int = 100,
        resolution: int = 512,
        scale_factor: int = 4,
        mask_type: str = "rectangle",
        device: str = "cuda:0",
    ):
        assert dataset_type == "nerfbusters"
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.path_prefix = path_prefix
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name
        self.fov = fov
        self.resolution = resolution
        self.scale_factor = scale_factor
        self.mask_type = mask_type
        self.device = device

        self.image_folder = self.path_prefix / self.dataset_type / self.dataset_name / "images"
        self.image_filenames = sorted(glob.glob(str(self.image_folder / "*")))

        if self.mask_type == "train-dist":
            self.mask_filenames = sorted(
                glob.glob(str(self.path_prefix / self.dataset_type / self.dataset_name / "masks" / "*"))
            )
            if len(self.mask_filenames) == 0:
                raise ValueError("no filenames in mask folder")
            self.mask_transforms = v2.Compose(
                [
                    v2.RandomResizedCrop(
                        size=(512, 512),
                        antialias=True,
                        interpolation=InterpolationMode.NEAREST,
                    ),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomVerticalFlip(p=0.5),
                ]
            )

        self.text_inputs = tokenize_prompt(self.tokenizer, self.instance_prompt, tokenizer_max_length=None)

        self.image_transforms = v2.Compose(
            [
                v2.RandomResizedCrop(size=(512, 512), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
            ]
        )

        if self.mask_type == "depth-aware":
            self.depth_inpainter = DepthInpainter(depth_method="zoedepth", device=self.device)
            self.idx_to_depth = {}

    def __len__(self):
        return len(self.image_filenames)

    def get_crop(self, index):
        idx = random.randint(0, len(self.image_filenames) - 1)
        filename = self.image_filenames[idx]
        image = torch.from_numpy(mediapy.read_image(filename)) / 255.0
        image = image.permute(2, 0, 1)
        if self.mask_type == "depth-aware":
            if idx not in self.idx_to_depth:
                with torch.no_grad():
                    depth = (
                        self.depth_inpainter.get_depth(image=image[None].to(self.depth_inpainter.device))[0]
                        .detach()
                        .cpu()
                    )
                self.idx_to_depth[idx] = depth
            else:
                depth = self.idx_to_depth[idx]
        else:
            depth = torch.zeros_like(image[:1])
        image, depth = torch.split(self.image_transforms(torch.cat([image, depth])), [3, 1])
        return image, depth

    def __getitem__(self, index):
        example = {}

        image, depth = self.get_crop(index)

        if self.mask_type == "rectangle":
            mask = random_mask_custom((self.resolution, self.resolution), max_np=20, min_np=10)[None]
        elif self.mask_type == "depth-aware":
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                mask, rendered_image = create_depth_aware_mask(
                    image[None].to(self.device),
                    depth[None].to(self.device),
                    self.fov,
                    max_distance=depth.median().item(),
                    resolution=self.resolution,
                    scale_factor=self.scale_factor,
                )
            mask = mask[None].detach().cpu()
        elif self.mask_type == "train-dist":
            idx = random.randint(0, len(self.mask_filenames) - 1)
            filename = self.mask_filenames[idx]
            mask = 1.0 - torch.from_numpy(mediapy.read_image(filename))[None] / 255.0
            mask = self.mask_transforms(mask)

        # shape is [3,H,W] or [1,H,W]
        example["image"] = image
        example["mask"] = mask
        example["depth"] = torch.zeros_like(mask)

        # text stuff
        example["input_ids"] = self.text_inputs.input_ids[0]  # shape (seq_len,)
        example["attention_mask"] = self.text_inputs.attention_mask[0]  # shape (seq_len,)

        return example


class EquiDataset(Dataset):
    """
    A dataset that operates on an equirectangular image and masks.
    """

    def __init__(
        self,
        instance_prompt: str,
        tokenizer,
        dataset_type: str,
        dataset_name: Path,
        fov: float = 90.0,
        length: int = 100,
        resolution: int = 512,
        scale_factor: int = 4,
        max_distance: float = 10.0,
        tile_images: bool = False,
        tile_images_percentage: float = 0.5,
        mask_type: str = "rectangle",
        device: str = "cuda:0",
    ):
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.fov = fov
        self.length = length
        self.resolution = resolution
        self.scale_factor = scale_factor
        self.max_distance = max_distance
        self.tile_images = tile_images
        self.tile_images_percentage = tile_images_percentage
        self.mask_type = mask_type
        self.device = device

        image, distance = image_distance_from_dataset(
            dataset_type=dataset_type,
            dataset_name=dataset_name,
            device=self.device,
            max_distance=self.max_distance,
        )
        self.diameter = distance[distance != 0.0].min()
        self.equi_image = EquiImage(
            num_images=1,
            width=image.shape[-1],
            height=image.shape[-2],
            device=self.device,
        )
        self.equi_image.image.data = image
        self.equi_image.distance = distance

        self.text_inputs = tokenize_prompt(self.tokenizer, self.instance_prompt, tokenizer_max_length=None)

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.length

    def get_crop(self, index):
        num_crops = 1
        image_indices = torch.zeros((num_crops)).int().to(self.device)
        yaws = torch.rand((num_crops), dtype=torch.float32) * 360
        pitch_range = [-55, 55]
        pitches = torch.rand((num_crops), dtype=torch.float32) * (pitch_range[1] - pitch_range[0]) + pitch_range[0]
        with torch.no_grad():
            ei_output = self.equi_image.forward(
                image_indices,
                yaws,
                pitches,
                self.fov,
                self.fov,
                self.resolution,
                self.resolution,
            )
            image, distance = ei_output.image, ei_output.distance
        return image, distance

    def get_info(self, index):
        image, distance = self.get_crop(index)

        if self.mask_type == "rectangle":
            mask = random_mask_custom((self.resolution, self.resolution), max_np=20, min_np=10)[None, None]
        elif self.mask_type == "depth-aware":
            depth = distance_to_depth(distance, self.fov, self.fov)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                mask, rendered_image = create_depth_aware_mask(
                    image,
                    distance,
                    self.fov,
                    max_distance=float(self.diameter),
                    resolution=self.resolution,
                    scale_factor=self.scale_factor,
                )

            mask = mask[None, None]
        rendered_image = rendered_image.permute(2, 0, 1)[None]

        return image, distance, depth, mask, rendered_image

    def __getitem__(self, index):
        example = {}

        if self.tile_images and torch.rand(1).item() < self.tile_images_percentage:
            # tile 4 images
            image1, distance1, depth1, mask1, rendered_image1 = self.get_info(index)
            image2, distance2, depth2, mask2, rendered_image2 = self.get_info(index)
            image3, distance3, depth3, mask3, rendered_image3 = self.get_info(index)
            image4, distance4, depth4, mask4, rendered_image4 = self.get_info(index)

            image = torch.cat(
                [
                    torch.cat([image1, image2], dim=-1),
                    torch.cat([image3, image4], dim=-1),
                ],
                dim=-2,
            )
            image = torch.nn.functional.interpolate(image, size=(self.resolution, self.resolution))

            distance = torch.cat(
                [
                    torch.cat([distance1, distance2], dim=-1),
                    torch.cat([distance3, distance4], dim=-1),
                ],
                dim=-2,
            )
            distance = torch.nn.functional.interpolate(distance, size=(self.resolution, self.resolution))

            depth = torch.cat(
                [
                    torch.cat([depth1, depth2], dim=-1),
                    torch.cat([depth3, depth4], dim=-1),
                ],
                dim=-2,
            )
            depth = torch.nn.functional.interpolate(depth, size=(self.resolution, self.resolution))

            mask = torch.cat(
                [
                    torch.cat([mask1, mask2], dim=-1),
                    torch.cat([mask3, mask4], dim=-1),
                ],
                dim=-2,
            )
            mask = torch.nn.functional.interpolate(mask, size=(self.resolution, self.resolution), mode="nearest")

            rendered_image = torch.cat(
                [
                    torch.cat([rendered_image1, rendered_image2], dim=-1),
                    torch.cat([rendered_image3, rendered_image4], dim=-1),
                ],
                dim=-2,
            )
            rendered_image = torch.nn.functional.interpolate(rendered_image, size=(self.resolution, self.resolution))
        else:
            image, distance, depth, mask, rendered_image = self.get_info(index)

        # shape is [3,H,W] or [1,H,W]
        example["image"] = image[0]
        example["distance"] = distance[0]
        example["depth"] = depth[0]
        example["rendered_image"] = rendered_image[0]
        example["mask"] = mask[0]

        # text stuff
        example["input_ids"] = self.text_inputs.input_ids[0]  # shape (seq_len,)
        example["attention_mask"] = self.text_inputs.attention_mask[0]  # shape (seq_len,)

        return example


if __name__ == "__main__":
    torch.manual_seed(1)
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-inpainting"
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    dataset = EquiDataset(
        instance_prompt="a photo of sks",
        tokenizer=tokenizer,
        fov=90.0,
        dataset_type="blender",
        dataset_name="village",
        tile_images=True,
        mask_type="depth-aware",
    )
    for _ in range(10):
        example = dataset[0]

        image_HW3 = example["image"].permute(1, 2, 0)
        distance_with_colormap_HW3 = apply_colormap(
            example["distance"].permute(1, 2, 0), ColormapOptions(normalize=True)
        )
        depth_with_colormap_HW3 = apply_colormap(example["depth"].permute(1, 2, 0), ColormapOptions(normalize=True))
        rendered_image_HW3 = example["rendered_image"].permute(1, 2, 0)
        mask_HW3 = example["mask"].permute(1, 2, 0).repeat(1, 1, 3)

        import mediapy

        im = torch.cat([image_HW3, mask_HW3, rendered_image_HW3, depth_with_colormap_HW3], dim=1)
        mediapy.write_image("image.png", im.cpu())

        import pdb

        pdb.set_trace()
