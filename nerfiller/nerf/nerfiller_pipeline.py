from dataclasses import dataclass, field
from typing import Literal, Optional, Type
import torch

from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig, VanillaPipeline
from nerfiller.inpaint.rgb_inpainter import RGBInpainter, RGBInpainterXL
from nerfiller.inpaint.depth_inpainter import DepthInpainter
from nerfiller.inpaint.upscaler import Upscaler

from nerfstudio.utils import profiler

from nerfiller.utils.image_utils import (
    get_inpainted_image_row,
)


import mediapy


from nerfstudio.utils.rich_utils import Console
from nerfstudio.utils.colormaps import apply_colormap, ColormapOptions

from jaxtyping import Float
from torch import Tensor
from nerfiller.utils.mask_utils import downscale_mask

from nerfiller.utils.typing import *
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes

CONSOLE = Console()


@dataclass
class NeRFillerPipelineConfig(VanillaPipelineConfig):
    """The config for the NeRFiller pipeline."""

    _target: Type = field(default_factory=lambda: NeRFillerPipeline)

    patch_size: int = 32

    # inpaint args
    rgb_inpainter: str = "sd"
    rgb_inpaint_device: Optional[str] = "cuda:1"
    """device to put the rgb inpainting module on"""
    rgb_inpaint_vae_device: Optional[str] = None
    """device to put the vae inpainting module on. defaults to rgb inpaint device"""
    depth_inpaint_device: Optional[str] = "cuda:0"
    """device to put the depth inpainting module on"""
    upscale_device: Optional[str] = "cuda:0"
    """device to put the upscaler module on"""
    prompt: str = "highly detailed, 4K, hdr, sharp focus, image"
    """positive prompt for text-conditioned inpainting"""
    negative_prompt: str = ""
    """negative prompt for text-conditionied inpainting"""
    depth_method: Literal["zoedepth", "irondepth"] = "zoedepth"
    """which depth network to use for depth prediction or depth completion"""

    # sds
    use_sds: bool = False

    # du (dataset update) args
    use_du: bool = True
    """how often to update the dataset via inpainting. if 0, don't do dataset updating"""
    edit_rate: int = 1000
    """how often to make an edit"""
    edit_num: int = 40
    """number of images to edit at a time"""
    edit_iters: int = 30001
    """how many iterations until we stop making changes"""
    num_inference_steps: int = 20
    multidiffusion_steps: int = 1
    randomize_latents: bool = True
    randomize_within_grid: bool = False
    use_annealing: bool = True
    lower_bound: float = 0.4
    """Lower bound for diffusion timesteps to use for image editing"""
    upper_bound: float = 1.0
    """Upper bound for diffusion timesteps to use for image editing"""
    denoise_in_grid: bool = True
    dilate_iters: int = 5
    dilate_kernel_size: int = 3
    allow_camera_mismatch: bool = False
    tile_resolution: int = 256
    upscale: bool = False
    inpaint_chunk_size: Optional[int] = None

    render_all_rate: int = 5000

    reference_image: Path = Path("reference.png")

    lora_model_path: Optional[str] = None
    only_sample_from_latest: bool = True
    """Only sample rays from the latest inpaints."""
    inpaint_method: str = "inpaint"
    """Strategy for inpainting a batch of images."""
    text_guidance_scale: float = 0.0
    image_guidance_scale: float = 1.5
    inpaint_index_start: int = 0
    """We will edit images starting from this index and onward."""
    sds_loss_mult: float = 1.0
    sds_guidance_mult: float = 10.0
    sds_downscale_factor: int = 1


class NeRFillerPipeline(VanillaPipeline):
    """The pipeline for the NeRFiller method."""

    def __init__(
        self,
        config: NeRFillerPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler=grad_scaler)

        if test_mode != "val":
            # skip the rest of setup if we aren't going to train
            return

        self.grad_scaler = grad_scaler

        self.start_step = None
        self.num_train_images = len(self.datamanager.train_dataparser_outputs.image_filenames)

        self.load_training_modules()

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        self.trainer_base_dir = training_callback_attributes.trainer.base_dir
        return super().get_training_callbacks(training_callback_attributes)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: Optional[bool] = None):
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}

        pipeline_state = {key: value for key, value in state_dict.items() if not key.startswith("_model.")}

        if self.config.allow_camera_mismatch:
            # Don't set the weights for the appearance embedding
            # This sets the weights to be zero.
            key = "field.embedding_appearance.embedding.weight"
            model_state[key] = torch.zeros(self.model.field.embedding_appearance.embedding.weight.shape)

        try:
            self.model.load_state_dict(model_state, strict=True)
        except RuntimeError:
            if not strict:
                self.model.load_state_dict(model_state, strict=False)
            else:
                raise

        super().load_state_dict(pipeline_state, strict=False)

    def load_training_modules(self):
        """Load the modules."""

        # RGB and depth inpainting
        rgb_inpaint_device = (
            self.config.rgb_inpaint_device if self.config.rgb_inpaint_device is not None else self.device
        )
        rgb_inpaint_vae_device = (
            self.config.rgb_inpaint_vae_device if self.config.rgb_inpaint_vae_device is not None else rgb_inpaint_device
        )
        if self.config.rgb_inpainter == "sd":
            self.rgb_inpainter = RGBInpainter(
                device=rgb_inpaint_device,
                vae_device=rgb_inpaint_vae_device,
                lora_model_path=self.config.lora_model_path,
            )
        elif self.config.rgb_inpainter == "sdxl":
            self.rgb_inpainter = RGBInpainterXL(
                device=rgb_inpaint_device,
                vae_device=rgb_inpaint_vae_device,
                lora_model_path=self.config.lora_model_path,
            )
        depth_inpaint_device = (
            self.config.depth_inpaint_device if self.config.depth_inpaint_device is not None else self.device
        )
        self.depth_inpainter = DepthInpainter(depth_method=self.config.depth_method, device=depth_inpaint_device)

        # Upscaling
        if self.config.upscale:
            upscale_device = self.config.upscale_device if self.config.upscale_device is not None else self.device
            self.upscaler = Upscaler(device=upscale_device)

        # Process the prompt
        self.text_embeddings = self.rgb_inpainter.compute_text_embeddings(
            self.config.prompt, self.config.negative_prompt
        )
        # self.rgb_inpainter.remove_pipe()

        # self.sd_version = "1-5"
        # self.guidance_scale = 20
        # self.diffusion_device = self.device
        # self._diffusion_model = StableDiffusion(self.diffusion_device, version=self.sd_version)
        # self.text_embedding = self._diffusion_model.get_text_embeds(self.config.prompt, self.config.negative_prompt)

        if self.config.inpaint_method == "reference":
            self.reference_image = (
                (torch.from_numpy(mediapy.read_image(self.config.datamanager.data / "reference.png")) / 255.0)
                .permute(2, 0, 1)[None]
                .to(self.device)
            )

    def render_and_save_images(self, image_indices: List[int], step: int):
        current_image = torch.cat(
            [
                self.datamanager.image_batch["image"][image_index].unsqueeze(dim=0).permute(0, 3, 1, 2)
                for image_index in image_indices
            ]
        )
        original_image = torch.cat(
            [
                self.datamanager.original_image_batch["image"][image_index].unsqueeze(dim=0).permute(0, 3, 1, 2)
                for image_index in image_indices
            ]
        )
        original_mask = torch.cat(
            [
                self.datamanager.original_image_batch["mask"][image_index].unsqueeze(dim=0).permute(0, 3, 1, 2)
                for image_index in image_indices
            ]
        )

        current_depth_colormap = torch.cat(
            [
                apply_colormap(
                    self.datamanager.image_batch["depth_image"][image_index],
                    ColormapOptions(normalize=True),
                )
                .unsqueeze(dim=0)
                .permute(0, 3, 1, 2)
                for image_index in image_indices
            ]
        )

        for idx, image_index in enumerate(image_indices):
            # save the current dataset images
            image_folder = self.trainer_base_dir / "dataset" / f"step_{step}" / "images"
            if not image_folder.exists():
                image_folder.mkdir(parents=True, exist_ok=True)
            filename = self.datamanager.train_dataset._dataparser_outputs.image_filenames[image_index].name
            mediapy.write_image(image_folder / filename, current_image.permute(0, 2, 3, 1)[idx].cpu())

            # save the current dataset depths
            image_folder = self.trainer_base_dir / "dataset" / f"step_{step}" / "depth"
            if not image_folder.exists():
                image_folder.mkdir(parents=True, exist_ok=True)
            filename = self.datamanager.train_dataset._dataparser_outputs.image_filenames[image_index].name
            mediapy.write_image(
                image_folder / filename,
                current_depth_colormap.permute(0, 2, 3, 1)[idx].cpu(),
            )

            # save the renders
            rendered_image = self.render_image(
                image_index,
                original_image=original_image[idx : idx + 1],
                original_mask=original_mask[idx : idx + 1],
            )
            image_folder = self.trainer_base_dir / "renders" / f"step_{step}" / "images"
            if not image_folder.exists():
                image_folder.mkdir(parents=True, exist_ok=True)
            filename = self.datamanager.train_dataset._dataparser_outputs.image_filenames[image_index].name
            mediapy.write_image(image_folder / filename, rendered_image.permute(0, 2, 3, 1)[0].cpu())

            # save the depth renders
            rendered_depth = apply_colormap(
                self.render_depth(
                    image_index,
                )[
                    0
                ].permute(1, 2, 0),
                ColormapOptions(normalize=True),
            ).permute(2, 0, 1)[None]
            image_folder = self.trainer_base_dir / "renders" / f"step_{step}" / "depth"
            if not image_folder.exists():
                image_folder.mkdir(parents=True, exist_ok=True)
            filename = self.datamanager.train_dataset._dataparser_outputs.image_filenames[image_index].name
            mediapy.write_image(image_folder / filename, rendered_depth.permute(0, 2, 3, 1)[0].cpu())

    # @profile
    def render_image(
        self,
        image_index: int,
        original_image: Optional[Float[Tensor, "1 3 H W"]] = None,
        original_mask: Optional[Float[Tensor, "1 1 H W"]] = None,
        keep_gradient: bool = False,
        downscale_factor: int = 1,
    ):
        """Render an image at a camera location in the dataset, specified by image_index.
        This needs access to both the datamanager and the model.

        Args:
            image_index: The index of the image where you want to render from.
            keep_gradient: Whether to keep the gradient or not. Faster and more memory efficient if False.
            downscale_factor: How much to downscale from the normal resolution.
        """

        # get current camera, include camera transforms from original optimizer
        camera_transforms = self.model.camera_optimizer(torch.tensor([image_index])).to(self.device)
        current_camera = self.datamanager.train_dataparser_outputs.cameras[image_index].to(self.device)
        height, width = current_camera.height.item(), current_camera.width.item()
        if downscale_factor != 1:
            current_camera.rescale_output_resolution(1.0 / downscale_factor)
        current_ray_bundle = current_camera.generate_rays(
            torch.tensor(list(range(1))).unsqueeze(-1),
            camera_opt_to_camera=camera_transforms,
        )

        use_mask = original_image is not None and original_mask is not None
        if use_mask:
            resized_image = torch.nn.functional.interpolate(
                original_image,
                size=(height // downscale_factor, width // downscale_factor),
                mode="bilinear",
            )
            resized_mask = torch.nn.functional.interpolate(
                original_mask,
                size=(height // downscale_factor, width // downscale_factor),
                mode="bilinear",
            )
            missing_mask = resized_mask[0, 0] != 1.0
            if missing_mask.sum() == 0:
                return original_image
            crb = current_ray_bundle[missing_mask]
            if keep_gradient:
                missing_camera_outputs = self.model.forward(crb.reshape(-1))
                missing_rgb = missing_camera_outputs["rgb"]
            else:
                missing_camera_outputs = self.model.get_outputs_for_camera_ray_bundle(crb.reshape((1, -1)))
                missing_rgb = missing_camera_outputs["rgb"].squeeze(0)

            rendered_image = resized_image
            rendered_image.permute(0, 2, 3, 1)[0][missing_mask] = missing_rgb
        else:
            if keep_gradient:
                raise ValueError("need to implement this")
            else:
                rendered_image = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)["rgb"].permute(
                    2, 0, 1
                )[None]

        rendered_image = torch.nn.functional.interpolate(rendered_image, size=(height, width), mode="bilinear")

        return rendered_image

    def render_depth(
        self,
        image_index: int,
        keep_gradient: bool = False,
        downscale_factor: int = 1,
    ):
        """Render an depth at a camera location in the dataset, specified by image_index.
        This needs access to both the datamanager and the model.

        Args:
            image_index: The index of the image where you want to render from.
            keep_gradient: Whether to keep the gradient or not. Faster and more memory efficient if False.
            downscale_factor: How much to downscale from the normal resolution.
        """

        # get current camera, include camera transforms from original optimizer
        camera_transforms = self.model.camera_optimizer(torch.tensor([image_index])).to(self.device)
        current_camera = self.datamanager.train_dataparser_outputs.cameras[image_index].to(self.device)
        height, width = current_camera.height.item(), current_camera.width.item()
        if downscale_factor != 1:
            current_camera.rescale_output_resolution(1.0 / downscale_factor)
        current_ray_bundle = current_camera.generate_rays(
            torch.tensor(list(range(1))).unsqueeze(-1),
            camera_opt_to_camera=camera_transforms,
        )

        if keep_gradient:
            raise ValueError("need to implement this")
        else:
            rendered_depth = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)["depth"].permute(2, 0, 1)[
                None
            ]

        rendered_depth = torch.nn.functional.interpolate(rendered_depth, size=(height, width), mode="bilinear")

        return rendered_depth

    def update_depth_image(self, image_indices: List[int], step: int):
        """Update the depth of an image."""

        image = torch.cat(
            [
                self.datamanager.image_batch["image"][image_index].unsqueeze(dim=0).permute(0, 3, 1, 2)
                for image_index in image_indices
            ]
        )

        CONSOLE.print(f"[green]Estimating depth for {len(image_indices)} images")
        for idx, image_index in enumerate(image_indices):
            with torch.no_grad():
                depth_image = self.depth_inpainter.get_depth(
                    image=image[idx : idx + 1].to(self.depth_inpainter.device)
                ).to(self.device)
            self.datamanager.image_batch["depth_image"][image_index] = depth_image[0].permute(1, 2, 0)

    def get_inpaint_info(
        self,
        image_indices: List[int],
        step: int,
        keep_gradient=False,
        downscale_factor=1,
    ):
        """Return the information needed to inpaint a set of images."""
        original_image = torch.cat(
            [
                self.datamanager.original_image_batch["image"][image_index].unsqueeze(dim=0).permute(0, 3, 1, 2)
                for image_index in image_indices
            ]
        )
        original_mask = torch.cat(
            [
                self.datamanager.original_image_batch["mask"][image_index].unsqueeze(dim=0).permute(0, 3, 1, 2)
                for image_index in image_indices
            ]
        )
        rendered_image = torch.cat(
            [
                self.render_image(
                    image_index,
                    original_image=original_image[idx : idx + 1],
                    original_mask=original_mask[idx : idx + 1],
                    keep_gradient=keep_gradient,
                    downscale_factor=downscale_factor,
                )
                for idx, image_index in enumerate(image_indices)
            ]
        )
        starting_image = torch.nn.functional.interpolate(
            rendered_image,
            size=(self.config.tile_resolution, self.config.tile_resolution),
            mode="bilinear",
        )
        image = torch.nn.functional.interpolate(
            original_image,
            size=(self.config.tile_resolution, self.config.tile_resolution),
            mode="bilinear",
        )
        mask = 1 - original_mask
        mask = downscale_mask(
            mask,
            size=(self.config.tile_resolution, self.config.tile_resolution),
            dilate_iters=self.config.dilate_iters,
            dilate_kernel_size=self.config.dilate_kernel_size,
        )
        depth = torch.zeros_like(mask)

        if self.config.use_annealing:
            starting_lower_bound = self.config.upper_bound - max(
                min(((step - self.start_step) / self.config.edit_iters), 1), 0
            ) * (self.config.upper_bound - self.config.lower_bound)
            starting_upper_bound = starting_lower_bound
        else:
            starting_lower_bound = (
                torch.rand((1,)) * (self.config.lower_bound - self.config.upper_bound) + self.config.upper_bound
            )
            starting_lower_bound = float(starting_lower_bound)
            starting_upper_bound = starting_lower_bound

        return (
            original_image,
            original_mask,
            rendered_image,
            starting_image,
            image,
            mask,
            depth,
            starting_lower_bound,
            starting_upper_bound,
        )

    def inpaint_depth_guided(
        self,
        image,
        mask,
        depth,
        starting_image,
        starting_lower_bound,
        starting_upper_bound,
    ):
        """Update a batch of images defeined by the image indices.
        Inpaints the first image and guides the remaining views to have the same texture.
        """

        # inpaint first image

        # inpaint with guidance

        edited_image = None
        return edited_image

    def inpaint_reference_guided(
        self,
        image,
        mask,
        depth,
        starting_image,
        starting_lower_bound,
        starting_upper_bound,
    ):
        """Update a batch of images defeined by the image indices.
        Inpaint with gridding technique where reference image is always present in each grid.
        """

        # add reference to grids
        reference_image = torch.nn.functional.interpolate(
            self.reference_image,
            size=(self.config.tile_resolution, self.config.tile_resolution),
            mode="bilinear",
        )
        num_images = image.shape[0]
        newstarting_image = []
        newimage = []
        newmask = []
        newdepth = []
        count = 0
        assert num_images % 3 == 0
        for i in range(0, num_images + int(num_images / 3)):
            if i % 4 == 0:
                newstarting_image.append(reference_image.to(starting_image.device))
                newimage.append(reference_image.to(image.device))
                newmask.append(torch.zeros_like(reference_image[:, 0:1]).to(mask.device))
                newdepth.append(torch.zeros_like(reference_image[:, 0:1]).to(depth.device))
            else:
                newstarting_image.append(starting_image[count : count + 1])
                newimage.append(image[count : count + 1])
                newmask.append(mask[count : count + 1])
                newdepth.append(depth[count : count + 1])
                count += 1
        starting_image = torch.cat(newstarting_image)
        image = torch.cat(newimage)
        mask = torch.cat(newmask)
        depth = torch.cat(newdepth)

        # inpaint
        edited_image = self.inpaint(
            image,
            mask,
            depth,
            starting_image,
            starting_lower_bound,
            starting_upper_bound,
        )

        # unpack
        edited_image = edited_image[torch.arange(len(image))[torch.arange(len(image)) % 4 != 0]]
        return edited_image

    def inpaint(
        self,
        image,
        mask,
        depth,
        starting_image,
        starting_lower_bound,
        starting_upper_bound,
    ):
        """Inpaint an image."""
        if starting_lower_bound == 0.0 and starting_upper_bound == 0.0:
            return starting_image.to(self.device)
        if self.config.inpaint_chunk_size is None:
            inpaint_chunk_size = len(image)
        else:
            inpaint_chunk_size = self.config.inpaint_chunk_size
        edited_image = []
        for i in range(0, len(image), inpaint_chunk_size):
            print(i, i + inpaint_chunk_size)
            ei = self.rgb_inpainter.get_image(
                text_embeddings=self.text_embeddings,
                starting_image=starting_image[i : i + inpaint_chunk_size].to(self.rgb_inpainter.device),
                starting_lower_bound=starting_lower_bound,
                starting_upper_bound=starting_upper_bound,
                image=image[i : i + inpaint_chunk_size].to(self.rgb_inpainter.device),
                mask=mask[i : i + inpaint_chunk_size].to(self.rgb_inpainter.device),
                depth=depth[i : i + inpaint_chunk_size].to(self.rgb_inpainter.device),
                denoise_in_grid=self.config.denoise_in_grid,
                multidiffusion_steps=self.config.multidiffusion_steps,
                randomize_latents=self.config.randomize_latents,
                randomize_within_grid=self.config.randomize_within_grid,
                text_guidance_scale=self.config.text_guidance_scale,
                image_guidance_scale=self.config.image_guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
            ).to(self.device)
            edited_image.append(ei)
        edited_image = torch.cat(edited_image)

        # Upscale inpainted images
        if self.config.upscale:
            upscaled_images = []
            for idx, image_index in enumerate(image_indices):
                ei = edited_image[idx : idx + 1].to(self.upscaler.device).to(torch.float32)
                upscaled_image = self.upscaler.upsample(ei).to(self.device).to(torch.float32)
                upscaled_images.append(upscaled_image)
            edited_image = torch.cat(upscaled_images)

        return edited_image

    def update_dataset_image(
        self,
        image_indices: List[int],
        step: int,
        text_guidance_scale: float = 0.0,
        image_guidance_scale: float = 1.5,
        method: str = "inpaint",
    ):
        """Update a batch of images defined by the image indices.
        Inpaint with gridding technique.
        """

        # Prepare for inpainting
        (
            original_image,
            original_mask,
            rendered_image,
            starting_image,
            image,
            mask,
            depth,
            starting_lower_bound,
            starting_upper_bound,
        ) = self.get_inpaint_info(image_indices, step)

        print(starting_upper_bound, starting_lower_bound)

        # Inpaint the images
        if method == "inpaint":
            edited_image = self.inpaint(
                image,
                mask,
                depth,
                starting_image,
                starting_lower_bound,
                starting_upper_bound,
            )
        elif method == "reference":
            edited_image = self.inpaint_reference_guided(
                image,
                mask,
                depth,
                starting_image,
                starting_lower_bound,
                starting_upper_bound,
            )
        elif method == "depth":
            edited_image = self.inpaint_depth_guided(
                image,
                mask,
                depth,
                starting_image,
                starting_lower_bound,
                starting_upper_bound,
            )
        else:
            raise ValueError(f"Inpaint method {method} not implemented")

        # we use bilinear interpolation on the mask to include more of the inpainted image than the ground truth
        edited_mask = torch.nn.functional.interpolate(mask, size=original_image.shape[-2:], mode="bilinear")
        edited_mask = (edited_mask != 0).float()
        edited_image = torch.nn.functional.interpolate(edited_image, size=original_image.shape[-2:], mode="bilinear")
        edited_image = torch.where(edited_mask == 0, original_image, edited_image)

        # save the edits
        image_folder = self.trainer_base_dir / "edits" / f"step_{step}" / "images"
        if not image_folder.exists():
            image_folder.mkdir(parents=True, exist_ok=True)
        for idx, image_index in enumerate(image_indices):
            filename = self.datamanager.train_dataset._dataparser_outputs.image_filenames[image_index].name
            imageinrow = get_inpainted_image_row(
                image=rendered_image[idx : idx + 1],
                mask=edited_mask[idx : idx + 1],
                inpainted_image=edited_image[idx : idx + 1],
                show_original=True,
            )
            mediapy.write_image(image_folder / filename, imageinrow.permute(0, 2, 3, 1)[0].cpu())

        # Update the dataset
        for idx, image_index in enumerate(image_indices):
            m = original_mask.permute(0, 2, 3, 1)[idx]
            self.datamanager.image_batch["image"][image_index] = edited_image.permute(0, 2, 3, 1)[idx]
            self.datamanager.image_batch["mask"][image_index] = torch.ones_like(m)

    def get_sds_loss(self, image_indices: List[int], step: int):
        """Get the SDS loss for a batch of images."""

        assert not self.config.denoise_in_grid, "SDS doesn't support gridding at the moment"
        assert not self.config.randomize_latents, "SDS doesn't support randomizing latents"
        assert not self.config.randomize_within_grid, "SDS doesn't support randomizing within grid"

        # Prepare for inpainting
        (
            original_image,
            original_mask,
            rendered_image,
            starting_image,
            image,
            mask,
            depth,
            starting_lower_bound,
            starting_upper_bound,
        ) = self.get_inpaint_info(
            image_indices,
            step,
            keep_gradient=True,
            downscale_factor=self.config.sds_downscale_factor,
        )

        assert starting_upper_bound <= 0.98, "Don't go below this"

        if self.config.inpaint_chunk_size is None:
            inpaint_chunk_size = len(image)
        else:
            inpaint_chunk_size = self.config.inpaint_chunk_size

        sds_loss = 0.0
        for i in range(0, len(image), inpaint_chunk_size):
            loss = self.config.sds_loss_mult * self.rgb_inpainter.sds_loss(
                text_embeddings=self.text_embeddings,
                starting_image=starting_image[i : i + inpaint_chunk_size].to(self.rgb_inpainter.device),
                starting_lower_bound=starting_lower_bound,
                starting_upper_bound=starting_upper_bound,
                image=image[i : i + inpaint_chunk_size].to(self.rgb_inpainter.device),
                mask=mask[i : i + inpaint_chunk_size].to(self.rgb_inpainter.device),
                text_guidance_scale=self.config.sds_guidance_mult * self.config.text_guidance_scale,
                image_guidance_scale=self.config.sds_guidance_mult * self.config.image_guidance_scale,
            ).to(self.device)
            sds_loss += loss
        return sds_loss

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        loss_dict = {}
        self.model.step = step
        self.model.trainer_base_dir = self.trainer_base_dir

        if self.start_step is None:
            self.start_step = step
            self.model.start_step = step

        # Typical Nerfacto losses (w/ whatever modifications we made in the model file)
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        if self.config.datamanager.camera_optimizer is not None:
            camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            if camera_opt_param_group in self.datamanager.get_param_groups():
                # Report the camera optimization metrics
                metrics_dict["camera_opt_translation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
                )
                metrics_dict["camera_opt_rotation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
                )

        if self.config.use_du or self.config.use_sds:
            image_indices = None
            if (step - self.start_step) % self.config.edit_rate == 0:
                edit_num = (
                    self.config.edit_num
                    if self.config.edit_num != -1
                    else (self.num_train_images - self.config.inpaint_index_start)
                )
                image_indices = [
                    self.config.inpaint_index_start + int(x)
                    for x in torch.randperm(self.num_train_images - self.config.inpaint_index_start)[:edit_num]
                ]

        # Inpaint Iterative Dataset Update
        if self.config.use_du and image_indices:
            self.update_dataset_image(image_indices, step, method=self.config.inpaint_method)
            if self.model.config.use_depth_ranking and (step - self.start_step >= self.model.config.start_depth_loss):
                self.update_depth_image(image_indices, step)
            if self.config.only_sample_from_latest:
                self.datamanager.sample_image_indices = image_indices

        # SDS
        if self.config.use_sds and image_indices:
            loss_dict["loss_sds"] = self.get_sds_loss(image_indices, step)

        # Render all the training images
        if self.config.render_all_rate != 0 and (step - self.start_step) % self.config.render_all_rate == 0:
            CONSOLE.print("[green]Rending all the training images...")
            self.render_and_save_images(torch.arange(self.num_train_images), step)

        loss_dict.update(self.model.get_loss_dict(model_outputs, batch, metrics_dict))
        return model_outputs, loss_dict, metrics_dict
