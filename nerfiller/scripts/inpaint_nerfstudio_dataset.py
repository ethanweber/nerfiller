import json
import shutil
from pathlib import Path

import mediapy
import torch
import tyro

from nerfiller.inpaint.rgb_inpainter import RGBInpainter
from nerfiller.inpaint.lama_inpainter import LaMaInpainter
from nerfiller.nerf.dataset_utils import parse_nerfstudio_frame
from nerfiller.utils.image_utils import get_inpainted_image_row
from nerfiller.utils.camera_utils import rescale_intrinsics

from nerfiller.configs.inpaint import InpaintConfig, AnnotatedBaseConfigUnion
from datetime import datetime
from nerfiller.utils.diff_utils import register_extended_attention
from nerfiller.utils.mask_utils import downscale_mask
import math


def main(
    config: InpaintConfig,
):
    """
    Inpaint a Nerfstudio dataset where the masks == 0.
    """

    if config.method_name == "individual-lama":
        rgb_inpainter = LaMaInpainter(device=config.device, model_path=Path("data/models/big-lama"))
    else:
        # Load the inpainting module.
        rgb_inpainter = RGBInpainter(
            half_precision_weights=config.half_precision_weights,
            lora_model_path=config.lora_model_path,
            device=config.device,
            vae_device=config.vae_device,
        )

        if config.text_guidance_scale != 0.0:
            assert config.prompt != "", "You need to set an actual prompt to use this method."

        # Process the text prompts.
        text_embeddings = rgb_inpainter.compute_text_embeddings(config.prompt, config.negative_prompt)

    if config.use_expanded_attention:
        register_extended_attention(rgb_inpainter.unet)

    # Setup the modules for guidance.
    # multiview_metric = ReprojectMetric(lossfeatmult=1.0)
    # feature_extractor = SuperPointExtractor(device=config.device)
    # TODO: make sure feature_extractor is half precision with half_precision_weights

    # Copy the original dataset besides the images, which we will inpaint.
    output_folder = (
        Path(str(config.nerfstudio_dataset) + "-" + "inpaint")
        / str(config.method_name)
        / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )
    output_folder.mkdir(parents=True)
    shutil.copytree(config.nerfstudio_dataset / "images", output_folder / "original_images")
    shutil.copytree(config.nerfstudio_dataset / "masks", output_folder / "original_masks")
    shutil.copytree(config.nerfstudio_dataset / "depth", output_folder / "depth")
    shutil.copytree(config.nerfstudio_dataset / "masks", output_folder / "masks")
    shutil.copy(config.nerfstudio_dataset / "transforms.json", output_folder / "transforms.json")
    (output_folder / "images").mkdir(parents=True)
    (output_folder / "inpaint").mkdir(parents=True)

    f = open(config.nerfstudio_dataset / "transforms.json")
    transforms = json.load(f)
    f.close()
    num_images = len(transforms["frames"])

    if config.randomize_image_order:
        indices = torch.randperm(num_images)
    else:
        indices = torch.arange(num_images)

    padded_num_images = config.chunk_size * math.ceil(num_images / config.chunk_size)
    if num_images != padded_num_images:
        indices = torch.cat([indices, indices[: padded_num_images - num_images]])

    for i in range(0, padded_num_images - config.chunk_size + 1, config.new_size):
        images = []
        masks = []
        depths = []
        Ks = []
        c2ws = []
        for j in range(i, i + config.chunk_size):
            if i == 0 or j >= (i + config.chunk_size - config.new_size):
                # new stuff to inpaint
                image, depth, mask, c2w, K = parse_nerfstudio_frame(
                    transforms,
                    config.nerfstudio_dataset,
                    indices[j],
                    depth_max=config.depth_max,
                    device=config.device,
                )
            else:
                # old stuff already inpainted
                image, depth, mask, c2w, K = parse_nerfstudio_frame(
                    transforms,
                    output_folder,
                    indices[j],
                    depth_max=config.depth_max,
                    device=config.device,
                )
            images.append(image)
            masks.append(mask)
            depths.append(depth)
            Ks.append(K)
            c2ws.append(c2w)
        images = torch.cat(images)
        masks = torch.cat(masks)
        depths = torch.cat(depths)
        Ks = torch.cat(Ks)
        c2ws = torch.cat(c2ws)

        # generator = [
        #     torch.Generator(device=config.device).manual_seed(int(indices[j])) for j in range(i, i + config.chunk_size)
        # ]

        generator = None

        image = torch.nn.functional.interpolate(images, scale_factor=config.scale_factor, mode="bilinear")
        depth = torch.nn.functional.interpolate(depths, scale_factor=config.scale_factor, mode="bilinear")
        mask = downscale_mask(
            masks,
            scale_factor=config.scale_factor,
            dilate_iters=config.dilate_iters,
            dilate_kernel_size=config.dilate_kernel_size,
        )

        if config.method_name == "individual-lama":
            imagein = rgb_inpainter.get_image(image=image, mask=mask)
        else:
            enable_gradient = config.num_guidance_steps > 0 and len(config.guidance_steps) > 0
            with torch.enable_grad() if enable_gradient else torch.no_grad():
                imagein = rgb_inpainter.get_image(
                    text_embeddings=text_embeddings,
                    image=image,
                    mask=mask,
                    depth=depth,
                    denoise_in_grid=config.denoise_in_grid,
                    multidiffusion_steps=config.multidiffusion_steps,
                    randomize_latents=config.randomize_latents,
                    text_guidance_scale=config.text_guidance_scale,
                    image_guidance_scale=config.image_guidance_scale,
                    num_inference_steps=config.num_inference_steps,
                    num_guidance_steps=config.num_guidance_steps,
                    classifier_guidance_scale=config.classifier_guidance_scale,
                    guidance_steps=config.guidance_steps,
                    multiview_guidance_scale=config.multiview_guidance_scale,
                    K=rescale_intrinsics(Ks, config.scale_factor, config.scale_factor),
                    c2w=c2ws,
                    output_folder=output_folder / "inpaint" / f"{i:06d}-{i+config.chunk_size:06d}"
                    if config.save_intermediates
                    else None,
                    show_multiview=False,
                    generator=generator,
                    use_decoder_approximation=config.use_decoder_approximation,
                )

        # TODO: use an upscaler here

        imagein = torch.nn.functional.interpolate(imagein, scale_factor=1 / config.scale_factor, mode="bilinear")
        imagein = torch.where(masks == 1, imagein, images)
        imageinrow = get_inpainted_image_row(image=images, mask=masks, inpainted_image=imagein, show_original=True)
        imageinrow_HW3 = torch.cat(list(imageinrow.permute(0, 2, 3, 1)), dim=1).detach().cpu()
        mediapy.write_image(
            output_folder / "inpaint" / f"{i:06d}-{i+config.chunk_size:06d}.png",
            imageinrow_HW3,
        )

        for j in range(i, i + config.chunk_size):
            imagein_HW3 = imagein[j - i].permute(1, 2, 0)
            mediapy.write_image(
                output_folder / "images" / f"image_{indices[j]:06d}.png",
                imagein_HW3.detach().cpu(),
            )
            mediapy.write_image(
                output_folder / "masks" / f"mask_{indices[j]:06d}.png",
                torch.ones_like(imagein_HW3[:, :, 0]).detach().cpu(),
            )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(AnnotatedBaseConfigUnion))


if __name__ == "__main__":
    entrypoint()
