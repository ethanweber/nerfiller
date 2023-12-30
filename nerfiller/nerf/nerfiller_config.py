"""
Define the nerfiller config.
"""

from __future__ import annotations


from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.data.pixel_samplers import (
    PatchPixelSamplerConfig,
)
from nerfiller.nerf.nerfiller_model import NeRFillerModelConfig
from nerfiller.nerf.nerfiller_pipeline import NeRFillerPipelineConfig
from nerfiller.nerf.nerfiller_datamanager import (
    NeRFillerDataManagerConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManagerConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.configs.method_configs import method_configs

PATCH_SIZE = 32
TRAIN_RAYS = 8192 * 4
TEXTURE_REFINEMENT_PATCH_SIZE = 32

import copy
from nerfiller.utils.image_utils import Colors

# nerfacto method with our modifications
nerfacto_nerfiller = copy.deepcopy(method_configs["nerfacto"])
nerfacto_nerfiller.method_name = "nerfacto-nerfiller"
nerfacto_nerfiller.pipeline.datamanager = VanillaDataManagerConfig(
    dataparser=NerfstudioDataParserConfig(),
    train_num_rays_per_batch=4096,
    eval_num_rays_per_batch=4096,
)
nerfacto_nerfiller_config = MethodSpecification(
    nerfacto_nerfiller,
    description="Nerfacto method with our modifications.",
)

# nerfacto method for visualization
nerfacto_nerfiller_visualize = copy.deepcopy(nerfacto_nerfiller)
nerfacto_nerfiller_visualize.method_name = "nerfacto-nerfiller-visualize"
nerfacto_nerfiller_visualize.pipeline.datamanager.pixel_sampler.ignore_mask = True
nerfacto_nerfiller_visualize.pipeline.datamanager.dataparser.mask_color = Colors.NEON_PINK.value
nerfacto_nerfiller_visualize_config = MethodSpecification(
    nerfacto_nerfiller_visualize,
    description="Nerfacto method where we color the mask, useful for visualization.",
)

# our base method
grid_prior_du = TrainerConfig(
    method_name="grid-prior-du",
    project_name="nerfiller-nerf",
    steps_per_eval_batch=0,
    steps_per_eval_image=0,
    steps_per_eval_all_images=0,
    steps_per_save=2000,
    max_num_iterations=30001,
    mixed_precision=True,
    pipeline=NeRFillerPipelineConfig(
        patch_size=PATCH_SIZE,
        datamanager=NeRFillerDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            pixel_sampler=PatchPixelSamplerConfig(patch_size=PATCH_SIZE),
            train_num_rays_per_batch=TRAIN_RAYS,
            eval_num_rays_per_batch=4096,
        ),
        model=NeRFillerModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
            patch_size=PATCH_SIZE,
            use_depth_ranking=True,  # use depth loss
            start_depth_loss=3000,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15, default_composite_depth=False, websocket_port=8081),
    vis="viewer",
)
grid_prior_du_config = MethodSpecification(
    grid_prior_du,
    description="The grid prior + dataset update method.",
)

grid_prior_du_no_depth = copy.deepcopy(grid_prior_du)
grid_prior_du_no_depth.method_name = "grid-prior-du-no-depth"
grid_prior_du_no_depth.pipeline.model.use_depth_ranking = False
grid_prior_du_no_depth_config = MethodSpecification(
    grid_prior_du_no_depth,
    description="The grid prior + dataset update method, without depth supervision.",
)

grid_prior_du_random_noise = copy.deepcopy(grid_prior_du)
grid_prior_du_random_noise.method_name = "grid-prior-du-random-noise"
grid_prior_du_random_noise.pipeline.use_annealing = False
grid_prior_du_random_noise.pipeline.lower_bound = 0.02
grid_prior_du_random_noise.pipeline.upper_bound = 0.98
grid_prior_du_random_noise_config = MethodSpecification(
    grid_prior_du_random_noise,
    description="The grid prior + dataset update method, with a random noise schedule.",
)

grid_prior_du_reference_config = MethodSpecification(
    TrainerConfig(
        method_name="grid-prior-du-reference",
        project_name="nerfiller-nerf",
        steps_per_eval_batch=0,
        steps_per_eval_image=0,
        steps_per_eval_all_images=0,
        steps_per_save=2000,
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=NeRFillerPipelineConfig(
            patch_size=PATCH_SIZE,
            datamanager=NeRFillerDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                pixel_sampler=PatchPixelSamplerConfig(patch_size=PATCH_SIZE),
                train_num_rays_per_batch=TRAIN_RAYS,
                eval_num_rays_per_batch=4096,
            ),
            model=NeRFillerModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                patch_size=PATCH_SIZE,
            ),
            inpaint_method="reference",
            multidiffusion_steps=8,
            edit_rate=1000,
            edit_num=30,
            randomize_latents=False,
            randomize_within_grid=True,
            only_sample_from_latest=False,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(
            num_rays_per_chunk=1 << 15,
            default_composite_depth=False,
            websocket_port=8081,
        ),
        vis="viewer",
    ),
    description="The grid prior + dataset update method.",
)

individual_inpaint_du_config = MethodSpecification(
    TrainerConfig(
        method_name="individual-inpaint-du",
        project_name="nerfiller-nerf",
        steps_per_eval_batch=0,
        steps_per_eval_image=0,
        steps_per_eval_all_images=0,
        steps_per_save=2000,
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=NeRFillerPipelineConfig(
            patch_size=PATCH_SIZE,
            datamanager=NeRFillerDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                pixel_sampler=PatchPixelSamplerConfig(patch_size=PATCH_SIZE),
                train_num_rays_per_batch=TRAIN_RAYS,
                eval_num_rays_per_batch=4096,
            ),
            model=NeRFillerModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                patch_size=PATCH_SIZE,
            ),
            tile_resolution=512,
            inpaint_chunk_size=10,
            denoise_in_grid=False,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(
            num_rays_per_chunk=1 << 15,
            default_composite_depth=False,
            websocket_port=8081,
        ),
        vis="viewer",
    ),
    description="Inpaint each image individually + dataset update method.",
)

individual_inpaint_sds_config = MethodSpecification(
    TrainerConfig(
        method_name="individual-inpaint-sds",
        project_name="nerfiller-nerf",
        steps_per_eval_batch=0,
        steps_per_eval_image=0,
        steps_per_eval_all_images=0,
        steps_per_save=2000,
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=NeRFillerPipelineConfig(
            patch_size=PATCH_SIZE,
            datamanager=NeRFillerDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                pixel_sampler=PatchPixelSamplerConfig(patch_size=PATCH_SIZE),
                train_num_rays_per_batch=TRAIN_RAYS,
                eval_num_rays_per_batch=4096,
            ),
            model=NeRFillerModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                patch_size=PATCH_SIZE,
            ),
            use_du=False,
            use_sds=True,
            tile_resolution=512,
            sds_downscale_factor=2,
            inpaint_chunk_size=1,
            denoise_in_grid=False,
            randomize_latents=False,
            randomize_within_grid=False,
            upper_bound=0.98,
            edit_num=1,
            edit_rate=1,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(
            num_rays_per_chunk=1 << 15,
            default_composite_depth=False,
            websocket_port=8081,
        ),
        vis="viewer",
    ),
    description="Inpaint each image individually + SDS method.",
)

individual_inpaint_once_config = MethodSpecification(
    TrainerConfig(
        method_name="individual-inpaint-once",
        project_name="nerfiller-nerf",
        steps_per_eval_batch=0,
        steps_per_eval_image=0,
        steps_per_eval_all_images=0,
        steps_per_save=2000,
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=NeRFillerPipelineConfig(
            patch_size=PATCH_SIZE,
            datamanager=NeRFillerDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                pixel_sampler=PatchPixelSamplerConfig(patch_size=PATCH_SIZE),
                train_num_rays_per_batch=TRAIN_RAYS,
                eval_num_rays_per_batch=4096,
            ),
            model=NeRFillerModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                patch_size=PATCH_SIZE,
            ),
            use_du=False,
            edit_num=-1,
            edit_rate=1000000,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(
            num_rays_per_chunk=1 << 15,
            default_composite_depth=False,
            websocket_port=8081,
        ),
        vis="viewer",
    ),
    description="Don't do inpainting during NeRF training. The dataset should be inpainted once before running this method.",
)
