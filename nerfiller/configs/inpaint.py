"""
Configs for inpainting.
"""

from dataclasses import dataclass, field
from nerfiller.utils.typing import *
import tyro


@dataclass
class InpaintConfig:
    method_name: str = "inpaint_method"
    """The method name."""
    nerfstudio_dataset: Path = Path("data/nerfstudio/billiards")
    """Path to the Nerfstudio Dataset to inpaint."""
    lora_model_path: Optional[Path] = None
    """Optional path to a lora checkpoint."""
    prompt: str = ""
    """Positive text prompt to use with text guidance."""
    negative_prompt: str = ""
    """Negative text prompt to use with text guidance."""
    text_guidance_scale: float = 0.0
    """Text guidance scale."""
    image_guidance_scale: float = 1.5
    """Image guidance scale."""
    depth_max: float = 100.0
    """TODO"""
    multidiffusion_steps: int = 4
    """How many noise predictions to average"""
    device: str = "cuda:0"
    vae_device: str = "cuda:0"
    # multiview_metric: MetricConfig = ReprojectMetricConfig
    # TODO
    # feature_extractor: ExtractorConfig = ColorExtractorConfig
    # TODO
    classifier_guidance_scale: float = 0.0
    num_guidance_steps: int = 50
    guidance_steps: List[int] = field(default_factory=lambda: [])
    multiview_guidance_scale: float = 0.0
    chunk_size: int = 40
    """How many images to process at once."""
    new_size: int = 20
    """How many new images to inpaint each iteration after the first chunk."""
    denoise_in_grid: bool = True
    """Whether or not to predict noise with 2x2 grid tiling."""
    randomize_latents: bool = True
    """Whether or not to randomize the latents ordering before each noise prediction."""
    num_inference_steps: int = 20
    """How many steps to use for denoising."""
    use_decoder_approximation: bool = False
    half_precision_weights: bool = True
    scale_factor: float = 0.5
    randomize_image_order: bool = True
    dilate_iters: int = 5
    dilate_kernel_size: int = 3
    use_expanded_attention: bool = False
    save_intermediates: bool = False
    """Whether to save diffusion intermediate results to an output folder."""


methods: Dict[str, InpaintConfig] = {}
descriptions = {
    "grid-prior": "Inpaint a Nerfstudio Dataset with the grid prior method.",
    "grid-prior-no-joint": "Inpaint a Nerfstudio Dataset with the grid prior method, but only 4 at a time.",
    "expanded-attention": "Inpaint a Nerfstudio Dataset with stable diffusion with expanded attention.",
    "individual-sd-image": "Inpaint a Nerfstudio Dataset with stable diffusion image CFG, each image independently.",
    "individual-sd-text": "Inpaint a Nerfstudio Dataset with stable diffusion text CFG, each image independently.",
    "individual-lama": "Inpaint a Nerfstudio Dataset with LaMa, each image independently",
}

methods["grid-prior"] = InpaintConfig(method_name="grid-prior")

methods["grid-prior-no-joint"] = InpaintConfig(
    method_name="grid-prior-no-joint",
    chunk_size=4,
    new_size=4,
)

methods["expanded-attention"] = InpaintConfig(
    method_name="expanded-attention",
    denoise_in_grid=False,
    multidiffusion_steps=1,
    scale_factor=1.0,
    chunk_size=5,
    new_size=5,
    use_expanded_attention=True,
)

methods["individual-sd-image"] = InpaintConfig(
    method_name="individual-sd-image",
    denoise_in_grid=False,
    multidiffusion_steps=1,
    scale_factor=1.0,
    chunk_size=8,
    new_size=8,
)

methods["individual-sd-text"] = InpaintConfig(
    method_name="individual-sd-text",
    denoise_in_grid=False,
    multidiffusion_steps=1,
    scale_factor=1.0,
    chunk_size=8,
    new_size=8,
    text_guidance_scale=7.5,
    image_guidance_scale=0.0,
)

methods["individual-lama"] = InpaintConfig(method_name="individual-lama", chunk_size=8, new_size=8, scale_factor=1.0)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[tyro.extras.subcommand_type_from_defaults(defaults=methods, descriptions=descriptions)]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
