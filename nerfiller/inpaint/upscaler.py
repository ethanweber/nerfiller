from diffusers import DDIMScheduler, StableDiffusionUpscalePipeline
import torch

from nerfiller.utils.typing import *


class Upscaler:
    def __init__(
        self,
        device: str = "cuda:0",
    ):
        self.device = device
        # load model and scheduler
        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            model_id, revision="fp16", torch_dtype=torch.float16
        )
        self.pipeline = self.pipeline.to(self.device)
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)

    @torch.cuda.amp.autocast(enabled=False)
    def upsample(
        self,
        image: Float[Tensor, "B 3 H W"],
        num_inference_steps: int = 20,
        noise_level: int = 20,
    ):
        batch_size = image.shape[0]
        prompt = [""] * batch_size
        upscaled_image = self.pipeline(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            noise_level=noise_level,
            output_type="pt",
        ).images
        return upscaled_image
