import os
from pathlib import Path

import yaml
from omegaconf import OmegaConf
import torch


from nerfiller.inpaint.saicinpainting.training.trainers import load_checkpoint

from nerfiller.utils.typing import *


class LaMaInpainter:
    """LaMa inpainter model."""

    def __init__(self, device: str = "cuda:0", model_path: Path = Path("data/models/big-lama")):
        print(f"Loading LaMa inpainter ...")

        self.device = device

        train_config_path = os.path.join(model_path, "config.yaml")
        with open(train_config_path, "r") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"

        checkpoint_path = os.path.join(model_path, "models", "best.ckpt")

        self.model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location="cpu")
        self.model.freeze()
        self.model.to(self.device)

    def get_image(self, image: Float[Tensor, "B 3 H W"], mask: Float[Tensor, "B 1 H W"]):
        with torch.no_grad():
            batch = {}
            batch["image"] = image
            batch["mask"] = mask
            batch = self.model(batch)
        inpainted_image = batch["inpainted"]
        return inpainted_image
