# Lint as: python3
"""A basic shader which returns textures without any lighting.
"""
import torch
import torch.nn as nn

from pytorch3d.renderer.blending import (
    BlendParams,
)
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials


class TextureShader(nn.Module):
    """
    Basic shader which just returns the texels.
    """

    def __init__(self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = materials if materials is not None else Materials(device=device)
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)
        return texels[:, :, :, 0, :]
