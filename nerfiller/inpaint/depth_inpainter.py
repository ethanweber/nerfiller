import logging

from nerfiller.utils.depth_utils import depth_to_distance, distance_to_depth
import torch

from nerfiller.utils.typing import *

class DepthInpainter:
    def __init__(
        self,
        max_depth: float = 10.0,
        tileX: bool = True,
        tileY: bool = False,
        depth_method: str = "zoedepth",
        device: str = "cuda:0",
    ):
        self.max_depth = max_depth
        self.tileX = tileX
        self.tileY = tileY
        self.depth_method = depth_method
        self.device = device
        self.configure()

    def configure(self) -> None:
        logging.info(f"Loading depth guidance ...")

        # setup the depth network

        # zoedepth
        if self.depth_method == "zoedepth":
            repo = "isl-org/ZoeDepth"
            self.zoe = torch.compile(torch.hub.load(repo, "ZoeD_NK", pretrained=True).to(self.device))

        # TODO: midas

        if self.depth_method == "midas":
            model_type = "DPT_Large"
            self.midas = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device)
            self.midas.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

    def get_depth(
        self,
        image: Float[Tensor, "B C H W"],
        rendered_depth: Optional[Float[Tensor, "B 1 H W"]] = None,
        overlapping_region_mask: Optional[Float[Tensor, "B 1 H W"]] = None,
        max_depth: Optional[float] = None,
        use_inverse=False,
        fov_x: Optional[float] = None,
        fov_y: Optional[float] = None,
    ):
        assert fov_x == fov_y
        batch_size, _, height, width = image.shape
        if self.depth_method != "zoedepth":
            assert batch_size == 1

        if self.depth_method == "zoedepth":
            predicted_depth = self.zoe.infer(image)
        elif self.depth_method == "midas":
            predicted_disparity = self.midas(image * 2 - 1).unsqueeze(1)
            predicted_depth = torch.where(predicted_disparity < 3, 0.0, 1 / predicted_disparity)
        else:
            raise NotImplementedError()

        if max_depth:
            predicted_depth[predicted_depth > max_depth] = 0.0

        return predicted_depth

    def get_distance(
        self,
        image: Float[Tensor, "B C H W"],
        fov_x: float,
        fov_y: float,
        rendered_distance: Optional[Float[Tensor, "B 1 H W"]] = None,
        overlapping_region_mask: Optional[Float[Tensor, "B 1 H W"]] = None,
        max_distance: Optional[float] = None,
        use_inverse=False,
    ):
        rendered_depth = distance_to_depth(rendered_distance, fov_x, fov_y)
        depth = self.get_depth(
            image,
            rendered_depth=rendered_depth,
            overlapping_region_mask=overlapping_region_mask,
            max_depth=max_distance,
            use_inverse=use_inverse,
            fov_x=fov_x,
            fov_y=fov_y,
        )
        distance = depth_to_distance(depth, fov_x, fov_y)

        if max_distance:
            distance[distance > max_distance] = 0.0
            overlapping_region_mask[distance > max_distance] = 0.0

        return distance
