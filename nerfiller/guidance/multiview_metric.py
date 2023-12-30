"""
Code for epipolar guidance.
"""

import mediapy
import torch
from kornia.geometry.epipolar import (
    compute_correspond_epilines,
    fundamental_from_projections,
)
from kornia.geometry.linalg import point_line_distance
from torchmetrics.functional import (
    pairwise_cosine_similarity,
)
from nerfiller.utils.draw_utils import (
    get_images_with_keypoints,
    get_images_with_lines,
)
from nerfiller.utils.depth_utils import (
    reproject,
)
from nerfiller.utils.camera_utils import (
    get_projection_matrix,
)
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap

from nerfiller.utils.typing import *


class MultiviewMetric(torch.nn.Module):
    """
    Computes multi-view consistency loss.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        features1: Float[Tensor, "B C H W"],
        features2: Float[Tensor, "B C H W"],
        image1: Float[Tensor, "B 3 Horig Worig"],
        image2: Float[Tensor, "B 3 Horig Worig"],
        depth1: Optional[Float[Tensor, "B 1 H W"]] = None,
        depth2: Optional[Float[Tensor, "B 1 H W"]] = None,
        mask1: Optional[Float[Tensor, "B 1 H W"]] = None,
        mask2: Optional[Float[Tensor, "B 1 H W"]] = None,
        K1: Optional[Float[Tensor, "B 3 3"]] = None,
        K2: Optional[Float[Tensor, "B 3 3"]] = None,
        c2w1: Optional[Float[Tensor, "B 3 4"]] = None,
        c2w2: Optional[Float[Tensor, "B 3 4"]] = None,
        output_folder: Optional[Path] = None,
        suffix: str = "",
        show: bool = False,
        display_height: int = 512,
    ):
        pass


class MatchingMetric(MultiviewMetric):
    """
    Computes a loss to encourage the depth to give good matches.
    """

    def __init__(
        self,
        lossfeatmult: float = 1.0,
        lossdistmult: float = 1.0,
        sigma_scalar: float = 1.0,
        height_scalar: float = 1.0,
        keypoint_size: int = 10,
        line_width: int = 4,
        eps: float = 1e-6,
        thresh: float = 0.018,
    ):
        super().__init__()
        self.sigma_scalar = sigma_scalar
        self.height_scalar = height_scalar
        self.lossfeatmult = lossfeatmult
        self.lossdistmult = lossdistmult
        self.keypoint_size = keypoint_size
        self.line_width = line_width
        self.eps = eps
        self.thresh = thresh

    def compute_matches(
        self,
        features1: Float[Tensor, "B C H W"],
        features2: Float[Tensor, "B C H W"],
        K1: Float[Tensor, "B 3 3"],
        K2: Float[Tensor, "B 3 3"],
        c2w1: Float[Tensor, "B 3 4"],
        c2w2: Float[Tensor, "B 3 4"],
        output_folder: Optional[Path] = None,
        suffix: str = "",
        show: bool = False,
        display_height: int = 512,
    ):
        P1 = get_projection_matrix(K1, c2w1)
        P2 = get_projection_matrix(K2, c2w2)

        c1_2 = []
        c2_1 = []
        for b in range(BS):
            c1_2_b = pairwise_cosine_similarity(
                features1[b].permute(1, 2, 0).view(-1, C) + self.eps,
                features2[b].permute(1, 2, 0).view(-1, C) + self.eps,
            ).view(H, W, H, W)
            c2_1_b = c1_2_b.clone().permute(2, 3, 0, 1)
            c1_2.append(c1_2_b)
            c2_1.append(c2_1_b)
        c1_2 = torch.stack(c1_2)
        c2_1 = torch.stack(c2_1)

        assert not c1_2.isnan().any()
        assert not c2_1.isnan().any()

        F_mat = fundamental_from_projections(P1, P2)
        e2_1 = compute_correspond_epilines(pts.view(BS, -1, 2), F_mat).view(BS, H, W, 3)
        e1_2 = compute_correspond_epilines(pts.view(BS, -1, 2), F_mat.permute(0, 2, 1)).view(BS, H, W, 3)

        def get_lines(E):
            # E of shape [..., 3] with a, b, c
            # output is [..., 2, 2]
            a = E[..., 0]
            b = E[..., 1]
            c = E[..., 2]
            # ax + by + c = 0
            x1 = torch.ones_like(a) * 0.5
            x2 = torch.ones_like(a) * (W - 0.5)
            y1 = (-a / b) * x1 - (c / b)
            y2 = (-a / b) * x2 - (c / b)
            lines = torch.stack([torch.stack([x1, y1], dim=-1), torch.stack([x2, y2], dim=-1)], dim=-2)
            return lines

        lines2_1 = (get_lines(e2_1) / size) * 2 - 1
        lines1_2 = (get_lines(e1_2) / size) * 2 - 1

        d2_1 = point_line_distance(
            pts.view(BS, 1, 1, H, W, 2).repeat(BS, H, W, 1, 1, 1),
            e2_1.view(BS, H, W, 1, 1, 3).repeat(BS, 1, 1, H, W, 1),
        )
        d1_2 = point_line_distance(
            pts.view(BS, 1, 1, H, W, 2).repeat(BS, H, W, 1, 1, 1),
            e1_2.view(BS, H, W, 1, 1, 3).repeat(BS, 1, 1, H, W, 1),
        )

        # FIND BEST MATCH
        # "distance with weighting" https://www.desmos.com/calculator/jbruxrrby9
        dw2_1 = self.height_scalar * torch.exp(-((d2_1 / self.sigma_scalar) ** 2))
        dw1_2 = self.height_scalar * torch.exp(-((d1_2 / self.sigma_scalar) ** 2))
        dwc2_1 = dw2_1 * c2_1  # "distance with weighting * correlation"
        dwc1_2 = dw1_2 * c1_2
        dwcs2_1 = torch.softmax(dwc2_1.view(BS, H * W, -1), dim=-1).view(BS, H, W, H, W)  # "... w/ softmax"
        dwcs1_2 = torch.softmax(dwc1_2.view(BS, H * W, -1), dim=-1).view(BS, H, W, H, W)
        m2_1 = l2.view(-1, 2)[torch.argmax(dwcs2_1.view(BS, H, W, -1), dim=-1)]
        m1_2 = l1.view(-1, 2)[torch.argmax(dwcs1_2.view(BS, H, W, -1), dim=-1)]

        assert not m2_1.isnan().any()
        assert not m1_2.isnan().any()

        if output_folder or show:
            keypoints = torch.tensor([[[0.0, 0.7]]], device=device).repeat(BS, 1, 1)
            colors1 = torch.tensor([[[0.0832, 0.7328, 0.8391]]], device=device)
            colors2 = torch.tensor([[[0.1416, 0.7433, 0.3251]]], device=device)

            # epipolar lines
            sampled_lines2_1 = (
                torch.nn.functional.grid_sample(
                    lines2_1.permute(0, 3, 4, 1, 2).view(BS, 4, H, W),
                    keypoints[:, None],
                    mode="nearest",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)[:, 0]
                .view(BS, -1, 2, 2)
            )
            sampled_lines1_2 = (
                torch.nn.functional.grid_sample(
                    lines1_2.permute(0, 3, 4, 1, 2).view(BS, 4, H, W),
                    keypoints[:, None],
                    mode="nearest",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)[:, 0]
                .view(BS, -1, 2, 2)
            )

            # keypoints
            sampled_m2_1 = (
                torch.nn.functional.grid_sample(
                    m2_1.permute(0, 3, 1, 2),
                    keypoints[:, None],
                    mode="nearest",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)[:, 0]
                .view(BS, -1, 2)
            )
            sampled_m1_2 = (
                torch.nn.functional.grid_sample(
                    m1_2.permute(0, 3, 1, 2),
                    keypoints[:, None],
                    mode="nearest",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)[:, 0]
                .view(BS, -1, 2)
            )

            # cycle keypoints
            sampled_m1_2_1 = (
                torch.nn.functional.grid_sample(
                    m1_2_1.permute(0, 3, 1, 2),
                    keypoints[:, None],
                    mode="nearest",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)[:, 0]
                .view(BS, -1, 2)
            )
            sampled_m2_1_2 = (
                torch.nn.functional.grid_sample(
                    m2_1_2.permute(0, 3, 1, 2),
                    keypoints[:, None],
                    mode="nearest",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)[:, 0]
                .view(BS, -1, 2)
            )

            # correlation
            sampled_c1_2 = (
                torch.nn.functional.grid_sample(
                    c1_2.permute(0, 3, 4, 1, 2).view(BS, -1, H, W),
                    keypoints[:, None],
                    mode="nearest",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)[:, 0]
                .view(BS, H, W)
            )
            sampled_c2_1 = (
                torch.nn.functional.grid_sample(
                    c2_1.permute(0, 3, 4, 1, 2).view(BS, -1, H, W),
                    keypoints[:, None],
                    mode="nearest",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)[:, 0]
                .view(BS, H, W)
            )

            matches1 = get_images_with_lines(image1, sampled_lines2_1, colors=colors2, line_width=self.line_width)
            matches1 = get_images_with_keypoints(matches1, keypoints, colors=colors1, keypoint_size=self.keypoint_size)
            matches1 = get_images_with_keypoints(matches1, sampled_m2_1, colors=colors2, keypoint_size=10, thickness=5)
            matches1 = get_images_with_keypoints(
                matches1, sampled_m1_2_1, colors=colors1, keypoint_size=20, thickness=5
            )

            matches2 = get_images_with_lines(image2, sampled_lines1_2, colors=colors1, line_width=self.line_width)
            matches2 = get_images_with_keypoints(matches2, keypoints, colors=colors2, keypoint_size=self.keypoint_size)
            matches2 = get_images_with_keypoints(matches2, sampled_m1_2, colors=colors1, keypoint_size=10, thickness=5)
            matches2 = get_images_with_keypoints(
                matches2, sampled_m2_1_2, colors=colors2, keypoint_size=20, thickness=5
            )

            # concatenate images
            image_matches = torch.cat(
                [
                    matches1.permute(0, 2, 3, 1).detach().cpu()[0],
                    matches2.permute(0, 2, 3, 1).detach().cpu()[0],
                ],
                dim=1,
            )

            image_correlation1_with_colormap = apply_colormap(
                sampled_c1_2.view(BS, H, W, 1), ColormapOptions(normalize=True)
            )[0]
            image_correlation2_with_colormap = apply_colormap(
                sampled_c2_1.view(BS, H, W, 1), ColormapOptions(normalize=True)
            )[0]
            image_correlation = (
                torch.cat(
                    [
                        image_correlation1_with_colormap,
                        image_correlation2_with_colormap,
                    ],
                    dim=1,
                )
                .detach()
                .cpu()
            )

            if output_folder:
                output_folder.mkdir(parents=True, exist_ok=True)
                mediapy.write_image(output_folder / f"matches{suffix}.png", image_matches)
                mediapy.write_image(output_folder / f"correlation{suffix}.png", image_correlation)

            if show:
                mediapy.show_image(image_matches, height=display_height, title="matches")
                mediapy.show_image(image_correlation, height=display_height, title="correlation")

        return m1_2, m2_1

    def forward(
        self,
        features1: Float[Tensor, "B C H W"],
        features2: Float[Tensor, "B C H W"],
        image1: Float[Tensor, "B 3 Horig Worig"],
        image2: Float[Tensor, "B 3 Horig Worig"],
        depth1: Optional[Float[Tensor, "B 1 H W"]] = None,
        depth2: Optional[Float[Tensor, "B 1 H W"]] = None,
        mask1: Optional[Float[Tensor, "B 1 H W"]] = None,
        mask2: Optional[Float[Tensor, "B 1 H W"]] = None,
        K1: Optional[Float[Tensor, "B 3 3"]] = None,
        K2: Optional[Float[Tensor, "B 3 3"]] = None,
        c2w1: Optional[Float[Tensor, "B 3 4"]] = None,
        c2w2: Optional[Float[Tensor, "B 3 4"]] = None,
        output_folder: Optional[Path] = None,
        suffix: str = "",
        show: bool = False,
        display_height: int = 512,
    ) -> Float[Tensor, "BxB"]:
        """
        Returns all pairs as distances in a matrix that is shape BxB.
        Inpaint where mask == 1.
        """
        assert features1.shape[0] == 1, "We don't handle more than one image pair at the moment."
        assert not features1.isnan().any()
        assert not features1.isnan().any()

        device = features1.device
        dtype = features1.dtype
        BS, C, H, W = features1.shape
        size = torch.tensor([W, H], device=device)

        losses = {}

        if mask1 is None:
            mask1 = torch.ones_like(features1[:, :1])
        if mask2 is None:
            mask2 = torch.ones_like(features2[:, :1])

        # pts[0,0] is top left and pts are (x,y)
        pts = (
            (
                torch.stack(
                    torch.meshgrid(torch.arange(H), torch.arange(W), indexing="xy"),
                    dim=-1,
                ).to(device)
            )
            .view(H, W, 2)
            .to(dtype)
        ) + 0.5
        pts = pts[None].repeat(BS, 1, 1, 1)

        l1 = (pts / size) * 2 - 1
        l2 = l1.clone()
        l2_1, depth2_1, valid_depth2_1 = reproject(
            from_l=l2, depth=depth2, from_K=K2, from_c2w=c2w2, to_K=K1, to_c2w=c2w1
        )
        l1_2, depth1_2, valid_depth1_2 = reproject(
            from_l=l1, depth=depth1, from_K=K1, from_c2w=c2w1, to_K=K2, to_c2w=c2w2
        )

        # to check for occlusions
        depthcheck2_1 = torch.nn.functional.grid_sample(depth1, l2_1, mode="bilinear", align_corners=False)
        depthcheck1_2 = torch.nn.functional.grid_sample(depth2, l1_2, mode="bilinear", align_corners=False)

        # cycle consistency
        # m1_2_1 = torch.nn.functional.grid_sample(
        #     m2_1.permute(0, 3, 1, 2), m1_2, mode="bilinear", align_corners=False
        # ).permute(0, 2, 3, 1)
        # m2_1_2 = torch.nn.functional.grid_sample(
        #     m1_2.permute(0, 3, 1, 2), m2_1, mode="bilinear", align_corners=False
        # ).permute(0, 2, 3, 1)

        # sample the image
        # resampled1 = torch.nn.functional.grid_sample(image1, m1_2_1, mode="bilinear", align_corners=False)
        # resampled2 = torch.nn.functional.grid_sample(image2, m2_1_2, mode="bilinear", align_corners=False)

        # feature loss - Uses depth and features to maximize similarity.
        if self.lossfeatmult != 0.0:
            f1 = torch.nn.functional.grid_sample(features1, l1, mode="bilinear", align_corners=False)
            f2 = torch.nn.functional.grid_sample(features2, l2, mode="bilinear", align_corners=False)
            f1_2 = torch.nn.functional.grid_sample(features2, l1_2, mode="bilinear", align_corners=False)
            f2_1 = torch.nn.functional.grid_sample(features1, l2_1, mode="bilinear", align_corners=False)
            lossfeat1 = torch.sum((f1 - f1_2) ** 2, dim=1, keepdim=True)
            lossfeat2 = torch.sum((f2 - f2_1) ** 2, dim=1, keepdim=True)
            lossfeat1 *= mask1
            lossfeat2 *= mask2
            lossfeat1 *= (torch.abs(depth1_2 - depthcheck1_2) < 0.1).float()
            lossfeat2 *= (torch.abs(depth2_1 - depthcheck2_1) < 0.1).float()
            assert not lossfeat1.isnan().any()
            assert not lossfeat2.isnan().any()
            lossfeat = lossfeat1.mean() + lossfeat2.mean()
            losses["feat"] = self.lossfeatmult * lossfeat

        # distance loss - Finds best features and encourages depth to be consistent with this
        if self.lossdistmult != 0.0:
            m1_2, m2_1 = self.compute_matches(
                features1=features1,
                features2=features2,
                K1=K1,
                K2=K2,
                c2w1=c2w1,
                c2w2=c2w2,
                output_folder=output_folder,
                suffix=suffix,
                show=show,
                display_height=display_height,
            )
            lossdist1 = ((m1_2 - l1_2) ** 2).mean(-1).unsqueeze(1)
            lossdist2 = ((m2_1 - l2_1) ** 2).mean(-1).unsqueeze(1)
            lossdist1 *= mask1
            lossdist2 *= mask2
            assert not lossdist1.isnan().any()
            assert not lossdist2.isnan().any()
            lossdist = lossdist1.mean() + lossdist2.mean()
            losses["dist"] = self.lossdistmult * lossdist

        if output_folder or show:
            # keypoints = torch.tensor([[[0.0, 0.7]]], device=device).repeat(BS, 1, 1)
            # colors1 = torch.tensor([[[0.0832, 0.7328, 0.8391]]], device=device)
            # colors2 = torch.tensor([[[0.1416, 0.7433, 0.3251]]], device=device)

            # keypoints_lines = torch.rand((BS, 10, 2), device=device) * 2 - 1
            # colors_lines = torch.rand((BS, 10, 3), device=device)
            # keypoints_lines = keypoints
            # colors_lines = colors1

            # print(keypoints.shape)
            # print(keypoints_lines.shape)

            # # epipolar lines
            # sampled_lines2_1 = (
            #     torch.nn.functional.grid_sample(
            #         lines2_1.permute(0, 3, 4, 1, 2).view(BS, 4, H, W),
            #         keypoints[:, None],
            #         mode="nearest",
            #         align_corners=False,
            #     )
            #     .permute(0, 2, 3, 1)[:, 0]
            #     .view(BS, -1, 2, 2)
            # )
            # sampled_lines1_2 = (
            #     torch.nn.functional.grid_sample(
            #         lines1_2.permute(0, 3, 4, 1, 2).view(BS, 4, H, W),
            #         keypoints[:, None],
            #         mode="nearest",
            #         align_corners=False,
            #     )
            #     .permute(0, 2, 3, 1)[:, 0]
            #     .view(BS, -1, 2, 2)
            # )

            # # keypoints
            # sampled_m2_1 = (
            #     torch.nn.functional.grid_sample(
            #         m2_1.permute(0, 3, 1, 2),
            #         keypoints[:, None],
            #         mode="nearest",
            #         align_corners=False,
            #     )
            #     .permute(0, 2, 3, 1)[:, 0]
            #     .view(BS, -1, 2)
            # )
            # sampled_m1_2 = (
            #     torch.nn.functional.grid_sample(
            #         m1_2.permute(0, 3, 1, 2),
            #         keypoints[:, None],
            #         mode="nearest",
            #         align_corners=False,
            #     )
            #     .permute(0, 2, 3, 1)[:, 0]
            #     .view(BS, -1, 2)
            # )

            # # cycle keypoints
            # sampled_m1_2_1 = (
            #     torch.nn.functional.grid_sample(
            #         m1_2_1.permute(0, 3, 1, 2),
            #         keypoints[:, None],
            #         mode="nearest",
            #         align_corners=False,
            #     )
            #     .permute(0, 2, 3, 1)[:, 0]
            #     .view(BS, -1, 2)
            # )
            # sampled_m2_1_2 = (
            #     torch.nn.functional.grid_sample(
            #         m2_1_2.permute(0, 3, 1, 2),
            #         keypoints[:, None],
            #         mode="nearest",
            #         align_corners=False,
            #     )
            #     .permute(0, 2, 3, 1)[:, 0]
            #     .view(BS, -1, 2)
            # )

            # # correlation
            # sampled_c1_2 = (
            #     torch.nn.functional.grid_sample(
            #         c1_2.permute(0, 3, 4, 1, 2).view(BS, -1, H, W),
            #         keypoints[:, None],
            #         mode="nearest",
            #         align_corners=False,
            #     )
            #     .permute(0, 2, 3, 1)[:, 0]
            #     .view(BS, H, W)
            # )
            # sampled_c2_1 = (
            #     torch.nn.functional.grid_sample(
            #         c2_1.permute(0, 3, 4, 1, 2).view(BS, -1, H, W),
            #         keypoints[:, None],
            #         mode="nearest",
            #         align_corners=False,
            #     )
            #     .permute(0, 2, 3, 1)[:, 0]
            #     .view(BS, H, W)
            # )

            # matches1 = get_images_with_lines(image1, sampled_lines2_1, colors=colors2, line_width=self.line_width)
            # matches1 = get_images_with_keypoints(matches1, keypoints, colors=colors1, keypoint_size=self.keypoint_size)
            # matches1 = get_images_with_keypoints(matches1, sampled_m2_1, colors=colors2, keypoint_size=10, thickness=5)
            # matches1 = get_images_with_keypoints(
            #     matches1, sampled_m1_2_1, colors=colors1, keypoint_size=20, thickness=5
            # )

            # matches2 = get_images_with_lines(image2, sampled_lines1_2, colors=colors1, line_width=self.line_width)
            # matches2 = get_images_with_keypoints(matches2, keypoints, colors=colors2, keypoint_size=self.keypoint_size)
            # matches2 = get_images_with_keypoints(matches2, sampled_m1_2, colors=colors1, keypoint_size=10, thickness=5)
            # matches2 = get_images_with_keypoints(
            #     matches2, sampled_m2_1_2, colors=colors2, keypoint_size=20, thickness=5
            # )

            # concatenate images
            # image_matches = torch.cat(
            #     [matches1.permute(0, 2, 3, 1).detach().cpu()[0], matches2.permute(0, 2, 3, 1).detach().cpu()[0]], dim=1
            # )

            # image_resampled = torch.cat(
            #     [resampled1.permute(0, 2, 3, 1).detach().cpu()[0], resampled2.permute(0, 2, 3, 1).detach().cpu()[0]],
            #     dim=1,
            # )

            image_images = torch.cat(
                [
                    image1.permute(0, 2, 3, 1).detach().cpu()[0],
                    image2.permute(0, 2, 3, 1).detach().cpu()[0],
                ],
                dim=1,
            )

            print(f1_2.shape)
            print(f1_2[:, :3].shape)

            image_features = torch.cat([f1_2[:, :3], f2_1[:, :3]], dim=-1).permute(0, 2, 3, 1).detach().cpu()[0]
            print(image_features.shape)

            # weights1_with_colormap = apply_colormap(weight1[..., None], ColormapOptions(normalize=True))
            # weights2_with_colormap = apply_colormap(weight2[..., None], ColormapOptions(normalize=True))
            # image_weights = torch.cat([weights1_with_colormap, weights2_with_colormap], dim=1).detach().cpu()

            # depth1_with_colormap = apply_colormap(valid_depth2_1[..., None], ColormapOptions(normalize=True))
            # depth2_with_colormap = apply_colormap(valid_depth1_2[..., None], ColormapOptions(normalize=True))
            # image_depth = torch.cat([depth1_with_colormap, depth2_with_colormap], dim=1).detach().cpu()

            # image_correlation1_with_colormap = apply_colormap(
            #     sampled_c1_2.view(BS, H, W, 1), ColormapOptions(normalize=True)
            # )[0]
            # image_correlation2_with_colormap = apply_colormap(
            #     sampled_c2_1.view(BS, H, W, 1), ColormapOptions(normalize=True)
            # )[0]
            # image_correlation = (
            #     torch.cat([image_correlation1_with_colormap, image_correlation2_with_colormap], dim=1).detach().cpu()
            # )

            # image_softmax1_with_colormap = apply_colormap(
            #     high1[ij[..., 1], ij[..., 0]].view(H, W, 1), ColormapOptions(normalize=True)
            # )
            # image_softmax2_with_colormap = apply_colormap(
            #     high2[ij[..., 1], ij[..., 0]].view(H, W, 1), ColormapOptions(normalize=True)
            # )
            # image_softmax = (
            #     torch.cat([image_softmax1_with_colormap, image_softmax2_with_colormap], dim=1).detach().cpu()
            # )

            # losses
            # lossdist1_with_colormap = apply_colormap(lossdist1[..., None], ColormapOptions(normalize=True))[0, 0]
            # lossdist2_with_colormap = apply_colormap(lossdist2[..., None], ColormapOptions(normalize=True))[0, 0]
            # image_lossdist = torch.cat([lossdist1_with_colormap, lossdist2_with_colormap], dim=1).detach().cpu()

            lossfeat1_with_colormap = apply_colormap(lossfeat1[..., None], ColormapOptions(normalize=True))[0, 0]
            lossfeat2_with_colormap = apply_colormap(lossfeat2[..., None], ColormapOptions(normalize=True))[0, 0]
            image_lossfeat = torch.cat([lossfeat1_with_colormap, lossfeat2_with_colormap], dim=1).detach().cpu()

            if output_folder:
                output_folder.mkdir(parents=True, exist_ok=True)
                # mediapy.write_image(output_folder / f"matches{suffix}.png", image_matches)
                # mediapy.write_image(output_folder / f"resampled{suffix}.png", image_resampled)
                #     mediapy.write_image(output_folder / f"weights{suffix}.png", image_weights)
                #     mediapy.write_image(output_folder / f"depth{suffix}.png", image_depth)
                # mediapy.write_image(output_folder / f"lossdist{suffix}.png", image_lossdist)
                mediapy.write_image(output_folder / f"lossfeat{suffix}.png", image_lossfeat)
                # mediapy.write_image(output_folder / f"correlation{suffix}.png", image_correlation)
            #     mediapy.write_image(output_folder / f"softmax{suffix}.png", image_softmax)

            if show:
                # mediapy.show_image(image_matches, height=display_height, title="matches")
                mediapy.show_image(image_images, height=display_height, title="images")
                mediapy.show_image(image_features, height=display_height, title="features")
                # #     mediapy.show_image(image_depth, height=display_height, title="depth")
                # mediapy.show_image(image_correlation, height=display_height, title="correlation")
                # #     mediapy.show_image(image_softmax, height=display_height, title="softmax")
                # #     mediapy.show_image(image_weights, height=display_height, title="weights")
                # mediapy.show_image(image_lossdist, height=display_height, title="lossdist")
                mediapy.show_image(image_lossfeat, height=display_height, title="lossfeat")
                # mediapy.show_image(image_resampled, height=display_height, title="resampled")

        loss = sum(losses.values())
        return loss, losses


class ReprojectMetric(MultiviewMetric):
    """
    Computes a loss for based on reprojection.
    """

    def __init__(
        self,
        lossfeatmult: float = 1.0,
        cycle_threshold: float = 0.0025,
        eps: float = 1e-6,
        thresh: float = 0.018,
    ):
        super().__init__()
        self.lossfeatmult = lossfeatmult
        self.cycle_threshold = cycle_threshold
        self.eps = eps
        self.thresh = thresh

    def forward(
        self,
        features1: Float[Tensor, "B C H W"],
        features2: Float[Tensor, "B C H W"],
        image1: Float[Tensor, "B 3 Horig Worig"],
        image2: Float[Tensor, "B 3 Horig Worig"],
        depth1: Optional[Float[Tensor, "B 1 H W"]] = None,
        depth2: Optional[Float[Tensor, "B 1 H W"]] = None,
        mask1: Optional[Float[Tensor, "B 1 H W"]] = None,
        mask2: Optional[Float[Tensor, "B 1 H W"]] = None,
        K1: Optional[Float[Tensor, "B 3 3"]] = None,
        K2: Optional[Float[Tensor, "B 3 3"]] = None,
        c2w1: Optional[Float[Tensor, "B 3 4"]] = None,
        c2w2: Optional[Float[Tensor, "B 3 4"]] = None,
        output_folder: Optional[Path] = None,
        suffix: str = "",
        show: bool = False,
        display_height: int = 512,
    ) -> Float[Tensor, "BxB"]:
        """
        Returns all pairs as distances in a matrix that is shape BxB.
        Inpaint where mask == 1.
        """
        assert features1.shape[0] == 1, "We don't handle more than one image pair at the moment."
        assert not features1.isnan().any()
        assert not features1.isnan().any()

        device = features1.device
        dtype = features1.dtype
        BS, C, H, W = features1.shape
        size = torch.tensor([W, H], device=device)

        losses = {}

        if mask1 is None:
            mask1 = torch.ones_like(features1[:, :1])
        if mask2 is None:
            mask2 = torch.ones_like(features2[:, :1])

        # pts[0,0] is top left and pts are (x,y)
        pts = (
            (
                torch.stack(
                    torch.meshgrid(torch.arange(H), torch.arange(W), indexing="xy"),
                    dim=-1,
                ).to(device)
            )
            .view(H, W, 2)
            .to(dtype)
        ) + 0.5
        pts = pts[None].repeat(BS, 1, 1, 1)

        l1 = (pts / size) * 2 - 1
        l2 = l1.clone()
        l2_1, depth2_1, valid_depth2_1 = reproject(
            from_l=l2, depth=depth2, from_K=K2, from_c2w=c2w2, to_K=K1, to_c2w=c2w1
        )
        l1_2, depth1_2, valid_depth1_2 = reproject(
            from_l=l1, depth=depth1, from_K=K1, from_c2w=c2w1, to_K=K2, to_c2w=c2w2
        )

        # cycle consistency
        l1_2_1 = torch.nn.functional.grid_sample(
            l2_1.permute(0, 3, 1, 2), l1_2, mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        l2_1_2 = torch.nn.functional.grid_sample(
            l1_2.permute(0, 3, 1, 2), l2_1, mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)

        dist_l1_2_1 = torch.sqrt(((l1.permute(0, 3, 1, 2) - l1_2_1.permute(0, 3, 1, 2)) ** 2).sum(1, keepdim=True))
        dist_l2_1_2 = torch.sqrt(((l2.permute(0, 3, 1, 2) - l2_1_2.permute(0, 3, 1, 2)) ** 2).sum(1, keepdim=True))

        # to check for occlusions
        depthcheck2_1 = torch.nn.functional.grid_sample(depth1, l2_1, mode="bilinear", align_corners=False)
        depthcheck1_2 = torch.nn.functional.grid_sample(depth2, l1_2, mode="bilinear", align_corners=False)

        # mediapy.show_image((dist_l2_1_2 < self.cycle_threshold)[0, 0].cpu())
        # mediapy.show_image((dist_l1_2_1 < self.cycle_threshold)[0, 0].cpu())

        # feature loss - Uses depth and features to maximize similarity.
        if self.lossfeatmult != 0.0:
            f1 = torch.nn.functional.grid_sample(features1, l1, mode="bilinear", align_corners=False)
            f2 = torch.nn.functional.grid_sample(features2, l2, mode="bilinear", align_corners=False)
            f1_2 = torch.nn.functional.grid_sample(features2, l1_2, mode="bilinear", align_corners=False)
            f2_1 = torch.nn.functional.grid_sample(features1, l2_1, mode="bilinear", align_corners=False)

            weights1 = (dist_l1_2_1 < self.cycle_threshold).float() * mask1.float()
            weights2 = (dist_l2_1_2 < self.cycle_threshold).float() * mask2.float()

            lossfeat1 = torch.sum((f1 - f1_2) ** 2, dim=1, keepdim=True)
            lossfeat2 = torch.sum((f2 - f2_1) ** 2, dim=1, keepdim=True)
            lossfeat1 *= weights1
            lossfeat2 *= weights1
            assert not lossfeat1.isnan().any()
            assert not lossfeat2.isnan().any()
            lossfeat = lossfeat1.mean() + lossfeat2.mean()
            losses["feat"] = self.lossfeatmult * lossfeat

        if output_folder or show:
            image_images = torch.cat(
                [
                    image1.permute(0, 2, 3, 1).detach().cpu()[0],
                    image2.permute(0, 2, 3, 1).detach().cpu()[0],
                ],
                dim=1,
            )
            image_features_source = torch.cat([f1[:, :3], f2[:, :3]], dim=-1).permute(0, 2, 3, 1).detach().cpu()[0]
            image_features = torch.cat([f1_2[:, :3], f2_1[:, :3]], dim=-1).permute(0, 2, 3, 1).detach().cpu()[0]
            image_weights = torch.cat([weights1, weights2], dim=-1).permute(0, 2, 3, 1).detach().cpu()[0, ..., 0]

            lossfeat1_with_colormap = apply_colormap(lossfeat1[..., None], ColormapOptions(normalize=True))[0, 0]
            lossfeat2_with_colormap = apply_colormap(lossfeat2[..., None], ColormapOptions(normalize=True))[0, 0]
            image_lossfeat = torch.cat([lossfeat1_with_colormap, lossfeat2_with_colormap], dim=1).detach().cpu()

            if output_folder:
                output_folder.mkdir(parents=True, exist_ok=True)
                mediapy.write_image(output_folder / f"featuressource{suffix}.png", image_features_source)
                mediapy.write_image(output_folder / f"features{suffix}.png", image_features)
                mediapy.write_image(output_folder / f"weights{suffix}.png", image_weights)

            if show:
                mediapy.show_image(image_images, height=display_height, title="images")
                mediapy.show_image(image_features_source, height=display_height, title="featuressource")
                mediapy.show_image(image_features, height=display_height, title="features")
                mediapy.show_image(image_lossfeat, height=display_height, title="lossfeat")
                mediapy.show_image(image_weights, height=display_height, title="weights")

        loss = sum(losses.values())
        return loss, losses
