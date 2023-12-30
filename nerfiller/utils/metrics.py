"""
This file contains NeRF metrics with masking capabilities.
"""

from abc import abstractmethod
from typing import Optional

import torch
from torch import nn
from torchmetrics.functional import structural_similarity_index_measure
from torchtyping import TensorType

from nerfiller.utils.lpips_utils import (
    LearnedPerceptualImagePatchSimilarityWithMasking,
)

# import tensorflow as tf
# import tensorflow_hub as hub
# from PIL import Image
# import io
from kornia.feature import LoFTR
from kornia.color import rgb_to_grayscale


class ImageMetricModule(nn.Module):
    """Computes image metrics with masking capabilities.
    We assume that the pred and target inputs are in the range [0, 1].
    """

    def __init__(self):
        super().__init__()
        self.populate_modules()

    def populate_modules(self):
        """Populates the modules that will be used to compute the metric."""

    @abstractmethod
    def forward(
        self,
        preds: TensorType["bs", 3, "H", "W"],
        target: TensorType["bs", 3, "H", "W"],
        mask: Optional[TensorType["bs", 1, "H", "W"]] = None,
    ) -> TensorType["bs"]:
        """Computes the metric.
        Args:
            preds: Predictions.
            target: Ground truth.
            mask: Mask to use to only compute the metrics where the mask is True.
        Returns:
            Metric value.
        """


class PSNRModule(ImageMetricModule):
    """Computes PSNR with masking capabilities."""

    def forward(
        self,
        preds: TensorType["bs", 3, "H", "W"],
        target: TensorType["bs", 3, "H", "W"],
        mask: Optional[TensorType["bs", 1, "H", "W"]] = None,
    ) -> TensorType["bs"]:
        bs, h, w = preds.shape[0], preds.shape[2], preds.shape[3]
        hw = h * w

        preds_reshaped = preds.view(bs, 3, hw)
        target_reshaped = target.view(bs, 3, hw)
        num = (preds_reshaped - target_reshaped) ** 2
        # the non-masked version
        if mask is None:
            den = hw
        else:
            mask_reshaped = mask.view(bs, 1, hw)
            num = num * mask_reshaped
            den = mask_reshaped.sum(-1)
        mse = num.sum(-1) / den
        psnr = 10 * torch.log10(1.0 / (mse + 1e-6))
        psnr = psnr.mean(-1)
        return psnr


class SSIMModule(ImageMetricModule):
    """Computes PSNR with masking capabilities."""

    def forward(
        self,
        preds: TensorType["bs", 3, "H", "W"],
        target: TensorType["bs", 3, "H", "W"],
        mask: Optional[TensorType["bs", 1, "H", "W"]] = None,
    ) -> TensorType["bs"]:
        bs, h, w = preds.shape[0], preds.shape[2], preds.shape[3]
        hw = h * w

        _, ssim_image = structural_similarity_index_measure(
            preds=preds,
            target=target,
            reduction="none",
            data_range=1.0,
            return_full_image=True,
        )
        ssim_image = ssim_image.mean(1)  # average over the channels
        assert ssim_image.shape == (bs, h, w)

        # the non-masked version
        if mask is None:
            ssim = ssim_image.view(bs, hw).mean(1)
            return ssim

        # the masked version
        ssim_reshaped = ssim_image.view(bs, hw)
        mask_reshaped = mask.view(bs, hw)
        den = mask_reshaped.sum(-1, keepdim=True)
        ssim = (ssim_reshaped * mask_reshaped / den).sum(-1)
        return ssim


class LPIPSModule(ImageMetricModule):
    """Computes LPIPS with masking capabilities."""

    def populate_modules(self):
        # by setting normalize=True, we assume that the pred and target inputs are in the range [0, 1]
        self.lpips_with_masking = LearnedPerceptualImagePatchSimilarityWithMasking(normalize=True)

    def forward(
        self,
        preds: TensorType["bs", 3, "H", "W"],
        target: TensorType["bs", 3, "H", "W"],
        mask: Optional[TensorType["bs", 1, "H", "W"]] = None,
    ) -> TensorType["bs"]:
        bs, h, w = preds.shape[0], preds.shape[2], preds.shape[3]
        hw = h * w

        with torch.no_grad():
            lpips_image = self.lpips_with_masking(preds, target)
        lpips_image = lpips_image.mean(1)  # average over the channels
        assert lpips_image.shape == (bs, h, w)

        # the non-masked version
        if mask is None:
            lpips = lpips_image.view(bs, hw).mean(1)
            return lpips

        # the masked version
        lpips_reshaped = lpips_image.view(bs, hw)
        mask_reshaped = mask.view(bs, hw)
        den = mask_reshaped.sum(-1, keepdim=True)
        lpips = (lpips_reshaped * mask_reshaped / den).sum(-1)
        return lpips


class MUSIQModule(ImageMetricModule):
    """Compute MUSIQ metric on images.
    https://tfhub.dev/google/musiq/ava/1
    """

    def populate_modules(self):
        # self.model = hub.load("https://tfhub.dev/google/musiq/ava/1")
        # self.predict_fn = self.model.signatures["serving_default"]
        pass

    def forward(
        self,
        preds: TensorType["bs", 3, "H", "W"],
        target: TensorType["bs", 3, "H", "W"],
        mask: Optional[TensorType["bs", 1, "H", "W"]] = None,
    ) -> TensorType["bs"]:
        bs, h, w = preds.shape[0], preds.shape[2], preds.shape[3]
        hw = h * w

        assert target is None
        assert mask is None

        # scores = []
        # for i in range(bs):
        #     image = preds[i].permute(1, 2, 0)
        #     img = Image.fromarray((image.detach().cpu().numpy() * 255).astype("uint8"))
        #     image_bytes = io.BytesIO()
        #     img.save(image_bytes, format="PNG")
        #     image_bytes = image_bytes.getvalue()
        #     x = tf.constant(image_bytes)
        #     assert x.device.endswith("GPU:0")
        #     aesthetic_score = self.predict_fn(x)
        #     score = float(tf.squeeze(aesthetic_score["output_0"]).numpy())
        #     scores.append(score)
        # scores = torch.tensor(scores)
        
        # TODO: remove this and fix the code above
        scores = torch.tensor([0.0] * bs)
        return scores


class CorrsModule(ImageMetricModule):
    """Compute the number of corrs between images with a confidence greater than the threshold."""

    def populate_modules(self):
        self.matcher = LoFTR(pretrained="indoor")
        self.confidence_thresh = 0.8

    def forward(
        self,
        preds: TensorType["bs", 3, "H", "W"],
        target: TensorType["bs", 3, "H", "W"],
        mask: Optional[TensorType["bs", 1, "H", "W"]] = None,
    ) -> TensorType["bs"]:
        bs, h, w = preds.shape[0], preds.shape[2], preds.shape[3]
        hw = h * w

        assert mask is None

        with torch.no_grad():
            scores = []
            for i in range(bs):
                image0 = rgb_to_grayscale(preds[i : i + 1])
                image1 = rgb_to_grayscale(target[i : i + 1])
                input_ = {"image0": image0, "image1": image1}
                correspondences_dict = self.matcher(input_)
                num_corrs = float((correspondences_dict["confidence"] >= self.confidence_thresh).sum())
                scores.append(num_corrs)
            scores = torch.tensor(scores)

        return scores
