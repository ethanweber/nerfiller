from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import mediapy as media
import numpy as np
import torch
import tyro
from typing_extensions import Annotated

from nerfiller.utils.typing import *
from nerfiller.utils.metrics import (
    LPIPSModule,
    PSNRModule,
    SSIMModule,
    MUSIQModule,
    CorrsModule,
)
from nerfiller.experiments.data import novel_view_pairs


@dataclass
class BaseMetrics:
    """Base class for metrics."""

    input_folder: Path = Path("input-folder")
    """Folder containing the renders."""
    output_folder: Path = Path("output-folder")
    """Folder to save the metrics."""
    device: str = "cuda:0"
    """Device to use for metrics."""
    video_sec: float = 10.0
    """Seconds for the video."""

    def main(self) -> None:
        """Main function."""

        # image metrics
        psnr_module = PSNRModule().to(self.device)
        ssim_module = SSIMModule().to(self.device)
        lpips_module = LPIPSModule().to(self.device)

        gt_rgb_filenames = sorted(glob.glob(str(self.input_folder / "gt-rgb" / "*")))
        metrics = defaultdict()
        video = []

        experiment_name = os.path.basename(self.input_folder)
        print(f"Processing experiment: {experiment_name} ...")
        if len(gt_rgb_filenames) == 0:
            print("No gt_rgb images found, skipping experiment")
            sys.exit(0)

        for idx, gt_rgb_filename in enumerate(gt_rgb_filenames):
            rgb_filename = gt_rgb_filename.replace("gt-rgb", "rgb")
            assert rgb_filename != gt_rgb_filename, "The image filenames should not be identical!"
            rgb = media.read_image(rgb_filename)
            gt_rgb = media.read_image(gt_rgb_filename)

            # move images to torch and to the correct device
            rgb = torch.from_numpy(rgb).float().to(self.device) / 255.0
            gt_rgb = torch.from_numpy(gt_rgb).float().to(self.device) / 255.0  # (H, W, 3)

            # compute the image metrics
            # reshape the images to (1, C, H, W)
            x = rgb.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            x_gt = gt_rgb.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

            psnr = float(psnr_module(x, x_gt)[0])
            ssim = float(ssim_module(x, x_gt)[0])
            lpips = float(lpips_module(x, x_gt)[0])

            metrics["psnr_list"] = metrics.get("psnr_list", []) + [psnr]
            metrics["ssim_list"] = metrics.get("ssim_list", []) + [ssim]
            metrics["lpips_list"] = metrics.get("lpips_list", []) + [lpips]

            # save the images
            gt_rgb = (gt_rgb * 255.0).cpu().numpy().astype(np.uint8)
            rgb = (rgb * 255.0).cpu().numpy().astype(np.uint8)
            image = np.concatenate([gt_rgb, rgb], axis=1)
            image_filename = self.input_folder / "composited" / f"{idx:04d}.png"
            image_filename.parent.mkdir(parents=True, exist_ok=True)
            media.write_image(image_filename, image)
            video.append(image)

        # write out the video
        video_filename = self.input_folder / f"{experiment_name}.mp4"
        fps = int(len(video) / self.video_sec)
        media.write_video(video_filename, video, fps=fps)

        # convert metrics dict to a proper dictionary
        metrics = dict(metrics)
        metrics["psnr"] = np.mean(metrics["psnr_list"])
        metrics["ssim"] = np.mean(metrics["ssim_list"])
        metrics["lpips"] = np.mean(metrics["lpips_list"])

        for metric_name in sorted(metrics.keys()):
            if "_list" not in metric_name:
                print(f"{metric_name}: {metrics[metric_name]}")

        # write to a json file
        metrics_filename = self.output_folder / f"{experiment_name}.json"
        os.makedirs(self.output_folder, exist_ok=True)
        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)


@dataclass
class NeRFillerMetrics(BaseMetrics):
    """Masked metrics."""

    gt_rgb_folder: Path = Path("gt-rgb-folder")
    """Which folder to use as ground truth RGB images"""
    novel_view_video: Optional[Path] = None
    """Optional novel view to compute metrics on"""
    novel_view_scale_factor: float = 0.25
    """How much to scale the novel view video before computing the metric."""

    def main(self) -> None:
        """Main function."""

        # image metrics
        psnr_module = PSNRModule().to(self.device)
        ssim_module = SSIMModule().to(self.device)
        lpips_module = LPIPSModule().to(self.device)

        gt_rgb_filenames = sorted(glob.glob(str(self.gt_rgb_folder / "*")))
        rgb_filenames = sorted(glob.glob(str(self.input_folder / "rgb" / "*")))
        assert len(gt_rgb_filenames) == len(rgb_filenames)
        metrics = defaultdict()
        video = []
        rgbs = []

        experiment_name = os.path.basename(self.input_folder)
        print(f"Processing experiment: {experiment_name} ...")

        if self.novel_view_video:
            print("Computing novel video metrics...")
            musiq_module = MUSIQModule().to(self.device)
            corrs_module = CorrsModule().to(self.device)

            frames = media.read_video(self.novel_view_video)

            for pair in novel_view_pairs:
                i = min(round(pair[0] * len(frames)), len(frames) - 1)
                j = min(round(pair[1] * len(frames)), len(frames) - 1)
                # rgb and depth concatenated in row, so take left only
                height = frames[i].shape[0]
                rgb_framei = frames[i][:, :height]
                height = frames[j].shape[0]
                rgb_framej = frames[j][:, :height]
                xi = (torch.from_numpy(rgb_framei).permute(2, 0, 1)[None] / 255.0).to(self.device)
                xj = (torch.from_numpy(rgb_framej).permute(2, 0, 1)[None] / 255.0).to(self.device)
                corrs = float(corrs_module(xi.to(self.device), xj.to(self.device))[0])
                metrics["corrs_list"] = metrics.get("corrs_list", []) + [corrs]

            for idx in range(len(frames)):
                # rgb and depth concatenated in row, so take left only
                height = frames[idx].shape[0]
                rgb_frame = frames[idx][:, :height]
                x = torch.from_numpy(rgb_frame).permute(2, 0, 1)[None] / 255.0
                x_down = torch.nn.functional.interpolate(x, scale_factor=self.novel_view_scale_factor, mode="bilinear")
                musiq = float(musiq_module(x_down, None)[0])
                metrics["musiq_list"] = metrics.get("musiq_list", []) + [musiq]

        if len(gt_rgb_filenames) == 0:
            print("No gt_rgb images found, skipping experiment")
            sys.exit(0)

        for idx, (gt_rgb_filename, rgb_filename) in enumerate(zip(gt_rgb_filenames, rgb_filenames)):
            assert rgb_filename != gt_rgb_filename, "The image filenames should not be identical!"
            rgb = media.read_image(rgb_filename)
            gt_rgb = media.read_image(gt_rgb_filename)

            # move images to torch and to the correct device
            rgb = torch.from_numpy(rgb).float().to(self.device) / 255.0
            gt_rgb = torch.from_numpy(gt_rgb).float().to(self.device) / 255.0  # (H, W, 3)

            # compute the image metrics
            # reshape the images to (1, C, H, W)
            x = rgb.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            x_gt = gt_rgb.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

            psnr = float(psnr_module(x, x_gt)[0])
            ssim = float(ssim_module(x, x_gt)[0])
            lpips = float(lpips_module(x, x_gt)[0])

            metrics["psnr_list"] = metrics.get("psnr_list", []) + [psnr]
            metrics["ssim_list"] = metrics.get("ssim_list", []) + [ssim]
            metrics["lpips_list"] = metrics.get("lpips_list", []) + [lpips]

            # save the images
            gt_rgb = (gt_rgb * 255.0).cpu().numpy().astype(np.uint8)
            rgb = (rgb * 255.0).cpu().numpy().astype(np.uint8)
            image = np.concatenate([gt_rgb, rgb], axis=1)
            rgbs.append(rgb)
            image_filename = self.input_folder / "composited" / f"{idx:04d}.png"
            image_filename.parent.mkdir(parents=True, exist_ok=True)
            media.write_image(image_filename, image)
            video.append(image)

        # write out the video
        video_filename = self.input_folder / f"{experiment_name}.mp4"
        fps = int(len(video) / self.video_sec)
        media.write_video(video_filename, video, fps=fps)
        media.write_video(str(video_filename).replace(".mp4", "_only_rgb.mp4"), rgbs, fps=fps)

        # convert metrics dict to a proper dictionary
        metrics = dict(metrics)
        for key in list(metrics.keys()):
            metrics[key.replace("_list", "")] = np.mean(metrics[key])

        for metric_name in sorted(metrics.keys()):
            if "_list" not in metric_name:
                print(f"{metric_name}: {metrics[metric_name]}")

        # write to a json file
        metrics_filename = self.output_folder / f"{experiment_name}.json"
        os.makedirs(self.output_folder, exist_ok=True)
        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[BaseMetrics, tyro.conf.subcommand(name="base-metrics")],
        Annotated[NeRFillerMetrics, tyro.conf.subcommand(name="nerfiller-metrics")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
