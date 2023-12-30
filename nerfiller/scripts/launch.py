"""
Commands for the NeRFiller paper. This script allows launching many commands on different GPUs, hence the name "launch".
"""

from typing import Union

import tyro

from typing_extensions import Annotated


from nerfiller.experiments.blender import (
    InpaintBlender,
    TrainBlender,
    RenderBlender,
    MetricsBlender,
    NeRFBlenderToNerfstudio,
)
from nerfiller.experiments.occluder import (
    InpaintOccluder,
    TrainOccluder,
    RenderOccluder,
    MetricsOccluder,
)
from nerfiller.experiments.reference import (
    TrainReference,
    RenderReference,
)
from nerfiller.experiments.base import (
    ExperimentConfig,
    EquiRGBDToNerfstudio,
    Inpaint,
    Dreambooth,
    Train,
    Render,
    Metrics,
)

Commands = Union[
    # commands to convert data into Nerfstudio format
    Annotated[
        NeRFBlenderToNerfstudio,
        tyro.conf.subcommand(name="blender-to-nerfstudio"),
    ],
    # TODO: command to convert mesh + occluder to Nerfstudio dataset (currently a notebook)
    Annotated[EquiRGBDToNerfstudio, tyro.conf.subcommand(name="equi-rgbd-to-nerfstudio")],
    # TODO: commands to convert Bayes Rays + novel view trajectory to Nerfstudio dataset
    # command to inpaint a Nerfstudio dataset once
    Annotated[Inpaint, tyro.conf.subcommand(name="inpaint")],
    Annotated[InpaintBlender, tyro.conf.subcommand(name="inpaint-blender")],
    Annotated[InpaintOccluder, tyro.conf.subcommand(name="inpaint-occluder")],
    # command to use train on different datasets with various methods
    Annotated[Train, tyro.conf.subcommand(name="train")],
    Annotated[TrainBlender, tyro.conf.subcommand(name="train-blender")],
    Annotated[TrainOccluder, tyro.conf.subcommand(name="train-occluder")],
    Annotated[TrainReference, tyro.conf.subcommand(name="train-reference")],
    Annotated[Render, tyro.conf.subcommand(name="render")],
    Annotated[RenderBlender, tyro.conf.subcommand(name="render-blender")],
    Annotated[RenderOccluder, tyro.conf.subcommand(name="render-occluder")],
    Annotated[RenderReference, tyro.conf.subcommand(name="render-reference")],
    Annotated[Metrics, tyro.conf.subcommand(name="metrics")],
    Annotated[MetricsBlender, tyro.conf.subcommand(name="metrics-blender")],
    Annotated[MetricsOccluder, tyro.conf.subcommand(name="metrics-occluder")],
    Annotated[Dreambooth, tyro.conf.subcommand(name="dreambooth")],
]


def main(
    experiment_config: ExperimentConfig,
):
    """Script to run the NeRFiller steps to get results.
    experiment_config: The experiment to run.
    """
    experiment_config.main(dry_run=experiment_config.dry_run)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Commands))


if __name__ == "__main__":
    entrypoint()
