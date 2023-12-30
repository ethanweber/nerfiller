"""
Blender experiments.
"""
from dataclasses import dataclass, field

from nerfiller.experiments.base import (
    Inpaint,
    Train,
    Render,
    Metrics,
    ExperimentConfig,
)
from nerfiller.experiments.data import blender_dataset_names
from nerfiller.utils.typing import *
from nerfiller.utils.launch_utils import launch_experiments

inpaint_methods = [
    "grid-prior-no-joint",
    "grid-prior",
    "expanded-attention",
    "individual-sd-image",
    "individual-sd-text",
    "individual-lama",
]

methods = [
    ("nerfacto-nerfiller", "none"),
    ("nerfacto-nerfiller", "grid-prior-no-joint"),
    ("nerfacto-nerfiller", "grid-prior"),
    ("nerfacto-nerfiller", "expanded-attention"),
    ("nerfacto-nerfiller", "individual-sd-image"),
    ("nerfacto-nerfiller", "individual-sd-text"),
    ("nerfacto-nerfiller", "individual-lama"),
]


@dataclass
class NeRFBlenderToNerfstudio(ExperimentConfig):
    """Convert NeRF synthetic blender datasets to Nerfstudio datasets"""

    dataset_names: List[str] = field(default_factory=lambda: blender_dataset_names)

    def main(self, dry_run: bool = False):
        jobs = []
        experiment_names = []
        argument_combinations = []

        for dataset_name in self.dataset_names:
            jobs.append(
                f"python nerfiller/scripts/blender_to_nerfstudio.py --input-folder data/blender/{dataset_name} --output-folder data/nerfstudio/{dataset_name}"
            )

        launch_experiments(jobs, dry_run=dry_run, gpu_ids=self.gpu_ids)


@dataclass
class InpaintBlender(Inpaint):
    """Inpaint Nerfstudio datasets"""

    dataset_names: List[str] = field(default_factory=lambda: blender_dataset_names)
    inpaint_methods: List[str] = field(default_factory=lambda: inpaint_methods)


@dataclass
class TrainBlender(Train):
    """Train Nerfacto on different inpainted datasets."""

    dataset_names: List[str] = field(default_factory=lambda: blender_dataset_names)
    methods: List[str] = field(default_factory=lambda: methods)
    gpus_per_job: int = 1


@dataclass
class RenderBlender(Render):
    """Render the blender scenes."""

    dataset_names: List[str] = field(default_factory=lambda: blender_dataset_names)
    methods: List[str] = field(default_factory=lambda: methods)


@dataclass
class MetricsBlender(Metrics):
    """Compute metrics for the blender scenes."""

    dataset_names: List[str] = field(default_factory=lambda: blender_dataset_names)
    methods: List[str] = field(default_factory=lambda: methods)
