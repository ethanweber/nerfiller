"""
3D Occluder Experiments. These are baseline comparisons.
"""

from dataclasses import dataclass, field

from nerfiller.experiments.base import Inpaint, Train, Render, Metrics
from nerfiller.utils.typing import *
from nerfiller.experiments.data import occluder_dataset_names

inpaint_methods = [
    "individual-sd-image",
    "individual-lama",
]

methods = [
    ("nerfacto-nerfiller-visualize", "none"),
    ("nerfacto-nerfiller", "none"), # this must be trained first, as the following depend on it
    ("individual-inpaint-once", "individual-sd-image"),
    ("individual-inpaint-once", "individual-lama"),
    ("grid-prior-du-no-depth", "none"),
    ("grid-prior-du", "none"),
    ("individual-inpaint-du", "none"),
    ("grid-prior-du-random-noise", "none"), # this is extra for an ablation
]


@dataclass
class InpaintOccluder(Inpaint):
    """Inpaint Nerfstudio datasets."""

    dataset_names: List[str] = field(default_factory=lambda: occluder_dataset_names)
    inpaint_methods: List[str] = field(default_factory=lambda: inpaint_methods)


@dataclass
class TrainOccluder(Train):
    """Train the baselines and our method."""

    dataset_names: List[str] = field(default_factory=lambda: occluder_dataset_names)
    methods: List[str] = field(default_factory=lambda: methods)


@dataclass
class RenderOccluder(Render):
    """Render the baselines and our method."""

    dataset_names: List[str] = field(default_factory=lambda: occluder_dataset_names)
    methods: List[str] = field(default_factory=lambda: methods)


@dataclass
class MetricsOccluder(Metrics):
    """Compute metrics on the baselines."""

    dataset_names: List[str] = field(default_factory=lambda: occluder_dataset_names)
    methods: List[str] = field(default_factory=lambda: methods)
