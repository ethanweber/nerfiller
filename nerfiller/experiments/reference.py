"""
Application where we inpaint one image and use it as a reference to guide the others.
"""

from dataclasses import dataclass, field

from nerfiller.experiments.base import Train, Render
from nerfiller.utils.typing import *
from nerfiller.experiments.data import occluder_dataset_names

methods = [
    ("grid-prior-du-reference", "none"),
]


@dataclass
class TrainReference(Train):
    """Train our method with reference images."""

    dataset_names: List[str] = field(default_factory=lambda: occluder_dataset_names)
    methods: List[str] = field(default_factory=lambda: methods)


@dataclass
class RenderReference(Render):
    """Render our method that used reference images."""

    dataset_names: List[str] = field(default_factory=lambda: occluder_dataset_names)
    methods: List[str] = field(default_factory=lambda: methods)
