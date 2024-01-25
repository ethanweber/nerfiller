"""
Base code for running experiments in parallel.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from nerfstudio.configs.base_config import PrintableConfig

from nerfstudio.utils.rich_utils import CONSOLE

from nerfiller.utils.typing import *
from nerfiller.experiments.data import *

from nerfiller.utils.launch_utils import launch_experiments


def get_dataname(dataset_name, dataset_inpaint_method, prefix=Path("data")):
    """Returns the folder of data to train the model with."""
    if dataset_inpaint_method == "none":
        dn = prefix / Path("nerfstudio/" + dataset_name)
    else:
        dn = sorted(
            list((prefix / Path("nerfstudio/" + dataset_name + "-inpaint") / dataset_inpaint_method).iterdir())
        )[-1]
    return dn


def get_loaddir(dataset_name, checkpoint_method, prefix=Path("outputs")):
    loaddir = str(sorted(list((prefix / Path(dataset_name + "-none" + "/" + checkpoint_method)).iterdir()))[-1])
    return loaddir


def get_experiment_folder(dataset_name, dataset_inpaint_method, method, prefix=Path("outputs")):
    experiment_name = dataset_name + "-" + dataset_inpaint_method
    folder = str(sorted(list((prefix / Path(experiment_name + "/" + method)).iterdir()))[-1])
    return folder


@dataclass
class ExperimentConfig(PrintableConfig):
    """Experiment config code."""

    dry_run: bool = False
    output_folder: Optional[Path] = None
    gpu_ids: Optional[List[int]] = None

    def main(self, dry_run: bool = False) -> None:
        """Run the code."""
        raise NotImplementedError


@dataclass
class EquiRGBDToNerfstudio(ExperimentConfig):
    """Convert 360 RGB-D image into Nerfstudio datasets"""

    dataset_names: List[str] = ("village", "bed")
    """Which datasets to convert."""
    dataset_types: List[str] = ("blender", "matterport")
    """Which dataset types the datasets are."""

    def main(self, dry_run: bool = False):
        jobs = []
        for dataset_name, dataset_type in zip(self.dataset_names, self.dataset_types):
            jobs.append(
                f"python nerfiller/scripts/equirgbd_to_nerfstudio_dataset.py --dataset-name {dataset_name} --dataset-type {dataset_type}"
            )

        launch_experiments(jobs, dry_run=dry_run, gpu_ids=self.gpu_ids)


@dataclass
class Inpaint(ExperimentConfig):
    """Inpaint Nerfstudio datasets"""

    dataset_names: List[str] = field(default_factory=lambda: occluder_dataset_names)
    """Which datasets to inpaint."""
    inpaint_methods: List[str] = (
        "individual-sd-image",
        "individual-lama",
    )
    """Which inpainting methods to use."""
    chunk_size: Optional[int] = None
    new_size: Optional[int] = None

    def main(self, dry_run: bool = False):
        jobs = []
        for dataset_name in self.dataset_names:
            for inpaint_method in self.inpaint_methods:
                command = f"ns-inpaint {inpaint_method} --nerfstudio-dataset data/nerfstudio/{dataset_name}"
                if inpaint_method == "individual-sd-text":
                    command += f' --prompt "{dataset_name_to_prompt[dataset_name]}"'
                if self.chunk_size:
                    command += f" --chunk-size {self.chunk_size}"
                if self.new_size:
                    command += f" --new-size {self.new_size}"
                jobs.append(command)
        launch_experiments(jobs, dry_run=dry_run, gpu_ids=self.gpu_ids)


@dataclass
class Train(ExperimentConfig):
    """Train methods."""

    dataset_names: List[str] = ("norway",)
    """Which dataset to train on."""
    methods: List[Tuple[str, str]] = (("grid-prior-du", "none"),)
    """Which methods to use and which dataset to start with (method, dataset inpaint method)."""
    gpus_per_job: int = 2
    """Number of gpus needed per job."""

    def main(self, dry_run: bool = False):
        jobs = []
        for method, dataset_inpaint_method in self.methods:
            for dataset_name in self.dataset_names:
                dn = get_dataname(dataset_name, dataset_inpaint_method)

                experiment_name = dataset_name + "-" + dataset_inpaint_method
                command = f"ns-train {method} --data {dn} --experiment-name {experiment_name} --viewer.websocket-port=8082 --viewer.quit_on_train_completion True --vis viewer --viewer.default-composite-depth False --pipeline.model.camera_optimizer.mode off --pipeline.datamanager.masks-on-gpu True --pipeline.datamanager.images-on-gpu True"
                if dataset_name in synthetic_dataset_names:
                    background_color = "black" if dataset_name in synthetic_black_background_dataset_names else "white"
                    command += f" --pipeline.model.background_color {background_color} --pipeline.model.disable_scene_contraction True --pipeline.model.distortion_loss_mult 0.0"

                if dataset_name in forward_facing_dataset_names:
                    command += f" {forward_facing_snippet}"

                if dataset_name in turn_off_view_dependence_dataset_names:
                    command += f" {turn_off_view_dependence_snippet}"

                if method.find("-lora-depth-aware") >= 0:
                    lora_model_path = sorted(list(Path(f"outputs/dreambooth/{dataset_name}/depth-aware").iterdir()))[-1]
                    command += f" --pipeline.lora-model-path {lora_model_path}"

                if method.find("-lora-train-dist") >= 0:
                    lora_model_path = sorted(list(Path(f"outputs/dreambooth/{dataset_name}/train-dist").iterdir()))[-1]
                    command += f" --pipeline.lora-model-path {lora_model_path}"

                if method.find("nerfacto") < 0 and dataset_name in nerfbusters_dataset_names:
                    # don't dilate the masks or we lose context
                    command += f" --pipeline.dilate-iters 0"

                if method.find("nerfacto") < 0 and dataset_name in synthetic_dataset_names:
                    # don't use depth supervision on synthetic scenes
                    command += f" --pipeline.model.use-depth-ranking False"

                # load from a checkpoint if not training a nerfacto model
                if method.find("nerfacto") < 0:
                    checkpoint_method = "nerfacto-nerfiller"
                    if method == "depth-refinement":
                        checkpoint_method = "grid-prior-du"
                        if dataset_name in nerfbusters_dataset_names:
                            checkpoint_method = "grid-prior-du-lora-train-dist"
                    if method == "texture-refinement":
                        # checkpoint_method = "depth-refinement"
                        checkpoint_method = "grid-prior-du"
                    loaddir = get_loaddir(dataset_name, checkpoint_method)
                    command += f" --load-dir {loaddir}/nerfstudio_models"

                dataset_name_method = dataset_name + "+" + method
                if dataset_name_method in dataset_name_method_modifications:
                    command += f" {dataset_name_method_modifications[dataset_name_method]}"

                if dataset_name in eval_all_dataset_names:
                    command += " nerfstudio-data --eval-mode=all"

                if dataset_name in nerfbusters_dataset_names:
                    command += " --orientation_method=none --center_method=none --auto_scale_poses=False"

                jobs.append(command)
        launch_experiments(jobs, dry_run=dry_run, gpu_ids=self.gpu_ids, gpus_per_job=self.gpus_per_job)


@dataclass
class Render(ExperimentConfig):
    """Render the results."""

    dataset_names: List[str] = ("norway",)
    """Which dataset to train on."""
    methods: List[Tuple[str, str]] = (("grid-prior-du", "none"),)
    """Which methods to use and which dataset to start with (method, dataset inpaint method)."""
    mode: str = "dataset+camera-path"
    """Which inpainting methods to use."""

    def main(self, dry_run: bool = False):
        jobs = []
        for method, dataset_inpaint_method in self.methods:
            for dataset_name in self.dataset_names:
                folder = get_experiment_folder(dataset_name, dataset_inpaint_method, method)
                load_config = folder + "/config.yml"
                output_path = folder + "/final-renders"
                for mode in self.mode.split("+"):
                    if mode == "dataset":
                        command = f"ns-render-nerfiller {mode} --load-config {load_config}"
                        if dataset_name in eval_all_dataset_names:
                            split = "test"
                        else:
                            split = "train+test"
                        command += f" --output-path {output_path} --split {split}"
                        command += f" --inference-near-plane {dataset_render_near_plane}"  # TODO: hardcoded
                    elif mode == "camera-path":
                        command = f"ns-render-nerfiller {mode} --load-config {load_config}"
                        camera_path_filename = Path("data/camera_paths") / Path(dataset_name + ".json")
                        output_path_mp4 = folder + "/camera_path.mp4"
                        command += f" --output-path {output_path_mp4} --camera-path-filename {camera_path_filename} --rendered-output-names rgb depth --depth-near-plane 0.0 --depth-far-plane 4.0"
                        if dataset_name in camera_path_inference_near_planes:
                            command += f" --inference-near-plane {camera_path_inference_near_planes[dataset_name]}"
                    else:
                        raise ValueError()
                    jobs.append(command)
        launch_experiments(jobs, dry_run=dry_run, gpu_ids=self.gpu_ids)


@dataclass
class Metrics(ExperimentConfig):
    """Compute metrics on the renders."""

    dataset_names: List[str] = ("norway",)
    """Which dataset to train on."""
    methods: List[Tuple[str, str]] = (("grid-prior-du", "none"),)
    """Which methods to use and which dataset to start with (method, dataset inpaint method)."""

    def main(self, dry_run: bool = False):
        jobs = []
        for method, dataset_inpaint_method in self.methods:
            for dataset_name in self.dataset_names:
                for split in ["train", "test"]:
                    folder = get_experiment_folder(dataset_name, dataset_inpaint_method, method)

                    input_folder = folder + f"/final-renders/{split}"
                    output_folder = folder + f"/final-renders/{split}-metrics"
                    novel_view_video = folder + f"/camera_path.mp4"

                    # check if folder exists
                    if not os.path.exists(input_folder):
                        CONSOLE.print(f"[red]Skipping {input_folder} since it doesn't exist!")
                        continue
                    if method.find("grid-prior-du") >= 0 or method.find("individual-inpaint-du") >= 0:
                        gt_rgb_folder = str(sorted(list(Path(folder + "/dataset").iterdir()))[-1]) + "/images"
                    else:
                        gt_rgb_folder = input_folder + "/gt-rgb"

                    command = f"python nerfiller/scripts/metrics.py nerfiller-metrics --input-folder {input_folder} --output-folder {output_folder} --novel-view-video {novel_view_video}"

                    extra_string = f" --gt-rgb-folder {gt_rgb_folder}"
                    command += extra_string
                    jobs.append(command)

        launch_experiments(jobs, dry_run=dry_run, gpu_ids=self.gpu_ids)


@dataclass
class Dreambooth(ExperimentConfig):
    """Fine-tune an inpainting model on a dataset."""

    dataset_names: List[str] = ("aloe", "table", "couch")
    """Which datasets to convert."""
    mask_types: List[str] = (
        "depth-aware",
        "train-dist",
    )

    def main(self, dry_run: bool = False):
        jobs = []
        for dataset_name in self.dataset_names:
            for mask_type in self.mask_types:
                if dataset_name in nerfbusters_dataset_names:
                    dataset_type = "nerfbusters"
                else:
                    raise ValueError()

                command = (
                    f"ns-dreambooth --dataset_type {dataset_type} --dataset_name {dataset_name} --mask_type {mask_type}"
                )
                jobs.append(command)

        launch_experiments(jobs, dry_run=dry_run, gpu_ids=self.gpu_ids)
