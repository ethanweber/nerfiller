"""Script to download data."""
from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import gdown
import tyro
from typing_extensions import Annotated

from nerfstudio.configs.base_config import PrintableConfig


@dataclass
class DataDownload(PrintableConfig):
    """Download a dataset"""

    capture_name = None

    save_dir: Path = Path("data/")
    """The directory to save the dataset to"""

    def download(self, save_dir: Path) -> None:
        """Download the dataset"""
        raise NotImplementedError


@dataclass
class BlenderDownload(DataDownload):
    """Download the blender dataset."""

    def download(self, save_dir: Path):
        """Download the blender dataset."""
        # TODO: give this code the same structure as download_nerfstudio

        # https://drive.google.com/uc?id=152kFJoxu5x1fblDmJhhAI6Q5XhpopUHM
        blender_file_id = "152kFJoxu5x1fblDmJhhAI6Q5XhpopUHM"

        final_path = save_dir / Path("blender")
        if os.path.exists(final_path):
            shutil.rmtree(str(final_path))
        url = f"https://drive.google.com/uc?id={blender_file_id}"
        download_path = save_dir / "blender_data.zip"
        gdown.download(url, output=str(download_path))
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(str(save_dir))
        unzip_path = save_dir / Path("nerf_synthetic")
        final_path = save_dir / Path("blender")
        unzip_path.rename(final_path)
        if download_path.exists():
            download_path.unlink()


@dataclass
class MeshesDownload(DataDownload):
    """Download the blender dataset."""

    def download(self, save_dir: Path):
        """Download the blender dataset."""

        url = "https://drive.google.com/drive/u/0/folders/10S4YYf-i31k9G_5i4Imqlge5kdSaVgOc"
        download_path = save_dir / Path("meshes")
        gdown.download_folder(url=url, output=str(download_path), quiet=False, use_cookies=False)


@dataclass
class NerfstudioDownload(DataDownload):
    """Download the nerfstudio dataset."""

    def download(self, save_dir: Path):
        """Download the nerfstudio dataset."""

        url = "https://drive.google.com/drive/folders/1WNe2ahkSgNip6meeZ5qhFB4CZWIAIGSE?usp=sharing"
        download_path = save_dir / Path("nerfstudio")
        gdown.download_folder(url=url, output=str(download_path), quiet=False, use_cookies=False)

        # get all the .zip files inside the nerfstudio folder, unzip, and remove the zip
        for file in download_path.glob("*.zip"):
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(str(download_path))
            file.unlink()


@dataclass
class CameraPathsDownload(DataDownload):
    """Download the camera paths."""

    def download(self, save_dir: Path):
        """Download the camera paths."""

        url = "https://drive.google.com/drive/folders/1Co75ghbaJPAws1JuKnIEpCeUI2rKvhf1?usp=sharing"
        download_path = save_dir / Path("camera_paths")
        gdown.download_folder(url=url, output=str(download_path), quiet=False, use_cookies=False)

@dataclass
class ModelsDownload(DataDownload):
    """Download pretrained model weights."""

    def download(self, save_dir: Path):
        """Download the camera paths."""

        url = "https://drive.google.com/drive/folders/1VOhG0sKrth9D808413aE0HhxQ5iadRdq?usp=sharing"
        download_path = save_dir / Path("models")
        gdown.download_folder(url=url, output=str(download_path), quiet=False, use_cookies=False)

Commands = Union[
    Annotated[BlenderDownload, tyro.conf.subcommand(name="blender")],
    Annotated[MeshesDownload, tyro.conf.subcommand(name="meshes")],
    Annotated[NerfstudioDownload, tyro.conf.subcommand(name="nerfstudio")],
    Annotated[CameraPathsDownload, tyro.conf.subcommand(name="camera-paths")],
    Annotated[ModelsDownload, tyro.conf.subcommand(name="models")],
]


def main(
    data_download: DataDownload,
):
    """Script to download data.
    - blender: Blender synthetic scenes realeased with NeRF.
    - meshes: Meshes to apply NeRFiller to.

    Args:
        dataset: The dataset to download (from).
    """
    data_download.save_dir.mkdir(parents=True, exist_ok=True)

    data_download.download(data_download.save_dir)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Commands))


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa
