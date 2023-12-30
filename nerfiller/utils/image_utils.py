from PIL import Image
from nerfstudio.utils.colormaps import apply_colormap, ColormapOptions
import wandb
import mediapy
import numpy as np
import torch
import math

from nerfiller.utils.typing import *

from enum import Enum
from typing import Tuple
import cv2


class Colors(Enum):
    NEON_PINK = (0.9, 0.4, 0.9)
    NEON_GREEN = (0.0, 1.0, 0.96)


def get_image_grid(images: List[Float[Tensor, "H W C"]], rows: int = None, cols: int = None):
    """Returns a grid of images."""

    def get_image(images, idx):
        # returns white if out of bounds
        if idx < len(images):
            return images[idx]
        else:
            return torch.ones_like(images[0])

    if cols is None:
        assert rows is not None
        cols = math.ceil(len(images) / rows)

    if rows is None:
        assert cols is not None
        rows = math.ceil(len(images) / cols)

    im_rows = []
    idx = 0
    for i in range(rows):
        im_row = []
        for j in range(cols):
            im_row.append(get_image(images, idx))
            idx += 1
        im_rows.append(torch.cat(im_row, dim=1))
    im = torch.cat(im_rows, dim=0)
    return im


def image_tensor_to_npy(image_tensor: Float[Tensor, "B C H W"]):
    """Returns a list of numpy images of shape [H, W, C]."""
    return list(image_tensor.permute(0, 2, 3, 1).detach().cpu().numpy())


def image_tensor_to_list(image_tensor: Float[Tensor, "B C H W"]):
    """Returns a list of torch images of shape [H, W, C]."""
    return list(image_tensor.permute(0, 2, 3, 1).detach().cpu())


def to_pil(im):
    return Image.fromarray((im * 255.0).numpy().astype("uint8"))


def ten_to_npy(tensor, normalize=False):
    if normalize:
        return apply_colormap(tensor.permute(1, 2, 0), ColormapOptions(normalize=True)).detach().cpu().numpy()
    return tensor.permute(1, 2, 0).detach().cpu().numpy()


def get_inpainted_image_row(
    image: Float[Tensor, "B 3 H W"],
    mask: Float[Tensor, "B 1 H W"],
    inpainted_image: Optional[Float[Tensor, "B 3 H W"]] = None,
    color: Tuple[float, float, float] = Colors.NEON_PINK.value,
    show_original: bool = False,
):
    """Returns an image concatenated along the x-axis. It has the following form:
        image with inpaint regions highlighted | image with inpainted regions
    Inpaint where mask == 1.
    The default color is neon pink.
    If the inpainted image is None, then just show the `image with inpaint regions highlighted`.
    """
    device = image.device
    c = torch.tensor(color, device=device).view(1, 3, 1, 1)
    color_image = torch.ones_like(image) * c
    image_with_highlights = torch.where(mask == 1, color_image, image)
    image_list = [image_with_highlights]
    if inpainted_image is not None:
        image_list = image_list + [inpainted_image]
    if show_original:
        image_list = [image] + image_list
    im = torch.cat(image_list, dim=-2)
    return im


def show_images(images_list, height=100, name="images"):
    """Show images with either mediapy (in a notebook) or wandb.
    The images list will be processed into one 3 channel numpy image of shape H,W*len(images_list),3.
    We concatenate along the x-axis.
    We convert the images to uin8.

    Args:
        image_list: A list of image tensors of shape H,W,3 between 0 and 1 or of shape H,W,1.
            The 1 channel images will be normalized.
        height: The height of the images.
    """
    images_list_processed = []
    for image in images_list:
        num_channels = image.shape[-1]
        if num_channels == 1:
            if len(torch.unique(image)) <= 2:
                # binary mask
                im = image.repeat(1, 1, 3).detach().cpu().numpy()
            else:
                # non-binary, so apply a mask
                im = apply_colormap(image, ColormapOptions(normalize=True)).detach().cpu().numpy()
        else:
            im = image.detach().cpu().numpy()
        im = (im * 255).astype("uint8")
        images_list_processed.append(im)
    image_row = np.concatenate(images_list_processed, axis=1)
    mediapy.show_image(image_row, height=height)
    wandb.log({name: wandb.Image(image_row)})


def save_video_from_path(path: Path, glob_str: str, sec: int = 10, output_filename: str = "video.mp4"):
    filenames = sorted(path.glob(glob_str))
    frames = []
    for filename in filenames:
        image = mediapy.read_image(filename)
        # Define the position and font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # White color in BGR
        font_thickness = 2
        position = (10, 30)  # (x, y) coordinates
        image = cv2.putText(
            image,
            str(filename.name),
            position,
            font,
            font_scale,
            font_color,
            font_thickness,
        )
        frames.append(image)
    fps = len(frames) / sec
    mediapy.write_video(output_filename, frames, fps=fps)
