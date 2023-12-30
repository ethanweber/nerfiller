import tyro
from typing import Optional
import torch
from pathlib import Path
import mediapy
from nerfiller.utils.image_utils import image_tensor_to_list, get_image_grid
import math


def main(
    output_dir: Path = Path("data/nerfstudio"),
    dataset_name: str = "village",
    inpaint_method: Optional[str] = None,
    radius: int = 1,
    seconds_per_frame: float = 0.5,
    device: str = "cpu",
):
    """
    Visualize a Nerfstudio dataset.

    Args:
        output_dir: Output directory path.
        dataset_name: Name of the dataset.
        inpaint_method: Inpainting method.
        device: Device to use for computation (e.g., 'cuda:0' or 'cpu').
    """

    # Check that the dataset exists and make new output folders.
    dn = Path(str(dataset_name) + f"{radius}") if radius != 1 else Path(dataset_name)
    output_folder = output_dir / dn
    assert output_folder.exists(), f"Output folder '{output_folder}' does not exist"

    folder_name = f"{dn}-{inpaint_method}" if inpaint_method is not None else dn
    data_folder = output_dir / folder_name

    images_folder = data_folder / "images"
    depth_folder = output_folder / "depth"
    masks_folder = output_folder / "masks"
    assert images_folder.exists(), f"Images folder '{images_folder}' does not exist"
    assert depth_folder.exists(), f"Depth folder '{depth_folder}' does not exist"
    assert masks_folder.exists(), f"Masks folder '{masks_folder}' does not exist"

    print(images_folder)

    image_files = sorted(list(images_folder.glob("*.png")))
    depth_files = sorted(list(depth_folder.glob("*.npy")))
    mask_files = sorted(list(masks_folder.glob("*.png")))
    num_images = len(image_files)

    images = []
    depths = []
    masks = []
    for i in range(num_images):
        print(i)

        # Load the images
        image = torch.from_numpy(mediapy.read_image(image_files[i]) / 255.0).to(device).permute(2, 0, 1)[None].float()
        # depth = torch.from_numpy(np.load(depth_files[i])).to(device)[None, None].float()
        mask = torch.from_numpy(mediapy.read_image(mask_files[i]) / 255.0).to(device)[None, None].float()

        if inpaint_method is None:
            image *= mask

        images.append(image)
        # depths.append(depth)
        # masks.append(masks)

        height, width = image.shape[-2], image.shape[-1]

        # idx = int(str(image_files[i])[str(image_files[i]).find("image_") + 6:str(image_files[i]).find(".png")])
        # mediapy.write_image(inpainted_images_folder / f"image_{idx:06d}.png", edited_image_HW3.detach().cpu())
        # mediapy.write_image(inpainted_depth_folder / f"depth_{idx:06d}.png", edited_depth_with_colormap_HW3)
        # np.save(inpainted_depth_folder / f"depth_{idx:06d}.npy", edited_depth_HW1[..., 0].detach().cpu().numpy())

    images_list = image_tensor_to_list(torch.cat(images))
    cols = math.ceil(math.sqrt(num_images) * 16 / 9)
    image_grid = get_image_grid(images_list, cols=cols)

    print("Saving images")
    mediapy.write_image(data_folder / Path(str(folder_name) + "_images.png"), image_grid)
    print("Saving video")
    images_list_np = [x.numpy() for x in images_list]
    mediapy.write_video(
        data_folder / Path(str(folder_name) + "_images.mp4"),
        images_list_np,
        fps=1 / seconds_per_frame,
        qp=18,
    )


if __name__ == "__main__":
    tyro.cli(main)
