"""
Code to create grids from images and go back and forth.
"""

import torch


def make_grid(tensors):
    """
    The batch size needs to be divisible by 4.
    Wraps with row major format.
    """
    batch_size, C, H, W = tensors.shape
    assert batch_size % 4 == 0
    num_grids = batch_size // 4
    t = tensors.view(num_grids, 4, C, H, W).transpose(0, 1)
    tensor = torch.cat(
        [
            torch.cat([t[0], t[1]], dim=-1),
            torch.cat([t[2], t[3]], dim=-1),
        ],
        dim=-2,
    )
    return tensor


def undo_grid(tensors):
    batch_size, C, H, W = tensors.shape
    num_squares = batch_size * 4
    hh = H // 2
    hw = W // 2
    t = tensors.view(batch_size, C, 2, hh, 2, hw).permute(0, 2, 4, 1, 3, 5)
    t = t.reshape(num_squares, C, hh, hw)
    return t


def test_grid_utils():
    """Test the grid utils visually. You can run this in a notebook."""
    import mediapy

    images = torch.ones((12, 3, 50, 50)) * torch.rand((12, 3, 1, 1))
    mediapy.show_images(images.permute(0, 2, 3, 1).detach().cpu())
    mg = make_grid(images)
    mediapy.show_images(mg.permute(0, 2, 3, 1).detach().cpu())
    ug = undo_grid(mg)
    mediapy.show_images(ug.permute(0, 2, 3, 1).detach().cpu())

    assert (images - ug).sum() == 0.0
