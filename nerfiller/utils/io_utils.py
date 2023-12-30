import os
from struct import unpack

import numpy as np
import open3d as o3d
import trimesh

from nerfiller.utils.typing import *


def read_dpt(dpt_file_path):
    """read depth map from *.dpt file.

    :param dpt_file_path: the dpt file path
    :type dpt_file_path: str
    :return: depth map data
    :rtype: numpy
    """
    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(dpt_file_path)[1]

    assert len(ext) > 0, "readFlowFile: extension required in fname %s" % dpt_file_path
    assert ext == ".dpt", exit("readFlowFile: fname %s should have extension " ".flo" "" % dpt_file_path)

    fid = None
    try:
        fid = open(dpt_file_path, "rb")
    except IOError:
        print("readFlowFile: could not open %s", dpt_file_path)

    tag = unpack("f", fid.read(4))[0]
    width = unpack("i", fid.read(4))[0]
    height = unpack("i", fid.read(4))[0]

    assert tag == TAG_FLOAT, "readFlowFile(%s): wrong tag (possibly due to big-endian machine?)" % dpt_file_path
    assert 0 < width and width < 100000, "readFlowFile(%s): illegal width %d" % (
        dpt_file_path,
        width,
    )
    assert 0 < height and height < 100000, "readFlowFile(%s): illegal height %d" % (
        dpt_file_path,
        height,
    )

    # arrange into matrix form
    depth_data = np.fromfile(fid, np.float32)
    depth_data = depth_data.reshape(height, width)

    fid.close()

    return depth_data


def save_pointcloud(
    vertices: Float[Tensor, "N 3"],
    colors: Float[Tensor, "N 3"],
    filename="pointcloud.ply",
):
    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices.detach().cpu().numpy())

    # Assign colors from the images to the point cloud
    point_cloud.colors = o3d.utility.Vector3dVector(colors.detach().cpu().numpy())

    # Save the point cloud to a file
    o3d.io.write_point_cloud(filename, point_cloud)


def save_mesh(
    vertices: Float[Tensor, "N 3"],
    vertex_colors: Float[Tensor, "N 3"],
    faces: Float[Tensor, "M 3"],
    filename="mesh.ply",
):
    mesh = trimesh.Trimesh(
        vertices=vertices.detach().cpu().numpy(),
        faces=faces.detach().cpu().numpy(),
        vertex_colors=vertex_colors.detach().cpu().numpy(),
    )
    mesh.remove_unreferenced_vertices()
    mesh.export(filename)
    return mesh
