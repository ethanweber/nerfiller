import torch

from nerfiller.utils.camera_utils import (
    c2wh_from_c2w,
    get_perspective_directions,
    get_equirectangular_directions,
    rot_x,
    rot_y,
    rot_z,
)
from nerfiller.utils.TextureShader import TextureShader
from nerfiller.utils.typing import *
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
    Textures,
)
from pytorch3d.structures import Meshes


class MeshRendererWithDepth(torch.nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf, fragments.pix_to_face


def get_mesh_from_perspective_images(
    images: Float[Tensor, "bs 3 H W"],
    distance: Float[Tensor, "bs 1 H W"],
    fov_x: float,
    fov_y: float,
    angle_threshold: Optional[float] = None,
    distance_threshold: Optional[float] = None,
    c2w: Optional[Float[Tensor, "3 4"]] = None,
) -> Tuple[Float[Tensor, "N 3"], Float[Tensor, "N 3"], Float[Tensor, "M 3"]]:
    """Returns a mesh from a perspective image.
    Fov is in radians.
    Assumes Blender camera conventions.
        - x right
        - y up
        - z back
    """

    device = images.device
    bs, _, H, W = images.shape
    assert bs == 1
    direction = (get_perspective_directions(H, W, fov_x, fov_y, device="cpu").unsqueeze(0).repeat(bs, 1, 1, 1)).to(
        device
    )
    vertices = direction * distance

    vertex_ids = torch.arange(bs * H * W).reshape(bs, 1, H, W).to(device)
    vertex_00 = vertex_ids[:, :, : H - 1, : W - 1]
    vertex_01 = vertex_00 + 1
    vertex_10 = vertex_00 + W
    vertex_11 = vertex_00 + W + 1

    # faces
    faces_ul = torch.cat([vertex_00, vertex_10, vertex_01], dim=1)
    faces_lr = torch.cat([vertex_10, vertex_11, vertex_01], dim=1)

    mask_ul = torch.ones_like(faces_ul[:, 0, :, :]) == 1.0
    mask_lr = torch.ones_like(faces_lr[:, 0, :, :]) == 1.0

    if angle_threshold:
        # face directions
        # NOTE(ethan): this might not be correct
        # faces_dir = 0.5 * (direction[:, :, :-1, :-1] + direction[:, :, 1:, 1:])
        # faces_dir = faces_dir / torch.norm(faces_dir, dim=1, keepdim=True)

        # faces normals
        faces_dir_ul = direction + torch.roll(direction, -1, dims=-1) + torch.roll(direction, -1, dims=-2)
        faces_dir_ul = faces_dir_ul[:, :, :-1, :-1]
        faces_dir_ul = faces_dir_ul / torch.norm(faces_dir_ul, dim=1, keepdim=True)
        faces_dir_lr = (
            torch.roll(direction, (-1, -1), dims=(-1, -2))
            + torch.roll(direction, -1, dims=-1)
            + torch.roll(direction, -1, dims=-2)
        )
        faces_dir_lr = faces_dir_lr[:, :, :-1, :-1]  # drop last row
        faces_dir_lr = faces_dir_lr / torch.norm(faces_dir_lr, dim=1, keepdim=True)

        # faces normals
        faces_nor_ul = get_face_normals(vertices, faces_ul)
        faces_nor_lr = get_face_normals(vertices, faces_lr)

        dot_ul = torch.abs(torch.sum(faces_dir_ul * faces_nor_ul, dim=1))
        dot_lr = torch.abs(torch.sum(faces_dir_lr * faces_nor_lr, dim=1))

        ang_ul = torch.acos(dot_ul)
        ang_lr = torch.acos(dot_lr)
        angle_threshold_rad = angle_threshold * (torch.pi / 180.0)
        mask_ul &= torch.isnan(ang_ul) | (ang_ul < angle_threshold_rad)
        mask_lr &= torch.isnan(ang_lr) | (ang_lr < angle_threshold_rad)

    faces = torch.cat([faces_ul.permute(0, 2, 3, 1), faces_lr.permute(0, 2, 3, 1)]).reshape(-1, 3)
    faces_mask = torch.cat([mask_ul, mask_lr]).reshape(-1)

    vertices = vertices.permute(0, 2, 3, 1).reshape(-1, 3)
    vertex_colors = images.clone().permute(0, 2, 3, 1).reshape(-1, 3)

    # apply rotation
    rot = torch.tensor([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).float().to(device)
    vertices = (rot @ vertices.permute(1, 0)).permute(1, 0)

    if c2w is not None:
        verticesh = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1)
        vertices = torch.matmul(c2w, verticesh.permute(1, 0)).permute(1, 0)

    return vertices, vertex_colors, faces, faces_mask


def get_mesh_from_equirectangular_images(
    images: Float[Tensor, "bs 3 H W"],
    distance: Float[Tensor, "bs 1 H W"],
    angle_threshold: Optional[float] = None,
    distance_threshold: Optional[float] = None,
    origins: Optional[Float[Tensor, "bs 3"]] = None,
    discard_distance_value: Optional[float] = None,
) -> Tuple[Float[Tensor, "N 3"], Float[Tensor, "N 3"], Float[Tensor, "M 3"]]:
    """Returns a mesh from an equirectangular image."""
    device = images.device
    bs, _, H, W = images.shape
    assert bs == 1
    direction = (
        get_equirectangular_directions(H, W, theta_fov=2 * torch.pi, phi_fov=torch.pi, device="cpu")
        .unsqueeze(0)
        .repeat(bs, 1, 1, 1)
    ).to(device)
    vertices = direction * distance

    # TODO(ethan): add extrinsics here
    if origins is not None:
        vertices = vertices + origins[:, :, None, None]

    vertex_ids = torch.arange(bs * H * W).reshape(bs, 1, H, W).to(device)
    vertex_00 = vertex_ids[:, :, : H - 1, :W]
    vertex_01 = vertex_00 + 1
    # wrap around because its equirectangular
    vertex_01 = torch.where(vertex_01 % W == 0, vertex_01 - W, vertex_01)
    vertex_10 = vertex_00 + W
    vertex_11 = vertex_00 + W + 1
    # wrap around because its equirectangular
    vertex_11 = torch.where(vertex_11 % W == 0, vertex_11 - W, vertex_11)

    # faces
    faces_ul = torch.cat([vertex_00, vertex_10, vertex_01], dim=1)
    faces_lr = torch.cat([vertex_10, vertex_11, vertex_01], dim=1)

    mask_ul = torch.ones_like(faces_ul[:, 0, :, :]) == 1.0
    mask_lr = torch.ones_like(faces_lr[:, 0, :, :]) == 1.0

    # remove faces where distance values should be discarded
    if discard_distance_value:
        mask_ul &= distance[:, 0, :-1, :] != discard_distance_value
        mask_lr &= torch.roll(torch.roll(distance, -1, dims=-1), -1, dims=-1)[:, 0, :-1, :] != discard_distance_value

    if angle_threshold:
        # face directions
        # NOTE(ethan): this might not be correct
        # faces_dir = 0.5 * (direction[:, :, :-1, :-1] + direction[:, :, 1:, 1:])
        # faces_dir = faces_dir / torch.norm(faces_dir, dim=1, keepdim=True)

        # faces normals
        faces_dir_ul = direction + torch.roll(direction, -1, dims=-1) + torch.roll(direction, -1, dims=-2)
        faces_dir_ul = faces_dir_ul[:, :, :-1, :]  # drop last row
        faces_dir_ul = faces_dir_ul / torch.norm(faces_dir_ul, dim=1, keepdim=True)
        faces_dir_lr = (
            torch.roll(direction, (-1, -1), dims=(-1, -2))
            + torch.roll(direction, -1, dims=-1)
            + torch.roll(direction, -1, dims=-2)
        )
        faces_dir_lr = faces_dir_lr[:, :, :-1, :]  # drop last row
        faces_dir_lr = faces_dir_lr / torch.norm(faces_dir_lr, dim=1, keepdim=True)

        # faces normals
        faces_nor_ul = get_face_normals(vertices, faces_ul)
        faces_nor_lr = get_face_normals(vertices, faces_lr)

        dot_ul = torch.abs(torch.sum(faces_dir_ul * faces_nor_ul, dim=1))
        dot_lr = torch.abs(torch.sum(faces_dir_lr * faces_nor_lr, dim=1))

        ang_ul = torch.acos(dot_ul)
        ang_lr = torch.acos(dot_lr)
        angle_threshold_rad = angle_threshold * (torch.pi / 180.0)
        mask_ul &= torch.isnan(ang_ul) | (ang_ul < angle_threshold_rad)
        mask_lr &= torch.isnan(ang_lr) | (ang_lr < angle_threshold_rad)

    if distance_threshold:
        faces_max_edge_dist_ul = get_face_max_edge_distances(vertices, faces_ul)
        faces_max_edge_dist_lr = get_face_max_edge_distances(vertices, faces_lr)
        mask_ul &= faces_max_edge_dist_ul <= distance_threshold
        mask_lr &= faces_max_edge_dist_lr <= distance_threshold

    faces = torch.cat([faces_ul.permute(0, 2, 3, 1), faces_lr.permute(0, 2, 3, 1)]).reshape(-1, 3)
    faces_mask = torch.cat([mask_ul, mask_lr]).reshape(-1)

    vertex_colors = images.clone()

    vertices = vertices.permute(0, 2, 3, 1).reshape(-1, 3)
    vertex_colors = vertex_colors.permute(0, 2, 3, 1).reshape(-1, 3)
    faces = faces.reshape(-1, 3)

    return vertices, vertex_colors, faces, faces_mask


def get_face_normals(
    vertices: Float[Tensor, "bs 3 H W"], faces: Float[Tensor, "bs 3 H W"]
) -> Float[Tensor, "bs 3 H W"]:
    """Get the face normals."""
    face_pos = vertices.permute(0, 2, 3, 1).reshape(-1, 3)[faces]
    ab = face_pos[:, 1] - face_pos[:, 0]
    ab = ab / torch.norm(ab, dim=-1, keepdim=True)
    ac = face_pos[:, 2] - face_pos[:, 0]
    ac = ac / torch.norm(ac, dim=-1, keepdim=True)
    normals = torch.cross(ab, ac, dim=-1)
    normals = normals / torch.norm(normals, dim=-1, keepdim=True)
    normals = normals.permute(0, 3, 1, 2)
    return normals


def get_face_max_edge_distances(
    vertices: Float[Tensor, "bs 3 H W"], faces: Float[Tensor, "bs 3 H W"]
) -> Float[Tensor, "bs H W"]:
    """Get the face normals."""
    face_pos = vertices.permute(0, 2, 3, 1).reshape(-1, 3)[faces]
    ab = face_pos[:, 1] - face_pos[:, 0]
    ab_dist = torch.norm(ab, dim=-1, keepdim=True)
    ac = face_pos[:, 2] - face_pos[:, 0]
    ac_dist = torch.norm(ac, dim=-1, keepdim=True)
    bc = face_pos[:, 2] - face_pos[:, 1]
    bc_dist = torch.norm(bc, dim=-1, keepdim=True)
    dist = torch.cat([ab_dist, ac_dist, bc_dist], dim=-1).permute(0, 3, 1, 2)
    dist = torch.max(dist, dim=1, keepdim=False).values
    return dist


def get_transformed_vertices(vertices: Float[Tensor, "N 3"], c2w=None, device=None):
    if c2w is None:
        return vertices

    # compute world to camera transformation
    c2wh = c2wh_from_c2w(c2w).to(device)
    w2ch = torch.inverse(c2wh)

    # apply the transformation
    transformed_vertices = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1)
    transformed_vertices = (w2ch @ transformed_vertices.permute(1, 0)).permute(1, 0)
    transformed_vertices = transformed_vertices[:, :3]
    return transformed_vertices


def project_mesh_into_perspective_image(
    vertices: Float[Tensor, "N 3"],
    colors: Union[Float[Tensor, "N 3"], None],
    faces: Float[Tensor, "M 3"],
    fov: float,
    image_size: int,
    c2w: Optional[Float[Tensor, "3 4"]] = None,
    faces_per_pixel=1,
    textures: Optional[Textures] = None,
    cull_backfaces: bool = True,
    device=None,
):
    """
    Projects a mesh into a camera. We only render front-facing triangles ordered in an anti-clockwise fashion.
    """
    if device is None:
        device = vertices.device

    assert device != "cpu", "Rendering with cpu will be slow!"

    transformed_vertices = get_transformed_vertices(vertices, c2w=c2w, device=device)

    if textures is None:
        assert colors is not None
        meshes = Meshes(
            verts=[transformed_vertices],
            faces=[faces],
            textures=Textures(verts_rgb=colors.unsqueeze(0)),
        )
    else:
        meshes = Meshes(verts=[transformed_vertices], faces=[faces], textures=textures)

    R = torch.eye(3).unsqueeze(0)
    T = torch.zeros(3).unsqueeze(0)

    # rotate 180 degrees around Y axis
    m = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).unsqueeze(0).float()
    R = torch.bmm(m, R)

    cameras = FoVPerspectiveCameras(R=R, T=T, fov=fov, device=device, znear=1e-6)

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=faces_per_pixel,
        cull_backfaces=cull_backfaces,
    )
    renderer = MeshRendererWithDepth(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=TextureShader(device=device, cameras=cameras),
    )

    images, depths, pix_to_face = renderer(meshes)
    image = images[0, :, :, :3]
    depth = depths[0, :, :, 0]

    return image, depth, pix_to_face[0]


def dilate(tensor, kernel_size=3):
    stride = 1
    padding = (kernel_size - 1) // 2
    return torch.nn.functional.max_pool2d(tensor, kernel_size, stride, padding)


def erode(tensor, kernel_size=3, keep_borders=False):
    x = 1 - tensor
    padding = (kernel_size - 1) // 2
    x = dilate(x, kernel_size=kernel_size)
    x = 1 - x
    if keep_borders:
        x[:, :, :padding, :] *= 0
        x[:, :, :, :padding] *= 0
        x[:, :, -padding:, :] *= 0
        x[:, :, :, -padding:] *= 0
    return x


def get_cube(
    center: Float[Tensor, "3"] = torch.tensor([0, 0, 0]),
    scale: Float[Tensor, "3"] = torch.tensor([1, 1, 1]),
    rotation: Float[Tensor, "3"] = torch.tensor([0, 0, 0]),
    device=None,
):
    # Define the vertices of the cube
    vertices = (
        torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=torch.float32,
            device=device,
        )
        * 2
        - 1
    )

    # Define the faces of the cube with triangular faces (each row represents a face by referring to vertex indices)
    faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 6, 2],
            [3, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 7, 3],
            [0, 4, 7],
        ],
        dtype=torch.int32,
        device=device,
    )

    rot = torch.tensor(rot_x(rotation[0].item())) @ (
        torch.tensor(rot_y(rotation[1].item())) @ torch.tensor(rot_z(rotation[2].item()))
    )
    vertices = (rot.to(vertices) @ vertices.permute(1, 0)).permute(1, 0)
    vertices = (vertices * scale.to(vertices)) + center.to(vertices)

    return vertices, faces
