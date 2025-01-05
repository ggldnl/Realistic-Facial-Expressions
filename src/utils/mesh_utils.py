from pathlib import Path
import torch
import numpy as np
import trimesh
from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    FoVPerspectiveCameras,
    look_at_view_transform
)


def read_mesh(
        obj_path,
        loader='pytorch3d',
        mesh_drop_percent=None,
        mesh_face_count=None,
        aggression=None,
        normalize=False,
):
    """
    Reads a mesh file using either PyTorch3D or trimesh loaders.

    Args:
        obj_path (str or Path): Path to the mesh file
        loader (str): Mesh loader to use ('pytorch3d' or 'trimesh')
        mesh_drop_percent (float): Percentage of faces to drop (between 0.0 and 1.0)
        mesh_face_count (int): Target number of faces in simplified mesh, overrides mesh_drop_percent if provided
        aggression (int): Simplification aggressiveness, 0 (slow/quality) to 10 (fast/rough)
        normalize (bool): Whether to normalize vertex coordinates

    Returns:
        For pytorch3d loader: PyTorch3D Meshes object
        For trimesh loader: trimesh.Trimesh object

    Raises:
        NotImplementedError: If loader not supported
        ValueError: If file extension not supported or invalid parameters
    """
    if isinstance(obj_path, str):
        obj_path = Path(obj_path)

    # Validate file extension
    if obj_path.suffix not in ['.obj', '.off', '.ply']:
        raise ValueError(f"Unsupported file extension for {obj_path}.")

    match loader:
        case 'pytorch3d':
            # Load using appropriate PyTorch3D loader based on extension
            if obj_path.suffix == '.obj':
                verts, faces_idx, _ = load_obj(obj_path)
                faces = faces_idx.verts_idx
            else:  # .ply
                verts, faces = load_ply(obj_path)

            # Normalize if requested
            if normalize:
                center = verts.mean(0)
                scale = max((verts - center).abs().max(0)[0])
                verts = (verts - center) / scale

            # Create Meshes object
            mesh = Meshes(
                verts=[verts],
                faces=[faces]
            )
            return mesh

        case 'trimesh':
            # Load the mesh using trimesh
            mesh = trimesh.load_mesh(obj_path, process=True)

            # Simplify the mesh if mesh_drop_percent or mesh_face_count are provided
            if mesh_drop_percent or mesh_face_count:
                # Validate mesh_face_count
                if mesh_face_count is not None:
                    if mesh_face_count < 4:
                        raise ValueError(
                            f"Target face count ({mesh_face_count}) too low. Minimum is 4 faces for watertight mesh.")
                    if mesh_face_count > len(mesh.faces):
                        raise ValueError(
                            f"Target face count ({mesh_face_count}) exceeds original count ({len(mesh.faces)}).")

                # Validate percentage if mesh_face_count not provided
                else:
                    if mesh_drop_percent and not 0 < mesh_drop_percent <= 1:
                        raise ValueError("mesh_drop_percent must be between 0.0 and 1.0")

                if aggression and not 0 <= aggression <= 10:
                    raise ValueError("aggression must be integer in range [0, 10]")

                mesh = mesh.simplify_quadric_decimation(
                    percent=mesh_drop_percent,
                    face_count=mesh_face_count,
                    aggression=aggression
                )

            # Normalize if requested
            if normalize:
                mesh.vertices = mesh.vertices / mesh.vertices.max()

            return mesh

        case _:
            raise NotImplementedError(f"Loader '{loader}' not implemented. Use 'pytorch3d' or 'trimesh'.")

def tensor_to_mesh(vertices, faces, vertex_normals=None):
    """
    Creates a trimesh object from tensor data.

    Args:
        vertices (torch.Tensor): Vertex positions tensor of shape (N, 3)
        faces (torch.Tensor): Face indices tensor of shape (M, 3)
        vertex_normals (torch.Tensor, optional): Vertex normals tensor of shape (N, 3)

    Returns:
        trimesh.Trimesh: The created mesh object
    """

    # If force is True this is equivalent to calling t.detach().cpu().resolve_conj().resolve_neg().numpy()
    vertices_np = vertices.numpy(force=True)
    faces_np = faces.numpy(force=True)

    # Create the mesh
    mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np
    )

    # If vertex normals are provided, set them
    if vertex_normals is not None:
        vertex_normals_np = vertex_normals.numpy(force=True)
        mesh.vertex_normals = vertex_normals_np

    return mesh


def get_pytorch3d_renderer(
        image_size: int = 512,
        dist: float = 1.5,
        elev: float = 0.0,
        azim: float = 0.0,
        device: str = "cuda"
) -> MeshRenderer:
    """
    Creates a PyTorch3D renderer with specified parameters.

    Args:
        image_size: Size of the rendered image
        dist: Distance of camera from origin
        elev: Elevation angle in degrees
        azim: Azimuth angle in degrees
        device: Device to place renderer on

    Returns:
        MeshRenderer configured with specified parameters
    """
    # Get camera position based on angles
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)

    # Initialize perspective camera
    cameras = FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device,
        fov=60  # Field of view in degrees
    )

    # Configure rasterization settings
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Set up lighting
    lights = PointLights(
        device=device,
        location=[[0.0, 0.0, 2.0]],
        ambient_color=[[0.7, 0.7, 0.7]],
        diffuse_color=[[0.3, 0.3, 0.3]],
        specular_color=[[0.2, 0.2, 0.2]]
    )

    # Create renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    return renderer


def visualize_mesh(
        mesh,
        image_size: int = 512,
        dist: float = 1.5,
        elev: float = 0.0,
        azim: float = 0.0,
        save_path: str = None,
        show: bool = True
) -> np.ndarray:
    """
    Visualizes a 3D mesh using either PyTorch3D or Trimesh.

    Args:
        mesh: Either a PyTorch3D Meshes object or a Trimesh object
        image_size: Size of the output image
        dist: Distance of camera from origin
        elev: Elevation angle in degrees
        azim: Azimuth angle in degrees
        save_path: Path to save the rendered image (optional)
        show: Whether to display the image

    Returns:
        numpy array containing the rendered image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    match mesh:
        case Meshes():
            # Ensure mesh is on correct device
            mesh = mesh.to(device)

            # Create renderer and render
            renderer = get_pytorch3d_renderer(image_size, dist, elev, azim, device)
            images = renderer(mesh)
            image = images[0, ..., :3].cpu().numpy()

        case trimesh.Trimesh():
            # Convert to pytorch3d format for rendering
            verts = torch.tensor(mesh.vertices, device=device, dtype=torch.float32)
            faces = torch.tensor(mesh.faces, device=device, dtype=torch.int64)

            # Create white vertex colors if none exist
            if mesh.visual.vertex_colors is None:
                vertex_colors = torch.ones_like(verts)[None]  # white
            else:
                vertex_colors = torch.tensor(
                    mesh.visual.vertex_colors[:, :3].astype(float) / 255.0,
                    device=device,
                    dtype=torch.float32
                )[None]

            # Create PyTorch3D mesh
            textures = TexturesVertex(verts_features=vertex_colors)
            pytorch3d_mesh = Meshes(
                verts=[verts],
                faces=[faces],
                textures=textures
            )

            # Render
            renderer = get_pytorch3d_renderer(image_size, dist, elev, azim, device)
            images = renderer(pytorch3d_mesh)
            image = images[0, ..., :3].cpu().numpy()

        case _:
            raise ValueError(f"Unsupported mesh type: {type(mesh)}. Must be either PyTorch3D Meshes or Trimesh object.")

    # Convert to uint8 for display/saving
    image = (image * 255).astype(np.uint8)

    if save_path:
        import imageio
        imageio.imsave(save_path, image)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    return image


def compute_face_normals(mesh):
    """Compute face normals."""
    verts = mesh.verts_packed()  # (V, 3)
    faces = mesh.faces_packed()  # (F, 3)

    # Compute edge vectors
    v0 = verts[faces[:, 0]]  # (F, 3)
    v1 = verts[faces[:, 1]]  # (F, 3)
    v2 = verts[faces[:, 2]]  # (F, 3)

    face_normals = torch.cross(v1 - v0, v2 - v0)
    face_normals = torch.nn.functional.normalize(face_normals, p=2, dim=1)

    return face_normals