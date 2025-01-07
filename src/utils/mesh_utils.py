from pathlib import Path
import torch
import numpy as np
import trimesh
from pytorch3d.io import load_obj
from pytorch3d.io import load_ply
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PointLights
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer import RasterizationSettings
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer import MeshRasterizer
from pytorch3d.renderer import SoftPhongShader
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer import FoVPerspectiveCameras


def read_mesh(
        obj_path,
        normalize=False,
):
    """
    Reads a mesh file using either PyTorch3D or trimesh loaders.

    Args:
        obj_path (str or Path): Path to the mesh file
        normalize (bool): Whether to normalize vertex coordinates

    Returns:
        PyTorch3D Meshes object

    Raises:
        ValueError: If file extension not supported or invalid parameters
    """

    if isinstance(obj_path, str):
        obj_path = Path(obj_path)

    # Validate file extension
    if obj_path.suffix not in ['.obj', '.off', '.ply']:
        raise ValueError(f"Unsupported file extension for {obj_path}.")

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

def batch_meshes(meshes, normalize=False):
    """
    Batch a list of mesh node features into a single tensor.

    Args:
        meshes (list of torch.Tensor | list of ): Each tensor has shape (num_nodes, num_features) representing the nodes of a mesh.
        normalize (bool): True to normalize the meshes, False otherwise.

    Returns:
        torch.Tensor: A tensor of shape (num_graphs, max_num_nodes, num_features), where `num_graphs` is the number of meshes,
                      `max_num_nodes` is the maximum number of nodes in any mesh, and `num_features` is the number of features per node.
    """

    meshes_obj = load_objs_as_meshes(meshes)

    if normalize:

        # Get vertices of the loaded meshes
        verts = meshes_obj.verts_padded()
        faces = meshes_obj.faces_padded()

        # We scale normalize and center the target meshes to fit in a sphere of radius 1 centered at (0,0,0).
        # (scale, center) will be used to bring the predicted mesh to its original center and scale
        center = verts.mean(dim=-2, keepdim=True)  # Compute mean along the num_nodes dimension
        verts_normalized = verts - center
        norms = torch.norm(verts_normalized, dim=-1, keepdim=True)  # Compute norms for each node
        max_norm = norms.max(dim=-2, keepdim=True).values
        verts_normalized /= max_norm

        meshes_obj = Meshes(verts=verts_normalized, faces=faces)

    return meshes_obj

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

def faces_to_edges(faces):
    """
    Converts a tensor containing face indices to a tensor containing edge indices.
    """

    # Convert faces to edges (undirected edges)
    edges = torch.cat([
        faces[:, [0, 1]],  # Edge between vertex 0 and 1
        faces[:, [1, 2]],  # Edge between vertex 1 and 2
        faces[:, [2, 0]]  # Edge between vertex 2 and 0
    ], dim=0)

    # Remove duplicate edges (if graph is undirected)
    edges = torch.cat([edges, edges[:, [1, 0]]], dim=0)  # Add reverse edges for undirected graph
    edges = torch.unique(edges, dim=0)  # Remove duplicates

    # Transpose edges to match PyTorch Geometric format
    edge_index = edges.t()  # Shape: [2, num_edges]

    return edge_index

def visualize_mesh(
        mesh,
        image_size: int = 512,
        dist: float = 1.5,
        elev: float = 0.0,
        azim: float = 0.0,
        save_path: str = None,
        show: bool = True,
        device='cuda'
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

    match mesh:
        case Meshes():

            # Create renderer and render
            renderer = get_pytorch3d_renderer(image_size, dist, elev, azim, device=device)
            images = renderer(mesh)
            image = images[0, ..., :3].cpu().numpy()

        case trimesh.Trimesh():

            # Convert to pytorch3d format for rendering
            verts = torch.tensor(mesh.vertices, dtype=torch.float32)
            faces = torch.tensor(mesh.faces, dtype=torch.int64)

            # Create white vertex colors if none exist
            if mesh.visual.vertex_colors is None:
                vertex_colors = torch.ones_like(verts)[None]  # white
            else:
                vertex_colors = torch.tensor(
                    mesh.visual.vertex_colors[:, :3].astype(float) / 255.0,
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
            renderer = get_pytorch3d_renderer(image_size, dist, elev, azim)
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
