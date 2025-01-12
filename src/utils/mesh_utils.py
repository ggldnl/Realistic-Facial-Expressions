from pathlib import Path
from pytorch3d.io import load_obj
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
import torch.nn.functional as F
import pyvista as pv
import numpy as np
import trimesh
import torch

from meshes import WeightedMeshes


def read_mesh(
        obj_path,
        normalize=False,
):
    """
    Reads a mesh file using either PyTorch3D or trimesh loaders.

    Parameters:
        obj_path (str or Path): Path to the mesh file.
        normalize (bool): Whether to normalize vertex coordinates.

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

    verts_data = []
    faces_data = []
    weights_data = []

    with open(obj_path, 'r') as obj_file:
        for line in obj_file:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v':  # Vertex line
                verts_data.append([float(coord) for coord in parts[1:4]])
            elif parts[0] == 'f':  # Face line
                # Split face into indices, considering only the first index if multiple formats are used (e.g., v/vt/vn).
                face = [int(part.split('/')[0]) - 1 for part in parts[1:4]]  # Convert to 0-based index.
                faces_data.append(face)
            elif parts[0] == 'w':  # Weights
                weights_data.append(float(parts[1]))

    verts = torch.tensor(verts_data, dtype=torch.float32)
    faces = torch.tensor(faces_data, dtype=torch.long)

    if len(weights_data) == 0:  # No weights in obj file
        weights_data = [1] * len(verts)
    weights = torch.tensor(weights_data, dtype=torch.float32)

    # Normalize if requested
    if normalize:
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        verts = (verts - center) / scale

    # Create Meshes object
    mesh = WeightedMeshes(
        verts=[verts],
        faces=[faces],
        weights=[weights]
    )

    return mesh

def read_meshes(meshes, normalize=False):
    """
    Batch a list of mesh node features into a single pytorch3d.Meshes object.

    Parameters:
        meshes (list[str | Path]): List of paths to meshes.
        normalize (bool): True to normalize the meshes, False otherwise.

    Returns:
        pytorch3d.Meshes: pytorch3d's Meshes object containing all the meshes (eventually normalized).
    """

    all_verts = []
    all_faces = []
    all_weights = []

    for obj_path in meshes:

        verts_data = []
        faces_data = []
        weights_data = []

        with open(obj_path, 'r') as obj_file:
            for line in obj_file:
                parts = line.strip().split()
                if not parts:
                    continue

                if parts[0] == 'v':  # Vertex line
                    verts_data.append([float(coord) for coord in parts[1:4]])
                elif parts[0] == 'f':  # Face line
                    # Split face into indices, considering only the first index if multiple formats are used (e.g., v/vt/vn).
                    face = [int(part.split('/')[0]) - 1 for part in parts[1:4]]  # Convert to 0-based index.
                    faces_data.append(face)
                elif parts[0] == 'w':  # Weights
                    weights_data.append(float(parts[1]))

        verts = torch.tensor(verts_data, dtype=torch.float32)
        faces = torch.tensor(faces_data, dtype=torch.long)

        if len(weights_data) == 0:  # No weights in obj file
            weights_data = [1] * len(verts)
        weights = torch.tensor(weights_data, dtype=torch.float32)

        # Normalize if requested
        if normalize:
            center = verts.mean(0)
            scale = max((verts - center).abs().max(0)[0])
            verts = (verts - center) / scale

        all_verts.append(verts)
        all_faces.append(faces)
        all_weights.append(weights)

    # Create a Meshes object
    meshes = WeightedMeshes(verts=all_verts, faces=all_faces, weights=all_weights)

    return meshes


def tensor_to_mesh(vertices, faces, vertex_normals=None):
    """
    Creates a trimesh object from tensor data.

    Parameters:
        vertices (torch.Tensor): Vertex positions tensor of shape (N, 3).
        faces (torch.Tensor): Face indices tensor of shape (M, 3).
        vertex_normals (torch.Tensor, optional): Vertex normals tensor of shape (N, 3).

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


def pad_mesh(tensor, target_size):
    """
    Pad a tensor to the desired size.
    """
    pad_size = target_size - tensor.size(0)
    return F.pad(tensor, (0, 0, 0, pad_size))


def unpack_meshes(pack, batch_indices):
    unique_indices = batch_indices.unique()
    return torch.stack([pack[batch_indices == idx] for idx in unique_indices])


def pack_meshes(batch):
    """
    Convert (num_meshes, max_num_nodes, num_features) a batch to a pack (num_meshes*max_num_nodes, num_features).
    It returns both the pack and the batch indices. The input meshes in the batch are supposed to be padded,
    otherwise the conversion will fail.
    """

    # Create an indexing tensor
    batch_indices = torch.cat([torch.full((batch[i].size(0),), i, dtype=torch.long) for i in range(len(batch))])

    # Concatenate the meshes into a single tensor
    pack = torch.cat(batch, dim=0)

    return pack, batch_indices


def faces_to_edges(faces):
    """
    Converts a tensor containing face indices to a tensor containing edge indices.

    Parameters:
        faces (torch.Tensor): Tensor containing face indices.

    Returns:
        torch.Tensor: Tensor containing edges indices.
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

def visualize_mesh(mesh, color=(0.0, 0.0, 1.0), show_edges=False):
    """
    Visualizes a 3D mesh using PyVista.

    Parameters:
        mesh (dict, tuple(torch.Tensor, torch.Tensor), pytorch3d.Meshes, trimesh.Trimesh):
            The input mesh to visualize. Can be one of the following:
            - A dictionary with keys "vertices" (Nx3 tensor/array) and "faces" (Fx3 tensor/array).
            - Two tensors containing vertex positions (Nx3) and a corresponding faces tensor (Fx3).
            - A PyTorch3D Meshes object (must contain exactly one mesh).
            - A Trimesh object.
        color (tuple(floats), torch.Tensor, np.nd.array): RGB color to apply to the mesh.
            If a tuple is provided, the same color will be applied to all the vertices.
            If a tensor or an array are provided, the number of colors must match the number of
            vertices and each color will be applied to the respective vertex.
        show_edges (bool): Whether to show the edges of the mesh. Default is False.
    """

    # Handle different mesh input types
    if isinstance(mesh, dict):
        vertices = mesh["vertices"]
        faces = mesh["faces"]

        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()

    elif isinstance(mesh, WeightedMeshes):
        if mesh.verts_padded().shape[0] != 1:
            raise ValueError("PyTorch3D Meshes object must contain exactly one mesh.")

        vertices = mesh.verts_packed().cpu().numpy()
        faces = mesh.faces_packed().cpu().numpy()
        weights = mesh.weights_packed().cpu().numpy()

        # Create the color by repeating the weights (bw).
        # This overrides the color provided from outside.
        color = np.tile(weights[:, np.newaxis], (1, 3))

    elif isinstance(mesh, Meshes):
        if mesh.verts_padded().shape[0] != 1:
            raise ValueError("PyTorch3D Meshes object must contain exactly one mesh.")

        vertices = mesh.verts_packed().cpu().numpy()
        faces = mesh.faces_packed().cpu().numpy()

    elif isinstance(mesh, trimesh.Trimesh):
        vertices = mesh.vertices
        faces = mesh.faces

    elif isinstance(mesh, tuple) and len(mesh) == 2:
        vertices, faces = mesh
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()

    else:
        raise TypeError("Unsupported mesh type. Must be dict, tuple, PyTorch3D Meshes, or trimesh.Trimesh.")

    # Convert faces to PyVista format
    pv_faces = np.insert(faces, 0, 3, axis=1).flatten()
    pv_mesh = pv.PolyData(vertices, pv_faces)
    num_vertices = vertices.shape[0]

    # Handle vertex colors: use provided colors or fallback to default
    if isinstance(color, tuple):
        if len(color) != 3:
            raise ValueError("Color tuple must contain 3 values (RGB).")
        vertex_colors = np.tile(color, (vertices.shape[0], 1))

    elif isinstance(color, np.ndarray):
        if color.shape != (num_vertices, 3):
            raise ValueError(f"Color array must have shape ({num_vertices}, 3), has shape {color.shape} instead.")
        vertex_colors = color

    elif isinstance(color, torch.Tensor):
        if (color.shape[0], color.shape[1]) != (num_vertices, 3):
            raise ValueError(f"Color tensor must have shape ({num_vertices}, 3), has shape {color.shape} instead.")
        vertex_colors = color.cpu().numpy()

    else:
        raise ValueError("Unsupported color type. Must be tuple, np.ndarray, torch.Tensor.")

    pv_mesh.point_data["Color"] = vertex_colors

    # Plot the mesh
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="Color", rgb=True, show_edges=show_edges)

    # Add axes for reference
    plotter.add_axes()
    plotter.show()


# Test the visualization
if __name__ == "__main__":

    root_dir = Path(__file__).parent.parent.parent
    mesh_path = Path(root_dir, 'datasets/facescape_highlighted/100/models_reg/1_neutral.obj')

    # Simple visualization with a Meshes object
    """
    color = torch.tensor([0.0, 0.0, 1.0])
    mesh = load_objs_as_meshes([mesh_path])
    visualize_mesh(mesh)
    """

    # Load a mesh with vertex colors
    vertices, faces, aux = load_obj(mesh_path)

    # Try to use the normals for the color
    if aux.normals is not None:
        color = aux.normals
    else:
        color = (0.0, 0.0, 0.1)  # Fallback to blue

    # Visualize the mesh
    visualize_mesh((vertices, faces.verts_idx), color=color)
