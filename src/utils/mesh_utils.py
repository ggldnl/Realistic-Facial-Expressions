from torch_geometric.data import Data
from pathlib import Path
import pyvista as pv
import numpy as np
import trimesh
import torch


def read_mesh(
        obj_path,
        loader='trimesh',
        mesh_drop_percent=0.5,
        mesh_face_count=None,
        aggression=0
):
    """
    Reads a mesh file and returns its processed data.

    Args:
        obj_path (str or Path): Path to the mesh file
        loader (str): Mesh loader to use (for now, only 'trimesh' is supported)
        mesh_drop_percent (float): Percentage of faces to drop (between 0.0 and 1.0)
        mesh_face_count (int, optional): Target number of faces in simplified mesh, overrides mesh_drop_percent if provided
        aggression (int): Simplification aggressiveness, 0 (slow/quality) to 10 (fast/rough)

    Returns:
        dict: Dictionary containing mesh data (vertices, faces, normals)

    Raises:
        NotImplementedError: If loader not present
        ValueError: If file extension is not supported or invalid parameters
    """
    if isinstance(obj_path, str):
        obj_path = Path(obj_path)

    # Validate file extension
    if obj_path.suffix not in ['.obj', '.off', '.ply']:
        raise ValueError(f"Unsupported file extension for {obj_path}.")

    match loader:
        case 'trimesh':
            # Load the mesh using trimesh
            mesh = trimesh.load_mesh(obj_path, process=True)

            # Simplify the mesh if mesh_drop_percent or mesh_face_count are provided
            if mesh_drop_percent or mesh_face_count:

                # Validate mesh_face_count
                if mesh_face_count is not None:
                    if mesh_face_count < 4:
                        raise ValueError(f"Target face count ({mesh_face_count}) too low. We can have a minimum of 4 faces for the mesh to be watertight.")
                    if mesh_face_count > len(mesh.faces):
                        raise ValueError(f"Target face count ({mesh_face_count}) exceeds original mesh face count ({len(mesh.faces)}).")

                # Validate percentage if mesh_face_count is not provided
                else:
                    if mesh_drop_percent and not 0 < mesh_drop_percent <= 1:
                        raise ValueError("percent must be between 0.0 and 1.0")

                if aggression and not 0 <= aggression <= 10:
                    raise ValueError("The aggression parameter must be an integer in range [0, 10]")

                mesh = mesh.simplify_quadric_decimation(
                    percent=mesh_drop_percent,
                    face_count=mesh_face_count,
                    aggression=aggression
                )


        case _:
            raise NotImplementedError(f"Loader '{loader}' is not implemented. Only 'trimesh' is currently supported.")

    return mesh


def read_graph(obj_path, **kwargs):
    """
    Reads the mesh file and returns a torch_geometric.data.Data graph.

    Args:
        obj_path (str or pathlib.Path): Path to the mesh.

    Returns:
        torch_geometric.data.Data: Graph generated from the mesh at obj_path.
    """

    mesh = read_mesh(obj_path, **kwargs)

    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)

    # Convert faces to edges
    edge_list = faces_to_edges(faces.numpy(force=True))
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Create graph object
    graph = Data(
        x=vertices,  # Nodes (vertex features)
        edge_index=edge_index  # Edges
    )
    return graph


def read_dict(obj_path, **kwargs):
    """
    Reads the mesh file and returns a dictionary containing its data.

    Args:
        obj_path (str or pathlib.Path): Path to the mesh.

    Returns:
        dict: Dictionary generated from the mesh at obj_path.
    """

    mesh = read_mesh(obj_path, **kwargs)

    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)

    # Generate normals if available
    vertex_normals = (torch.tensor(mesh.vertex_normals, dtype=torch.float32)
                      if mesh.vertex_normals is not None else None)
    face_normals = None

    return {
        "vertices": vertices,
        "faces": faces,
        "vertex_normals": vertex_normals,
        "face_normals": face_normals
    }


def visualize_mesh(mesh, color=(0.0, 0.0, 1.0), show_normals=False):
    """
    Visualizes the mesh and its normals using common data from read_mesh.
    """

    # Extract vertices and faces
    if isinstance(mesh, dict):
        vertices = mesh["vertices"].numpy(force=True)
        faces = mesh["faces"].numpy(force=True).flatten()

        # Check if vertex normals are available
        vertex_normals = (mesh["vertex_normals"].numpy(force=True)
                          if mesh["vertex_normals"] is not None else None)

    else:
        vertices = mesh.vertices
        faces = mesh.faces

        # Generate normals if available
        vertex_normals = mesh.vertex_normals if mesh.vertex_normals is not None else None

    # Convert faces to PyVista format (prefix each face with the number of vertices, 3 for triangles)
    pv_faces = np.insert(faces.reshape(-1, 3), 0, 3, axis=1)  # PyVista expects this format
    mesh = pv.PolyData(vertices, pv_faces)

    # Add normals if available
    if show_normals and vertex_normals is not None:
        mesh["Normals"] = vertex_normals

    # Add uniform color to all vertices
    uniform_color = np.tile(color, (vertices.shape[0], 1))
    mesh.point_data["Color"] = uniform_color

    # Plot the mesh
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars="Color", rgb=True, show_edges=False)

    # Optionally add normals
    if show_normals and vertex_normals is not None:
        plotter.add_arrows(vertices, vertex_normals, mag=0.1, color="white", label="Normals")

    # Add axes for reference
    plotter.add_axes()
    plotter.show()


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


def faces_to_edges(faces):
    """
    Given the faces of a mesh, create the unique edges.

    Args:
        faces (list of list or tuple, tensor): A list of faces, where each face is
            represented as a list or tuple of vertex indices (e.g., [v1, v2, v3]).

    Returns:
        list of tuple: A list of unique edges, where each edge is a tuple of two
            vertex indices (e.g., (v1, v2)). The vertex indices in each edge are
            sorted in ascending order.
    """
    edges = set()
    for face in faces:
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))
    return list(edges)


# Test the visualization
if __name__ == "__main__":

    root_dir = Path(__file__).parent.parent.parent
    mesh_path = Path(root_dir, 'datasets/facescape/100/models_reg/1_neutral.obj')

    color = torch.tensor([0.0, 0.0, 1.0])
    mesh_data = read_dict(mesh_path)
    visualize_mesh(mesh_data, color=color)
