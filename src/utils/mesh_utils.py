from pathlib import Path
import pyvista as pv
import numpy as np
import trimesh
import torch


def read_mesh(obj_path, loader='trimesh', simplify=False, percent=0.5, face_count=None, aggression=None):
    """
    Reads a mesh file and returns its processed data.

    Args:
        obj_path (str or Path): Path to the mesh file
        loader (str): Mesh loader to use ('trimesh' or 'kaolin')
        simplify (bool): Whether to simplify the mesh (only works with trimesh loader)
        percent (float): Percentage of faces to drop (between 0.0 and 1.0), used if face_count is None
        face_count (int, optional): Target number of faces in simplified mesh, overrides percent if provided
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

            # Simplify mesh if requested
            if simplify:
                # Validate aggression parameter
                if aggression and not 0 <= aggression <= 10:
                    raise ValueError("aggression must be between 0 and 10")

                # If face_count is provided, validate it
                if face_count is not None:
                    if face_count < 4:
                        raise ValueError(f"Target face count ({face_count}) too low. Minimum is 4 faces.")
                    if face_count > len(mesh.faces):
                        raise ValueError(f"Target face count ({face_count}) exceeds original mesh face count ({len(mesh.faces)})")
                else:
                    # Validate percentage
                    if percent and not 0 < percent <= 1:
                        raise ValueError("percent must be between 0.0 and 1.0")

                mesh = mesh.simplify_quadric_decimation(
                    percent=percent,
                    face_count=face_count,
                    aggression=aggression
                )

            vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
            faces = torch.tensor(mesh.faces, dtype=torch.long)

            # Generate normals if available
            vertex_normals = (torch.tensor(mesh.vertex_normals, dtype=torch.float32)
                              if mesh.vertex_normals is not None else None)
            face_normals = None

        case _:
            raise NotImplementedError(f"Loader '{loader}' is not implemented. Only 'trimesh' is currently supported.")

    return {
        "vertices": vertices,
        "faces": faces,
        "vertex_normals": vertex_normals,
        "face_normals": face_normals
    }


def visualize_mesh(mesh_data, color=(0.0, 0.0, 1.0), show_normals=False):
    """
    Visualizes the mesh and its normals using common data from read_mesh.
    """
    # Extract vertices and faces
    vertices = mesh_data["vertices"].cpu().numpy()
    faces = mesh_data["faces"].cpu().numpy().flatten()

    # Check if vertex normals are available
    vertex_normals = (mesh_data["vertex_normals"].cpu().numpy()
                      if mesh_data["vertex_normals"] is not None else None)

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


def create_trimesh_from_tensors(vertices, faces, vertex_normals=None):
    """
    Creates a trimesh object from tensor data.

    Args:
        vertices (torch.Tensor): Vertex positions tensor of shape (N, 3)
        faces (torch.Tensor): Face indices tensor of shape (M, 3)
        vertex_normals (torch.Tensor, optional): Vertex normals tensor of shape (N, 3)

    Returns:
        trimesh.Trimesh: The created mesh object
    """
    # Convert tensors to numpy arrays
    vertices_np = vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    # Create the mesh
    mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        process=False
    )

    # If vertex normals are provided, set them
    if vertex_normals is not None:
        vertex_normals_np = vertex_normals.detach().cpu().numpy()
        mesh.vertex_normals = vertex_normals_np

    return mesh
