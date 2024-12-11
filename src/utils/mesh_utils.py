try:

    import torch

    if torch.cuda.is_available():
        import kaolin as kal
        import kaolin.ops.mesh
    else:
        print("No NVIDIA GPU detected. Using a stub for kaolin.")
        from src.utils.kaolin_stub import KaolinStub
        kal = KaolinStub()

except ImportError as e:
    print(f"ImportError: {e}. Using a stub for kaolin.")
    from src.utils.kaolin_stub import KaolinStub
    kal = KaolinStub()

from pathlib import Path
import pyvista as pv
import numpy as np
import trimesh


# TODO check H and W as they should be parametric
def get_texture_map_from_color(mesh, color, H=224, W=224):
    num_faces = mesh.faces.shape[0]
    texture_map = torch.zeros(1, H, W, 3)
    texture_map[:, :, :] = color
    return texture_map.permute(0, 3, 1, 2)

def get_face_attributes_from_color(mesh, color):
    num_faces = mesh.faces.shape[0]
    face_attributes = torch.zeros(1, num_faces, 3, 3)
    face_attributes[:, :, :] = color
    return face_attributes

def read_mesh(obj_path):
    """
    Reads a mesh file and returns its processed data.
    """

    if isinstance(obj_path, str):
        obj_path = Path(obj_path)

    # Load the mesh based on file extension
    loader = 'kaolin'
    if obj_path.suffix == ".obj":
        mesh = kal.io.obj.import_mesh(obj_path, with_normals=True)
    elif obj_path.suffix == ".off":
        mesh = kal.io.off.import_mesh(obj_path)
    elif obj_path.suffix == ".ply":
        print("Kaolin does not support PLY. Falling back to trimesh.")
        mesh = trimesh.load_mesh(obj_path, process=True)
        loader = 'trimesh'
    else:
        raise ValueError(f"Unsupported file extension for {obj_path}.")

    if loader == 'trimesh':

        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.long)

        # Generate normals if available
        vertex_normals = (torch.tensor(mesh.vertex_normals, dtype=torch.float32)
                            if mesh.vertex_normals is not None else None)
        face_normals = None  # Trimesh doesn't directly provide face normals

    else:

        # Process vertices and faces
        vertices = mesh.vertices
        faces = mesh.faces

        # Initialize normals and UV mappings
        vertex_normals = (torch.nn.functional.normalize(mesh.vertex_normals.float())
                        if mesh.vertex_normals is not None else None)
        face_normals = (torch.nn.functional.normalize(mesh.face_normals.float())
                        if mesh.face_normals is not None else None)

    # Return the mesh data as a dictionary
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


# Test the visualization
if __name__ == "__main__":

    import src.config.config as config

    mesh_path = Path(config.DATA_DIR, '100/1_neutral.ply')

    color = torch.tensor([0.0, 0.0, 1.0])
    mesh_data = read_mesh(mesh_path)
    visualize_mesh(mesh_data, color=color)
