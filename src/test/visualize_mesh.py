import argparse
from pathlib import Path

from src.utils.mesh_utils import visualize_mesh
from src.utils.mesh_utils import read_meshes


if __name__ == "__main__":

    """
    Visualizes the mesh before and after the simplification.
    """


    # Argument parser setup
    parser = argparse.ArgumentParser(description="Visualize mesh before and after simplification.")
    parser.add_argument('-m', '--mesh', type=str, required=True, help="Path to the mesh file (.obj).")
    parser.add_argument('-n', '--normalize', type=bool, default=True, help="Normalize mesh between -1 and 1.")

    args = parser.parse_args()

    # Resolve mesh path
    mesh_path = Path(args.mesh)

    # Read the mesh
    mesh = read_meshes([mesh_path])
    print(f"Original mesh:")
    print(f"Vertices: {mesh.verts_padded().shape}")
    print(f"Faces: {mesh.faces_padded().shape}")

    # Visualize the mesh
    visualize_mesh(mesh)
