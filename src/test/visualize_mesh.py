import argparse
from pathlib import Path
import torch

from src.utils.mesh_utils import visualize_mesh
from src.utils.mesh_utils import batch_meshes


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
    mesh = batch_meshes([mesh_path])
    print(f"Original mesh:")
    print(f"Vertices: {mesh.verts_padded().shape}")
    print(f"Faces: {mesh.faces_padded().shape}")

    # Visualize the mesh
    visualize_mesh(mesh, device='cpu')

    """
    # Read and display the original mesh
    mesh_data = read_dict(mesh_path)
    print(f"Original mesh:")
    print(f"Vertices: {len(mesh_data['vertices'])}")
    print(f"Faces: {len(mesh_data['faces'])}")

    # Simplify and display the mesh
    mesh_data = read_dict(
        mesh_path,
        mesh_drop_percent=args.percent,
        mesh_face_count=args.face_count,
        aggression=args.aggression,
        normalize=args.normalize
    )
    print(f"\nSimplified mesh:")
    print(f"Vertices: {len(mesh_data['vertices'])}")
    print(f"Faces: {len(mesh_data['faces'])}")

    # Visualize the mesh
    visualize_mesh(mesh_data, color=color)
    """
