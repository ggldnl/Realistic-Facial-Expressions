import argparse
from pathlib import Path
import torch

from src.utils.mesh_utils import read_dict, visualize_mesh


if __name__ == "__main__":

    """
    Visualizes the mesh before and after the simplification.
    """

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Visualize mesh before and after simplification.")
    parser.add_argument('-m', '--mesh', type=str, required=True, help="Path to the mesh file (.obj).")
    parser.add_argument('-p', '--percent', type=float, default=0.8, help="Percentage to simplify the mesh (default: 0.8).")

    args = parser.parse_args()

    # Default color for the mesh (blue)
    color = torch.tensor([0.0, 0.0, 1.0])

    # Resolve mesh path
    mesh_path = Path(args.mesh)

    # Read and display the original mesh
    mesh_data = read_dict(mesh_path)
    print(f"Original mesh:")
    print(f"Vertices: {len(mesh_data['vertices'])}")
    print(f"Faces: {len(mesh_data['faces'])}")

    # Simplify and display the mesh
    mesh_data = read_dict(mesh_path, mesh_drop_percent=args.percent, mesh_face_count=None)
    print(f"\nSimplified mesh:")
    print(f"Vertices: {len(mesh_data['vertices'])}")
    print(f"Faces: {len(mesh_data['faces'])}")

    # Visualize the mesh
    visualize_mesh(mesh_data, color=color)
