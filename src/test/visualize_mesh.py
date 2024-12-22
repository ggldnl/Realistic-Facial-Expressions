from pathlib import Path

import torch

from src.utils.mesh_utils import read_mesh, visualize_mesh

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / "datasets" / "facescape"
    mesh_path = Path(data_dir, '100/models_reg/1_neutral.obj')

    color = torch.tensor([0.0, 0.0, 1.0])
    mesh_data = read_mesh(mesh_path, simplify=False)
    print(f"Original mesh:")
    print(f"Vertices: {len(mesh_data['vertices'])}")
    print(f"Faces: {len(mesh_data['faces'])}")

    mesh_data = read_mesh(mesh_path, simplify=True, percent=0.8, face_count=None, aggression=None)
    print(f"\nSimplified mesh:")
    print(f"Vertices: {len(mesh_data['vertices'])}")
    print(f"Faces: {len(mesh_data['faces'])}")

    visualize_mesh(mesh_data, color=color)