from pathlib import Path
import argparse
import torch
from pytorch3d.io import load_objs_as_meshes


def process_dir(source):
    """
    Recursively check for consistency of .obj files in the source directory
    """
    for file_path in source.rglob("*.obj"):
        process_file(file_path)

def process_file(file_path):
    """
    Check consistency of the input file
    """

    print(f"Processing {file_path}")
    mesh = load_objs_as_meshes([str(file_path)])

    # Get the number of faces and vertices
    num_faces = mesh.num_faces_per_mesh().item()
    num_vertices = mesh.num_verts_per_mesh().item()

    # Get the faces and vertices
    faces = mesh.faces_packed()  # Faces are represented as indices of vertices

    # Compute the number of edges per face
    num_edges_per_face = torch.tensor([3] * num_faces)  # Assumes all faces are triangles

    min_edges = num_edges_per_face.min().item()
    max_edges = num_edges_per_face.max().item()
    mean_edges = num_edges_per_face.float().mean().item()

    # Compute vertex neighbors
    vertex_neighbors = [[] for _ in range(num_vertices)]
    for face in faces:
        for i in range(3):
            v1, v2 = face[i].item(), face[(i + 1) % 3].item()
            vertex_neighbors[v1].append(v2)
            vertex_neighbors[v2].append(v1)

    num_neighbors_per_vertex = [len(set(neighbors)) for neighbors in vertex_neighbors]
    min_neighbors = min(num_neighbors_per_vertex)
    max_neighbors = max(num_neighbors_per_vertex)
    mean_neighbors = sum(num_neighbors_per_vertex) / len(num_neighbors_per_vertex)

    # Output the results
    print('-' * 50)
    print(f"Number of faces: {num_faces}")
    print(f"Number of vertices: {num_vertices}")
    print(f"Min number of edges per face: {min_edges}")
    print(f"Mean number of edges per face: {mean_edges}")
    print(f"Max number of edges per face: {max_edges}")
    print(f"Min number of neighbors per vertex: {min_neighbors}")
    print(f"Mean number of neighbors per vertex: {mean_neighbors}")
    print(f"Max number of neighbors per vertex: {max_neighbors}")
    print()

if __name__ == '__main__':

    """
    Check the provided mesh(s) and tell the min/mid/max number of faces and number of neighbors.
    """

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Check the input mesh for consistency.")
    parser.add_argument('-s', '--source', type=str, required=True, help="Path to the source file/directory.")

    args = parser.parse_args()

    # Resolve source path and set device
    source = Path(args.source)

    if source.is_dir():
        process_dir(source)
    else:
        process_file(source)
