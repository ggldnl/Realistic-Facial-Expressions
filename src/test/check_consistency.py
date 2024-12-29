from pathlib import Path
import argparse

from src.utils.mesh_utils import read_mesh


def process_dir(source):
    """
    Recursively check for consistency the .obj files in the source directory
    """

    for file_path in source.rglob("*.obj"):
        process_file(file_path)

def process_file(file_path):
    """
    Check consistency of the input file
    """

    print(f"Processing {file_path}")
    mesh = read_mesh(file_path)

    num_faces = len(mesh.faces)
    num_vertices = len(mesh.vertices)

    num_edges_per_face = [len(face) for face in mesh.faces]
    min_edges = min(num_edges_per_face)
    max_edges = max(num_edges_per_face)
    mean_edges = sum(num_edges_per_face) / len(num_edges_per_face)

    vertex_neighbors = mesh.vertex_neighbors
    num_neighbors_per_vertex = [len(neighbors) for neighbors in vertex_neighbors]
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

    # Resolve source and destination paths
    source = Path(args.source)
    if source.is_dir():
        process_dir(source)
    else:
        process_file(source)
