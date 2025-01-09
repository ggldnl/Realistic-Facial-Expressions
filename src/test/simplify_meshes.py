from pathlib import Path
from tqdm import tqdm
import trimesh
import argparse


def read_and_simplify(
        path,
        mesh_drop_percent=None,
        mesh_face_count=None,
        aggression=None,
        normalize=False
):
    """
    Read a mesh as a Trimesh object and, optionally, simplify it using the provided parameters.
    """

    if isinstance(path, str):
        path = Path(path)

        # Validate file extension
    if path.suffix not in ['.obj', '.off', '.ply']:
        raise ValueError(f"Unsupported file extension for {path}.")

    # Load the mesh using trimesh
    mesh = trimesh.load_mesh(path, process=True)

    # Simplify the mesh if mesh_drop_percent or mesh_face_count are provided
    if mesh_drop_percent or mesh_face_count:

        # Validate mesh_face_count
        if mesh_face_count is not None:
            if mesh_face_count < 4:
                raise ValueError(
                    f"Target face count ({mesh_face_count}) too low. We can have a minimum of 4 faces for the mesh to be watertight.")
            if mesh_face_count > len(mesh.faces):
                raise ValueError(
                    f"Target face count ({mesh_face_count}) exceeds original mesh face count ({len(mesh.faces)}).")

        # Validate percentage if mesh_face_count is not provided
        else:
            if mesh_drop_percent and not 0 < mesh_drop_percent <= 1:
                raise ValueError("percent must be between 0.0 and 1.0")

        if aggression and not 0 <= aggression <= 10:
            raise ValueError("The aggression parameter must be an integer in range [0, 10]")

        print(f'Number of faces before simplification: {len(mesh.faces)}')

        mesh = mesh.simplify_quadric_decimation(
            percent=mesh_drop_percent,
            face_count=mesh_face_count,
            aggression=aggression
        )

        print(f'Number of faces after simplification: {len(mesh.faces)}')

        if normalize:
            mesh.vertices = mesh.vertices / mesh.vertices.max()

    return mesh


def process_dir(
        source,
        destination,
        mesh_drop_percent=None,
        mesh_face_count=None,
        aggression=None,
        normalize=False
):
    """
    Recursively preprocess .obj files in the source directory, simplifying it
    and saving the results in the destination directory.

    Parameters:
        source (str or Path): Path to the mesh file.
        destination (str or Path): Path where to save the processed file.
        mesh_drop_percent (float): Percentage of faces to drop (between 0.0 and 1.0).
        mesh_face_count (int, optional): Target number of faces in simplified mesh, overrides mesh_drop_percent if provided.
        aggression (int): Simplification aggressiveness, 0 (slow/quality) to 10 (fast/rough)
    """

    for file_path in tqdm(source.rglob("*.obj")):

        # Compute the relative path to maintain directory structure
        relative_path = file_path.relative_to(source)
        destination_file = destination / relative_path

        # Ensure the destination directory exists
        destination_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Processing {file_path} -> {destination_file}")
        input_mesh = read_and_simplify(file_path,
                               mesh_drop_percent=mesh_drop_percent,
                               mesh_face_count=mesh_face_count,
                               aggression=aggression,
                               normalize=normalize
        )

        input_mesh.export(destination_file)


if __name__ == '__main__':

    """
    Preprocess the dataset by simplifying all the meshes we encounter (.obj)
    in the given folder by a certain amount (default 80%).
    We store the result in another directory, keeping the directory tree unchanged.
    """

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Preprocess .obj meshes by simplifying them.")
    parser.add_argument('-s', '--source', type=str, required=True, help="Path to the source directory containing .obj files.")
    parser.add_argument('-d', '--destination', type=str, required=True, help="Path to the destination directory to save simplified meshes.")
    parser.add_argument('-p', '--percent', type=float, default=0.8, help="Percentage to simplify the mesh (default: 0.8).")
    parser.add_argument('-f', '--face-count', type=int, default=None, help="Percentage to simplify the mesh (default: 0.8).")
    parser.add_argument('-a', '--aggression', type=int, default=5, help="Percentage to simplify the mesh (default: 0.8).")
    parser.add_argument('-n', '--normalize', type=bool, default=True, help="Normalize mesh between -1 and 1.")

    args = parser.parse_args()

    # Resolve source and destination paths
    source_dir = Path(args.source)
    destination_dir = Path(args.destination)

    # Run the processing
    process_dir(
        source_dir,
        destination_dir,
        mesh_drop_percent=args.percent,
        mesh_face_count=args.face_count,
        aggression=args.aggression,
        normalize=args.normalize
    )
