import argparse
from pathlib import Path

from src.utils.mesh_utils import read_mesh


def process_dir(source, destination, percent=None):
    """
    Recursively preprocess .obj files in the source directory, simplifying it
    and saving the results in the destination directory.
    """

    for file_path in source.rglob("*.obj"):

        # Compute the relative path to maintain directory structure
        relative_path = file_path.relative_to(source)
        destination_file = destination / relative_path

        # Ensure the destination directory exists
        destination_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Processing {file_path} -> {destination_file}")
        input_mesh = read_mesh(file_path, mesh_drop_percent=percent)
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

    args = parser.parse_args()

    # Resolve source and destination paths
    source_dir = Path(args.source)
    destination_dir = Path(args.destination)

    # Run the processing
    process_dir(source_dir, destination_dir, percent=args.percent)
