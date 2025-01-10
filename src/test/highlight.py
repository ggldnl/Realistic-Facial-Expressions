from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import trimesh


def load_mesh_file(file_path):
    """
    Load a mesh file using trimesh.
    Supports various formats including .obj, .stl, .ply, .glb, .gltf

    Parameters:
    file_path: str, path to the mesh file

    Returns:
    vertices: np.array of shape (N, 3)
    faces: np.array of shape (M, 3)
    """
    mesh = trimesh.load(file_path)
    return np.array(mesh.vertices), np.array(mesh.faces)


def cartesian_to_spherical(cartesian_points):
    """
    Convert Cartesian coordinates to spherical coordinates (r, theta, phi).

    Parameters:
        cartesian_points (np.ndarray): Array of shape (N, 3) with Cartesian coordinates.

    Returns:
        np.ndarray: Array of shape (N, 3) with spherical coordinates (r, theta, phi):
            - r: radial distance
            - theta: polar angle (0 to pi)
            - phi: azimuthal angle (-pi to pi)
    """
    x, y, z = cartesian_points[:, 0], cartesian_points[:, 1], cartesian_points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1, 1))  # Handle numerical precision issues
    phi = np.arctan2(y, x)  # Handles edge cases gracefully
    return np.column_stack((r, theta, phi))


def spherical_to_cartesian(spherical_points):
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian coordinates.

    Parameters:
        spherical_points (np.ndarray): Array of shape (N, 3) with spherical coordinates:
            - r: radial distance
            - theta: polar angle (0 to pi)
            - phi: azimuthal angle (-pi to pi)

    Returns:
        np.ndarray: Array of shape (N, 3) with Cartesian coordinates (x, y, z).
    """
    r, theta, phi = spherical_points[:, 0], spherical_points[:, 1], spherical_points[:, 2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack((x, y, z))


def gaussian_weight(distance, sigma):
    """
    Compute Gaussian weight based on distance.

    Parameters:
        distance: Angular distance in radians
        sigma: Spread parameter controlling how quickly weight decreases with distance

    Returns:
        float: Weight between 0 and 1
    """
    return np.exp(-0.5 * (distance / sigma) ** 2)


def angular_distance(theta1, phi1, theta2, phi2):
    """
    Compute the angular distance between two points on a unit sphere.
    This uses the great circle distance formula (haversine formula).

    Parameters:
        theta1, theta2: polar angles (0 to pi)
        phi1, phi2: azimuthal angles (-pi to pi)

    Returns:
        float: Angular distance in radians (0 to pi)
    """
    # Haversine formula for angular distance on a sphere
    cos_distance = (np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2) +
                    np.cos(theta1) * np.cos(theta2))
    # Handle numerical precision issues
    cos_distance = np.clip(cos_distance, -1.0, 1.0)
    return np.arccos(cos_distance)


def add_weights_to_obj(input_obj, output_obj, weights):
    """
    Add weights to the OBJ file as vertex normals.

    Parameters:
        input_obj (str): Path to the input OBJ file.
        output_obj (str): Path to save the modified OBJ file.
        weights (list or np.ndarray): List or array of weights (one per vertex).
    """
    with open(input_obj, 'r') as infile, open(output_obj, 'w') as outfile:

        for line in infile:
            # Copy all other lines unchanged
            outfile.write(line)

        for weight in weights:

            # Normalize weight to [0, 1] (if not already in range)
            weight = np.clip(weight, 0.0, 1.0)

            # Use weight as grayscale color (R=G=B=weight)
            r = g = b = weight

            # Write the vertex with color
            outfile.write(f"vn {r} {g} {b}\n")


def process_dir(
        source,
        destination,
        sigma=0.1
):
    """
    Recursively preprocess .obj files in the source directory, adding weights to each node based on
    their positions on the surface (cover front face) and saving the results in the destination directory.

    Parameters:
        source (str or Path): Path to the mesh file.
        destination (str or Path): Path where to save the processed file.
        sigma (float): Spread parameter for weight calculation.
    """

    pbar = tqdm(source.rglob("*.obj"))
    for file_path in pbar:

        # Compute the relative path to maintain directory structure
        relative_path = file_path.relative_to(source)
        destination_file = destination / relative_path

        # Ensure the destination directory exists
        destination_file.parent.mkdir(parents=True, exist_ok=True)

        # Compute the weights
        # print(f"Processing {file_path} -> {destination_file}")
        pbar.set_description(f"Processing {file_path} -> {destination_file}")

        vertices, _ = load_mesh_file(file_path)

        # Compute the parameters of a sphere around the mesh.
        # This sphere will be big enough to include all the vertices.
        # We will use its radius to define the region of interest.
        sphere_center = np.mean(vertices, axis=0)
        vertices_centered = vertices - sphere_center
        sphere_radius = np.max(np.linalg.norm(vertices_centered, axis=1))  # Max distance of vertex from origin

        # Convert the mesh points to spherical coordinates
        mesh_spherical_points = cartesian_to_spherical(vertices_centered)
        theta_points, phi_points = mesh_spherical_points[:, 1], mesh_spherical_points[:, 2]

        # Define regions of interest on the sphere.
        # The centers of these regions could be anywhere, they will be projected on the sphere.
        # We will define only one region that takes the upper part of the sphere.
        roi_centers = np.array([(0, 0, 10)])  # Only one point at z=10
        roi_radii = [sphere_radius]  # Same radius as before, the region should cover a whole hemisphere

        # Project the roi centers on the sphere and then translate to spherical coordinates
        roi_centers_spherical = cartesian_to_spherical(roi_centers)
        theta_roi_centers, phi_roi_centers = roi_centers_spherical[:, 1], roi_centers_spherical[:, 2]  # Take theta and phi of the roi centers

        # Compute the actual weights
        weights = np.zeros(vertices_centered.shape[0])
        for theta_center, phi_center, radius in zip(theta_roi_centers, phi_roi_centers, roi_radii):

            # For each vertex, compute its angular distance to the current ROI center
            distances = angular_distance(
                theta_points,
                phi_points,
                theta_center,
                phi_center
            )

            # Compute weights using a Gaussian function
            current_weights = gaussian_weight(distances, sigma)

            # Add the weights for this ROI center to the total weights
            weights += current_weights

        # Normalize weights to [0, 1] range
        if np.max(weights) > 0:  # Avoid division by zero
            weights = weights / np.max(weights)

        # Add the weights to the obj file as color
        add_weights_to_obj(file_path, destination_file, weights)


if __name__ == '__main__':

    """
    Preprocess the dataset adding weights to all the mesh objects (obj) we encounter.
    We store the result in another directory, keeping the directory tree unchanged.
    The weights are stored as vertex normals.
    """

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Add a weight to each node of the mesh.")
    parser.add_argument('-s', '--source', type=str, required=True, help="Path to the source directory containing .obj files.")
    parser.add_argument('-d', '--destination', type=str, required=True, help="Path to the destination directory to save simplified meshes.")
    parser.add_argument('-g', '--sigma', type=float, default=1.0, help="Spread parameter for weight calculation.")
    args = parser.parse_args()

    # Resolve source and destination paths
    source_dir = Path(args.source)
    destination_dir = Path(args.destination)

    # Run the processing
    process_dir(
        source_dir,
        destination_dir,
        sigma=args.sigma
    )
