from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
from pytorch3d.loss import mesh_edge_loss
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.loss import mesh_normal_consistency
import torch
import numpy as np

def get_chamfer_distances(pred, target):

    # Add batch dimension if needed
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)  # Shape: (1, n_samples, 3)
    if target.dim() == 2:
        target = target.unsqueeze(0)  # Shape: (1, n_samples, 3)

    # Compute the nearest neighbors in both directions
    # pred -> target: for each point in pred, find nearest in target
    nearest_pred = knn_points(pred, target, K=1)
    # target -> pred: for each point in target, find nearest in pred
    nearest_target = knn_points(target, pred, K=1)

    # Get the distances.
    # dists has shape (batch_size, n_points, K), but since K=1, we can squeeze it
    dist1 = nearest_pred.dists.squeeze(-1)  # Shape: (batch_size, n_samples)
    dist2 = nearest_target.dists.squeeze(-1)  # Shape: (batch_size, n_samples)

    return dist1, dist2


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


def mask_face_landmarks(vertices, sigma=1):
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
    theta_roi_centers, phi_roi_centers = roi_centers_spherical[:, 1], roi_centers_spherical[:,
                                                                      2]  # Take theta and phi of the roi centers

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

    return weights

def custom_loss(
        pred,
        target,
        w_chamfer=1.0,
        w_normal=1.0,
        w_laplacian=1.0,
        n_samples=5000
):

    # Sample a set of points from the surface of each mesh.
    # Along with all the sampled points we take a combination of the weights of the vertices
    # from where the points are derived
    target_sample, target_mask = target.sample_points(n_samples)  # (batch_size, n_samples, num_features)
    pred_sample, pred_mask = pred.sample_points(n_samples)

    # Get bidirectional distances. By default, chamfer_distance returns the mean of both directions.
    # We need the two raw distances instead.
    # dist1, dist2 = chamfer_distance(sample_pred, sample_target, return_raw=True)
    dist1, dist2 = get_chamfer_distances(pred_sample, target_sample)

    # Apply mask to both directions of the chamfer distance
    # dist1: for each point in pred, distance to nearest point in target
    # dist2: for each point in target, distance to nearest point in pred
    loss_chamfer = (dist1 * pred_mask).mean() + (dist2 * target_mask).mean()

    # Compute the chamfer loss (this is batched -> expects tensors with shape (num_graphs, max_num_vertices, 3))
    # loss_chamfer, _ = chamfer_distance(sample_target, sample_pred)

    # Compute consistency loss
    loss_normal = mesh_normal_consistency(pred)

    # Compute laplacian smoothing loss
    loss_laplacian = mesh_laplacian_smoothing(pred, method="uniform")

    return loss_chamfer * w_chamfer + loss_normal * w_normal + loss_laplacian * w_laplacian
