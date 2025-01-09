from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
from pytorch3d.loss import mesh_edge_loss
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.loss import mesh_normal_consistency
import torch


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

def custom_loss(
        pred,
        target,
        w_chamfer=1.0,
        w_edge=1.0,
        w_normal=1.0,
        w_laplacian=1.0,
        n_samples=5000
):

    # Sample a set of points from the surface of each mesh.
    # The pred Meshes object should have the normals, that we arbitrarily used to store weights
    # for the chamfer distance. If the normals are available in the pred Meshes, use them as a mask
    sample_target = sample_points_from_meshes(target, n_samples)  # (batch_size, n_samples, num_features)
    sample_pred, mask = sample_points_from_meshes(pred, n_samples, return_normals=True)

    if mask is None:
        mask = torch.ones_like(sample_pred)

    # Select the first value of the last dimension of the mask.
    # During creation, we arbitrarily set the weights in the normals. All the components of the
    # normals are the same normalized weight, so we can select the first one and it will do.
    mask = mask[:, :, 0]

    # Get bidirectional distances. By default, chamfer_distance returns the mean of both directions.
    # We need the two raw distances instead.
    # dist1, dist2 = chamfer_distance(sample_pred, sample_target, return_raw=True)
    dist1, dist2 = get_chamfer_distances(sample_pred, sample_target)

    # Apply mask to both directions of the chamfer distance
    # dist1: for each point in pred, distance to nearest point in target
    # dist2: for each point in target, distance to nearest point in pred
    loss_chamfer = (dist1 * mask).mean() + dist2.mean()

    # Compute the chamfer loss (this is batched -> expects tensors with shape (num_graphs, max_num_vertices, 3))
    # loss_chamfer, _ = chamfer_distance(sample_target, sample_pred)

    # Compute edge loss (length of the edges)
    loss_edge = mesh_edge_loss(pred)

    # Compute consistency loss
    loss_normal = mesh_normal_consistency(pred)

    # Compute laplacian smoothing loss
    loss_laplacian = mesh_laplacian_smoothing(pred, method="uniform")

    return loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
