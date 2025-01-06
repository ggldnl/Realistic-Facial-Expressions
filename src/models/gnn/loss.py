from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)


def loss(pred, target, w_chamfer=1.0, w_edge=1.0, w_normal=1.0, w_laplacian=1.0, n_samples=5000):

    # Sample a set of points from the surface of each mesh
    sample_target = sample_points_from_meshes(target, n_samples)
    sample_pred = sample_points_from_meshes(pred, n_samples)

    # Compute the chamfer loss (this is batched -> expects tensors with shape (num_graphs, max_num_vertices, 3))
    loss_chamfer, _ = chamfer_distance(sample_target, sample_pred)

    # Compute edge loss (length of the edges)
    loss_edge = mesh_edge_loss(pred)

    # Compute consistency loss
    loss_normal = mesh_normal_consistency(pred)

    # Compute laplacian smoothing loss
    loss_laplacian = mesh_laplacian_smoothing(pred, method="uniform")

    return loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
