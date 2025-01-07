from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.loss import mesh_edge_loss
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.loss import mesh_normal_consistency

import torch.nn.functional as F


def pad_mesh(tensor, target_size):
    """
    Pad a tensor to the desired size.
    """
    pad_size = target_size - tensor.size(0)
    return F.pad(tensor, (0, 0, 0, pad_size))


def unpack_meshes(pack, batch_indices):
    unique_indices = batch_indices.unique()
    return torch.stack([pack[batch_indices == idx] for idx in unique_indices])


def pack_meshes(batch):
    """
    Convert (num_meshes, max_num_nodes, num_features) a batch to a pack (num_meshes*max_num_nodes, num_features).
    It returns both the pack and the batch indices. The input meshes in the batch are supposed to be padded,
    otherwise the conversion will fail.
    """

    # Create an indexing tensor
    batch_indices = torch.cat([torch.full((batch[i].size(0),), i, dtype=torch.long) for i in range(len(batch))])

    # Concatenate the meshes into a single tensor
    pack = torch.cat(batch, dim=0)

    return pack, batch_indices


def custom_loss(pred, target, w_chamfer=1.0, w_edge=1.0, w_normal=1.0, w_laplacian=1.0, n_samples=5000):

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


if __name__ == '__main__':
    """
    Convert a single tensor containing all the meshed in the graph into 
    multiple tensors. The result will have shape (num_meshes, max_num_nodes, num_features).
    """

    import torch

    # Define three meshes, each with a different number of nodes
    m1 = torch.randn((100, 3))
    m2 = torch.randn((105, 3))
    m3 = torch.randn((95, 3))

    # Pad the meshes
    meshes = [m1, m2, m3]
    max_num_nodes = max([m.size(0) for m in meshes])
    padded_meshes = [pad_mesh(m, max_num_nodes) for m in meshes]

    pack, pack_indices = pack_meshes(padded_meshes)
    print('Packed meshes (three meshes padded to the same length and put in the same tensor):')
    print(pack.size())

    print('Batched meshes (three meshes padded to the same length and stacked in the same tensor, '
          'with the additional batch dimension):')
    batch = unpack_meshes(pack, pack_indices)
    print(batch.size())
