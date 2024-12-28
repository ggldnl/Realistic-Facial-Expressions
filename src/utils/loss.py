import torch
import torch.nn.functional as F
from torchmetrics.functional import pairwise_euclidean_distance
from torch_geometric.utils import to_dense_adj


def chamfer_distance(pred, target):
    """
    Compute the chamfer distance between prediction and target tensors.
    Chamfer distance is efficient to compute and works for tensors with
    different sizes. This requires us to compute a distance matrix that
    is too big to fit to memory in most cases.
    """

    # Compute pairwise distances
    distances = pairwise_euclidean_distance(pred, target)

    # For each point in pred, find the closest point in target
    min_pred_to_target = distances.min(dim=1)[0]

    # For each point in target, find the closest point in min
    min_target_to_pred = distances.min(dim=0)[0]

    # Compute average of these distances
    chamfer = torch.mean(min_pred_to_target) + torch.mean(min_target_to_pred)

    return chamfer

def mse_loss(pred, target):
    """
    Computes the Mean Squared Error (MSE) loss between the predicted and target values.
    Penalizes larger errors more heavily, which can help converge to accurate predictions,
    but it is sensitive to outliers as larger errors dominate the loss due to squaring.
    """
    return F.mse_loss(pred, target)

def compute_adjacency_matrix_batch(data_batch):
    """
    Computes the adjacency matrix for the meshes given a batch.
    """
    edge_index = data_batch.edge_index
    batch = data_batch.batch
    num_nodes = data_batch.num_nodes

    # Dense adjacency matrices (batch_size, num_nodes, num_nodes)
    adj_matrix = to_dense_adj(edge_index, batch=batch, max_num_nodes=num_nodes)
    return adj_matrix

def compute_adjacency_matrix_mesh(vertices, faces):
    """
    Computes the adjacency matrix for a mesh given vertices and faces.
    """
    num_vertices = vertices.shape[0]

    # Initialize an empty adjacency matrix
    adj_matrix = torch.zeros((num_vertices, num_vertices), dtype=torch.float32)

    for face in faces:

        v0, v1, v2 = face

        # Add bidirectional edges to each pair of vertices of the face
        adj_matrix[v0, v1] = 1
        adj_matrix[v1, v0] = 1
        adj_matrix[v1, v2] = 1
        adj_matrix[v2, v1] = 1
        adj_matrix[v2, v0] = 1
        adj_matrix[v0, v2] = 1

    return adj_matrix

def smoothness_regularization(vertices, adj):
    """
    Uses the Laplacian operator to encourage smoothness in the mesh. This does not explicitly enforce 
    alignment with a target shape or data, it is just a regularization term.

    It requires the adjacency matrix, that can be computed starting from vertices and edges.
    """
    laplacian = adj @ vertices - vertices  # Laplacian operator
    loss = torch.mean(torch.norm(laplacian, dim=-1))  # Smoothness penalty
    return loss

def stability_regularization(vertex_values, initial_vertex_values):
    """
    Computes the penalty for large changes in vertex values. It requires the initial vertex values.
    It makes the vertices less likely to have big changes.
    """
    return torch.sum((vertex_values - initial_vertex_values) ** 2)

def mesh_custom_loss(pred, target, neutral, lambda_smoothness=1, lambda_stability=1):

    # Compute the actual loss (chamfer for meshes)
    task_loss = chamfer_distance(pred.x, target.x)

    # Compute regularization terms
    adjacency_matrix = compute_adjacency_matrix_batch(neutral)
    smoothness_loss = smoothness_regularization(neutral.x, adjacency_matrix)
    stability_loss = stability_regularization(pred, neutral)

    result = task_loss + lambda_stability * stability_loss + lambda_smoothness * smoothness_loss
    result.requires_grad = True

    return result
