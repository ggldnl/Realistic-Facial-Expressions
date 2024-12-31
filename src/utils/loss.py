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

def compute_adjacency_matrix(vertices, faces):
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

def L2(pred):
    """
    L2 penalty on the predicted node positions (sum of squared positions)
    """
    l2_reg = torch.sum(pred ** 2)
    return l2_reg


def smoothness_regularization(displacements, adj, batch):
    """
    Computes the smoothness regularization loss for a batch of graphs using the Laplacian.
    The Laplacian at a vertex is the difference between the vertex displacement and
    the average displacement of its neighbors.
    Args:
        displacements: Tensor of shape [num_total_nodes, 3] containing vertex displacements
        adj: Tensor of shape [num_graphs, max_nodes, max_nodes] containing adjacency matrices
        batch: Tensor of shape [num_total_nodes] indicating which graph each node belongs to
    Returns:
        Scalar tensor containing the mean smoothness loss across all graphs
    """
    num_graphs = batch.max().item() + 1
    losses = []

    for i in range(num_graphs):
        # Get mask for current graph's nodes
        node_mask = batch == i

        # Get displacements for current graph
        graph_displacements = displacements[node_mask]  # [num_nodes_in_graph, 3]

        # Get adjacency matrix for current graph
        graph_adj = adj[i]  # [max_nodes, max_nodes]

        # Trim adjacency matrix to match actual number of nodes in this graph
        num_nodes = graph_displacements.shape[0]
        graph_adj = graph_adj[:num_nodes, :num_nodes]  # [num_nodes, num_nodes]

        # Normalize adjacency matrix
        degree = graph_adj.sum(dim=-1, keepdim=True).clamp(min=1)  # [num_nodes, 1]
        normalized_adj = graph_adj / degree  # [num_nodes, num_nodes]

        # Compute Laplacian: L = x - Ax (where A is normalized adjacency)
        neighbor_mean = torch.matmul(normalized_adj, graph_displacements)  # [num_nodes, 3]
        laplacian = graph_displacements - neighbor_mean  # [num_nodes, 3]

        # Compute L2 norm of Laplacian per vertex and take mean
        loss = torch.norm(laplacian, p=2, dim=-1).mean()  # scalar
        losses.append(loss)

    return torch.stack(losses).mean()

def stability_regularization(vertex_values, initial_vertex_values):
    """
    Computes the penalty for large changes in vertex values. It requires the initial vertex values.
    It makes the vertices less likely to have big changes.
    """
    return torch.sum((vertex_values - initial_vertex_values) ** 2)

def mesh_custom_loss(
        displaced,      # Predicted neutral vertices (full batch)
        target,         # Initial expression vertices (full batch)
        vertices=None,  # Initial neutral vertices (full batch)
        edges=None,     # Initial neutral edges (full batch)
        batch=None,     # Batch tensor (indicates which graph each node in the batch belongs to)
        lambda_smoothness=1,
        lambda_stability=1
):

    # Compute the actual loss (chamfer for meshes)
    task_loss = chamfer_distance(displaced, target)

    # Compute regularization terms

    if edges is not None and vertices is not None and batch is not None:
        adjacency_matrix = to_dense_adj(edges, batch=batch)
        smoothness_loss = smoothness_regularization(displaced, adjacency_matrix, batch)
    else:
        smoothness_loss = L2(displaced)

    stability_loss = stability_regularization(displaced, vertices)

    result = task_loss + lambda_stability * stability_loss + lambda_smoothness * smoothness_loss
    # result.requires_grad = True

    return result
