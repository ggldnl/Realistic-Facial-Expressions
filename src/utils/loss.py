import torch
import torch.nn.functional as F
from torchmetrics.functional import pairwise_euclidean_distance


def chamfer_distance(pred, target):
    """
    Compute the chamfer distance between prediction and target tensors.
    Chamfer distance is efficient to compute and works for tensors with
    different sizes.
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

def chamfer_distance(pred, target):
    """
    Computes the Chamfer Distance between two sets of points. The Chamfer Distance measures
    the distance between the points of the predicted and target meshes in a point cloud.
    It works even if predicted and target tensors have different shapes. It is not sensitive
    to the global structure of the point clouds, as it only considers nearest neighbors.
    May fail to capture fine-grained details.
    """
    pred_expand = pred.unsqueeze(1)  # (N, 1, 3)
    target_expand = target.unsqueeze(0)  # (1, M, 3)
    distances = torch.cdist(pred_expand, target_expand)  # Pairwise distances

    pred_to_target = torch.min(distances, dim=1)[0].mean()
    target_to_pred = torch.min(distances, dim=0)[0].mean()

    return pred_to_target + target_to_pred

def surface_normal_consistency_loss(pred_normals, target_normals):
    """
    Computes the Surface Normal Consistency Loss, which measures the cosine similarity
    between predicted and target normals. It encourages smooth surfaces by penalizing
    deviations in normal directions.
    """
    # Ensure the normals are unit vectors
    pred_normals = torch.nn.functional.normalize(pred_normals, dim=-1)
    target_normals = torch.nn.functional.normalize(target_normals, dim=-1)

    # Compute cosine similarity
    cos_similarity = torch.sum(pred_normals * target_normals, dim=-1)  # Dot product
    loss = 1 - cos_similarity.mean()  # 1 - cosine similarity
    return loss

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

def smoothness_regularization(vertices, adj):
    """
    Uses the Laplacian operator to encourage smoothness in the mesh, reducing irregularities in the geometry.
    This does not explicitly enforce alignment with a target shape or data, it is just a regularization term.

    It requires the adjacency matrix, that can be computed starting from vertices and edges.
    """
    laplacian = adj @ vertices - vertices  # Laplacian operator
    loss = torch.mean(torch.norm(laplacian, dim=-1))  # Smoothness penalty
    return loss
