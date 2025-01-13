from transformers import DistilBertModel, DistilBertTokenizer
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
import torch.nn as nn
import torch

from src.utils.loss import custom_loss
from src.utils.loss import hausdorff_distance
from src.utils.loss import chamfer_distance
from src.utils.meshes import WeightedMeshes


class TextEncoder(pl.LightningModule):

    def __init__(self, d_out, hidden_dim=None):

        super().__init__()

        # Create the text transformer
        self.model_name = "distilbert-base-multilingual-cased"
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.text_model = DistilBertModel.from_pretrained(self.model_name)

        # Take out dimension of the last linear layer
        last_linear_layer = self.text_model.transformer.layer[-1].output_layer_norm
        last_linear_layer_out = last_linear_layer.normalized_shape[0]

        self.d_out = d_out
        self.hidden_dim = hidden_dim if hidden_dim is not None else min(last_linear_layer_out, d_out)

        # Create the projection layers
        self.projection = nn.Sequential(
            nn.Linear(last_linear_layer_out, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.d_out)
        )

        # Freeze the text model parameters
        for p in self.text_model.parameters():
            p.requires_grad = False

    def forward(self, x):

        # Pass the text to the tokenizer and get a dictionary containing a tensor as result
        model_input = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        model_input = {k: v.to(self.device) for k, v in model_input.items()}

        # Give the tensor to the model and take  the last hidden state
        model_output = self.text_model(**model_input)
        last_hidden_states = model_output.last_hidden_state

        # Extract CLS token representation
        features = last_hidden_states[:, 0, :]

        # Project the last hidden state to the new embedding space
        projected_vec = self.projection(features)

        # Normalize
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


class Model(pl.LightningModule):
    def __init__(self,
                 latent_size,
                 input_dim=3,
                 lr=1e-3,
                 batch_size=4,
                 w_chamfer=1.0,
                 w_normal=1.0,
                 w_laplacian=1.0,
                 n_samples=5000
                 ):
        super().__init__()

        self.latent_size = latent_size
        self.input_dim = input_dim
        self.lr = lr
        self.batch_size = batch_size

        # Loss weights
        self.w_chamfer = w_chamfer
        self.w_normal = w_normal
        self.w_laplacian = w_laplacian
        self.n_samples = n_samples

        # Define the architecture
        self.gcn1 = GCNConv(self.input_dim, self.latent_size)
        self.gcn2 = GCNConv(self.latent_size, self.input_dim)
        self.text_encoder = TextEncoder(self.latent_size)

        # Define the loss function
        self.loss_fn = custom_loss

        self.save_hyperparameters()

    def forward(self, neutral_meshes, descriptions):
        # Ensure meshes are on the correct device
        neutral_meshes = neutral_meshes.to(self.device)

        # Get a packed representation of the meshes
        neutral_meshes_vertices_packed = neutral_meshes.verts_packed()
        neutral_meshes_edges_packed = neutral_meshes.edges_packed().T

        # Text conditioning
        text_condition = self.text_encoder(descriptions)  # (batch_size, latent_space)

        # We can use neutral_graph.batch to sum each text condition to the respective subgraph in the batch
        mesh_idx_per_vertex = neutral_meshes.verts_packed_to_mesh_idx()
        text_condition_per_subgraph = text_condition[mesh_idx_per_vertex]

        # Ensure all tensors are on the same device
        # TODO remove .to(device) once we are sure everything works
        neutral_meshes_vertices_packed = neutral_meshes_vertices_packed.to(self.device)
        neutral_meshes_edges_packed = neutral_meshes_edges_packed.to(self.device)
        text_condition_per_subgraph = text_condition_per_subgraph.to(self.device)

        x = self.gcn1(neutral_meshes_vertices_packed, neutral_meshes_edges_packed)
        x = x + text_condition_per_subgraph  # Add text conditioning
        x = torch.relu(x)
        x = self.gcn2(x, neutral_meshes_edges_packed)

        return x

    def inference(self, neutral_meshes, descriptions):
        displacements = self(neutral_meshes, descriptions)
        expression_meshes = neutral_meshes.offset_verts(displacements)
        return WeightedMeshes(
            expression_meshes.verts_list(),
            neutral_meshes.faces_list(),
            neutral_meshes.weights_list()
        )

    def common_step(self, batch):
        # Move batch data to device
        neutral_meshes = batch['neutral_meshes'].to(self.device)
        expression_meshes = batch['expression_meshes'].to(self.device)
        descriptions = batch['descriptions']

        # Nodes with updated features that will become offsets with training
        displacements = self(neutral_meshes, descriptions)
        predicted_meshes = neutral_meshes.offset_verts(displacements)  # returns new object

        loss = self.loss_fn(
            predicted_meshes,
            expression_meshes,
            w_chamfer=self.w_chamfer,
            w_normal=self.w_normal,
            w_laplacian=self.w_laplacian,
            n_samples=self.n_samples
        )

        self.log("disp_mean",
                 torch.abs(displacements).mean().item(),
                 batch_size=self.batch_size,
                 prog_bar=True,
                 logger=True)

        return predicted_meshes, loss

    def compute_metrics(self, pred, target):

        # Compute the metrices
        chamfer = chamfer_distance(pred, target)
        hausdorff = hausdorff_distance(pred, target)

        # Collect all metrics in a dictionary
        metrics = {
            'Chamfer Distance': chamfer,
            'Hausdorff Distance': hausdorff,
        }

        return metrics

    def training_step(self, batch, batch_idx):
        pred, loss = self.common_step(batch)
        self.compute_metrics(pred, batch)
        self.log("train_loss",
                 loss,
                 batch_size=self.batch_size,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, loss = self.common_step(batch)
        self.compute_metrics(pred, batch)
        self.log("val_loss",
                 loss,
                 batch_size=self.batch_size,
                 prog_bar=True,
                 logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        pred, loss = self.common_step(batch)
        self.log("test_loss",
                 loss,
                 batch_size=self.batch_size,
                 prog_bar=True,
                 logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
