from transformers import DistilBertModel, DistilBertTokenizer
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pytorch_lightning as pl
import torch.nn as nn
import torch

from src.utils.renderer import Renderer

from src.utils.loss import mesh_custom_loss


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
                 ):

        super().__init__()

        self.latent_size = latent_size
        self.input_dim = input_dim
        self.lr = lr
        self.batch_size = batch_size

        # Define the architecture
        self.gcn1 = GCNConv(self.input_dim, self.latent_size)
        self.gcn2 = GCNConv(self.latent_size, self.input_dim)
        self.text_encoder = TextEncoder(self.latent_size)

        self.renderer = Renderer()

        # Define the loss function
        self.loss_fn = mesh_custom_loss

    def forward(self, neutral_graph, descriptions):

        # Text conditioning
        text_condition = self.text_encoder(descriptions).unsqueeze(1)

        x = self.gcn1(neutral_graph.x, neutral_graph.edge_index)
        x = x + text_condition
        x = torch.relu(x)
        x = self.gcn2(x, neutral_graph.edge_index)

        return x

    def common_step(self, batch):

        neutral_graph = batch['neutral_graph']
        target_graph = batch['expression_graph']
        descriptions = batch['description']

        displaced_vertices = self(neutral_graph, descriptions)

        # Reconstruct the graph by adding the same vertex index as before
        pred_graph = Data(x=displaced_vertices, edge_index=neutral_graph.edge_index)

        loss = self.loss_fn(pred_graph, target_graph, neutral_graph)  # pred batch graph, target batch graph, neutral batch graph

        return pred_graph, loss

    def compute_metrics(self, pred, batch):
        computed_rendered_images = []
        target_rendered_images = []

        """
        for idx in range(self.batch_size):

            computed_mesh = tensor_to_mesh(vertices=pred[idx],
                                            faces=batch['neutral_graph'][idx].faces,)

            target_mesh = tensor_to_mesh(vertices=batch['expression_graph'][idx].x,
                                          faces=batch['expression_graph'][idx].faces,)

            # Render views
            computed_rendered_images.append(self.renderer.render_viewpoints(
                model_in=computed_mesh,
                num_views=8,
                radius=600,
                elevation=0,
                scale=1.0,
                rend_size=(1024, 1024),
                return_images=True
            ))
            target_rendered_images.append(self.renderer.render_viewpoints(
                model_in=target_mesh,
                num_views=8,
                radius=600,
                elevation=0,
                scale=1.0,
                rend_size=(1024, 1024),
                return_images=True
            ))

            for computed, target in zip(computed_rendered_images, target_rendered_images):

                # Calculate metrics
                mse_loss = nn.MSELoss()
                mse = mse_loss(computed, target)
            """


    def training_step(self, batch, batch_idx):
        pred, loss = self.common_step(batch)
        self.compute_metrics(pred, batch)
        self.log("train_loss",
                 loss,
                 prog_bar=True,
                 logger=True,
                 batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, loss = self.common_step(batch)
        self.compute_metrics(pred, batch)
        self.log("val_loss",
                 loss,
                 prog_bar=True,
                 logger=True,
                 batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        pred, loss = self.common_step(batch)
        self.log("test_loss",
                 loss,
                 prog_bar=True,
                 logger=True,
                 batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
