from transformers import DistilBertModel, DistilBertTokenizer
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
import torch.nn as nn
import torch

from src.utils.loss import custom_loss
from src.utils.loss import hausdorff_distance
from src.utils.loss import chamfer_distance


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
        features = last_hidden_states[:, 0, :]
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

        assert self.latent_size / 2 >= self.input_dim, \
            "Latent size must be greater (at least double) than input dimension"

        # Encoder layers
        self.gcn_encoder = nn.ModuleList([
            GCNConv(self.input_dim, int(self.latent_size / 2)),
            GCNConv(int(self.latent_size / 2), self.latent_size)
        ])

        # Text encoder
        self.text_encoder = TextEncoder(self.latent_size)

        # Fusion module
        self.fusion = nn.Sequential(
            nn.Linear(self.latent_size * 2, self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, self.latent_size)
        )

        # Decoder layers with residual connections
        self.gcn_decoder = nn.ModuleList([
            GCNConv(self.latent_size, self.latent_size),
            GCNConv(self.latent_size, self.latent_size)
        ])

        # Final projection to get vertex offsets
        self.offset_proj = nn.Linear(self.latent_size, self.input_dim)

        # Activation functions
        self.activation = nn.LeakyReLU(0.2)

        self.loss_fn = custom_loss

        self.save_hyperparameters()

    def forward(self, neutral_meshes, descriptions):
        # Get packed representations
        vertices = neutral_meshes.verts_packed()
        edges = neutral_meshes.edges_packed().T

        # Encode mesh
        x = vertices
        for gcn in self.gcn_encoder:
            x = self.activation(gcn(x, edges))
        mesh_features = x

        # Encode text
        text_features = self.text_encoder(descriptions)
        mesh_idx_per_vertex = neutral_meshes.verts_packed_to_mesh_idx()
        text_features = text_features[mesh_idx_per_vertex]

        # Fuse features
        concat_features = torch.cat([mesh_features, text_features], dim=-1)
        fused_features = self.fusion(concat_features)

        # Decode with residual connections
        x = fused_features
        for gcn in self.gcn_decoder:
            x_new = self.activation(gcn(x, edges))
            x = x + x_new  # Residual connection

        # Project to offset space
        offsets = self.offset_proj(x)

        return offsets

    def common_step(self, batch):
        neutral_meshes = batch['neutral_meshes'].to(self.device)
        expression_meshes = batch['expression_meshes'].to(self.device)
        descriptions = batch['descriptions']

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

        self.log("loss", loss.item(), batch_size=self.batch_size, prog_bar=True, logger=True)
        self.log("disp_mean", torch.abs(displacements).mean().item(), batch_size=self.batch_size, prog_bar=True,
                 logger=True)

        return predicted_meshes, loss

    def compute_metrics(self, pred, target, stage='train'):
        chamfer = chamfer_distance(pred, target)
        hausdorff = hausdorff_distance(pred, target)

        self.log(f'{stage}_Chamfer', chamfer.item(), batch_size=self.batch_size, prog_bar=True, logger=True)
        self.log(f'{stage}_Hausdorff', hausdorff.item(), batch_size=self.batch_size, prog_bar=True, logger=True)

    def inference(self, neutral_meshes, descriptions):
        displacements = self(neutral_meshes, descriptions)
        expression_meshes = neutral_meshes.offset_verts(displacements)
        return expression_meshes

    def training_step(self, batch, batch_idx):
        pred, loss = self.common_step(batch)
        self.compute_metrics(pred, batch['expression_meshes'], stage='train')
        self.log("train_loss", loss, batch_size=self.batch_size, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, loss = self.common_step(batch)
        self.compute_metrics(pred, batch['expression_meshes'], stage='validation')
        self.log("val_loss", loss, batch_size=self.batch_size, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        pred, loss = self.common_step(batch)
        self.log("test_loss", loss, batch_size=self.batch_size, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
