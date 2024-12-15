from transformers import DistilBertModel, DistilBertTokenizer
import pytorch_lightning as pl
import torch.nn as nn
import torch

from src.utils.loss import mse_loss


class Projection(pl.LightningModule):

    def __init__(self, layer_sizes, activation_fn=nn.ReLU, d=0.4):

        super().__init__()

        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least two elements (input and output).")

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No activation or dropout after the last layer
                layers.append(activation_fn())
                layers.append(nn.Dropout(p=d))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class TextEncoder(pl.LightningModule):

    def __init__(self, d_out):

        super().__init__()

        self.d_out = d_out

        # Create the text transformer
        self.model_name = "distilbert-base-multilingual-cased"
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.text_model = DistilBertModel.from_pretrained(self.model_name)

        # Take in and out dimensions of the last linear layer
        last_linear_layer = self.text_model.transformer.layer[-1].output_layer_norm
        last_linear_layer_out = last_linear_layer.normalized_shape[0]

        # Create the projection layers
        self.projection = Projection([last_linear_layer_out, last_linear_layer_out, d_out])

        # Freeze the text model parameters
        for p in self.text_model.parameters():
            p.requires_grad = False

    def forward(self, x):

        # Pass the text to the tokenizer and get a dictionary containing a tensor as result
        model_input = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)

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

class MeshAutoEncoder(pl.LightningModule):

    def __init__(self, layers_in, latent_size, layers_out=None, text_embed_size=None, lr=1e-3):

        super().__init__()

        self.layers_in = layers_in
        self.latent_size = latent_size
        self.layers_out = layers_out if layers_out is not None \
            else [layers_in[i] for i in range(len(layers_in) - 1, -1, -1)]  # else mirror layers_in
        self.text_embed_size = text_embed_size if text_embed_size is not None else latent_size
        self.lr = lr

        self.projection_in = Projection(self.layers_in + [self.latent_size])
        self.text_encoder = TextEncoder(self.text_embed_size)
        self.projection_out = Projection([self.latent_size + self.text_embed_size] + self.layers_out)

        # TODO choose a better loss
        self.loss_fn = mse_loss

    def common_step(self, batch):

        neutral_vertices = batch['neutral_vertices']
        target = batch['expression_vertices']
        descriptions = batch['description']
        batch_size = neutral_vertices.shape[0]

        # TODO this should be fixed using normals or something
        features = neutral_vertices.view(batch_size, -1)
        projection_in_result = self.projection_in(features)

        text_embed = self.text_encoder(descriptions)
        concat_vec = torch.hstack([projection_in_result, text_embed])

        projection_out_result = self.projection_out(concat_vec)
        displacements = projection_out_result.view(batch_size, -1, 3)
        pred = neutral_vertices + displacements  # Meshes with artificial expression

        loss = self.loss_fn(pred, target)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):

        trainable_parameters = list(self.projection_in.parameters()) + \
            list(self.projection_out.parameters()) + \
            list(self.text_encoder.projection.parameters())

        optimizer = torch.optim.AdamW(trainable_parameters, lr=self.lr)

        return optimizer


if __name__ == '__main__':

    # Test autoencoder
    layers_in = [26404, 10000, 5000, 2500]
    latent_size = 512
    text_embed_size = 512

    model = MeshAutoEncoder(layers_in, latent_size, text_embed_size=text_embed_size)
    print(model)
