import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Tuple, Optional, Dict, Any
from src.models.mlp.clip import CLIP
from src.utils.mesh_utils import tensor_to_mesh
from src.utils.renderer import Renderer


class FourierFeatureTransform(pl.LightningModule):
    """
    Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
    https://arxiv.org/abs/2006.10739
    """

    def __init__(self,
                 num_input_channels: int,
                 mapping_size: int = 256,
                 scale: float = 10,
                 exclude: int = 0):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.mapping_size = mapping_size
        self.exclude = exclude

        # Initialize and sort B matrix by L2 norm
        B = torch.randn((num_input_channels, mapping_size)) * scale
        B_sort = sorted(B, key=lambda x: torch.norm(x, p=2))
        self.B = nn.Parameter(torch.stack(B_sort), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input to 2D if needed: (batch_size, ..., channels) -> (batch_size * ..., channels)
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.reshape(-1, original_shape[-1])

        assert x.shape[-1] == self.num_input_channels, \
            f"Expected {self.num_input_channels} channels, got {x.shape[-1]}"

        res = x @ self.B.to(x.device)
        res = 2 * np.pi * res
        output = torch.cat([x, torch.sin(res), torch.cos(res)], dim=-1)

        # Reshape back to original dimensions
        if len(original_shape) > 2:
            output = output.reshape(*original_shape[:-1], -1)

        return output


class ProgressiveEncoding(pl.LightningModule):
    """Progressive encoding module for feature learning"""

    def __init__(self,
                 mapping_size: int,
                 T: int,
                 d: int = 3,
                 apply: bool = True):
        super().__init__()
        self.t = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.mapping_size = mapping_size
        self.T = T
        self.d = d
        self.tau = 2 * self.mapping_size / self.T
        self.indices = nn.Parameter(torch.arange(self.mapping_size), requires_grad=False)
        self.apply = apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.apply:
            alpha = torch.ones(2 * self.mapping_size + self.d, device=x.device)
        else:
            alpha = ((self.t - self.tau * self.indices) / self.tau).clamp(0, 1).repeat(2)
            alpha = torch.cat([torch.ones(self.d, device=x.device), alpha], dim=0)

        self.t += 1
        return x * alpha.view(1, -1)  # Add dimension for broadcasting


class NeuralStyleField(pl.LightningModule):
    """Neural Style Field implementation using PyTorch Lightning"""

    def __init__(
            self,
            sigma: float = 10.0,
            depth: int = 4,
            width: int = 256,
            colordepth: int = 2,
            normdepth: int = 2,
            normratio: float = 0.1,
            clamp: Optional[str] = 'tanh',
            normclamp: Optional[str] = 'tanh',
            n_iter: int = 6000,
            input_dim: int = 3,  #todo check
            progressive_encoding: bool = True,
            exclude: int = 0,
            learning_rate: float = 1e-3,  #todo check
            weight_decay: float = 0.0,
            lr_decay: float = 1.0,
            decay_step: int = 100
    ):
        super().__init__()
        self.sigma = sigma
        self.depth = depth
        self.width = width
        self.colordepth = colordepth
        self.normdepth = normdepth
        self.normratio = normratio
        self.clamp = clamp
        self.normclamp = normclamp
        self.n_iter = n_iter
        self.input_dim = input_dim
        self.progressive_encoding = progressive_encoding
        self.exclude = exclude
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.decay_step = decay_step

        self.pe = ProgressiveEncoding(
            mapping_size=width,
            T=n_iter,
            d=input_dim
        )

        # Build base network
        self.base = nn.ModuleList([
            FourierFeatureTransform(input_dim, width, sigma, exclude),
            self.pe if progressive_encoding else nn.Identity(),
            nn.Linear(width * 2 + input_dim, width),
            nn.ReLU()
        ])

        # Add depth layers
        for _ in range(depth):
            self.base.extend([
                nn.Linear(width, width),
                nn.ReLU()
            ])

        # Normal branch
        self.mlp_normal = self._build_branch(width, normdepth, output_dim=1)

        self.save_hyperparameters()

    def _build_branch(self, width: int, depth: int, output_dim: int) -> nn.ModuleList:
        """Helper method to build network branches"""
        layers = []
        for _ in range(depth):
            layers.extend([
                nn.Linear(width, width),
                nn.ReLU()
            ])
        layers.append(nn.Linear(width, output_dim))
        return nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base network
        for layer in self.base:
            x = layer(x)

        displ = x
        for layer in self.mlp_normal:
            displ = layer(displ)

        if self.normclamp == "tanh":
            displ = F.tanh(displ) * self.normratio
        elif self.normclamp == "clamp":
            displ = torch.clamp(displ, -self.normratio, self.normratio)

        return displ


class NeuralStyleTransfer(pl.LightningModule):
    """Lightning module for neural style transfer training"""

    def __init__(self,
                 clip_model: str = "openai/clip-vit-base-patch32",
                 prompt: Optional[str] = None,
                 reference_image: Optional[str] = None,
                 sigma: float = 10.0,
                 depth: int = 4,
                 width: int = 256,
                 colordepth: int = 2,
                 normdepth: int = 2,
                 normratio: float = 0.1,
                 learning_rate: float = 1e-3,
                 lr_decay: float = 0.9,
                 decay_step: int = 1000,
                 batch_size: int = 4,
                 res: int = 224):
        super().__init__()
        self.save_hyperparameters()

        # Initialize CLIP
        self.clip = CLIP(model_name=clip_model, res=res)

        # Set up renderer
        self.renderer = Renderer()

        # Get target features
        if prompt:
            self.target_features = self.clip.encode_prompt(prompt)
        elif reference_image:
            self.target_features = self.clip.encode_image(reference_image)
        else:
            print('No prompt or reference image, training...')

        # Initialize neural style field
        self.style_field = NeuralStyleField(
            sigma=sigma,
            depth=depth,
            width=width,
            colordepth=colordepth,
            normdepth=normdepth,
            normratio=normratio,
            learning_rate=learning_rate
        )

        # Store settings
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.decay_step = decay_step
        self.batch_size = batch_size

    def forward(self, vertices):
        return self.style_field(vertices)

    def common_step(self, batch, batch_idx):
        for idx in range(self.batch_size):
            vertices = batch['neutral_graph'][idx].x

            # Get style field output
            displacements = self(vertices)

            # Apply displacements
            deformed_vertices = vertices + displacements

            computed_mesh = tensor_to_mesh(vertices=deformed_vertices,
                                           faces=batch['neutral_graph'][idx].faces, )

            target_mesh = tensor_to_mesh(vertices=batch['expression_graph'][idx].x,
                                         faces=batch['expression_graph'][idx].faces, )

            # Render views
            computed_rendered_images = self.renderer.render_viewpoints(
                model_in=computed_mesh,
                num_views=8,
                radius=600,
                elevation=0,
                scale=1.0,
                rend_size=(1024, 1024),
            )
            target_rendered_images = self.renderer.render_viewpoints(
                model_in=target_mesh,
                num_views=8,
                radius=600,
                elevation=0,
                scale=1.0,
                rend_size=(1024, 1024),
            )

            # Calculate CLIP loss with augmentations
            loss = 0
            for idx, rendered_image in enumerate(computed_rendered_images):
                encoded_renders = self.clip.encode_augmented_renders(rendered_image)
                loss -= torch.mean(torch.cosine_similarity(
                    rendered_image,
                    target_rendered_images[idx]
                    # self.target_features.expand(batch_size, -1)
                ))

            return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("val_loss",
                 loss,
                 prog_bar=True,
                 logger=True,
                 batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("test_loss",
                 loss,
                 prog_bar=True,
                 logger=True,
                 batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )

        if self.lr_decay < 1 and self.decay_step > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.decay_step,
                gamma=self.lr_decay
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "train_loss"
            }

        return optimizer
