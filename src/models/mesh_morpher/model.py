import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Tuple, Optional, Dict, Any


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
        batches, channels = x.shape
        assert channels == self.num_input_channels, \
            f"Expected {self.num_input_channels} channels, got {channels}"

        res = x @ self.B.to(x.device)
        res = 2 * np.pi * res
        return torch.cat([x, torch.sin(res), torch.cos(res)], dim=1)


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
        return x * alpha


class NeuralStyleField(pl.LightningModule):
    """Neural Style Field implementation using PyTorch Lightning"""

    def __init__(
            self,
            sigma: float = 10.0,
            depth: int= 4,
            width: int = 256,
            colordepth: int = 2,
            normdepth: int = 2,
            normratio: float = 0.1,
            clamp: Optional[str] = 'tanh',
            normclamp: Optional[str] = 'tanh',
            n_iter: int = 6000,
            input_dim: int = 3, #todo check
            progressive_encoding: bool = True,
            exclude: int = 0,
            learning_rate: float = 1e-3,#todo check
            weight_decay: float = 0.0,
            lr_decay: float = 1.0,
            decay_step: int = 100
    ):
        super().__init__()

        # Store configuration
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

        # Initialize progressive encoding
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

        # Color branch
        self.mlp_rgb = self._build_branch(width, colordepth, output_dim=3)

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Base network
        for layer in self.base:
            x = layer(x)

        # Color branch
        colors = x
        for layer in self.mlp_rgb:
            colors = layer(colors)

        # Normal branch
        displ = x
        for layer in self.mlp_normal:
            displ = layer(displ)

        # Apply clamping if specified
        if self.clamp == "tanh":
            colors = F.tanh(colors) / 2
        elif self.clamp == "clamp":
            colors = torch.clamp(colors, 0, 1)

        if self.normclamp == "tanh":
            displ = F.tanh(displ) * self.normratio
        elif self.normclamp == "clamp":
            displ = torch.clamp(displ, -self.normratio, self.normratio)

        return colors, displ

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
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

    def training_step(self, batch, batch_idx):
        x = batch
        colors, displ = self(x)
        loss = self._calculate_loss(colors, displ)
        self.log('train_loss', loss)
        return loss

    def _calculate_loss(self, colors, displ):
        pass