"""Reproducible model definitions for the GJI resubmission.

This module centralizes the Transformer model definitions used to reproduce
Fig. 5 and Fig. 7 for the resubmitted GJI manuscript:

"A Forward-Inverse Dual-Constrained Transformer Framework for Rayleigh-Wave
Shear-Wave Velocity Inversion".

The default constructor arguments match the ForwardModel and InverseModel used
in the original notebooks and root-level models.py. Shape-related constants from
the notebooks are exposed as parameters so the same computation can be reused
with different period/depth grids when needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ForwardModel(nn.Module):
    """Transformer forward model mapping Vs profiles to dispersion curves.

    Default behavior is equivalent to the original implementation:
    input shape ``(batch, 301, 1)`` -> output shape ``(batch, 16, 2)``.
    """

    def __init__(
        self,
        n_periods: int = 16,
        n_channels: int = 2,
        n_depths: int = 301,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        input_channels: int = 1,
        dim_feedforward: int = 512,
    ) -> None:
        super().__init__()
        self.n_periods = n_periods
        self.n_channels = n_channels
        self.n_depths = n_depths
        self.input_channels = input_channels

        self.per_layer = nn.Linear(input_channels, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pred_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * n_depths, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_periods * n_channels),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        x = self.per_layer(src)
        x = self.encoder(x)
        return self.pred_layer(x).view(-1, self.n_periods, self.n_channels)


class InverseModel(nn.Module):
    """Transformer inverse model mapping dispersion curves to Vs profiles.

    Default behavior is equivalent to the original implementation:
    input shape ``(batch, 16, 2)`` -> output shape ``(batch, 301, 1)``.
    """

    def __init__(
        self,
        n_periods: int = 16,
        n_channels: int = 2,
        n_depths: int = 301,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_channels: int = 1,
        dim_feedforward: int = 512,
    ) -> None:
        super().__init__()
        self.n_periods = n_periods
        self.n_channels = n_channels
        self.n_depths = n_depths
        self.output_channels = output_channels

        self.embedding = nn.Linear(n_channels, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * n_periods, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_depths * output_channels),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        x = self.embedding(src)
        x = self.encoder(x)
        return self.decoder(x).view(-1, self.n_depths, self.output_channels)
