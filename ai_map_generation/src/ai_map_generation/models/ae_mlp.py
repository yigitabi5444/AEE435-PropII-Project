from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn


def _activation_layer(name: str) -> nn.Module:
    name_lower = name.lower()
    if name_lower == "relu":
        return nn.ReLU()
    if name_lower == "tanh":
        return nn.Tanh()
    if name_lower == "gelu":
        return nn.GELU()
    if name_lower in {"leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU(0.1)
    raise ValueError(f"Unsupported activation '{name}'.")


def build_mlp(input_dim: int, hidden_layers: Iterable[int], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(_activation_layer(activation))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class MLPAutoencoder(nn.Module):
    def __init__(
        self, input_shape: tuple[int, int, int], latent_dim: int, hidden_layers: Iterable[int], activation: str
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        input_dim = int(np.prod(input_shape))
        hidden_layers = list(hidden_layers)

        self.encoder = build_mlp(input_dim, hidden_layers, latent_dim, activation)
        self.decoder = build_mlp(latent_dim, list(reversed(hidden_layers)), input_dim, activation)

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size = tensor.shape[0]
        return self.encoder(tensor.view(batch_size, -1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(latent)
        return decoded.view(latent.shape[0], *self.input_shape)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        latent = self.encode(tensor)
        return self.decode(latent)
