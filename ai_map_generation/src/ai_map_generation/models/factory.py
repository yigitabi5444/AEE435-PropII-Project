from __future__ import annotations

from typing import Iterable

from .ae_mlp import MLPAutoencoder


def build_model_arch(hidden_layers: Iterable[int], activation: str) -> dict:
    return {
        "type": "mlp_ae",
        "encoder_layers": list(hidden_layers),
        "decoder_layers": list(reversed(list(hidden_layers))),
        "activation": activation,
    }


def build_autoencoder(
    input_shape: tuple[int, int, int], latent_dim: int, model_arch: dict
) -> MLPAutoencoder:
    model_type = model_arch.get("type", "mlp_ae")
    if model_type != "mlp_ae":
        raise ValueError(f"Unsupported model type '{model_type}'.")

    hidden_layers = model_arch.get("encoder_layers", [])
    activation = model_arch.get("activation", "relu")
    return MLPAutoencoder(input_shape=input_shape, latent_dim=latent_dim, hidden_layers=hidden_layers, activation=activation)
