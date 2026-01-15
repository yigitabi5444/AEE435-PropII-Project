from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..common.grids import axis_metadata
from ..models.factory import build_autoencoder


def build_model_artifact(
    model: torch.nn.Module,
    map_type: str,
    latent_dim: int,
    axis0: np.ndarray,
    axis1: np.ndarray,
    model_arch: dict,
    training_meta: dict,
) -> dict:
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    return {
        "format_version": "1.0",
        "map_type": map_type,
        "timestamp_utc": timestamp_utc,
        "framework": "pytorch",
        "latent_dim": int(latent_dim),
        "grid": axis_metadata(map_type, axis0, axis1),
        "model_arch": model_arch,
        "state_dict": {
            "encoder": model.encoder.state_dict(),
            "decoder": model.decoder.state_dict(),
        },
        "training_meta": training_meta,
    }


def save_model_artifact(path: str | Path, artifact: dict) -> None:
    torch.save(artifact, path)


def load_model_artifact(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu")


def build_model_from_artifact(artifact: dict) -> torch.nn.Module:
    map_type = artifact["map_type"]
    grid = artifact["grid"]
    axis0 = np.asarray(grid["axis0"], dtype=np.float32)
    axis1 = np.asarray(grid["axis1"], dtype=np.float32)
    input_shape = (2, axis0.size, axis1.size)
    latent_dim = int(artifact["latent_dim"])
    model_arch = artifact["model_arch"]
    model = build_autoencoder(input_shape=input_shape, latent_dim=latent_dim, model_arch=model_arch)
    model.encoder.load_state_dict(artifact["state_dict"]["encoder"])
    model.decoder.load_state_dict(artifact["state_dict"]["decoder"])
    model.eval()
    return model
