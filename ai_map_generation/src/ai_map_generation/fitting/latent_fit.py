from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from ..common.interpolation import build_bilinear_sampler
from ..common.utils import ensure_float32
from .losses import latent_l2, point_loss


@dataclass
class FitConfig:
    init_strategy: str
    iterations: int
    learning_rate: float
    latent_l2_weight: float
    optimizer: str
    tolerance: float


@dataclass
class FitResult:
    latent_z: np.ndarray
    loss_history: list[float]
    final_loss: float
    reconstructed: np.ndarray
    predicted: np.ndarray
    targets: np.ndarray
    inputs: np.ndarray


def _init_latent(latent_dim: int, strategy: str) -> torch.Tensor:
    if strategy == "zeros":
        return torch.zeros(1, latent_dim)
    if strategy == "random":
        return torch.randn(1, latent_dim) * 0.1
    raise ValueError(f"Unknown init strategy '{strategy}'.")


def fit_latent(
    decoder: torch.nn.Module,
    latent_dim: int,
    output_shape: tuple[int, int, int],
    axis0: np.ndarray,
    axis1: np.ndarray,
    points: np.ndarray,
    targets: np.ndarray,
    config: FitConfig,
    device: torch.device,
    progress_callback: Callable[[int, float], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> FitResult:
    axis0 = ensure_float32(axis0)
    axis1 = ensure_float32(axis1)
    points = ensure_float32(points)
    targets = ensure_float32(targets)
    if points.shape[0] != targets.shape[0]:
        raise ValueError("points and targets must have the same length.")

    sampler = build_bilinear_sampler(axis0, axis1, points).to(device)
    target_tensor = torch.from_numpy(targets).to(device)

    latent = _init_latent(latent_dim=latent_dim, strategy=config.init_strategy).to(device)
    latent.requires_grad_(True)

    def decode_latent(current_latent: torch.Tensor) -> torch.Tensor:
        decoded_flat = decoder(current_latent)
        return decoded_flat.view(current_latent.shape[0], *output_shape)

    loss_history: list[float] = []

    if config.optimizer.lower() == "lbfgs":
        optimizer = torch.optim.LBFGS([latent], lr=config.learning_rate, max_iter=1)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            decoded = decode_latent(latent)
            decoded = decoded[0]
            predicted = torch.stack(
                [sampler.sample(decoded[0]), sampler.sample(decoded[1])], dim=1
            )
            loss_value = point_loss(predicted, target_tensor) + latent_l2(latent, config.latent_l2_weight)
            loss_value.backward()
            return loss_value

        for iteration in range(config.iterations):
            if stop_requested and stop_requested():
                break
            loss_value = optimizer.step(closure)
            loss_scalar = float(loss_value.item())
            loss_history.append(loss_scalar)
            if progress_callback:
                progress_callback(iteration + 1, loss_scalar)
            if iteration > 0 and abs(loss_history[-2] - loss_scalar) < config.tolerance:
                break
    else:
        optimizer = torch.optim.Adam([latent], lr=config.learning_rate)
        for iteration in range(config.iterations):
            if stop_requested and stop_requested():
                break
            optimizer.zero_grad(set_to_none=True)
            decoded = decode_latent(latent)[0]
            predicted = torch.stack([sampler.sample(decoded[0]), sampler.sample(decoded[1])], dim=1)
            loss_value = point_loss(predicted, target_tensor) + latent_l2(latent, config.latent_l2_weight)
            loss_value.backward()
            optimizer.step()
            loss_scalar = float(loss_value.item())
            loss_history.append(loss_scalar)
            if progress_callback:
                progress_callback(iteration + 1, loss_scalar)
            if iteration > 0 and abs(loss_history[-2] - loss_scalar) < config.tolerance:
                break

    with torch.no_grad():
        decoded = decode_latent(latent)[0]
        predicted = torch.stack([sampler.sample(decoded[0]), sampler.sample(decoded[1])], dim=1)

    return FitResult(
        latent_z=latent.detach().cpu().numpy().squeeze(),
        loss_history=loss_history,
        final_loss=float(loss_history[-1]) if loss_history else float("nan"),
        reconstructed=decoded.detach().cpu().numpy(),
        predicted=predicted.detach().cpu().numpy(),
        targets=targets,
        inputs=points,
    )
