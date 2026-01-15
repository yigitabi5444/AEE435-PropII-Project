from __future__ import annotations

import torch


def point_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum((predicted - target) ** 2)


def latent_l2(latent: torch.Tensor, weight: float) -> torch.Tensor:
    return weight * torch.sum(latent**2)
