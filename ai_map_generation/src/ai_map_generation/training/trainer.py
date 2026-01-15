from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float


def train_autoencoder(
    model: nn.Module,
    dataloader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    progress_callback: Callable[[int, float], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> list[float]:
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    losses: list[float] = []

    for epoch_index in range(config.epochs):
        if stop_requested and stop_requested():
            break
        running_loss = 0.0
        batch_count = 0

        for batch in dataloader:
            if stop_requested and stop_requested():
                break
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batch_count += 1

        if batch_count == 0:
            break
        average_loss = running_loss / batch_count
        losses.append(average_loss)
        if progress_callback:
            progress_callback(epoch_index + 1, average_loss)

    return losses
