from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

from .config import get_map_config

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - handled at runtime
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_ERROR = exc
else:
    _MATPLOTLIB_ERROR = None

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _require_matplotlib():
    if plt is None:
        message = (
            "Matplotlib failed to import. Install/upgrade matplotlib to a build "
            "compatible with NumPy 2 (e.g. matplotlib>=3.9)."
        )
        raise RuntimeError(message) from _MATPLOTLIB_ERROR
    return plt


def save_loss_curves(losses: Sequence[float], png_path: str | Path, csv_path: str | Path) -> None:
    plot_module = _require_matplotlib()
    loss_array = np.asarray(losses, dtype=float)
    figure, axis = plot_module.subplots(figsize=(6, 4))
    axis.plot(np.arange(1, loss_array.size + 1), loss_array, color="tab:blue")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training Loss")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(png_path, dpi=150)
    plot_module.close(figure)

    dataframe = pd.DataFrame({"epoch": np.arange(1, loss_array.size + 1), "loss": loss_array})
    dataframe.to_csv(csv_path, index=False)


def build_sample_figure(
    tensor: np.ndarray, axis0: np.ndarray, axis1: np.ndarray, map_type: str
) -> "Figure":
    plot_module = _require_matplotlib()
    config = get_map_config(map_type)
    figure, axes = plot_module.subplots(1, 2, figsize=(8, 3.5))
    extent = [axis1.min(), axis1.max(), axis0.min(), axis0.max()]

    for channel_index, axis in enumerate(axes):
        image = axis.imshow(
            tensor[channel_index],
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
        )
        axis.set_title(config.channels[channel_index])
        axis.set_xlabel(config.axis1_name)
        axis.set_ylabel(config.axis0_name)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.tight_layout()
    return figure
