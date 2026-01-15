from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import get_map_config

try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
except Exception as exc:  # pragma: no cover - handled at runtime
    Figure = None  # type: ignore[assignment]
    FigureCanvasAgg = None  # type: ignore[assignment]
    _MATPLOTLIB_ERROR = exc
else:
    _MATPLOTLIB_ERROR = None


def _require_matplotlib() -> tuple[type[Figure], type[FigureCanvasAgg]]:  # type: ignore[name-defined]
    if Figure is None or FigureCanvasAgg is None:
        message = (
            "Matplotlib failed to import. Install/upgrade matplotlib to a build "
            "compatible with NumPy 2 (e.g. matplotlib>=3.9)."
        )
        raise RuntimeError(message) from _MATPLOTLIB_ERROR
    return Figure, FigureCanvasAgg


def save_loss_curves(losses: Sequence[float], png_path: str | Path, csv_path: str | Path) -> None:
    figure_cls, canvas_cls = _require_matplotlib()
    loss_array = np.asarray(losses, dtype=float)
    figure = figure_cls(figsize=(6, 4))
    canvas_cls(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(np.arange(1, loss_array.size + 1), loss_array, color="tab:blue")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training Loss")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(png_path, dpi=150)

    dataframe = pd.DataFrame({"epoch": np.arange(1, loss_array.size + 1), "loss": loss_array})
    dataframe.to_csv(csv_path, index=False)


def build_sample_figure(
    tensor: np.ndarray, axis0: np.ndarray, axis1: np.ndarray, map_type: str
) -> Figure:  # type: ignore[name-defined]
    figure_cls, _ = _require_matplotlib()
    config = get_map_config(map_type)
    figure = figure_cls(figsize=(8, 3.5))
    axes = figure.subplots(1, 2)
    extent = [axis0.min(), axis0.max(), axis1.min(), axis1.max()]
    axis0_values = np.asarray(axis0)
    axis1_values = np.asarray(axis1)

    for channel_index, axis in enumerate(axes):
        image_data = tensor[channel_index].T
        image = axis.imshow(
            image_data,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
        )
        axis.set_title(config.channels[channel_index])
        axis.set_xlabel(config.axis0_name)
        axis.set_ylabel(config.axis1_name)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label=config.channels[channel_index])

        def format_coord(x: float, y: float, channel: int = channel_index) -> str:
            if x < axis0_values.min() or x > axis0_values.max():
                return f"{config.axis0_name}={x:.3f}, {config.axis1_name}={y:.3f}"
            if y < axis1_values.min() or y > axis1_values.max():
                return f"{config.axis0_name}={x:.3f}, {config.axis1_name}={y:.3f}"
            x_index = int(np.argmin(np.abs(axis0_values - x)))
            y_index = int(np.argmin(np.abs(axis1_values - y)))
            value = tensor[channel, x_index, y_index]
            return (
                f"{config.axis0_name}={x:.3f}, {config.axis1_name}={y:.3f}, "
                f"{config.channels[channel]}={value:.4f}"
            )

        axis.format_coord = format_coord

    figure.tight_layout()
    return figure
