from __future__ import annotations

import numpy as np

from .config import get_map_config


DEFAULT_GRID_SPECS = {
    "Nc": (0.0, 1.0, 20),
    "mdotc": (0.0, 1.0, 20),
    "pi_t": (0.0, 5.0, 20),
}


def make_default_axis(axis_name: str) -> np.ndarray:
    if axis_name not in DEFAULT_GRID_SPECS:
        raise ValueError(f"No default grid spec for axis '{axis_name}'.")
    start, end, length = DEFAULT_GRID_SPECS[axis_name]
    return np.linspace(start, end, length, dtype=np.float32)


def make_default_axes(map_type: str) -> tuple[np.ndarray, np.ndarray]:
    config = get_map_config(map_type)
    axis0 = make_default_axis(config.axis0_name)
    axis1 = make_default_axis(config.axis1_name)
    return axis0, axis1


def validate_axis(axis: np.ndarray, name: str) -> None:
    if axis.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if axis.size < 2:
        raise ValueError(f"{name} must have at least two points.")
    if not np.all(np.diff(axis) > 0):
        raise ValueError(f"{name} must be strictly increasing.")


def validate_grid(axis0: np.ndarray, axis1: np.ndarray) -> None:
    validate_axis(axis0, "axis0")
    validate_axis(axis1, "axis1")


def axis_metadata(
    map_type: str,
    axis0: np.ndarray,
    axis1: np.ndarray,
    raw_ranges: dict | None = None,
) -> dict:
    config = get_map_config(map_type)
    metadata = {
        "axis0_name": config.axis0_name,
        "axis0": axis0.astype(float).tolist(),
        "axis1_name": config.axis1_name,
        "axis1": axis1.astype(float).tolist(),
        "channels": list(config.channels),
    }
    if raw_ranges:
        metadata.update(raw_ranges)
    return metadata
