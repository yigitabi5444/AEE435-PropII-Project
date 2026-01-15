from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass
class BilinearSampler:
    axis0_lower: torch.Tensor
    axis0_upper: torch.Tensor
    axis1_lower: torch.Tensor
    axis1_upper: torch.Tensor
    weight_00: torch.Tensor
    weight_01: torch.Tensor
    weight_10: torch.Tensor
    weight_11: torch.Tensor

    def to(self, device: torch.device) -> "BilinearSampler":
        return BilinearSampler(
            axis0_lower=self.axis0_lower.to(device),
            axis0_upper=self.axis0_upper.to(device),
            axis1_lower=self.axis1_lower.to(device),
            axis1_upper=self.axis1_upper.to(device),
            weight_00=self.weight_00.to(device),
            weight_01=self.weight_01.to(device),
            weight_10=self.weight_10.to(device),
            weight_11=self.weight_11.to(device),
        )

    def sample(self, grid: torch.Tensor) -> torch.Tensor:
        value_00 = grid[self.axis0_lower, self.axis1_lower]
        value_01 = grid[self.axis0_lower, self.axis1_upper]
        value_10 = grid[self.axis0_upper, self.axis1_lower]
        value_11 = grid[self.axis0_upper, self.axis1_upper]
        return (
            self.weight_00 * value_00
            + self.weight_01 * value_01
            + self.weight_10 * value_10
            + self.weight_11 * value_11
        )


def _validate_points_in_bounds(axis0: np.ndarray, axis1: np.ndarray, points: np.ndarray) -> None:
    axis0_min, axis0_max = axis0.min(), axis0.max()
    axis1_min, axis1_max = axis1.min(), axis1.max()
    for point in points:
        axis0_value, axis1_value = float(point[0]), float(point[1])
        if axis0_value < axis0_min or axis0_value > axis0_max:
            raise ValueError(
                f"Point axis0={axis0_value} outside grid range [{axis0_min}, {axis0_max}]."
            )
        if axis1_value < axis1_min or axis1_value > axis1_max:
            raise ValueError(
                f"Point axis1={axis1_value} outside grid range [{axis1_min}, {axis1_max}]."
            )


def build_bilinear_sampler(
    axis0: np.ndarray, axis1: np.ndarray, points: Iterable[Iterable[float]]
) -> BilinearSampler:
    point_array = np.asarray(list(points), dtype=np.float32)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")
    _validate_points_in_bounds(axis0, axis1, point_array)

    axis0_indices = np.searchsorted(axis0, point_array[:, 0], side="right") - 1
    axis1_indices = np.searchsorted(axis1, point_array[:, 1], side="right") - 1
    axis0_indices = np.clip(axis0_indices, 0, axis0.size - 2)
    axis1_indices = np.clip(axis1_indices, 0, axis1.size - 2)

    axis0_lower = axis0[axis0_indices]
    axis0_upper = axis0[axis0_indices + 1]
    axis1_lower = axis1[axis1_indices]
    axis1_upper = axis1[axis1_indices + 1]

    axis0_span = axis0_upper - axis0_lower
    axis1_span = axis1_upper - axis1_lower

    axis0_fraction = (point_array[:, 0] - axis0_lower) / axis0_span
    axis1_fraction = (point_array[:, 1] - axis1_lower) / axis1_span

    weight_00 = (1.0 - axis0_fraction) * (1.0 - axis1_fraction)
    weight_01 = (1.0 - axis0_fraction) * axis1_fraction
    weight_10 = axis0_fraction * (1.0 - axis1_fraction)
    weight_11 = axis0_fraction * axis1_fraction

    return BilinearSampler(
        axis0_lower=torch.from_numpy(axis0_indices.astype(np.int64)),
        axis0_upper=torch.from_numpy((axis0_indices + 1).astype(np.int64)),
        axis1_lower=torch.from_numpy(axis1_indices.astype(np.int64)),
        axis1_upper=torch.from_numpy((axis1_indices + 1).astype(np.int64)),
        weight_00=torch.from_numpy(weight_00.astype(np.float32)),
        weight_01=torch.from_numpy(weight_01.astype(np.float32)),
        weight_10=torch.from_numpy(weight_10.astype(np.float32)),
        weight_11=torch.from_numpy(weight_11.astype(np.float32)),
    )
