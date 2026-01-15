from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import get_map_config, normalize_map_type
from .grids import validate_grid


@dataclass
class MapSample:
    tensor: np.ndarray
    map_type: str
    axis0: np.ndarray
    axis1: np.ndarray
    sample_id: str | None


@dataclass
class OperatingPoints:
    inputs: np.ndarray
    targets: np.ndarray
    columns: list[str]


def load_map_npz(path: str | Path, expected_map_type: str | None = None) -> MapSample:
    data = np.load(path, allow_pickle=True)
    if "X" not in data:
        raise ValueError(f"{path} missing 'X' array.")
    tensor = np.asarray(data["X"], dtype=np.float32)
    expected = normalize_map_type(expected_map_type) if expected_map_type else None
    actual = None
    if "map_type" in data:
        actual = normalize_map_type(str(data["map_type"]))
    if expected and actual and expected != actual:
        raise ValueError(f"{path} map_type '{actual}' does not match expected '{expected}'.")
    map_type = actual or expected
    if not map_type:
        raise ValueError(f"{path} missing map_type.")

    axis0 = data.get("axis0")
    axis1 = data.get("axis1")
    if axis0 is None or axis1 is None:
        raise ValueError(f"{path} missing axis0/axis1 arrays.")

    axis0 = np.asarray(axis0, dtype=np.float32)
    axis1 = np.asarray(axis1, dtype=np.float32)
    validate_grid(axis0, axis1)

    if tensor.ndim != 3 or tensor.shape[0] != 2:
        raise ValueError(f"{path} has invalid X shape {tensor.shape}.")
    if tensor.shape[1] != axis0.size or tensor.shape[2] != axis1.size:
        raise ValueError(
            f"{path} X shape {tensor.shape} does not match grid sizes "
            f"({axis0.size}, {axis1.size})."
        )

    sample_id = str(data["id"]) if "id" in data else None
    return MapSample(tensor=tensor, map_type=map_type, axis0=axis0, axis1=axis1, sample_id=sample_id)


def load_points_csv(path: str | Path, map_type: str) -> OperatingPoints:
    dataframe = pd.read_csv(path)

    if map_type == "compressor":
        required_columns = ["Nc", "mdotc", "eta", "pi"]
    else:
        required_columns = ["Nc", "pi_t", "eta", "mdotc"]

    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {', '.join(missing)}")

    input_values = dataframe[required_columns[:2]].to_numpy(dtype=np.float32)
    target_values = dataframe[required_columns[2:]].to_numpy(dtype=np.float32)

    return OperatingPoints(inputs=input_values, targets=target_values, columns=required_columns)


def load_points_json(path: str | Path, map_type: str) -> OperatingPoints:
    with open(path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if map_type == "compressor":
        required_columns = ["Nc", "mdotc", "eta", "pi"]
    else:
        required_columns = ["Nc", "pi_t", "eta", "mdotc"]

    rows = []
    for entry in payload:
        row = [entry.get(column) for column in required_columns]
        if any(value is None for value in row):
            raise ValueError("JSON entries must include all required columns.")
        rows.append(row)

    values = np.asarray(rows, dtype=np.float32)
    return OperatingPoints(inputs=values[:, :2], targets=values[:, 2:], columns=required_columns)


def save_created_map_npz(
    path: str | Path,
    tensor: np.ndarray,
    axis0: np.ndarray,
    axis1: np.ndarray,
    channels: Iterable[str],
    map_type: str,
    latent_z: np.ndarray,
    fit_meta: dict,
) -> None:
    np.savez(
        path,
        X_hat=np.asarray(tensor, dtype=np.float32),
        axis0=np.asarray(axis0, dtype=np.float32),
        axis1=np.asarray(axis1, dtype=np.float32),
        channels=np.array(list(channels)),
        map_type=str(map_type),
        latent_z=np.asarray(latent_z, dtype=np.float32),
        fit_meta=json.dumps(fit_meta),
    )


def save_created_map_csv(
    path: str | Path,
    tensor: np.ndarray,
    axis0: np.ndarray,
    axis1: np.ndarray,
    map_type: str,
) -> None:
    config = get_map_config(map_type)
    axis0_grid, axis1_grid = np.meshgrid(axis0, axis1, indexing="ij")
    dataframe = pd.DataFrame(
        {
            config.axis0_name: axis0_grid.reshape(-1),
            config.axis1_name: axis1_grid.reshape(-1),
            config.channels[0]: tensor[0].reshape(-1),
            config.channels[1]: tensor[1].reshape(-1),
        }
    )
    dataframe.to_csv(path, index=False)


def save_latent_npz(
    path: str | Path,
    latent_z: np.ndarray,
    map_type: str,
    latent_dim: int,
    model_hash: str,
    fit_meta: dict,
) -> None:
    np.savez(
        path,
        z=np.asarray(latent_z, dtype=np.float32),
        map_type=str(map_type),
        latent_dim=int(latent_dim),
        model_hash=str(model_hash),
        fit_meta=json.dumps(fit_meta),
    )
