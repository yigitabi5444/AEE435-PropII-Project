from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class GasTurbTable:
    arguments: np.ndarray
    parameters: np.ndarray
    values: np.ndarray


@dataclass
class CompressorMap:
    name: str
    nc: np.ndarray
    beta: np.ndarray
    mdot: np.ndarray
    eta: np.ndarray
    pi: np.ndarray
    surge_mdot: np.ndarray | None
    surge_pi: np.ndarray | None


@dataclass
class TurbineMap:
    name: str
    nc: np.ndarray
    beta: np.ndarray
    mdot: np.ndarray
    eta: np.ndarray
    pi_min_nc: np.ndarray
    pi_min: np.ndarray
    pi_max_nc: np.ndarray
    pi_max: np.ndarray


@dataclass
class MapGrid:
    axis0: np.ndarray
    axis1: np.ndarray


def _decode_key(key: float) -> tuple[int, int]:
    rows = int(key)
    columns = int(round((key - rows) * 1000))
    if columns <= 0:
        raise ValueError(f"Invalid table key '{key}'.")
    return rows, columns


def _read_floats(lines: list[str], start_index: int, count: int) -> tuple[list[float], int]:
    values: list[float] = []
    index = start_index
    while index < len(lines) and len(values) < count:
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        tokens = line.replace(",", " ").split()
        for token in tokens:
            if len(values) >= count:
                break
            values.append(float(token))
    if len(values) < count:
        raise ValueError("Unexpected end of table while reading numeric values.")
    return values, index


def _parse_table(lines: list[str], start_index: int) -> tuple[GasTurbTable, int]:
    index = start_index
    key = None
    arguments: list[float] = []

    while index < len(lines) and key is None:
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        tokens = line.replace(",", " ").split()
        if not tokens:
            continue
        try:
            key = float(tokens[0])
            arguments = [float(token) for token in tokens[1:]]
        except ValueError:
            key = None
            arguments = []
            continue

    if key is None:
        raise ValueError("Missing table key.")

    rows, columns = _decode_key(key)

    if len(arguments) < columns - 1:
        remaining = (columns - 1) - len(arguments)
        more_args, index = _read_floats(lines, index, remaining)
        arguments.extend(more_args)
    else:
        arguments = arguments[: columns - 1]

    parameters: list[float] = []
    data_rows: list[list[float]] = []

    for _ in range(rows - 1):
        row_values, index = _read_floats(lines, index, columns)
        parameters.append(row_values[0])
        data_rows.append(row_values[1:])

    table = GasTurbTable(
        arguments=np.asarray(arguments, dtype=np.float32),
        parameters=np.asarray(parameters, dtype=np.float32),
        values=np.asarray(data_rows, dtype=np.float32),
    )
    return table, index


def _load_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def parse_compressor_map(path: str | Path) -> CompressorMap:
    file_path = Path(path)
    lines = _load_lines(file_path)

    mass_table = None
    eff_table = None
    pr_table = None
    surge_table = None

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        lower = line.lower()
        if lower.startswith("mass flow"):
            mass_table, index = _parse_table(lines, index + 1)
            continue
        if lower.startswith("efficiency"):
            eff_table, index = _parse_table(lines, index + 1)
            continue
        if lower.startswith("pressure ratio"):
            pr_table, index = _parse_table(lines, index + 1)
            continue
        if lower.startswith("surge line"):
            surge_table, index = _parse_table(lines, index + 1)
            continue
        index += 1

    if mass_table is None or eff_table is None or pr_table is None:
        raise ValueError(f"Missing required tables in compressor map {file_path.name}.")

    surge_mdot = None
    surge_pi = None
    if surge_table is not None and surge_table.values.size > 0:
        surge_mdot = surge_table.arguments
        surge_pi = surge_table.values[0]

    return CompressorMap(
        name=file_path.stem,
        nc=mass_table.parameters,
        beta=mass_table.arguments,
        mdot=mass_table.values,
        eta=eff_table.values,
        pi=pr_table.values,
        surge_mdot=surge_mdot,
        surge_pi=surge_pi,
    )


def _is_pressure_min(line: str) -> bool:
    lower = line.lower()
    return "min pressure" in lower or "beta=0" in lower or "beta = 0" in lower


def _is_pressure_max(line: str) -> bool:
    lower = line.lower()
    return "max pressure" in lower or "beta=1" in lower or "beta = 1" in lower


def parse_turbine_map(path: str | Path) -> TurbineMap:
    file_path = Path(path)
    lines = _load_lines(file_path)

    min_table = None
    max_table = None
    mass_table = None
    eff_table = None

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        lower = line.lower()
        if _is_pressure_min(line):
            min_table, index = _parse_table(lines, index + 1)
            continue
        if _is_pressure_max(line):
            max_table, index = _parse_table(lines, index + 1)
            continue
        if lower.startswith("mass flow"):
            mass_table, index = _parse_table(lines, index + 1)
            continue
        if lower.startswith("efficiency"):
            eff_table, index = _parse_table(lines, index + 1)
            continue
        index += 1

    if min_table is None or max_table is None or mass_table is None or eff_table is None:
        raise ValueError(f"Missing required tables in turbine map {file_path.name}.")

    pi_min = min_table.values[0]
    pi_max = max_table.values[0]

    return TurbineMap(
        name=file_path.stem,
        nc=mass_table.parameters,
        beta=mass_table.arguments,
        mdot=mass_table.values,
        eta=eff_table.values,
        pi_min_nc=min_table.arguments,
        pi_min=pi_min,
        pi_max_nc=max_table.arguments,
        pi_max=pi_max,
    )


def _interp_row(x_query: float, x_row: np.ndarray, y_row: np.ndarray) -> float:
    order = np.argsort(x_row)
    x_sorted = x_row[order]
    y_sorted = y_row[order]
    return float(np.interp(x_query, x_sorted, y_sorted))


def _interpolate_surface(
    nc_values: np.ndarray,
    x_table: np.ndarray,
    y_table: np.ndarray,
    grid_nc: np.ndarray,
    grid_x: np.ndarray,
) -> np.ndarray:
    row_values = np.full((nc_values.size, grid_x.size), np.nan, dtype=np.float32)

    for row_index, nc_value in enumerate(nc_values):
        x_row = x_table[row_index]
        y_row = y_table[row_index]
        for col_index, x_query in enumerate(grid_x):
            row_values[row_index, col_index] = _interp_row(x_query, x_row, y_row)

    grid = np.full((grid_nc.size, grid_x.size), np.nan, dtype=np.float32)
    for col_index in range(grid_x.size):
        column = row_values[:, col_index]
        valid = ~np.isnan(column)
        if np.count_nonzero(valid) < 2:
            continue
        grid[:, col_index] = np.interp(grid_nc, nc_values[valid], column[valid])

    return grid


def compressor_to_tensor(map_data: CompressorMap, grid: MapGrid) -> np.ndarray:
    nc_values = map_data.nc
    mdot_values = map_data.mdot

    eta_grid = _interpolate_surface(nc_values, mdot_values, map_data.eta, grid.axis0, grid.axis1)
    pi_grid = _interpolate_surface(nc_values, mdot_values, map_data.pi, grid.axis0, grid.axis1)

    mdot_min = np.nanmin(mdot_values, axis=1)
    mdot_min_grid = np.interp(grid.axis0, nc_values, mdot_min)

    for row_index, min_mdot in enumerate(mdot_min_grid):
        mask = grid.axis1 < min_mdot
        eta_grid[row_index, mask] = 0.0
        pi_grid[row_index, mask] = 0.0

    eta_grid = np.nan_to_num(eta_grid, nan=0.0)
    pi_grid = np.nan_to_num(pi_grid, nan=0.0)

    eta_grid = np.clip(eta_grid, 0.0, 1.0)
    pi_grid = np.clip(pi_grid, 0.0, None)

    return np.stack([eta_grid, pi_grid]).astype(np.float32)


def turbine_to_tensor(map_data: TurbineMap, grid: MapGrid) -> np.ndarray:
    nc_values = map_data.nc
    beta = map_data.beta

    pi_min = np.interp(nc_values, map_data.pi_min_nc, map_data.pi_min)
    pi_max = np.interp(nc_values, map_data.pi_max_nc, map_data.pi_max)
    pi_table = pi_min[:, None] + beta[None, :] * (pi_max - pi_min)[:, None]

    eta_grid = _interpolate_surface(nc_values, pi_table, map_data.eta, grid.axis0, grid.axis1)
    mdot_grid = _interpolate_surface(nc_values, pi_table, map_data.mdot, grid.axis0, grid.axis1)

    pi_min_grid = np.interp(grid.axis0, nc_values, pi_min)
    pi_max_grid = np.interp(grid.axis0, nc_values, pi_max)

    for row_index in range(grid.axis0.size):
        mask = (grid.axis1 < pi_min_grid[row_index]) | (grid.axis1 > pi_max_grid[row_index])
        eta_grid[row_index, mask] = 0.0
        mdot_grid[row_index, mask] = 0.0

    eta_grid = np.nan_to_num(eta_grid, nan=0.0)
    mdot_grid = np.nan_to_num(mdot_grid, nan=0.0)

    eta_grid = np.clip(eta_grid, 0.0, 1.0)
    mdot_grid = np.clip(mdot_grid, 0.0, None)

    return np.stack([eta_grid, mdot_grid]).astype(np.float32)


def compute_grid(axis0_min: float, axis0_max: float, axis1_min: float, axis1_max: float, size: int) -> MapGrid:
    axis0 = np.linspace(axis0_min, axis0_max, size, dtype=np.float32)
    axis1 = np.linspace(axis1_min, axis1_max, size, dtype=np.float32)
    return MapGrid(axis0=axis0, axis1=axis1)


def _normalize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value == min_value:
        return np.zeros_like(values, dtype=np.float32), min_value, max_value
    scaled = (values - min_value) / (max_value - min_value)
    return scaled.astype(np.float32), min_value, max_value


def normalize_compressor(map_data: CompressorMap) -> tuple[CompressorMap, dict]:
    nc_scaled, nc_min, nc_max = _normalize(map_data.nc)
    mdot_scaled, mdot_min, mdot_max = _normalize(map_data.mdot)
    surge_mdot = None
    if map_data.surge_mdot is not None:
        surge_mdot = (map_data.surge_mdot - mdot_min) / (mdot_max - mdot_min)
    normalized = CompressorMap(
        name=map_data.name,
        nc=nc_scaled,
        beta=map_data.beta,
        mdot=mdot_scaled,
        eta=map_data.eta,
        pi=map_data.pi,
        surge_mdot=surge_mdot,
        surge_pi=map_data.surge_pi,
    )
    meta = {
        "axis0_raw_min": nc_min,
        "axis0_raw_max": nc_max,
        "axis1_raw_min": mdot_min,
        "axis1_raw_max": mdot_max,
    }
    return normalized, meta


def normalize_turbine(map_data: TurbineMap) -> tuple[TurbineMap, dict]:
    nc_scaled, nc_min, nc_max = _normalize(map_data.nc)
    if nc_max == nc_min:
        pi_min_nc_scaled = np.zeros_like(map_data.pi_min_nc, dtype=np.float32)
        pi_max_nc_scaled = np.zeros_like(map_data.pi_max_nc, dtype=np.float32)
    else:
        pi_min_nc_scaled = ((map_data.pi_min_nc - nc_min) / (nc_max - nc_min)).astype(np.float32)
        pi_max_nc_scaled = ((map_data.pi_max_nc - nc_min) / (nc_max - nc_min)).astype(np.float32)

    pi_min_raw = float(np.min(map_data.pi_min))
    pi_max_raw = float(np.max(map_data.pi_max))
    if pi_max_raw == pi_min_raw:
        pi_min_scaled = np.zeros_like(map_data.pi_min, dtype=np.float32)
        pi_max_scaled = np.zeros_like(map_data.pi_max, dtype=np.float32)
    else:
        pi_min_scaled = ((map_data.pi_min - pi_min_raw) / (pi_max_raw - pi_min_raw)).astype(np.float32)
        pi_max_scaled = ((map_data.pi_max - pi_min_raw) / (pi_max_raw - pi_min_raw)).astype(np.float32)
    normalized = TurbineMap(
        name=map_data.name,
        nc=nc_scaled,
        beta=map_data.beta,
        mdot=map_data.mdot,
        eta=map_data.eta,
        pi_min_nc=pi_min_nc_scaled,
        pi_min=pi_min_scaled * 5.0,
        pi_max_nc=pi_max_nc_scaled,
        pi_max=pi_max_scaled * 5.0,
    )
    meta = {
        "axis0_raw_min": nc_min,
        "axis0_raw_max": nc_max,
        "axis1_raw_min": pi_min_raw,
        "axis1_raw_max": pi_max_raw,
    }
    return normalized, meta


def perturb_tensor(tensor: np.ndarray, rng: np.random.Generator, noise_scale: float) -> np.ndarray:
    noisy = tensor.copy()
    mask = noisy[0] <= 0.0

    scale = rng.normal(1.0, 0.03, size=(noisy.shape[0], 1, 1)).astype(np.float32)
    noisy *= scale

    for channel in range(noisy.shape[0]):
        channel_data = noisy[channel]
        amplitude = float(np.nanmax(channel_data) - np.nanmin(channel_data))
        sigma = noise_scale * (amplitude if amplitude > 0 else 1.0)
        channel_data += rng.normal(0.0, sigma, size=channel_data.shape)
        noisy[channel] = channel_data

    noisy[0] = np.clip(noisy[0], 0.0, 1.0)
    noisy[1] = np.clip(noisy[1], 0.0, None)
    noisy[:, mask] = 0.0

    return noisy.astype(np.float32)
