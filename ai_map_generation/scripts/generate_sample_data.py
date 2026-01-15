from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_map_generation.common.config import get_map_config
from ai_map_generation.common.grids import make_default_axes


def generate_compressor_map(axis0: np.ndarray, axis1: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    nc_grid, mdot_grid = np.meshgrid(axis0, axis1, indexing="ij")

    nc_center = rng.uniform(0.55, 0.85)
    mdot_center = rng.uniform(0.45, 0.8)
    nc_sigma = rng.uniform(0.15, 0.3)
    mdot_sigma = rng.uniform(0.12, 0.25)

    eta_peak = rng.uniform(0.78, 0.92)
    eta_base = rng.uniform(0.15, 0.3)
    eta = eta_base + (eta_peak - eta_base) * np.exp(
        -(((nc_grid - nc_center) / nc_sigma) ** 2 + ((mdot_grid - mdot_center) / mdot_sigma) ** 2)
    )
    eta += 0.02 * np.sin(4 * np.pi * nc_grid) * np.cos(3 * np.pi * mdot_grid)
    eta = np.clip(eta, 0.0, 1.0)

    pi_gain = rng.uniform(1.0, 2.5)
    pi_base = rng.uniform(1.05, 1.4)
    pi = pi_base + pi_gain * np.exp(
        -(((nc_grid - nc_center) / (nc_sigma * 1.3)) ** 2 + ((mdot_grid - mdot_center) / (mdot_sigma * 1.2)) ** 2)
    )
    pi += 0.2 * nc_grid - 0.15 * mdot_grid
    pi = np.clip(pi, 0.0, None)

    surge_offset = rng.uniform(-0.05, 0.05)
    surge_line = 0.12 + 0.55 * nc_grid + surge_offset
    mask = mdot_grid < surge_line
    eta[mask] = 0.0
    pi[mask] = 0.0

    return np.stack([eta, pi]).astype(np.float32)


def generate_turbine_map(axis0: np.ndarray, axis1: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    nc_grid, pi_grid = np.meshgrid(axis0, axis1, indexing="ij")

    nc_center = rng.uniform(0.4, 0.7)
    pi_center = rng.uniform(1.0, 3.5)
    nc_sigma = rng.uniform(0.2, 0.35)
    pi_sigma = rng.uniform(0.9, 1.6)

    eta_peak = rng.uniform(0.82, 0.95)
    eta_base = rng.uniform(0.35, 0.5)
    eta = eta_base + (eta_peak - eta_base) * np.exp(
        -(((nc_grid - nc_center) / nc_sigma) ** 2 + ((pi_grid - pi_center) / pi_sigma) ** 2)
    )
    eta += 0.015 * np.sin(2 * np.pi * nc_grid) * np.cos(0.6 * np.pi * pi_grid)
    eta = np.clip(eta, 0.0, 1.0)

    mdot_base = rng.uniform(0.2, 0.35)
    mdot_gain = rng.uniform(0.6, 0.9)
    mdot = mdot_base + mdot_gain * (0.6 + 0.4 * nc_grid) * (1.0 - np.exp(-pi_grid / 2.5))
    mdot += rng.normal(0.0, 0.01, size=mdot.shape)
    mdot = np.clip(mdot, 0.0, None)

    return np.stack([eta, mdot]).astype(np.float32)


def save_sample(
    output_dir: Path,
    map_type: str,
    index: int,
    tensor: np.ndarray,
    axis0: np.ndarray,
    axis1: np.ndarray,
) -> None:
    sample_id = f"{map_type}_sample_{index:04d}"
    path = output_dir / f"sample_{index:04d}.npz"
    np.savez(
        path,
        X=tensor.astype(np.float32),
        map_type=map_type,
        id=sample_id,
        axis0=axis0.astype(np.float32),
        axis1=axis1.astype(np.float32),
    )


def select_point_indices(eta_grid: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    candidates = np.argwhere(eta_grid > 0.05)
    if candidates.shape[0] < count:
        candidates = np.argwhere(~np.isnan(eta_grid))
    rng.shuffle(candidates)
    return candidates[:count]


def save_point_sets(
    output_dir: Path,
    map_type: str,
    tensor: np.ndarray,
    axis0: np.ndarray,
    axis1: np.ndarray,
    rng: np.random.Generator,
) -> None:
    max_points = 20
    indices = select_point_indices(tensor[0], max_points, rng)
    if indices.shape[0] < max_points:
        raise RuntimeError("Not enough valid points to sample.")

    config = get_map_config(map_type)
    input_axis0 = axis0[indices[:, 0]]
    input_axis1 = axis1[indices[:, 1]]
    eta_values = tensor[0, indices[:, 0], indices[:, 1]]
    second_values = tensor[1, indices[:, 0], indices[:, 1]]

    if map_type == "compressor":
        columns = ["Nc", "mdotc", "eta", "pi"]
        data = np.column_stack([input_axis0, input_axis1, eta_values, second_values])
    else:
        columns = ["Nc", "pi_t", "eta", "mdotc"]
        data = np.column_stack([input_axis0, input_axis1, eta_values, second_values])

    for count in (20, 6, 3):
        subset = data[:count]
        dataframe = pd.DataFrame(subset, columns=columns)
        dataframe.to_csv(output_dir / f"test_points_{count:02d}.csv", index=False)


def generate_dataset(map_type: str, output_dir: Path, num_samples: int, seed: int, overwrite: bool) -> None:
    axis0, axis1 = make_default_axes(map_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(output_dir.glob("*.npz")):
        raise FileExistsError(f"Output directory {output_dir} already contains .npz files.")

    rng = np.random.default_rng(seed)
    reference_tensor = None

    for index in range(1, num_samples + 1):
        if map_type == "compressor":
            tensor = generate_compressor_map(axis0, axis1, rng)
        else:
            tensor = generate_turbine_map(axis0, axis1, rng)
        save_sample(output_dir, map_type, index, tensor, axis0, axis1)
        if reference_tensor is None:
            reference_tensor = tensor

    if reference_tensor is not None:
        save_point_sets(output_dir, map_type, reference_tensor, axis0, axis1, rng)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic training maps and test point sets.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--map-type", choices=["compressor", "turbine", "both"], default="compressor")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.map_type == "both":
        generate_dataset("compressor", args.output_dir / "compressor", args.num_samples, args.seed, args.overwrite)
        generate_dataset("turbine", args.output_dir / "turbine", args.num_samples, args.seed + 1, args.overwrite)
    else:
        generate_dataset(args.map_type, args.output_dir / args.map_type, args.num_samples, args.seed, args.overwrite)


if __name__ == "__main__":
    main()
