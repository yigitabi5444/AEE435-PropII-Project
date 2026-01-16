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
from ai_map_generation.common.gasturb import (
    MapGrid,
    compressor_to_tensor,
    normalize_compressor,
    normalize_turbine,
    parse_compressor_map,
    parse_turbine_map,
    perturb_tensor,
    turbine_to_tensor,
)
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
    sample_id: str,
    tensor: np.ndarray,
    axis0: np.ndarray,
    axis1: np.ndarray,
    filename: str | None = None,
    extra_meta: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"{sample_id}.npz"
    path = output_dir / filename
    payload = {
        "X": tensor.astype(np.float32),
        "map_type": map_type,
        "id": sample_id,
        "axis0": axis0.astype(np.float32),
        "axis1": axis1.astype(np.float32),
    }
    if extra_meta:
        payload.update(extra_meta)
    np.savez(path, **payload)


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


def generate_dataset_analytic(map_type: str, output_dir: Path, num_samples: int, seed: int, overwrite: bool) -> None:
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
        sample_id = f"{map_type}_synthetic_{index:04d}"
        save_sample(output_dir, map_type, sample_id, tensor, axis0, axis1, filename=f"sample_{index:04d}.npz")
        if reference_tensor is None:
            reference_tensor = tensor

    if reference_tensor is not None:
        save_point_sets(output_dir, map_type, reference_tensor, axis0, axis1, rng)


def _map_files(folder: Path) -> list[Path]:
    return sorted({*folder.glob("*.MAP"), *folder.glob("*.map"), *folder.glob("*.Map")})


def _load_compressor_maps(maps_dir: Path) -> list:
    compressor_dir = maps_dir / "compressor"
    return [parse_compressor_map(path) for path in _map_files(compressor_dir)]


def _load_turbine_maps(maps_dir: Path) -> list:
    turbine_dir = maps_dir / "turbine"
    return [parse_turbine_map(path) for path in _map_files(turbine_dir)]


def _grid_for_compressors(grid_size: int) -> MapGrid:
    axis0 = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    axis1 = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    return MapGrid(axis0=axis0, axis1=axis1)


def _grid_for_turbines(grid_size: int) -> MapGrid:
    axis0 = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    axis1 = np.linspace(0.0, 5.0, grid_size, dtype=np.float32)
    return MapGrid(axis0=axis0, axis1=axis1)


def generate_dataset_from_maps(
    map_type: str,
    maps_dir: Path,
    output_dir: Path,
    num_samples: int,
    seed: int,
    overwrite: bool,
    mode: str,
    noise_scale: float,
    grid_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(output_dir.glob("*.npz")):
        raise FileExistsError(f"Output directory {output_dir} already contains .npz files.")

    rng = np.random.default_rng(seed)

    if map_type == "compressor":
        maps = _load_compressor_maps(maps_dir)
        if not maps:
            raise FileNotFoundError(f"No compressor maps found in {maps_dir}.")
        grid = _grid_for_compressors(grid_size)
        convert = compressor_to_tensor
        normalizer = normalize_compressor
    else:
        maps = _load_turbine_maps(maps_dir)
        if not maps:
            raise FileNotFoundError(f"No turbine maps found in {maps_dir}.")
        grid = _grid_for_turbines(grid_size)
        convert = turbine_to_tensor
        normalizer = normalize_turbine

    base_tensors: list[np.ndarray] = []
    base_meta: list[dict] = []
    reference_tensor: np.ndarray | None = None

    if mode in {"real", "both"}:
        for map_item in maps:
            normalized_map, meta = normalizer(map_item)
            tensor = convert(normalized_map, grid)
            base_tensors.append(tensor)
            base_meta.append(meta)
            sample_id = f"{map_type}_{map_item.name}"
            filename = f"real_{map_item.name}.npz"
            save_sample(
                output_dir,
                map_type,
                sample_id,
                tensor,
                grid.axis0,
                grid.axis1,
                filename=filename,
                extra_meta=meta,
            )
            if reference_tensor is None:
                reference_tensor = tensor

    if not base_tensors:
        for map_item in maps:
            normalized_map, meta = normalizer(map_item)
            base_tensors.append(convert(normalized_map, grid))
            base_meta.append(meta)

    if mode in {"synthetic", "both"}:
        for index in range(1, num_samples + 1):
            base_index = int(rng.integers(0, len(base_tensors)))
            base_tensor = base_tensors[base_index]
            synthetic = perturb_tensor(base_tensor, rng, noise_scale)
            sample_id = f"{map_type}_synth_{index:04d}"
            filename = f"synth_{index:04d}.npz"
            save_sample(
                output_dir,
                map_type,
                sample_id,
                synthetic,
                grid.axis0,
                grid.axis1,
                filename=filename,
                extra_meta=base_meta[base_index],
            )
            if reference_tensor is None:
                reference_tensor = synthetic

    if reference_tensor is not None:
        save_point_sets(output_dir, map_type, reference_tensor, grid.axis0, grid.axis1, rng)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate training maps and test point sets.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--maps-dir", type=Path, default=PROJECT_ROOT / "maps")
    parser.add_argument("--map-type", choices=["compressor", "turbine", "both"], default="both")
    parser.add_argument("--source", choices=["maps", "analytic"], default="maps")
    parser.add_argument("--mode", choices=["real", "synthetic", "both"], default="both")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--noise-scale", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "analytic":
        if args.map_type == "both":
            generate_dataset_analytic(
                "compressor", args.output_dir / "compressor", args.num_samples, args.seed, args.overwrite
            )
            generate_dataset_analytic(
                "turbine", args.output_dir / "turbine", args.num_samples, args.seed + 1, args.overwrite
            )
        else:
            generate_dataset_analytic(
                args.map_type, args.output_dir / args.map_type, args.num_samples, args.seed, args.overwrite
            )
        return

    if args.map_type == "both":
        generate_dataset_from_maps(
            "compressor",
            args.maps_dir,
            args.output_dir / "compressor",
            args.num_samples,
            args.seed,
            args.overwrite,
            args.mode,
            args.noise_scale,
            args.grid_size,
        )
        generate_dataset_from_maps(
            "turbine",
            args.maps_dir,
            args.output_dir / "turbine",
            args.num_samples,
            args.seed + 1,
            args.overwrite,
            args.mode,
            args.noise_scale,
            args.grid_size,
        )
    else:
        generate_dataset_from_maps(
            args.map_type,
            args.maps_dir,
            args.output_dir / args.map_type,
            args.num_samples,
            args.seed,
            args.overwrite,
            args.mode,
            args.noise_scale,
            args.grid_size,
        )


if __name__ == "__main__":
    main()
