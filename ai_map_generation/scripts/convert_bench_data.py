from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

T_REF = 288.15
P_REF = 101.325
R_GAS = 287.0
GAMMA_COMP = 1.4
GAMMA_TURB = 1.33

SENSOR_KEYS = ["p03_raw", "p04_raw", "p05_raw"]
TEMP_KEYS = ["t02_raw", "t03_raw", "t04_raw", "t05_raw"]


def _parse_raw_file(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {"source_file": path.name}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            continue
        if key in {"timestamp", "operating_point_name"}:
            data[key] = value
            continue
        try:
            data[key] = float(value)
        except ValueError:
            data[key] = value
    return data


def _scale_p02(value: float, unit: str) -> float:
    if unit == "bar":
        return value * 100.0
    if unit == "kpa":
        return value
    if value < 20.0:
        return value * 100.0
    return value


def _calibration_from_zero(points: list[dict[str, Any]], p02_unit: str) -> tuple[dict[str, float], float]:
    if not points:
        raise ValueError("No data points provided.")
    zero_point = min(points, key=lambda item: abs(float(item.get("shaft_speed_rpm", 0.0))))
    p02_raw = float(zero_point.get("p02_raw", 0.0))
    p02_kpa = _scale_p02(p02_raw, p02_unit)
    if p02_kpa <= 0:
        raise ValueError("Invalid p02_raw value for calibration.")

    slopes: dict[str, float] = {}
    for key in SENSOR_KEYS:
        voltage = float(zero_point.get(key, 0.0))
        if voltage <= 0:
            raise ValueError(f"Invalid calibration voltage for {key}.")
        slopes[key] = p02_kpa / voltage

    return slopes, p02_kpa


def _pressure_from_voltage(raw_value: float | None, slope: float) -> float | None:
    if raw_value is None:
        return None
    return float(raw_value) * slope


def _to_kelvin(value_c: float | None) -> float | None:
    if value_c is None:
        return None
    return float(value_c) + 273.15


def _mass_flow(pt5_kpa: float, tt5_k: float, throat_area: float) -> float:
    if pt5_kpa <= 0 or tt5_k <= 0 or throat_area <= 0:
        return float("nan")
    pt5_pa = pt5_kpa * 1000.0
    gamma = GAMMA_TURB
    factor = math.sqrt(gamma / R_GAS) * ((gamma + 1.0) / 2.0) ** (
        -(gamma + 1.0) / (2.0 * (gamma - 1.0))
    )
    return pt5_pa * throat_area / math.sqrt(tt5_k) * factor


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _compute_comp_point(record: dict[str, Any], speed_source: str) -> dict[str, float] | None:
    tt2 = record["t02_k"]
    tt3 = record["t03_k"]
    pt2 = record["pt2_kpa"]
    pt3 = record["pt3_kpa"]
    mdot = record["mdot"]

    if any(value is None for value in (tt2, tt3, pt2, pt3)):
        return None

    theta_c = tt2 / T_REF
    delta_c = pt2 / P_REF

    if speed_source == "nc_turb":
        nc = record.get("nc_turb")
    else:
        n_rpm = float(record.get("shaft_speed_rpm", 0.0))
        nc = _safe_divide(n_rpm, math.sqrt(theta_c))

    if nc is None:
        return None

    mdot_c = mdot * math.sqrt(theta_c) / delta_c if delta_c > 0 else float("nan")
    pi_c = _safe_divide(pt3, pt2)

    temp_ratio = _safe_divide(tt3, tt2)
    eta_den = temp_ratio - 1.0
    eta_num = pi_c ** ((GAMMA_COMP - 1.0) / GAMMA_COMP) - 1.0
    eta = _safe_divide(eta_num, eta_den)

    if not all(np.isfinite(value) for value in (nc, mdot_c, pi_c, eta)):
        return None

    record["nc_comp"] = nc
    return {"Nc": nc, "mdotc": mdot_c, "eta": eta, "pi": pi_c}


def _compute_turb_point(record: dict[str, Any], speed_source: str) -> dict[str, float] | None:
    tt4 = record["t04_k"]
    tt5 = record["t05_k"]
    pt4 = record["pt4_kpa"]
    pt5 = record["pt5_kpa"]
    mdot = record["mdot"]

    if any(value is None for value in (tt4, tt5, pt4, pt5)):
        return None

    theta_t = tt4 / T_REF
    delta_t = pt4 / P_REF

    if speed_source == "nc_comp":
        nc = record.get("nc_comp")
    else:
        n_rpm = float(record.get("shaft_speed_rpm", 0.0))
        nc = _safe_divide(n_rpm, math.sqrt(theta_t))

    if nc is None:
        return None

    mdot_c = mdot * math.sqrt(theta_t) / delta_t if delta_t > 0 else float("nan")
    pi_t = _safe_divide(pt4, pt5)

    eta_num = 1.0 - _safe_divide(tt5, tt4)
    eta_den = 1.0 - pi_t ** (-(GAMMA_TURB - 1.0) / GAMMA_TURB)
    eta = _safe_divide(eta_num, eta_den)

    if not all(np.isfinite(value) for value in (nc, mdot_c, pi_t, eta)):
        return None

    record["nc_turb"] = nc
    return {"Nc": nc, "pi_t": pi_t, "eta": eta, "mdotc": mdot_c}


def convert_points(
    input_dir: Path,
    output_dir: Path,
    include_zero: bool,
    p02_unit: str,
    comp_speed_source: str,
    turb_speed_source: str,
) -> tuple[Path, Path]:
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}.")

    points = [_parse_raw_file(path) for path in files]
    slopes, p02_cal = _calibration_from_zero(points, p02_unit)

    comp_rows: list[dict[str, float]] = []
    turb_rows: list[dict[str, float]] = []

    for record in points:
        n_rpm = float(record.get("shaft_speed_rpm", 0.0))
        if n_rpm == 0.0 and not include_zero:
            continue

        p02_kpa = _scale_p02(float(record.get("p02_raw", p02_cal)), p02_unit)
        record["pt2_kpa"] = p02_kpa
        record["pt3_kpa"] = _pressure_from_voltage(record.get("p03_raw"), slopes["p03_raw"])
        record["pt4_kpa"] = _pressure_from_voltage(record.get("p04_raw"), slopes["p04_raw"])
        record["pt5_kpa"] = _pressure_from_voltage(record.get("p05_raw"), slopes["p05_raw"])

        for key in TEMP_KEYS:
            record[key.replace("_raw", "_k")] = _to_kelvin(record.get(key))

        tt5 = record["t05_k"]
        if record["pt5_kpa"] is None or tt5 is None:
            continue
        record["mdot"] = _mass_flow(record["pt5_kpa"], tt5, float(record.get("throat_area", 0.0)))

        comp_point = _compute_comp_point(record, comp_speed_source)
        if comp_point:
            comp_rows.append(comp_point)

        turb_point = _compute_turb_point(record, turb_speed_source)
        if turb_point:
            turb_rows.append(turb_point)

    output_dir.mkdir(parents=True, exist_ok=True)
    comp_path = output_dir / "compressor_points.csv"
    turb_path = output_dir / "turbine_points.csv"

    comp_frame = pd.DataFrame(comp_rows)
    turb_frame = pd.DataFrame(turb_rows)
    comp_frame = comp_frame[["Nc", "mdotc", "eta", "pi"]]
    turb_frame = turb_frame[["Nc", "pi_t", "eta", "mdotc"]]
    comp_frame.to_csv(comp_path, index=False)
    turb_frame.to_csv(turb_path, index=False)

    return comp_path, turb_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert bench raw TXT files into compressor/turbine CSVs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("bench_data/raw"),
        help="Folder containing raw operating_point_*.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bench_data/processed"),
        help="Output folder for compressor/turbine CSVs.",
    )
    parser.add_argument("--include-zero", action="store_true", help="Include the zero-rpm calibration point.")
    parser.add_argument(
        "--p02-unit",
        choices=["auto", "kpa", "bar"],
        default="auto",
        help="Units of p02_raw (auto treats values < 20 as bar).",
    )
    parser.add_argument(
        "--comp-speed-source",
        choices=["shaft_speed_rpm", "nc_turb"],
        default="shaft_speed_rpm",
        help="Source for compressor corrected speed (default uses shaft speed).",
    )
    parser.add_argument(
        "--turb-speed-source",
        choices=["shaft_speed_rpm", "nc_comp"],
        default="shaft_speed_rpm",
        help="Source for turbine corrected speed (default uses shaft speed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comp_path, turb_path = convert_points(
        args.input_dir,
        args.output_dir,
        args.include_zero,
        args.p02_unit,
        args.comp_speed_source,
        args.turb_speed_source,
    )
    print(f"Wrote {comp_path} and {turb_path}")


if __name__ == "__main__":
    main()
