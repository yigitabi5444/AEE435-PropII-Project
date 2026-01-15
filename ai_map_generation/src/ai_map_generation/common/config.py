from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MapConfig:
    map_type: str
    axis0_name: str
    axis1_name: str
    channels: tuple[str, str]


MAP_CONFIGS = {
    "compressor": MapConfig(
        map_type="compressor",
        axis0_name="Nc",
        axis1_name="mdotc",
        channels=("eta", "pi"),
    ),
    "turbine": MapConfig(
        map_type="turbine",
        axis0_name="Nc",
        axis1_name="pi_t",
        channels=("eta", "mdotc"),
    ),
}


def normalize_map_type(map_type: str) -> str:
    if not map_type:
        raise ValueError("map_type is required")
    return map_type.strip().lower()


def get_map_config(map_type: str) -> MapConfig:
    normalized = normalize_map_type(map_type)
    if normalized not in MAP_CONFIGS:
        raise ValueError(f"Unknown map_type '{map_type}'. Expected compressor or turbine.")
    return MAP_CONFIGS[normalized]
