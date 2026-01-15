from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def parse_int_list(text: str) -> list[int]:
    if not text:
        return []
    parts = [item.strip() for item in text.split(",") if item.strip()]
    values: list[int] = []
    for item in parts:
        if not item.isdigit():
            raise ValueError(f"Invalid integer value '{item}'.")
        values.append(int(item))
    return values


def ensure_float32(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def to_json(data: dict) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def ensure_numpy_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float32)
