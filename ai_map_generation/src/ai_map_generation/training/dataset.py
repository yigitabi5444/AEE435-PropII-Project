from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..common.io_formats import load_map_npz


class MapDataset(Dataset):
    def __init__(self, folder: str | Path, expected_map_type: str) -> None:
        self.folder = Path(folder)
        if not self.folder.exists():
            raise FileNotFoundError(f"Dataset folder '{self.folder}' not found.")

        self.files = sorted(self.folder.glob("*.npz"))
        if not self.files:
            raise ValueError(f"No .npz files found in {self.folder}.")

        first_sample = load_map_npz(self.files[0], expected_map_type)
        self.map_type = first_sample.map_type
        self.axis0 = first_sample.axis0
        self.axis1 = first_sample.axis1
        self.input_shape = first_sample.tensor.shape

        for sample_path in self.files[1:]:
            sample = load_map_npz(sample_path, self.map_type)
            if sample.tensor.shape != self.input_shape:
                raise ValueError(
                    f"Sample {sample_path} has shape {sample.tensor.shape} "
                    f"expected {self.input_shape}."
                )
            if not np.allclose(sample.axis0, self.axis0) or not np.allclose(sample.axis1, self.axis1):
                raise ValueError(f"Sample {sample_path} grid does not match dataset grid.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = load_map_npz(self.files[index], self.map_type)
        return torch.from_numpy(sample.tensor)
