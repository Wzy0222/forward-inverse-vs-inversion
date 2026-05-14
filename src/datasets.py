"""HDF5 reading helpers for the public reproduction package."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


DEPTH_AXIS_301 = np.arange(0.0, 150.0 + 0.5, 0.5, dtype=np.float32)
PERIODS_16 = np.array(
    [
        10.0,
        12.0,
        15.0,
        18.0,
        20.0,
        22.0,
        25.0,
        28.0,
        30.0,
        32.0,
        35.0,
        38.0,
        40.0,
        42.0,
        45.0,
        50.0,
    ],
    dtype=np.float32,
)


def read_hdf5_arrays(path: str | Path, keys: Iterable[str]) -> dict[str, np.ndarray]:
    """Read selected arrays from an HDF5 file."""

    h5_path = Path(path)
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as handle:
        available = sorted(handle.keys())
        for key in keys:
            if key not in handle:
                raise KeyError(f"Missing HDF5 key {key!r} in {h5_path}. Available keys: {available}")
            arrays[key] = handle[key][:]
    return arrays
