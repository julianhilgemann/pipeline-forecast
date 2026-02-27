from __future__ import annotations

import numpy as np
import pandas as pd


def build_arrivals_surface(
    arrivals_fc: pd.DataFrame, kernel: pd.DataFrame, horizon: int
) -> np.ndarray:
    win_mass = kernel["win_kernel_mass"].to_numpy(dtype=float)
    arrivals = arrivals_fc.sort_values("day_offset")["forecast_arrivals"].to_numpy(dtype=float)[:horizon]
    mat = np.zeros((horizon, horizon), dtype=float)
    for create_ix in range(horizon):
        n = arrivals[create_ix]
        for close_ix in range(create_ix, horizon):
            age = close_ix - create_ix
            if age >= len(win_mass):
                break
            mat[create_ix, close_ix] = n * win_mass[age]
    return mat


def smooth_series(values: np.ndarray, window: int = 9, sigma: float = 2.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, arr.size if arr.size % 2 == 1 else max(3, arr.size - 1))
    if window < 3:
        return arr.copy()

    radius = window // 2
    x = np.arange(-radius, radius + 1, dtype=float)
    weights = np.exp(-(x * x) / (2.0 * sigma * sigma))
    weights = weights / weights.sum()
    padded = np.pad(arr, (radius, radius), mode="edge")
    return np.convolve(padded, weights, mode="valid")


def smooth_matrix(values: np.ndarray, passes: int = 1) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return arr.copy()
    out = arr.copy()
    for _ in range(max(1, passes)):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            padded[:-2, :-2]
            + 2.0 * padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + 2.0 * padded[1:-1, :-2]
            + 4.0 * padded[1:-1, 1:-1]
            + 2.0 * padded[1:-1, 2:]
            + padded[2:, :-2]
            + 2.0 * padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 16.0
    return out
