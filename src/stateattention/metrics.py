"""Evaluation metrics.

MAPE and alpha-accuracy follow the BatteryLife benchmark (KDD 2025) exactly, so
numbers here are directly comparable to their published table. Both are reported
as *fractions*, not percentages: their best Li-ion MAPE of 0.179 means 17.9%.
"""

from __future__ import annotations

import numpy as np

__all__ = ["mape", "alpha_accuracy", "rmse", "mae", "compute_metrics"]


def _as_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, as a fraction.

    BatteryLife eq. 10: (1/N) * sum(|y - yhat| / y). Labels are cycle lives, so
    y > 0 always holds; we guard anyway rather than emit inf.
    """
    y, p = _as_1d(y_true), _as_1d(y_pred)
    denom = np.where(np.abs(y) < 1e-8, np.nan, y)
    return float(np.nanmean(np.abs(y - p) / np.abs(denom)))


def alpha_accuracy(y_true, y_pred, alpha: float = 0.15) -> float:
    """Fraction of predictions within +/- alpha relative error.

    BatteryLife eq. 11. alpha=0.15 gives the "15%-Acc" column, i.e. the share of
    predictions an engineer would call usable.
    """
    y, p = _as_1d(y_true), _as_1d(y_pred)
    return float(np.mean(np.abs(y - p) <= alpha * np.abs(y)))


def rmse(y_true, y_pred) -> float:
    y, p = _as_1d(y_true), _as_1d(y_pred)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y_true, y_pred) -> float:
    y, p = _as_1d(y_true), _as_1d(y_pred)
    return float(np.mean(np.abs(y - p)))


def compute_metrics(y_true, y_pred, alphas=(0.15,)) -> dict[str, float]:
    """Full metric bundle for one evaluation set."""
    out = {
        "mape": mape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "n": int(_as_1d(y_true).size),
    }
    for a in alphas:
        out[f"acc@{int(round(a * 100))}%"] = alpha_accuracy(y_true, y_pred, a)
    return out


def aggregate_seeds(runs: list[dict[str, float]]) -> dict[str, str]:
    """mean +/- std across seeds, formatted like the BatteryLife paper.

    The paper runs every config 3x and reports mean+/-std; single-seed numbers
    are not comparable to theirs, so always aggregate before you claim a win.
    """
    if not runs:
        return {}
    keys = [k for k in runs[0] if k != "n"]
    return {
        k: f"{np.mean([r[k] for r in runs]):.3f}±{np.std([r[k] for r in runs]):.3f}" for k in keys
    }
