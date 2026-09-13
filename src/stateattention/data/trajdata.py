"""Dataset for SOH trajectory forecasting.

    input   first S cycles of curves   [S, curve_len, 3]   (from the curve cache)
    target  SOH at every cycle to EOL  [t_max] + mask      (from soh-trajectories)

Artefact smoothing runs here rather than in the cache because its thresholds are
percentiles of cycle-to-cycle SOH change computed on the *training split only*.
Baking them into the cache would let test-set behaviour decide what counts as an
artefact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .batterylife import BMF_PREFIX, SPLITS_FILE, cache_key, load_split_spec
from .trajectory import (
    T_MAX_DEFAULT,
    delta_thresholds,
    estimate_eol,
    smooth_trajectory,
    to_fixed_horizon,
)

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    class Dataset:  # type: ignore[no-redef]
        pass


def load_pool(curve_dir: Path, traj_dir: Path, names: list[str]) -> dict:
    """Read every cell that has BOTH an input curve array and a target trajectory."""
    out = {}
    for n in names:
        stem = n.removesuffix(".pkl")
        c, t = curve_dir / f"{stem}.npz", traj_dir / f"{stem}.npz"
        if not (c.exists() and t.exists()):
            continue
        tz = np.load(t, allow_pickle=True)
        soh = np.asarray(tz["soh_raw"], dtype=np.float32)
        if len(soh) < 20 or not np.isfinite(soh).any():
            continue
        gap = np.asarray(tz["gap_s"], dtype=np.float32) if "gap_s" in tz else None
        out[stem] = {
            "curve": c,
            "soh_raw": soh,
            "gap_s": gap,
            "gap_usable": bool(tz["gap_usable"]) if "gap_usable" in tz else False,
        }
    return out


class TrajDataset(Dataset):
    def __init__(self, pool: dict, names: list[str], gamma, curve_key: str,
                 t_max: int = T_MAX_DEFAULT, max_cycles: int = 100,
                 gap_thresh: float = 3 * 3600.0, tau: float = 0.80,
                 normalize_capacity: bool = True, require_eol: bool = False):
        self.items = []
        self.max_cycles, self.t_max = max_cycles, t_max
        skipped = 0
        for n in names:
            stem = n.removesuffix(".pkl")
            rec = pool.get(stem)
            if rec is None:
                continue
            soh = smooth_trajectory(
                rec["soh_raw"], rec["gap_s"] if rec["gap_usable"] else None,
                gamma=gamma, gap_thresh=gap_thresh if rec["gap_usable"] else None,
            )
            eol, how = estimate_eol(soh, tau=tau)
            if eol is None:
                # BatteryMFormer's mask marks whether ground-truth SOH is
                # *available* at a cycle, not whether the cell reached EOL. A
                # cell that never faded to tau still supervises over its
                # observed window, so keep it and mask to what was measured.
                # Dropping these loses whole domains: CALB's processed data
                # stops at ~99 cycles against a labelled life near 1242, so 26
                # of its 27 cells never reach any threshold.
                if require_eol:
                    skipped += 1
                    continue
                eol, how = len(soh), "observed-window-only"
            if eol < 20:
                skipped += 1
                continue
            y, m = to_fixed_horizon(soh, t_max, eol)
            if m.sum() < 20:
                skipped += 1
                continue
            self.items.append({"stem": stem, "curve": rec["curve"], "y": y, "m": m,
                               "eol": eol, "how": how})
        self.skipped = skipped
        self.normalize_capacity = normalize_capacity
        self.curve_key = curve_key

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        z = np.load(it["curve"])
        cur = np.asarray(z["curves"], dtype=np.float32)
        if self.normalize_capacity:
            q0 = float(z["nominal_capacity"])
            cur = cur.copy()
            cur[..., 2] /= max(q0, 1e-12)
        n = min(self.max_cycles, cur.shape[0])
        x = np.zeros((self.max_cycles, cur.shape[1], cur.shape[2]), dtype=np.float32)
        x[:n] = cur[:n]
        mask = np.zeros(self.max_cycles, dtype=np.float32)
        mask[:n] = 1.0
        return {"x": x, "mask": mask, "y": it["y"], "ymask": it["m"]}


# EOL threshold per domain. BatteryLife uses 80% of nominal for lab Li-ion and
# 90% for industrial cells; na-ion cells rarely fade past ~0.88 in the recorded
# window, so 0.80 leaves its val/test splits empty.
DOMAIN_TAU = {"li-ion": 0.80, "zn-ion": 0.80, "na-ion": 0.90, "calb": 0.90}


def censoring_rate(pool: dict, names: list[str], tau: float) -> float:
    """Fraction of cells whose recorded SOH never reaches the EOL threshold.

    A censored cell's target is the flat opening stretch of a much longer
    trajectory, so MAPE on it scores how well a model reproduces SOH ~ 1.0
    rather than whether it forecasts degradation. BatteryLife's public CALB
    release stops at 99-100 cycles while its own life labels put CALB EOL past
    1000, so CALB is ~100% censored: on it a constant predictor beats a trained
    model, and neither can be compared with a model fit on the full curves.
    """
    have = [pool[n.removesuffix(".pkl")]["soh_raw"] for n in names
            if n.removesuffix(".pkl") in pool]
    if not have:
        return 1.0
    return sum(1 for s in have if not (np.nanmin(s) <= tau)) / len(have)


def build_splits(root: Path, group: str = "MIX_large", t_max: int = T_MAX_DEFAULT,
                 max_cycles: int = 100, curve_len: int = 300,
                 min_coverage: float = 0.99, max_censoring: float = 0.5, **kw):
    """-> (train, val, test) TrajDatasets, with gamma from the training split only.

    For a `bmf:` group the point is that the split is *exactly* BatteryMFormer's,
    so a fold we can only partially populate is refused rather than quietly run
    and reported against their table. `min_coverage` is the escape hatch.
    """
    curve_dir = Path(root) / ".cache" / cache_key(max_cycles, curve_len)
    traj_dir = Path(root) / ".cache" / "soh_trajectories"
    sp = load_split_spec(group)
    pool = load_pool(curve_dir, traj_dir, sp["train"] + sp["val"] + sp["test"])

    if group.startswith(BMF_PREFIX):
        short = {
            s: (n_have, len(sp[s]))
            for s in ("train", "val", "test")
            for n_have in [sum(c.removesuffix(".pkl") in pool for c in sp[s])]
            if n_have < min_coverage * len(sp[s])
        }
        if short:
            detail = ", ".join(f"{s} {h}/{n}" for s, (h, n) in short.items())
            raise ValueError(
                f"group {group!r} is only partially covered ({detail}). Running it "
                "would not be their split, so the result could not be compared with "
                "their Table 2. Cache the missing cells, or pass min_coverage."
            )

    train_raw = [pool[n.removesuffix(".pkl")]["soh_raw"]
                 for n in sp["train"] if n.removesuffix(".pkl") in pool]
    gamma = delta_thresholds(train_raw)

    tau = kw.get("tau", 0.80)
    censored = censoring_rate(pool, sp["test"], tau)
    if group.startswith(BMF_PREFIX) and censored > max_censoring:
        raise ValueError(
            f"group {group!r}: {censored:.0%} of test cells never reach SOH {tau} "
            f"-- their recorded curves end while still near 1.0, so the target is a "
            f"censored stub of the real trajectory. A MAPE on it is not comparable "
            f"with a number computed on full-length curves. Pass max_censoring to "
            f"override deliberately."
        )

    ck = cache_key(max_cycles, curve_len)
    ds = tuple(
        TrajDataset(pool, sp[s], gamma, ck, t_max, max_cycles, **kw)
        for s in ("train", "val", "test")
    )
    empty = [n for n, d in zip(("train", "val", "test"), ds, strict=True) if len(d) == 0]
    if empty:
        raise ValueError(
            f"group {group!r}: empty {'/'.join(empty)} split "
            f"({len(pool)} cells had both curves and a trajectory). "
            "Usually the EOL threshold: raise tau, or set require_eol=False."
        )
    return ds, gamma, len(pool)
