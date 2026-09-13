"""SOH degradation trajectories, following BatteryMFormer's methodology.

Target for trajectory forecasting: given the first S cycles, predict SOH at
every cycle to end of life. Storage is one float per cycle, so all 1021 cells
come to about 3 MB against the 239 MB curve cache that supplies the input.

BatteryMFormer (Tan et al., KDD 2026) defines

    SOH_i = Cap_i / (Cap_0 * DoD)                                        (their eq. 2)

The DoD divisor matters and is easy to miss. A cell cycled 20-80% can only ever
discharge 60% of its nominal capacity, so without it that cell's SOH starts near
0.60 rather than 1.0 -- which is exactly the artefact seen earlier, where RWTH
cells appeared to start at 0.58 and HUST at 1.10. That was never inconsistent
bookkeeping; it was the missing DoD.

Only 120 of 1021 cells state DoD in their filename, so where it is absent it is
estimated from the data as the early-cycle capacity ratio. That is a
substitution, not their method: they read DoD from protocol metadata. It targets
the same thing -- SOH beginning at 1.0 -- but folds any pre-cycle-1 capacity
loss into the estimate.

Raw (unsmoothed) SOH is what gets cached. Artefact smoothing happens at load
time, because its thresholds are percentiles that must be computed on the
training split alone.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

# 90.2% of Li-ion cells reach EOL by cycle 2000, 96.8% by 3000, 100% by 5000.
T_MAX_DEFAULT = 5000

# Cells whose stored nominal capacity disagrees with their actual rating.
_NOMINAL_OVERRIDE = {"RWTH": 1.85, "SNL_18650_NCA_25C_20-80": 3.2}

_DOD_RE = re.compile(r"_(\d{1,3})-(\d{1,3})_")


def dod_from_name(name: str) -> float | None:
    """Parse depth of discharge from an SOC window in the filename, e.g. `_20-80_`."""
    m = _DOD_RE.search(name)
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    return (hi - lo) / 100.0 if 0 <= lo < hi <= 100 else None


def _nominal_for(name: str, data: dict) -> float:
    for prefix, val in _NOMINAL_OVERRIDE.items():
        if name.startswith(prefix):
            return val
    return float(data["nominal_capacity_in_Ah"])


def extract_trajectory(pkl_path: str | Path, early: int = 10) -> dict:
    """Read one processed .pkl -> raw SOH trajectory plus what smoothing needs.

    Returns soh_raw (unsmoothed), inter-cycle time gaps, and the DoD used.
    Smoothing is deliberately not applied here: its thresholds are training-split
    percentiles, so doing it now would leak test statistics into the cache.
    """
    name = Path(pkl_path).stem
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    nominal = _nominal_for(name, data)
    if not np.isfinite(nominal) or nominal <= 0:
        raise ValueError(f"{name}: bad nominal capacity {nominal!r}")

    caps, starts, ends = [], [], []
    for cyc in data["cycle_data"]:
        q = np.asarray(cyc.get("discharge_capacity_in_Ah", []), dtype=np.float64)
        q = q[np.isfinite(q)]
        caps.append(float(q.max()) if q.size else np.nan)
        t = np.asarray(cyc.get("time_in_s", []), dtype=np.float64)
        t = t[np.isfinite(t)]
        starts.append(float(t[0]) if t.size else np.nan)
        ends.append(float(t[-1]) if t.size else np.nan)

    caps = np.asarray(caps)
    ratio = caps / nominal

    dod = dod_from_name(name)
    dod_source = "filename"
    if dod is None:
        # Fall back to the early-cycle capacity ratio, which is what DoD scales.
        head = ratio[:early]
        head = head[np.isfinite(head)]
        dod = float(np.median(head)) if head.size else 1.0
        dod_source = "estimated"
    dod = float(np.clip(dod, 0.05, 1.5))

    # Gaps between consecutive cycles. Only meaningful if time_in_s is
    # cumulative across cycles; where it resets per cycle the differences go
    # negative, and the caller should fall back to the percentile rule.
    s, e = np.asarray(starts), np.asarray(ends)
    gap = np.full(len(caps), np.nan)
    gap[1:] = s[1:] - e[:-1]
    cumulative = bool(np.nanmedian(gap[1:]) >= 0) if len(gap) > 1 else False

    return {
        "soh_raw": (ratio / dod).astype(np.float32),
        "gap_s": gap.astype(np.float32),
        "nominal": np.float32(nominal),
        "dod": np.float32(dod),
        "dod_source": dod_source,
        "gap_usable": cumulative,
    }


# --------------------------------------------------------------------------
# artefact smoothing -- applied at load time, with training-split thresholds
# --------------------------------------------------------------------------
def delta_thresholds(train_sohs, lo_pct: float = 1.0, hi_pct: float = 99.0):
    """(gamma_minus, gamma_plus): percentiles of cycle-to-cycle SOH change.

    BatteryMFormer computes these on the training split only. Doing it over all
    cells would let test-set behaviour set the flagging thresholds.
    """
    d = np.concatenate([np.diff(np.asarray(s, dtype=np.float64)) for s in train_sohs if len(s) > 1])
    d = d[np.isfinite(d)]
    if d.size == 0:
        return -np.inf, np.inf
    return float(np.percentile(d, lo_pct)), float(np.percentile(d, hi_pct))


def find_artifacts(soh, gap_s=None, gamma=None, gap_thresh=None):
    """Cycle indices where a measurement artefact begins.

    Two detectors, matching the paper. Where inter-cycle timestamps exist, a
    long rest before a cycle marks a reference-performance test -- the cell
    recovers capacity while resting, producing a spurious upward step that is
    an artefact of the test schedule, not degradation. Otherwise a step outside
    the training-split percentiles is flagged.
    """
    soh = np.asarray(soh, dtype=np.float64)
    onsets = set()
    if gap_s is not None and gap_thresh is not None:
        g = np.asarray(gap_s, dtype=np.float64)
        onsets |= {int(i) for i in np.nonzero(g > gap_thresh)[0] if i > 0}
    if gamma is not None:
        lo, hi = gamma
        d = np.diff(soh)
        onsets |= {int(i) + 1 for i in np.nonzero((d > hi) | (d < lo))[0]}
    return sorted(onsets)


def smooth_trajectory(soh, gap_s=None, gamma=None, gap_thresh=None,
                      tol: float = 0.005, window: int = 5, anchors: int = 3):
    """PCHIP-interpolate over artefact regions, leaving normal cycles untouched.

    For each onset k_s, scan forward for the recovery point k_e: the first cycle
    whose SOH comes back to the pre-artefact level and stays there for `window`
    consecutive cycles. The region between is replaced by PCHIP interpolation
    anchored on `anchors` normal cycles either side.

    One deviation from the paper, which asks for SOH within a tolerance of the
    pre-artefact level. On a steadily fading curve that condition never holds
    for long -- the trajectory keeps descending past the pre-artefact value --
    so recovery is never detected and nothing is smoothed. The criterion here is
    one-sided in the direction of the excursion: an upward artefact ends once
    SOH has come back down to at or below its pre-artefact level (and symmetrically
    for a downward one), which does terminate on a declining trajectory.

    PCHIP rather than a spline because it is shape-preserving: it will not
    introduce overshoot, which on a monotonically fading curve would invent
    degradation that never happened.
    """
    from scipy.interpolate import PchipInterpolator

    y = np.asarray(soh, dtype=np.float64).copy()
    n = len(y)
    if n < 8:
        return y.astype(np.float32)

    for ks in find_artifacts(y, gap_s, gamma, gap_thresh):
        if ks <= 0 or ks >= n - 1:
            continue
        pre = y[ks - 1]
        upward = y[ks] > pre  # capacity recovery after a rest is an upward step
        ke = None
        for k in range(ks + 1, n):
            seg = y[k : min(k + window, n)]
            if seg.size == 0:
                break
            done = np.all(seg <= pre + tol) if upward else np.all(seg >= pre - tol)
            if done:
                ke = k
                break
        if ke is None or ke <= ks:
            continue
        left = np.arange(max(0, ks - anchors), ks)
        right = np.arange(ke, min(n, ke + anchors))
        idx = np.concatenate([left, right])
        if len(idx) < 2 or not np.isfinite(y[idx]).all():
            continue
        fill = np.arange(ks, ke)
        y[fill] = PchipInterpolator(idx, y[idx], extrapolate=False)(fill)
    return np.nan_to_num(y, nan=0.0).astype(np.float32)


def estimate_eol(soh, tau: float = 0.80, margin: float = 0.025, fit_last: int = 50):
    """Cycle index where SOH first reaches tau, extrapolating if it stops short.

    BatteryMFormer keeps cells that reached tau + 2.5% and linearly extrapolates
    the remaining distance. Measured SOH is used where it exists; extrapolation
    only fixes where the trajectory ends. Returns None if the cell never got
    close enough to estimate honestly.
    """
    soh = np.asarray(soh, dtype=np.float64)
    below = np.nonzero(soh <= tau)[0]
    if below.size:
        return int(below[0]) + 1, "measured"
    if soh.min() > tau + margin:
        return None, "insufficient degradation"
    tail = soh[-fit_last:]
    x = np.arange(len(soh) - len(tail), len(soh))
    slope, intercept = np.polyfit(x, tail, 1)
    if slope >= -1e-9:
        return None, "no fade in the tail"
    return int(np.ceil((tau - intercept) / slope)) + 1, "extrapolated"


def to_fixed_horizon(soh, t_max: int = T_MAX_DEFAULT, eol: int | None = None):
    """-> (y[t_max], mask[t_max]); mask marks cycles the cell actually reached."""
    y = np.zeros(t_max, dtype=np.float32)
    m = np.zeros(t_max, dtype=np.float32)
    n = min(len(soh), t_max if eol is None else min(t_max, eol))
    v = np.isfinite(soh[:n])
    y[:n][v] = np.asarray(soh[:n])[v]
    m[:n][v] = 1.0
    return y, m
