"""Tests for SOH trajectory construction (BatteryMFormer methodology)."""

from __future__ import annotations

import numpy as np
import pytest

from stateattention.data.trajectory import (
    delta_thresholds,
    dod_from_name,
    estimate_eol,
    find_artifacts,
    smooth_trajectory,
    to_fixed_horizon,
)


def test_dod_parsed_from_filename():
    assert dod_from_name("SNL_18650_NCA_25C_20-80_0.5-0.5C_c") == pytest.approx(0.60)
    assert dod_from_name("MICH_17C_pouch_NMC_-5C_50-100_0.2-1.5C") == pytest.approx(0.50)
    assert dod_from_name("MATR_b1c0") is None          # no SOC window -> estimate it
    assert dod_from_name("HUST_1-1") is None           # cell id, not an SOC window


def test_dod_divisor_puts_soh_at_one():
    """The point of the DoD divisor: a 20-80% cell must still start at SOH 1.0.

    Without it that cell reads 0.60 at beginning of life, which is what made
    RWTH look like it started at 0.58.
    """
    nominal, dod = 3.0, 0.60
    cap = np.array([1.80, 1.79, 1.78])                 # 60% of nominal, as designed
    assert (cap / nominal)[0] == pytest.approx(0.60)   # without DoD -- wrong
    assert (cap / (nominal * dod))[0] == pytest.approx(1.00)  # with DoD -- right


def test_delta_thresholds_use_only_what_they_are_given():
    """Thresholds come from the training split; a wild test cell must not widen
    them, or artefact flagging would be set by the data being evaluated."""
    calm = [np.linspace(1.0, 0.9, 200)]
    lo_a, hi_a = delta_thresholds(calm)
    wild = calm + [np.array([1.0, 0.5, 1.0, 0.5] * 50)]
    lo_b, hi_b = delta_thresholds(wild)
    assert hi_b > hi_a * 5, "sanity: the wild cell would move the thresholds a lot"
    assert hi_a == pytest.approx(-0.1 / 199, abs=1e-4), "calm-only threshold stays tight"


def test_artifact_detected_by_time_gap():
    soh = np.linspace(1.0, 0.9, 50)
    gap = np.zeros(50)
    gap[20] = 86400.0                                   # a day of rest before cycle 20
    assert 20 in find_artifacts(soh, gap_s=gap, gap_thresh=3600.0)


def test_artifact_detected_by_percentile():
    soh = np.linspace(1.0, 0.9, 50).copy()
    soh[30] += 0.05                                     # spurious capacity recovery
    lo, hi = delta_thresholds([np.linspace(1.0, 0.9, 50)])
    assert 30 in find_artifacts(soh, gamma=(lo, hi))


def test_smoothing_repairs_the_spike_and_leaves_the_rest_alone():
    clean = np.linspace(1.0, 0.85, 120)
    dirty = clean.copy()
    dirty[60:64] += 0.04                                # a recovery bump
    lo, hi = delta_thresholds([clean])
    out = smooth_trajectory(dirty, gamma=(lo, hi), tol=0.005, window=5, anchors=3)
    assert np.abs(out[60:64] - clean[60:64]).max() < np.abs(dirty[60:64] - clean[60:64]).max()
    untouched = np.r_[0:50, 80:120]
    assert np.allclose(out[untouched], clean[untouched], atol=1e-4), "smoothing spilled"


def test_smoothing_does_not_overshoot():
    """PCHIP is shape-preserving; a cubic spline would invent fade that never
    happened on a monotone curve."""
    clean = np.linspace(1.0, 0.85, 120)
    dirty = clean.copy()
    dirty[60:64] += 0.04
    lo, hi = delta_thresholds([clean])
    out = smooth_trajectory(dirty, gamma=(lo, hi))
    assert out.max() <= clean.max() + 1e-3
    assert out.min() >= clean.min() - 1e-3


def test_eol_measured_extrapolated_and_refused():
    reached = np.linspace(1.0, 0.75, 400)
    k, how = estimate_eol(reached)
    assert how == "measured" and reached[k - 1] <= 0.80

    close = np.linspace(1.0, 0.815, 400)                # stops just above tau
    k, how = estimate_eol(close)
    assert how == "extrapolated" and k > 400

    barely = np.linspace(1.0, 0.95, 400)                # nowhere near
    assert estimate_eol(barely)[0] is None


def test_fixed_horizon_masks_past_eol():
    y, m = to_fixed_horizon(np.linspace(1.0, 0.8, 300), t_max=1000, eol=300)
    assert y.shape == (1000,) and m.sum() == 300
    assert m[299] == 1.0 and m[300] == 0.0
