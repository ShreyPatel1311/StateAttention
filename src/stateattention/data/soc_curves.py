"""SOC-aligned curve preparation, following BatteryMFormer's published method.

Each cycle is resampled onto an equally spaced *state-of-charge* grid rather
than uniformly in sample index, and carries four channels (voltage, current in
C-rate, capacity, SOC). SOC itself is reconstructed per charge/discharge segment
from the protocol interval, as described in Appendix B of

    R. Tan et al., "BatteryMFormer: Multi-level Learning for Battery
    Degradation Trajectory Forecasting", KDD 2026.

        SOC_ik = SOC_start + (Q_ik - Q_start) / (Q_end - Q_start)
                             * (SOC_end - SOC_start)

This matters because a uniform-index resampling destroys the SOC axis, and
degradation signatures are localised to particular SOC windows. The
implementation follows their reference code at
github.com/Ruifeng-Tan/BatteryMFormer (no licence declared at time of writing;
see NOTICE in the README).

Correctness check: their loader asserts the resampled SOC channel meets the
protocol interval endpoints within 2e-3. 1003 of 1006 cells pass here, the
three failures being ISU-ILCC cells off by 0.009.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NEED_KEYS = ["current_in_A", "voltage_in_V", "charge_capacity_in_Ah",
             "discharge_capacity_in_Ah", "time_in_s"]

# Their hard-coded list of ZN-coin cells that charge before discharging.
ZN_COIN_CHARGE_FIRST = {
    "ZN-coin_402-1_20231209225636_01_1.pkl", "ZN-coin_402-2_20231209225727_01_2.pkl",
    "ZN-coin_402-3_20231209225844_01_3.pkl", "ZN-coin_403-1_20231209225922_01_4.pkl",
    "ZN-coin_428-1_20231212185048_01_2.pkl", "ZN-coin_428-2_20231212185058_01_4.pkl",
    "ZN-coin_429-1_20231212185129_01_5.pkl", "ZN-coin_429-2_20231212185157_01_8.pkl",
    "ZN-coin_430-1_20231212185250_02_6.pkl", "ZN-coin_430-2_20231212185305_02_7.pkl",
    "ZN-coin_430-3_20231212185323_03_2.pkl",
}
DISCHARGE_FIRST_PREFIXES = {"RWTH", "CALB_0", "CALB_25", "CALB_45"}


def _sanitize(values, fill_value=0.0):
    arr = np.asarray(values, dtype=np.float32)
    if np.isnan(arr).any() or np.isinf(arr).any():
        return np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return arr


def _fast_resample(curve, target_length):
    if len(curve) == 0:
        return np.zeros(target_length, dtype=np.float32)
    curve = _sanitize(curve)
    old_idx = np.linspace(0, len(curve) - 1, len(curve))
    new_idx = np.linspace(0, len(curve) - 1, target_length)
    return _sanitize(np.interp(new_idx, old_idx, curve))


def _soc_based_resample(voltage, current, capacity, soc, target_length, is_charge=True):
    """Interpolate v/i/c as functions of SOC onto an equally-spaced SOC grid."""
    if len(soc) == 0 or len(voltage) == 0:
        z = np.zeros(target_length, dtype=np.float32)
        return z, z, z, z

    voltage, current = _sanitize(voltage), _sanitize(current)
    capacity, soc = _sanitize(capacity), _sanitize(soc)

    soc_mono = np.maximum.accumulate(soc) if is_charge else np.minimum.accumulate(soc)
    ds = np.diff(soc_mono)
    valid = (np.concatenate([[True], ds > 1e-8]) if is_charge
             else np.concatenate([[True], ds < -1e-8]))

    if valid.sum() < 2:
        return (_fast_resample(voltage, target_length), _fast_resample(current, target_length),
                _fast_resample(capacity, target_length), _fast_resample(soc, target_length))

    soc_valid, v_valid = soc_mono[valid], voltage[valid]
    i_valid, c_valid = current[valid], capacity[valid]
    soc_min, soc_max = float(soc_valid.min()), float(soc_valid.max())
    target_soc = (np.linspace(soc_min, soc_max, target_length, dtype=np.float32) if is_charge
                  else np.linspace(soc_max, soc_min, target_length, dtype=np.float32))

    if is_charge:
        x = soc_valid
        v = np.interp(target_soc, x, v_valid)
        i = np.interp(target_soc, x, i_valid)
        c = np.interp(target_soc, x, c_valid)
    else:
        x = soc_valid[::-1]
        v = np.interp(target_soc[::-1], x, v_valid[::-1])[::-1]
        i = np.interp(target_soc[::-1], x, i_valid[::-1])[::-1]
        c = np.interp(target_soc[::-1], x, c_valid[::-1])[::-1]
    return _sanitize(v), _sanitize(i), _sanitize(c), _sanitize(target_soc)


def prepare_curves(file_name, cycle_data, nominal_capacity, start_soc, end_soc,
                   charge_discharge_len=300, early_cycle_threshold=100, seq_len=1):
    """-> curves [n_cycles, 4, L] (V, I in C-rate, Q, SOC), CE [n], EE [n]."""
    curves, ces, ees = [], [], []
    soc_range = float(end_soc - start_soc)
    prefix = file_name.split("_")[0]
    if prefix == "CALB":
        prefix = "_".join(file_name.split("_")[:2])
    half = charge_discharge_len // 2

    def pad():
        curves.append(np.zeros((1, 4, charge_discharge_len), dtype=np.float32))
        ces.append(0.0)
        ees.append(0.0)

    for cycle_idx in range(seq_len, early_cycle_threshold + 1):
        if cycle_idx >= len(cycle_data):
            pad()
            continue
        cycle = cycle_data[cycle_idx - 1]

        df = pd.DataFrame({k: cycle[k] for k in NEED_KEYS})
        df.loc[df["charge_capacity_in_Ah"] < 0] = np.nan
        df.loc[df["discharge_capacity_in_Ah"] < 0] = np.nan
        df.bfill(inplace=True)
        df.ffill(inplace=True)

        t_h = _sanitize(df["time_in_s"]) / 3600.0
        voltage = _sanitize(df["voltage_in_V"])
        current = _sanitize(df["current_in_A"])
        charge_cap = _sanitize(df["charge_capacity_in_Ah"])
        discharge_cap = _sanitize(df["discharge_capacity_in_Ah"])
        c_rate = current / float(nominal_capacity)

        cut_chg = np.nonzero(c_rate >= 0.01)
        cut_dis = np.nonzero(c_rate <= -0.01)
        if cut_chg[0].size == 0 or cut_dis[0].size == 0:
            pad()
            continue

        charge_end = cut_dis[0][0] - 1 if "MATR_b1c18" in file_name else cut_chg[0][-1]
        discharge_end = cut_dis[0][-1]

        discharge_first = prefix in DISCHARGE_FIRST_PREFIXES or (
            prefix == "ZN-coin" and file_name not in ZN_COIN_CHARGE_FIRST)

        if discharge_first:
            v_dis, c_dis = voltage[:discharge_end], discharge_cap[:discharge_end]
            i_dis, t_dis = current[:discharge_end], t_h[:discharge_end]
            v_chg, c_chg = voltage[discharge_end:], charge_cap[discharge_end:]
            i_chg, t_chg = current[discharge_end:], t_h[discharge_end:]
            keep = np.abs(i_chg / float(nominal_capacity)) > 0.01
            v_chg, c_chg, i_chg, t_chg = v_chg[keep], c_chg[keep], i_chg[keep], t_chg[keep]
        else:
            v_dis, c_dis = voltage[charge_end:], discharge_cap[charge_end:]
            i_dis, t_dis = current[charge_end:], t_h[charge_end:]
            keep = np.abs(i_dis / float(nominal_capacity)) > 0.01
            v_dis, c_dis, i_dis, t_dis = v_dis[keep], c_dis[keep], i_dis[keep], t_dis[keep]
            v_chg, c_chg = voltage[:charge_end], charge_cap[:charge_end]
            i_chg, t_chg = current[:charge_end], t_h[:charge_end]

        e_dis = np.trapz(v_dis * np.abs(i_dis), t_dis) if t_dis.size > 1 else 0.0
        e_chg = np.trapz(v_chg * np.abs(i_chg), t_chg) if t_chg.size > 1 else 1e-6
        ees.append(float(e_dis) / float(max(e_chg, 1e-6)))
        ces.append(float(max(c_dis) / max(max(c_chg), 1e-6))
                   if len(c_dis) > 0 and len(c_chg) > 0 else 0.0)

        if len(c_chg) > 0:
            d = float(max(c_chg.max() - c_chg.min(), 1e-6))
            soc_chg = float(start_soc) + ((c_chg - c_chg[0]) / d) * soc_range
        else:
            soc_chg = np.array([], dtype=np.float32)
        if len(c_dis) > 0:
            d = float(max(c_dis.max() - c_dis.min(), 1e-6))
            soc_dis = float(end_soc) - ((c_dis - c_dis[0]) / d) * soc_range
        else:
            soc_dis = np.array([], dtype=np.float32)

        cv, ci, cc, cs = _soc_based_resample(v_chg, i_chg, c_chg, soc_chg, half, True)
        dv, di, dc, ds = _soc_based_resample(v_dis, i_dis, c_dis, soc_dis, half, False)

        v_curve = _sanitize(np.concatenate([cv, dv]))
        i_curve = _sanitize(np.concatenate([ci, di])) / float(nominal_capacity)
        c_curve = _sanitize(np.concatenate([cc, dc]))
        s_curve = _sanitize(np.concatenate([cs, ds]))
        curve = _sanitize(np.stack([v_curve, i_curve, c_curve, s_curve]))
        curves.append(curve.reshape(1, 4, charge_discharge_len))

    return (np.concatenate(curves, axis=0).astype(np.float32),
            np.asarray(ces, dtype=np.float32), np.asarray(ees, dtype=np.float32))
