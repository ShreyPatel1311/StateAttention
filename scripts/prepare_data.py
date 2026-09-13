"""Download BatteryLife and build every cache the models need.

Three caches come out of this, all under <root>/.cache/:

    c100_l300/         our 3-channel curves, uniform resampling  (input to hx)
    bmf_c100_l300/     4-channel SOC-aligned curves              (input to their model)
    soh_trajectories/  SOH target per cell, to end of life

Only the derived caches are needed for training (~600 MB); the raw archives are
~31 GB and can be deleted afterwards with --clean-raw.

    python scripts/prepare_data.py --root data/batterylife --domains zn-ion,na-ion
    python scripts/prepare_data.py --root data/batterylife --domains all --clean-raw
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import quote

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from stateattention.data.batterylife import (  # noqa: E402
    bmf_split_groups, build_cell, cache_key, load_split_spec,
)
from stateattention.data.soc_curves import prepare_curves  # noqa: E402
from stateattention.data.trajectory import extract_trajectory  # noqa: E402

ZENODO_RECORD = "19688272"
ZENODO = "https://zenodo.org/api/records/{rec}/files/{name}/content"

# Which Zenodo archives each domain needs. Li-ion spans 13 source datasets.
ARCHIVES = {
    "zn-ion": ["ZN-coin.zip"],
    "na-ion": ["NA-ion.zip"],
    "calb": ["CALB.zip"],
    "li-ion": ["ISU_ILCC.zip", "Stanford.zip", "Stanford_2.zip", "MATR.zip",
               "Tongji.zip", "HUST.zip", "RWTH.zip", "XJTU.zip", "MICH.zip",
               "MICH_EXP.zip", "SNL.zip", "CALCE.zip", "HNEI.zip", "UL_PUR.zip"],
}


def fetch(raw: Path, member: str) -> None:
    stem = member.removesuffix(".zip")
    if (raw / stem).exists() and any((raw / stem).rglob("*.pkl")):
        print(f"  {stem}: present")
        return
    raw.mkdir(parents=True, exist_ok=True)
    tmp = raw / (member + ".part")
    print(f"  {stem}: downloading …", flush=True)
    with urllib.request.urlopen(ZENODO.format(rec=ZENODO_RECORD,
                                              name=quote(member))) as r, \
            open(tmp, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    with zipfile.ZipFile(tmp) as z:
        z.extractall(raw)
    tmp.unlink()
    print(f"  {stem}: {len(list((raw / stem).rglob('*.pkl')))} cells")


def cells_for(domain: str) -> list[str]:
    """Every cell BatteryMFormer's folds name for this domain."""
    out: set[str] = set()
    for g in bmf_split_groups(domain):
        sp = load_split_spec(g)
        out |= {c for s in ("train", "val", "test") for c in sp[s]}
    return sorted(out)


def build(root: Path, domain: str, max_cycles: int = 100, curve_len: int = 300) -> None:
    raw = root / "raw"
    index = {p.name: p for p in raw.rglob("*.pkl")}
    cdir = root / ".cache" / cache_key(max_cycles, curve_len)
    bdir = root / ".cache" / "bmf_c100_l300"
    tdir = root / ".cache" / "soh_trajectories"
    for d in (cdir, bdir, tdir):
        d.mkdir(parents=True, exist_ok=True)

    names = cells_for(domain)
    missing = [n for n in names if n not in index]
    if missing:
        print(f"  WARNING: {len(missing)} cells have no raw file, e.g. {missing[:3]}")

    made = {"curves": 0, "soc": 0, "traj": 0}
    for i, name in enumerate(names, 1):
        src = index.get(name)
        if src is None:
            continue
        stem = name.removesuffix(".pkl")

        if not (tdir / f"{stem}.npz").exists():
            r = extract_trajectory(src)
            soh = r["soh_raw"]
            if len(soh) >= 10 and np.isfinite(soh).any():
                np.savez_compressed(tdir / f"{stem}.npz", soh_raw=soh,
                                    gap_s=r["gap_s"], nominal=r["nominal"],
                                    dod=r["dod"], dod_source=r["dod_source"],
                                    gap_usable=r["gap_usable"])
                made["traj"] += 1

        if not (cdir / f"{stem}.npz").exists():
            with open(src, "rb") as f:
                d = pickle.load(f)
            cell = build_cell(src, len(d["cycle_data"]), max_cycles, curve_len)
            np.savez_compressed(cdir / f"{stem}.npz", curves=cell.curves,
                                eol=cell.eol, nominal_capacity=cell.nominal_capacity)
            made["curves"] += 1

        if not (bdir / f"{stem}.npz").exists():
            with open(src, "rb") as f:
                d = pickle.load(f)
            lo, hi = d["SOC_interval"][0], d["SOC_interval"][1]
            curves, ce, ee = prepare_curves(
                name, d["cycle_data"], d["nominal_capacity_in_Ah"], lo, hi,
                charge_discharge_len=curve_len, early_cycle_threshold=max_cycles,
                seq_len=1)
            np.savez_compressed(bdir / f"{stem}.npz", curves=curves, ce=ce, ee=ee,
                                nominal=float(d["nominal_capacity_in_Ah"]),
                                n_cycles=len(d["cycle_data"]),
                                soc_interval=np.array([lo, hi], dtype=np.float32))
            made["soc"] += 1

        if i % 50 == 0:
            print(f"    {i}/{len(names)}", flush=True)
    print(f"  built: {made['curves']} curve, {made['soc']} SOC-aligned, "
          f"{made['traj']} trajectory")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/batterylife")
    ap.add_argument("--domains", default="zn-ion,na-ion,calb",
                    help="comma-separated, or 'all'. li-ion needs ~28 GB of archives.")
    ap.add_argument("--clean-raw", action="store_true",
                    help="delete the raw archives once the caches are built")
    a = ap.parse_args()

    doms = (list(ARCHIVES) if a.domains == "all"
            else [d.strip() for d in a.domains.split(",") if d.strip()])
    root = Path(a.root)

    for dom in doms:
        print(f"\n=== {dom} ===")
        for m in ARCHIVES[dom]:
            fetch(root / "raw", m)
        build(root, dom)

    if a.clean_raw:
        shutil.rmtree(root / "raw", ignore_errors=True)
        print("\nraw archives removed; the caches are all training needs")
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
