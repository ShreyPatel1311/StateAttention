"""BatteryLife (KDD 2025) loader.

Task, following the paper: given the first S <= 100 cycles of (voltage, current,
capacity), predict the cycle number at which SOH first drops to <= 80%. Every
cell yields ~100 samples, one per prefix length, which is how 990 cells become
99k samples.

We reuse the paper's official cell-level train/val/test split lists verbatim
(vendored in splits/batterylife_official.json) so results are comparable to
their published table. Curve extraction here is our own clean reimplementation,
not a byte-for-byte copy of their preprocessing -- see NOTE below.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .download import DOMAIN_SUBSETS

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - data-prep-only environments
    # Building the npz cache needs numpy and nothing else. Keeping torch
    # optional here lets a cache job run in a ~200 MB image instead of ~1.5 GB.
    class Dataset:  # type: ignore[no-redef]
        pass

SPLITS_FILE = Path(__file__).parent / "splits" / "batterylife_official.json"

# BatteryMFormer's own published split files, vendored from
# github.com/Ruifeng-Tan/BatteryMFormer @ master (data_provider/split_json).
# Training on these puts our numbers in the same table as their Table 2 without
# reproducing their code. Their protocol is aging-condition-exclusive: Li-ion
# and Zn-ion get three random splits at a 6:2:2 aging-condition ratio, while
# CALB (4 conditions) and Na-ion (12) use leave-one-aging-condition-out.
BMF_SPLITS_DIR = Path(__file__).parent / "splits" / "batterymformer"
BMF_PREFIX = "bmf:"


def bmf_split_groups(domain: str) -> list[str]:
    """Every BatteryMFormer split group for a domain, in published order.

    Their mean/std is taken over these folds, not over seeds -- for CALB that
    is 4 held-out aging conditions, which is why their CALB std is so wide.
    """
    stem = {"calb": "loao/CALB_loao_cond*", "na-ion": "loao/NA-ion_loao_cond*",
            "zn-ion": "pure_ood/ZNcoin_split_seed*",
            "li-ion": "pure_ood/Liion_split_seed*"}[domain]
    return [BMF_PREFIX + p.stem for p in sorted(BMF_SPLITS_DIR.glob(stem + ".json"))]


def load_split_spec(group: str) -> dict[str, list[str]]:
    """Resolve a split-group name to {train, val, test} lists of cell names.

    A `bmf:` prefix selects one of BatteryMFormer's published folds; anything
    else is a group in BatteryLife's own vendored split file.
    """
    if not group.startswith(BMF_PREFIX):
        groups = json.loads(SPLITS_FILE.read_text())
        if group not in groups:
            raise KeyError(f"unknown split group {group!r}; have {sorted(groups)}")
        return groups[group]

    stem = group[len(BMF_PREFIX):]
    hits = sorted(BMF_SPLITS_DIR.glob(f"*/{stem}.json"))
    if not hits:
        avail = sorted(p.stem for p in BMF_SPLITS_DIR.glob("*/*.json"))
        raise KeyError(f"unknown BatteryMFormer split {stem!r}; have {avail}")
    d = json.loads(hits[0].read_text())
    return {s: list(d[s]) for s in ("train", "val", "test")}

# Which vendored split group each domain uses, per seed. Li-ion keeps one fixed
# split across seeds; the three small domains resample the split per seed, which
# is why their published std is so much wider.
DOMAIN_SPLIT_GROUPS: dict[str, tuple[str, ...]] = {
    "li-ion": ("MIX_large", "MIX_large", "MIX_large"),
    "zn-ion": ("ZNcoin", "ZN_42", "ZN_2024"),
    "na-ion": ("NAion_2021", "NAion_42", "NAion_2024"),
    "calb": ("CALB", "CALB_42", "CALB_2024"),
}

# Cells whose protocol discharges before charging. Getting this backwards
# silently mangles the curve, so it is worth carrying the list over explicitly.
DISCHARGE_FIRST_PREFIXES = ("RWTH", "CALB_0", "CALB_25", "CALB_45")

RAW_KEYS = (
    "current_in_A",
    "voltage_in_V",
    "charge_capacity_in_Ah",
    "discharge_capacity_in_Ah",
    "time_in_s",
)


@dataclass
class Cell:
    name: str
    curves: np.ndarray  # [n_cycles, curve_len, 3] float32
    eol: int
    nominal_capacity: float


def cache_key(max_cycles: int, curve_len: int) -> str:
    """Cache directory name. Both parameters change the stored arrays, so both
    are in the key -- bumping either must not silently reuse stale npz files."""
    return f"c{max_cycles}_l{curve_len}"


def load_life_labels(root: Path, required: bool = True) -> dict[str, int]:
    """Merge every *_labels.json under 'Life labels/' into one filename -> eol map.

    The upstream loader picks a label file per dataset prefix with a handful of
    special cases; merging sidesteps that entirely since keys are unique.

    With `required=False`, a missing directory yields an empty map rather than
    raising. That is the right behaviour when training from a downloaded npz
    cache: labels are baked into the cache, and are only needed to build it.
    """
    labels: dict[str, int] = {}
    label_dir = root / "Life labels"
    if not label_dir.exists():
        if not required:
            return labels
        raise FileNotFoundError(f"missing {label_dir} -- run `mf download --dataset batterylife`")
    for jf in sorted(label_dir.glob("*.json")):
        labels.update(json.loads(jf.read_text()))
    # Tongji filenames use '-#' in the label file but '--' on disk.
    for k in list(labels):
        if "-#" in k:
            labels.setdefault(k.replace("-#", "--"), labels[k])
    return labels


def _resample(x: np.ndarray, n: int) -> np.ndarray:
    """Linear resample a 1-D segment to exactly n points."""
    if x.size == 0:
        return np.zeros(n, dtype=np.float32)
    if x.size == 1:
        return np.full(n, x[0], dtype=np.float32)
    src = np.linspace(0.0, 1.0, x.size)
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, x).astype(np.float32)


def _cycle_to_curve(cycle: dict, nominal_capacity: float, half: int, discharge_first: bool):
    """One cycle -> [2*half, 3] array of (voltage, current_in_C, capacity).

    Charge and discharge segments are resampled separately and concatenated, so
    the model always sees the two phases at fixed, aligned positions regardless
    of how long the raw cycle ran.
    """
    v = np.asarray(cycle["voltage_in_V"], dtype=np.float64)
    i = np.asarray(cycle["current_in_A"], dtype=np.float64)
    qc = np.asarray(cycle["charge_capacity_in_Ah"], dtype=np.float64)
    qd = np.asarray(cycle["discharge_capacity_in_Ah"], dtype=np.float64)

    if v.size == 0:
        return None
    i_c = i / max(nominal_capacity, 1e-8)  # current in C-rate

    chg = np.nonzero(i_c >= 0.01)[0]
    dis = np.nonzero(i_c <= -0.01)[0]
    if chg.size == 0 or dis.size == 0:
        return None

    if discharge_first:
        d_end = dis[-1]
        d_sl, c_sl = slice(0, d_end), slice(d_end, None)
    else:
        c_end = chg[-1]
        c_sl, d_sl = slice(0, c_end), slice(c_end, None)

    def seg(sl, q):
        return np.stack(
            [_resample(v[sl], half), _resample(i_c[sl], half), _resample(q[sl], half)], axis=-1
        )

    return np.concatenate([seg(c_sl, qc), seg(d_sl, qd)], axis=0).astype(np.float32)


def build_cell(
    pkl_path: Path, eol: int, max_cycles: int, curve_len: int
) -> Cell | None:
    """Read one processed .pkl and turn its early cycles into a fixed-shape array."""
    name = pkl_path.stem
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Two cells carry a nominal capacity that disagrees with their metadata.
    if name.startswith("RWTH"):
        nominal = 1.85
    elif name.startswith("SNL_18650_NCA_25C_20-80"):
        nominal = 3.2
    else:
        nominal = float(data["nominal_capacity_in_Ah"])

    discharge_first = name.startswith(DISCHARGE_FIRST_PREFIXES)
    half = curve_len // 2

    curves = []
    for cyc in data["cycle_data"][:max_cycles]:
        c = _cycle_to_curve(cyc, nominal, half, discharge_first)
        if c is None:
            break
        curves.append(c)
    if len(curves) < 2:
        return None

    arr = np.stack(curves)  # [n, curve_len, 3]
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return Cell(name=name, curves=arr, eol=int(eol), nominal_capacity=nominal)


class BatteryLifeDataset(Dataset):
    """Samples of (prefix of cycles, mask, cycle life) for one split.

    NOTE ON PARITY: the split lists are the paper's. The curve extraction is our
    own -- simpler than upstream's per-chemistry special-casing. Expect small
    deviations from their absolute MAPE. Before claiming you beat a published
    number, re-run at least one baseline through this pipeline and compare
    against that, not against the paper's table alone.
    """

    def __init__(
        self,
        root: str | Path,
        domain: str = "li-ion",
        split: str = "train",
        seed_idx: int = 0,
        max_cycles: int = 100,
        min_cycles: int = 1,
        curve_len: int = 300,
        cache: bool = True,
        normalize_capacity: bool = True,
        max_cells: int = 0,
        sample_stride: int = 1,
    ):
        self.root = Path(root)
        self.domain = domain
        self.split = split
        self.max_cycles = max_cycles
        self.min_cycles = min_cycles
        self.curve_len = curve_len
        self.normalize_capacity = normalize_capacity
        self.sample_stride = max(1, sample_stride)

        group = DOMAIN_SPLIT_GROUPS[domain][seed_idx % len(DOMAIN_SPLIT_GROUPS[domain])]
        all_splits = json.loads(SPLITS_FILE.read_text())
        self.files: list[str] = all_splits[group][split]
        if max_cells:
            # Stride rather than truncate: the split lists are grouped by source
            # dataset, so taking the first N would sample one chemistry only.
            step = max(1, len(self.files) // max_cells)
            self.files = self.files[::step][:max_cells]

        self.cells = self._load_cells(cache)
        self.index = self._build_index()

    # -- loading -----------------------------------------------------------
    def _find(self, fname: str) -> Path | None:
        for sub in DOMAIN_SUBSETS[self.domain]:
            p = self.root / sub / fname
            if p.exists():
                return p
        hits = list(self.root.glob(f"*/{fname}"))
        return hits[0] if hits else None

    def _load_cells(self, cache: bool) -> list[Cell]:
        cache_dir = self.root / ".cache" / cache_key(self.max_cycles, self.curve_len)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Labels are only needed to build the cache. Loading them lazily means a
        # bare cache pulled from the Hub is enough to train from -- no raw .pkl
        # and no "Life labels" directory required.
        labels: dict[str, int] | None = None

        cells, missing = [], []
        for fname in self.files:
            stem = fname.removesuffix(".pkl")
            cp = cache_dir / f"{stem}.npz"
            if cache and cp.exists():
                z = np.load(cp)
                cells.append(
                    Cell(stem, z["curves"], int(z["eol"]), float(z["nominal_capacity"]))
                )
                continue

            # Not cached: fall back to building from raw .pkl, which needs
            # labels. If there are none (cache-only checkout), skip the cell --
            # six MICH_EXP cells are listed in the splits but were never
            # labelled upstream, so this path is reached on a healthy cache.
            if labels is None:
                labels = load_life_labels(self.root, required=False)
            eol = labels.get(fname, labels.get(stem))
            if eol is None:
                missing.append(fname)  # cell never reached EOL -> no label, skip
                continue
            path = self._find(fname)
            if path is None:
                missing.append(fname)
                continue

            cell = build_cell(path, int(eol), self.max_cycles, self.curve_len)
            if cell is None:
                missing.append(fname)
                continue
            if cache:
                np.savez_compressed(
                    cp,
                    curves=cell.curves,
                    eol=cell.eol,
                    nominal_capacity=cell.nominal_capacity,
                )
            cells.append(cell)

        if missing:
            print(f"[batterylife] {self.domain}/{self.split}: skipped {len(missing)} cells")
        if not cells:
            raise RuntimeError(
                f"no cells loaded for {self.domain}/{self.split}. "
                "Did the download finish, and does data.root point at data/batterylife?"
            )

        if self.normalize_capacity:
            # The cache stores capacity in raw Ah, which spans three orders of
            # magnitude across chemistries -- a MATR Li-ion cell holds 1.1 Ah, a
            # ZN-coin cell 0.00044 Ah. Dividing by nominal capacity makes the
            # channel a dimensionless state-of-charge, matching what we already
            # do for current (C-rate) and making cross-domain transfer possible
            # at all. Done at load time so the cache stays raw and re-scalable.
            for c in cells:
                c.curves[..., 2] /= max(c.nominal_capacity, 1e-12)

        return cells

    def _build_index(self) -> list[tuple[int, int]]:
        """One sample per (cell, prefix length), matching the paper's expansion."""
        idx = []
        for ci, cell in enumerate(self.cells):
            if cell.eol <= self.min_cycles:
                continue
            upper = min(self.max_cycles, cell.curves.shape[0], cell.eol - 1)
            for n in range(self.min_cycles, upper + 1, self.sample_stride):
                idx.append((ci, n))
        return idx

    # -- torch API ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        ci, n = self.index[i]
        cell = self.cells[ci]

        x = np.zeros((self.max_cycles, self.curve_len, 3), dtype=np.float32)
        x[:n] = cell.curves[:n]
        mask = np.zeros(self.max_cycles, dtype=np.float32)
        mask[:n] = 1.0

        return {
            "x": x,
            "mask": mask,
            "y": np.float32(cell.eol),
            "n_cycles": np.int64(n),
            "cell": ci,
        }

    @property
    def labels(self) -> np.ndarray:
        return np.array([self.cells[ci].eol for ci, _ in self.index], dtype=np.float64)
