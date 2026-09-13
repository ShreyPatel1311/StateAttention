"""BatteryMFormer's published splits, vendored so our numbers join their Table 2.

Their protocol is aging-condition-exclusive, which is not what BatteryLife's own
split file does, so these are the splits -- not ours -- that make a comparison
with their reported MAPE meaningful.
"""

from __future__ import annotations

import pytest

from stateattention.data.batterylife import (
    BMF_SPLITS_DIR,
    bmf_split_groups,
    load_split_spec,
)

# BatteryMFormer's split files are fetched by scripts/setup.sh rather than
# vendored -- their repository declares no licence, so redistributing its
# contents is not ours to do.
pytestmark = pytest.mark.skipif(
    not any(BMF_SPLITS_DIR.glob("*/*.json")),
    reason="BatteryMFormer splits absent; run scripts/setup.sh to fetch them",
)

# Table 1 of the paper: aging conditions per domain. CALB and Na-ion use
# leave-one-aging-condition-out, so folds == conditions; Li-ion and Zn-ion get
# three random splits instead.
EXPECTED_FOLDS = {"calb": 4, "na-ion": 12, "zn-ion": 3, "li-ion": 3}
# Table 1: batteries per domain (Li-ion's published 963 differs from the 979
# named across their three split files; we assert the split files, not the paper).
EXPECTED_CELLS = {"calb": 27, "na-ion": 31, "zn-ion": 95}


def test_all_published_folds_are_vendored():
    assert len(list(BMF_SPLITS_DIR.glob("*/*.json"))) == 22


@pytest.mark.parametrize("domain,n", sorted(EXPECTED_FOLDS.items()))
def test_fold_counts_match_table_1(domain, n):
    assert len(bmf_split_groups(domain)) == n


@pytest.mark.parametrize("domain", sorted(EXPECTED_FOLDS))
def test_every_fold_is_a_usable_disjoint_split(domain):
    for group in bmf_split_groups(domain):
        sp = load_split_spec(group)
        tr, va, te = (set(sp[s]) for s in ("train", "val", "test"))
        assert tr and va and te, f"{group} has an empty split"
        assert not (tr & va) and not (tr & te) and not (va & te), f"{group} leaks"


@pytest.mark.parametrize("domain", ["calb", "na-ion"])
def test_leave_one_aging_condition_out_partitions_the_cells(domain):
    """Each cell must be tested exactly once across the folds.

    That is the defining property of leave-one-out, and it is what makes their
    reported std a spread over held-out conditions rather than over seeds.
    """
    groups = bmf_split_groups(domain)
    seen: dict[str, int] = {}
    for g in groups:
        for c in load_split_spec(g)["test"]:
            seen[c] = seen.get(c, 0) + 1
    assert set(seen.values()) == {1}, f"{domain}: some cell is tested {sorted(set(seen.values()))} times"
    assert len(seen) == EXPECTED_CELLS[domain]


@pytest.mark.parametrize("domain", sorted(EXPECTED_CELLS))
def test_cell_universe_is_stable_across_folds(domain):
    groups = bmf_split_groups(domain)
    universe = {
        frozenset(sum((load_split_spec(g)[s] for s in ("train", "val", "test")), []))
        for g in groups
    }
    assert len(universe) == 1, f"{domain}: folds disagree on which cells exist"
    assert len(next(iter(universe))) == EXPECTED_CELLS[domain]


def test_calb_conditions_are_temperatures():
    """CALB's 4 aging conditions are its 4 temperatures.

    This is why CALB is degenerate: three of the four groups barely fade, so a
    fold that holds out CALB_0 trains only on near-flat cells. Pinning it here
    keeps that fact visible if the vendored files are ever refreshed.
    """
    temps = set()
    for g in bmf_split_groups("calb"):
        held = {c.split("_")[1] for c in load_split_spec(g)["test"]}
        assert len(held) == 1, f"{g} holds out more than one temperature: {held}"
        temps |= held
    assert temps == {"0", "25", "35", "45"}


def test_unknown_group_names_are_rejected():
    with pytest.raises(KeyError, match="unknown BatteryMFormer split"):
        load_split_spec("bmf:no_such_fold")
    with pytest.raises(KeyError, match="unknown split group"):
        load_split_spec("no_such_group")


def test_partial_coverage_of_a_bmf_fold_is_refused(tmp_path):
    """A fold we can only half-populate must not be run and reported as theirs.

    Li-ion is the live case: their split names 184 Stanford cells and our cache
    holds 38, so ~60% of each Li-ion fold is missing. Silently training on the
    remainder would produce a number that looks comparable to their Table 2 and
    is not.
    """
    import numpy as np

    from stateattention.data.batterylife import cache_key
    from stateattention.data.trajdata import build_splits

    group = "bmf:CALB_loao_cond396_seed2021"
    sp = load_split_spec(group)
    root = tmp_path / "batterylife"
    cdir = root / ".cache" / cache_key(100, 300)
    tdir = root / ".cache" / "soh_trajectories"
    cdir.mkdir(parents=True)
    tdir.mkdir(parents=True)

    # populate every cell except one held-out test cell -> 7/8 test coverage
    for name in sum((sp[s] for s in ("train", "val", "test")), [])[:-1]:
        stem = name.removesuffix(".pkl")
        np.savez_compressed(cdir / f"{stem}.npz",
                            curves=np.random.rand(100, 300, 3).astype(np.float32),
                            eol=500, nominal_capacity=1.1)
        np.savez_compressed(tdir / f"{stem}.npz",
                            soh_raw=np.linspace(1.0, 0.8, 300).astype(np.float32),
                            gap_s=np.zeros(300, dtype=np.float32), gap_usable=False)

    with pytest.raises(ValueError, match="partially covered"):
        build_splits(root, group=group)

    # the escape hatch still lets it run deliberately
    ds, _, _ = build_splits(root, group=group, min_coverage=0.0)
    assert all(len(d) > 0 for d in ds)


def _fake_root(tmp_path, names, soh_min):
    """Build a cache whose cells fade only to `soh_min`."""
    import numpy as np

    from stateattention.data.batterylife import cache_key

    root = tmp_path / "batterylife"
    cdir = root / ".cache" / cache_key(100, 300)
    tdir = root / ".cache" / "soh_trajectories"
    cdir.mkdir(parents=True)
    tdir.mkdir(parents=True)
    for name in names:
        stem = name.removesuffix(".pkl")
        np.savez_compressed(cdir / f"{stem}.npz",
                            curves=np.random.rand(100, 300, 3).astype(np.float32),
                            eol=500, nominal_capacity=1.1)
        np.savez_compressed(tdir / f"{stem}.npz",
                            soh_raw=np.linspace(1.0, soh_min, 300).astype(np.float32),
                            gap_s=np.zeros(300, dtype=np.float32), gap_usable=False)
    return root


def test_censored_targets_are_refused(tmp_path):
    """A target that never reaches EOL must not be scored against their table.

    This is the CALB failure: BatteryLife's public release stops at 99-100
    cycles while its life labels put EOL past 1000, so the SOH target never
    leaves ~1.0 and an untrained constant beats a trained model. Guarding it
    keeps that from being read as a result.
    """
    from stateattention.data.trajdata import build_splits

    group = "bmf:CALB_loao_cond396_seed2021"
    sp = load_split_spec(group)
    names = sum((sp[s] for s in ("train", "val", "test")), [])

    root = _fake_root(tmp_path, names, soh_min=0.996)  # censored, like real CALB
    with pytest.raises(ValueError, match="never reach SOH"):
        build_splits(root, group=group, tau=0.90)

    # a domain whose cells actually reach EOL is fine
    root2 = _fake_root(tmp_path / "ok", names, soh_min=0.80)
    ds, _, _ = build_splits(root2, group=group, tau=0.90)
    assert all(len(d) > 0 for d in ds)


def test_censoring_rate_counts_cells_that_never_fade():
    import numpy as np

    from stateattention.data.trajdata import censoring_rate

    pool = {
        "a": {"soh_raw": np.linspace(1.0, 0.75, 50)},   # reaches 0.80
        "b": {"soh_raw": np.linspace(1.0, 0.99, 50)},   # censored
        "c": {"soh_raw": np.linspace(1.0, 0.995, 50)},  # censored
    }
    assert censoring_rate(pool, ["a.pkl", "b.pkl", "c.pkl"], tau=0.80) == pytest.approx(2 / 3)
    assert censoring_rate(pool, ["a.pkl"], tau=0.80) == 0.0
    assert censoring_rate({}, ["a.pkl"], tau=0.80) == 1.0
