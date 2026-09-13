"""Aggregate result JSONs written by the SLURM array, with paired statistics.

Fold difficulty dominates the unpaired std by 4-7x, so the per-fold paired view
is the one that can actually separate models.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from itertools import combinations
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--domain", default="")
    a = ap.parse_args()

    root = Path(a.results)
    doms = [a.domain] if a.domain else sorted(p.name for p in root.iterdir() if p.is_dir())

    for dom in doms:
        rows = [json.loads(p.read_text()) for p in sorted((root / dom).glob("*.json"))]
        rows = [r for r in rows if not r.get("skipped")]
        if not rows:
            print(f"\n{dom}: nothing")
            continue
        by: dict[str, dict[str, float]] = {}
        for r in rows:
            by.setdefault(r["label"], {})[r["group"]] = r["mape_pct"]

        print(f"\n### {dom}")
        print("| model | MAPE (%) | folds |")
        print("| --- | --- | --- |")
        for lab in sorted(by, key=lambda k: st.mean(by[k].values())):
            v = list(by[lab].values())
            sd = st.pstdev(v) if len(v) > 1 else 0.0
            print(f"| `{lab}` | {st.mean(v):.3f}±{sd:.3f} | {len(v)} |")

        print("\npaired per-fold (same test set, so fold difficulty cancels):")
        for x, y in combinations(sorted(by), 2):
            shared = sorted(set(by[x]) & set(by[y]))
            if len(shared) < 2:
                continue
            d = [by[x][g] - by[y][g] for g in shared]
            md = st.mean(d)
            se = (st.stdev(d) / len(d) ** 0.5) if len(d) > 1 else 0.0
            wins_y = sum(1 for v in d if v > 0)
            print(f"  {x:16} - {y:16} {md:+.3f} ± {se:.3f}   "
                  f"{y} better on {wins_y}/{len(shared)} folds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
