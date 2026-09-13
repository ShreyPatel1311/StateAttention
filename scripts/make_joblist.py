"""Write the (domain, model, fold, seed) job list a SLURM array indexes into."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from stateattention.data.batterylife import (  # noqa: E402
    DOMAIN_SPLIT_GROUPS, bmf_split_groups)

# Li-ion has ~600 training cells, so their full prefix ladder is ~60k samples an
# epoch; stride 10 keeps it comparable with the other domains. t_max follows the
# longest trajectory in the domain.
PER_DOMAIN = {
    "li-ion":  dict(prefix_stride=10, t_max=5000, max_censoring=0.5),
    "zn-ion":  dict(prefix_stride=1,  t_max=3000, max_censoring=0.5),
    "na-ion":  dict(prefix_stride=1,  t_max=3000, max_censoring=0.5),
    "calb":    dict(prefix_stride=1,  t_max=3000, max_censoring=1.0),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="li-ion,zn-ion")
    ap.add_argument("--models", default="BatteryMFormer,hx,mambaformer")
    ap.add_argument("--seeds", default="2024")
    ap.add_argument("--out", default="joblist.txt")
    ap.add_argument("--prefix-stride", type=int, default=0,
                    help="override the per-domain stride. Their launch script "
                         "sets seq_len=1, the full prefix ladder on every "
                         "domain; stride 10 on li-ion was our compute "
                         "concession, not their protocol.")
    ap.add_argument("--pooled", action="store_true",
                    help="one BatteryLife cell-level split per domain instead of "
                         "BatteryMFormer's per-aging-condition folds. Removes the "
                         "fold axis, and with it comparability to their published "
                         "numbers, which are fold means under leave-one-out.")
    ap.add_argument("--variants", default="baseline",
                    help="comma-separated: baseline | socview+rateshape | socview")
    a = ap.parse_args()

    VAR_FLAGS = {
        "baseline": "",
        "socview": "--soc-view",
        "rateshape": "--head-type rateshape",
        "socview+rateshape": "--soc-view --head-type rateshape",
    }
    rows = []
    for dom in [d.strip() for d in a.domains.split(",") if d.strip()]:
        opts = PER_DOMAIN[dom]
        groups = ([DOMAIN_SPLIT_GROUPS[dom][0]] if a.pooled
                  else bmf_split_groups(dom))
        for g in groups:
            for m in [x.strip() for x in a.models.split(",") if x.strip()]:
                for s in [int(x) for x in a.seeds.split(",") if x.strip()]:
                    # our architecture flags do not apply to their model, and
                    # tagging its results with them would be misleading
                    vs = ([ "baseline" ] if m == "BatteryMFormer"
                          else [x.strip() for x in a.variants.split(",") if x.strip()])
                    for v in vs:
                        rows.append(
                            f"--domain {dom} --model {m} --group {g} --seed {s} "
                            f"--prefix-stride {a.prefix_stride or opts['prefix_stride']} "
                            f"--t-max {opts['t_max']} "
                            f"--max-censoring {opts['max_censoring']} "
                            f"{VAR_FLAGS[v]}".rstrip())
    Path(a.out).write_text("\n".join(rows) + "\n")
    print(f"{len(rows)} jobs -> {a.out}")
    print(f"submit with:  sbatch --array=1-{len(rows)} scripts/submit.sbatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
