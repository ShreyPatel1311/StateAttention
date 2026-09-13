"""Train one (model, fold) on a local machine or cluster node.

Intended to be driven by a SLURM array (see submit.sbatch), but runs standalone
on any machine with a GPU.

    python scripts/run_fold.py --domain li-ion --model hx \
        --group bmf:Liion_split_seed2021 --data-root $DATA --out $RESULTS
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--model", required=True,
                    help="BatteryMFormer | hx | mambaformer | constant | trainmean")
    ap.add_argument("--group", required=True, help="e.g. bmf:Liion_split_seed2021")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--data-root", required=True,
                    help="dir containing .cache/c100_l300 and .cache/soh_trajectories")
    ap.add_argument("--bmf-root", default="",
                    help="dir of 4-channel SOC-aligned curves (default: <data>/.cache/bmf_c100_l300)")
    ap.add_argument("--out", required=True, help="directory for result JSON")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--t-max", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--prefix-aug", action="store_true", default=True)
    ap.add_argument("--no-prefix-aug", dest="prefix_aug", action="store_false")
    ap.add_argument("--their-inputs", action="store_true", default=True)
    ap.add_argument("--no-their-inputs", dest="their_inputs", action="store_false")
    ap.add_argument("--prefix-stride", type=int, default=1)
    ap.add_argument("--max-censoring", type=float, default=0.5)
    ap.add_argument("--embed-type", default="flatten",
                    help="flatten | patch | patchattn | conv | delta")
    ap.add_argument("--ssm-core", default="diagonal", help="diagonal | s6")
    ap.add_argument("--head-type", default="free", help="free | monotone | rateshape")
    ap.add_argument("--soc-view", action="store_true", default=False)
    a = ap.parse_args()

    from stateattention.bmf_eval import run_one

    data_root = Path(a.data_root)
    bmf_root = Path(a.bmf_root) if a.bmf_root else data_root / ".cache" / "bmf_c100_l300"
    bmf_repo = REPO / "third_party" / "batterymformer"
    emb_path = bmf_repo / "Qwen3_total.pkl"
    if a.model == "BatteryMFormer" and not emb_path.exists():
        print(f"ERROR: their aging-condition embeddings are missing at {emb_path}.\n"
              "Fetch with:\n  curl -sfL https://raw.githubusercontent.com/"
              "Ruifeng-Tan/BatteryMFormer/master/data_provider/prompt_embeddings/"
              "Qwen3_total.pkl -o " + str(emb_path), file=sys.stderr)
        return 2

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cfgd = dict(epochs=a.epochs, patience=a.patience, d_model=a.d_model, lr=a.lr,
                batch_size=a.batch_size, t_max=a.t_max, stride=a.stride,
                prefix_aug=a.prefix_aug, their_inputs=a.their_inputs,
                prefix_stride=a.prefix_stride, max_censoring=a.max_censoring,
                embed_type=a.embed_type, ssm_core=a.ssm_core,
                head_type=a.head_type, soc_view=a.soc_view)

    res = run_one(a.domain, a.model, a.group, a.seed, cfgd,
                  data_root=data_root, bmf_root=bmf_root, emb_path=emb_path,
                  bmf_repo=bmf_repo, results_dir=out)
    print(json.dumps({k: v for k, v in res.items() if k != "protocol"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
