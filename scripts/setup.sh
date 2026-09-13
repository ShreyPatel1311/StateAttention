#!/usr/bin/env bash
# One-time setup on a SLURM cluster.
#
#   export WORK=$HOME/stateattention
#   bash scripts/setup.sh
#
# Fetches BatteryMFormer's split definitions and, optionally, their model code
# for the baseline comparison. Their repository declares no licence, so nothing
# from it is redistributed here -- it is fetched at setup time instead.
set -euo pipefail

: "${WORK:=$HOME/stateattention}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BMF_RAW="https://raw.githubusercontent.com/Ruifeng-Tan/BatteryMFormer/master"

mkdir -p "$WORK"
cd "$WORK"

echo "== python environment =="
module purge 2>/dev/null || true
module load cuda/12.3.0/gcc/11.3.0/zen2 2>/dev/null || module load cuda 2>/dev/null || true
if [ ! -d .venv ]; then
    python3.11 -m venv .venv || python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet --index-url https://download.pytorch.org/whl/cu121 torch
pip install --quiet numpy scipy pandas einops

echo "== BatteryMFormer split definitions =="
# The evaluation protocol: which cells are train/val/test in each
# aging-condition fold. Without these the folds cannot be reproduced.
SPLITS="$REPO/src/stateattention/data/splits/batterymformer"
mkdir -p "$SPLITS/loao" "$SPLITS/pure_ood"
fetch() { curl -sfL "$BMF_RAW/data_provider/split_json/$1" -o "$SPLITS/$1" && echo "   $1"; }
for s in 2021 2024 42; do
    fetch "pure_ood/Liion_split_seed${s}.json"
    fetch "pure_ood/ZNcoin_split_seed${s}.json"
done
for c in 396 397 398 399; do fetch "loao/CALB_loao_cond${c}_seed2021.json"; done
for c in $(seq 400 411); do fetch "loao/NA-ion_loao_cond${c}_seed2021.json"; done
echo "   $(find "$SPLITS" -name '*.json' | wc -l) split files (expect 22)"

echo
echo "== optional: BatteryMFormer baseline =="
echo "   Their model is NOT redistributed here (their repo declares no licence)."
echo "   To reproduce the baseline column, fetch it yourself:"
echo "     bash scripts/fetch_baseline.sh"
echo
echo "== next =="
echo "   python scripts/prepare_data.py --root $WORK/data/batterylife --domains zn-ion"
echo "   python scripts/make_joblist.py --domains zn-ion --out joblist.txt"
echo "   sbatch --array=1-\$(wc -l < joblist.txt) scripts/submit.sbatch"
