#!/usr/bin/env bash
#SBATCH --job-name=stateattention
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# ---- set these for your allocation ----
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --account=<your_account>
##SBATCH --qos=<your_qos>
# ---------------------------------------
set -euo pipefail

: "${WORK:=$HOME/stateattention}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="$WORK/data/batterylife"
RESULTS="$WORK/results"
JOBLIST="${JOBLIST:-$WORK/joblist.txt}"

mkdir -p logs "$RESULTS"
module purge
module load cuda/12.3.0/gcc/11.3.0/zen2 || module load cuda || true
source "$WORK/.venv/bin/activate"

ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBLIST")
if [ -z "$ARGS" ]; then
  echo "no job at index ${SLURM_ARRAY_TASK_ID}"; exit 0
fi

DOMAIN=$(echo "$ARGS" | sed -n 's/.*--domain \([^ ]*\).*/\1/p')
echo "[$(date +%T)] task ${SLURM_ARRAY_TASK_ID}: $ARGS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Each run writes its own JSON before returning, so a preempted or timed-out
# array element never costs the runs that already finished.
srun python "$REPO/scripts/run_fold.py" \
  $ARGS \
  --data-root "$DATA" \
  --out "$RESULTS/$DOMAIN" \
  --epochs "${EPOCHS:-80}" \
  --patience "${PATIENCE:-25}" \
  --batch-size "${BATCH:-128}"
