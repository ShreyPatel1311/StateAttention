#!/usr/bin/env bash
# Fetch BatteryMFormer's model code, for the baseline column only.
#
# Their repository (github.com/Ruifeng-Tan/BatteryMFormer) declares no licence,
# so this project does not redistribute any of it. This script downloads it into
# third_party/ on your machine, which is gitignored. Check their terms before
# using it for anything beyond reproducing the comparison.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="https://raw.githubusercontent.com/Ruifeng-Tan/BatteryMFormer/master"
DEST="$REPO/third_party/batterymformer"
mkdir -p "$DEST"/{models,layers,utils}

for f in models/BatteryMFormer.py layers/Embed.py layers/SelfAttention_Family.py \
         layers/Transformer_EncDec.py layers/StandardNorm.py layers/__init__.py \
         utils/masking.py utils/__init__.py; do
    curl -sfL "$RAW/$f" -o "$DEST/$f" && echo "  $f"
done
touch "$DEST/models/__init__.py"

# Their aging-condition embeddings, keyed by cell filename (4.7 MB).
curl -sfL "$RAW/data_provider/prompt_embeddings/Qwen3_total.pkl" \
     -o "$DEST/Qwen3_total.pkl" && echo "  Qwen3_total.pkl"

# Their attention module imports reformer_pytorch for an LSH variant the model
# does not use, but the import is unconditional.
pip install --quiet reformer_pytorch
echo
echo "baseline available. Add --models hx,BatteryMFormer to make_joblist.py."
