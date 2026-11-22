#!/bin/bash
# E2: DAMR đúng như code công bố, 3 seed, có ghi log mở rộng.
# Cách dùng (chạy từ gốc repo):  bash scripts/run_e2.sh PSM [GPU_ID] [EPOCHS]
#   Không truyền EPOCHS thì dùng mặc định của code gốc (40).
set -euo pipefail
DATASET=${1:?Cần tên dataset: PSM | MSL | SMD}
GPU=${2:-0}
EPOCHS=${3:-}
SEEDS=${SEEDS:-"0 1 2"}

case $DATASET in        # learning rate tác giả dùng trong DAMR/run/run_*.sh
  PSM) LR=1e-4 ;;
  MSL) LR=5e-4 ;;
  SMD) LR=1e-6 ;;
  SWaT) LR=5e-5 ;;
  WADI) LR=1e-4 ;;
  *) echo "Dataset lạ: $DATASET"; exit 1 ;;
esac
DATA_DIR=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/logs" "$ROOT/results" "$ROOT/checkpoints/E2"

cd "$ROOT/DAMR"
for SEED in $SEEDS; do
  echo "=== E2 $DATASET seed $SEED (lr=$LR) ==="
  python3 main.py --dataset="$DATASET" --data_path="./data/$DATA_DIR" --seed="$SEED" \
    --device="cuda:$GPU" --lr="$LR" ${EPOCHS:+--epochs=$EPOCHS} \
    --log_extra=True --save_dir="$ROOT/checkpoints/E2" --variant=full \
    --results_csv="$ROOT/results/E2_DAMR.csv" \
    2>&1 | tee "$ROOT/logs/E2_${DATASET}_seed${SEED}.log"
done
