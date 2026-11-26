#!/bin/bash
# E4: ablation DAMR (Bảng 3, 4 của bài báo), 3 seed chung, có ghi log mở rộng.
# Cách dùng (chạy từ gốc repo):  bash scripts/run_e4.sh PSM [GPU_ID] [EPOCHS]
#   Chọn biến thể: VARIANTS="wo_MSASB fixed_tau_0.25" bash scripts/run_e4.sh PSM 0
# Mọi biến thể dùng đúng learning rate và siêu tham số như E2; chỉ khác cờ trong bảng VARIANT_FLAGS.
# Bản "full" chính là E2 (seed 16, 17, 18), không chạy lại ở đây.
set -euo pipefail
DATASET=${1:?Cần tên dataset: PSM | MSL | SMD}
GPU=${2:-0}
EPOCHS=${3:-}
SEEDS=${SEEDS:-"16 17 18"}
VARIANTS=${VARIANTS:-"wo_MSASB wo_MSASB_G wo_MSASB_L fixed_tau_0.25 wo_temp_multihead"}
# tau dùng cho biến thể bỏ cả temperature và multi-head. CẦN ĐỐI CHIẾU BÀI BÁO (docs/code_vs_paper.md mục 12).
TAU_NOTEMP=${TAU_NOTEMP:-0.25}

declare -A VARIANT_FLAGS=(
  [wo_MSASB]="--MSASB=False"                                  # bỏ cả khối MSASB
  [wo_MSASB_G]="--global_filter=False"                        # bỏ nhánh lọc toàn cục (giả định G = global)
  [wo_MSASB_L]="--adaptive_filter=False"                      # bỏ nhánh lọc thích nghi (giả định L = local/adaptive)
  [fixed_tau_0.25]="--fixed_temp=0.25"                        # tau cố định thay cho AdaptiveTemperature
  [wo_temp_multihead]="--num_heads=1 --fixed_temp=$TAU_NOTEMP" # 1 head, tau cố định
)

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
mkdir -p "$ROOT/logs" "$ROOT/results" "$ROOT/checkpoints/E4"

cd "$ROOT/DAMR"
for V in $VARIANTS; do
  FLAGS=${VARIANT_FLAGS[$V]:?Biến thể lạ: $V}
  for SEED in $SEEDS; do
    echo "=== E4 $DATASET $V seed $SEED (lr=$LR, $FLAGS) ==="
    # shellcheck disable=SC2086
    python3 main.py --dataset="$DATASET" --data_path="./data/$DATA_DIR" --seed="$SEED" \
      --device="cuda:$GPU" --lr="$LR" ${EPOCHS:+--epochs=$EPOCHS} $FLAGS \
      --log_extra=True --save_dir="$ROOT/checkpoints/E4" --variant="$V" \
      --results_csv="$ROOT/results/E4_DAMR_ablation.csv" \
      2>&1 | tee "$ROOT/logs/E4_${DATASET}_${V}_seed${SEED}.log"
  done
done
