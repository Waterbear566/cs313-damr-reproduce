#!/usr/bin/env bash
# Chạy lại DeepSVDD (2 biến thể) và ALOCC theo pipeline chung, seed 0/1/2, PSM -> MSL -> SMD.
# Chạy từ gốc repo:  bash scripts/run_e3a_rerun.sh [dataset ...]     (mặc định: PSM MSL SMD)
# Lượt đã có trong CSV sẽ được bỏ qua, nên chạy lại sau khi bị ngắt là an toàn.
set -euo pipefail
export PYTHONWARNINGS="ignore::UserWarning"
DATASETS=("${@:-PSM MSL SMD}")
mkdir -p logs
for ds in ${DATASETS[@]}; do
  for seed in 0 1 2; do
    for variant in full bn_train_each_epoch; do
      python baselines/DeepSVDD/run.py --dataset "$ds" --seed "$seed" --variant "$variant" 2>&1 \
        | tee -a "logs/E3_DeepSVDD_${variant}_${ds}_seed${seed}.log"
    done
    python baselines/ALOCC/run.py --dataset "$ds" --seed "$seed" 2>&1 | tee -a "logs/E3_ALOCC_${ds}_seed${seed}.log"
  done
done
