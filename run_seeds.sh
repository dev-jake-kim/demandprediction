#!/usr/bin/env bash
# merged 모델 다중 시드 실행기.
#
#   ./run_seeds.sh <dataset> <loss_type> [seed ...]
#
# GPU는 CUDA_VISIBLE_DEVICES로 고른다.
set -euo pipefail

DATASET="${1:?dataset(ulsan|porto)이 필요함}"
LOSS_TYPE="${2:?loss_type(combined|mae)이 필요함}"
shift 2
if [ "$#" -eq 0 ]; then
  SEEDS=(245 6835 851 5123 535)  # 저장소 공용 5시드
else
  SEEDS=("$@")
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

for seed in "${SEEDS[@]}"; do
  out="output/merged/runs/${DATASET}_${LOSS_TYPE}_seed${seed}.json"
  log="output/merged/logs/${DATASET}_${LOSS_TYPE}_seed${seed}.log"
  if [ -f "$out" ]; then
    echo "[skip] $out 이미 있음"
    continue
  fi
  mkdir -p output/merged/runs output/merged/logs
  echo "[run ] dataset=$DATASET loss=$LOSS_TYPE seed=$seed -> $out"
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n DA \
    python train_merged.py \
      dataset="$DATASET" train.seed="$seed" model.loss_type="$LOSS_TYPE" \
      ablation=full run_json="$out" > "$log" 2>&1
  echo "[done] seed=$seed $(python3 -c "
import json;d=json.load(open('$out'));t=d['test']
print('best_ep=%d mae=%.5f rmse=%.5f mape+1=%.3f mape0x=%.3f' % (d['best_epoch'],t['mae'],t['rmse'],t['mape_plus1'],t['mape_excl_zero']))")"
done
