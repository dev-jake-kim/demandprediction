#!/usr/bin/env bash
# merged 모델 ablation 큐 러너.


#   ./run_ablation.sh build      # 큐 파일 생성(우선순위 순)
#   ./run_ablation.sh worker <gpu> <slot>

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
QUEUE="output/merged/.ablation_queue"
LOCK="output/merged/.ablation_queue.lock"

SEEDS=(245 6835 851)
# 중단 시 앞쪽 ablation부터 완료된다.
ABLATIONS=(no-ir no-periodic no-extra no-branch-attn no-neighbors)

# ablation 이름에 대응하는 Hydra 오버라이드.
declare -A ABLATION_OVERRIDES=(
  [full]=""
  [no-ir]="model.use_retrieval=false"
  [no-periodic]="model.use_daily=false model.use_weekly=false"
  [no-daily]="model.use_daily=false"
  [no-weekly]="model.use_weekly=false"
  [no-weather]="model.use_weather=false"
  [no-calendar]="model.use_calendar=false"
  [no-extra]="model.use_weather=false model.use_calendar=false"
  [no-branch-attn]="model.use_branch_attention=false"
  # 이 ablation은 인코더의 이웃 공간 정보만 끈다.
  [no-neighbors]="model.use_neighbors=false"
  # weather_injection 변형.
  [weather-cls-add]="model.weather_injection=cls_add"
  # softplus 양수 보정을 끈다.
  [no-softplus]="model.use_softplus=false"
)

case "${1:-}" in
  build)
    mkdir -p output/merged
    : > "$QUEUE"
    for ab in "${ABLATIONS[@]}"; do
      for city in ulsan porto; do
        for seed in "${SEEDS[@]}"; do
          out="output/merged/runs/${city}_mae_${ab}_zero_seed${seed}.json"
          [ -f "$out" ] && continue
          echo "$ab $city $seed" >> "$QUEUE"
        done
      done
    done
    echo "큐 $(wc -l < "$QUEUE") 건 생성"
    ;;
  worker)
    GPU="${2:?gpu 번호 필요}"; SLOT="${3:?slot 번호 필요}"
    mkdir -p output/merged/runs output/merged/logs
    while true; do
      # 큐에서 첫 줄을 원자적으로 꺼낸다.
      task=$(flock "$LOCK" bash -c "head -1 '$QUEUE' 2>/dev/null; sed -i '1d' '$QUEUE' 2>/dev/null")
      [ -z "$task" ] && { echo "[w$GPU-$SLOT] 큐 비어 종료 $(date +%H:%M:%S)"; break; }
      read -r ab city seed <<< "$task"
      out="output/merged/runs/${city}_mae_${ab}_zero_seed${seed}.json"
      log="output/merged/logs/${city}_mae_${ab}_zero_seed${seed}.log"
      overrides="${ABLATION_OVERRIDES[$ab]:-}"
      # no-ir 외 ablation은 검색을 켠다.
      if [ "$ab" != no-ir ]; then
        overrides="model.use_retrieval=true $overrides"
      fi
      echo "[w$GPU-$SLOT] $(date +%H:%M:%S) 시작 $ab/$city/$seed"
      # shellcheck disable=SC2086
      # 도시별 model 그룹을 선택하려면 --config-name이 필요하다.
      PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n DA \
        python train.py --config-name "config_${city}" train.seed="$seed" \
          model.loss_type=mae ablation="$ab" run_json="$out" $overrides > "$log" 2>&1
      if [ -f "$out" ]; then
        echo "[w$GPU-$SLOT] $(date +%H:%M:%S) 완료 $ab/$city/$seed $(python3 -c "
import json;d=json.load(open('$out'));t=d['test']
print('mae=%.5f rmse=%.5f mape+1=%.3f mape0x=%.3f ep=%d'%(t['mae'],t['rmse'],t['mape_plus1'],t['mape_excl_zero'],d['best_epoch']))")"
      else
        echo "[w$GPU-$SLOT] $(date +%H:%M:%S) 실패 $ab/$city/$seed — $log 확인"
      fi
    done
    ;;
  *) echo "사용법: $0 build | worker <gpu> <slot>"; exit 1 ;;
esac
