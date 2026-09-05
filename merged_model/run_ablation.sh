#!/usr/bin/env bash
# merged_model ablation 큐 러너.
#
# 워커 여러 개가 flock으로 보호된 공용 큐에서 한 줄씩 꺼내 실행한다. 정적으로 나누지 않는
# 이유는 porto(~104분)와 ulsan(~42분)의 소요가 2.5배 차이라, 미리 쪼개면 한쪽 워커만
# 오래 남기 때문이다.
#
#   ./merged_model/run_ablation.sh build      # 큐 파일 생성(우선순위 순)
#   ./merged_model/run_ablation.sh worker <gpu> <slot>
#
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
QUEUE="merged_model/.ablation_queue"
LOCK="merged_model/.ablation_queue.lock"
# 3시드 운영: 셀 내부에서 평균과 표준오차를 직접 낼 수 있다. 기준선 full은 기존 5시드 런을
# 그대로 쓴다(0-치환 리팩터 후에도 소수점 16자리까지 재현됨을 확인).
SEEDS=(245 6835 851)
# 우선순위 순. 앞쪽 셀부터 완성되므로 중단 시점에 완결된 셀이 최대가 된다.
ABLATIONS=(no-ir no-periodic no-extra no-branch-attn no-neighbors)

case "${1:-}" in
  build)
    : > "$QUEUE"
    for ab in "${ABLATIONS[@]}"; do
      for city in ulsan porto; do
        for seed in "${SEEDS[@]}"; do
          out="merged_model/runs/${city}_mae_${ab}_zero_seed${seed}.json"
          [ -f "$out" ] && continue
          echo "$ab $city $seed" >> "$QUEUE"
        done
      done
    done
    echo "큐 $(wc -l < "$QUEUE") 건 생성"
    ;;
  worker)
    GPU="${2:?gpu 번호 필요}"; SLOT="${3:?slot 번호 필요}"
    mkdir -p merged_model/logs
    while true; do
      # 큐에서 첫 줄을 원자적으로 꺼낸다.
      task=$(flock "$LOCK" bash -c "head -1 '$QUEUE' 2>/dev/null; sed -i '1d' '$QUEUE' 2>/dev/null")
      [ -z "$task" ] && { echo "[w$GPU-$SLOT] 큐 비어 종료 $(date +%H:%M:%S)"; break; }
      read -r ab city seed <<< "$task"
      out="merged_model/runs/${city}_mae_${ab}_zero_seed${seed}.json"
      log="merged_model/logs/${city}_mae_${ab}_zero_seed${seed}.log"
      echo "[w$GPU-$SLOT] $(date +%H:%M:%S) 시작 $ab/$city/$seed"
      PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n DA \
        python -m merged_model.train --dataset "$city" --device cuda:0 --seed "$seed" \
          --loss-type mae --ablation "$ab" --output "$out" > "$log" 2>&1
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
