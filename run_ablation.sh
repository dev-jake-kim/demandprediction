#!/usr/bin/env bash
# merged 모델 ablation 큐 러너.
#
# 워커 여러 개가 flock으로 보호된 공용 큐에서 한 줄씩 꺼내 실행한다. 정적으로 나누지 않는
# 이유는 porto(~104분)와 ulsan(~42분)의 소요가 2.5배 차이라, 미리 쪼개면 한쪽 워커만
# 오래 남기 때문이다.
#
#   ./run_ablation.sh build      # 큐 파일 생성(우선순위 순)
#   ./run_ablation.sh worker <gpu> <slot>
#
# merged_model/ 자체 실행 스크립트를 Hydra + Trainer 구조로 옮기면서, --ablation <name> CLI
# 인자가 아래 ABLATION_OVERRIDES의 Hydra 오버라이드 문자열로 바뀌었다. 큐/락/경로 로직과
# 결과 JSON 파일명은 그대로다(기존 output/merged_model/runs/*.json과 같은 규칙).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
QUEUE="output/merged/.ablation_queue"
LOCK="output/merged/.ablation_queue.lock"
# 3시드 운영: 셀 내부에서 평균과 표준오차를 직접 낼 수 있다. 기준선 full은 기존 5시드 런을
# 그대로 쓴다(0-치환 리팩터 후에도 소수점 16자리까지 재현됨을 확인).
SEEDS=(245 6835 851)
# 우선순위 순. 앞쪽 셀부터 완성되므로 중단 시점에 완결된 셀이 최대가 된다.
ABLATIONS=(no-ir no-periodic no-extra no-branch-attn no-neighbors)

# ablation 이름 -> Hydra 오버라이드. 원본 merged_model/train.py의 ABLATIONS dict와 1:1이다.
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
  # (2a+1)^2 로컬 창에서 중앙(자기 노드)만 남기고 이웃 공간 정보를 끈다.
  # 검색기 질의는 원래 크롭을 그대로 쓴다 — 인코더의 공간 정보만 분리해서 재려는 것이다.
  [no-neighbors]="model.use_neighbors=false"
  # 임베딩 방식 변형: 날씨를 LSTM concat 대신 ir-weather식으로 CLS에 더한다.
  [weather-cls-add]="model.weather_injection=cls_add"
  # 최종 예측의 softplus 양수 보정을 없애고 raw 선형 출력을 그대로 쓴다.
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
      # 도시 기본값은 검색 pass. no-ir 외의 이전 ablation은 검색 켬 기준으로 유지한다.
      if [ "$ab" != no-ir ]; then
        overrides="model.use_retrieval=true $overrides"
      fi
      echo "[w$GPU-$SLOT] $(date +%H:%M:%S) 시작 $ab/$city/$seed"
      # shellcheck disable=SC2086
      # 도시마다 최적 하이퍼파라미터가 달라 루트 config가 나뉘어 있다(config_ulsan/config_porto)
      # — dataset="$city" 오버라이드만으로는 model: 그룹이 안 바뀌므로 --config-name을 쓴다.
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
