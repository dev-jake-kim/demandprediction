#!/usr/bin/env bash
# ulsan multirun(PID 인자)이 끝나면 porto 5시드를 이어서 돌린다. GPU는 1번 고정.
set -uo pipefail
WAIT_PID="$1"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[chain] ulsan multirun(pid=$WAIT_PID) 종료, porto 시작: $(date -Is)"
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n DA \
  python train.py -m dataset=porto model.loss_type=mae seed=245,6835,851,5123,535 \
  hydra.sweep.dir=output/adformer/multirun/mae_porto > logs/adformer_mae_porto.log 2>&1
echo "[chain] porto 종료: $(date -Is)"
