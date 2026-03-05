#!/usr/bin/env bash
set -u -o pipefail

ROOT="/home/dataset-assist-0/tmp/zsl/zsl/Protenix"
LOG_DIR="${ROOT}/logs"
TARGET_STEPS=50000
RETRY_SLEEP_SECONDS=1800
RETRY_SHORT_SECONDS=60
TRAIN_PID=""

mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="${LOG_DIR}/train_5w_${TS}.log"
MONITOR_LOG="${LOG_DIR}/monitor_5w_${TS}.log"

: "${WANDB_ENTITY:=1553731627-tianjin-university}"
: "${WANDB_PROJECT:=debug_qbiolip}"

get_free_port() {
  python - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

extract_latest_step() {
  local step
  step="$(rg -o 'Step [0-9]+ train' "${TRAIN_LOG}" -N 2>/dev/null | tail -n 1 | rg -o '[0-9]+' -N 2>/dev/null || true)"
  if [[ -z "${step}" ]]; then
    step="$(rg -o 'Finish training after [0-9]+ steps' "${TRAIN_LOG}" -N 2>/dev/null | tail -n 1 | rg -o '[0-9]+' -N | tail -n 1 2>/dev/null || true)"
  fi
  echo "${step:-0}"
}

launch_train() {
  local port="$1"
  (
    cd "${ROOT}" || exit 1
    export PROTENIX_DATA_ROOT_DIR="${ROOT}/release_data"
    export CC=/usr/bin/gcc
    export CXX=/usr/bin/g++
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_DEBUG=warn
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export WANDB_MODE=online
    export WANDB_CONSOLE=off
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
      export WANDB_API_KEY
    fi
    export WANDB_ENTITY
    export WANDB_PROJECT
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    conda run --no-capture-output -n protenix311 \
      torchrun --standalone \
      --rdzv_backend=c10d \
      --rdzv_endpoint="127.0.0.1:${port}" \
      --nproc_per_node=8 \
      runner/train.py \
      --model_name protenix_mini_esm650m_unimol_contrast_v0.2.7 \
      --project "${WANDB_PROJECT}" \
      --run_name DDP8-contrast-drugclipmask-v027-50k \
      --base_dir ./output \
      --batch_size 8 \
      --data.batch_size 8 \
      --data.num_dl_workers 0 \
      --model.N_cycle 1 \
      --sample_diffusion.N_sample 1 \
      --max_steps "${TARGET_STEPS}" \
      --log_interval 10 \
      --eval_interval 999999999 \
      --checkpoint_interval 2000 \
      --chain_permutation.train.mini_rollout False \
      --atom_permutation.train.mini_rollout False \
      --loss.weight.alpha_diffusion 0 \
      --loss.weight.alpha_distogram 0 \
      --loss.weight.alpha_confidence 0 \
      --use_wandb True \
      --data.train_sets qbiolip_nonredund \
      --data.test_sets qbiolip_nonredund \
      --data.qbiolip_nonredund.sampler_configs.sampler_type uniform \
      --lr 0.0003
  ) >>"${TRAIN_LOG}" 2>&1 &

  TRAIN_PID="$!"
}

echo "[$(date '+%F %T')] monitor started" | tee -a "${MONITOR_LOG}"
echo "[$(date '+%F %T')] train_log=${TRAIN_LOG}" | tee -a "${MONITOR_LOG}"
echo "[$(date '+%F %T')] wandb_entity=${WANDB_ENTITY} project=${WANDB_PROJECT}" | tee -a "${MONITOR_LOG}"

attempt=1
while true; do
  port="$(get_free_port)"
  launch_train "${port}"
  pid="${TRAIN_PID}"
  echo "[$(date '+%F %T')] attempt=${attempt} launched pid=${pid} rdzv_port=${port}" | tee -a "${MONITOR_LOG}"

  while kill -0 "${pid}" 2>/dev/null; do
    step="$(extract_latest_step)"
    echo "[$(date '+%F %T')] pid=${pid} latest_step=${step}" | tee -a "${MONITOR_LOG}"
    if [[ "${step}" =~ ^[0-9]+$ ]] && (( step >= TARGET_STEPS )); then
      echo "[$(date '+%F %T')] target step reached (>=${TARGET_STEPS})" | tee -a "${MONITOR_LOG}"
      wait "${pid}" || true
      exit 0
    fi
    sleep 60
  done

  if wait "${pid}"; then
    exit_code=0
  else
    exit_code=$?
  fi

  step="$(extract_latest_step)"
  echo "[$(date '+%F %T')] pid=${pid} exited code=${exit_code} latest_step=${step}" | tee -a "${MONITOR_LOG}"
  if [[ "${step}" =~ ^[0-9]+$ ]] && (( step >= TARGET_STEPS )) && (( exit_code == 0 )); then
    echo "[$(date '+%F %T')] completed target successfully" | tee -a "${MONITOR_LOG}"
    exit 0
  fi

  if rg -n "FileNotFoundError|No such file or directory|找不到字典文件" "${TRAIN_LOG}" -S >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] detected missing file error, sleep ${RETRY_SLEEP_SECONDS}s before retry" | tee -a "${MONITOR_LOG}"
    sleep "${RETRY_SLEEP_SECONDS}"
  else
    echo "[$(date '+%F %T')] detected runtime failure, sleep ${RETRY_SHORT_SECONDS}s before retry" | tee -a "${MONITOR_LOG}"
    sleep "${RETRY_SHORT_SECONDS}"
  fi

  attempt=$((attempt + 1))
done
