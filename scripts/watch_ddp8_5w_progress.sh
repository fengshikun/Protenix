#!/usr/bin/env bash
set -u -o pipefail

ROOT="/home/dataset-assist-0/tmp/zsl/zsl/Protenix"
LOG_DIR="${ROOT}/logs"
TARGET_STEPS=50000
SLEEP_SECONDS=60
RUN_NAME_PREFIX="DDP8-contrast-drugclipmask-v027-50k"
PROC_PATTERN="runner/train.py --model_name protenix_mini_esm650m_unimol_contrast_v0.2.7"

mkdir -p "${LOG_DIR}"

if [[ $# -ge 1 ]]; then
  TRAIN_LOG="$1"
else
  TRAIN_LOG="$(ls -1t "${LOG_DIR}"/train_5w_*.log 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${TRAIN_LOG}" || ! -f "${TRAIN_LOG}" ]]; then
  echo "train log not found. pass log path as first arg." >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
WATCH_LOG="${LOG_DIR}/watch_5w_${TS}.log"

extract_step() {
  local step
  step="$(rg -o 'Step [0-9]+ train' "${TRAIN_LOG}" -N 2>/dev/null | tail -n 1 | rg -o '[0-9]+' -N 2>/dev/null || true)"
  if [[ -z "${step}" ]]; then
    step="$(rg -o '\\[GRAD_DEBUG\\]\\[step=[0-9]+\\]' "${TRAIN_LOG}" -N 2>/dev/null | tail -n 1 | rg -o '[0-9]+' -N 2>/dev/null || true)"
  fi
  if [[ -z "${step}" ]]; then
    step="$(rg -o 'self.step [0-9]+' "${TRAIN_LOG}" -N 2>/dev/null | tail -n 1 | rg -o '[0-9]+' -N 2>/dev/null || true)"
  fi
  if [[ -z "${step}" ]]; then
    step="$(rg -o 'Finish training after [0-9]+ steps' "${TRAIN_LOG}" -N 2>/dev/null | tail -n 1 | rg -o '[0-9]+' -N 2>/dev/null | tail -n 1 || true)"
  fi
  echo "${step:-0}"
}

is_running() {
  pgrep -f "${PROC_PATTERN}" >/dev/null 2>&1
}

echo "[$(date '+%F %T')] watch started" | tee -a "${WATCH_LOG}"
echo "[$(date '+%F %T')] target_steps=${TARGET_STEPS}" | tee -a "${WATCH_LOG}"
echo "[$(date '+%F %T')] train_log=${TRAIN_LOG}" | tee -a "${WATCH_LOG}"

while true; do
  step="$(extract_step)"
  if is_running; then
    status="running"
  else
    status="stopped"
  fi

  echo "[$(date '+%F %T')] status=${status} step=${step}" | tee -a "${WATCH_LOG}"

  if [[ "${step}" =~ ^[0-9]+$ ]] && (( step >= TARGET_STEPS )); then
    echo "[$(date '+%F %T')] target reached: step=${step}" | tee -a "${WATCH_LOG}"
    exit 0
  fi

  if [[ "${status}" == "stopped" ]]; then
    echo "[$(date '+%F %T')] training process stopped before reaching ${TARGET_STEPS}" | tee -a "${WATCH_LOG}"
    exit 1
  fi

  sleep "${SLEEP_SECONDS}"
done
