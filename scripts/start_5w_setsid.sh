#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dataset-assist-0/tmp/zsl/zsl/Protenix"
cd "${ROOT}"

: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
WANDB_ENTITY="${WANDB_ENTITY:-1553731627-tianjin-university}"
WANDB_PROJECT="${WANDB_PROJECT:-debug_qbiolip}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="DDP8-contrast-drugclipmask-v027-50k-${TS}"
LOG_PATH="${ROOT}/logs/train_5w_daemon_${TS}.log"
META_PATH="${ROOT}/logs/current_5w_run.txt"
RUN_SCRIPT="${ROOT}/logs/run_5w_daemon_${TS}.sh"

mkdir -p "${ROOT}/logs"

pkill -f "runner/train.py --model_name protenix_mini_esm650m_unimol_contrast_v0.2.7" || true
pkill -f "torchrun --standalone" || true
sleep 1

cat > "${RUN_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_ENTITY="${WANDB_ENTITY}"
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_MODE=online
export WANDB_CONSOLE=off
export PROTENIX_DATA_ROOT_DIR="${ROOT}/release_data"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PORT=\$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)
conda run --no-capture-output -n protenix311 \\
  torchrun --standalone --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:\${PORT} --nproc_per_node=8 \\
  runner/train.py \\
    --model_name protenix_mini_esm650m_unimol_contrast_v0.2.7 \\
    --project "\${WANDB_PROJECT}" \\
    --run_name "${RUN_NAME}" \\
    --base_dir ./output \\
    --batch_size 8 \\
    --data.batch_size 8 \\
    --data.num_dl_workers 0 \\
    --model.N_cycle 1 \\
    --sample_diffusion.N_sample 1 \\
    --max_steps 50000 \\
    --log_interval 10 \\
    --eval_interval 999999999 \\
    --checkpoint_interval 2000 \\
    --chain_permutation.train.mini_rollout False \\
    --atom_permutation.train.mini_rollout False \\
    --loss.weight.alpha_diffusion 0 \\
    --loss.weight.alpha_distogram 0 \\
    --loss.weight.alpha_confidence 0 \\
    --use_wandb True \\
    --data.train_sets qbiolip_nonredund \\
    --data.test_sets qbiolip_nonredund \\
    --data.qbiolip_nonredund.base_info.mmcif_dir "${ROOT}/biolip/nonredund_pl/all_data/mmcif" \\
    --data.qbiolip_nonredund.base_info.bioassembly_dict_dir "${ROOT}/biolip/nonredund_pl/all_data/prepared/bioassembly" \\
    --data.qbiolip_nonredund.base_info.indices_fpath "${ROOT}/biolip/nonredund_pl/all_data/prepared/indices_PL_only.csv" \\
    --data.qbiolip_nonredund.sampler_configs.sampler_type uniform \\
    --lr 0.0003
EOF
chmod +x "${RUN_SCRIPT}"

setsid -f bash "${RUN_SCRIPT}" > "${LOG_PATH}" 2>&1
sleep 2

{
  echo "started_at=${TS}"
  echo "run_name=${RUN_NAME}"
  echo "log_path=${LOG_PATH}"
  echo "run_script=${RUN_SCRIPT}"
  echo "wandb_entity=${WANDB_ENTITY}"
  echo "wandb_project=${WANDB_PROJECT}"
} > "${META_PATH}"

echo "RUN_NAME=${RUN_NAME}"
echo "LOG_PATH=${LOG_PATH}"
echo "META_PATH=${META_PATH}"
echo "PROCS:"
pgrep -af "torchrun --standalone|runner/train.py --model_name protenix_mini_esm650m_unimol_contrast_v0.2.7" || true
