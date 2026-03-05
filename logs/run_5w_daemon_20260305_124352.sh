#!/usr/bin/env bash
set -euo pipefail
cd "/home/dataset-assist-0/tmp/zsl/zsl/Protenix"
export WANDB_API_KEY="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"
export WANDB_ENTITY="1553731627-tianjin-university"
export WANDB_PROJECT="debug_qbiolip"
export WANDB_MODE=online
export WANDB_CONSOLE=off
export PROTENIX_DATA_ROOT_DIR="/home/dataset-assist-0/tmp/zsl/zsl/Protenix/release_data"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PORT=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)
conda run --no-capture-output -n protenix311 \
  torchrun --standalone --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:${PORT} --nproc_per_node=8 \
  runner/train.py \
    --model_name protenix_mini_esm650m_unimol_contrast_v0.2.7 \
    --project "${WANDB_PROJECT}" \
    --run_name "DDP8-contrast-drugclipmask-v027-50k-20260305_124352" \
    --base_dir ./output \
    --batch_size 8 \
    --data.batch_size 8 \
    --data.num_dl_workers 0 \
    --model.N_cycle 1 \
    --sample_diffusion.N_sample 1 \
    --max_steps 50000 \
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
    --data.qbiolip_nonredund.base_info.mmcif_dir "/home/dataset-assist-0/tmp/zsl/zsl/Protenix/biolip/nonredund_pl/all_data/mmcif" \
    --data.qbiolip_nonredund.base_info.bioassembly_dict_dir "/home/dataset-assist-0/tmp/zsl/zsl/Protenix/biolip/nonredund_pl/all_data/prepared/bioassembly" \
    --data.qbiolip_nonredund.base_info.indices_fpath "/home/dataset-assist-0/tmp/zsl/zsl/Protenix/biolip/nonredund_pl/all_data/prepared/indices_PL_only.csv" \
    --data.qbiolip_nonredund.sampler_configs.sampler_type uniform \
    --lr 0.0003
