# 8-GPU training command

```bash
# W&B online mode (team entity from work_log_zhou.md)
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>" && \
export WANDB_ENTITY="1553731627-tianjin-university" && \
export WANDB_PROJECT=debug_qbiolip && \
export WANDB_MODE=online && \
export WANDB_CONSOLE=off && \
conda run --no-capture-output -n protenix311 wandb login --relogin "$WANDB_API_KEY" && \

cd /home/dataset-assist-0/tmp/zsl/zsl/Protenix && \
export PROTENIX_DATA_ROOT_DIR=/home/dataset-assist-0/tmp/zsl/zsl/Protenix/release_data && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 && \
export NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
export WANDB_ENTITY=$WANDB_ENTITY && \
export WANDB_PROJECT=$WANDB_PROJECT && \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 conda run --no-capture-output -n protenix311 \
torchrun --standalone --nproc_per_node=8 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.7 \
  --project $WANDB_PROJECT \
  --run_name DDP8-contrast-drugclipmask-v027-50k \
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
  --data.qbiolip_nonredund.sampler_configs.sampler_type uniform \
  --lr 0.0003
```

## 2026-03-05 修复与变更记录

### 1) 训练稳定性修复（避免中途崩溃）

- 文件：`protenix/model/unimol/models/dataset.py`
- 修改：
  - `TokenizeDataset.__getitem__` 增加容错：
    - 当 `raw_data` 为空时，返回 `unk` token，避免 `assert len(raw_data) > 0` 崩溃。
    - 当长度超限时，裁剪到 `max_seq_len - 1`。
  - `RemoveHydrogenDataset.__cached_item__` 增加容错：
    - 去氢后如果原子列表为空，回退到去氢前的原始 `atoms/coordinates`，避免后续空数组异常。

### 2) 路径修复（从旧机器路径切到当前路径）

- 文件：`configs/configs_model_type.py`
- 文件：`configs/configs_data.py`
- 修改：
  - 将硬编码路径 `/home/dataset-local/tmp/zsl/Protenix` 全部替换为当前环境路径：
    `/home/dataset-assist-0/tmp/zsl/zsl/Protenix`
  - 解决 `dict_mol.txt`、UniMol 权重、qbiolip indices 路径找不到问题。

### 3) 启动脚本增强（8卡 5w steps 稳定后台运行）

- 文件：`scripts/start_5w_setsid.sh`
- 修改：
  - 使用 `setsid` 后台启动 + 自动生成独立 `run_script` 与日志。
  - 自动固定 qbiolip 数据路径：
    - `--data.qbiolip_nonredund.base_info.mmcif_dir`
    - `--data.qbiolip_nonredund.base_info.bioassembly_dict_dir`
    - `--data.qbiolip_nonredund.base_info.indices_fpath`
  - 自动写入当前运行元信息：`logs/current_5w_run.txt`

### 4) 当前运行状态

- 当前 run_name：`DDP8-contrast-drugclipmask-v027-50k-20260305_124352`
- 当前日志：`logs/train_5w_daemon_20260305_124352.log`
- 当前 W&B：
  - `https://wandb.ai/1553731627-tianjin-university/debug_qbiolip/runs/4saok1qu`
- 最新确认：训练已稳定跑过早期故障点，日志已到 `Step 59 train`，并继续增长（`self.step` 持续递增）。

### 5) 推荐启动方式（后续统一用这个）

```bash
cd /home/dataset-assist-0/tmp/zsl/zsl/Protenix && \
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>" && \
export WANDB_ENTITY="1553731627-tianjin-university" && \
export WANDB_PROJECT="debug_qbiolip" && \
scripts/start_5w_setsid.sh
```
