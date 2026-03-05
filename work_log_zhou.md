# 1.ESM 和 unimol 接入 protenix

整体上，我希望借助protenix和qbiolip的数据库进行训练，完成虚拟筛选的对比学习，我提供给你一些代码，请告诉我下面应该干什么？
esm 编码为[L,1280] 是残基级别的表示，需要进行mean聚合
unimol 编码为[L,512] 是原子级别的表示，
protenix 维度为258
我首先将pro通过esm编码，将lig通过unimol编码，然后投影到258维，进行对比学习，这个时候得到loss1，然后将生成结构形成protenix的损失loss2，将这两个损失一起进行训练
注意的是，我们需要冻结protenix，esm unimol，只训练我们的MLP投影层

1. 我是用16张卡训练，batchaseize = 4，所以是方案A ，修改batchsize ，
2. 投影层改成MLP，至少两层relu
3. 

# 2.Biolip 的数据整理

Number of entries for regular ligands: 501072

1. 筛选Biolip里面的pro+lig部分，用esm + unimol 进行对比学习，然后和protenix一起优化
2. biolip_nr.txt.gz 内部每一行是protein-ligand交互样本，解析注释文件，过滤ligand的金属离子，合并receptorPDB 和ligandPDB
   1. receptor:与配体相互作用的蛋白结构（按链切分）
   2. ligand：对应配体结构，每个binding site/serial一份
      官方说明了怎么从这个文件中获得PDB结构的数据
      然而，困难的是我无法从其中下载，所以我选择了Q-biolip，其中包括四级结构https://yanglab.qd.sdu.edu.cn/Q-BioLiP/Download/index_biolip.html
3. protenix 有实验用的预训练脚本，默认会做一堆过滤（去水、去氢、去异常链/原子等,如果你输入的 CIF 不是“标准从 RCSB 下载的原始 mmCIF”，而是你自己生成/裁剪的，通常建议加 -d：-d 会禁用这些 filters，并且不会尝试 expand 到 Assembly 1
   1. input:cif文件
   2. output: .pkl.gz 的“训练可用缓存”（包含 atom_array/token_array、序列、分辨率等信息;
   3. output: output_csv（indices），用于catalog chains 和 interfaces，训练时按这个采样
      以上，数据准备已经结束.
      wget -c https://yanglab.qd.sdu.edu.cn/Q-BioLiP/DATA/rec_biolip.tar.gz

# 3. Q-biolip 数据下载及处理

## 3.1 下载

https://yanglab.qd.sdu.edu.cn/Q-BioLiP/Download/
Interaction-based → Protein–small molecules interaction → Non-redundant

## 3.2 run to clean

我通过Q-biolip的数据进行通过ESM 和UNIMOL 改良的protenix的模型训练，具体是lig通过unimol prot通过esm，然后简单投影到258维度进行对比学习，过程中，一个batchsize = 4，用16张卡进行训练，得到对比学习的loss；同时我们的复合物prot-lig的结构进行protenix的结构损失形成loss2
我首先进行投影层的训练，冻结protenix，然后根据训练情况，是否冻结unimol和esm
现在我有两个文件：

- run_clean_qbiolip.sh
- clean_qbiolip.py
  实现两个过程，将下载的lig和prot结构进行清洗获得干净的训练数据（可以进入我修改的unimol+protenix+esm的模型），同时需要调用protenix的数处理py文件进行数据规范化。

```bash
cd /home/dataset-local/tmp/zsl/Protenix/biolip/Q_biolip/qbiolip_PL_nonredund_mmcif

# 关键：指定 Protenix 根目录（按你的路径）
export PROTENIX_ROOT=/home/dataset-local/tmp/zsl/Protenix

# CPU 并行（按机器调整）
export WORKERS=32
export PREP_WORKERS=32

# 输出目录（可改）
export OUT_ROOT=$PWD/clean_qbiolip_nonredund

# （可选）更严格的“小分子”过滤：只保留 ligand.json 里有 SMILES 的 ligid
# export REQUIRE_SMILES=1

# （可选）如果你数据包含大量 2024-06-08 之后的新 CCD code，建议打开 CCD 更新（需要联网）
# export UPDATE_CCD=1

bash run_clean_qbiolip_nonredund_mmcif.sh

```

### 3.2.1 location

[DONE] complexes: /home/dataset-local/tmp/zsl/Protenix/biolip/Q_biolip/qbiolip_PL_nonredund_mmcif/mmcif_output/complexes
[DONE] manifest:  /home/dataset-local/tmp/zsl/Protenix/biolip/Q_biolip/qbiolip_PL_nonredund_mmcif/mmcif_output/manifest.tsv
[DONE] failed:    /home/dataset-local/tmp/zsl/Protenix/biolip/Q_biolip/qbiolip_PL_nonredund_mmcif/mmcif_output/failed.tsv
[DONE] stats:     /home/dataset-local/tmp/zsl/Protenix/biolip/Q_biolip/qbiolip_PL_nonredund_mmcif/mmcif_output/stats.txt
[DONE] cif_list:  /home/dataset-local/tmp/zsl/Protenix/biolip/Q_biolip/qbiolip_PL_nonredund_mmcif/mmcif_output/complexes_cif_paths.txt
[DONE] indices:   /home/dataset-local/tmp/zsl/Protenix/biolip/Q_biolip/qbiolip_PL_nonredund_mmcif/mmcif_output/qbiolip_nonredund_indices.csv
[DONE] bioasm:    /home/dataset-local/tmp/zsl/Protenix/biolip/Q_biolip/qbiolip_PL_nonredund_mmcif/mmcif_output/qbiolip_nonredund_bioassembly

### 3.2.2 expliation

原始结构：受体-配体结构
annotations: csv ID PDB Assembly ligandID 位点 binding Site residues
ligand.json : smiles mw 小分子集合

1. 合并rec 和lig 形成complex cif
2. 清洗ligand 只保留一个model 去掉alt conf H water，选择最佳的ligand residue
3. 清洗rec：保留model 去掉alter conf /H/water
   调用protenix标准预处理脚本
   find complexes -name "*.cif" > complexes_cif_paths.txt

python3 $PROTENIX_ROOT/scripts/prepare_training_data.py -i `<list>` -o indices.csv -b bioassembly -n `<cpu>` -d

run_clean_qbiolip_nonredund_mmc…

这里用了 -d，含义是：
“这批 CIF 不是 RCSB 原生下载的 WeightedPDB 格式，不要再套一堆 WeightedPDB 过滤/assembly 扩展规则。”

# 模块整理

调用方式

```bash
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

python runner/train.py \
      --model_name "protenix_mini_esm650m_unimol_contrast_v0.2.0" \
      --run_name "test_run_final" \
      --base_dir "./output" \
      --model.N_cycle 1 \
      --sample_diffusion.N_sample 1 \
      --max_steps 10 \
      --log_interval 1 \
      --eval_interval 10 \
      --checkpoint_interval 0 \
      --use_wandb False \
      --project "test_debug"
```
## 如何将loss加入进行训练
qbiolip 输出为:
1. receptor 和 ligand 合成一个complex cif 输出到out_root/complect/../8cif
2. complex cif 输入到prepare_training_data.py 生成
   1. qbiolip_*_indices.csv
   2. qbiolip_*_bioassembly/*.pkl.gz
# 问题

1. batchsize=1是用ddp吗？ 不是，用16张卡，batchsize=4，先简单训练一下，看看可不可以跑通，如果不行，我们再调整一下状态
2. 数据是149w的结构数据 40w结构数据？直接下载mmcif的文件 有4.5w蛋白质，同时有13w小分子对应，进行数据清洗和合并，形成新的结构
3. 对比学习投影，我就只用了一个line投影到258维？不行，用MLP
4. 接下来是冻结unimol 和 esm protenix 直接训练 投影吗？不一定是冻结esm和unimol，反正protenix是一定要修改的，需要开放mlp和unimol，esm不一定要开放
5.
### 修改完成qbiolip(单卡)
export CC=/usr/bin/gcc && export CXX=/usr/bin/g++ && conda activate protenix311 && python -u runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.0 \
  --run_name debug_qbiolip_single2 \
  --base_dir ./output \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 2 \
  --log_interval 1 \
  --eval_interval 10 \
  --checkpoint_interval 0 \
  --use_wandb False \
  --project debug_qbiolip \
  --data.train_sets qbiolip_nonredund

  ### q1 存在ligand不识别的问题?
1.   根据qbiolip 下载select ligand的黑名单
2.   python3 /home/dataset-local/tmp/zsl/Protenix/scripts/gen_ccd_cache.py -n 32 下载ccd数据
```bash
CCD_COMPONENTS_FILE_PATH = os.path.join(DATA_ROOT_DIR, "ccd_cache","components.cif")
CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
    DATA_ROOT_DIR,"ccd_cache","components.cif.rdkit_mol.pkl"
)
```
3. 为了避免DDP过程中出现ligand=0的情况会导致NCCL Hang，需要对过程进行补充(解决方法：对缺少的部分进行补充，让他们ligand=0的部分也要进行同样的处理流程，但是loss不做贡献)

### qbiolip（四卡）
cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
source ~/.bashrc 2>/dev/null || true && \
conda activate protenix311 && \
export NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.0 \
  --run_name debug_qbiolip_ddp4 \
  --base_dir ./output \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 2 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 999999999 \
  --use_wandb False \
  --project debug_qbiolip \
  --data.train_sets qbiolip_nonredund
#### q2 不同的rank走了不同的分支，有的有ligand 有的没有，导致某些需要同步梯度的参数在部分rank上变成了unused 从而梯度上死锁超时
find_unused_parameters: true
static_graph=False,# 原本是true
## 考虑到cropsize导致的ligand删除
            bioassembly_dict["token_array"], bioassembly_dict["atom_array"], _, _,_= (




### wandb true
cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
source ~/.bashrc 2>/dev/null || true && \
conda activate protenix311 && \
export NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.0 \
  --run_name debug_qbiolip_ddp4 \
  --base_dir ./output \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 2 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 999999999 \
  --use_wandb True \
  --project debug_qbiolip \
  --data.train_sets qbiolip_nonredund



  ## 直接训练的代码
  cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
source ~/.bashrc 2>/dev/null || true && \
conda activate protenix311 && \
export NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.0 \
  --run_name qbiolip_PLonly_ddp4_bs1_e100 \
  --base_dir ./output \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 3236900 \
  --log_interval 10 \
  --eval_interval 999999999 \
  --checkpoint_interval 20000 \
  --use_wandb True \
  --project debug_qbiolip \
  --data.train_sets qbiolip_nonredund

### structure-only sanity check (Protenix loss without contrastive term)
cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
source ~/.bashrc 2>/dev/null || true && \
conda activate protenix311 && \
export NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_structure_only_v0.1.0 \
  --run_name structure_only_protenix \
  --base_dir ./output \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 50000 \
  --log_interval 10 \
  --eval_interval 999999999 \
  --checkpoint_interval 20000 \
  --use_wandb True \
  --project debug_qbiolip \
  --data.train_sets qbiolip_nonredund \
  --contrast.enable False

该命令复用了 `protenix_mini_structure_only_v0.1.0` 配置，它继承了现有的 ESM/UniMol 设置但把 contrast_loss 关闭，便于观察仅用 Protenix 结构损失的收敛行为。

## 本次修改完成（2026-03-03）

### 1) 运行方式修正
- 当前环境里 `conda activate` 不可用，统一改为 `conda run -n protenix311 ...`
- NCCL 异步报错变量改为 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`

### 2) 纯 Protenix 训练（无对比损失）
- 已新增模型配置：`protenix_mini_structure_only_v0.1.0`
- 使用该模型时 `contrast.enable=False`，只优化结构相关损失（diffusion/distogram/confidence）

### 3) 联合训练路径代码对齐（核心改动）
- 文件：`protenix/model/protenix.py`
- 已将同一套 ESM/UniMol MLP 输出同时用于：
  1. 结构分支（映射后加到 `s_inputs`，进入 pairformer/protenix）
  2. 对比分支（contrast loss）
- 新增 `esm_token_embedding` 预构建函数并在 forward 中统一写入
- 修正 `contrast_logit_scale` 传递，去掉前向中的提前 `exp`，避免与 loss 侧重复处理

### 4) 快速验证命令（先看 loss 是否正常刷新）
```bash
cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
export WANDB_MODE=online WANDB_CONSOLE=off WANDB_API_KEY=... && \
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n protenix311 \
torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.0 \
  --project debug_qbiolip \
  --run_name ddp4_contrast_joint_quickcheck \
  --base_dir ./output \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 50 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 0 \
  --use_wandb True \
  --data.train_sets qbiolip_nonredund
```

### 5) 可直接训练（含对比学习）
```bash
cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
export WANDB_MODE=online WANDB_CONSOLE=off WANDB_API_KEY=... && \
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n protenix311 \
torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.0 \
  --project debug_qbiolip \
  --run_name ddp4_contrast_joint_train \
  --base_dir ./output \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 1000 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 999999999 \
  --use_wandb True \
  --data.train_sets qbiolip_nonredund
```

备注：为避免 DDP 报 `logit_scale marked ready twice`，`contrast_logit_scale` 在前向输出时使用了 detached tensor。

## 稳定性改动（2026-03-03，contrast 联训）

### 6) 默认训练策略改为“联合训练”
- 模型配置 `protenix_mini_esm650m_unimol_contrast_v0.2.0` 已调整：
  - `contrast.train_projection_only: False`
  - `contrast.loss_weight: 0.2`
  - `contrast.loss_weight_warmup_steps: 2000`
- 目的：避免只训投影头导致结构分支几乎不下降，同时减轻 early-stage 多任务冲突。

### 7) 新增稳定版模型配置
- 新增：`protenix_mini_esm650m_unimol_contrast_v0.2.2`
- 关键默认：
  - `contrast.enable: True`
  - `contrast.train_projection_only: False`
  - `contrast.loss_weight: 0.2`
  - `contrast.loss_weight_warmup_steps: 2000`

### 8) loss 侧新增 contrast 权重 warmup
- 文件：`protenix/model/loss.py`
- 新增按 step 的动态权重：
  - `effective_weight = contrast.loss_weight * min(1, (step+1)/warmup_steps)`
- 并将实际权重写入日志指标：`contrast_loss/weight`。

### 9) 训练步数注入到 loss
- 文件：`runner/train.py`
- 在 `get_loss` 前向中注入：`input_feature_dict["current_step"] = self.step`
- 用于 loss 侧 warmup 权重计算。

### 10) 推荐训练命令（稳定版）
```bash
cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
export WANDB_MODE=online WANDB_CONSOLE=off WANDB_API_KEY=... && \
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n protenix311 \
torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.2 \
  --project debug_qbiolip \
  --run_name ddp4_contrast_joint_train_v022 \
  --base_dir ./output \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 1000 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 999999999 \
  --use_wandb True \
  --data.train_sets qbiolip_nonredund
```

### 11) 运行排障（W&B 无 loss 刷新）
- 现象：W&B run 已创建，但 `wandb-history.jsonl` 未生成，页面无 loss。
- 定位：训练在 mini-rollout 的链置换阶段反复报错并重启，错误样本见：
  - `output/ddp4_contrast_joint_train_v022_20260303_132037/errors/chain_permutation/T_20260303_132136.pt`
  - 核心报错：`The size of tensor a (0) must match the size of tensor b (235)`

### 12) 临时稳定启动参数（先让 loss 正常写入）
- 在命令里增加：
  - `--chain_permutation.train.mini_rollout False`
  - `--atom_permutation.train.mini_rollout False`
- 当前在跑：
  - `run_name: ddp4_contrast_joint_train_v022_noperm`
  - `model_name: protenix_mini_esm650m_unimol_contrast_v0.2.2`

### 13) contrast 缺失 key 统计与日志化
- 文件：`protenix/model/loss.py`
- 新增 `contrast` 输入字段缺失统计，不再只是静默 fallback。
- 记录维度：
  - `contrast_loss/missing_input_total`
  - `contrast_loss/missing_input_prot_z`
  - `contrast_loss/missing_input_lig_z`
  - `contrast_loss/missing_input_logit_scale`
  - `contrast_loss/missing_input_valid_mask`
- 机制：
  - 当以下任一字段缺失时注入稳定 fallback tensor，并将计数加一：
    - `contrast_prot_z`
    - `contrast_lig_z`
    - `contrast_logit_scale`
    - `contrast_valid_mask`
  - 将总缺失次数与各 key 缺失次数写入 WandB 统计。

### 14) 已明确为“一个卡一个样本”场景做适配
- 已按你要求按卡级输入处理（每 rank 处理独立样本）继续走训练流程，便于快速定位异常样本与重启场景。

### 15) 可配置每卡 batch（先行 2 卡样本）
- 文件：
  - `configs/configs_base.py`：新增 `data.batch_size` 默认值（`1`）。
  - `protenix/data/dataloader.py`：训练 `DataLoader` 的 `batch_size` 改为 `configs.data.batch_size`，替换原先硬编码 `1`。
- 已支持通过命令行设置每卡 batch：
  - `--data.batch_size 2`
- 说明：
  - `4 卡 × 2` 时，全局训练 batch 大小为 8；
  - 建议先保留 `--iters_to_accumulate 1`，观察单步显存；
  - 如 OOM，可回退 `--data.batch_size 1`，或改 `--iters_to_accumulate 2`（同等“等效 batch”但延迟梯度同步）。

### 15) contrast 支持每卡 `B>1`（目标：4卡 x 每卡4 = global 16）
- 文件：`protenix/model/loss.py`
- 背景：此前 loss 侧存在“强制 `B=1`”路径，导致即使你设置了每卡 `batch_size=4`，W&B 里的 `contrast_loss/global_batch` 仍长期接近 `4`。

- 本次核心修改：
  - `ContrastiveCLIPLoss` 从“仅支持 `B=1`”改为“支持 `B>=1`”。
  - `_force_B1` 改为 `_force_BD`，不再把 `[B,D]` 压成 `[1,D]`。
  - 去掉 gather 前的 `B==1` 断言；全局偏移由 `rank` 改为 `rank * local_b`。
  - 对比标签由单值改为批量：`labels = torch.arange(B) + offset`。
  - `valid_mask` 改为与 `B` 对齐（按需 repeat/pad/truncate），不再截断为单元素。
  - 缺失 key 的 fallback tensor 从固定 `[1,D]` / `[1]` 改为 `[B,D]` / `[B]`。
  - invalid 样本处理从“整批处理”改为“按行处理”（`valid_mask<=0` 的样本逐行填充）。

- 预期观测（W&B）：
  - `train/contrast_loss/valid_local.avg` 接近 `4`
  - `train/contrast_loss/global_batch.avg` 接近 `16`
  - 若仍是 `1/4`，说明上游 forward/data 仍只产出每卡 1 对 embedding，需要继续改 model/data 侧。

### 16) 按复盘意见补充：contrast 训练与诊断能力增强（2026-03-03）
- 文件：`protenix/model/protenix.py`
  - 新增配置：`contrast.learn_logit_scale`（默认 `False`）
  - 行为：
    - `False`：`logit_scale` 作为 buffer（稳定模式）
    - `True`：`logit_scale` 作为 `nn.Parameter`（可学习温度）
  - forward 输出 `contrast_logit_scale` 改为按上述配置分支：可学习时不 detach，稳定模式下 detach。

- 文件：`protenix/model/loss.py`
  - `ContrastiveCLIPLoss` 新增配置开关：
    - `use_grad_gather`（由 `contrast.full_grad_gather` 控制，默认 `False`）
    - `detach_logit_scale`（由 `learn_logit_scale` 反向控制）
  - gather 模式：
    - 默认仍为 no-grad all_gather（稳定）
    - 可选 full-grad gather（用于更接近标准 CLIP 的负样本梯度传播）
  - 指标增强：新增
    - `contrast_loss/raw`
    - `contrast_loss/local_batch`
  - 现有指标保留：
    - `contrast_loss/weight`
    - `contrast_loss/global_batch`
    - `contrast_loss/valid_local`
    - `contrast_loss/valid_global`

- 结论：
  - 现在可以直接区分“weighted 上升是否仅由 warmup 导致”；
  - 可按配置在“稳定优先（默认）”与“对比学习强优化（full-grad + learn logit_scale）”间切换。

### 17) 重新整理并重跑配置：`v0.2.3`（full-grad contrast）
- 文件：`configs/configs_model_type.py`
- 新增模型配置：`protenix_mini_esm650m_unimol_contrast_v0.2.3`
- 关键差异（相对 `v0.2.2`）：
  - `contrast.full_grad_gather: True`（启用带 backward 的全局 gather）
  - `contrast.learn_logit_scale: False`（先保持稳定，避免历史 DDP ready twice 风险）
  - `contrast.loss_weight_warmup_steps: 0`（去掉 warmup，避免 weighted 指标被线性拉升干扰判断）
- 目的：
  - 提升对比负样本梯度质量；
  - 让 `weighted_contrast_loss` 与 `raw contrast_loss` 趋势更一致，便于直接判断是否收敛。

### 18) 已按整理后的配置重启训练（2026-03-03）
- 运行名称（W&B）：`DDP4-对比学习-修改loss`
- 模型配置：`protenix_mini_esm650m_unimol_contrast_v0.2.3`
- 关键参数：
  - `--batch_size 4`
  - `--chain_permutation.train.mini_rollout False`
  - `--atom_permutation.train.mini_rollout False`
  - `--max_steps 1000`
- 目的：使用 `full_grad_gather` 且关闭 contrast weight warmup，观察 `contrast_loss/raw` 是否出现更明确下降趋势。

### 19) 实现“真每卡4样本”训练路径（2026-03-03）
- 文件：`protenix/data/dataloader.py`
  - 训练集 `batch_size>1` 时不再用 `collate_fn_first` 丢弃样本，改为 `collate_fn_identity` 返回完整 batch 列表。
- 文件：`protenix/utils/torch_utils.py`
  - `to_device` 新增对 `list/tuple` 的递归搬运支持，兼容训练 batch 为 `list[dict]`。
- 文件：`runner/train.py`
  - `train_step` 新增“local batch list”分支：
    - 同一步内逐样本前向并累计结构 loss（取均值）；
    - contrast 不再逐样本算 B=1，而是收集本地全部样本 embedding 后一次性计算（`B_local=4`）。
  - 新增辅助函数：
    - `_merge_loss_dicts_mean`
    - `_compute_group_contrast_loss`
- 预期：
  - `train/contrast_loss/local_batch.avg` 接近 `4`
  - `train/contrast_loss/global_batch.avg` 接近 `16`（4卡）

### 20) 状态确认：真每卡4样本改造已完成（2026-03-03）
- 已完成代码改动并可用于训练：
  - `protenix/data/dataloader.py`：`batch_size>1` 时训练使用 `collate_fn_identity`，不再丢样本。
  - `protenix/utils/torch_utils.py`：`to_device` 支持 `list/tuple` 递归搬运。
  - `runner/train.py`：训练步支持 `list` batch；同一步内按本地多样本聚合 contrast（local batch 真正为 4）。
- 预期训练指标：
  - `train/contrast_loss/local_batch.avg ≈ 4`
  - `train/contrast_loss/global_batch.avg ≈ 16`（4 卡）
- 建议启动命令（确保启用真实每卡4）：
```bash
cd /home/dataset-local/tmp/zsl/Protenix && \
export WANDB_ENTITY="1553731627-tianjin-university" && \
export WANDB_PROJECT="debug_qbiolip" && \
export WANDB_MODE=online && \
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n protenix311 \
torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.2 \
  --project debug_qbiolip \
  --run_name DDP4-对比学习-20260303_trueB4 \
  --base_dir ./output \
  --batch_size 4 \
  --data.batch_size 4 \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 1000 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 999999999 \
  --chain_permutation.train.mini_rollout False \
  --atom_permutation.train.mini_rollout False \
  --use_wandb True \
  --data.train_sets qbiolip_nonredund
```

## 本次追加（2026-03-04，contrast-only 稳定性修复）

### 21) 对比学习 loss 增加“防塌缩正则”与诊断指标
- 文件：`protenix/model/loss.py`
- 变更：
  - `ContrastiveCLIPLoss.__init__` 新增参数：
    - `std_floor`
    - `std_reg_weight`
  - forward 中将 loss 拆分为：
    - `raw_loss`（原始 InfoNCE）
    - `final_loss = raw_loss + std_reg_weight * collapse_penalty`
  - `collapse_penalty` 定义：
    - `relu(std_floor - prot_std) + relu(std_floor - lig_std)`
  - 新增日志指标：
    - `contrast_loss/final`
    - `contrast_loss/collapse_penalty`
    - `contrast_loss/std_floor`
    - `contrast_loss/std_reg_weight`
  - `ProtenixLoss` 已接入配置读取并传给 `ContrastiveCLIPLoss`。

### 22) `v0.2.6` 默认 contrast 配置改为更稳模式
- 文件：`configs/configs_model_type.py`
- 模型：`protenix_mini_esm650m_unimol_contrast_v0.2.6`
- 变更：
  - `learn_logit_scale: False`
  - `init_temp: 0.2`
  - `std_floor: 0.02`
  - `std_reg_weight: 1.0`

### 23) 训练启动参数修正（关键）
- 结论：
  - 想真正覆盖优化器学习率，需使用 `--lr`。
  - 仅使用 `--adam.lr` / `--af3_lr_scheduler.lr` 时，实测 optimizer param group 仍可能保持默认 `0.0018`。
- 现已验证 `--lr 0.0003` 生效（日志显示 `optimizer.param_group[0].lr=0.0003`）。

### 24) W&B 权限问题修复
- 原因：`WANDB_ENTITY=1553731627` 触发 403（personal entities disabled）。
- 修复：统一改为团队实体：
  - `WANDB_ENTITY=1553731627-tianjin-university`

### 25) 成功调用命令（改动后可跑通）

#### 25.1 成功上线（团队实体，4卡，每卡8）
```bash
cd /home/dataset-local/tmp/zsl/Protenix

export WANDB_API_KEY="..."
export WANDB_ENTITY=1553731627-tianjin-university
export WANDB_PROJECT=debug_qbiolip
export WANDB_MODE=online
export WANDB_CONSOLE=off
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1,2,3 conda run --no-capture-output -n protenix311 \
torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.6 \
  --project debug_qbiolip \
  --run_name DDP4-contrast-only-online-bs8-team-20260304_1445 \
  --base_dir ./output \
  --batch_size 8 \
  --data.batch_size 8 \
  --data.num_dl_workers 0 \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 1000 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 200 \
  --chain_permutation.train.mini_rollout False \
  --atom_permutation.train.mini_rollout False \
  --loss.weight.alpha_diffusion 0 \
  --loss.weight.alpha_distogram 0 \
  --loss.weight.alpha_confidence 0 \
  --use_wandb True \
  --data.train_sets qbiolip_nonredund
```

#### 25.2 成功运行（防塌缩 + 真 lr=3e-4 + 300 step 诊断）
```bash
cd /home/dataset-local/tmp/zsl/Protenix

export WANDB_API_KEY="..."
export WANDB_ENTITY=1553731627-tianjin-university
export WANDB_PROJECT=debug_qbiolip
export WANDB_MODE=online
export WANDB_CONSOLE=off
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1,2,3 conda run --no-capture-output -n protenix311 \
torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.6 \
  --project debug_qbiolip \
  --run_name DDP4-contrast-only-stdreg-lr3e4-global-300-20260304_1501 \
  --base_dir ./output \
  --lr 0.0003 \
  --batch_size 8 \
  --data.batch_size 8 \
  --data.num_dl_workers 0 \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 300 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 200 \
  --chain_permutation.train.mini_rollout False \
  --atom_permutation.train.mini_rollout False \
  --loss.weight.alpha_diffusion 0 \
  --loss.weight.alpha_distogram 0 \
  --loss.weight.alpha_confidence 0 \
  --use_wandb True \
  --data.train_sets qbiolip_nonredund
```

### 26) 当前运行状态
- 当前任务正在运行：
  - `run_name: DDP4-contrast-only-stdreg-lr3e4-global-300-20260304_1501`
  - `nproc_per_node=4`
  - 每卡 `batch_size=8`，全局 batch `=32`
  - 学习率 `=0.0003`（已在 optimizer param group 日志中确认）

### 27) 框架级问题定位与修复（2026-03-04，contrast 训练稳定性）
- 结论（核心）：
  - 冻结的 ESM / UniMol 编码器在训练态下可能被 `model.train()` 重新切到 train mode，导致 dropout 噪声持续注入，对比学习目标不稳定。
  - 这属于 contrast 框架链路问题，不是是否叠加 Protenix 结构损失的问题。

- 文件：`protenix/model/protenix.py`
  - 修复1：当 `unimol_trainable=False` 时，显式冻结 UniMol 参数并 `eval()`：
    - 位置：`__init__` 附近（`if not self.unimol_trainable:`）
  - 修复2：新增 `train(self, mode)` 覆写，确保冻结编码器在训练期间保持 `eval`：
    - ESM（`esm_trainable=False`）强制 `self.esm_model.eval()`
    - UniMol（`unimol_trainable=False`）强制 `self.unimol_model.eval()`
  - 修复3：在 `forward` 中二次兜底：
    - `if not self.esm_trainable: self.esm_model.eval()`
    - `if not self.unimol_trainable: self.unimol_model.eval()`

- 文件：`protenix/data/compute_esm.py`
  - 修复4：`compute_esm2_embeddings_online(..., trainable=False)` 时，函数入口强制 `model.eval()`，避免上层模式切换带来的随机性。

- 备注：
  - `GRAD_DEBUG` 显示对比投影头已有有效梯度（非 0），说明并非“对比头没被 optimizer 更新”。

### 28) 成功调用命令（框架修复后）
```bash
cd /home/dataset-local/tmp/zsl/Protenix

export WANDB_API_KEY="..."
export WANDB_ENTITY=1553731627-tianjin-university
export WANDB_PROJECT=debug_qbiolip
export WANDB_MODE=online
export WANDB_CONSOLE=off
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1,2,3 conda run --no-capture-output -n protenix311 \
torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.6 \
  --project debug_qbiolip \
  --run_name DDP4-contrast-only-encoder-evalfix-bs4-20260304_1521 \
  --base_dir ./output \
  --lr 0.0003 \
  --batch_size 4 \
  --data.batch_size 4 \
  --data.num_dl_workers 0 \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 120 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 200 \
  --chain_permutation.train.mini_rollout False \
  --atom_permutation.train.mini_rollout False \
  --loss.weight.alpha_diffusion 0 \
  --loss.weight.alpha_distogram 0 \
  --loss.weight.alpha_confidence 0 \
  --use_wandb True \
  --data.train_sets qbiolip_nonredund
```

### 29) 框架修复后再次启动（当前在跑）
- `run_name`: `DDP4-contrast-only-encoder-evalfix-bs4-20260304_1533`
- 关键参数：
  - `--lr 0.0003`
  - `--batch_size 4 --data.batch_size 4`（4 卡全局 batch=16）
  - `--max_steps 120`
  - 仅对比学习（`alpha_diffusion/distogram/confidence = 0`）
- 命令：
```bash
cd /home/dataset-local/tmp/zsl/Protenix

export WANDB_API_KEY="..."
export WANDB_ENTITY=1553731627-tianjin-university
export WANDB_PROJECT=debug_qbiolip
export WANDB_MODE=online
export WANDB_CONSOLE=off
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1,2,3 conda run --no-capture-output -n protenix311 \
torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.6 \
  --project debug_qbiolip \
  --run_name DDP4-contrast-only-encoder-evalfix-bs4-20260304_1533 \
  --base_dir ./output \
  --lr 0.0003 \
  --batch_size 4 \
  --data.batch_size 4 \
  --data.num_dl_workers 0 \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 120 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 200 \
  --chain_permutation.train.mini_rollout False \
  --atom_permutation.train.mini_rollout False \
  --loss.weight.alpha_diffusion 0 \
  --loss.weight.alpha_distogram 0 \
  --loss.weight.alpha_confidence 0 \
  --use_wandb True \
  --data.train_sets qbiolip_nonredund
```

## 诊断追加（2026-03-04）

### 30) contrast 诊断增强（pair id + 基线差值 + 梯度连通）
- 文件：`runner/train.py`
- 改动：
  - `pair_id` 提取逻辑新增从 `basic` 字段读取：`pdb_id/assembly_id/chain_1_id/chain_2_id/ligand_id`，减少 fallback。
  - 新增基线对照指标：
    - `contrast_loss/random_top1`
    - `contrast_loss/raw_minus_ln_global_batch`
    - `contrast_loss/top1_minus_random`
  - 保留并继续记录：
    - `contrast_loss/raw_shuffled_local`
    - `contrast_loss/raw_gap_vs_shuffled`
  - 新增/扩展梯度连通指标（rank0 前 30 step）：
    - `contrast_loss/grad_norm_sum_*`
    - `contrast_loss/params_in_opt_with_grad_*`

### 31) 调整 v0.2.6 的防塌缩正则
- 文件：`configs/configs_model_type.py`
- 模型：`protenix_mini_esm650m_unimol_contrast_v0.2.6`
- 修改：
  - `contrast.std_floor: 0.02 -> 0.05`
  - `contrast.std_reg_weight: 1.0 -> 2.0`

### 32) 当前正在运行的 300 step 诊断命令（4卡，每卡4）
```bash
cd /home/dataset-local/tmp/zsl/Protenix && \
export WANDB_API_KEY="..." WANDB_ENTITY=1553731627-tianjin-university WANDB_PROJECT=debug_qbiolip WANDB_MODE=online WANDB_CONSOLE=off && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run --no-capture-output -n protenix311 \
torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.6 \
  --project debug_qbiolip \
  --run_name DDP4-contrast-diagnose-v3-pairid-stdreg-bs4 \
  --base_dir ./output \
  --lr 0.0003 \
  --batch_size 4 \
  --data.batch_size 4 \
  --data.num_dl_workers 0 \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 300 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 200 \
  --chain_permutation.train.mini_rollout False \
  --atom_permutation.train.mini_rollout False \
  --loss.weight.alpha_diffusion 0 \
  --loss.weight.alpha_distogram 0 \
  --loss.weight.alpha_confidence 0 \
  --use_wandb True \
  --data.train_sets qbiolip_nonredund
```

### 33) 当前已观察到的早期信号（本次 run）
- `contrast_loss/pair_id_missing = 0`（已不再 fallback）
- `contrast_loss/pair_id_unique = 4`，`pair_id_dup = 0`
- `contrast_loss/raw_minus_ln_global_batch` 已出现负值（例如 step3 约 `-0.0173`）
- `contrast_loss/raw_gap_vs_shuffled` 出现正值（例如 step3 约 `+0.0156`）
- `esm_contrast_proj/unimol_contrast_proj` 梯度非零，且参数在 optimizer 内。

### 34) 新增 contrast-only-forward 训练路径（2026-03-04）
- 文件：`protenix/model/protenix.py`
- 改动：
  - 新增开关 `contrast.only_forward`。
  - 当 `contrast.only_forward=True` 时，forward 在生成 `contrast_out` 后直接返回，仅包含：
    - `contrast_prot_z`
    - `contrast_lig_z`
    - `contrast_logit_scale`
    - `contrast_valid_mask`
  - 不再进入结构主干计算（pairformer/diffusion/confidence）。

- 文件：`runner/train.py`
- 改动：
  - `train_step` 新增 `contrast_only_forward` 分支：
    - `len(batch_list)==1`：直接计算一次 contrast loss，不再走 `get_loss` 结构损失路径。
    - `len(batch_list)>1`：对本地 batch 每个样本做 `model_forward`，再聚合计算一次 contrast loss。
  - 保留并记录 `contrast_loss/*` 诊断指标。

- 文件：`configs/configs_model_type.py`
- 模型：`protenix_mini_esm650m_unimol_contrast_v0.2.6`
- 改动：
  - `contrast.only_forward: True`
  - `contrast.std_floor: 0.05`
  - `contrast.std_reg_weight: 2.0`

### 35) contrast-only-forward 启动命令（4卡，每卡4）
```bash
cd /home/dataset-local/tmp/zsl/Protenix && \
export WANDB_API_KEY="..." WANDB_ENTITY=1553731627-tianjin-university WANDB_PROJECT=debug_qbiolip WANDB_MODE=online WANDB_CONSOLE=off && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run --no-capture-output -n protenix311 \
torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.6 \
  --project debug_qbiolip \
  --run_name DDP4-contrast-only-forward-v026-bs4 \
  --base_dir ./output \
  --lr 0.0003 \
  --batch_size 4 \
  --data.batch_size 4 \
  --data.num_dl_workers 0 \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 300 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 200 \
  --chain_permutation.train.mini_rollout False \
  --atom_permutation.train.mini_rollout False \
  --use_wandb True \
  --data.train_sets qbiolip_nonredund
```

### 36) only_forward 兼容性修复（2026-03-04）
- 文件：`runner/train.py`
- 修复：`model_forward` 现在兼容 `self.model(...)` 返回 `pred_dict`（dict）场景。
  - 之前按三元组解包导致报错：`ValueError: too many values to unpack (expected 3)`。
  - 现在新增 `_unpack_model_out`，支持：
    - `(pred_dict, label_dict, log_dict)`
    - `pred_dict`（only-forward）

### 37) only_forward 实跑状态
- 当前运行：`DDP4-contrast-only-forward-v026-bs4_20260304_164751`
- W&B run：`tvp6b88d`
- 观测：训练日志仅包含 contrast 分支指标，结构 loss 不再参与。

## 2026-03-05 新一轮定位与修复（“loss 不下降”专项）

### 38) 问题复述（本轮目标）
- 现象（4 卡 contrast-only）：
  - `train/contrast_loss/raw.avg` 长时间接近 `ln(global_batch)`（例如 global=32 时约 3.465）。
  - `train/contrast_loss/raw_gap_vs_shuffled.avg` 逐步接近 0。
  - `train/contrast_loss/prot_std.avg`、`lig_std.avg` 降到很小（可到 `1e-4` 量级）。
- 解释：
  - 这表示模型进入“接近随机 + 表示塌缩”的区域，不是简单的“学习率太小”。

### 39) 复现实验 A（单卡 20 step，先排除 DDP 干扰）
- 为什么做：
  - 先确认在最小分布式复杂度下是否也有同类趋势，区分“模型问题”还是“DDP 放大问题”。
- 调用命令：
```bash
cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 && \
export NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n protenix311 \
torchrun --standalone --nproc_per_node=1 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.6 \
  --project debug_qbiolip \
  --run_name debug-contrast-v026-n1-20steps \
  --base_dir ./output \
  --batch_size 8 \
  --data.batch_size 8 \
  --data.num_dl_workers 0 \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 20 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 999999999 \
  --chain_permutation.train.mini_rollout False \
  --atom_permutation.train.mini_rollout False \
  --loss.weight.alpha_diffusion 0 \
  --loss.weight.alpha_distogram 0 \
  --loss.weight.alpha_confidence 0 \
  --use_wandb False \
  --data.train_sets qbiolip_nonredund
```
- 结果摘要：
  - 单卡未快速塌缩到完全随机，但 `raw` 仍在 `ln(8)` 附近波动，下降不明显。
  - 梯度连通正常：`esm_contrast_proj/unimol_contrast_proj` 有梯度；`esm_proj/unimol_proj` 无梯度（符合 only-forward）。

### 40) 复现实验 B（4 卡 40 step，复现用户问题）
- 为什么做：
  - 需要在与你一致的 4 卡设置下确认“是否确实很快塌缩”。
- 调用命令：
```bash
cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 && \
export NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run --no-capture-output -n protenix311 \
torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.6 \
  --project debug_qbiolip \
  --run_name debug-contrast-v026-n4-40steps \
  --base_dir ./output \
  --batch_size 8 \
  --data.batch_size 8 \
  --data.num_dl_workers 0 \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 40 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 999999999 \
  --chain_permutation.train.mini_rollout False \
  --atom_permutation.train.mini_rollout False \
  --loss.weight.alpha_diffusion 0 \
  --loss.weight.alpha_distogram 0 \
  --loss.weight.alpha_confidence 0 \
  --use_wandb False \
  --data.train_sets qbiolip_nonredund
```
- 结果摘要：
  - 前几步有信号，随后迅速走向塌缩：
    - `raw -> 3.465...`（接近 `ln(32)`）
    - `raw_gap_vs_shuffled -> 0`
    - `prot_std/lig_std -> 1e-4 ~ 1e-3`
  - 说明“loss 不下降”在 4 卡上可稳定复现。

### 41) 修改 1：`train_projection_only + only_forward` 时仅解冻对比头
- 为什么改：
  - only-forward 路径只用到 `esm_contrast_proj/unimol_contrast_proj`。
  - 原逻辑会额外解冻结构投影头（`esm_proj/unimol_proj/*_to_s_inputs`），这些参数不参与当前 loss，增加优化器噪声与诊断干扰。
- 改了哪里：
  - 文件：`runner/train.py`
  - 位置：`AF3Trainer.init_model` 的 `train_projection_only` 分支。
  - 改动：
    - 新增 `contrast_only_forward` 判断；
    - 当 `only_forward=True` 时，`module_names` 只保留：
      - `esm_contrast_proj`
      - `unimol_contrast_proj`
- 结果：
  - 日志从 “30 个可训练参数张量” 变成 “14 个可训练参数张量”，与当前任务严格一致。

### 42) 修改 2：新增 `v0.2.7` 抗塌缩配置
- 为什么改：
  - `v0.2.6` 在 4 卡下易进入“高置信 + 低方差”退化区，需要更稳的温度和方差约束。
- 改了哪里：
  - 文件：`configs/configs_model_type.py`
  - 新增模型：`protenix_mini_esm650m_unimol_contrast_v0.2.7`
- 具体参数及原因：
  - `contrast.full_grad_gather: False`
    - 原因：在当前对比-only场景里，no-grad gather 更稳，减少分布式梯度放大带来的不稳定。
  - `contrast.init_temp: 0.5`（对应 `logit_scale=2`）
    - 原因：降低初始温度（相较 v0.2.6 的 `init_temp=0.2` -> `scale=5`），避免早期 logits 过尖导致快速塌缩。
  - `contrast.std_floor: 0.03` + `contrast.std_reg_weight: 10.0`
    - 原因：明确提高“反塌缩”约束强度，让 embedding 方差维持在安全区间。
  - 其余保持 contrast-only 主策略不变：
    - `train_projection_only=True`
    - `only_forward=True`
    - `learn_logit_scale=False`

### 43) 一次失败尝试（记录）
- 为什么失败：
  - CLI 参数解析器不接受 `--contrast.full_grad_gather` 这种层级参数覆盖（当前 argparse 注册里没有该 key）。
- 失败命令（原样记录）：
```bash
cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 && \
export NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run --no-capture-output -n protenix311 \
torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.6 \
  --project debug_qbiolip \
  --run_name debug-contrast-v026-n4-lr3e4-nofullgather \
  --base_dir ./output \
  --batch_size 8 \
  --data.batch_size 8 \
  --data.num_dl_workers 0 \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 25 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 999999999 \
  --chain_permutation.train.mini_rollout False \
  --atom_permutation.train.mini_rollout False \
  --loss.weight.alpha_diffusion 0 \
  --loss.weight.alpha_distogram 0 \
  --loss.weight.alpha_confidence 0 \
  --use_wandb False \
  --data.train_sets qbiolip_nonredund \
  --lr 0.0003 \
  --contrast.full_grad_gather False
```
- 报错：`unrecognized arguments: --contrast.full_grad_gather False`

### 44) 验证实验 C（新配置 `v0.2.7`，4 卡 25 step）
- 为什么做：
  - 验证“参数策略 + 解冻策略”能否阻止塌缩。
- 调用命令：
```bash
cd /home/dataset-local/tmp/zsl/Protenix && \
export CC=/usr/bin/gcc CXX=/usr/bin/g++ && \
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 && \
export NCCL_DEBUG=warn NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && \
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run --no-capture-output -n protenix311 \
torchrun --standalone --nproc_per_node=4 runner/train.py \
  --model_name protenix_mini_esm650m_unimol_contrast_v0.2.7 \
  --project debug_qbiolip \
  --run_name debug-contrast-v027-n4-25steps \
  --base_dir ./output \
  --batch_size 8 \
  --data.batch_size 8 \
  --data.num_dl_workers 0 \
  --model.N_cycle 1 \
  --sample_diffusion.N_sample 1 \
  --max_steps 25 \
  --log_interval 1 \
  --eval_interval 999999999 \
  --checkpoint_interval 999999999 \
  --chain_permutation.train.mini_rollout False \
  --atom_permutation.train.mini_rollout False \
  --loss.weight.alpha_diffusion 0 \
  --loss.weight.alpha_distogram 0 \
  --loss.weight.alpha_confidence 0 \
  --use_wandb False \
  --data.train_sets qbiolip_nonredund \
  --lr 0.0003
```
- 结果摘要（关键对比）：
  - 未出现 v0.2.6 那种“方差掉到 1e-4 且 `raw_gap_vs_shuffled=0` 的锁死状态”。
  - 例如：
    - step 23：`raw≈3.098`，`raw_gap_vs_shuffled≈0.263`，`prot_std≈0.038`，`lig_std≈0.038`
    - step 24：`raw≈3.482`，`raw_gap_vs_shuffled≈0.049`
  - 结论：`v0.2.7` 在短程 4 卡实验中显著改善塌缩。

### 45) 运行过程中的中断命令（记录）
- 用途：停止长跑诊断任务，避免占卡。
- 命令：
```bash
pkill -f 'debug-contrast-v026-n4-40steps' || true
```

### 46) 代码正确性快速检查
- 为什么做：
  - 确保本轮改动没有引入语法错误。
- 命令：
```bash
cd /home/dataset-local/tmp/zsl/Protenix && \
python -m py_compile runner/train.py configs/configs_model_type.py
```
- 结果：通过。

### 47) 与 `v0.2.8（learn_logit_scale=True）` 的区别（A/B 计划说明）
- `v0.2.7`（当前）：
  - `learn_logit_scale=False`，温度固定，训练更稳。
- `v0.2.8`（计划）：
  - 仅切换 `learn_logit_scale=True`，其余保持与 `v0.2.7` 一致，做纯 A/B。
- 做 A/B 的原因：
  - 判断“可学习温度”是否能进一步降低 `raw`，同时不引入新的不稳定。
