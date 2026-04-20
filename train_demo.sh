# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export LAYERNORM_TYPE=fast_layernorm # fast_layernorm, torch
# Kernel options:
# - triangle_attention: supports 'triattention', 'cuequivariance', 'deepspeed', 'torch'
# - triangle_multiplicative: supports 'cuequivariance', 'torch'

torchrun \
    --nproc_per_node=8 \
    --master-port 29501 \
     ./runner/train.py \
--run_name protenix_train \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_wandb true \
--wandb_id a46eaf1ea4fdcf3a2a93022568aa1c730c208b50 \
--diffusion_batch_size 48 \
--eval_interval 400 \
--log_interval 50 \
--checkpoint_interval 400 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--triangle_attention "triattention" \
--triangle_multiplicative "cuequivariance" \
--data.train_sets weightedPDB_before2109_wopb_nometalc_0925 \
--data.test_sets recentPDB_1536_sample384_0925,posebusters_0925 \
--data.posebusters_0925.base_info.max_n_token 768


# abi study, use apo pose as condition, pdbbind v1 train and test set
/vepfs-mlp2/mlp-public/shikunfeng/Envs/alphainteract2/bin/torchrun \
--nproc_per_node=8 \
--master-port 29501 \
    ./runner/train.py \
--run_name protenix_mini_pdbbind_v1 \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_wandb true \
--diffusion_batch_size 48 \
--eval_interval 400 \
--log_interval 50 \
--checkpoint_interval 400 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--triangle_attention "triattention" \
--triangle_multiplicative "cuequivariance" \
--data.train_sets pdbbind_prot_ligand \
--data.test_sets pdbbind_test,posebusters_0925 \
--data.posebusters_0925.base_info.max_n_token 768 \
--data.pdbbind_test.base_info.max_n_token 768 \
--model_name protenix_mini_esm_online_v0.1.0  \
--model.input_embedder.use_apo_pos True \
--model.diffusion_module.use_apo_pos True \
--find_unused_parameters True \
> protenix_mini_pdbbind_v1.log 2>&1


# baseline
/vepfs-mlp2/mlp-public/shikunfeng/Envs/alphainteract2/bin/torchrun \
--nproc_per_node=8 \
--master-port 29501 \
    ./runner/train.py \
--run_name protenix_mini_pdbbind_v1_baseline \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_wandb true \
--diffusion_batch_size 48 \
--eval_interval 400 \
--log_interval 50 \
--checkpoint_interval 400 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--triangle_attention "triattention" \
--triangle_multiplicative "cuequivariance" \
--data.train_sets pdbbind_prot_ligand \
--data.test_sets pdbbind_test,posebusters_0925 \
--data.posebusters_0925.base_info.max_n_token 768 \
--data.pdbbind_test.base_info.max_n_token 768 \
--model_name protenix_mini_esm_online_v0.1.0  \
--find_unused_parameters True \
> protenix_mini_pdbbind_v1_baseline.log 2>&1

# --run_name protenix_train --seed 42 --base_dir ./output --dtype bf16 --project protenix --use_wandb false --diffusion_batch_size 48 --eval_interval 2 --log_interval 50 --checkpoint_interval 400 --ema_decay 0.999 --train_crop_size 384 --max_steps 100000 --warmup_steps 2000 --lr 0.001 --sample_diffusion.N_step 20 --triangle_attention triattention --triangle_multiplicative cuequivariance --data.train_sets pdbbind_prot_ligand --model.input_embedder.use_apo_pos True --model.diffusion_module.use_apo_pos True --data.test_sets pdbbind_test --data.posebusters_0925.base_info.max_n_token 768 --data.num_dl_workers 0 --model_name protenix_mini_esm_online_v0.1.0 


# ddbm
/vepfs-mlp2/mlp-public/shikunfeng/Envs/alphainteract2/bin/torchrun \
--nproc_per_node=8 \
--master-port 29501 \
    ./runner/train.py \
--run_name protenix_mini_pdbbind_v1_ddbm \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_wandb true \
--diffusion_batch_size 48 \
--eval_interval 400 \
--log_interval 50 \
--checkpoint_interval 400 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--triangle_attention "triattention" \
--triangle_multiplicative "cuequivariance" \
--data.train_sets pdbbind_prot_ligand \
--data.test_sets pdbbind_test \
--data.posebusters_0925.base_info.max_n_token 768 \
--data.pdbbind_test.base_info.max_n_token 768 \
--model_name protenix_mini_esm_online_v0.1.0  \
--model.input_embedder.use_apo_pos True \
--model.diffusion_module.use_apo_pos True \
--find_unused_parameters True \
--ddbm True \
> protenix_mini_pdbbind_v1_ddbm.log 2>&1



# prepare the dataset
python -u scripts/prepare_training_data.py -i /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/pdbbind_cif_v2.txt -o /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/tools/v2_data/pdbbind_v2.csv -b /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/tools/v2_data/pdbbind/


python -u scripts/prepare_training_data.py -i /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/pdbbind_cif_v2.txt -o /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/tools/v3_data/pdbbind_v3.csv -b /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/tools/v3_data/pdbbind/ > prepare_v3.log 2>&1 &


python -u scripts/prepare_training_data.py -i /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/posebusters_v2_cif.txt -o /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/tools/posebusterv2/posbuster.csv -b /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/tools/posebusterv2/posebusterv2 -p  > posebuster_v2.log 2>&1 &

# use apo structure as condition, finetune from protenix_mini_esm_online_v0.1.0

/vepfs-mlp2/mlp-public/shikunfeng/Envs/alphainteract2/bin/torchrun \
--nproc_per_node=8 \
--master-port 29501 \
    ./runner/train.py \
--run_name protenix_mini_pdbbind_v3_apo_condition_finetune_diffusion \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_wandb true \
--diffusion_batch_size 48 \
--eval_interval 400 \
--log_interval 50 \
--checkpoint_interval 400 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--triangle_attention "triattention" \
--triangle_multiplicative "cuequivariance" \
--data.train_sets pdbbind_prot_ligand \
--data.test_sets pdbbind_test \
--data.posebusters_0925.base_info.max_n_token 768 \
--data.pdbbind_test.base_info.max_n_token 768 \
--model_name protenix_mini_esm_online_v0.1.0  \
--model.diffusion_module.use_apo_pos True \
--find_unused_parameters True \
--load_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--load_ema_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--model.only_diffusion_module_train True \
> protenix_mini_pdbbind_v3_apo_condition_finetune_diffusion.log 2>&1


# Protenix-mini 序列sft ( Only fine-tune diffusion head)

/vepfs-mlp2/mlp-public/shikunfeng/Envs/alphainteract2/bin/torchrun \
--nproc_per_node=8 \
--master-port 29501 \
    ./runner/train.py \
--run_name protenix_mini_pdbbind_v3_finetune_diffusion \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_wandb true \
--diffusion_batch_size 48 \
--eval_interval 400 \
--log_interval 50 \
--checkpoint_interval 400 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--triangle_attention "triattention" \
--triangle_multiplicative "cuequivariance" \
--data.train_sets pdbbind_prot_ligand \
--data.test_sets pdbbind_test,posebusters_0925 \
--data.posebusters_0925.base_info.max_n_token 768 \
--data.pdbbind_test.base_info.max_n_token 768 \
--model_name protenix_mini_esm_online_v0.1.0  \
--find_unused_parameters True \
--load_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--load_ema_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--model.only_diffusion_module_train True \
> protenix_mini_pdbbind_v3_finetune_diffusion.log 2>&1


# Protenix-mini 序列sft (diffusion head train from scratch)
/vepfs-mlp2/mlp-public/shikunfeng/Envs/alphainteract2/bin/torchrun \
--nproc_per_node=8 \
--master-port 29501 \
    ./runner/train.py \
--run_name protenix_mini_pdbbind_v3_finetune_diffusion_scratch \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_wandb true \
--diffusion_batch_size 48 \
--eval_interval 400 \
--log_interval 50 \
--checkpoint_interval 400 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--triangle_attention "triattention" \
--triangle_multiplicative "cuequivariance" \
--data.train_sets pdbbind_prot_ligand \
--data.test_sets pdbbind_test,posebusters_0925 \
--data.posebusters_0925.base_info.max_n_token 768 \
--data.pdbbind_test.base_info.max_n_token 768 \
--model_name protenix_mini_esm_online_v0.1.0  \
--find_unused_parameters True \
--load_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--load_ema_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--model.only_diffusion_module_train True \
--skip_load_diffusion_module True \
> protenix_mini_pdbbind_v3_finetune_diffusion_scratch.log 2>&1

# 4. Bridge train from scratch (only diffusion head)

/vepfs-mlp2/mlp-public/shikunfeng/Envs/alphainteract2/bin/torchrun \
--nproc_per_node=8 \
--master-port 29501 \
    ./runner/train.py \
--run_name protenix_mini_pdbbind_v3_ddbm_vp \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_wandb true \
--diffusion_batch_size 48 \
--eval_interval 400 \
--log_interval 50 \
--checkpoint_interval 400 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--triangle_attention "triattention" \
--triangle_multiplicative "cuequivariance" \
--data.train_sets pdbbind_prot_ligand \
--data.test_sets pdbbind_test \
--data.posebusters_0925.base_info.max_n_token 768 \
--data.pdbbind_test.base_info.max_n_token 768 \
--model_name protenix_mini_esm_online_v0.1.0  \
--find_unused_parameters True \
--load_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--load_ema_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--model.only_diffusion_module_train True \
--skip_load_diffusion_module True \
--model.diffusion_module.use_apo_pos True \
--ddbm True \
> protenix_mini_pdbbind_v3_ddbm_vp.log 2>&1



cd /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix


export CUTLASS_PATH=/opt/cutlass
export WANDB_DIR="/vepfs-mlp2/mlp-public/shikunfeng/Datas/WANDB_LOGS"
export WANDB_API_KEY="a46eaf1ea4fdcf3a2a93022568aa1c730c208b50"

/vepfs-mlp2/mlp-public/shikunfeng/Envs/alphainteract2/bin/torchrun \
--nproc_per_node=8 \
--master-port 29501 \
    ./runner/train.py \
--run_name protenix_mini_pdbbind_v3_ddbm_ve \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_wandb true \
--diffusion_batch_size 48 \
--eval_interval 400 \
--log_interval 50 \
--checkpoint_interval 400 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--triangle_attention "triattention" \
--triangle_multiplicative "cuequivariance" \
--data.train_sets pdbbind_prot_ligand \
--data.test_sets pdbbind_test \
--data.posebusters_0925.base_info.max_n_token 768 \
--data.pdbbind_test.base_info.max_n_token 768 \
--model_name protenix_mini_esm_online_v0.1.0  \
--find_unused_parameters True \
--load_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--load_ema_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--model.only_diffusion_module_train True \
--skip_load_diffusion_module True \
--model.diffusion_module.use_apo_pos True \
--ddbm True \
--ddbm_configs.pred_mode ve \
--ddbm_configs.sigma_max 80.0 \
--ddbm_configs.sigma_min 0.002 \
> protenix_mini_pdbbind_v3_ddbm_ve.log 2>&1




## test command
/vepfs-mlp2/mlp-public/shikunfeng/Envs/alphainteract2/bin/torchrun \
--nproc_per_node=1 \
--master-port 29501 \
    ./runner/train.py \
--run_name protenix_mini_pdbbind_v3_ddbm_ve_test \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_wandb true \
--diffusion_batch_size 48 \
--eval_interval 400 \
--log_interval 50 \
--checkpoint_interval 400 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--triangle_attention "triattention" \
--triangle_multiplicative "cuequivariance" \
--data.train_sets pdbbind_prot_ligand \
--data.test_sets pdbbind_test \
--data.posebusters_0925.base_info.max_n_token 768 \
--data.pdbbind_test.base_info.max_n_token 768 \
--model_name protenix_mini_esm_online_v0.1.0  \
--find_unused_parameters True \
--load_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/protenix_mini_pdbbind_v3_ddbm_ve_sigma_min1_fix_align_20260105_155045/checkpoints/10399.pt \
--model.diffusion_module.use_apo_pos True \
--ddbm True \
--ddbm_configs.pred_mode ve \
--ddbm_configs.sigma_max 80.0 \
--ddbm_configs.sigma_min 1 \
--eval_only True \
> protenix_mini_pdbbind_v3_ddbm_ve_test.log 2>&1



python -u runner/train.py --run_name protenix_train --seed 42 --base_dir ./output --dtype bf16 --project protenix --use_wandb false --diffusion_batch_size 48 --eval_interval 200 --log_interval 50 --checkpoint_interval 400 --ema_decay 0.999 --train_crop_size 384 --max_steps 100000 --warmup_steps 2000 --lr 0.001 --sample_diffusion.N_step 20 --triangle_attention triattention --triangle_multiplicative cuequivariance --data.train_sets pdbbind_prot_ligand --model.diffusion_module.use_apo_pos True --data.test_sets posebustersv2 --data.posebusters_0925.base_info.max_n_token 768 --data.num_dl_workers 0 --eval_only True --load_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/ve_sigma_min0.4_max40_dbim_align_2_20260116_170603/checkpoints/27999_ema_0.999.pt --model_name protenix_mini_esm_online_v0.1.0 --model.only_diffusion_module_train True --ddbm True --ddbm_configs.pred_mode ve --ddbm_configs.sigma_max 40.0 --ddbm_configs.sigma_min 0.4 --ddbm_configs.train_sampler RealUniformSamplerSquare --ddbm_configs.infer_sampler dbim --ddbm_configs.align_af3 True  > ve_sigma_min0.4_max40_dbim_align_2_test_posebustersv2_step20.log 2>&1 &



python -u runner/train.py --run_name protenix_train --seed 42 --base_dir ./output --dtype bf16 --project protenix --use_wandb false --diffusion_batch_size 48 --eval_interval 200 --log_interval 50 --checkpoint_interval 400 --ema_decay 0.999 --train_crop_size 384 --max_steps 100000 --warmup_steps 2000 --lr 0.001 --sample_diffusion.N_step 2 --triangle_attention triattention --triangle_multiplicative cuequivariance --data.train_sets pdbbind_prot_ligand --model.diffusion_module.use_apo_pos False --data.test_sets posebustersv2 --data.posebusters_0925.base_info.max_n_token 768 --data.num_dl_workers 0 --eval_only True --load_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt --model_name protenix_mini_esm_online_v0.1.0   > protenixmin_test_posebustersv2.log 2>&1 &


python -u runner/train.py --run_name protenix_train --seed 42 --base_dir ./output --dtype bf16 --project protenix --use_wandb false --diffusion_batch_size 48 --eval_interval 200 --log_interval 50 --checkpoint_interval 400 --ema_decay 0.999 --train_crop_size 384 --max_steps 100000 --warmup_steps 2000 --lr 0.001 --sample_diffusion.N_step 20 --triangle_attention triattention --triangle_multiplicative cuequivariance --data.train_sets pdbbind_prot_ligand --model.diffusion_module.use_apo_pos True --data.test_sets posebustersv2 --data.posebusters_0925.base_info.max_n_token 768 --data.num_dl_workers 0 --eval_only True --load_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/bridge_exps/protenix_mini_pdbbind_v3_apo_condition_finetune_diffusion_20251226_090013/99999_ema_0.999.pt --model_name protenix_mini_esm_online_v0.1.0   > apo_min_sft_posebustersv2_step20.log 2>&1 &



### use new data to align
cd /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix


export CUTLASS_PATH=/opt/cutlass
export WANDB_DIR="/vepfs-mlp2/mlp-public/shikunfeng/Datas/WANDB_LOGS"
export WANDB_API_KEY="a46eaf1ea4fdcf3a2a93022568aa1c730c208b50"

/vepfs-mlp2/mlp-public/shikunfeng/Envs/alphainteract2/bin/torchrun \
--nproc_per_node=8 \
--master-port 29501 \
    ./runner/train.py \
--run_name ve_sigma_min0.4_max40_dbim_align_4w_apo \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_wandb true \
--diffusion_batch_size 48 \
--eval_interval 400 \
--log_interval 50 \
--checkpoint_interval 4000 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--triangle_attention "triattention" \
--triangle_multiplicative "cuequivariance" \
--data.train_sets weightedPDB_4w_prot_lig_apo \
--data.test_sets pdbbind_test,posebustersv2 \
--data.posebustersv2.base_info.max_n_token 768 \
--data.pdbbind_test.base_info.max_n_token 768 \
--model_name protenix_mini_esm_online_v0.1.0  \
--find_unused_parameters True \
--load_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--load_ema_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--model.only_diffusion_module_train True \
--model.diffusion_module.use_apo_pos True \
--ddbm True \
--ddbm_configs.pred_mode ve \
--ddbm_configs.sigma_max 40.0 \
--ddbm_configs.sigma_min 0.4 \
--ddbm_configs.align_af3 True \
> ve_sigma_min0.4_max40_dbim_align_4w_apo.log 2>&1

/vepfs-mlp2/mlp-public/shikunfeng/Envs/alphainteract2/bin/torchrun \
--nproc_per_node=8 \
--master-port 29502 \
    ./runner/train.py \
--run_name ve_sigma_min0.4_max40_dbim_align_4w_apo_rmsd_lt_100 \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_wandb true \
--diffusion_batch_size 48 \
--eval_interval 400 \
--log_interval 50 \
--checkpoint_interval 4000 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--triangle_attention "triattention" \
--triangle_multiplicative "cuequivariance" \
--data.train_sets weightedPDB_4w_prot_lig_apo_rmsd_lt_100 \
--data.test_sets pdbbind_test,posebustersv2 \
--data.posebustersv2.base_info.max_n_token 768 \
--data.pdbbind_test.base_info.max_n_token 768 \
--model_name protenix_mini_esm_online_v0.1.0  \
--find_unused_parameters True \
--load_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--load_ema_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--model.only_diffusion_module_train True \
--model.diffusion_module.use_apo_pos True \
--ddbm True \
--ddbm_configs.pred_mode ve \
--ddbm_configs.sigma_max 40.0 \
--ddbm_configs.sigma_min 0.4 \
--ddbm_configs.align_af3 True \
> ve_sigma_min0.4_max40_dbim_align_4w_apo_rmsd_lt_100.log 2>&1

/vepfs-mlp2/mlp-public/shikunfeng/Envs/alphainteract2/bin/torchrun \
--nproc_per_node=8 \
--master-port 29503 \
    ./runner/train.py \
--run_name ve_sigma_min0.4_max40_dbim_align_4w_apo_rmsd_lt_30 \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_wandb true \
--diffusion_batch_size 48 \
--eval_interval 400 \
--log_interval 50 \
--checkpoint_interval 4000 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--triangle_attention "triattention" \
--triangle_multiplicative "cuequivariance" \
--data.train_sets weightedPDB_4w_prot_lig_apo_rmsd_lt_30 \
--data.test_sets pdbbind_test,posebustersv2 \
--data.posebustersv2.base_info.max_n_token 768 \
--data.pdbbind_test.base_info.max_n_token 768 \
--model_name protenix_mini_esm_online_v0.1.0  \
--find_unused_parameters True \
--load_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--load_ema_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--model.only_diffusion_module_train True \
--model.diffusion_module.use_apo_pos True \
--ddbm True \
--ddbm_configs.pred_mode ve \
--ddbm_configs.sigma_max 40.0 \
--ddbm_configs.sigma_min 0.4 \
--ddbm_configs.align_af3 True \
> ve_sigma_min0.4_max40_dbim_align_4w_apo_rmsd_lt_30.log 2>&1


## use 4w apo data to finetune
/vepfs-mlp2/mlp-public/shikunfeng/Envs/alphainteract2/bin/torchrun \
--nproc_per_node=8 \
--master-port 29501 \
    ./runner/train.py \
--run_name protenix_mini_4w_apo_condition_finetune_diffusion \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_wandb true \
--diffusion_batch_size 48 \
--eval_interval 400 \
--log_interval 50 \
--checkpoint_interval 400 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--triangle_attention "triattention" \
--triangle_multiplicative "cuequivariance" \
--data.train_sets weightedPDB_4w_prot_lig_apo \
--data.test_sets pdbbind_test,posebustersv2  \
--data.posebusters_0925.base_info.max_n_token 768 \
--data.pdbbind_test.base_info.max_n_token 768 \
--model_name protenix_mini_esm_online_v0.1.0  \
--model.diffusion_module.use_apo_pos True \
--find_unused_parameters True \
--load_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--load_ema_checkpoint_path /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt \
--model.only_diffusion_module_train True > protenix_mini_4w_apo_condition_finetune_diffusion.log 2>&1
