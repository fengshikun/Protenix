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