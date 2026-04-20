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

import torch.distributed as dist
import datetime
import hashlib
import json
import logging
import os
import time
from argparse import Namespace
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
import wandb
from ml_collections.config_dict import ConfigDict
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import copy
import numpy as np


import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add it to the start of sys.path
sys.path.insert(0, parent_dir)

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_model_type import model_configs
from protenix.config import parse_configs, parse_sys_args
from protenix.config.config import save_config
from protenix.data.dataloader import get_dataloaders
from protenix.metrics.lddt_metrics import (
    LDDTMetrics,
    get_complex_level_rankers,
    kabsch,
    partially_aligned_rmsd,
)
from protenix.model.loss import ProtenixLoss
from protenix.model.protenix import Protenix
from protenix.model import sample_confidence
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.lr_scheduler import FinetuneLRScheduler, get_lr_scheduler
from protenix.utils.metrics import SimpleMetricAggregator
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import autocasting_disable_decorator, to_device
from protenix.utils.training import get_optimizer, is_loss_nan_check
from runner.ema import EMAWrapper

# Disable WANDB's console output capture to reduce unnecessary logging
os.environ["WANDB_CONSOLE"] = "off"

torch.serialization.add_safe_globals([Namespace])

LIGAND_RMSD_EVAL_SETS = {
    "gen_apo_check_pdbbind_test_proteinix_overlap",
    "posebusters_0925",
    "pdbbind_test",
    "pdbbind_test_proteinix_overlap",
    "pbbind_test_v2",
    "posebustersv2",
}


def _base_test_name(test_name: str) -> str:
    return test_name.split("__", 1)[0]


class AF3Trainer(object):
    def __init__(self, configs):
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_log()
        self.init_data()
        self.init_model()
        self.init_loss()
        # self.init_data()
        self.try_load_checkpoint()

    def init_basics(self):
        # Step means effective step considering accumulation
        self.step = 0
        # Global_step equals to self.step * self.iters_to_accumulate
        self.global_step = 0
        self.start_step = 0
        # Add for grad accumulation, it can increase real batch size
        self.iters_to_accumulate = self.configs.iters_to_accumulate

        self.run_name = self.configs.run_name + "_" + time.strftime("%Y%m%d_%H%M%S")
        run_names = DIST_WRAPPER.all_gather_object(
            self.run_name if DIST_WRAPPER.rank == 0 else None
        )
        self.run_name = [name for name in run_names if name is not None][0]
        self.run_dir = f"{self.configs.base_dir}/{self.run_name}"
        self.checkpoint_dir = f"{self.run_dir}/checkpoints"
        self.prediction_dir = f"{self.run_dir}/predictions"
        self.structure_dir = f"{self.run_dir}/structures"
        self.dump_dir = f"{self.run_dir}/dumps"
        self.error_dir = f"{self.run_dir}/errors"

        if DIST_WRAPPER.rank == 0:
            self.print("rank 0 creating directories")
            # import pdb; pdb.set_trace()
            os.makedirs(self.run_dir)
            os.makedirs(self.checkpoint_dir)
            os.makedirs(self.prediction_dir)
            os.makedirs(self.structure_dir)
            os.makedirs(self.dump_dir)
            os.makedirs(self.error_dir)
            save_config(
                self.configs,
                os.path.join(self.configs.base_dir, self.run_name, "config.yaml"),
            )
        
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

            

        self.print(
            f"Using run name: {self.run_name}, run dir: {self.run_dir}, checkpoint_dir: "
            + f"{self.checkpoint_dir}, prediction_dir: {self.prediction_dir}, structure_dir: "
            + f"{self.structure_dir}, error_dir: {self.error_dir}"
        )

    def init_log(self):
        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            wandb.init(
                project=self.configs.project,
                name=self.run_name,
                config=vars(self.configs),
                id=self.configs.wandb_id or None,
            )
        self.train_metric_wrapper = SimpleMetricAggregator(["avg"])

    def init_env(self):
        """Init pytorch/cuda envs."""
        logging.info(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            + f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if DIST_WRAPPER.world_size > 1:
            timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
            dist.init_process_group(
                backend="nccl", timeout=datetime.timedelta(seconds=timeout_seconds)
            )
        if not self.configs.deterministic_seed:
            # use rank-specific seed
            hash_string = f"({self.configs.seed},{DIST_WRAPPER.rank},init_seed)"
            rank_seed = int(hashlib.sha256(hash_string.encode("utf8")).hexdigest(), 16)
            rank_seed = rank_seed % (2**32)
        else:
            rank_seed = self.configs.seed
        seed_everything(
            seed=rank_seed,
            deterministic=self.configs.deterministic,
        )  # diff ddp process got diff seeds

        if self.configs.triangle_attention == "deepspeed":
            env = os.getenv("CUTLASS_PATH", None)
            print(f"env: {env}")
            assert (
                env is not None
            ), "if use ds4sci, set env as https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
        logging.info("Finished init ENV.")

    def init_loss(self):
        self.loss = ProtenixLoss(self.configs)
        self.symmetric_permutation = SymmetricPermutation(
            self.configs, error_dir=self.error_dir
        )
        self.lddt_metrics = LDDTMetrics(self.configs)

    def init_model(self):
        if 'distill_model_config' in self.configs:
            org_model_config = copy.deepcopy(self.configs.model)
            self.configs.model = self.configs.distill_model_config
            self.configs.distill_mode = 2
            self.raw_distill_model = Protenix(self.configs).to(self.device)
            self.configs.model = org_model_config
            self.distill_mode = True
            self.configs.distill_mode = 1
        else:
            self.configs.distill_mode = 0
            self.distill_mode = False
        
        self.raw_model = Protenix(self.configs).to(self.device)

        self.only_diffusion_module_train = self.configs.model.only_diffusion_module_train
        if self.only_diffusion_module_train:
            # 1. 先 freeze 全部参数
            for p in self.raw_model.parameters():
                p.requires_grad = False

            # 2. 只解冻 diffusion_module
            for p in self.raw_model.diffusion_module.parameters():
                p.requires_grad = True

            # 3. 设置 mode
            self.raw_model.eval()                      # 全模型 eval
            self.raw_model.diffusion_module.train()    # diffusion_module 单独 train
            
        self.use_ddp = False
        if DIST_WRAPPER.world_size > 1:
            self.print(f"Using DDP")
            self.use_ddp = True
            # Fix DDP/checkpoint https://discuss.pytorch.org/t/ddp-and-gradient-checkpointing/132244
            self.model = DDP(
                self.raw_model,
                find_unused_parameters=self.configs.find_unused_parameters,
                device_ids=[DIST_WRAPPER.local_rank],
                output_device=DIST_WRAPPER.local_rank,
                static_graph=False,
            )
            if 'distill_model_config' in self.configs:
                self.distill_model = DDP(
                    self.raw_distill_model,
                    find_unused_parameters=self.configs.find_unused_parameters,
                    device_ids=[DIST_WRAPPER.local_rank],
                    output_device=DIST_WRAPPER.local_rank,
                    static_graph=False,
                )
        else:
            self.model = self.raw_model
            if 'distill_model_config' in self.configs:
                self.distill_model = self.raw_distill_model

        def count_parameters(model):
            total_params = sum(p.numel() for p in model.parameters())
            return total_params / 1000.0 / 1000.0

        self.print(f"Model Parameters: {count_parameters(self.model)}")
        if self.configs.get("ema_decay", -1) > 0:
            assert self.configs.ema_decay < 1
            self.ema_wrapper = EMAWrapper(
                self.model,
                self.configs.ema_decay,
                self.configs.ema_mutable_param_keywords,
            )
            self.ema_wrapper.register()

        torch.cuda.empty_cache()
        self.optimizer = get_optimizer(
            self.configs,
            self.model,
            param_names=self.configs.get("finetune_params_with_substring", [""]),
        )
        self.init_scheduler()

    def init_scheduler(self, **kwargs):
        # init finetune lr scheduler if available
        finetune_params = self.configs.get("finetune_params_with_substring", [""])
        is_finetune = len(finetune_params[0]) > 0

        if is_finetune:
            self.lr_scheduler = FinetuneLRScheduler(
                self.optimizer,
                self.configs,
                self.configs.finetune,
                **kwargs,
            )
        else:
            self.lr_scheduler = get_lr_scheduler(self.configs, self.optimizer, **kwargs)

    def init_data(self):
        self.train_dl, self.test_dls = get_dataloaders(
            self.configs,
            DIST_WRAPPER.world_size,
            seed=self.configs.seed,
            error_dir=self.error_dir,
        )

    def save_checkpoint(self, ema_suffix=""):
        if DIST_WRAPPER.rank == 0:
            path = f"{self.checkpoint_dir}/{self.step}{ema_suffix}.pt"
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (
                    self.lr_scheduler.state_dict()
                    if self.lr_scheduler is not None
                    else None
                ),
                "step": self.step,
            }
            torch.save(checkpoint, path)
            self.print(f"Saved checkpoint to {path}")

    def try_load_checkpoint(self):

        def _load_checkpoint(
            checkpoint_path: str,
            load_params_only: bool,
            skip_load_optimizer: bool = False,
            skip_load_step: bool = False,
            skip_load_scheduler: bool = False,
            skip_load_diffusion_module: bool = False,
            load_step_for_scheduler: bool = True,
        ):
            if not os.path.exists(checkpoint_path):
                raise Exception(f"Given checkpoint path not exist [{checkpoint_path}]")
            self.print(
                f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
            )
            checkpoint = torch.load(checkpoint_path, self.device)
            sample_key = [k for k in checkpoint["model"].keys()][0]
            self.print(f"Sampled key: {sample_key}")
            if sample_key.startswith("module.") and not self.use_ddp:
                # DDP checkpoint has module. prefix
                print('replace the model as module......')
                checkpoint["model"] = {
                    k[len("module.") :]: v for k, v in checkpoint["model"].items()
                }

            
            ckpt_state_dict = checkpoint["model"]
            param_group = set()
            for k, v in self.model.named_parameters():
                param_group.add(k.split('.')[0])

            if skip_load_diffusion_module:
                sample_key = [k for k in ckpt_state_dict.keys()][0]
                if sample_key.startswith("module."):
                    diffusion_module_prefix = "module.diffusion_module."
                else:
                    diffusion_module_prefix = "diffusion_module."
                
                
                ckpt_state_dict = {
                    k: v
                    for k, v in ckpt_state_dict.items()
                    if not k.startswith(diffusion_module_prefix)
                }
            

            if self.configs.model.diffusion_module.use_apo_pos and not skip_load_diffusion_module:
                sample_key = [k for k in ckpt_state_dict.keys()][0]
                if sample_key.startswith("module."):
                    key = "module.diffusion_module.atom_attention_encoder.linear_no_bias_r.weight"
                else:
                    print(f'sample_key {sample_key}, and {self.use_ddp}')
                    key = "diffusion_module.atom_attention_encoder.linear_no_bias_r.weight"
                # print ckpt_state_dict keys
                # print(ckpt_state_dict.keys())
                # import pdb; pdb.set_trace()
                # dist.barrier()
                ckpt_w = ckpt_state_dict[key]
                model_w = self.model.state_dict()[key]

                if ckpt_w.shape != model_w.shape:
                    assert ckpt_w.dim() == 2
                    assert ckpt_w.shape[0] == model_w.shape[0]
                    assert ckpt_w.shape[1] < model_w.shape[1]

                    # 新权重：和模型 shape 一致
                    new_w = model_w.clone()          # 保留模型原始初始化
                    new_w.zero_()                    # 你要求后半部分为 0
                    new_w[:, :ckpt_w.shape[1]] = ckpt_w

                    ckpt_state_dict[key] = new_w

                    print(
                        f"[INFO] Expand weight {key}: "
                        f"{tuple(ckpt_w.shape)} -> {tuple(model_w.shape)}"
                    )
            
            missing_keys, unexpected_keys = self.model.load_state_dict(
                    ckpt_state_dict,
                    strict=self.configs.load_strict,
                )

            if skip_load_diffusion_module:
                print(f"[INFO] Skip loading diffusion_module, "
                    f"missing keys: {len(missing_keys)}, "
                    f"unexpected keys: {len(unexpected_keys)}")
            else:
                print(f"[INFO] Load full model, "
                    f"missing keys: {len(missing_keys)}, "
                    f"unexpected keys: {len(unexpected_keys)}")

            
            
            # self.model.load_state_dict(
            #     state_dict=checkpoint["model"],
            #     strict=self.configs.load_strict,
            # )
            if not load_params_only:
                if not skip_load_optimizer:
                    self.print(f"Loading optimizer state")
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                if not skip_load_step:
                    self.print(f"Loading checkpoint step")
                    self.step = checkpoint["step"] + 1
                    self.start_step = self.step
                    self.global_step = self.step * self.iters_to_accumulate
                if not skip_load_scheduler:
                    self.print(f"Loading scheduler state")
                    self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
                elif load_step_for_scheduler:
                    assert (
                        not skip_load_step
                    ), "if load_step_for_scheduler is True, you must load step first"
                    # reinitialize LR scheduler using the updated optimizer and step
                    self.init_scheduler(last_epoch=self.step - 1)

            self.print(f"Finish loading checkpoint, current step: {self.step}")

        # Load EMA model parameters
        if self.configs.load_ema_checkpoint_path:
            _load_checkpoint(
                self.configs.load_ema_checkpoint_path,
                load_params_only=True,
                skip_load_diffusion_module=self.configs.skip_load_diffusion_module
            )
            self.ema_wrapper.register()

        # Load model
        if self.configs.load_checkpoint_path:
            _load_checkpoint(
                self.configs.load_checkpoint_path,
                self.configs.load_params_only,
                skip_load_optimizer=self.configs.skip_load_optimizer,
                skip_load_scheduler=self.configs.skip_load_scheduler,
                skip_load_step=self.configs.skip_load_step,
                load_step_for_scheduler=self.configs.load_step_for_scheduler,
                skip_load_diffusion_module=self.configs.skip_load_diffusion_module
            )
        
        if self.distill_mode:
            # freeze self.model
            # load the embedder part of model to distill_model
            
            if DIST_WRAPPER.world_size > 1:
                src_sd = self.model.module.input_embedder.state_dict()
                self.distill_model.module.input_embedder.load_state_dict(src_sd, strict=True)
                
                for p in self.distill_model.module.input_embedder.parameters():
                    p.requires_grad = False
                
                self.distill_model.module.input_embedder.eval()
                self.model.module.eval()
            else:
                src_sd = self.model.input_embedder.state_dict()
                self.distill_model.input_embedder.load_state_dict(src_sd, strict=True)
                
                for p in self.distill_model.input_embedder.parameters():
                    p.requires_grad = False
                
                self.distill_model.input_embedder.eval()
                self.model.eval()

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logging.info(msg)

    def model_forward(self, batch: dict, mode: str = "train") -> tuple[dict, dict]:
        assert mode in ["train", "eval"]
        
        if mode == 'train' and self.distill_mode:
            try:
                batch['label_dict'] = self.model(
                    input_feature_dict=batch["input_feature_dict"],
                    label_dict=batch["label_dict"],
                    label_full_dict=batch["label_full_dict"],
                    mode=mode,
                    current_step=self.step if mode == "train" else None,
                    symmetric_permutation=self.symmetric_permutation,
                ) # teacher return the mimic label
                batch["pred_dict"], _, log_dict = self.distill_model(
                    input_feature_dict=batch["input_feature_dict"],
                    label_dict=batch["label_dict"],
                    label_full_dict=batch["label_full_dict"],
                    mode=mode,
                    current_step=self.step if mode == "train" else None,
                    symmetric_permutation=self.symmetric_permutation,
                ) # student model return the prediction of feature
            except Exception as e:
                print("❌ Exception caught during model forward:")
                print(e)
                # 打印更完整的堆栈信息（便于调试）
                import traceback
                traceback.print_exc()

                # 打印 batch 基本信息
                if "basic" in batch:
                    print("Batch basic info:")
                    print(batch["basic"])
                else:
                    print("⚠️ batch['basic'] not found.")
                
                # 你可以选择继续抛出异常或跳过这个 batch
                raise
        elif mode == 'eval' and self.distill_mode:
            batch["pred_dict"], batch["label_dict"], log_dict = self.distill_model(
                input_feature_dict=batch["input_feature_dict"],
                label_dict=batch["label_dict"],
                label_full_dict=batch["label_full_dict"],
                mode=mode,
                current_step=self.step if mode == "train" else None,
                symmetric_permutation=self.symmetric_permutation,
            ) # get the z, s, etc. from student model

            batch['input_feature_dict']['s_inputs'] = batch['pred_dict']['s_inputs'].detach()
            batch['input_feature_dict']['s'] = batch['pred_dict']['s'].detach()
            batch['input_feature_dict']['z'] = batch['pred_dict']['z'].detach()
            
            batch["pred_dict"], batch["label_dict"], log_dict = self.model(
                input_feature_dict=batch["input_feature_dict"],
                label_dict=batch["label_dict"],
                label_full_dict=batch["label_full_dict"],
                mode=mode,
                current_step=self.step if mode == "train" else None,
                symmetric_permutation=self.symmetric_permutation,
            ) # get the z, s, etc. from student model
            
            
        else: # eval model
            # self.symmetric_permutation = None
            batch["pred_dict"], batch["label_dict"], log_dict = self.model(
                input_feature_dict=batch["input_feature_dict"],
                label_dict=batch["label_dict"],
                label_full_dict=batch["label_full_dict"],
                mode=mode,
                current_step=self.step if mode == "train" else None,
                symmetric_permutation=self.symmetric_permutation,
            )
        return batch, log_dict

    def get_loss(
        self, batch: dict, mode: str = "train"
    ) -> tuple[torch.Tensor, dict, dict]:
        assert mode in ["train", "eval"]

        loss, loss_dict = autocasting_disable_decorator(self.configs.skip_amp.loss)(
            self.loss
        )(
            feat_dict=batch["input_feature_dict"],
            pred_dict=batch["pred_dict"],
            label_dict=batch["label_dict"],
            mode=mode,
        )
        return loss, loss_dict, batch

    @torch.no_grad()
    def get_metrics(self, batch: dict) -> dict:

        lddt_dict = self.lddt_metrics.compute_lddt(
            batch["pred_dict"], batch["label_dict"]
        )
        # calculate the coordinate of ligand
        if 'interested_ligand_mask' in batch['label_dict']:
            complex_gt = batch['label_dict']['coordinate']
            complex_pred = batch['pred_dict']['coordinate']
            ligand_mask = batch['label_dict']['interested_ligand_mask']
            
            rmsd_dict = self.lddt_metrics.compute_ligand_rmsd_with_kabsch(complex_gt.cpu().numpy(),  complex_pred.cpu().numpy(), ligand_mask[0].cpu().numpy().astype(bool))
            
            rmsd_dict2 = {'ligand_rmsds': []}
            pocket_mask = batch['label_dict']['pocket_mask'][0].cpu().numpy()
            ligand_mask = batch['label_dict']['interested_ligand_mask'][0].cpu().numpy()
            complex_gt = complex_gt.cpu().numpy()
            complex_pred = complex_pred.cpu().numpy()
            for complex_p in complex_pred:
                pocket_rmsd_w_refl, lig_rmsd_w_refl, _, _ = partially_aligned_rmsd(complex_p, complex_gt, align_mask=pocket_mask, rmsd_mask=ligand_mask, reduce=True, allow_reflection=True,)
                rmsd_dict2['ligand_rmsds'].append(lig_rmsd_w_refl)
            
            return lddt_dict, rmsd_dict2

        return lddt_dict

    @torch.no_grad()
    def aggregate_metrics(self, lddt_dict: dict, batch: dict) -> dict:

        simple_metrics, _ = self.lddt_metrics.aggregate_lddt(
            lddt_dict, batch["pred_dict"]["summary_confidence"]
        )

        return simple_metrics

    @torch.no_grad()
    def aggregate_ligand_rmsd_sample_metrics(
        self,
        ligand_rmsds: list[float],
        per_sample_summary_confidence: list[dict],
    ) -> dict[str, torch.Tensor]:
        if not ligand_rmsds:
            return {}

        ligand_rmsd_tensor = torch.as_tensor(ligand_rmsds, dtype=torch.float32)
        n_sample = ligand_rmsd_tensor.size(0)
        median_index = n_sample // 2

        confidence_scores = sample_confidence.merge_per_sample_confidence_scores(
            per_sample_summary_confidence
        )
        complex_level_ranker = get_complex_level_rankers(
            confidence_scores, self.lddt_metrics.complex_ranker_keys
        )

        # For RMSD, smaller is better. Rank1 selectors still follow confidence scores.
        sample_aggregators = {
            "best": lambda x: x.min(dim=0)[0],
            "worst": lambda x: x.max(dim=0)[0],
            "random": lambda x: x.select(dim=0, index=0),
            "mean": lambda x: x.mean(dim=0),
            "median": lambda x: x.sort(dim=0, descending=False)[0].select(
                dim=0, index=median_index
            ),
            **complex_level_ranker,
        }

        ligand_rmsd_lt_2a = (ligand_rmsd_tensor < 2.0).to(torch.float32)
        ligand_rmsd_lt_5a = (ligand_rmsd_tensor < 5.0).to(torch.float32)

        metrics = {}
        for agg_name, agg_func in sample_aggregators.items():
            metrics[f"ligand_rmsd/{agg_name}"] = agg_func(ligand_rmsd_tensor)
            metrics[f"ligand_rmsd_lt_2A/{agg_name}"] = agg_func(ligand_rmsd_lt_2a)
            metrics[f"ligand_rmsd_lt_5A/{agg_name}"] = agg_func(ligand_rmsd_lt_5a)

        return metrics

    @torch.no_grad()
    def aggregate_ligand_rmsd_metrics(
        self,
        test_name: str,
        ligand_rmsds: list[float],
        pdb_rmsd_dict: dict[str, float],
    ) -> dict[str, float]:
        gathered_ligand_rmsds = DIST_WRAPPER.all_gather_object(ligand_rmsds)
        all_ligand_rmsds = [
            float(rmsd)
            for rank_ligand_rmsds in gathered_ligand_rmsds
            for rmsd in rank_ligand_rmsds
        ]

        if len(all_ligand_rmsds) == 0:
            return {}

        ligand_rmsd_array = np.asarray(all_ligand_rmsds, dtype=np.float32)
        ratio_lt_2A = float(np.mean(ligand_rmsd_array < 2.0))
        ratio_lt_5A = float(np.mean(ligand_rmsd_array < 5.0))
        mean_rmsd = float(np.mean(ligand_rmsd_array))
        median_rmsd = float(np.median(ligand_rmsd_array))

        self.print(
            "ratio 2A: "
            f"{ratio_lt_2A}; ratio 5A: {ratio_lt_5A}; "
            f"mean_rmsd: {mean_rmsd}; median_rmsd: {median_rmsd}"
        )

        merged_pdb_rmsd_dict = {}
        for rank_pdb_rmsd_dict in DIST_WRAPPER.all_gather_object(pdb_rmsd_dict):
            merged_pdb_rmsd_dict.update(rank_pdb_rmsd_dict)

        if DIST_WRAPPER.rank == 0 and merged_pdb_rmsd_dict:
            rmsd_path = os.path.join(self.run_dir, f"{test_name}_ligand_rmsd.txt")
            merged_pdb_rmsd_dict = dict(
                sorted(merged_pdb_rmsd_dict.items(), key=lambda item: item[1])
            )
            with open(rmsd_path, "w") as f:
                for pdb_id, rmsd in merged_pdb_rmsd_dict.items():
                    f.write(f"{pdb_id}\t{rmsd}\n")

        return {
            f"{test_name}/ligand_rmsd_lt_2A.avg": ratio_lt_2A,
            f"{test_name}/ligand_rmsd_lt_5A.avg": ratio_lt_5A,
            f"{test_name}/ligand_rmsd.avg": mean_rmsd,
            f"{test_name}/ligand_rmsd.median": median_rmsd,
        }

    def _wrapped_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _clone_eval_value(self, value):
        if torch.is_tensor(value):
            return value.clone()
        if isinstance(value, dict):
            return {k: self._clone_eval_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._clone_eval_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._clone_eval_value(v) for v in value)
        return copy.deepcopy(value)

    def _clone_eval_batch(self, batch: dict) -> dict:
        return self._clone_eval_value(batch)

    def _jsonify_eval_value(self, value):
        if torch.is_tensor(value):
            if value.dim() == 0:
                return value.item()
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {k: self._jsonify_eval_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._jsonify_eval_value(v) for v in value]
        if isinstance(value, tuple):
            return [self._jsonify_eval_value(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return copy.deepcopy(value)

    def _get_test_output_folder(self, test_name: str) -> str:
        test_output_folder = f"{self.configs.base_dir}/test_predictions"
        if "__" in test_name or any("__" in dataset_name for dataset_name in self.test_dls):
            test_output_folder = os.path.join(test_output_folder, test_name)
        return test_output_folder

    def _save_test_predictions(self, test_name: str, batch: dict) -> None:
        test_output_folder = self._get_test_output_folder(test_name)
        os.makedirs(test_output_folder, exist_ok=True)
        pdb_id = batch["basic"]["pdb_id"]
        save_file_name = f"{test_output_folder}/{pdb_id}.npy"
        results_numpy = batch["pred_dict"]["coordinate"].detach().cpu().numpy()
        np.save(save_file_name, results_numpy)

        summary_confidence = batch["pred_dict"].get("summary_confidence", None)
        if summary_confidence:
            ranking_scores = [
                float(each_sample["ranking_score"]) for each_sample in summary_confidence
            ]
            best_sample_index = int(np.argmax(ranking_scores))
            summary_payload = {
                "pdb_id": pdb_id,
                "best_sample_index": best_sample_index,
                "ranking_scores": ranking_scores,
                "summary_confidence": [
                    self._jsonify_eval_value(each_sample)
                    for each_sample in summary_confidence
                ],
            }
            meta_save_file = f"{test_output_folder}/{pdb_id}.summary_confidence.json"
            with open(meta_save_file, "w", encoding="utf-8") as handle:
                json.dump(summary_payload, handle, indent=2, ensure_ascii=True)

    @contextmanager
    def _temporary_inference_n_step(self, n_step: int):
        refs = []
        seen = set()
        for cfg in (
            self.configs,
            getattr(self.raw_model, "configs", None),
            getattr(self._wrapped_model(), "configs", None),
        ):
            if cfg is None or id(cfg) in seen:
                continue
            seen.add(id(cfg))
            refs.append((cfg.sample_diffusion, cfg.sample_diffusion["N_step"]))
            cfg.sample_diffusion["N_step"] = int(n_step)
        try:
            yield
        finally:
            for sample_diffusion_cfg, original_n_step in refs:
                sample_diffusion_cfg["N_step"] = original_n_step

    def _estimate_step1_bootstrap_center(
        self,
        base_batch: dict,
        step1_pred_coordinate: torch.Tensor,
    ) -> torch.Tensor:
        pred_index = int(
            self.configs.ddbm_configs.get("eval_step1_bootstrap_prediction_index", 0)
        )
        if step1_pred_coordinate.dim() == 3:
            pred_index = max(0, min(pred_index, step1_pred_coordinate.size(0) - 1))
            pred_coordinate = step1_pred_coordinate[pred_index]
        else:
            pred_coordinate = step1_pred_coordinate

        apo_atom_array = base_batch["input_feature_dict"]["apo_atom_array"]
        ligand_mask = base_batch["input_feature_dict"]["is_ligand"].bool()
        if "is_protein" in base_batch["input_feature_dict"]:
            protein_mask = base_batch["input_feature_dict"]["is_protein"].bool()
        else:
            protein_mask = ~ligand_mask

        pred_coordinate_np = pred_coordinate.detach().cpu().float().numpy()
        apo_atom_array_np = apo_atom_array.detach().cpu().float().numpy()
        ligand_mask_np = ligand_mask.detach().cpu().numpy().astype(bool)
        protein_mask_np = protein_mask.detach().cpu().numpy().astype(bool)

        aligned_pred_coordinate_np = pred_coordinate_np
        if int(protein_mask_np.sum()) >= 3:
            rotation, translation = kabsch(
                pred_coordinate_np[protein_mask_np],
                apo_atom_array_np[protein_mask_np],
            )
            aligned_pred_coordinate_np = pred_coordinate_np @ rotation + translation

        ligand_coords_np = aligned_pred_coordinate_np[ligand_mask_np]
        if ligand_coords_np.shape[0] == 0:
            raise ValueError("No ligand atoms found for step1 bootstrap center estimation.")

        center_np = ligand_coords_np.mean(axis=0)
        protein_coords_np = aligned_pred_coordinate_np[protein_mask_np]
        if protein_coords_np.shape[0] > 0:
            shell_min = float(
                self.configs.ddbm_configs.get("eval_step1_bootstrap_shell_min", 6.0)
            )
            shell_max = float(
                self.configs.ddbm_configs.get("eval_step1_bootstrap_shell_max", 10.0)
            )
            pairwise_dist = np.linalg.norm(
                protein_coords_np[:, None, :] - ligand_coords_np[None, :, :],
                axis=-1,
            )
            min_dist = pairwise_dist.min(axis=1)
            shell_mask = (min_dist >= shell_min) & (min_dist <= shell_max)
            if np.any(shell_mask):
                center_np = protein_coords_np[shell_mask].mean(axis=0)

        return torch.as_tensor(
            center_np,
            device=apo_atom_array.device,
            dtype=apo_atom_array.dtype,
        )

    def _build_step1_bootstrap_batch(
        self,
        base_batch: dict,
        bootstrap_center: torch.Tensor,
    ) -> dict:
        bootstrap_batch = self._clone_eval_batch(base_batch)
        apo_atom_array = bootstrap_batch["input_feature_dict"]["apo_atom_array"].clone()
        ligand_mask = bootstrap_batch["input_feature_dict"]["is_ligand"].bool()
        ligand_coords = apo_atom_array[ligand_mask]
        ligand_coords = ligand_coords - ligand_coords.mean(dim=0, keepdim=True)
        apo_atom_array[ligand_mask] = ligand_coords + bootstrap_center.unsqueeze(0)
        bootstrap_batch["input_feature_dict"]["apo_atom_array"] = apo_atom_array
        if "apo_center_ligand_to_origin" in bootstrap_batch["input_feature_dict"]:
            bootstrap_batch["input_feature_dict"]["apo_center_ligand_to_origin"] = (
                torch.zeros_like(
                    bootstrap_batch["input_feature_dict"]["apo_center_ligand_to_origin"]
                )
            )
        return bootstrap_batch

    def _build_protein_center_init_batch(self, base_batch: dict) -> dict:
        protein_center_batch = self._clone_eval_batch(base_batch)
        apo_atom_array = protein_center_batch["input_feature_dict"]["apo_atom_array"].clone()
        ligand_mask = protein_center_batch["input_feature_dict"]["is_ligand"].bool()
        ligand_coords = apo_atom_array[ligand_mask]
        ligand_coords = ligand_coords - ligand_coords.mean(dim=0, keepdim=True)
        apo_atom_array[ligand_mask] = ligand_coords
        protein_center_batch["input_feature_dict"]["apo_atom_array"] = apo_atom_array
        if "apo_center_ligand_to_origin" in protein_center_batch["input_feature_dict"]:
            protein_center_batch["input_feature_dict"]["apo_center_ligand_to_origin"] = (
                torch.ones_like(
                    protein_center_batch["input_feature_dict"]["apo_center_ligand_to_origin"]
                )
            )
        return protein_center_batch

    def _evaluate_batch_once(
        self,
        batch: dict,
        test_name: str,
        mode: str,
        enable_amp,
    ) -> tuple[dict, dict, list[float]]:
        simple_metrics = {}
        ligand_rmsd_values: list[float] = []

        with enable_amp:
            batch, _ = self.model_forward(batch, mode=mode)
            self._save_test_predictions(test_name, batch)

            _, loss_dict, batch = self.get_loss(batch, mode="eval")
            if "interested_ligand_mask" in batch["label_dict"]:
                lddt_dict, rmsd_dict = self.get_metrics(batch)
                ligand_rmsd_values = [
                    float(rmsd) for rmsd in rmsd_dict.get("ligand_rmsds", [])
                ]
                simple_metrics.update(
                    self.aggregate_ligand_rmsd_sample_metrics(
                        ligand_rmsds=ligand_rmsd_values,
                        per_sample_summary_confidence=batch["pred_dict"][
                            "summary_confidence"
                        ],
                    )
                )
            else:
                lddt_dict = self.get_metrics(batch)

            lddt_metrics = self.aggregate_metrics(lddt_dict, batch)
            simple_metrics.update(
                {k: v for k, v in lddt_metrics.items() if "diff" not in k}
            )
            simple_metrics.update(loss_dict)

        return batch, simple_metrics, ligand_rmsd_values

    def _evaluate_batch_with_step1_bootstrap(
        self,
        batch: dict,
        test_name: str,
        mode: str,
        enable_amp,
    ) -> tuple[dict, dict, list[float]]:
        step1_n_step = int(
            self.configs.ddbm_configs.get("eval_step1_bootstrap_n_step", 1)
        )
        base_batch = self._clone_eval_batch(batch)
        step1_batch = self._build_protein_center_init_batch(base_batch)

        with self._temporary_inference_n_step(step1_n_step):
            with enable_amp:
                step1_batch, _ = self.model_forward(step1_batch, mode=mode)

        bootstrap_center = self._estimate_step1_bootstrap_center(
            base_batch=base_batch,
            step1_pred_coordinate=step1_batch["pred_dict"]["coordinate"],
        )
        del step1_batch
        torch.cuda.empty_cache()

        bootstrap_batch = self._build_step1_bootstrap_batch(
            base_batch=base_batch,
            bootstrap_center=bootstrap_center,
        )
        return self._evaluate_batch_once(
            batch=bootstrap_batch,
            test_name=test_name,
            mode=mode,
            enable_amp=enable_amp,
        )

    def _should_eval_step1_bootstrap(self, test_name: str, test_dl) -> bool:
        if not bool(self.configs.ddbm_configs.get("eval_with_step1_bootstrap", False)):
            return False
        if test_name.endswith("__step1_bootstrap"):
            return False
        if not (
            self.configs.model.diffusion_module.use_apo_pos
            or self.configs.model.input_embedder.use_apo_pos
        ):
            return False
        dataset = getattr(test_dl, "dataset", None)
        if dataset is None:
            return False
        return getattr(dataset, "apo_ligand_init_mode", "") == "pocket_center"

    def _evaluate_loader(
        self,
        simple_metric_wrapper: SimpleMetricAggregator,
        test_name: str,
        test_dl,
        mode: str,
        enable_amp,
        ema_suffix: str = "",
        use_step1_bootstrap: bool = False,
    ) -> None:
        self.print(f"Testing on {test_name}")
        evaluated_pids = []
        total_batch_num = len(test_dl)
        base_test_name = _base_test_name(test_name)
        if base_test_name in LIGAND_RMSD_EVAL_SETS:
            ligand_rmsds = []
            pdb_rmsd_dict = {}
        else:
            ligand_rmsds = None
            pdb_rmsd_dict = None

        for index, batch in enumerate(tqdm(test_dl)):
            batch = to_device(batch, self.device)
            pid = batch["basic"]["pdb_id"]

            if index + 1 == total_batch_num and DIST_WRAPPER.world_size > 1:
                all_data_ids = DIST_WRAPPER.all_gather_object(evaluated_pids)
                dedup_ids = set(sum(all_data_ids, []))
                if pid in dedup_ids:
                    print(
                        f"Rank {DIST_WRAPPER.rank}: Drop data_id {pid} as it is already evaluated."
                    )
                    break
            evaluated_pids.append(pid)

            if use_step1_bootstrap:
                batch, simple_metrics, batch_ligand_rmsds = (
                    self._evaluate_batch_with_step1_bootstrap(
                        batch=batch,
                        test_name=test_name,
                        mode=mode,
                        enable_amp=enable_amp,
                    )
                )
            else:
                batch, simple_metrics, batch_ligand_rmsds = self._evaluate_batch_once(
                    batch=batch,
                    test_name=test_name,
                    mode=mode,
                    enable_amp=enable_amp,
                )

            if ligand_rmsds is not None:
                ligand_rmsds.extend(batch_ligand_rmsds)
                if batch_ligand_rmsds:
                    pdb_rmsd_dict[batch["basic"]["pdb_id"]] = float(
                        np.mean(batch_ligand_rmsds)
                    )

            for key, value in simple_metrics.items():
                simple_metric_wrapper.add(
                    f"{ema_suffix}{key}", value, namespace=test_name
                )

            del batch, simple_metrics
            if index % 5 == 0:
                torch.cuda.empty_cache()

        metrics = simple_metric_wrapper.calc()
        if ligand_rmsds is not None:
            metrics.update(
                self.aggregate_ligand_rmsd_metrics(
                    test_name=test_name,
                    ligand_rmsds=ligand_rmsds,
                    pdb_rmsd_dict=pdb_rmsd_dict,
                )
            )

        self.print(f"Step {self.step}, eval {test_name}: {metrics}")
        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            wandb.log(metrics, step=self.step)

    @torch.no_grad()
    def evaluate(self, mode: str = "eval"):
        if not self.configs.eval_ema_only:
            self._evaluate(only_save_af3=False)
        # if hasattr(self, "ema_wrapper"):
        #     self.ema_wrapper.apply_shadow()
        #     self._evaluate(ema_suffix=f"ema{self.ema_wrapper.decay}_", mode=mode)
        #     self.ema_wrapper.restore()

    @torch.no_grad()
    def _evaluate(self, ema_suffix: str = "", mode: str = "eval", only_save_af3: bool = False):
        # Init Metric Aggregator
        
        if only_save_af3:
            for test_name, test_dl in self.test_dls.items():
                dataset = test_dl.dataset
                for i in tqdm(range(len(dataset))):
                    try:
                        _ = dataset[i]
                    except Exception as e:
                        print(f'{e} happens')
            return
        
        
        simple_metric_wrapper = SimpleMetricAggregator(["avg"])
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )
        self.model.eval()

        for test_name, test_dl in self.test_dls.items():
            self._evaluate_loader(
                simple_metric_wrapper=simple_metric_wrapper,
                test_name=test_name,
                test_dl=test_dl,
                mode=mode,
                enable_amp=enable_amp,
                ema_suffix=ema_suffix,
                use_step1_bootstrap=False,
            )
            if self._should_eval_step1_bootstrap(test_name, test_dl):
                self._evaluate_loader(
                    simple_metric_wrapper=simple_metric_wrapper,
                    test_name=f"{test_name}__step1_bootstrap",
                    test_dl=test_dl,
                    mode=mode,
                    enable_amp=enable_amp,
                    ema_suffix=ema_suffix,
                    use_step1_bootstrap=True,
                )

    def update(self):
        # Clip the gradient
        if self.configs.grad_clip_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.configs.grad_clip_norm
            )

    def train_step(self, batch: dict):
        self.model.train()
        # FP16 training has not been verified yet
        train_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(
                device_type="cuda", dtype=train_precision, cache_enabled=False
            )
            if torch.cuda.is_available()
            else nullcontext()
        )

        scaler = torch.GradScaler(
            device="cuda" if torch.cuda.is_available() else "cpu",
            enabled=(self.configs.dtype == "float16"),
        )

        with enable_amp:
            batch, _ = self.model_forward(batch, mode="train")
            loss, loss_dict, _ = self.get_loss(batch, mode="train")

        if self.configs.dtype in ["bf16", "fp32"]:
            if is_loss_nan_check(loss):
                self.print(f"Skip iteration with NaN loss: {self.step} steps")
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        scaler.scale(loss / self.iters_to_accumulate).backward()

        # For simplicity, the global training step is used
        if (self.global_step + 1) % self.iters_to_accumulate == 0:
            self.print(
                f"self.step {self.step}, self.iters_to_accumulate: {self.iters_to_accumulate}"
            )
            # Unscales the gradients of optimizer's assigned parameters in-place
            scaler.unscale_(self.optimizer)
            # Do grad clip only
            self.update()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
        for key, value in loss_dict.items():
            if "loss" not in key:
                continue
            self.train_metric_wrapper.add(key, value, namespace="train")
        torch.cuda.empty_cache()

    def progress_bar(self, desc: str = ""):
        if DIST_WRAPPER.rank != 0:
            return
        if self.global_step % (
            self.configs.eval_interval * self.iters_to_accumulate
        ) == 0 or (not hasattr(self, "_ipbar")):
            # Start a new progress bar
            self._pbar = tqdm(
                range(
                    self.global_step
                    % (self.iters_to_accumulate * self.configs.eval_interval),
                    self.iters_to_accumulate * self.configs.eval_interval,
                )
            )
            self._ipbar = iter(self._pbar)

        step = next(self._ipbar)
        self._pbar.set_description(
            f"[step {self.step}: {step}/{self.iters_to_accumulate * self.configs.eval_interval}] {desc}"
        )
        return

    def run(self):
        """
        Main entry for the AF3Trainer.

        This function handles the training process, evaluation, logging, and checkpoint saving.
        """
        if self.configs.eval_only or self.configs.eval_first:
            self.evaluate()
            if self.configs.eval_only:
                return
        use_ema = hasattr(self, "ema_wrapper")
        self.print(f"Using ema: {use_ema}")

        while True:
            for batch in self.train_dl:
                is_update_step = (self.global_step + 1) % self.iters_to_accumulate == 0
                is_last_step = (self.step + 1) == self.configs.max_steps
                step_need_log = (self.step + 1) % self.configs.log_interval == 0

                step_need_eval = (
                    self.configs.eval_interval > 0
                    and (self.step + 1) % self.configs.eval_interval == 0
                )
                step_need_save = (
                    self.configs.checkpoint_interval > 0
                    and (self.step + 1) % self.configs.checkpoint_interval == 0
                )

                is_last_step &= is_update_step
                step_need_log &= is_update_step
                step_need_eval &= is_update_step
                step_need_save &= is_update_step

                batch = to_device(batch, self.device)
                self.progress_bar()
                self.train_step(batch)
                if use_ema and is_update_step:
                    self.ema_wrapper.update()
                if step_need_log or is_last_step:
                    metrics = self.train_metric_wrapper.calc()
                    self.print(f"Step {self.step} train: {metrics}")
                    last_lr = self.lr_scheduler.get_last_lr()
                    if DIST_WRAPPER.rank == 0:
                        if self.configs.use_wandb:
                            lr_dict = {"train/lr": last_lr[0]}
                            for group_i, group_lr in enumerate(last_lr):
                                lr_dict[f"train/group{group_i}_lr"] = group_lr
                            wandb.log(lr_dict, step=self.step)
                        self.print(f"Step {self.step}, lr: {last_lr}")
                    if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                        wandb.log(metrics, step=self.step)

                if step_need_save or is_last_step:
                    self.save_checkpoint()
                    if use_ema:
                        self.ema_wrapper.apply_shadow()
                        self.save_checkpoint(
                            ema_suffix=f"_ema_{self.ema_wrapper.decay}"
                        )
                        self.ema_wrapper.restore()

                if step_need_eval or is_last_step:
                    self.evaluate()
                self.global_step += 1
                if self.global_step % self.iters_to_accumulate == 0:
                    self.step += 1
                if self.step >= self.configs.max_steps:
                    self.print(f"Finish training after {self.step} steps")
                    break
            if self.step >= self.configs.max_steps:
                break


def main():
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    configs_base["triangle_attention"] = os.environ.get(
        "TRIANGLE_ATTENTION", "triattention"
    )
    configs_base["triangle_multiplicative"] = os.environ.get(
        "TRIANGLE_MULTIPLICATIVE", "cuequivariance"
    )
    configs = {**configs_base, **{"data": data_configs}}
    configs = parse_configs(
        configs,
        parse_sys_args(),
    )
    model_name = configs.model_name
    model_specfics_configs = ConfigDict(model_configs[model_name])
    # update model specific configs
    configs.update(model_specfics_configs)

    if configs.distill_model_type:
        distill_model_config = ConfigDict(model_configs[configs.distill_model_type])
        configs['distill_model_config'] = copy.deepcopy(configs['model'])
        configs['distill_model_config'].update(distill_model_config.model)
        
        
        # model_config = configs.get("model_config", {})

        # # 复制缺失项
        # for k, v in model_config.items():
        #     if k not in distill_model_config:
        #         distill_model_config[k] = v

        # configs["distill_model_config"] = distill_model_config
    
    
    print(configs.run_name)
    print(configs)
    trainer = AF3Trainer(configs)
    trainer.run()


if __name__ == "__main__":
    main()
