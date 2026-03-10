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
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
try:
    import wandb
except Exception:
    wandb = None
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
from protenix.metrics.lddt_metrics import LDDTMetrics
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.lr_scheduler import FinetuneLRScheduler, get_lr_scheduler
from protenix.utils.metrics import SimpleMetricAggregator
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import autocasting_disable_decorator, to_device
from protenix.utils.training import get_optimizer, is_loss_nan_check
from runner.ema import EMAWrapper
from scripts.benchmark.eval_protenix_vs import evaluate_dataset

# Disable WANDB's console output capture to reduce unnecessary logging
os.environ["WANDB_CONSOLE"] = "off"

torch.serialization.add_safe_globals([Namespace])


class InMemoryProtenixEmbedder:
    def __init__(self, model, device):
        from protenix.data.compute_esm import compute_esm2_embeddings_online
        from protenix.model.unimol.models.dataset import load_mols_dataset

        self.model = model
        self.device = device
        self.compute_esm2_embeddings_online = compute_esm2_embeddings_online
        self.load_mols_dataset = load_mols_dataset
        self.prot_proj = (
            self.model.esm_contrast_proj
            if self.model.esm_contrast_proj is not None
            else self.model.esm_proj
        )
        self.lig_proj = (
            self.model.unimol_contrast_proj
            if self.model.unimol_contrast_proj is not None
            else self.model.unimol_proj
        )
        if self.prot_proj is None or self.lig_proj is None:
            raise RuntimeError("Missing ESM/UniMol projection heads for LIT-PCBA eval.")
        if not getattr(self.model, "esm_model_online", False):
            raise RuntimeError("LIT-PCBA eval requires online ESM model loading.")
        if not getattr(self.model, "use_unimol", False):
            raise RuntimeError("LIT-PCBA eval requires UniMol model.")

    @torch.no_grad()
    def encode_protein_sequence(self, sequence: str) -> np.ndarray:
        embs = self.compute_esm2_embeddings_online(
            model=self.model.esm_model,
            alphabet=self.model.alphabet,
            labels=["0"],
            sequences=[sequence],
            trainable=False,
            truncation_seq_length=self.model.truncation_seq_length,
        )
        pooled = self.model._pool_esm_global(embs)
        if pooled is None:
            raise RuntimeError("Failed to build ESM global embedding for LIT-PCBA eval.")
        pooled = pooled.to(device=self.device, dtype=torch.float32)
        prot_z = self.prot_proj(pooled)
        prot_z = F.normalize(prot_z, dim=-1)
        return prot_z.detach().cpu().numpy()

    @torch.no_grad()
    def encode_ligands(self, lig_records: list[dict], batch_size: int) -> np.ndarray:
        dataset_dict = []
        for rec in lig_records:
            atoms = np.asarray(rec["atoms"])
            coords = torch.as_tensor(rec["coordinates"], dtype=torch.float32)
            dataset_dict.append({"atoms": atoms, "coordinates": coords})

        ds = self.load_mols_dataset(dataset_dict, self.model.unimol_model.dictionary)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, collate_fn=ds.collater, shuffle=False
        )

        out = []
        for sample in loader:
            st = sample["net_input"]["mol_src_tokens"].to(self.device)
            dist = sample["net_input"]["mol_src_distance"].to(self.device)
            et = sample["net_input"]["mol_src_edge_type"].to(self.device)
            pad_mask = st.eq(self.model.unimol_model.padding_idx)
            x = self.model.unimol_model.embed_tokens(st)
            n = dist.size(-1)
            gbf = self.model.unimol_model.gbf(dist, et)
            gab = (
                self.model.unimol_model.gbf_proj(gbf)
                .permute(0, 3, 1, 2)
                .contiguous()
                .view(-1, n, n)
            )
            token_states = self.model.unimol_model.encoder(
                x, padding_mask=pad_mask, attn_mask=gab
            )[0]
            lig_global = token_states[:, 0, :].to(dtype=torch.float32)
            lig_z = self.lig_proj(lig_global)
            lig_z = F.normalize(lig_z, dim=-1)
            out.append(lig_z.detach().cpu().numpy())
        return np.concatenate(out, axis=0)


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
        self.benchmark_eval_dir = f"{self.run_dir}/benchmark_eval"
        self.benchmark_history_path = f"{self.run_dir}/benchmark_eval_history.json"
        self.eval_schedule_path = f"{self.run_dir}/eval_schedule.json"

        if DIST_WRAPPER.rank == 0:
            self.print("rank 0 creating directories")
            # import pdb; pdb.set_trace()
            os.makedirs(self.run_dir)
            os.makedirs(self.checkpoint_dir)
            os.makedirs(self.prediction_dir)
            os.makedirs(self.structure_dir)
            os.makedirs(self.dump_dir)
            os.makedirs(self.error_dir)
            os.makedirs(self.benchmark_eval_dir)
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
        self.eval_step_schedule = self._build_eval_step_schedule(self.configs.max_steps)
        if DIST_WRAPPER.rank == 0:
            with open(self.eval_schedule_path, "w", encoding="utf-8") as f:
                json.dump(self.eval_step_schedule, f, indent=2, ensure_ascii=False)
        self.print(f"Eval step schedule: {self.eval_step_schedule}")

    def _build_eval_step_schedule(self, max_steps: int) -> list[int]:
        schedule = set()
        for step in getattr(self.configs, "extra_eval_steps", []):
            try:
                step = int(step)
            except Exception:
                continue
            if step > 0 and step <= max_steps:
                schedule.add(step)

        interval = int(getattr(self.configs, "extra_eval_interval", 0) or 0)
        if interval > 0:
            schedule.update(range(interval, max_steps + 1, interval))

        return sorted(schedule)

    def _get_eval_triggers(self, next_step: int) -> list[str]:
        triggers = []
        if self.configs.eval_interval > 0 and next_step % self.configs.eval_interval == 0:
            triggers.append("eval_interval")
        if next_step in getattr(self, "eval_step_schedule", []):
            triggers.append("custom_eval_schedule")
        return triggers

    def init_log(self):
        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            wandb.init(
                project=self.configs.project,
                name=self.run_name,
                config=vars(self.configs),
                id=self.configs.wandb_id or None,
            )
        #self.train_metric_wrapper = SimpleMetricAggregator(["avg"])
        # 改成：训练不做对象 gather（避免 all_gather_object 的 GPU OOM）
        self.train_metric_wrapper = SimpleMetricAggregator(["avg"], need_gather=False)


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
        from protenix.model.loss import ProtenixLoss
        self.loss = ProtenixLoss(self.configs)
        self.symmetric_permutation = SymmetricPermutation(
            self.configs, error_dir=self.error_dir
        )
        self.lddt_metrics = LDDTMetrics(self.configs)

    def init_model(self):
        from protenix.model.protenix import Protenix
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

        # Optionally freeze backbone and only train projection layers (ESM/UniMol projection + logit_scale)
        contrast_cfg = getattr(self.configs, "contrast", {})
        if isinstance(contrast_cfg, dict):
            train_proj_only = contrast_cfg.get("train_projection_only", False)
            contrast_only_forward = contrast_cfg.get("only_forward", False)
        else:
            train_proj_only = getattr(contrast_cfg, "get", lambda k, d: d)("train_projection_only", False)
            contrast_only_forward = getattr(contrast_cfg, "get", lambda k, d: d)("only_forward", False)

        if train_proj_only:
            # 1. freeze all params
            for p in self.raw_model.parameters():
                p.requires_grad = False

            # 2. unfreeze only encoder + shared MLP.
            # NOTE:
            # - shared projection heads: esm_proj / unimol_proj
            # - keep Protenix trunk frozen, including *_proj_to_s_inputs
            # - optional encoder finetune: controlled by esm_trainable / unimol_trainable
            module_names = [
                "esm_proj",
                "unimol_proj",
            ]
            partial_train_modules = []
            partial_train_messages = []

            # If requested in model config, also finetune frozen encoders while keeping
            # Protenix trunk frozen.
            if bool(getattr(self.raw_model, "esm_trainable", False)):
                esm_last_n = int(
                    getattr(self.raw_model, "esm_trainable_last_n_layers", -1)
                )
                esm_model = getattr(self.raw_model, "esm_model", None)
                esm_layers = getattr(esm_model, "layers", None) if esm_model is not None else None
                if esm_last_n > 0 and esm_layers is not None and len(esm_layers) > 0:
                    n_layers = len(esm_layers)
                    start_idx = max(0, n_layers - esm_last_n)
                    for layer_idx in range(start_idx, n_layers):
                        layer = esm_layers[layer_idx]
                        for p in layer.parameters():
                            p.requires_grad = True
                        partial_train_modules.append(
                            (f"esm_model.layers.{layer_idx}", layer)
                        )
                    emb_ln_after = getattr(esm_model, "emb_layer_norm_after", None)
                    if emb_ln_after is not None:
                        for p in emb_ln_after.parameters():
                            p.requires_grad = True
                        partial_train_modules.append(
                            ("esm_model.emb_layer_norm_after", emb_ln_after)
                        )
                    partial_train_messages.append(
                        "[train_projection_only] unfreeze esm_model last "
                        f"{n_layers - start_idx} layers: {start_idx}-{n_layers - 1}"
                    )
                else:
                    module_names.append("esm_model")
            if bool(getattr(self.raw_model, "use_unimol", False)) and bool(
                getattr(self.raw_model, "unimol_trainable", False)
            ):
                module_names.append("unimol_model")
            unfreeze_modules = []
            _seen_module_ids = set()
            for module_name in module_names:
                module = getattr(self.raw_model, module_name, None)
                if module is None:
                    continue
                mid = id(module)
                if mid in _seen_module_ids:
                    continue
                _seen_module_ids.add(mid)
                unfreeze_modules.append((module_name, module))

            # also unfreeze logit scale only when it is a Parameter
            if isinstance(getattr(self.raw_model, "logit_scale", None), torch.nn.Parameter):
                self.raw_model.logit_scale.requires_grad = True

            for _, m in unfreeze_modules:
                for p in m.parameters():
                    p.requires_grad = True

            # set appropriate train/eval modes: backbone eval, projection train
            self.raw_model.eval()
            for module_name, m in unfreeze_modules:
                m.train()
                self.print(f"[train_projection_only] unfreeze module: {module_name}")
            for module_name, m in partial_train_modules:
                m.train()
            for msg in partial_train_messages:
                self.print(msg)

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
            find_unused = bool(getattr(self.configs, "find_unused_parameters", False))
            self.model = DDP(
                self.raw_model,
                find_unused_parameters=find_unused,
                device_ids=[DIST_WRAPPER.local_rank],
                output_device=DIST_WRAPPER.local_rank,
                static_graph=(not find_unused),
            )
            if 'distill_model_config' in self.configs:
                self.distill_model = DDP(
                    self.raw_distill_model,
                    find_unused_parameters=find_unused,
                    device_ids=[DIST_WRAPPER.local_rank],
                    output_device=DIST_WRAPPER.local_rank,
                    static_graph=(not find_unused),
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
        # Debug print: when training only projection MLPs, print optimizer param groups and requires_grad
        try:
            if train_proj_only:
                trainable = [
                    (name, p.shape, p.requires_grad)
                    for name, p in self.model.named_parameters()
                    if p.requires_grad
                ]
                self.print(f"train_proj_only active: {len(trainable)} parameter tensors trainable")
                for name, shape, req in trainable:
                    self.print(f"param: {name}, shape={tuple(shape)}, requires_grad={req}")
                for i, g in enumerate(self.optimizer.param_groups):
                    n = sum(p.numel() for p in g["params"])
                    self.print(f"optimizer.param_group[{i}] size={n}, lr={g.get('lr', None)}")
        except Exception as e:
            self.print(f"Failed to print optimizer param groups: {e}")

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
            
            if not self.configs.load_strict:
                model_state_dict = self.model.state_dict()
                shape_mismatch_keys = []
                filtered_ckpt_state_dict = {}
                for k, v in ckpt_state_dict.items():
                    if k in model_state_dict and model_state_dict[k].shape != v.shape:
                        shape_mismatch_keys.append(
                            (k, tuple(v.shape), tuple(model_state_dict[k].shape))
                        )
                        continue
                    filtered_ckpt_state_dict[k] = v
                ckpt_state_dict = filtered_ckpt_state_dict

                if shape_mismatch_keys:
                    preview_n = 20
                    self.print(
                        f"[WARN] Skip {len(shape_mismatch_keys)} mismatched checkpoint tensors "
                        f"because load_strict=False"
                    )
                    for k, ckpt_shape, model_shape in shape_mismatch_keys[:preview_n]:
                        self.print(
                            f"[WARN] shape mismatch: {k}, ckpt={ckpt_shape}, model={model_shape}"
                        )
                    if len(shape_mismatch_keys) > preview_n:
                        self.print(
                            f"[WARN] ... truncated {len(shape_mismatch_keys) - preview_n} more mismatched tensors"
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

        def _unpack_model_out(model_out):
            if isinstance(model_out, tuple) and len(model_out) == 3:
                return model_out
            if isinstance(model_out, dict):
                # contrast-only-forward path returns only pred_dict
                return model_out, batch["label_dict"], {}
            raise ValueError(f"Unexpected model output type: {type(model_out)}")

        if mode == 'train' and self.distill_mode:
            try:
                teacher_out = self.model(
                    input_feature_dict=batch["input_feature_dict"],
                    label_dict=batch["label_dict"],
                    label_full_dict=batch["label_full_dict"],
                    mode=mode,
                    current_step=self.step if mode == "train" else None,
                    symmetric_permutation=self.symmetric_permutation,
                ) # teacher return the mimic label
                if isinstance(teacher_out, dict):
                    batch["label_dict"] = teacher_out
                elif isinstance(teacher_out, tuple):
                    if len(teacher_out) >= 2 and isinstance(teacher_out[1], dict):
                        batch["label_dict"] = teacher_out[1]
                    elif len(teacher_out) >= 1 and isinstance(teacher_out[0], dict):
                        batch["label_dict"] = teacher_out[0]
                    else:
                        raise ValueError(
                            f"Unexpected teacher output tuple for distill train: {type(teacher_out)}"
                        )
                else:
                    raise ValueError(
                        f"Unexpected teacher output type for distill train: {type(teacher_out)}"
                    )
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
            
            model_out = self.model(
                input_feature_dict=batch["input_feature_dict"],
                label_dict=batch["label_dict"],
                label_full_dict=batch["label_full_dict"],
                mode=mode,
                current_step=self.step if mode == "train" else None,
                symmetric_permutation=self.symmetric_permutation,
            ) # get the z, s, etc. from student model
            batch["pred_dict"], batch["label_dict"], log_dict = _unpack_model_out(model_out)
            
            
        else: # eval model
            # self.symmetric_permutation = None
            model_out = self.model(
                input_feature_dict=batch["input_feature_dict"],
                label_dict=batch["label_dict"],
                label_full_dict=batch["label_full_dict"],
                mode=mode,
                current_step=self.step if mode == "train" else None,
                symmetric_permutation=self.symmetric_permutation,
            )
            batch["pred_dict"], batch["label_dict"], log_dict = _unpack_model_out(model_out)
        return batch, log_dict

    def get_loss(
        self, batch: dict, mode: str = "train"
    ) -> tuple[torch.Tensor, dict, dict]:
        assert mode in ["train", "eval"]
        batch["input_feature_dict"]["current_step"] = int(self.step)

        loss, loss_dict = autocasting_disable_decorator(self.configs.skip_amp.loss)(
            self.loss
        )(
            feat_dict=batch["input_feature_dict"],
            pred_dict=batch["pred_dict"],
            label_dict=batch["label_dict"],
            mode=mode,
        )
        return loss, loss_dict, batch

    def _merge_loss_dicts_mean(self, loss_dicts: list[dict]) -> dict:
        if len(loss_dicts) == 1:
            return loss_dicts[0]
        merged = {}
        all_keys = set()
        for d in loss_dicts:
            all_keys.update(d.keys())
        for k in all_keys:
            vals = []
            device = None
            for d in loss_dicts:
                if k not in d:
                    continue
                v = d[k]
                if torch.is_tensor(v):
                    if device is None:
                        device = v.device
                    vals.append(v.detach().float())
                else:
                    if device is None:
                        device = self.device
                    vals.append(torch.tensor(float(v), device=device, dtype=torch.float32))
            if len(vals) == 0:
                continue
            vals = [x.to(device=device, dtype=torch.float32) for x in vals]
            merged[k] = torch.stack(vals).mean()
        return merged

    def _compute_group_contrast_loss(self, batch_list: list[dict]) -> tuple[torch.Tensor, dict]:
        device = self.device
        D = 256
        for b in batch_list:
            pred = b.get("pred_dict", {})
            for kk in ["contrast_prot_z", "contrast_lig_z"]:
                if kk in pred and isinstance(pred[kk], torch.Tensor) and pred[kk].dim() >= 1:
                    D = int(pred[kk].shape[-1])
                    device = pred[kk].device
                    break

        def _to_scalar_text(value):
            try:
                if torch.is_tensor(value):
                    if value.numel() == 0:
                        return None
                    value = value.reshape(-1)[0].item()
                elif isinstance(value, (list, tuple)):
                    if len(value) == 0:
                        return None
                    value = value[0]
                value = str(value)
            except Exception:
                return None
            value = value.strip()
            return value if value else None

        def _extract_pair_id(sample: dict, idx: int) -> tuple[str, bool]:
            """Best-effort sample id extraction for alignment diagnostics."""
            candidate_dicts = []
            if isinstance(sample, dict):
                candidate_dicts.extend(
                    [
                        sample.get("basic", {}),
                        sample.get("input_feature_dict", {}),
                        sample.get("label_dict", {}),
                        sample.get("label_full_dict", {}),
                        sample,
                    ]
                )
            for src in candidate_dicts:
                if not isinstance(src, dict):
                    continue
                for key in (
                    "pair_id",
                    "sample_id",
                    "data_id",
                    "uid",
                    "id",
                    "name",
                    "pdb_id",
                    "assembly_id",
                    "chain_1_id",
                    "chain_2_id",
                    "ligand_id",
                ):
                    if key not in src:
                        continue
                    value = src[key]
                    value = _to_scalar_text(value)
                    if value is None:
                        continue
                    if value:
                        return value, False
            # Compose a stable id from nested "basic" metadata when direct keys are absent.
            try:
                b = sample.get("basic", {}) if isinstance(sample, dict) else {}
                if isinstance(b, dict):
                    pid = str(b.get("pdb_id", "")).strip()
                    aid = str(b.get("assembly_id", "")).strip()
                    c1 = str(b.get("chain_1_id", "")).strip()
                    c2 = str(b.get("chain_2_id", "")).strip()
                    lid = str(b.get("ligand_id", "")).strip()
                    comp = [x for x in [pid, aid, c1, c2, lid] if x]
                    if len(comp) > 0:
                        return "|".join(comp), False
            except Exception:
                pass
            return f"fallback_idx_{idx}", True

        def _stable_uid_hash(text: str, salt: str) -> int:
            digest = hashlib.blake2b(
                f"{salt}|{text}".encode("utf-8"), digest_size=8
            ).digest()
            return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)

        def _extract_uid(
            sample: dict,
            idx: int,
            uid_type: str,
            pair_id: str,
            lig_z_hint: torch.Tensor | None = None,
        ) -> tuple[int, bool]:
            candidate_dicts = []
            if isinstance(sample, dict):
                candidate_dicts.extend(
                    [
                        sample.get("basic", {}),
                        sample.get("input_feature_dict", {}),
                        sample.get("label_dict", {}),
                        sample.get("label_full_dict", {}),
                        sample,
                    ]
                )

            if uid_type == "prot":
                keys = [
                    "prot_uid",
                    "protein_uid",
                    "pocket_uid",
                    "protein_id",
                    "pdb_id",
                    "bioassembly_dict_fpath",
                    "chain_id",
                    "entity_id",
                ]
            else:
                keys = [
                    "lig_uid",
                    "ligand_uid",
                    "ligand_id",
                    "ligand_name",
                    "entity_2_id",
                    "smiles",
                    "canonical_smiles",
                    "inchi_key",
                    "inchi",
                    "ccd_id",
                    "res_name",
                ]

            for src in candidate_dicts:
                if not isinstance(src, dict):
                    continue
                for key in keys:
                    if key not in src:
                        continue
                    value = _to_scalar_text(src[key])
                    if value is None:
                        continue
                    return _stable_uid_hash(value, uid_type), False

            if uid_type == "lig":
                try:
                    feat = sample.get("input_feature_dict", {}) if isinstance(sample, dict) else {}
                    is_lig = feat.get("is_ligand", None)
                    ref_element = feat.get("ref_element", None)
                    if isinstance(is_lig, torch.Tensor) and isinstance(ref_element, torch.Tensor):
                        is_lig = is_lig.view(-1) > 0
                        ref_element = ref_element.view(-1).to(dtype=torch.long)
                        if is_lig.numel() == ref_element.numel() and is_lig.any():
                            lig_ele = ref_element[is_lig].clamp(min=0, max=255)
                            hist = torch.bincount(lig_ele, minlength=256)
                            nz = torch.nonzero(hist, as_tuple=False).view(-1)
                            sig = ";".join([f"{int(i)}:{int(hist[i].item())}" for i in nz[:64]])
                            if len(sig) > 0:
                                return _stable_uid_hash(sig, "lig_sig"), False
                except Exception:
                    pass
                try:
                    if isinstance(lig_z_hint, torch.Tensor):
                        zz = lig_z_hint.detach().float()
                        if zz.dim() == 2 and zz.size(0) > 0:
                            zz = zz.mean(dim=0)
                        zz = zz.view(-1)
                        if zz.numel() > 0:
                            q = torch.round(zz[:128] * 100.0).to(dtype=torch.int32).tolist()
                            sig = ",".join(str(int(x)) for x in q)
                            if len(sig) > 0:
                                return _stable_uid_hash(sig, "lig_emb"), False
                except Exception:
                    pass

            fallback = f"{pair_id}|idx={idx}|{uid_type}"
            return _stable_uid_hash(fallback, uid_type), True

        def _to_row(x: torch.Tensor | None) -> torch.Tensor:
            if not isinstance(x, torch.Tensor):
                return torch.zeros((1, D), device=device, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            elif x.dim() > 2:
                x = x.reshape(-1, x.shape[-1])
            if x.dim() != 2:
                x = torch.zeros((1, D), device=device, dtype=torch.float32)
            if x.shape[-1] != D:
                if x.shape[-1] > D:
                    x = x[..., :D]
                else:
                    x = F.pad(x, (0, D - x.shape[-1]))
            if x.shape[0] == 0:
                x = torch.zeros((1, D), device=device, dtype=torch.float32)
            # If multiple rows are present unexpectedly, use mean pooling instead of
            # taking the first row to avoid index bias in contrast pairing.
            return x.to(device=device, dtype=torch.float32).mean(dim=0, keepdim=True)

        prot_rows = []
        lig_rows = []
        valid_rows = []
        missing = {
            "contrast_prot_z": 0.0,
            "contrast_lig_z": 0.0,
            "contrast_logit_scale": 0.0,
            "contrast_valid_mask": 0.0,
        }
        pair_ids = []
        pair_id_missing = 0.0
        prot_uid_rows = []
        lig_uid_rows = []
        prot_uid_missing = 0.0
        lig_uid_missing = 0.0
        logit_scale = None

        for i, b in enumerate(batch_list):
            pred = b.get("pred_dict", {})

            p = pred.get("contrast_prot_z", None)
            l = pred.get("contrast_lig_z", None)
            v = pred.get("contrast_valid_mask", None)
            s = pred.get("contrast_logit_scale", None)
            pair_id, is_missing = _extract_pair_id(b, i)
            pair_ids.append(pair_id)
            if is_missing:
                pair_id_missing += 1.0
            prot_uid, prot_miss = _extract_uid(b, i, "prot", pair_id)
            lig_uid, lig_miss = _extract_uid(
                b, i, "lig", pair_id, lig_z_hint=l if isinstance(l, torch.Tensor) else None
            )
            prot_uid_rows.append(prot_uid)
            lig_uid_rows.append(lig_uid)
            prot_uid_missing += float(prot_miss)
            lig_uid_missing += float(lig_miss)

            if not isinstance(p, torch.Tensor):
                missing["contrast_prot_z"] += 1.0
            if not isinstance(l, torch.Tensor):
                missing["contrast_lig_z"] += 1.0
            if not isinstance(v, torch.Tensor):
                missing["contrast_valid_mask"] += 1.0
            if not isinstance(s, torch.Tensor):
                missing["contrast_logit_scale"] += 1.0

            prot_rows.append(_to_row(p))
            lig_rows.append(_to_row(l))

            if isinstance(v, torch.Tensor):
                vv = v.to(device=device, dtype=torch.float32).view(-1)
                vv = vv[:1] if vv.numel() > 0 else torch.zeros((1,), device=device, dtype=torch.float32)
            else:
                vv = torch.zeros((1,), device=device, dtype=torch.float32)
            valid_rows.append(vv)

            if logit_scale is None and isinstance(s, torch.Tensor):
                logit_scale = s.to(device=device, dtype=torch.float32)

        if logit_scale is None:
            logit_scale = torch.zeros((), device=device, dtype=torch.float32)

        prot_z = torch.cat(prot_rows, dim=0)   # [B_local, D]
        lig_z = torch.cat(lig_rows, dim=0)     # [B_local, D]
        valid_mask = torch.cat(valid_rows, dim=0).view(-1)  # [B_local]
        prot_uid = torch.tensor(prot_uid_rows, device=device, dtype=torch.long).view(-1)
        lig_uid = torch.tensor(lig_uid_rows, device=device, dtype=torch.long).view(-1)

        # Local pairing diagnostics (without DDP gather) to detect index mismatch.
        with torch.no_grad():
            p_local = F.normalize(prot_z, dim=-1)
            l_local = F.normalize(lig_z, dim=-1)
            sim_local = p_local @ l_local.t()  # [B_local, B_local]
            b_local = sim_local.size(0)
            labels_local = torch.arange(b_local, device=device, dtype=torch.long)
            valid_local = (valid_mask > 0).to(sim_local.dtype)
            denom_local = valid_local.sum().clamp(min=1.0)

            top1_local = (sim_local.argmax(dim=-1) == labels_local).to(sim_local.dtype)
            top1_local = (top1_local * valid_local).sum() / denom_local

            pos_local = sim_local.gather(1, labels_local[:, None]).squeeze(-1)
            if b_local > 1:
                off_mask = ~torch.eye(b_local, device=device, dtype=torch.bool)
                off_cnt = off_mask.sum(dim=-1).clamp(min=1).to(sim_local.dtype)
                off_sum = sim_local.masked_fill(~off_mask, 0.0).sum(dim=-1)
                off_local = off_sum / off_cnt
            else:
                off_local = torch.zeros_like(pos_local)
            diag_minus_offdiag_local = (pos_local - off_local)
            diag_minus_offdiag_local = (
                diag_minus_offdiag_local * valid_local
            ).sum() / denom_local

        contrast_loss, contrast_metrics = self.loss.contrast_loss(
            prot_z=prot_z,
            lig_z=lig_z,
            logit_scale=logit_scale,
            valid_mask=valid_mask,
            prot_uid=prot_uid,
            lig_uid=lig_uid,
        )
        # Baseline-aware diagnostics:
        # - raw_minus_ln_global_batch < 0 indicates better-than-random CE baseline.
        # - top1_minus_random > 0 indicates better-than-random retrieval.
        try:
            gb = contrast_metrics.get("global_batch", None)
            raw_v = contrast_metrics.get("raw", None)
            top1_v = contrast_metrics.get("top1", None)
            if isinstance(gb, torch.Tensor):
                gb_safe = gb.detach().float().clamp(min=1.0)
                rand_top1 = (1.0 / gb_safe).detach()
                contrast_metrics["random_top1"] = rand_top1
                if isinstance(raw_v, torch.Tensor):
                    contrast_metrics["raw_minus_ln_global_batch"] = (
                        raw_v.detach().float() - gb_safe.log()
                    )
                if isinstance(top1_v, torch.Tensor):
                    contrast_metrics["top1_minus_random"] = (
                        top1_v.detach().float() - rand_top1
                    )
        except Exception:
            pass
        with torch.no_grad():
            b_local = int(prot_z.size(0))
            perm = (
                torch.roll(torch.arange(b_local, device=device), shifts=1)
                if b_local > 1
                else torch.arange(b_local, device=device)
            )
            lig_z_shuf = lig_z.index_select(0, perm)
            valid_mask_shuf = valid_mask.index_select(0, perm)
            shuffled_loss, _ = self.loss.contrast_loss(
                prot_z=prot_z.detach(),
                lig_z=lig_z_shuf.detach(),
                logit_scale=logit_scale.detach(),
                valid_mask=valid_mask_shuf.detach(),
            )
        contrast_metrics["local_top1_wo_gather"] = top1_local.detach()
        contrast_metrics["local_diag_minus_offdiag_wo_gather"] = (
            diag_minus_offdiag_local.detach()
        )
        contrast_metrics["raw_shuffled_local"] = shuffled_loss.detach()
        contrast_metrics["raw_gap_vs_shuffled"] = (
            shuffled_loss.detach() - contrast_loss.detach()
        )
        contrast_metrics["pair_id_missing"] = torch.tensor(
            pair_id_missing, device=device, dtype=torch.float32
        )
        contrast_metrics["pair_id_unique"] = torch.tensor(
            float(len(set(pair_ids))), device=device, dtype=torch.float32
        )
        contrast_metrics["pair_id_dup"] = torch.tensor(
            float(max(0, len(pair_ids) - len(set(pair_ids)))),
            device=device,
            dtype=torch.float32,
        )
        contrast_metrics["prot_uid_unique"] = torch.tensor(
            float(len(set(prot_uid_rows))), device=device, dtype=torch.float32
        )
        contrast_metrics["lig_uid_unique"] = torch.tensor(
            float(len(set(lig_uid_rows))), device=device, dtype=torch.float32
        )
        contrast_metrics["prot_uid_missing"] = torch.tensor(
            float(prot_uid_missing), device=device, dtype=torch.float32
        )
        contrast_metrics["lig_uid_missing"] = torch.tensor(
            float(lig_uid_missing), device=device, dtype=torch.float32
        )
        contrast_metrics["pair_index_ok"] = torch.tensor(
            1.0 if (len(prot_rows) == len(lig_rows) == len(batch_list)) else 0.0,
            device=device,
            dtype=torch.float32,
        )

        contrast_metrics["missing_input_total"] = torch.tensor(
            sum(missing.values()), device=device, dtype=torch.float32
        )
        contrast_metrics["missing_input_prot_z"] = torch.tensor(
            missing["contrast_prot_z"], device=device, dtype=torch.float32
        )
        contrast_metrics["missing_input_lig_z"] = torch.tensor(
            missing["contrast_lig_z"], device=device, dtype=torch.float32
        )
        contrast_metrics["missing_input_logit_scale"] = torch.tensor(
            missing["contrast_logit_scale"], device=device, dtype=torch.float32
        )
        contrast_metrics["missing_input_valid_mask"] = torch.tensor(
            missing["contrast_valid_mask"], device=device, dtype=torch.float32
        )
        return contrast_loss, contrast_metrics

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
            return lddt_dict, rmsd_dict

        return lddt_dict

    @torch.no_grad()
    def aggregate_metrics(self, lddt_dict: dict, batch: dict) -> dict:

        simple_metrics, _ = self.lddt_metrics.aggregate_lddt(
            lddt_dict, batch["pred_dict"]["summary_confidence"]
        )

        return simple_metrics

    @torch.no_grad()
    def evaluate(self, mode: str = "eval"):
        if not self.configs.eval_ema_only:
            self._evaluate()
        if hasattr(self, "ema_wrapper"):
            self.ema_wrapper.apply_shadow()
            self._evaluate(ema_suffix=f"ema{self.ema_wrapper.decay}_", mode=mode)
            self.ema_wrapper.restore()

    @torch.no_grad()
    def _evaluate_lit_pcba(self, ema_suffix: str = ""):
        benchmark_cfg = getattr(self.configs, "benchmark", {})
        enable_lit_pcba = bool(benchmark_cfg.get("enable_lit_pcba", False))
        if not enable_lit_pcba:
            return

        lit_pcba_root = benchmark_cfg.get("lit_pcba_root", "")
        if not lit_pcba_root or (not os.path.isdir(lit_pcba_root)):
            self.print(f"Skip LIT-PCBA eval: invalid root [{lit_pcba_root}]")
            return

        if DIST_WRAPPER.world_size > 1:
            dist.barrier()

        if DIST_WRAPPER.rank == 0:
            try:
                model_for_eval = self.model.module if hasattr(self.model, "module") else self.model
                embedder = InMemoryProtenixEmbedder(model_for_eval, self.device)
                summary, per_target = evaluate_dataset(
                    dataset_name="lit_pcba",
                    dataset_root=lit_pcba_root,
                    embedder=embedder,
                    lig_batch_size=int(benchmark_cfg.get("lit_pcba_lig_batch_size", 64)),
                    require_mols="mols.lmdb",
                    require_pocket="pockets.lmdb"
                    if bool(benchmark_cfg.get("lit_pcba_require_pocket", False))
                    else "",
                    limit_targets=int(benchmark_cfg.get("lit_pcba_limit_targets", 0)),
                )
                metric_alias = {
                    "AUCROC": "AUCROC",
                    "BEDROC_80.5": "BEDROC_80_5",
                    "EF0.5%": "EF0_5_pct",
                    "EF1%": "EF1_pct",
                    "EF2%": "EF2_pct",
                    "EF5%": "EF5_pct",
                }
                dataset_tag = "lit_pcba" if not ema_suffix else f"lit_pcba_{ema_suffix.rstrip('_')}"
                wandb_metrics = {
                    f"eval/{dataset_tag}/n_targets": int(summary.get("n_targets", 0))
                }
                for raw_key, wandb_key in metric_alias.items():
                    value = summary.get(raw_key, float("nan"))
                    try:
                        value = float(value)
                    except Exception:
                        value = float("nan")
                    wandb_metrics[f"eval/{dataset_tag}/{wandb_key}"] = value

                result_payload = {
                    "step": int(self.step),
                    "ema_suffix": ema_suffix,
                    "dataset": "lit_pcba",
                    "summary": summary,
                    "per_target": per_target,
                }
                suffix = f"_{ema_suffix.rstrip('_')}" if ema_suffix else ""
                out_json = os.path.join(
                    self.benchmark_eval_dir,
                    f"lit_pcba_step_{self.step}{suffix}.json",
                )
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(result_payload, f, indent=2, ensure_ascii=False)

                history = []
                if os.path.exists(self.benchmark_history_path):
                    try:
                        with open(self.benchmark_history_path, "r", encoding="utf-8") as f:
                            history = json.load(f)
                    except Exception:
                        history = []
                history.append(
                    {
                        "step": int(self.step),
                        "ema_suffix": ema_suffix,
                        "summary": summary,
                        "result_file": out_json,
                    }
                )
                with open(self.benchmark_history_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)

                self.print(f"Step {self.step}, eval {dataset_tag}: {summary}")
                if self.configs.use_wandb:
                    wandb.log(wandb_metrics, step=self.step)
            except Exception as e:
                self.print(f"LIT-PCBA eval failed at step {self.step}: {e}")
            finally:
                torch.cuda.empty_cache()

        if DIST_WRAPPER.world_size > 1:
            dist.barrier()

    @torch.no_grad()
    def _evaluate(self, ema_suffix: str = "", mode: str = "eval"):
        # Init Metric Aggregator
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
            self.print(f"Testing on {test_name}")
            evaluated_pids = []
            total_batch_num = len(test_dl)
            if test_name in ['posebusters_0925', "pdbbind_test", "pbbind_test_v2"]:
                ligand_rmsds = []
                pdb_rmsd_dict = {}
            
            for index, batch in enumerate(tqdm(test_dl)):
                batch = to_device(batch, self.device)
                pid = batch["basic"]["pdb_id"]

                if index + 1 == total_batch_num and DIST_WRAPPER.world_size > 1:
                    # Gather all pids across ranks for avoiding duplicated evaluations when drop_last = False
                    all_data_ids = DIST_WRAPPER.all_gather_object(evaluated_pids)
                    dedup_ids = set(sum(all_data_ids, []))
                    if pid in dedup_ids:
                        print(
                            f"Rank {DIST_WRAPPER.rank}: Drop data_id {pid} as it is already evaluated."
                        )
                        break
                evaluated_pids.append(pid)

                simple_metrics = {}
                with enable_amp:
                    # Model forward
                    batch, _ = self.model_forward(batch, mode=mode)
                    # Loss forward
                    loss, loss_dict, batch = self.get_loss(batch, mode="eval")
                    # lDDT metrics
                    if 'interested_ligand_mask' in batch['label_dict']:
                        lddt_dict, rmsd_dict = self.get_metrics(batch)
                        ligand_rmsds.extend(rmsd_dict['ligand_rmsds'])
                        pdb_rmsd_dict[batch['basic']['pdb_id']] = rmsd_dict['ligand_rmsds'].mean()
                    else:
                        lddt_dict = self.get_metrics(batch)
                    
                    
                    
                    lddt_metrics = self.aggregate_metrics(lddt_dict, batch)
                    simple_metrics.update(
                        {k: v for k, v in lddt_metrics.items() if "diff" not in k}
                    )
                    simple_metrics.update(loss_dict)

                # Metrics
                for key, value in simple_metrics.items():
                    simple_metric_wrapper.add(
                        f"{ema_suffix}{key}", value, namespace=test_name
                    )
                # 调试wandb
                if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                    eval_log_interval = getattr(self.configs, "eval_log_interval", 5)  # 每5个batch打一条
                    if (index % eval_log_interval == 0) or (index + 1 == total_batch_num):
                        wandb.log(
                            {
                                f"eval/{test_name}/progress": (index + 1) / total_batch_num,
                                f"eval/{test_name}/batch_idx": index + 1,
                                # 用本 batch 的 loss（或者你也可以换成某个 loss_dict 里的 key）
                                f"eval/{test_name}/loss_batch": float(loss.detach().float().cpu()),
                            },
                            step=self.step,  # 关键：保持 step 不倒退
                        )
                # 调试结束
                del batch, simple_metrics
                if index % 5 == 0:
                    # Release some memory periodically
                    torch.cuda.empty_cache()

            metrics = simple_metric_wrapper.calc()
            
            if test_name in ['posebusters_0925', "pdbbind_test", "pbbind_test_v2"]: # calculate the mean rmsd and rmsd < 2 or 5 A ratio
                ratio_lt_2A = np.mean(np.array(ligand_rmsds) < 2.0)
                ratio_lt_5A = np.mean(np.array(ligand_rmsds) < 5.0)
                mean_rmsd = np.mean(ligand_rmsds)
                self.print(f'ratio 2A:  {ratio_lt_2A}; ratio 5A: {ratio_lt_5A}, mean_rmsd: {mean_rmsd}')
                # save pdb rmsd dict as txt
                # sort the dict by rmsd
                pdb_rmsd_dict = dict(sorted(pdb_rmsd_dict.items(), key=lambda item: item[1]))
                with open(f'{test_name}_ligand_rmsd.txt', 'w') as f:
                    for pdb_id, rmsd in pdb_rmsd_dict.items():
                        f.write(f'{pdb_id}\t{rmsd}\n')
                
                
                
            self.print(f"Step {self.step}, eval {test_name}: {metrics}")
            if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                wandb.log(metrics, step=self.step)
        self._evaluate_lit_pcba(ema_suffix=ema_suffix)

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

        batch_list = batch if isinstance(batch, list) else [batch]
        contrast_cfg = getattr(self.configs, "contrast", {})
        try:
            contrast_only_forward = bool(
                contrast_cfg.get("only_forward", False)
                if isinstance(contrast_cfg, dict)
                else contrast_cfg.get("only_forward", False)
            )
        except Exception:
            contrast_only_forward = bool(
                getattr(contrast_cfg, "only_forward", False)
            )
        with enable_amp:
            if len(batch_list) == 1:
                batch, _ = self.model_forward(batch_list[0], mode="train")
                if contrast_only_forward:
                    contrast_loss, contrast_metrics = self._compute_group_contrast_loss([batch])
                    contrast_weight = float(
                        self.loss._get_effective_loss_weight(
                            "contrast_loss", batch["input_feature_dict"], mode="train"
                        )
                    )
                    loss = contrast_weight * contrast_loss
                    loss_dict = {
                        "contrast_loss": contrast_loss.detach().clone(),
                        "weighted_contrast_loss": (contrast_weight * contrast_loss).detach().clone(),
                        "contrast_loss/weight": torch.tensor(
                            contrast_weight, device=contrast_loss.device, dtype=contrast_loss.dtype
                        ),
                    }
                    for key, value in contrast_metrics.items():
                        loss_dict[f"contrast_loss/{key}"] = (
                            value.detach().clone()
                            if torch.is_tensor(value)
                            else torch.tensor(float(value), device=contrast_loss.device, dtype=torch.float32)
                        )
                    loss_dict["loss"] = loss.detach().clone()
                else:
                    # 调试信息
                    if DIST_WRAPPER.rank == 0 and self.step < 50:
                        try:
                            is_lig_sum = int(batch["input_feature_dict"]["is_ligand"].sum().item())
                            if is_lig_sum == 0:
                                b = batch.get("basic", {})
                                print(
                                    f"[NO_LIGAND][step={self.step}] pdb_id={b.get('pdb_id')} assembly_id={b.get('assembly_id')} "
                                    f"chain_1_id={b.get('chain_1_id', None)} chain_2_id={b.get('chain_2_id', None)} "
                                    f"asym_id_unique={batch['input_feature_dict']['asym_id'].unique().tolist()[:20]} "
                                    f"num_tokens={b.get('num_tokens', None)}"
                                )
                        except Exception as e:
                            print("[NO_LIGAND] debug print failed:", e)
                    # 调试结束
                    loss, loss_dict, _ = self.get_loss(batch, mode="train")
            else:
                # True local batch path: each item contributes to structure loss;
                # contrast is computed once on stacked local embeddings.
                contrast_enabled = bool(getattr(self.loss, "contrast_enable", False))
                if contrast_only_forward:
                    for i, sample in enumerate(batch_list):
                        sample, _ = self.model_forward(sample, mode="train")
                        batch_list[i] = sample
                    if contrast_enabled:
                        contrast_loss, contrast_metrics = self._compute_group_contrast_loss(batch_list)
                        contrast_weight = float(
                            self.loss._get_effective_loss_weight(
                                "contrast_loss", batch_list[0]["input_feature_dict"], mode="train"
                            )
                        )
                        loss = contrast_weight * contrast_loss
                        loss_dict = {
                            "contrast_loss": contrast_loss.detach().clone(),
                            "weighted_contrast_loss": (contrast_weight * contrast_loss).detach().clone(),
                            "contrast_loss/weight": torch.tensor(
                                contrast_weight, device=contrast_loss.device, dtype=contrast_loss.dtype
                            ),
                        }
                        for key, value in contrast_metrics.items():
                            loss_dict[f"contrast_loss/{key}"] = (
                                value.detach().clone()
                                if torch.is_tensor(value)
                                else torch.tensor(float(value), device=contrast_loss.device, dtype=torch.float32)
                            )
                        loss_dict["loss"] = loss.detach().clone()
                    else:
                        loss = torch.tensor(
                            0.0, device=self.device, dtype=torch.float32, requires_grad=True
                        )
                        loss_dict = {"loss": loss.detach().clone()}
                else:
                    structure_losses = []
                    structure_loss_dicts = []
                    if contrast_enabled:
                        self.loss.contrast_enable = False
                    try:
                        for sample in batch_list:
                            sample, _ = self.model_forward(sample, mode="train")
                            loss_i, loss_dict_i, _ = self.get_loss(sample, mode="train")
                            structure_losses.append(loss_i)
                            structure_loss_dicts.append(loss_dict_i)
                    finally:
                        if contrast_enabled:
                            self.loss.contrast_enable = True

                    loss = torch.stack(structure_losses).mean()
                    loss_dict = self._merge_loss_dicts_mean(structure_loss_dicts)

                    if contrast_enabled:
                        contrast_loss, contrast_metrics = self._compute_group_contrast_loss(batch_list)
                        contrast_weight = float(
                            self.loss._get_effective_loss_weight(
                                "contrast_loss", batch_list[0]["input_feature_dict"], mode="train"
                            )
                        )
                        loss = loss + contrast_weight * contrast_loss
                        loss_dict["contrast_loss"] = contrast_loss.detach().clone()
                        loss_dict["weighted_contrast_loss"] = (
                            contrast_weight * contrast_loss
                        ).detach().clone()
                        loss_dict["contrast_loss/weight"] = torch.tensor(
                            contrast_weight, device=contrast_loss.device, dtype=contrast_loss.dtype
                        )
                        for key, value in contrast_metrics.items():
                            loss_dict[f"contrast_loss/{key}"] = (
                                value.detach().clone()
                                if torch.is_tensor(value)
                                else torch.tensor(float(value), device=contrast_loss.device, dtype=torch.float32)
                            )

                    loss_dict["loss"] = loss.detach().clone()

        if self.configs.dtype in ["bf16", "fp32"]:
            if is_loss_nan_check(loss):
                self.print(f"Skip iteration with NaN loss: {self.step} steps")
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

        # If loss does not require grad (no trainable params connected), skip backward
        if not getattr(loss, 'requires_grad', False):
            self.print(f"Skipping backward: loss has no grad (step {self.step})")
        else:
            scaler.scale(loss / self.iters_to_accumulate).backward()

        # For simplicity, the global training step is used
        if (self.global_step + 1) % self.iters_to_accumulate == 0:
            self.print(
                f"self.step {self.step}, self.iters_to_accumulate: {self.iters_to_accumulate}"
            )
            # Unscales the gradients of optimizer's assigned parameters in-place
            scaler.unscale_(self.optimizer)
            # Contrast grad/optimizer connectivity debug (rank0, early steps only)
            grad_debug_metrics = {}
            if DIST_WRAPPER.rank == 0 and self.step < 30:
                try:
                    opt_param_ids = {
                        id(p)
                        for group in self.optimizer.param_groups
                        for p in group["params"]
                    }

                    def _report_grad(key: str):
                        total = 0
                        in_opt = 0
                        with_grad = 0
                        in_opt_with_grad = 0
                        grad_norm_sum = 0.0
                        for n, p in self.model.named_parameters():
                            if key not in n:
                                continue
                            total += 1
                            if id(p) in opt_param_ids:
                                in_opt += 1
                            if p.grad is not None:
                                with_grad += 1
                                grad_norm_sum += p.grad.detach().float().norm(2).item()
                                if id(p) in opt_param_ids:
                                    in_opt_with_grad += 1
                        self.print(
                            f"[GRAD_DEBUG][step={self.step}] key={key} "
                            f"total={total} in_opt={in_opt} with_grad={with_grad} "
                            f"in_opt_with_grad={in_opt_with_grad} grad_norm_sum={grad_norm_sum:.6e}"
                        )
                        grad_debug_metrics[f"contrast_loss/grad_norm_sum_{key}"] = torch.tensor(
                            float(grad_norm_sum), device=self.device, dtype=torch.float32
                        )
                        grad_debug_metrics[f"contrast_loss/params_total_{key}"] = torch.tensor(
                            float(total), device=self.device, dtype=torch.float32
                        )
                        grad_debug_metrics[f"contrast_loss/params_in_opt_{key}"] = torch.tensor(
                            float(in_opt), device=self.device, dtype=torch.float32
                        )
                        grad_debug_metrics[f"contrast_loss/params_with_grad_{key}"] = torch.tensor(
                            float(with_grad), device=self.device, dtype=torch.float32
                        )
                        grad_debug_metrics[
                            f"contrast_loss/params_in_opt_with_grad_{key}"
                        ] = torch.tensor(
                            float(in_opt_with_grad), device=self.device, dtype=torch.float32
                        )

                    for _k in [
                        "esm_proj",
                        "unimol_proj",
                        "logit_scale",
                    ]:
                        _report_grad(_k)
                except Exception as e:
                    self.print(f"[GRAD_DEBUG] failed: {e}")
            if len(grad_debug_metrics) > 0:
                loss_dict.update(grad_debug_metrics)
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
        contrast_cfg = getattr(self.configs, "contrast", {})
        try:
            contrast_only_forward = bool(
                contrast_cfg.get("only_forward", False)
                if isinstance(contrast_cfg, dict)
                else contrast_cfg.get("only_forward", False)
            )
        except Exception:
            contrast_only_forward = bool(getattr(contrast_cfg, "only_forward", False))

        if self.configs.eval_only or self.configs.eval_first:
            if contrast_only_forward:
                self.print("Skip eval: contrast.only_forward=True (no structure outputs).")
            else:
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
                eval_triggers = self._get_eval_triggers(self.step + 1)

                step_need_eval = len(eval_triggers) > 0
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
                    if contrast_only_forward:
                        self.print(
                            "Skip eval at step boundary: contrast.only_forward=True."
                        )
                    else:
                        if len(eval_triggers) > 0:
                            self.print(
                                f"Trigger eval at step {self.step + 1} by {eval_triggers}"
                            )
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
    def _deep_update(dst: dict, src: dict) -> dict:
        for k, v in src.items():
            if (
                isinstance(v, dict)
                and isinstance(dst.get(k), dict)
            ):
                _deep_update(dst[k], v)
            else:
                dst[k] = v
        return dst

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
    arg_str = parse_sys_args()
    # Resolve model_name directly from argv so model-specific keys can be registered
    # before argparse parsing (e.g. --contrast.*).
    model_name = configs_base.get("model_name")
    argv = sys.argv[1:]
    for k, v in zip(argv[::2], argv[1::2]):
        if k == "--model_name":
            model_name = v
            break

    configs = {**configs_base, **{"data": data_configs}}
    model_specfics_configs = ConfigDict(model_configs[model_name])
    # update model specific configs
    configs = _deep_update(configs, model_specfics_configs.to_dict())
    # Single pass parse with model-specific keys already present.
    configs = parse_configs(
        configs,
        arg_str,
    )

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
