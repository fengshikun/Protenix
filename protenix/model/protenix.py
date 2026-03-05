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

import random
import time
import pdb
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from protenix.model import sample_confidence
from protenix.model.generator import (
    InferenceNoiseScheduler,
    TrainingNoiseSampler,
    RealUniformSampler,
    RealUniformSamplerSquare,
    RealUnifromSamplerLogisticnorm,
    KarrasSigmaSampler,
    UniformSigmaSampler,
    sample_diffusion,
    sample_diffusion_ddbm,
    sample_diffusion_training,
    sample_diffusion_training_ddbm,
    
)
from protenix.model.utils import simple_merge_dict_list
from protenix.openfold_local.model.primitives import LayerNorm
from protenix.utils.logger import get_logger
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.torch_utils import autocasting_disable_decorator

from .modules.confidence import ConfidenceHead
from .modules.diffusion import DiffusionModule
from .modules.embedders import (
    ConstraintEmbedder,
    InputFeatureEmbedder,
    RelativePositionEncoding,
)
from .modules.head import DistogramHead
from .modules.pairformer import MSAModule, PairformerStack, TemplateEmbedder
from .modules.primitives import LinearNoBias


from protenix.data.compute_esm import compute_ESM_embeddings, compute_esm2_embeddings_online, load_esm_model # here for the esm model loading and forwarding

# load the pre-trained molecular model if needed
from protenix.model.unimol.build_model import build_default_unimol_model

from protenix.model.unimol.models.dataset import load_mols_dataset
from rdkit import Chem

logger = get_logger(__name__)


class Protenix(nn.Module):
    """
    Implements Algorithm 1 [Main Inference/Train Loop] in AF3
    """

    def __init__(self, configs) -> None:

        super(Protenix, self).__init__()
        self.configs = configs

        # Some constants
        self.N_cycle = self.configs.model.N_cycle
        self.N_model_seed = self.configs.model.N_model_seed
        self.train_confidence_only = configs.train_confidence_only
        if self.train_confidence_only:  # the final finetune stage
            assert configs.loss.weight.alpha_diffusion == 0.0
            assert configs.loss.weight.alpha_distogram == 0.0

        # Diffusion scheduler
        self.ddbm = configs.ddbm
        if self.ddbm:
            
            train_sampler_type = configs.ddbm_configs.get('train_sampler', 'RealUniformSampler')
            if train_sampler_type == 'RealUniformSampler':
                self.train_noise_sampler = RealUniformSampler(sigma_max=configs.ddbm_configs['sigma_max'], sigma_min=configs.ddbm_configs['sigma_min'])
            elif train_sampler_type == 'RealUniformSamplerSquare':
                self.train_noise_sampler = RealUniformSamplerSquare(sigma_max=configs.ddbm_configs['sigma_max'], sigma_min=configs.ddbm_configs['sigma_min'])
            else:
                self.train_noise_sampler = RealUnifromSamplerLogisticnorm(sigma_max=configs.ddbm_configs['sigma_max'], sigma_min=configs.ddbm_configs['sigma_min'], lognorm_std=configs.ddbm_configs.get('lognorm_std', 1.0), lognorm_mean=configs.ddbm_configs.get('lognorm_mean', 0.0))
            
            
            # inference_noise_schedule = {}
            # inference_noise_schedule['s_min'] = configs.ddbm_configs['sigma_min']
            # inference_noise_schedule['s_max'] = configs.ddbm_configs['sigma_max'] - 1e-4
            # inference_noise_schedule['sigma_data'] = configs.ddbm_configs['sigma_data']
            # inference_noise_schedule['rho'] = configs.ddbm_configs['rho']
            # self.inference_noise_scheduler = InferenceNoiseScheduler(
            #     **inference_noise_schedule
            # )
            test_sampler = configs.ddbm_configs.get("infer_sampler", "ddbm")
            if test_sampler == 'ddbm':
                self.inference_noise_scheduler = KarrasSigmaSampler(
                    sigma_max=configs.ddbm_configs['sigma_max'] - 1e-4,
                    sigma_min=configs.ddbm_configs['sigma_min'],
                    rho=configs.ddbm_configs['rho'],
                )
            else:
                self.inference_noise_scheduler = UniformSigmaSampler(t_min=configs.ddbm_configs['sigma_min'], t_max=configs.ddbm_configs['sigma_max'] - 1e-3)
            
            # configs.inference_noise_scheduler.sigma_data = 1.0
            # configs.inference_noise_scheduler.s_min = 0.0001
            # configs.inference_noise_scheduler.s_max = 0.9999
            # configs.inference_noise_scheduler.rho = 7
        else:
            self.train_noise_sampler = TrainingNoiseSampler(**configs.train_noise_sampler)
            self.inference_noise_scheduler = InferenceNoiseScheduler(
                **configs.inference_noise_scheduler
            )
        self.diffusion_batch_size = self.configs.diffusion_batch_size

        # Model
        esm_configs = configs.get("esm", {})  # This is used in InputFeatureEmbedder
        
        
        # if esm_configs is not empty, load the esm model here
        if esm_configs.get("esm_model_online", False):
            self.esm_model, self.alphabet = load_esm_model(esm_configs['model_name'], esm_configs['local_model_path'])
            self.esm_trainable = esm_configs.get("esm_trainable", False)
            self.truncation_seq_length = esm_configs.get("truncation_seq_length", 4096)
            if not self.esm_trainable:
                for param in self.esm_model.parameters():
                    param.requires_grad = False
                self.esm_model.eval()
            self.esm_model_online = True
            logger.info("ESM model loaded successfully, load from the path {}".format(esm_configs.get('local_model_path', 'default path')))
            # 记录ESM隐藏维度，可以简写一些
            """
            self.esm_dim = (
                getattr(self.esm_model,"embed_dim",None)
                or getattr(getattr(self.esm_model, "args", None), "embed_dim", None)
                or getattr(getattr(self.esm_model, "embed_tokens", None), "embedding_dim", None)
            )
            if self.esm_dim is None:
                if hasattr(self.esm_model, "embed_tokens") and hasattr(self.esm_model.embed_tokens, "weight"):
                    self.esm_dim = int(self.esm_model.embed_tokens.weight.shape[1])
                else:
                    raise ValueError("Cannot infer esm_dim from esm_model; please set it manually.")
            """
            self.esm_dim = esm_configs.get("embedding_dim",None)
        else:
            self.esm_model_online = False
            self.esm_dim = (
                esm_configs.get("embedding_dim", None)  # 配置里面的config状态
            )
            
            
        # load the molecular model
        unimol_configs = configs.get("unimol", None)
        if unimol_configs is not None:
            self.use_unimol = True
            self.unimol_trainable = bool(unimol_configs.get("unimol_trainable", False))  # unimol trainable
            self.unimol_model = build_default_unimol_model(unimol_configs['mode'], unimol_configs['dict_path'])
            self.unimol_dim = self.unimol_model.embed_tokens.embedding_dim  # unimol hidden size
            self.unimol_to_s_inputs = LinearNoBias(self.unimol_dim, configs.c_s_inputs)  # 投影
            nn.init.zeros_(self.unimol_to_s_inputs.weight)  # 零初始化

            # load the pre-trained weights
            if 'pretrained_path' in unimol_configs:
                unimol_state_dict = torch.load(unimol_configs['pretrained_path'], map_location='cpu')
                missing_keys, unexpected_keys = self.unimol_model.load_state_dict(unimol_state_dict['model'], strict=False)
                logger.info("UniMol model loaded successfully, load from the path {}".format(unimol_configs['pretrained_path']))
                logger.info("Missing keys: {}".format(missing_keys))
                logger.info("Unexpected keys: {}".format(unexpected_keys))
            if not self.unimol_trainable:
                for param in self.unimol_model.parameters():
                    param.requires_grad = False
                self.unimol_model.eval()
        else:
            self.use_unimol = False
            self.unimol_dim = None  # unimol

        self.input_embedder = InputFeatureEmbedder(
            **configs.model.input_embedder, esm_configs=esm_configs
        )
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(**configs.model.template_embedder)
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data.get("msa", {}),
        )
        self.constraint_embedder = ConstraintEmbedder(
            **configs.model.constraint_embedder
        )
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        self.distogram_head = DistogramHead(**configs.model.distogram_head)
        self.confidence_head = ConfidenceHead(**configs.model.confidence_head)

        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )
        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)
        
        
        self.distill_mode = configs.get('distill_mode', 0) # 0 forbidden distill; 1 means teacher; 2 means student;
        self._enable_forward_debug_log = bool(
            getattr(self.configs, "debug", False)
            or getattr(self.configs, "verbose", False)
            or getattr(self.configs, "debug_forward", False)
        )

        # Zero init the recycling layer
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

        # contrast model
        self.contrast_dim = configs.get("contrast", {}).get("contrast_dim", 256)
        contrast_cfg = configs.get("contrast",{})
        mlp_hidden = int(contrast_cfg.get("mlp_hidden",1024))
        # clip风格
        init_temp = float(contrast_cfg.get("init_temp",0.07))
        self._contrast_learn_logit_scale = bool(
            contrast_cfg.get("learn_logit_scale", False)
        )
        _logit_init = torch.tensor(np.log(1.0 / max(init_temp, 1e-6)), dtype=torch.float32)
        if self._contrast_learn_logit_scale:
            self.logit_scale = nn.Parameter(_logit_init)
        else:
            # Default stable mode: keep non-trainable to avoid DDP "marked ready twice".
            self.register_buffer("logit_scale", _logit_init, persistent=True)
        #self.logit_scale = nn.Parameter(torch.tensor(1.0))  # 或者用 CLIP 那种 exp(logit_sca
        
        # mlp
        def _mlp(in_dim: int, out_dim: int):
            # 至少两层 ReLU => 至少 3 个 Linear
            return nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden, mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden, out_dim, bias=False),
            )


        # 当esm 和unimol存在时建立投影头
        if self.esm_dim is None or self.unimol_dim is None:
            self.esm_proj = None
            self.unimol_proj = None
            self.esm_contrast_proj = None
            self.unimol_contrast_proj = None
            self.esm_proj_to_s_inputs = None
            self.unimol_proj_to_s_inputs = None
        else:
            # structure branch projection heads
            self.esm_proj = _mlp(self.esm_dim, self.contrast_dim)
            self.unimol_proj = _mlp(self.unimol_dim, self.contrast_dim)
            # contrast branch projection heads
            # default: separate heads to reduce multi-task interference with structure branch
            separate_proj = bool(contrast_cfg.get("separate_projection_head", True))
            if separate_proj:
                self.esm_contrast_proj = _mlp(self.esm_dim, self.contrast_dim)
                self.unimol_contrast_proj = _mlp(self.unimol_dim, self.contrast_dim)
            else:
                self.esm_contrast_proj = self.esm_proj
                self.unimol_contrast_proj = self.unimol_proj
            # Reuse the same MLP embeddings for the structure branch by mapping to s_inputs dim.
            self.esm_proj_to_s_inputs = LinearNoBias(self.contrast_dim, self.c_s_inputs)
            self.unimol_proj_to_s_inputs = LinearNoBias(
                self.contrast_dim, self.c_s_inputs
            )

    def train(self, mode: bool = True):
        """
        Keep frozen encoder backbones in eval mode during training.
        This avoids stochastic dropout/noise from ESM/UniMol when they are not trainable.
        """
        super().train(mode)
        if getattr(self, "esm_model_online", False) and (not getattr(self, "esm_trainable", False)):
            self.esm_model.eval()
        if getattr(self, "use_unimol", False) and (not getattr(self, "unimol_trainable", False)):
            self.unimol_model.eval()
        return self

    def get_pairformer_output(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        The forward pass from the input to pairformer output

        Args:
            input_feature_dict (dict[str, Any]): input features
            N_cycle (int): number of cycles
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            Tuple[torch.Tensor, ...]: s_inputs, s, z
        """
        if self.train_confidence_only:
            self.input_embedder.eval()
            self.template_embedder.eval()
            self.msa_module.eval()
            self.pairformer_stack.eval()

        # Line 1-5
        s_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 449]

        # 加入 UniMol 条件（token 级对齐后）
        if self.use_unimol and ("unimol_s_inputs_add" in input_feature_dict):
            s_inputs = s_inputs + input_feature_dict["unimol_s_inputs_add"].to(
                device=s_inputs.device, dtype=s_inputs.dtype
            )
        # Add MLP-projected ESM/UniMol token features to the structure trunk when provided.
        if "esm_s_inputs_add_mlp" in input_feature_dict:
            s_inputs = s_inputs + input_feature_dict["esm_s_inputs_add_mlp"].to(
                device=s_inputs.device, dtype=s_inputs.dtype
            )
        if "unimol_s_inputs_add_mlp" in input_feature_dict:
            s_inputs = s_inputs + input_feature_dict["unimol_s_inputs_add_mlp"].to(
                device=s_inputs.device, dtype=s_inputs.dtype
            )
        z_constraint = None

        if "constraint_feature" in input_feature_dict:
            z_constraint = self.constraint_embedder(
                input_feature_dict["constraint_feature"]
            )

        s_init = self.linear_no_bias_sinit(s_inputs)  #  [..., N_token, c_s]
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  #  [..., N_token, N_token, c_z]
        if inplace_safe:
            z_init += self.relative_position_encoding(
                input_feature_dict["asym_id"],
                input_feature_dict["residue_index"],
                input_feature_dict["entity_id"],
                input_feature_dict["token_index"],
                input_feature_dict["sym_id"],
            )
            z_init += self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init += z_constraint
        else:
            z_init = z_init + self.relative_position_encoding(
                input_feature_dict["asym_id"],
                input_feature_dict["residue_index"],
                input_feature_dict["entity_id"],
                input_feature_dict["token_index"],
                input_feature_dict["sym_id"],
            )
            z_init = z_init + self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init = z_init + z_constraint
        # Line 6
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        # Line 7-13 recycling
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                self.training
                and (not self.train_confidence_only)
                and cycle_no == (N_cycle - 1)
            ):
                z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                if inplace_safe:
                    if self.template_embedder.n_blocks > 0:
                        z += self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                    )
                else:
                    if self.template_embedder.n_blocks > 0:
                        z = z + self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                s, z = self.pairformer_stack(
                    s,
                    z,
                    pair_mask=None,
                    triangle_multiplicative=self.configs.triangle_multiplicative,
                    triangle_attention=self.configs.triangle_attention,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

        if self.train_confidence_only:
            self.input_embedder.train()
            self.template_embedder.train()
            self.msa_module.train()
            self.pairformer_stack.train()
        if self._enable_forward_debug_log and DIST_WRAPPER.rank == 0:
            print("pairformer output 运行中!")
            print("s_inputs", s_inputs.shape)
        return s_inputs, s, z

    # pool_esm_embedding
    @staticmethod
    def _pool_esm_global(esm_embeddings, debug=False):
        if esm_embeddings is None:
            return None
        total = None #D
        n = 0 # 总残基
        for v in esm_embeddings.values():
            if v is None:
                continue
            if total is None:
                total = v.new_zeros(v.size(-1))
            total  += v.sum(dim=0)
            n += v.size(0)
        # 解决esm没有蛋白质时的崩溃
        if total is None or n ==0:
            if debug:
                print("没有esm的信息，回退到0")
            return None

        return total/n

    @staticmethod
    def _build_esm_token_embedding(
        input_feature_dict: dict[str, Any], embedding_dim: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if (
            "esm_embeddings" not in input_feature_dict
            or "esm_token_map_idx" not in input_feature_dict
        ):
            return None
        token_map = input_feature_dict["esm_token_map_idx"]
        n_token = int(input_feature_dict["token_index"].shape[-1])
        token_emb = torch.zeros(
            size=(n_token, embedding_dim),
            dtype=torch.float32,
            device=device,
        )
        for i in range(n_token):
            if token_map[i][0] < 0:
                continue
            token_emb[i] = input_feature_dict["esm_embeddings"][
                str(int(token_map[i][1]))
            ][int(token_map[i][0])]
        return token_emb
    
    def sample_diffusion(self, **kwargs) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )
    
        
        
        if self.ddbm:
            _configs.update(
                {
                    "ddbm_configs": self.configs.ddbm_configs,
                })
            return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
                sample_diffusion_ddbm
            )(**_configs, **kwargs)
        else:
            return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
                sample_diffusion
            )(**_configs, **kwargs)

    def run_confidence_head(self, *args, **kwargs):
        """
        Runs the confidence head with optional automatic mixed precision (AMP) disabled.

        Returns:
            Any: The output of the confidence head.
        """
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.confidence_head
        )(*args, **kwargs)

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        N_model_seed: int = 1,
        symmetric_permutation: SymmetricPermutation = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (multiple model seeds) for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_dict (dict[str, Any]): Label dictionary.
            N_cycle (int): Number of cycles.
            mode (str): Mode of operation (e.g., 'inference').
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to 4.
            N_model_seed (int): Number of model seeds. Defaults to 1.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        pred_dicts = []
        log_dicts = []
        time_trackers = []
        for _ in range(N_model_seed):
            pred_dict, log_dict, time_tracker = self._main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                symmetric_permutation=symmetric_permutation,
            )
            pred_dicts.append(pred_dict)
            log_dicts.append(log_dict)
            time_trackers.append(time_tracker)

        # Combine outputs of multiple models
        def _cat(dict_list, key):
            return torch.cat([x[key] for x in dict_list], dim=0)

        def _list_join(dict_list, key):
            return sum([x[key] for x in dict_list], [])

        if self.distill_mode == 2:
            all_pred_dict = {
                "s_inputs": _cat(pred_dicts, "s_inputs"),
                "s": _cat(pred_dicts, "s"),
                "z": _cat(pred_dicts, "z"),
            }
        else:
            all_pred_dict = {
                "coordinate": _cat(pred_dicts, "coordinate"),
                "summary_confidence": _list_join(pred_dicts, "summary_confidence"),
                "full_data": _list_join(pred_dicts, "full_data"),
                "plddt": _cat(pred_dicts, "plddt"),
                "pae": _cat(pred_dicts, "pae"),
                "pde": _cat(pred_dicts, "pde"),
                "resolved": _cat(pred_dicts, "resolved"),
            }

        all_log_dict = simple_merge_dict_list(log_dicts)
        all_time_dict = simple_merge_dict_list(time_trackers)
        return all_pred_dict, all_log_dict, all_time_dict

    def _main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        symmetric_permutation: SymmetricPermutation = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (single model seed) for the Alphafold3 model.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        step_st = time.time()
        N_token = input_feature_dict["token_index"].shape[-1]

        log_dict = {}
        pred_dict = {}
        time_tracker = {}

        if self.distill_mode == 1 and mode == 'eval':
            s_inputs, s, z = input_feature_dict['s_inputs'], input_feature_dict['s'], input_feature_dict['z'] # teacher get the output from student to eval
        else:            
            s_inputs, s, z = self.get_pairformer_output(
                input_feature_dict=input_feature_dict,
                N_cycle=N_cycle,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        
        if self.distill_mode == 2 and mode == 'eval':
            return {'s_inputs': s_inputs, 's': s, 'z': z}, log_dict, time_tracker # student return the output to mimic the teacher
       
        
        
        if mode == "inference":
            keys_to_delete = []
            for key in input_feature_dict.keys():
                if "template_" in key or key in [
                    "msa",
                    "has_deletion",
                    "deletion_value",
                    "profile",
                    "deletion_mean",
                    "token_bonds",
                ]:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del input_feature_dict[key]
            torch.cuda.empty_cache()
        step_trunk = time.time()
        time_tracker.update({"pairformer": step_trunk - step_st})
        # Sample diffusion
        # [..., N_sample, N_atom, 3]
        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]

        noise_schedule = self.inference_noise_scheduler(
            N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype
        )
        pred_dict["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            N_sample=N_sample,
            noise_schedule=noise_schedule,
            inplace_safe=inplace_safe,
        )

        step_diffusion = time.time()
        time_tracker.update({"diffusion": step_diffusion - step_trunk})
        if mode == "inference" and N_token > 2000:
            torch.cuda.empty_cache()
        # Distogram logits: log contact_probs only, to reduce the dimension
        pred_dict["contact_probs"] = autocasting_disable_decorator(True)(
            sample_confidence.compute_contact_prob
        )(
            distogram_logits=self.distogram_head(z),
            **sample_confidence.get_bin_params(self.configs.loss.distogram),
        )  # [N_token, N_token]

        # Confidence logits
        (
            pred_dict["plddt"],
            pred_dict["pae"],
            pred_dict["pde"],
            pred_dict["resolved"],
        ) = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=pred_dict["coordinate"],
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        step_confidence = time.time()
        time_tracker.update({"confidence": step_confidence - step_diffusion})
        time_tracker.update({"model_forward": time.time() - step_st})

        # Permutation: when label is given, permute coordinates and other heads
        if label_dict is not None and symmetric_permutation is not None:
            pred_dict, log_dict = symmetric_permutation.permute_inference_pred_dict(
                input_feature_dict=input_feature_dict,
                pred_dict=pred_dict,
                label_dict=label_dict,
                permute_by_pocket=("pocket_mask" in label_dict)
                and ("interested_ligand_mask" in label_dict),
            )
            last_step_seconds = step_confidence
            time_tracker.update({"permutation": time.time() - last_step_seconds})

        # Summary Confidence & Full Data
        # Computed after coordinates and logits are permuted
        if label_dict is None:
            interested_atom_mask = None
        else:
            interested_atom_mask = label_dict.get("interested_ligand_mask", None)
        (
            pred_dict["summary_confidence"],
            pred_dict["full_data"],
        ) = autocasting_disable_decorator(True)(
            sample_confidence.compute_full_data_and_summary
        )(
            configs=self.configs,
            pae_logits=pred_dict["pae"],
            plddt_logits=pred_dict["plddt"],
            pde_logits=pred_dict["pde"],
            contact_probs=pred_dict.get(
                "per_sample_contact_probs", pred_dict["contact_probs"]
            ),
            token_asym_id=input_feature_dict["asym_id"],
            token_has_frame=input_feature_dict["has_frame"],
            atom_coordinate=pred_dict["coordinate"],
            atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            atom_is_polymer=1 - input_feature_dict["is_ligand"],
            N_recycle=N_cycle,
            interested_atom_mask=interested_atom_mask,
            return_full_data=True,
            mol_id=(input_feature_dict["mol_id"] if mode != "inference" else None),
            elements_one_hot=(
                input_feature_dict["ref_element"] if mode != "inference" else None
            ),
        )

        return pred_dict, log_dict, time_tracker

    def main_train_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict,
        N_cycle: int,
        symmetric_permutation: SymmetricPermutation,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main training loop for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict): Label dictionary (cropped).
            N_cycle (int): Number of cycles.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """
        N_token = input_feature_dict["token_index"].shape[-1]

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        
        
        log_dict = {}
        pred_dict = {}
        
        if self.distill_mode == 2:
            pred_dict = {
                's_inputs': s_inputs,
                's': s,
                'z': z,
            }
            label_dict = {}
            return pred_dict, label_dict, log_dict # student return the output to mimic the teacher
        

        

        # Mini-rollout: used for confidence and label permutation
        with torch.no_grad():
            # [..., 1, N_atom, 3]
            N_sample_mini_rollout = self.configs.sample_diffusion[
                "N_sample_mini_rollout"
            ]  # =1
            N_step_mini_rollout = self.configs.sample_diffusion["N_step_mini_rollout"]

            coordinate_mini = self.sample_diffusion(
                denoise_net=self.diffusion_module,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs.detach(),
                s_trunk=s.detach(),
                z_trunk=z.detach(),
                N_sample=N_sample_mini_rollout,
                noise_schedule=self.inference_noise_scheduler(
                    N_step=N_step_mini_rollout,
                    device=s_inputs.device,
                    dtype=s_inputs.dtype,
                ),
            )
            coordinate_mini.detach_()
            pred_dict["coordinate_mini"] = coordinate_mini

            # Permute ground truth to match mini-rollout prediction
            label_dict, perm_log_dict = (
                symmetric_permutation.permute_label_to_match_mini_rollout(
                    coordinate_mini,
                    input_feature_dict,
                    label_dict,
                    label_full_dict,
                )
            )
            log_dict.update(perm_log_dict)

        # Confidence: use mini-rollout prediction, and detach token embeddings
        drop_embedding = (
            random.random() < self.configs.model.confidence_embedding_drop_rate
        )
        plddt_pred, pae_pred, pde_pred, resolved_pred = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=coordinate_mini,
            use_embedding=not drop_embedding,
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        pred_dict.update(
            {
                "plddt": plddt_pred,
                "pae": pae_pred,
                "pde": pde_pred,
                "resolved": resolved_pred,
            }
        )

        if self.train_confidence_only:
            # Skip diffusion loss and distogram loss. Return now.
            return pred_dict, label_dict, log_dict

        # Denoising: use permuted coords to generate noisy samples and perform denoising
        # x_denoised: [..., N_sample, N_atom, 3]
        # x_noise_level: [..., N_sample]
        N_sample = self.diffusion_batch_size
        drop_conditioning = (
            random.random() < self.configs.model.condition_embedding_drop_rate
        )
        if self.ddbm:
            # _, x_denoised, x_noise_level, mse_weights = sample_diffusion_training_ddbm(
            #     noise_sampler=self.train_noise_sampler,
            #     denoise_net=self.diffusion_module,
            #     label_dict=label_dict,
            #     input_feature_dict=input_feature_dict,
            #     s_inputs=s_inputs,
            #     s_trunk=s,
            #     z_trunk=z,
            #     N_sample=N_sample,
            #     diffusion_chunk_size=self.configs.diffusion_chunk_size,
            #     use_conditioning=not drop_conditioning,
            #     ddbm_configs=self.configs.ddbm_configs,
            # )
            _, x_denoised, x_noise_level, mse_weights = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion_training
            )(sample_diffusion_training_ddbm)(
                noise_sampler=self.train_noise_sampler,
                denoise_net=self.diffusion_module,
                label_dict=label_dict,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s,
                z_trunk=z,
                N_sample=N_sample,
                diffusion_chunk_size=self.configs.diffusion_chunk_size,
                use_conditioning=not drop_conditioning,
                ddbm_configs=self.configs.ddbm_configs,
            )
        else:
            _, x_denoised, x_noise_level = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion_training
            )(sample_diffusion_training)(
                noise_sampler=self.train_noise_sampler,
                denoise_net=self.diffusion_module,
                label_dict=label_dict,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s,
                z_trunk=z,
                N_sample=N_sample,
                diffusion_chunk_size=self.configs.diffusion_chunk_size,
                use_conditioning=not drop_conditioning,
            )
        pred_dict.update(
            {
                "distogram": autocasting_disable_decorator(True)(self.distogram_head)(
                    z
                ),
                # [..., N_sample=48, N_atom, 3]: diffusion loss
                "coordinate": x_denoised,
                "noise_level": x_noise_level,
            }
        )
        if self.ddbm:
            pred_dict.update(
                {
                    "mse_weights": mse_weights,
                }
            )

        # Permute symmetric atom/chain in each sample to match true structure
        # Note: currently chains cannot be permuted since label is cropped
        pred_dict, perm_log_dict, _, _ = (
            symmetric_permutation.permute_diffusion_sample_to_match_label(
                input_feature_dict, pred_dict, label_dict, stage="train"
            )
        )
        log_dict.update(perm_log_dict)

        return pred_dict, label_dict, log_dict

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        mode: str = "inference",
        current_step: Optional[int] = None,
        symmetric_permutation: SymmetricPermutation = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward pass of the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict[str, Any]): Label dictionary (cropped).
            mode (str): Mode of operation ('train', 'inference', 'eval'). Defaults to 'inference'.
            current_step (Optional[int]): Current training step. Defaults to None.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        assert mode in ["train", "inference", "eval"]
        inplace_safe = not (self.training or torch.is_grad_enabled())
        chunk_size = self.configs.infer_setting.chunk_size if inplace_safe else None

        if self.esm_model_online:
            if not self.esm_trainable:
                self.esm_model.eval()
            sequences = input_feature_dict['sequences']  # list of strings
            # pickout the protein sequences only
            protein_entity_ids = input_feature_dict['protein_entity_ids']  # list of ints
            # get the protein sequences
            protein_sequences = [sequences[str(eid)] for eid in protein_entity_ids]
            esm_embeddings = compute_esm2_embeddings_online(
                model=self.esm_model,
                alphabet=self.alphabet,
                labels=protein_entity_ids,
                sequences=protein_sequences,
                trainable=self.esm_trainable,
                truncation_seq_length=self.truncation_seq_length
            )
            input_feature_dict['esm_embeddings'] = esm_embeddings
            # print(esm_embeddings.shape)
            if self.esm_dim is not None:
                esm_token_embedding = self._build_esm_token_embedding(
                    input_feature_dict=input_feature_dict,
                    embedding_dim=int(self.esm_dim),
                    device=next(self.parameters()).device,
                )
                if esm_token_embedding is not None:
                    input_feature_dict["esm_token_embedding"] = esm_token_embedding
        
        if self.use_unimol:           
            if not self.unimol_trainable:
                self.unimol_model.eval()
            # get the ligand atom and positions
            ligand_mask = input_feature_dict['is_ligand'] == 1  # [N_Atoms,]
            if ligand_mask.sum() > 0:
                ligand_positions = input_feature_dict['ref_pos'][ligand_mask]  # [N_ligand_atoms, 3]
                ligand_elements = input_feature_dict['ref_element'][ligand_mask]  # [N_ligand_atoms,
                n_ligand = int(ligand_positions.size(0))
                # 128 
                indices = torch.argmax(ligand_elements, dim=1).detach().cpu().numpy()
                pt = Chem.GetPeriodicTable()
                symbols_list = [pt.GetElementSymbol(int(idx) + 1) for idx in indices]
                symbols_np = np.array(symbols_list)
                
                ligand_dict = [{"atoms": symbols_np,
                               "coordinates": ligand_positions}]
                
                mol_dataset = load_mols_dataset(ligand_dict, self.unimol_model.dictionary)
                sample = next(iter(torch.utils.data.DataLoader(
                    mol_dataset, batch_size=1, collate_fn=mol_dataset.collater
                )))
                unimol_device = next(self.unimol_model.parameters()).device
                st   = sample["net_input"]["mol_src_tokens"].to(unimol_device)        # [1, L]
                dist = sample["net_input"]["mol_src_distance"].to(unimol_device)      # [1, L, L]
                et   = sample["net_input"]["mol_src_edge_type"].to(unimol_device)     # [1, L, L]

                pad_mask = st.eq(self.unimol_model.padding_idx)  # [1, L]
                x = self.unimol_model.embed_tokens(st)  
                n = dist.size(-1)
                gbf = self.unimol_model.gbf(dist, et)
                gab = self.unimol_model.gbf_proj(gbf).permute(0, 3, 1, 2).contiguous().view(-1, n, n)
                if self.unimol_trainable:
                    token_states = self.unimol_model.encoder(x, padding_mask=pad_mask, attn_mask=gab)[0]
                else:
                    with torch.no_grad():
                        token_states = self.unimol_model.encoder(x,padding_mask=pad_mask,attn_mask=gab)[0]
                """保留这个循环，还没明白循环的用途
                bsz = 1
                loader = torch.utils.data.DataLoader(
                    mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater
                )
                
                with torch.no_grad():
                    
                    for sample in loader:
                        # sample = unicore.utils.move_to_cuda(sample)
                        dist = sample["net_input"]["mol_src_distance"]
                        et   = sample["net_input"]["mol_src_edge_type"].cuda()
                        st   = sample["net_input"]["mol_src_tokens"].cuda()

                        pad = st.eq(self.unimol_model.padding_idx)
                        x   = self.unimol_model.embed_tokens(st) # [B ,L ,D unimol]
                        n   = dist.size(-1)
                        gbf = self.unimol_model.gbf(dist, et) # 几何特征编码
                        gab = self.unimol_model.gbf_proj(gbf).permute(0,3,1,2).contiguous().view(-1, n, n) # 几何特征投影

                        out = self.unimol_model.encoder(x, padding_mask=pad, attn_mask=gab)
                        token_states = out[0]  #  [B, L, D_unimol] [1,16,512]
                        rep = out[0][:,0,:]                                  #[B, D] [1,512]
                        
                        # rep = self.model.mol_project(rep)                    # [B, d_proj]
                        # rep = rep / rep.norm(dim=-1, keepdim=True)           # cosine/IP 可互换
                        # reps.append(rep.detach().cpu().to(torch.float32).numpy())
                    
                    # unimol_output = self.unimol_model(
                    #     atom_positions=ligand_positions,
                    #     atom_elements=ligand_elements,
                    # )  # assume the unimol model returns a dict
                # get the unimol embeddings
                #unimol_embeddings = out['embeddings']  # [N_ligand_atoms, unimol_dim]
                # atoms
                pad_mask = pad
                """
                # 序列为BOS+ATOMS+EOS+PAD
                valid_len = int((~pad_mask[0]).sum().item()) # 非pad token数
                # 全局信息（constract用）
                unimol_global = token_states[0,0,:] # D
                # atoms 建模
                atom_states_all = token_states[0,1:valid_len-1,:]
                n_atoms_u = atom_states_all.size(0)

                n_take = min(ligand_positions.size(0),n_atoms_u)
                #atom_states = atom_states_all[:,n_take] # n_take,D
                atom_states = atom_states_all[:n_take]        # [n_take, D_unimol]

                # 对齐到protenix token n_token
                ligand_atom_idx = ligand_mask.nonzero(as_tuple=False).squeeze(-1)[:n_take] # [n_take]
                token_idx = input_feature_dict["atom_to_token_idx"][ligand_atom_idx].long().to(unimol_device)

                N_token = int(input_feature_dict["token_index"].shape[-1])
                D_unimol = int(atom_states.size(-1)) 

                unimol_token = atom_states.new_zeros((N_token,D_unimol))
                cnt = atom_states.new_zeros((N_token,1))  

                unimol_token.index_add_(0,token_idx,atom_states)
                cnt.index_add_(0,token_idx,torch.ones((n_take,1),device=unimol_device, dtype=atom_states.dtype))
                unimol_token = unimol_token / (cnt +1e-6)

                input_feature_dict["unimol_global_embedding"] = unimol_global # D
                input_feature_dict["unimol_atom_embeddings"] = atom_states #n_toke,D
                input_feature_dict["unimol_token_embeddings"] = unimol_token # N_token,D 结构生成
                input_feature_dict["unimol_s_inputs_add"] = self.unimol_to_s_inputs(unimol_token) # [N_token,c_s_inputs]

                if self._enable_forward_debug_log and DIST_WRAPPER.rank == 0:
                    print("unimol 完成了！")
                    print("valid_len", valid_len, "n_atoms_u", n_atoms_u, "n_ligand", n_ligand, "n_take", n_take)
                    print("is_ligand_sum", input_feature_dict.get("is_ligand", None).sum() if input_feature_dict.get("is_ligand", None) is not None else None)
                    print("unimol_token_embeddings", input_feature_dict["unimol_token_embeddings"].shape)
                    print("unimol_s_inputs_add", input_feature_dict["unimol_s_inputs_add"].shape)
                #unimol_embeddings = out[0]  # [N_ligand_atoms, unimol_dim]
                #input_feature_dict['unimol_embeddings'] = unimol_embeddings
            else:
                if self._enable_forward_debug_log and DIST_WRAPPER.rank == 0:
                    print("缺少ligand！全部填写为0！")
                    print("DEBUG: input_feature_dict keys:", list(input_feature_dict.keys()))
                    print("DEBUG: is_ligand present?", "is_ligand" in input_feature_dict)
                    if "is_ligand" in input_feature_dict:
                        try:
                            print("DEBUG: is_ligand sum", input_feature_dict["is_ligand"].sum().item())
                        except Exception:
                            pass
                N_token = int(input_feature_dict["token_index"].shape[-1])
                unimol_device = next(self.unimol_model.parameters()).device
                D_unimol = int(self.unimol_dim)

                input_feature_dict["unimol_global_embedding"] = torch.zeros(D_unimol, device=unimol_device)
                input_feature_dict["unimol_token_embeddings"] = torch.zeros((N_token, D_unimol), device=unimol_device)
                input_feature_dict["unimol_s_inputs_add"] = self.unimol_to_s_inputs(input_feature_dict["unimol_token_embeddings"])



        # contrast
        contrast_cfg = self.configs.get("contrast",{})
        contrast_enable = bool(contrast_cfg.get("enable",False))
        contrast_out = {}
        """之前需要控制是否进入
        if contrast_enable and (self.esm_proj is not None) and (self.unimol_proj is not None):
            # 只有ligand存在的时候才进行contrast
            has_ligand = (input_feature_dict["is_ligand"].sum() >0 ).item()
            if has_ligand:
                # get input
                lig_global = input_feature_dict["unimol_global_embedding"].to(device=unimol_device)   # [512] 或 [B,512]
                prot_global_raw = input_feature_dict.get("esm_embeddings", None)
                
                prot_global = self._pool_esm_global(
                    prot_global_raw,
                    debug=self._enable_forward_debug_log and DIST_WRAPPER.rank == 0,
                )
                # if esm embeddings missing, fallback to zero vector to avoid crash
                if prot_global is None:
                    esm_dim = getattr(self, "esm_dim", None) or 1280
                    prot_global = torch.zeros(esm_dim, device=unimol_device)
                else:
                    prot_global = prot_global.to(device=unimol_device) # D_esm
                # B dim for 1 dim
                if lig_global.dim() == 1:
                    lig_global = lig_global.unsqueeze(0)
                if prot_global.dim() == 1:
                    prot_global = prot_global.unsqueeze(0)
                # others b dim 

                

                prot_z = torch.nn.functional.normalize(self.esm_proj(prot_global), dim=-1)
                lig_z  = torch.nn.functional.normalize(self.unimol_proj(lig_global), dim=-1)
                # valid mask per-sample (batch dim)
                try:
                    valid_mask = torch.ones(prot_global.shape[0], device=prot_global.device, dtype=torch.float32)
                except Exception:
                    valid_mask = torch.tensor([1.0], device=prot_global.device)
                # 投影到 contrast_dim
                contrast_out = {
                                "contrast_prot_z":prot_z,
                                "contrast_lig_z":lig_z,
                                "contrast_logit_scale": self.logit_scale.detach(),
                                "contrast_valid_mask": valid_mask,
                                }

        """
        # contrast
        contrast_cfg = self.configs.get("contrast", {})
        contrast_enable = bool(contrast_cfg.get("enable", False))
        contrast_out = {}

        # Structure branch uses the same ESM/UniMol MLP embeddings when available.
        if (self.esm_proj is not None) and (self.unimol_proj is not None):
            if (
                "esm_token_embedding" in input_feature_dict
                and self.esm_proj_to_s_inputs is not None
            ):
                esm_token_z = self.esm_proj(
                    input_feature_dict["esm_token_embedding"].to(
                        device=unimol_device, dtype=torch.float32
                    )
                )
                input_feature_dict["esm_s_inputs_add_mlp"] = self.esm_proj_to_s_inputs(
                    esm_token_z
                )
            if (
                "unimol_token_embeddings" in input_feature_dict
                and self.unimol_proj_to_s_inputs is not None
            ):
                unimol_token_z = self.unimol_proj(
                    input_feature_dict["unimol_token_embeddings"].to(
                        device=unimol_device, dtype=torch.float32
                    )
                )
                input_feature_dict["unimol_s_inputs_add_mlp"] = (
                    self.unimol_proj_to_s_inputs(unimol_token_z)
                )

        if (
            contrast_enable
            and (self.esm_contrast_proj is not None)
            and (self.unimol_contrast_proj is not None)
        ):
            # 是否真的有 ligand（用于 mask，不用于控制是否计算）
            has_ligand = (input_feature_dict["is_ligand"].sum() > 0).item()

            # --- always build embeddings on every rank ---
            lig_global = input_feature_dict["unimol_global_embedding"].to(device=unimol_device)  # [D_unimol]
            prot_global_raw = input_feature_dict.get("esm_embeddings", None)
            prot_global = self._pool_esm_global(
                prot_global_raw,
                debug=self._enable_forward_debug_log and DIST_WRAPPER.rank == 0,
            )

            if prot_global is None:
                esm_dim = getattr(self, "esm_dim", None) or 1280
                prot_global = torch.zeros(esm_dim, device=unimol_device)
            else:
                prot_global = prot_global.to(device=unimol_device)

            # --- always run projections so parameters are always used ---
            prot_z = self.esm_contrast_proj(prot_global)      # [D_proj]
            lig_z  = self.unimol_contrast_proj(lig_global)    # [D_proj]

            # valid mask: 1 if has ligand else 0
            valid = torch.tensor([1.0 if has_ligand else 0.0], device=unimol_device, dtype=torch.float32)

            # contrast_out = {
            #     "contrast_prot_z": prot_z,
            #     "contrast_lig_z": lig_z,
            #     "contrast_logit_scale": self.logit_scale,   # IMPORTANT: use the PARAMETER here
            #     "contrast_valid_mask": valid,
            # }
            contrast_logit_scale = (
                self.logit_scale
                if self._contrast_learn_logit_scale
                else self.logit_scale.detach()
            )
            contrast_out = {
                "contrast_prot_z": prot_z,
                "contrast_lig_z": lig_z,
                "contrast_logit_scale": contrast_logit_scale,
                "contrast_valid_mask": valid,
            }



        #pred_dict.update(contrast_out)
        contrast_only_forward = bool(contrast_cfg.get("only_forward", False))
        if contrast_only_forward:
            pred_dict = {}
            if contrast_out:
                pred_dict.update(contrast_out)
            return pred_dict

        #pdb.set_trace()
        
        if mode == "train":
            nc_rng = np.random.RandomState(current_step)
            N_cycle = nc_rng.randint(1, self.N_cycle + 1)
            assert self.training
            assert label_dict is not None
            assert symmetric_permutation is not None
            
            # if is teacher, in train mode, only provide the pairwise label to regression
            
            if self.distill_mode == 1:
                # student get the output from teacher to mimic
                label_dict = {}
                with torch.no_grad():
                    s_inputs, s, z = self.get_pairformer_output(
                        input_feature_dict=input_feature_dict,
                        N_cycle=N_cycle,
                        inplace_safe=True,
                        chunk_size=self.configs.infer_setting.chunk_size,
                    )
                label_dict.update({
                    's_inputs': s_inputs,
                    's': s,
                    'z': z,
                })
                return label_dict
            

            pred_dict, label_dict, log_dict = self.main_train_loop(
                input_feature_dict=input_feature_dict,
                label_full_dict=label_full_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                symmetric_permutation=symmetric_permutation,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        elif mode == "inference":
            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=None,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=None,
            )
            log_dict.update({"time": time_tracker})
        elif mode == "eval":
            if label_dict is not None:
                assert (
                    label_dict["coordinate"].size()
                    == label_full_dict["coordinate"].size()
                )
                label_dict.update(label_full_dict)

            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=symmetric_permutation,
            )
            log_dict.update({"time": time_tracker})
            # lig_unimol 和 prot_esm加入到pred_dict 中
        # 全局增加contrast_out 
        if contrast_out:
            pred_dict.update(contrast_out)

        return pred_dict, label_dict, log_dict
