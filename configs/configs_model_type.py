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

# model configs for inference and training,
# such as: protenix-base, protenix-mini, protenix-tiny, protenix-constraint.
# protenix_{model_size}_{features}_{version}
# model_size: base, mini, tiny
# features: default, constraint, esm, etc, if multiple split by "-"
# version: v{x}.{y}.{z}

"""
Currently, the model_name support the following models.

|           Model Name                |    ESM/MSA/Constraint    | Model Parameters(M ) |
|-------------------------------------|--------------------------|----------------------|
| `protenix_base_default_v0.5.0`      |      ❌ / ✅ / ❌         |         368.09       |
| `protenix_base_constraint_v0.5.0`   |      ❌ / ✅ / ✅         |         368.30       |
| `protenix_mini_esm_v0.5.0`          |      ✅ / ✅ / ❌         |         135.22       |
| `protenix_mini_ism_v0.5.0`          |      ✅ / ✅ / ❌         |         135.22       |
| `protenix_mini_default_v0.5.0`      |      ❌ / ✅ / ❌         |         134.06       |
| `protenix_tiny_default_v0.5.0`      |      ❌ / ✅ / ❌         |         109.50       |
| "protenix_mini_esm650m_unimol_contrast_v0.2.0"| yes/yes/yes|?| 完成了基础的模型搭建
| 将qbiolip接入
"""
model_configs = {
    "protenix_base_default_v0.5.0": {
        "model": {"N_cycle": 10},
        "sample_diffusion": {
            "N_step": 200,
        },  # the default setting for base model
    },
    "protenix_base_constraint_v0.5.0": {
        "model": {
            "sample_diffusion": {
                "N_step": 200,
            },  # the default setting for constraint model
            "N_cycle": 10,
            "constraint_embedder": {
                "pocket_embedder": {
                    "enable": True,
                },
                "contact_embedder": {
                    "enable": True,
                },
                "substructure_embedder": {
                    "enable": True,
                },
                "contact_atom_embedder": {
                    "enable": True,
                },
            },
        },
        "data": {
            "weightedPDB_before2109_wopb_nometalc_0925": {
                "constraint": {
                    "enable": True,
                    "pocket": {
                        "prob": 0.2,
                        "max_distance_range": {
                            "PP": [4, 15],
                            "LP": [3, 10],
                        },
                    },
                    "contact": {
                        "prob": 0.1,
                    },
                    "substructure": {
                        "prob": 0.5,
                        "size": 1,
                        "coord_noise_scale": 1,
                    },
                    "contact_atom": {
                        "prob": 0.1,
                        "max_distance_range": {
                            "PP": [2, 12],
                            "PL": [2, 15],
                        },
                        "min_distance": -1,
                        "group": "complex",
                        "distance_type": "atom",
                        "feature_type": "continuous",
                    },
                },
            },
            "recentPDB_1536_sample384_0925": {
                "constraint": {
                    "enable": True,
                },
            },
            "posebusters_0925": {
                "constraint": {
                    "enable": True,
                },
            },
        },
        "load_strict": False,  # If finetuning from base model, model arch has been changed,
        # it should be False, for inference, it should be True.
        "finetune_params_with_substring": [
            "constraint_embedder.substructure_z_embedder",
            "constraint_embedder.pocket_z_embedder",
            "constraint_embedder.contact_z_embedder",
            "constraint_embedder.contact_atom_z_embedder",
        ],
    },
    "protenix_mini_default_v0.5.0": {
        "sample_diffusion": {
            "gamma0": 0,
            "step_scale_eta": 1.0,
            "N_step": 5,
        },  # the default setting for mini model
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "load_strict": False,  # For inference, it should be True.
    },
    "protenix_tiny_default_v0.5.0": {
        "sample_diffusion": {
            "gamma0": 0,
            "step_scale_eta": 1.0,
            "N_step": 5,
        },  # the default setting for tiny model
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 8,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "load_strict": False,  # For inference, it should be True.
    },
    "protenix_mini_esm_v0.5.0": {
        "sample_diffusion": {
            "gamma0": 0,
            "step_scale_eta": 1.0,
            "N_step": 5,
        },  # the default setting for mini model
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "enable": True,
            "model_name": "esm2-3b",
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    "protenix_mini_ism_v0.5.0": {
        "sample_diffusion": {
            "gamma0": 0,
            "step_scale_eta": 1.0,
            "N_step": 5,
        },  # the default setting for mini model
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "enable": True,
            "model_name": "esm2-3b-ism",
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    "protenix_mini_esm_online_v0.1.0": {
        # "sample_diffusion": {
        #     "gamma0": 0,
        #     "step_scale_eta": 1.0,
        #     "N_step": 5,
        # },  # the default setting for mini model, we didn't change diffusion settings here
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2-3b",
            "local_model_path": '/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/./release_data/checkpoint/', 
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    "protenix_mini_esm_trainable_v0.1.0": {
        # "sample_diffusion": {
        #     "gamma0": 0,
        #     "step_scale_eta": 1.0,
        #     "N_step": 5,
        # },  # the default setting for mini model, we didn't change diffusion settings here
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2-3b",
            "esm_trainable": True,
            "local_model_path": '/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/./release_data/checkpoint/', 
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    "protenix_mini_baseline_v0.1.0": {
        # "sample_diffusion": {
        #     "gamma0": 0,
        #     "step_scale_eta": 1.0,
        #     "N_step": 5,
        # },  # the default setting for mini model, we didn't change diffusion settings here
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    # "protenix_mini_esm_online_v0.1.0": {
    #     # "sample_diffusion": {
    #     #     "gamma0": 0,
    #     #     "step_scale_eta": 1.0,
    #     #     "N_step": 5,
    #     # },  # the default setting for mini model, we didn't change diffusion settings here
    #     "model": {
    #         "N_cycle": 4,
    #         "msa_module": {
    #             "n_blocks": 1,
    #         },
    #         "pairformer": {
    #             "n_blocks": 16,
    #         },
    #         "diffusion_module": {
    #             "atom_encoder": {
    #                 "n_blocks": 1,
    #             },
    #             "transformer": {
    #                 "n_blocks": 8,
    #             },
    #             "atom_decoder": {
    #                 "n_blocks": 1,
    #             },
    #         },
    #     },
    #     "esm": {
    #         "esm_model_online": True,
    #         "enable": True,
    #         "model_name": "esm2-3b",
    #         "local_model_path": '/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/./release_data/checkpoint/', 
    #     },
    #     "load_strict": False,  # For inference, it should be True.
    #     "use_msa": False,  # For efficiency, this model does not use MSA by default.
    # },
    "protenix_mini_esm_trainable_v0.1.0": {
        # "sample_diffusion": {
        #     "gamma0": 0,
        #     "step_scale_eta": 1.0,
        #     "N_step": 5,
        # },  # the default setting for mini model, we didn't change diffusion settings here
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2-3b",
            "esm_trainable": True,
            "local_model_path": '/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/./release_data/checkpoint/', 
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    "protenix_mini_esm_trainable_v0.1.0": {
        # "sample_diffusion": {
        #     "gamma0": 0,
        #     "step_scale_eta": 1.0,
        #     "N_step": 5,
        # },  # the default setting for mini model, we didn't change diffusion settings here
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2-3b",
            "esm_trainable": True,
            "local_model_path": '/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/./release_data/checkpoint/', 
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    # esm 650M training
    "protenix_mini_esm_trainable_650m_v0.1.0": {
        # "sample_diffusion": {
        #     "gamma0": 0,
        #     "step_scale_eta": 1.0,
        #     "N_step": 5,
        # },  # the default setting for mini model, we didn't change diffusion settings here
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2_t33_650M_UR50D",
            "esm_trainable": True,
            "embedding_dim": 1280,
            "local_model_path": '/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/./release_data/checkpoint/',
            "truncation_seq_length": 1024, 
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    # esm 650M training
    "protenix_mini_esm_trainable_650m_v0.1.0_unimol": {
        # 扩散模型
        # "sample_diffusion": {
        #     "gamma0": 0,
        #     "step_scale_eta": 1.0,
        #     "N_step": 5,
        # },  # the default setting for mini model, we didn't change diffusion settings here
        # protenix主干网络结构配置
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        # ESM2语言模型相关配置
        "esm": {
            "esm_model_online": True,
            "enable": True, 
            "model_name": "esm2_t33_650M_UR50D",
            "esm_trainable": True,
            "embedding_dim": 1280,
            "local_model_path": '/vepfs-mlp2/c20250601/252705034/Protenix/weight/',
            "truncation_seq_length": 1024, 
        },
        "unimol": {
            "mode": "train",
            "dict_path": "/vepfs-mlp2/c20250601/252705034/Protenix/protenix/model/unimol/data",
            "pretrained_path": "/vepfs-mlp2/c20250601/252705034/Protenix/weight/mol_pre_no_h_220816.pt",
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    # esm 650M freeze
    "protenix_mini_esm_650m_v0.1.0": {
        # "sample_diffusion": {
        #     "gamma0": 0,
        #     "step_scale_eta": 1.0,
        #     "N_step": 5,
        # },  # the default setting for mini model, we didn't change diffusion settings here
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2_t33_650M_UR50D",
            "esm_trainable": False,
            "embedding_dim": 1280,
            "local_model_path": '/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/./release_data/checkpoint/', 
        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
    },
    # esm unimol contrast base
    "protenix_mini_esm650m_unimol_contrast_v0.2.0":{
        "model":{
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True, 
            "model_name": "esm2_t33_650M_UR50D",
            "esm_trainable": False,
            "embedding_dim": 1280,
            "local_model_path": '/vepfs-mlp2/c20250601/252705034/Protenix/weight/',
            "truncation_seq_length": 1024, 
        },
        "unimol": {
            "mode": "train",
            "dict_path": "/vepfs-mlp2/c20250601/252705034/Protenix/protenix/model/unimol/data",
            "pretrained_path": "/vepfs-mlp2/c20250601/252705034/Protenix/weight/mol_pre_no_h_220816.pt",
            "unimol_trainable":False,
            
        },
        "contrast":{
            "enable":True,
            "contrast_dim": 256,
            "loss_weight": 0.2,
            "loss_weight_warmup_steps": 2000,
            # If true, freeze backbone (protenix, esm, unimol) and only train projection MLP and logit_scale
            "train_projection_only": False,

        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
        
    },
    # v2.1 增加qbiolip
    "protenix_mini_esm650m_unimol_contrast_v0.2.1":{
        "model":{
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True, 
            "model_name": "esm2_t33_650M_UR50D",
            "esm_trainable": False,
            "embedding_dim": 1280,
            "local_model_path": '/vepfs-mlp2/c20250601/252705034/Protenix/weight/',
            "truncation_seq_length": 1024, 
        },
        "unimol": {
            "mode": "train",
            "dict_path": "/vepfs-mlp2/c20250601/252705034/Protenix/protenix/model/unimol/data",
            "pretrained_path": "/vepfs-mlp2/c20250601/252705034/Protenix/weight/mol_pre_no_h_220816.pt",
            "unimol_trainable":False,
            
        },
        "contrast":{
            "enable":True,
            "contrast_dim": 256,
            "loss_weight": 1.0,

        },
        "load_strict": False,  # For inference, it should be True.
        "use_msa": False,  # For efficiency, this model does not use MSA by default.
        
    },
    # v2.2 contrast stabilized defaults (joint optimization + contrast warmup)
    "protenix_mini_esm650m_unimol_contrast_v0.2.2":{
        "model":{
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2_t33_650M_UR50D",
            "esm_trainable": False,
            "embedding_dim": 1280,
            "local_model_path": '/vepfs-mlp2/c20250601/252705034/Protenix/weight/',
            "truncation_seq_length": 1024,
        },
        "unimol": {
            "mode": "train",
            "dict_path": "/vepfs-mlp2/c20250601/252705034/Protenix/protenix/model/unimol/data",
            "pretrained_path": "/vepfs-mlp2/c20250601/252705034/Protenix/weight/mol_pre_no_h_220816.pt",
            "unimol_trainable":False,
        },
        "contrast":{
            "enable":True,
            "contrast_dim": 256,
            "loss_weight": 0.2,
            "loss_weight_warmup_steps": 2000,
            "train_projection_only": False,
        },
        "load_strict": False,
        "use_msa": False,
    },
    # v0.2.3: contrast rerun profile (full-grad gather, no warmup for clearer loss trend)
    "protenix_mini_esm650m_unimol_contrast_v0.2.3":{
        "model":{
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2_t33_650M_UR50D",
            "esm_trainable": False,
            "embedding_dim": 1280,
            "local_model_path": '/vepfs-mlp2/c20250601/252705034/Protenix/weight/',
            "truncation_seq_length": 1024,
        },
        "unimol": {
            "mode": "train",
            "dict_path": "/vepfs-mlp2/c20250601/252705034/Protenix/protenix/model/unimol/data",
            "pretrained_path": "/vepfs-mlp2/c20250601/252705034/Protenix/weight/mol_pre_no_h_220816.pt",
            "unimol_trainable":False,
        },
        "contrast":{
            "enable":True,
            "contrast_dim": 256,
            "loss_weight": 0.2,
            "loss_weight_warmup_steps": 0,
            "train_projection_only": False,
            "full_grad_gather": True,
            "learn_logit_scale": False,
        },
        "load_strict": False,
        "use_msa": False,
    },
    # v0.2.4: stronger contrast weight for joint-training diagnosis
    "protenix_mini_esm650m_unimol_contrast_v0.2.4":{
        "model":{
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2_t33_650M_UR50D",
            "esm_trainable": False,
            "embedding_dim": 1280,
            "local_model_path": '/vepfs-mlp2/c20250601/252705034/Protenix/weight/',
            "truncation_seq_length": 1024,
        },
        "unimol": {
            "mode": "train",
            "dict_path": "/vepfs-mlp2/c20250601/252705034/Protenix/protenix/model/unimol/data",
            "pretrained_path": "/vepfs-mlp2/c20250601/252705034/Protenix/weight/mol_pre_no_h_220816.pt",
            "unimol_trainable":False,
        },
        "contrast":{
            "enable":True,
            "contrast_dim": 256,
            "loss_weight": 0.4,
            "loss_weight_warmup_steps": 0,
            "train_projection_only": False,
            "full_grad_gather": True,
            "learn_logit_scale": False,
        },
        "load_strict": False,
        "use_msa": False,
    },
    # v0.2.5: decouple structure/contrast projection heads + learnable logit scale
    "protenix_mini_esm650m_unimol_contrast_v0.2.5":{
        "model":{
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2_t33_650M_UR50D",
            "esm_trainable": False,
            "embedding_dim": 1280,
            "local_model_path": '/vepfs-mlp2/c20250601/252705034/Protenix/weight/',
            "truncation_seq_length": 1024,
        },
        "unimol": {
            "mode": "train",
            "dict_path": "/vepfs-mlp2/c20250601/252705034/Protenix/protenix/model/unimol/data",
            "pretrained_path": "/vepfs-mlp2/c20250601/252705034/Protenix/weight/mol_pre_no_h_220816.pt",
            "unimol_trainable":False,
        },
        "contrast":{
            "enable":True,
            "contrast_dim": 256,
            "loss_weight": 0.2,
            "loss_weight_warmup_steps": 0,
            "train_projection_only": False,
            "full_grad_gather": True,
            "learn_logit_scale": True,
            "separate_projection_head": True,
        },
        "load_strict": False,
        "use_msa": False,
    },
    # v0.2.6: contrast bootstrap profile (projection-only, fixed high contrast weight)
    "protenix_mini_esm650m_unimol_contrast_v0.2.6":{
        "model":{
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2_t33_650M_UR50D",
            "esm_trainable": False,
            "embedding_dim": 1280,
            "local_model_path": "/vepfs-mlp2/c20250601/252705034/Protenix/weight/",
            "truncation_seq_length": 1024,
        },
        "unimol": {
            "mode": "train",
            "dict_path": "/vepfs-mlp2/c20250601/252705034/Protenix/protenix/model/unimol/data",
            "pretrained_path": "/vepfs-mlp2/c20250601/252705034/Protenix/weight/mol_pre_no_h_220816.pt",
            "unimol_trainable": False,
        },
        "contrast":{
            "enable": True,
            "contrast_dim": 256,
            "loss_weight": 1.0,
            "loss_weight_warmup_steps": 0,
            "train_projection_only": True,
            "only_forward": True,
            "full_grad_gather": True,
            "learn_logit_scale": False,
            "init_temp": 0.2,
            "std_floor": 0.05,
            "std_reg_weight": 2.0,
            "separate_projection_head": True,
        },
        "load_strict": False,
        "use_msa": False,
    },
    # v0.2.7: contrast-only anti-collapse profile for multi-GPU bootstrap
    "protenix_mini_esm650m_unimol_contrast_v0.2.7":{
        "model":{
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2_t33_650M_UR50D",
            "esm_trainable": False,
            "embedding_dim": 1280,
            "local_model_path": "/vepfs-mlp2/c20250601/252705034/Protenix/weight/",
            "truncation_seq_length": 1024,
        },
        "unimol": {
            "mode": "train",
            "dict_path": "/vepfs-mlp2/c20250601/252705034/Protenix/protenix/model/unimol/data",
            "pretrained_path": "/vepfs-mlp2/c20250601/252705034/Protenix/weight/mol_pre_no_h_220816.pt",
            "unimol_trainable": False,
        },
        "contrast":{
            "enable": True,
            "contrast_dim": 256,
            "loss_weight": 1.0,
            "loss_weight_warmup_steps": 0,
            "train_projection_only": True,
            "only_forward": True,
            # no-grad gather is more stable for 4-GPU contrast bootstrap
            "full_grad_gather": False,
            "learn_logit_scale": False,
            # lower initial scale than v0.2.6 to avoid early over-confident collapse
            "init_temp": 0.5,
            "std_floor": 0.03,
            "std_reg_weight": 10.0,
            "separate_projection_head": True,
        },
        "load_strict": False,
        "use_msa": False,
    },
    # v0.3.0: freeze Protenix trunk, finetune ESM/UniMol + projection MLPs,
    # run LIT-PCBA eval every 500 steps, and train on the de-leaked qbiolip split.
    "protenix_mini_esm650m_unimol_contrast_litpcba_ft_v0.3.0": {
        "batch_size": 4,
        "iters_to_accumulate": 3,
        "eval_interval": 999999999,
        "extra_eval_steps": [0],
        "extra_eval_interval": 500,
        "checkpoint_interval": 500,
        "lr": 3e-5,
        "load_checkpoint_path": "/vepfs-mlp2/c20250601/252705034/Protenix/release_data/checkpoint/protenix_mini_default_v0.5.0.pt",
        "load_params_only": True,
        "load_strict": False,
        "benchmark": {
            "enable_lit_pcba": True,
            "lit_pcba_root": "/vepfs-mlp2/c20250601/252705034/Protenix/benchmark/lit_pcba",
            "lit_pcba_lig_batch_size": 64,
            "lit_pcba_require_pocket": False,
            "lit_pcba_limit_targets": 0,
        },
        "data": {
            "batch_size": 4,
            "num_dl_workers": 0,
            "train_sets": ["qbiolip_nonredund_exclude_litpcba_overlap"],
            "qbiolip_nonredund_exclude_litpcba_overlap": {
                "cropping_configs": {
                    "crop_size": 384,
                },
            },
        },
        "model": {
            "N_cycle": 1,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "sample_diffusion": {
            "N_sample": 1,
        },
        "chain_permutation": {
            "train": {
                "mini_rollout": False,
            },
        },
        "atom_permutation": {
            "train": {
                "mini_rollout": False,
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2_t33_650M_UR50D",
            "esm_trainable": True,
            "trainable_last_n_layers": 4,
            "embedding_dim": 1280,
            "local_model_path": "/vepfs-mlp2/c20250601/252705034/Protenix/weight/",
            "truncation_seq_length": 1024,
        },
        "unimol": {
            "mode": "train",
            "dict_path": "/vepfs-mlp2/c20250601/252705034/Protenix/protenix/model/unimol/data",
            "pretrained_path": "/vepfs-mlp2/c20250601/252705034/Protenix/weight/mol_pre_no_h_220816.pt",
            "unimol_trainable": True,
        },
        "contrast": {
            "enable": True,
            "contrast_dim": 256,
            "loss_weight": 0.2,
            "loss_weight_warmup_steps": 2000,
            "train_projection_only": True,
            "only_forward": False,
            "full_grad_gather": False,
            "learn_logit_scale": False,
            "init_temp": 0.5,
            "std_floor": 0.03,
            "std_reg_weight": 10.0,
            "separate_projection_head": True,
        },
        "loss": {
            "weight": {
                "alpha_diffusion": 4.0,
                "alpha_distogram": 0.03,
                "alpha_confidence": 1e-4,
            },
        },
        "use_msa": False,
    },
    # structure-only sanity check (no contrast loss, just Protenix structure loss)
    "protenix_mini_structure_only_v0.1.0": {
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "esm": {
            "esm_model_online": True,
            "enable": True,
            "model_name": "esm2_t33_650M_UR50D",
            "esm_trainable": False,
            "embedding_dim": 1280,
            "local_model_path": "/vepfs-mlp2/c20250601/252705034/Protenix/weight/",
            "truncation_seq_length": 1024,
        },
        "unimol": {
            "mode": "train",
            "dict_path": "/vepfs-mlp2/c20250601/252705034/Protenix/protenix/model/unimol/data",
            "pretrained_path": "/vepfs-mlp2/c20250601/252705034/Protenix/weight/mol_pre_no_h_220816.pt",
            "unimol_trainable": False,
        },
        "contrast": {
            "enable": False,
        },
        "load_strict": False,
        "use_msa": False,
    }
}
