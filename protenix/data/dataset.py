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

import json
import os
import random
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional, Union
from protenix.data.esm_featurizer import ESMFeaturizer

import numpy as np
import pandas as pd
import torch
from biotite.structure.atoms import AtomArray
from ml_collections.config_dict import ConfigDict
from torch.utils.data import Dataset

from protenix.data.constants import EvaluationChainInterface
from protenix.data.constraint_featurizer import ConstraintFeatureGenerator
from protenix.data.data_pipeline import DataPipeline
from protenix.data.featurizer import Featurizer
from protenix.data.msa_featurizer import MSAFeaturizer
from protenix.data.tokenizer import TokenArray
from protenix.data.utils import (
    data_type_transform,
    get_antibody_clusters,
    make_dummy_feature,
)
from protenix.utils.cropping import CropData
from protenix.utils.file_io import read_indices_csv
from protenix.utils.logger import get_logger
from protenix.utils.torch_utils import dict_to_tensor
from scipy.spatial.transform import Rotation
import gemmi
from typing import Dict, List, Union, Any
import string


logger = get_logger(__name__)


import json
from collections import defaultdict
import string

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


_STD3 = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
}

# 仅用于把三字母里识别出的“修饰/非标准残基”写到 AF3 modifications.ptmType
# 这里不再需要 parent AA 映射，因为 sequence 直接来自 sequences_dict
def _as_res3_list(x: Union[Sequence[str], "np.ndarray"]) -> List[str]:
    if np is not None and isinstance(x, np.ndarray):
        return [str(r).strip().upper() for r in x.tolist()]
    return [str(r).strip().upper() for r in x]  # type: ignore[arg-type]


from typing import Dict, List, Union, Any

def unique_stable(arr):
    seen = set()
    out = []
    for x in arr:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return np.array(out)


def _is_weighted_apo_dataset(dataset_name: str) -> bool:
    return dataset_name.startswith("weightedPDB_4w_prot_lig_apo")


def _af3_chain_ids(n: int) -> List[str]:
    """Generate chain IDs in AF3 'reverse spreadsheet style':
    A..Z, AA, BA, CA..ZA, AB, BB..ZB, ...
    """
    if n <= 0:
        return []
    letters = list(string.ascii_uppercase)
    ids = []
    # 1-letter IDs
    for c in letters:
        ids.append(c)
        if len(ids) == n:
            return ids
    # 2+ letter IDs: first vary the first char, then second char (reverse spreadsheet)
    # AA, BA, CA, ... ZA, AB, BB, ...
    k = 2
    while len(ids) < n:
        for second in letters:          # A, B, C...
            for first in letters:       # A, B, C...
                ids.append(first + second)
                if len(ids) == n:
                    return ids
        k += 1
        # If you ever need 3+ letters, extend similarly; most use-cases won't.
        # Implement 3+ letters only if necessary.
        if k > 2:
            raise ValueError("Need >702 chain IDs; extend generator for 3+ letters.")
    return ids

def build_af3_config(
    entity_to_seq: Dict[str, str],                      # sequences_dict: 1-letter
    entity_to_count: Dict[str, int],
    entity_id_seq_dict: Optional[Dict[str, Union[Sequence[str], "np.ndarray"]]] = None,  # 3-letter
    name: str = "job_from_entities",
    model_seeds: List[int] = None,
    version: int = 1,
    template_free: bool = False,
    msa_free: bool = False,
    print_mod_summary: bool = True,
) -> Dict[str, Any]:
    """
    Build AlphaFold3 (dialect=alphafold3) JSON config for protein-only systems.

    - protein.sequence 直接使用 entity_to_seq (1-letter).
    - 若提供 entity_id_seq_dict，则用其三字母序列识别非标准残基，并写入 protein.modifications:
        {"ptmType": "<CCD>", "ptmPosition": <1-based>}
    """
    if model_seeds is None or len(model_seeds) == 0:
        model_seeds = [1]

    # Validate sequences
    for eid, seq in entity_to_seq.items():
        if not isinstance(seq, str) or len(seq.strip()) == 0:
            raise ValueError(f"Empty sequence for entity {eid}")

    # Validate counts
    for eid, cnt in entity_to_count.items():
        if cnt is None:
            continue
        if not isinstance(cnt, int) or cnt <= 0:
            raise ValueError(f"Invalid chain count for entity {eid}: {cnt}")

    sequences_block: List[Dict[str, Any]] = []
    total_chains = sum(entity_to_count.get(eid, 1) for eid in entity_to_seq.keys())
    chain_ids_pool = _af3_chain_ids(total_chains)  # 复用你已有的 helper
    cursor = 0

    job_modified: List[Tuple[str, int, str]] = []  # (eid, pos, ptmType)

    for eid in sorted(entity_to_seq.keys(), key=lambda x: str(x)):
        seq1 = entity_to_seq[eid].replace(" ", "").replace("\n", "").strip()
        cnt = entity_to_count.get(eid, 1)

        ids = chain_ids_pool[cursor: cursor + cnt]
        cursor += cnt
        id_field: Union[str, List[str]] = ids[0] if cnt == 1 else ids

        protein_obj: Dict[str, Any] = {
            "id": id_field,
            "sequence": seq1,
        }

        # Add modifications if 3-letter residues provided
        if entity_id_seq_dict is not None and eid in entity_id_seq_dict and entity_id_seq_dict[eid] is not None:
            res3_list = _as_res3_list(entity_id_seq_dict[eid])  # type: ignore[arg-type]
            if len(res3_list) != len(seq1):
                raise ValueError(
                    f"Length mismatch for entity {eid}: "
                    f"len(sequences_dict[1-letter])={len(seq1)} vs len(entity_id_seq_dict[3-letter])={len(res3_list)}"
                )

            mods: List[Dict[str, Any]] = []
            for i0, r3 in enumerate(res3_list):
                if r3 in _STD3:
                    continue
                pos = i0 + 1  # 1-based
                mods.append({"ptmType": r3, "ptmPosition": pos})
                job_modified.append((eid, pos, r3))

            if mods:
                protein_obj["modifications"] = mods

        if template_free or msa_free:
            protein_obj["templates"] = []
        if msa_free:
            protein_obj["unpairedMsa"] = ""
            protein_obj["pairedMsa"] = ""

        sequences_block.append({"protein": protein_obj})

    if print_mod_summary and job_modified:
        uniq_ptm = sorted({ptm for _, _, ptm in job_modified})
        print(
            f"[{name}] contains modified residues: {uniq_ptm} "
            f"(total modified positions={len(job_modified)}). Examples: {job_modified[:10]}"
        )

    return {
        "name": name,
        "modelSeeds": model_seeds,
        "sequences": sequences_block,
        "dialect": "alphafold3",
        "version": version,
    }

# def build_af3_config(
#     entity_to_seq: Dict[str, str],
#     entity_to_count: Dict[str, int],
#     name: str = "job_from_entities",
#     model_seeds: List[int] = None,
#     version: int = 1,
#     template_free: bool = False,
#     msa_free: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Build AlphaFold3 (dialect=alphafold3) JSON config for protein-only systems.

#     template_free=True  -> add "templates": []
#     msa_free=True       -> set unpairedMsa="" and pairedMsa="" (and typically templates:[])
#     """
#     if model_seeds is None or len(model_seeds) == 0:
#         model_seeds = [1]

#     # Validate
#     for eid, seq in entity_to_seq.items():
#         if not isinstance(seq, str) or len(seq.strip()) == 0:
#             raise ValueError(f"Empty sequence for entity {eid}")
#     for eid, cnt in entity_to_count.items():
#         if cnt is None:
#             continue
#         if not isinstance(cnt, int) or cnt <= 0:
#             raise ValueError(f"Invalid chain count for entity {eid}: {cnt}")

#     sequences_block = []
#     chain_ids_pool = _af3_chain_ids(sum(entity_to_count.get(eid, 1) for eid in entity_to_seq.keys()))
#     cursor = 0

#     # Deterministic order: sort by entity_id string
#     for eid in sorted(entity_to_seq.keys(), key=lambda x: str(x)):
#         seq = entity_to_seq[eid].replace(" ", "").replace("\n", "").strip()
#         cnt = entity_to_count.get(eid, 1)

#         ids = chain_ids_pool[cursor: cursor + cnt]
#         cursor += cnt
#         id_field: Union[str, List[str]] = ids[0] if cnt == 1 else ids

#         protein_obj: Dict[str, Any] = {
#             "id": id_field,
#             "sequence": seq,
#         }
#         if template_free or msa_free:
#             protein_obj["templates"] = []
#         if msa_free:
#             protein_obj["unpairedMsa"] = ""
#             protein_obj["pairedMsa"] = ""

#         sequences_block.append({"protein": protein_obj})

#     return {
#         "name": name,
#         "modelSeeds": model_seeds,
#         "sequences": sequences_block,
#         "dialect": "alphafold3",
#         "version": version,
#     }


def build_af3_from_entity_info(entity_seq_dict, entity_chain_count):
    """
    仅基于:
      - entity_seq_dict: {entity_id: sequence}
      - entity_chain_count: {entity_id: num_chains}
    
    构造 AlphaFold3 multimer 输入 config
    """

    # 1. 构造 entities（序列级别）
    entities = []
    for ent_id, seq in entity_seq_dict.items():
        entities.append({
            "id": f"entity_{ent_id}",
            "type": "protein",
            "sequence": seq
        })

    # 2. 构造 chains（结构链级别）
    chains = []
    chain_letters = list(string.ascii_uppercase)  # A, B, C, ...
    chain_counter = 0

    for ent_id, n_chain in entity_chain_count.items():
        for i in range(n_chain):
            chain_id = chain_letters[chain_counter]
            chains.append({
                "chain_id": f"Chain_{chain_id}",     # 结构链ID
                "entity_id": f"entity_{ent_id}",      # 指向同一序列实体
                "sym_id": i                            # 对称编号
            })
            chain_counter += 1

    # 3. AF3 输入结构
    af3_input = {
        "entities": entities,
        "chains": chains,
        "options": {
            "model_preset": "default",
            "use_msa": True,
            "use_templates": False
        }
    }

    return af3_input


def read_mmcif_to_chain_coords(mmcif_path, atom_name_filter=None):
    """
    读取 mmCIF 文件
    返回:
      {
        "A": np.ndarray(shape=(M,3)),
        "B": np.ndarray(shape=(N,3)),
        ...
      }

    atom_name_filter:
      None        -> 所有原子
      "CA"        -> 只取 Cα
      ["N","CA"]  -> 指定原子
    """

    st = gemmi.read_structure(mmcif_path)
    model = st[0]   # 默认只取第一个 model

    chain_coords = {}

    for chain in model:
        coords = []
        for res in chain:
            for atom in res:
                if atom_name_filter is not None:
                    if isinstance(atom_name_filter, str):
                        if atom.name != atom_name_filter:
                            continue
                    else:
                        if atom.name not in atom_name_filter:
                            continue

                pos = atom.pos
                coords.append([pos.x, pos.y, pos.z])

        if len(coords) > 0:
            chain_coords[chain.name] = np.array(coords, dtype=np.float32)

    return chain_coords


def read_mmcif_to_chain_coords_types(mmcif_path, atom_name_filter=None):
    """
    读取 mmCIF 文件
    返回:
      {
        "A": {
            "coords": np.ndarray(shape=(M, 3)),
            "atom_types": List[str]
        },
        ...
      }

    atom_name_filter:
      None        -> 所有原子
      "CA"        -> 只取 Cα
      ["N","CA"]  -> 指定原子
    """

    st = gemmi.read_structure(mmcif_path)
    model = st[0]   # 默认只取第一个 model

    chain_data = {}

    for chain in model:
        coords = []
        atom_types = []

        for res in chain:
            for atom in res:
                if atom_name_filter is not None:
                    if isinstance(atom_name_filter, str):
                        if atom.name != atom_name_filter:
                            continue
                    else:
                        if atom.name not in atom_name_filter:
                            continue

                pos = atom.pos
                coords.append([pos.x, pos.y, pos.z])
                atom_types.append(atom.name)

        if coords:
            chain_data[chain.name] = {
                "coords": np.array(coords, dtype=np.float32),
                "atom_types": atom_types
            }

    return chain_data

def unique_stable(arr):
    seen = set()
    out = []
    for x in arr:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return np.array(out)

def _af3_chain_ids(n: int) -> List[str]:
    """Generate chain IDs in AF3 'reverse spreadsheet style':
    A..Z, AA, BA, CA..ZA, AB, BB..ZB, ...
    """
    if n <= 0:
        return []
    letters = list(string.ascii_uppercase)
    ids = []
    # 1-letter IDs
    for c in letters:
        ids.append(c)
        if len(ids) == n:
            return ids
    # 2+ letter IDs: first vary the first char, then second char (reverse spreadsheet)
    # AA, BA, CA, ... ZA, AB, BB, ...
    k = 2
    while len(ids) < n:
        for second in letters:          # A, B, C...
            for first in letters:       # A, B, C...
                ids.append(first + second)
                if len(ids) == n:
                    return ids
        k += 1
        # If you ever need 3+ letters, extend similarly; most use-cases won't.
        # Implement 3+ letters only if necessary.
        if k > 2:
            raise ValueError("Need >702 chain IDs; extend generator for 3+ letters.")
    return ids


class BaseSingleDataset(Dataset):
    """
    dataset for a single data source
    data = self.__item__(idx)
    return a dict of features and labels, the keys and the shape are defined in protenix.data.utils
    """

    def __init__(
        self,
        mmcif_dir: Union[str, Path],
        bioassembly_dict_dir: Optional[Union[str, Path]],
        indices_fpath: Union[str, Path],
        cropping_configs: dict[str, Any],
        msa_featurizer: Optional[MSAFeaturizer] = None,
        template_featurizer: Optional[Any] = None,
        name: str = None,
        **kwargs,
    ) -> None:
        super(BaseSingleDataset, self).__init__()

        # Configs
        self.mmcif_dir = mmcif_dir
        self.bioassembly_dict_dir = bioassembly_dict_dir
        self.indices_fpath = indices_fpath
        self.cropping_configs = cropping_configs
        self.name = name
        # General dataset configs
        self.ref_pos_augment = kwargs.get("ref_pos_augment", True)
        self.lig_atom_rename = kwargs.get("lig_atom_rename", False)
        self.reassign_continuous_chain_ids = kwargs.get(
            "reassign_continuous_chain_ids", False
        )
        self.shuffle_mols = kwargs.get("shuffle_mols", False)
        self.shuffle_sym_ids = kwargs.get("shuffle_sym_ids", False)

        # Typically used for test sets
        self.find_pocket = kwargs.get("find_pocket", False)
        self.find_all_pockets = kwargs.get("find_all_pockets", False)  # for dev
        self.find_eval_chain_interface = kwargs.get("find_eval_chain_interface", False)
        self.group_by_pdb_id = kwargs.get("group_by_pdb_id", False)  # for test set
        self.sort_by_n_token = kwargs.get("sort_by_n_token", False)

        # Typically used for training set
        self.random_sample_if_failed = kwargs.get("random_sample_if_failed", False)
        self.use_reference_chains_only = kwargs.get("use_reference_chains_only", False)
        self.is_distillation = kwargs.get("is_distillation", False)

        # Configs for data filters
        self.max_n_token = kwargs.get("max_n_token", -1)
        self.pdb_list = kwargs.get("pdb_list", None)
        if len(self.pdb_list) == 0:
            self.pdb_list = None
        # Used for removing rows in the indices list. Column names and excluded values are specified in this dict.
        self.exclusion_dict = kwargs.get("exclusion", {})
        self.limits = kwargs.get(
            "limits", -1
        )  # Limit number of indices rows, mainly for test
        # Configs for constraint
        self.constraint = kwargs.get("constraint", {})
        if self.constraint.get("enable", False):
            logger.info(f"[{self.name}] constraint config: {self.constraint}")
            # Do not rely on new files for users who do not use constraint feature
            self.ab_top2_clusters = get_antibody_clusters()
            self.constraint_generator = ConstraintFeatureGenerator(
                self.constraint, self.ab_top2_clusters
            )

        self.error_dir = kwargs.get("error_dir", None)
        if self.error_dir is not None:
            os.makedirs(self.error_dir, exist_ok=True)

        self.msa_featurizer = msa_featurizer
        self.template_featurizer = template_featurizer

        # Read data
        self.indices_list = self.read_indices_list(indices_fpath)
        
        self.use_apo_pos = kwargs.get("use_apo_pos", False)
        
        # protein encoder: esm
        esm_info = kwargs.get("esm_config", {})
        esm_info.embedding_dir = f"./esm_embeddings_train/{name}/{esm_info.model_name}"
        esm_info.sequence_fpath = (
            f"./esm_embeddings_train/{name}_prot_sequences.csv"
        )
        self.esm_enable = esm_info.get("enable", False)
        if self.esm_enable:
            os.makedirs(esm_info.embedding_dir, exist_ok=True)
            os.makedirs(os.path.dirname(esm_info.sequence_fpath), exist_ok=True)
            # ESMFeaturizer.precompute_esm_embedding(
            #     self.inputs,
            #     esm_info.model_name,
            #     esm_info.embedding_dir,
            #     esm_info.sequence_fpath,
            #     configs.load_checkpoint_dir,
            # )
            # self.esm_featurizer = ESMFeaturizer(
            #     embedding_dir=esm_info.embedding_dir,
            #     sequence_fpath=esm_info.sequence_fpath,
            #     embedding_dim=esm_info.embedding_dim,
            #     error_dir="./esm_embeddings/",
            # )

    @staticmethod
    def read_pdb_list(pdb_list: Union[list, str]) -> Optional[list]:
        """
        Reads a list of PDB IDs from a file or directly from a list.

        Args:
            pdb_list: A list of PDB IDs or a file path containing PDB IDs.

        Returns:
            A list of PDB IDs if the input is valid, otherwise None.
        """
        if pdb_list is None:
            return None

        if isinstance(pdb_list, list):
            return pdb_list

        with open(pdb_list, "r") as f:
            pdb_filter_list = []
            for l in f.readlines():
                l = l.strip()
                if l:
                    pdb_filter_list.append(l)
        return pdb_filter_list

    def read_indices_list(self, indices_fpath: Union[str, Path]) -> pd.DataFrame:
        """
        Reads and processes a list of indices from a CSV file.

        Args:
            indices_fpath: Path to the CSV file containing the indices.

        Returns:
            A DataFrame containing the processed indices.
        """
        indices_list = read_indices_csv(indices_fpath)
        num_data = len(indices_list)
        logger.info(f"#Rows in indices list: {num_data}")
        # Filter by pdb_list
        if self.pdb_list is not None:
            pdb_filter_list = set(self.read_pdb_list(pdb_list=self.pdb_list))
            indices_list = indices_list[indices_list["pdb_id"].isin(pdb_filter_list)]
            logger.info(f"[filtered by pdb_list] #Rows: {len(indices_list)}")

        # Filter by max_n_token
        if self.max_n_token > 0:
            valid_mask = indices_list["num_tokens"].astype(int) <= self.max_n_token
            removed_list = indices_list[~valid_mask]
            indices_list = indices_list[valid_mask]
            logger.info(f"[removed] #Rows: {len(removed_list)}")
            logger.info(f"[removed] #PDB: {removed_list['pdb_id'].nunique()}")
            logger.info(
                f"[filtered by n_token ({self.max_n_token})] #Rows: {len(indices_list)}"
            )

        # Filter by exclusion_dict
        for col_name, exclusion_list in self.exclusion_dict.items():
            cols = col_name.split("|")
            exclusion_set = {tuple(excl.split("|")) for excl in exclusion_list}

            def is_valid(row):
                return tuple(row[col] for col in cols) not in exclusion_set

            valid_mask = indices_list.apply(is_valid, axis=1)
            indices_list = indices_list[valid_mask].reset_index(drop=True)
            logger.info(
                f"[Excluded by {col_name} -- {exclusion_list}] #Rows: {len(indices_list)}"
            )
        self.print_data_stats(indices_list)

        # Group by pdb_id
        # A list of dataframe. Each contains one pdb with multiple rows.
        if self.group_by_pdb_id:
            indices_list = [
                df.reset_index() for _, df in indices_list.groupby("pdb_id", sort=True)
            ]

        if self.sort_by_n_token:
            # Sort the dataset in a descending order, so that if OOM it will raise Error at an early stage.
            if self.group_by_pdb_id:
                indices_list = sorted(
                    indices_list,
                    key=lambda df: int(df["num_tokens"].iloc[0]),
                    reverse=True,
                )
            else:
                indices_list = indices_list.sort_values(
                    by="num_tokens", key=lambda x: x.astype(int), ascending=False
                ).reset_index(drop=True)

        if self.find_eval_chain_interface:
            # Remove data that does not contain eval_type in the EvaluationChainInterface list
            if self.group_by_pdb_id:
                indices_list = [
                    df
                    for df in indices_list
                    if len(
                        set(df["eval_type"].to_list()).intersection(
                            set(EvaluationChainInterface)
                        )
                    )
                    > 0
                ]
            else:
                indices_list = indices_list[
                    indices_list["eval_type"].apply(
                        lambda x: x in EvaluationChainInterface
                    )
                ]
        if self.limits > 0 and len(indices_list) > self.limits:
            logger.info(
                f"Limit indices list size from {len(indices_list)} to {self.limits}"
            )
            indices_list = indices_list[: self.limits]
        return indices_list

    def print_data_stats(self, df: pd.DataFrame) -> None:
        """
        Prints statistics about the dataset, including the distribution of molecular group types.

        Args:
            df: A DataFrame containing the indices list.
        """
        if self.name:
            logger.info("-" * 10 + f" Dataset {self.name}" + "-" * 10)
        df["mol_group_type"] = df.apply(
            lambda row: "_".join(
                sorted(
                    [
                        str(row["mol_1_type"]),
                        str(row["mol_2_type"]).replace("nan", "intra"),
                    ]
                )
            ),
            axis=1,
        )

        group_size_dict = dict(df["mol_group_type"].value_counts())
        for i, n_i in group_size_dict.items():
            logger.info(f"{i}: {n_i}/{len(df)}({round(n_i*100/len(df), 2)}%)")

        logger.info("-" * 30)
        if "cluster_id" in df.columns:
            n_cluster = df["cluster_id"].nunique()
            for i in group_size_dict:
                n_i = df[df["mol_group_type"] == i]["cluster_id"].nunique()
                logger.info(f"{i}: {n_i}/{n_cluster}({round(n_i*100/n_cluster, 2)}%)")
            logger.info("-" * 30)

        logger.info(f"Final pdb ids: {len(set(df.pdb_id.tolist()))}")
        logger.info("-" * 30)

    def __len__(self) -> int:
        return len(self.indices_list)

    def save_error_data(self, idx: int, error_message: str) -> None:
        """
        Saves the error data for a specific index to a JSON file in the error directory.

        Args:
            idx: The index of the data sample that caused the error.
            error_message: The error message to be saved.
        """
        if self.error_dir is not None:
            sample_indice = self._get_sample_indice(idx=idx)
            data = sample_indice.to_dict()
            data["error"] = error_message
            if self.bioassembly_dict_dir is not None and "pdb_id" in sample_indice:
                data["bioassembly_dict_fpath"] = os.path.join(
                    self.bioassembly_dict_dir, f"{sample_indice.pdb_id}.pkl.gz"
                )

            filename = f"{sample_indice.pdb_id}-{sample_indice.chain_1_id}-{sample_indice.chain_2_id}.json"
            fpath = os.path.join(self.error_dir, filename)
            if not os.path.exists(fpath):
                with open(fpath, "w") as f:
                    json.dump(data, f)

    def _format_sample_error_context(self, idx: int) -> str:
        sample_indice = self._get_sample_indice(idx=idx)
        pdb_id = sample_indice.get("pdb_id", "N/A")
        chain_1_id = sample_indice.get("chain_1_id", "N/A")
        chain_2_id = sample_indice.get("chain_2_id", "N/A")
        bioassembly_dict_fpath = (
            os.path.join(self.bioassembly_dict_dir, f"{pdb_id}.pkl.gz")
            if self.bioassembly_dict_dir is not None and pdb_id != "N/A"
            else "N/A"
        )
        return (
            f"idx={idx}, pdb_id={pdb_id}, chain_1_id={chain_1_id}, "
            f"chain_2_id={chain_2_id}, bioassembly_dict_fpath={bioassembly_dict_fpath}"
        )

    def __getitem__(self, idx: int):
        """
        Retrieves a data sample by processing the given index.
        If an error occurs, it attempts to handle it by either saving the error data or randomly sampling another index.

        Args:
            idx: The index of the data sample to retrieve.

        Returns:
            A dictionary containing the processed data sample.
        """
        # Try at most 50 times.
        last_error_message = ""
        for _ in range(50):
            try:
                data = self.process_one(idx)
                return data
            except Exception as e:
                sample_ctx = self._format_sample_error_context(idx=idx)
                last_error_message = (
                    f"{e}\n[sample_context] {sample_ctx}\n{traceback.format_exc()}"
                )
                self.save_error_data(idx, last_error_message)

                if self.random_sample_if_failed:
                    logger.exception(f"[skip data] {sample_ctx}")
                    # Randomly sample another index and continue.
                    idx = random.choice(range(len(self.indices_list)))
                    continue
                raise RuntimeError(f"[data sample failed] {sample_ctx}") from e
        raise RuntimeError(
            f"[data sample failed after retries] {last_error_message[:2000]}"
        )

    def _get_bioassembly_data(
        self, idx: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        sample_indice = self._get_sample_indice(idx=idx)
        # sample_indice.pdb_id = '3jb0'
        
        if self.bioassembly_dict_dir is not None:
            bioassembly_dict_fpath = os.path.join(
                self.bioassembly_dict_dir, sample_indice.pdb_id + ".pkl.gz"
            )
        else:
            bioassembly_dict_fpath = None

        bioassembly_dict = DataPipeline.get_data_bioassembly(
            bioassembly_dict_fpath=bioassembly_dict_fpath
        )
        bioassembly_dict["pdb_id"] = sample_indice.pdb_id
        return sample_indice, bioassembly_dict, bioassembly_dict_fpath

    @staticmethod
    def _reassign_atom_array_chain_id(atom_array: AtomArray):
        """
        In experiments conducted to observe overfitting effects using training sets,
        the pre-stored AtomArray in the training set may experience issues with discontinuous chain IDs due to filtering.
        Consequently, a temporary patch has been implemented to resolve this issue.

        e.g. 3x6u asym_id_int = [0, 1, 2, ... 18, 20] -> reassigned_asym_id_int [0, 1, 2, ..., 18, 19]
        """

        def _get_contiguous_array(array):
            array_uniq = np.sort(np.unique(array))
            map_dict = {i: idx for idx, i in enumerate(array_uniq)}
            new_array = np.vectorize(map_dict.get)(array)
            return new_array

        atom_array.asym_id_int = _get_contiguous_array(atom_array.asym_id_int)
        atom_array.entity_id_int = _get_contiguous_array(atom_array.entity_id_int)
        atom_array.sym_id_int = _get_contiguous_array(atom_array.sym_id_int)
        return atom_array

    @staticmethod
    def _shuffle_array_based_on_mol_id(token_array: TokenArray, atom_array: AtomArray):
        """
        Shuffle both token_array and atom_array.
        Atoms/tokens with the same mol_id will be shuffled as a integrated component.
        """

        # Get token mol_id
        centre_atom_indices = token_array.get_annotation("centre_atom_index")
        token_mol_id = atom_array[centre_atom_indices].mol_id

        # Get unique molecule IDs and shuffle them in place
        shuffled_mol_ids = np.unique(token_mol_id).copy()
        np.random.shuffle(shuffled_mol_ids)

        # Get shuffled token indices
        original_token_indices = np.arange(len(token_mol_id))
        shuffled_token_indices = []
        for mol_id in shuffled_mol_ids:
            mol_token_indices = original_token_indices[token_mol_id == mol_id]
            shuffled_token_indices.append(mol_token_indices)
        shuffled_token_indices = np.concatenate(shuffled_token_indices)

        # Get shuffled token and atom array
        # Use `CropData.select_by_token_indices` to shuffle safely
        token_array, atom_array, _, _ = CropData.select_by_token_indices(
            token_array=token_array,
            atom_array=atom_array,
            selected_token_indices=shuffled_token_indices,
        )

        return token_array, atom_array

    @staticmethod
    def _assign_random_sym_id(atom_array: AtomArray):
        """
        Assign random sym_id for chains of the same entity_id
        e.g.
        when entity_id = 0
            sym_id_int = [0, 1, 2] -> random_sym_id_int = [2, 0, 1]
        when entity_id = 1
            sym_id_int = [0, 1, 2, 3] -> random_sym_id_int = [3, 0, 1, 2]
        """

        def _shuffle(x):
            x_unique = np.sort(np.unique(x))
            x_shuffled = x_unique.copy()
            np.random.shuffle(x_shuffled)  # shuffle in-place
            map_dict = dict(zip(x_unique, x_shuffled))
            new_x = np.vectorize(map_dict.get)(x)
            return new_x.copy()

        for entity_id in np.unique(atom_array.label_entity_id):
            mask = atom_array.label_entity_id == entity_id
            atom_array.sym_id_int[mask] = _shuffle(atom_array.sym_id_int[mask])
        return atom_array

    def process_one(
        self, idx: int, return_atom_token_array: bool = False
    ) -> dict[str, dict]:
        """
        Processes a single data sample by retrieving bioassembly data, applying various transformations, and cropping the data.
        It then extracts features and labels, and optionally returns the processed atom and token arrays.

        Args:
            idx: The index of the data sample to process.
            return_atom_token_array: Whether to return the processed atom and token arrays.

        Returns:
            A dict containing the input features, labels, basic_info and optionally the processed atom and token arrays.
        """
        # idx = 2333
        sample_indice, bioassembly_dict, bioassembly_dict_fpath = (
            self._get_bioassembly_data(idx=idx)
        )
        
        # sample code to get the token and protein sequence correspondence
        # if self.name == 'posebusters_0925':
        if self.name == 'gen_apo':
            # pass
            pdb_id = bioassembly_dict['pdb_id']
            sequences_dict = bioassembly_dict['sequences']
            
            new_sequence_dict = {}
            for k in sequences_dict:
                seq = sequences_dict[k]
                keep_idx = [i for i, aa in enumerate(seq) if aa != "X"]
                new_seq = "".join(seq[i] for i in keep_idx)
                if len(new_seq) != len(seq):
                    print(f'[{pdb_id}] seq found x')
                new_sequence_dict[k] = new_seq
            sequences_dict = new_sequence_dict
            
            
            # protein_mask = bioassembly_dict['atom_array'].is_protein.astype(bool)
            # protein_entity_id = bioassembly_dict['atom_array'].label_entity_id[protein_mask]
            
            entity_id_seq_dict = {}
            entity_id_chain_num = {}
            for entity_id in sequences_dict.keys():
                entiy_id_mask = bioassembly_dict['atom_array'].label_entity_id == entity_id
                asym_ids = bioassembly_dict['atom_array'][entiy_id_mask].asym_id_int
                chain_num = len(np.unique(asym_ids))
                entity_id_chain_num[entity_id] = chain_num
                
                
                asym_id = asym_ids[0]
                asym_id_mask = bioassembly_dict['atom_array'][entiy_id_mask].asym_id_int == asym_id
                res_names = bioassembly_dict['atom_array'][entiy_id_mask][asym_id_mask].res_name
                res_ids = bioassembly_dict['atom_array'][entiy_id_mask][asym_id_mask].res_id
                merged = []
                last_resid = None

                for r, s in zip(res_names, res_ids):
                    if s != last_resid:
                        merged.append(r)
                        last_resid = s

                merged = np.array(merged)
                merged = merged[merged != 'UNK'] # delete UNK
                
                assert len(merged) == len(sequences_dict[entity_id])
                
                entity_id_seq_dict[entity_id] = merged
            
            
            af3_json = build_af3_config(sequences_dict, entity_id_chain_num, entity_id_seq_dict, name=pdb_id)

            
            
            # af3_json = build_af3_input(asym_id, entity_id, sym_id, sequences_dict)
            output_folder = f"{self.name}_af3_input_all"
            os.makedirs(output_folder, exist_ok=True)
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                else:
                    return obj
                
            clean_af3_json = convert_numpy(af3_json)

            with open(f"{output_folder}/{pdb_id}.json", "w") as f:
                json.dump(clean_af3_json, f, indent=2)
            # continue
            return
            
            
            
            
            
            
            
        # if self.name == 'posebusters_0925':    
            # pass
        if (
            self.name == 'gen_apo_check'
            or self.name == 'posebusters_0925'
            or _is_weighted_apo_dataset(self.name)
        ):
            pdb_id = bioassembly_dict['pdb_id']
            protein_mask = bioassembly_dict['atom_array'].is_protein.astype(bool)
            protein_entity_id = bioassembly_dict['atom_array'].label_entity_id[protein_mask]
            aysm_protein_id = bioassembly_dict['atom_array'].asym_id_int[protein_mask]
            sidx = np.lexsort((aysm_protein_id, protein_entity_id))
            asym_sorted = aysm_protein_id[sidx]
            unique_ids = unique_stable(asym_sorted)
            chain_ids = _af3_chain_ids(len(unique_ids))
            mapping = dict(zip(unique_ids, chain_ids))
            
            mapping_reversed = dict(zip(chain_ids, unique_ids))
            
            # mapped_sorted = np.array([mapping[x] for x in asym_sorted])
            if self.name == 'posebusters_0925':
                apo_protein_path = (
                    f'/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/'
                    f'posebusters_0925_af3_input/af3_predictions/{pdb_id}/{pdb_id}_model.cif'
                )
                if not os.path.exists(apo_protein_path):
                    raise FileNotFoundError(f"apo protein path not found: {apo_protein_path}")
            else:
                apo_protein_path = (
                    f'/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/'
                    f'gen_apo_af3_input/af3_predictions/{pdb_id}/{pdb_id}_model.cif'
                )
                if not os.path.exists(apo_protein_path):
                    apo_protein_path = (
                        f'/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/'
                        f'gen_apo_af3_input_all/af3_predictions/{pdb_id}/{pdb_id}_model.cif'
                    )
                    if not os.path.exists(apo_protein_path):
                        raise FileNotFoundError(f"apo protein path not found: {apo_protein_path}")
            apo_coords_dict = read_mmcif_to_chain_coords_types(apo_protein_path)
            
            apo_atom_coords = np.zeros_like(bioassembly_dict['atom_array'].ref_pos)
            is_protein_num = 0
            for chain_id in chain_ids:
                protein_idx = np.where(protein_mask)[0]   # shape: (N_protein,)
                full_mask = np.zeros_like(protein_mask, dtype=bool)
                asym_id = mapping_reversed[chain_id]
                full_mask[protein_idx] = (aysm_protein_id == asym_id)
                # full_mask[protein_idx] = (mapped_sorted == chain_id)
                apo_atom_coords[full_mask] = apo_coords_dict[chain_id]['coords']
                # assert (bioassembly_dict['atom_array'][full_mask].atom_name == apo_coords_dict[chain_id]['atom_types']).all()
                
                atom_names_1 = bioassembly_dict['atom_array'][full_mask].atom_name
                atom_names_2 = apo_coords_dict[chain_id]['atom_types']

                if not (atom_names_1 == atom_names_2).all():
                    mismatch_idx = np.where(atom_names_1 != atom_names_2)[0][0]
                    print(f"[Atom mismatch] chain {chain_id}, index = {mismatch_idx}")
                    print(f"  bioassembly atom : {atom_names_1[mismatch_idx]}")
                    print(f"  apo atom         : {atom_names_2[mismatch_idx]}")
                    raise ValueError("Atom names do not match")
                is_protein_num += apo_coords_dict[chain_id]['coords'].shape[0]
            
            assert is_protein_num == protein_mask.sum()
            is_ligand = bioassembly_dict['atom_array'].is_ligand.astype(bool)
            rdkit_coords = bioassembly_dict['atom_array'].ref_pos[is_ligand]
            rdkit_coords = rdkit_coords - rdkit_coords.mean(axis=0)
            
            bioassembly_dict['rdkit_coords'] = rdkit_coords
            
            assert is_protein_num + is_ligand.sum()  == bioassembly_dict['atom_array'].shape[0]
            
            # random rotation for the rdkit
            rotation = Rotation.random(num=1)
            rot_matrix = torch.from_numpy(rotation.as_matrix()).float()
            rot_matrix = rot_matrix.squeeze(0).numpy()
            rdkit_coords = rdkit_coords @ rot_matrix.T
            
            
            apo_atom_coords[is_ligand] = rdkit_coords
            bioassembly_dict['apo_atom_array'] = apo_atom_coords
            
            protein_coords = bioassembly_dict['apo_atom_array'][protein_mask]
            protein_coords = protein_coords - protein_coords.mean(axis=0)
            bioassembly_dict['apo_atom_array'][protein_mask] = protein_coords
            complex_apo_coords = bioassembly_dict['apo_atom_array']

        
        # rdkit coordinate
        if 'pdbbind' in self.name or 'posebustersv2' in self.name:
            rdkit_coords = bioassembly_dict['rdkit_coords']
            apo_coords = bioassembly_dict['apo_coords']
            rdkit_idx = np.random.randint(rdkit_coords.shape[0])
            rdkit_coords = rdkit_coords[rdkit_idx]
            # align to the original point
            rdkit_coords = rdkit_coords - rdkit_coords.mean(axis=0)
            if self.name == 'pdbbind_prot_ligand': # only training rotate
                rotation = Rotation.random(num=1)
                rot_matrix = torch.from_numpy(rotation.as_matrix()).float()
                rot_matrix = rot_matrix.squeeze(0).numpy()
                rdkit_coords = rdkit_coords @ rot_matrix.T
            apo_coords = apo_coords - apo_coords.mean(axis=0)
            complex_apo_coords = np.concatenate((apo_coords, rdkit_coords), axis=0)
            bioassembly_dict['apo_atom_array'] = complex_apo_coords
            
            # ref_mask all set to 1
            n_atoms = len(bioassembly_dict['atom_array'].ref_mask)
            bioassembly_dict['atom_array'].ref_mask = np.ones((n_atoms,), dtype=np.int64)
            # print('set all ref_mask to 1')
            # fix the ligand res_id which is all 1
            ligand_atom_array = bioassembly_dict['atom_array'][bioassembly_dict['atom_array'].is_ligand.astype(bool)]
            res_ids = ligand_atom_array.res_id
            unique_res_ids = np.unique(res_ids)
            assert len(unique_res_ids) == 1
            prot_atom_array = bioassembly_dict['atom_array'][~bioassembly_dict['atom_array'].is_ligand.astype(bool)]
            max_prot_res_id = prot_atom_array.res_id.max()
            org_ligand_res_id = unique_res_ids[0]
            assert org_ligand_res_id == 1
            new_ligand_res_id = max_prot_res_id + 1
            bioassembly_dict['atom_array'].res_id[bioassembly_dict['atom_array'].is_ligand.astype(bool)] = new_ligand_res_id
        # elif 'pbbind_test_v2' in self.name:
        #     bioassembly_dict['atom_array'].ref_pos = np.zeros_like(bioassembly_dict['atom_array'].ref_pos)
        #     bioassembly_dict['atom_array'].ref_mask[bioassembly_dict['atom_array'].is_ligand.astype(bool)] = 0
        #     print('pbbind_test_v2 set all ligand ref_mask to 0')
        
        token_num = len(bioassembly_dict["token_array"])
        
        esm_embeddings_dim = 1280
        x = torch.zeros([token_num, esm_embeddings_dim])
        
        token_centre_atom_indices = bioassembly_dict["token_array"].get_annotation(
                "centre_atom_index"
        )
        centre_atom_array = bioassembly_dict["atom_array"][token_centre_atom_indices]
        is_protein = centre_atom_array.is_protein.astype(bool)
        # print('is_protein shape:', is_protein)
        # print('entity ids shape:', centre_atom_array.label_entity_id)
        protein_entity_ids = set(centre_atom_array.label_entity_id[is_protein])
        # iterate the entity_ids to get sequences, get the esm feature
        # for entity_id in protein_entity_ids:
        #     continue
        #     sequence = bioassembly_dict["sequences"][str(entity_id)]
        #     # x_esm = self.esm_featurizer.load_esm_embedding(sequence)
        #     # print(f"entity_id: {entity_id}, sequence: {sequence}, esm feature shape: {x_esm.shape}")
        #     # get the residue id
        #     # res_idx =
        #     entity_protein_mask = (centre_atom_array.label_entity_id == entity_id)
            
        #     # entity_atom_array = centre_atom_array[
        #     #     centre_atom_array.label_entity_id == entity_id
        #     # ]
        #     res_idx = entity_atom_array.res_id - 1  # res_id starts with 1
        #     x[entity_protein_mask] = x_esm[res_idx]
            
        

        if self.use_reference_chains_only:
            # Get the reference chains
            ref_chain_ids = [sample_indice.chain_1_id, sample_indice.chain_2_id]
            if sample_indice.type == "chain":
                ref_chain_ids.pop(-1)
            # Remove other chains from the bioassembly_dict
            # Remove them safely using the crop method
            token_centre_atom_indices = bioassembly_dict["token_array"].get_annotation(
                "centre_atom_index"
            )
            token_chain_id = bioassembly_dict["atom_array"][
                token_centre_atom_indices
            ].chain_id
            is_ref_chain = np.isin(token_chain_id, ref_chain_ids)
            bioassembly_dict["token_array"], bioassembly_dict["atom_array"], _, _ = (
                CropData.select_by_token_indices(
                    token_array=bioassembly_dict["token_array"],
                    atom_array=bioassembly_dict["atom_array"],
                    selected_token_indices=np.arange(len(is_ref_chain))[is_ref_chain],
                )
            )

        if self.shuffle_mols:
            bioassembly_dict["token_array"], bioassembly_dict["atom_array"] = (
                self._shuffle_array_based_on_mol_id(
                    token_array=bioassembly_dict["token_array"],
                    atom_array=bioassembly_dict["atom_array"],
                )
            )

        if self.shuffle_sym_ids:
            bioassembly_dict["atom_array"] = self._assign_random_sym_id(
                bioassembly_dict["atom_array"]
            )

        if self.reassign_continuous_chain_ids:
            bioassembly_dict["atom_array"] = self._reassign_atom_array_chain_id(
                bioassembly_dict["atom_array"]
            )

        max_entity_mol_id = bioassembly_dict["atom_array"].entity_mol_id.max()

        # Crop
        (
            crop_method,
            cropped_token_array,
            cropped_atom_array,
            cropped_msa_features,
            cropped_template_features,
            reference_token_index,
            selected_indices,
            cropped_atom_indices,
        ) = self.crop(
            sample_indice=sample_indice,
            bioassembly_dict=bioassembly_dict,
            **self.cropping_configs,
        )

        feat, label, label_full = self.get_feature_and_label(
            idx=idx,
            token_array=cropped_token_array,
            atom_array=cropped_atom_array,
            msa_features=cropped_msa_features,
            template_features=cropped_template_features,
            full_atom_array=bioassembly_dict["atom_array"],
            is_spatial_crop="spatial" in crop_method.lower(),
            max_entity_mol_id=max_entity_mol_id,
        )

        # pass the orginal tokens 
        # feat['org_token_num'] = token_num # the token number before cropping
        # feat['select_tokens'] = selected_indices # the selected token indices after cropping
        if (
            'pdbbind' in self.name
            or 'posebustersv2' in self.name
            or 'posebusters_0925' in self.name
            or _is_weighted_apo_dataset(self.name)
        ) and self.use_apo_pos:
            feat['apo_atom_array'] = torch.tensor(complex_apo_coords, dtype=feat['ref_pos'].dtype)
            if self.cropping_configs['crop_size'] > -1:
                feat['apo_atom_array'] = feat['apo_atom_array'][cropped_atom_indices]
                 
        feat['sequences'] = bioassembly_dict["sequences"] # esm embedding for all tokens
        feat['protein_entity_ids'] = protein_entity_ids
        
        # print(f'protein sequence entity ids: {protein_entity_ids}')
        # print(f'sequences: {bioassembly_dict["sequences"]}')
        
        # Construct an index list that maps each token to its corresponding ESM embedding index.
        full_token_map_idx = np.zeros((token_num, 2)) - 1  # initialize with -1
        for entity_id in protein_entity_ids:
            sequence = bioassembly_dict["sequences"][str(entity_id)]
            # x_esm = self.esm_featurizer.load_esm_embedding(sequence)
            # print(f"entity_id: {entity_id}, sequence: {sequence}, esm feature shape: {x_esm.shape}")
            # get the residue id
            # res_idx =
            entity_protein_mask = (centre_atom_array.label_entity_id == entity_id)
            
            entity_atom_array = centre_atom_array[
                centre_atom_array.label_entity_id == entity_id
            ]
            res_idx = entity_atom_array.res_id - 1  # res_id starts with 1
            token_map_idx = [[x, entity_id] for x in res_idx]
            full_token_map_idx[entity_protein_mask] = token_map_idx
        
        if isinstance(selected_indices, torch.Tensor):
            feat['esm_token_map_idx'] = full_token_map_idx[selected_indices] # cropped
        else:
            feat['esm_token_map_idx'] = full_token_map_idx # no cropping, for example, test set
        feat['protein_entity_ids'] = protein_entity_ids
        # feat['centre_atom_array'] = centre_atom_array
        
        
        
        # Basic info, e.g. dimension related items
        basic_info = {
            "pdb_id": (
                bioassembly_dict["pdb_id"]
                if self.is_distillation is False
                else sample_indice["pdb_id"]
            ),
            "N_asym": torch.tensor([len(torch.unique(feat["asym_id"]))]),
            "N_token": torch.tensor([feat["token_index"].shape[0]]),
            "N_atom": torch.tensor([feat["atom_to_token_idx"].shape[0]]),
            "N_msa": torch.tensor([feat["msa"].shape[0]]),
            "bioassembly_dict_fpath": bioassembly_dict_fpath,
            "N_msa_prot_pair": torch.tensor([feat["prot_pair_num_alignments"]]),
            "N_msa_prot_unpair": torch.tensor([feat["prot_unpair_num_alignments"]]),
            "N_msa_rna_pair": torch.tensor([feat["rna_pair_num_alignments"]]),
            "N_msa_rna_unpair": torch.tensor([feat["rna_unpair_num_alignments"]]),
        }

        for mol_type in ("protein", "ligand", "rna", "dna"):
            abbr = {"protein": "prot", "ligand": "lig"}
            abbr_type = abbr.get(mol_type, mol_type)
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            n_atom = int(mol_type_mask.sum(dim=-1).item())
            n_token = len(torch.unique(feat["atom_to_token_idx"][mol_type_mask]))
            basic_info[f"N_{abbr_type}_atom"] = torch.tensor([n_atom])
            basic_info[f"N_{abbr_type}_token"] = torch.tensor([n_token])

        # Add chain level chain_id
        asymn_id_to_chain_id = {
            atom.asym_id_int: atom.chain_id for atom in cropped_atom_array
        }
        chain_id_list = [
            asymn_id_to_chain_id[asymn_id_int]
            for asymn_id_int in sorted(asymn_id_to_chain_id.keys())
        ]
        basic_info["chain_id"] = chain_id_list

        data = {
            "input_feature_dict": feat,
            "label_dict": label,
            "label_full_dict": label_full,
            "basic": basic_info,
        }

        if return_atom_token_array:
            data["cropped_atom_array"] = cropped_atom_array
            data["cropped_token_array"] = cropped_token_array
        return data

    def crop(
        self,
        sample_indice: pd.Series,
        bioassembly_dict: dict[str, Any],
        crop_size: int,
        method_weights: list[float],
        contiguous_crop_complete_lig: bool = True,
        spatial_crop_complete_lig: bool = True,
        drop_last: bool = True,
        remove_metal: bool = True,
    ) -> tuple[str, TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
        """
        Crops the bioassembly data based on the specified configurations.

        Returns:
            A tuple containing the cropping method, cropped token array, cropped atom array,
                cropped MSA features, and cropped template features.
        """
        return DataPipeline.crop(
            one_sample=sample_indice,
            bioassembly_dict=bioassembly_dict,
            crop_size=crop_size,
            msa_featurizer=self.msa_featurizer,
            template_featurizer=self.template_featurizer,
            method_weights=method_weights,
            contiguous_crop_complete_lig=contiguous_crop_complete_lig,
            spatial_crop_complete_lig=spatial_crop_complete_lig,
            drop_last=drop_last,
            remove_metal=remove_metal,
        )

    def _get_sample_indice(self, idx: int) -> pd.Series:
        """
        Retrieves the sample indice for a given index. If the dataset is grouped by PDB ID, it returns the first row of the PDB-idx.
        Otherwise, it returns the row at the specified index.

        Args:
            idx: The index of the data sample to retrieve.

        Returns:
            A pandas Series containing the sample indice.
        """
        if self.group_by_pdb_id:
            # Row-0 of PDB-idx
            sample_indice = self.indices_list[idx].iloc[0]
        else:
            sample_indice = self.indices_list.iloc[idx]
        return sample_indice

    def _get_pdb_indice(self, idx: int) -> pd.core.series.Series:
        if self.group_by_pdb_id:
            pdb_indice = self.indices_list[idx].copy()
        else:
            pdb_indice = self.indices_list.iloc[idx : idx + 1].copy()
        return pdb_indice

    def _get_eval_chain_interface_mask(
        self, idx: int, atom_array_chain_id: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Retrieves the evaluation chain/interface mask for a given index.

        Args:
            idx: The index of the data sample.
            atom_array_chain_id: An array containing the chain IDs of the atom array.

        Returns:
            A tuple containing the evaluation type, cluster ID, chain 1 mask, and chain 2 mask.
        """
        if self.group_by_pdb_id:
            df = self.indices_list[idx]
        else:
            df = self.indices_list.iloc[idx : idx + 1]

        # Only consider chain/interfaces defined in EvaluationChainInterface
        df = df[df["eval_type"].apply(lambda x: x in EvaluationChainInterface)].copy()
        if len(df) < 1:
            raise ValueError(
                f"Cannot find a chain/interface for evaluation in the PDB."
            )

        def get_atom_mask(row):
            chain_1_mask = atom_array_chain_id == row["chain_1_id"]
            if row["type"] == "chain":
                chain_2_mask = chain_1_mask
            else:
                chain_2_mask = atom_array_chain_id == row["chain_2_id"]
            chain_1_mask = torch.tensor(chain_1_mask).bool()
            chain_2_mask = torch.tensor(chain_2_mask).bool()
            if chain_1_mask.sum() == 0 or chain_2_mask.sum() == 0:
                return None, None
            return chain_1_mask, chain_2_mask

        df["chain_1_mask"], df["chain_2_mask"] = zip(*df.apply(get_atom_mask, axis=1))
        df = df[df["chain_1_mask"].notna()]  # drop NaN

        if len(df) < 1:
            raise ValueError(
                f"Cannot find a chain/interface for evaluation in the atom_array."
            )

        eval_type = np.array(df["eval_type"].tolist())
        cluster_id = np.array(df["cluster_id"].tolist())
        # [N_eval, N_atom]
        chain_1_mask = torch.stack(df["chain_1_mask"].tolist())
        # [N_eval, N_atom]
        chain_2_mask = torch.stack(df["chain_2_mask"].tolist())

        return eval_type, cluster_id, chain_1_mask, chain_2_mask

    def get_constraint_feature(
        self,
        idx,
        atom_array,
        token_array,
        msa_features,
        max_entity_mol_id,
        full_atom_array,
    ):
        sample_indice = self._get_sample_indice(idx=idx)
        pdb_indice = self._get_pdb_indice(idx=idx)
        features_dict = {}
        (
            token_array,
            atom_array,
            msa_features,
            constraint_feature_dict,
            feature_info,
            log_dict,
            full_atom_array,
        ) = self.constraint_generator.generate(
            atom_array,
            token_array,
            sample_indice,
            pdb_indice,
            msa_features,
            max_entity_mol_id,
            full_atom_array,
        )
        features_dict["constraint_feature"] = constraint_feature_dict
        features_dict.update(feature_info)
        features_dict["constraint_log_info"] = log_dict
        return token_array, atom_array, features_dict, msa_features, full_atom_array

    def get_feature_and_label(
        self,
        idx: int,
        token_array: TokenArray,
        atom_array: AtomArray,
        msa_features: dict[str, Any],
        template_features: dict[str, Any],
        full_atom_array: AtomArray,
        is_spatial_crop: bool = True,
        max_entity_mol_id: int = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Get feature and label information for a given data point.
        It uses a Featurizer object to obtain input features and labels, and applies several
        steps to add other features and labels. Finally, it returns the feature dictionary, label
        dictionary, and a full label dictionary.

        Args:
            idx: Index of the data point.
            token_array: Token array representing the amino acid sequence.
            atom_array: Atom array containing atomic information.
            msa_features: Dictionary of MSA features.
            template_features: Dictionary of template features.
            full_atom_array: Full atom array containing all atoms.
            is_spatial_crop: Flag indicating whether spatial cropping is applied, by default True.
            max_entity_mol_id: Maximum entity mol ID in the full atom array.
        Returns:
            A tuple containing the feature dictionary and the label dictionary.

        Raises:
            ValueError: If the ligand cannot be found in the data point.
        """
        features_dict = {}
        if self.constraint.get("enable", False):
            token_array, atom_array, features_dict, msa_features, full_atom_array = (
                self.get_constraint_feature(
                    idx,
                    atom_array,
                    token_array,
                    msa_features,
                    max_entity_mol_id,
                    full_atom_array,
                )
            )

        # Get feature and labels from Featurizer
        feat = Featurizer(
            cropped_token_array=token_array,
            cropped_atom_array=atom_array,
            ref_pos_augment=self.ref_pos_augment,
            lig_atom_rename=self.lig_atom_rename,
        )
        features_dict.update(feat.get_all_input_features())
        labels_dict = feat.get_labels()

        # Permutation list for atom permutation
        features_dict["atom_perm_list"] = feat.get_atom_permutation_list()

        # Labels for multi-chain permutation
        # Note: the returned full_atom_array may contain fewer atoms than the input
        label_full_dict, full_atom_array = Featurizer.get_gt_full_complex_features(
            atom_array=full_atom_array,
            cropped_atom_array=atom_array,
            get_cropped_asym_only=is_spatial_crop,
        )

        # Masks for Pocket Metrics
        if self.find_pocket:
            # Get entity_id of the interested ligand
            sample_indice = self._get_sample_indice(idx=idx)
            if sample_indice.mol_1_type == "ligand":
                lig_entity_id = str(int(sample_indice.entity_1_id))
                lig_chain_id = str(sample_indice.chain_1_id)
            elif sample_indice.mol_2_type == "ligand":
                lig_entity_id = str(int(float(sample_indice.entity_2_id)))
                lig_chain_id = str(sample_indice.chain_2_id)
            else:
                raise ValueError(f"Cannot find ligand from this data point.")
            # Make sure the cropped array contains interested ligand
            assert lig_entity_id in set(atom_array.label_entity_id)
            assert lig_chain_id in set(atom_array.chain_id)

            # Get asym ID of the specific ligand in the `main` pocket
            lig_asym_id = atom_array.label_asym_id[atom_array.chain_id == lig_chain_id]
            assert len(np.unique(lig_asym_id)) == 1
            lig_asym_id = lig_asym_id[0]
            ligands = [lig_asym_id]

            if self.find_all_pockets:
                # Get asym ID of other ligands with the same entity_id
                all_lig_asym_ids = set(
                    full_atom_array[
                        full_atom_array.label_entity_id == lig_entity_id
                    ].label_asym_id
                )
                ligands.extend(list(all_lig_asym_ids - set([lig_asym_id])))

            # Note: the `main` pocket is the 0-indexed one.
            # [N_pocket, N_atom], [N_pocket, N_atom].
            # If not find_all_pockets, then N_pocket = 1.
            interested_ligand_mask, pocket_mask = feat.get_lig_pocket_mask(
                atom_array=full_atom_array, lig_label_asym_id=ligands
            )

            label_full_dict["pocket_mask"] = pocket_mask
            label_full_dict["interested_ligand_mask"] = interested_ligand_mask
            

        # Masks for Chain/Interface Metrics
        if self.find_eval_chain_interface:
            eval_type, cluster_id, chain_1_mask, chain_2_mask = (
                self._get_eval_chain_interface_mask(
                    idx=idx, atom_array_chain_id=full_atom_array.chain_id
                )
            )
            labels_dict["eval_type"] = eval_type  # [N_eval]
            labels_dict["cluster_id"] = cluster_id  # [N_eval]
            labels_dict["chain_1_mask"] = chain_1_mask  # [N_eval, N_atom]
            labels_dict["chain_2_mask"] = chain_2_mask  # [N_eval, N_atom]

        # Make dummy features for not implemented features
        dummy_feats = []
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
        if len(template_features) == 0:
            dummy_feats.append("template")
        else:
            template_features = dict_to_tensor(template_features)
            features_dict.update(template_features)

        features_dict = make_dummy_feature(
            features_dict=features_dict, dummy_feats=dummy_feats
        )
        # Transform to right data type
        features_dict = data_type_transform(feat_or_label_dict=features_dict)
        labels_dict = data_type_transform(feat_or_label_dict=labels_dict)

        # Is_distillation
        features_dict["is_distillation"] = torch.tensor([self.is_distillation])
        if self.is_distillation is True:
            features_dict["resolution"] = torch.tensor([-1.0])
        return features_dict, labels_dict, label_full_dict


def get_msa_featurizer(configs, dataset_name: str, stage: str) -> Optional[Callable]:
    """
    Creates and returns an MSAFeaturizer object based on the provided configurations.

    Args:
        configs: A dictionary containing the configurations for the MSAFeaturizer.
        dataset_name: The name of the dataset.
        stage: The stage of the dataset (e.g., 'train', 'test').

    Returns:
        An MSAFeaturizer object if MSA is enabled in the configurations, otherwise None.
    """
    if "msa" in configs["data"] and configs["data"]["msa"]["enable"]:
        msa_info = configs["data"]["msa"]
        msa_args = deepcopy(msa_info)

        if "msa" in (dataset_config := configs["data"][dataset_name]):
            for k, v in dataset_config["msa"].items():
                if k not in ["prot", "rna"]:
                    msa_args[k] = v
                else:
                    for kk, vv in dataset_config["msa"][k].items():
                        msa_args[k][kk] = vv

        prot_msa_args = msa_args["prot"]
        prot_msa_args.update(
            {
                "dataset_name": dataset_name,
                "merge_method": msa_args["merge_method"],
                "max_size": msa_args["max_size"][stage],
            }
        )

        rna_msa_args = msa_args["rna"]
        rna_msa_args.update(
            {
                "dataset_name": dataset_name,
                "merge_method": msa_args["merge_method"],
                "max_size": msa_args["max_size"][stage],
            }
        )

        return MSAFeaturizer(
            prot_msa_args=prot_msa_args,
            rna_msa_args=rna_msa_args,
            enable_rna_msa=configs.data.msa.enable_rna_msa,
        )

    else:
        return None


class WeightedMultiDataset(Dataset):
    """
    A weighted dataset composed of multiple datasets with weights.
    """

    def __init__(
        self,
        datasets: list[Dataset],
        dataset_names: list[str],
        datapoint_weights: list[list[float]],
        dataset_sample_weights: list[torch.tensor],
    ):
        """
        Initializes the WeightedMultiDataset.
        Args:
            datasets: A list of Dataset objects.
            dataset_names: A list of dataset names corresponding to the datasets.
            datapoint_weights: A list of lists containing sampling weights for each datapoint in the datasets.
            dataset_sample_weights: A list of torch tensors containing sampling weights for each dataset.
        """
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.datapoint_weights = datapoint_weights
        self.dataset_sample_weights = torch.Tensor(dataset_sample_weights)
        self.iteration = 0
        self.offset = 0
        self.init_datasets()

    def init_datasets(self):
        """Calculate global weights of each datapoint in datasets for future sampling."""
        self.merged_datapoint_weights = []
        self.weight = 0.0
        self.dataset_indices = []
        self.within_dataset_indices = []
        for dataset_index, (
            dataset,
            datapoint_weight_list,
            dataset_weight,
        ) in enumerate(
            zip(self.datasets, self.datapoint_weights, self.dataset_sample_weights)
        ):
            # normalize each dataset weights
            weight_sum = sum(datapoint_weight_list)
            datapoint_weight_list = [
                dataset_weight * w / weight_sum for w in datapoint_weight_list
            ]
            self.merged_datapoint_weights.extend(datapoint_weight_list)
            self.weight += dataset_weight
            self.dataset_indices.extend([dataset_index] * len(datapoint_weight_list))
            self.within_dataset_indices.extend(list(range(len(datapoint_weight_list))))
        self.merged_datapoint_weights = torch.tensor(
            self.merged_datapoint_weights, dtype=torch.float64
        )

    def __len__(self) -> int:
        return len(self.merged_datapoint_weights)

    def __getitem__(self, index: int) -> dict[str, dict]:
        return self.datasets[self.dataset_indices[index]][
            self.within_dataset_indices[index]
        ]


def get_weighted_pdb_weight(
    data_type: str,
    cluster_size: int,
    chain_count: dict,
    eps: float = 1e-9,
    beta_dict: Optional[dict] = None,
    alpha_dict: Optional[dict] = None,
) -> float:
    """
    Get sample weight for each example in a weighted PDB dataset.

        data_type (str): Type of data, either 'chain' or 'interface'.
        cluster_size (int): Cluster size of this chain/interface.
        chain_count (dict): Count of each kind of chains, e.g., {"prot": int, "nuc": int, "ligand": int}.
        eps (float, optional): A small epsilon value to avoid division by zero. Default is 1e-9.
        beta_dict (Optional[dict], optional): Dictionary containing beta values for 'chain' and 'interface'.
        alpha_dict (Optional[dict], optional): Dictionary containing alpha values for different chain types.

    Returns:
         float: Calculated weight for the given chain/interface.
    """
    if not beta_dict:
        beta_dict = {
            "chain": 0.5,
            "interface": 1,
        }
    if not alpha_dict:
        alpha_dict = {
            "prot": 3,
            "nuc": 3,
            "ligand": 1,
        }

    assert cluster_size > 0
    assert data_type in ["chain", "interface"]
    beta = beta_dict[data_type]
    assert set(chain_count.keys()).issubset(set(alpha_dict.keys()))
    weight = (
        beta
        * sum(
            [alpha * chain_count[data_mode] for data_mode, alpha in alpha_dict.items()]
        )
        / (cluster_size + eps)
    )
    return weight


def calc_weights_for_df(
    indices_df: pd.DataFrame, beta_dict: dict[str, Any], alpha_dict: dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate weights for each example in the dataframe.

    Args:
        indices_df: A pandas DataFrame containing the indices.
        beta_dict: A dictionary containing beta values for different data types.
        alpha_dict: A dictionary containing alpha values for different data types.

    Returns:
        A pandas DataFrame with an column 'weights' containing the calculated weights.
    """
    # Specific to assembly, and entities (chain or interface)
    indices_df["pdb_sorted_entity_id"] = indices_df.apply(
        lambda x: f"{x['pdb_id']}_{x['assembly_id']}_{'_'.join(sorted([str(x['entity_1_id']), str(x['entity_2_id'])]))}",
        axis=1,
    )

    entity_member_num_dict = {}
    for pdb_sorted_entity_id, sub_df in indices_df.groupby("pdb_sorted_entity_id"):
        # Number of repeatative entities in the same assembly
        entity_member_num_dict[pdb_sorted_entity_id] = len(sub_df)
    indices_df["pdb_sorted_entity_id_member_num"] = indices_df.apply(
        lambda x: entity_member_num_dict[x["pdb_sorted_entity_id"]], axis=1
    )

    cluster_size_record = {}
    for cluster_id, sub_df in indices_df.groupby("cluster_id"):
        cluster_size_record[cluster_id] = len(set(sub_df["pdb_sorted_entity_id"]))

    weights = []
    for _, row in indices_df.iterrows():
        data_type = row["type"]
        cluster_size = cluster_size_record[row["cluster_id"]]
        chain_count = {"prot": 0, "nuc": 0, "ligand": 0}
        for mol_type in [row["mol_1_type"], row["mol_2_type"]]:
            if chain_count.get(mol_type) is None:
                continue
            chain_count[mol_type] += 1
        # Weight specific to (assembly, entity(chain/interface))
        weight = get_weighted_pdb_weight(
            data_type=data_type,
            cluster_size=cluster_size,
            chain_count=chain_count,
            beta_dict=beta_dict,
            alpha_dict=alpha_dict,
        )
        weights.append(weight)
    indices_df["weights"] = weights / indices_df["pdb_sorted_entity_id_member_num"]
    return indices_df


def get_sample_weights(
    sampler_type: str,
    indices_df: pd.DataFrame = None,
    beta_dict: dict = {
        "chain": 0.5,
        "interface": 1,
    },
    alpha_dict: dict = {
        "prot": 3,
        "nuc": 3,
        "ligand": 1,
    },
    force_recompute_weight: bool = False,
) -> Union[pd.Series, list[float]]:
    """
    Computes sample weights based on the specified sampler type.

    Args:
        sampler_type: The type of sampler to use ('weighted' or 'uniform').
        indices_df: A pandas DataFrame containing the indices.
        beta_dict: A dictionary containing beta values for different data types.
        alpha_dict: A dictionary containing alpha values for different data types.
        force_recompute_weight: Whether to force recomputation of weights even if they already exist.

    Returns:
        A list of sample weights.

    Raises:
        ValueError: If an unknown sampler type is provided.
    """
    if sampler_type == "weighted":
        assert indices_df is not None
        if "weights" not in indices_df.columns or force_recompute_weight:
            indices_df = calc_weights_for_df(
                indices_df=indices_df,
                beta_dict=beta_dict,
                alpha_dict=alpha_dict,
            )
        return indices_df["weights"].astype("float32")
    elif sampler_type == "uniform":
        assert indices_df is not None
        return [1 / len(indices_df) for _ in range(len(indices_df))]
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def get_datasets(
    configs: ConfigDict, error_dir: Optional[str],
) -> tuple[WeightedMultiDataset, dict[str, BaseSingleDataset]]:
    """
    Get training and testing datasets given configs

    Args:
        configs: A ConfigDict containing the dataset configurations.
        error_dir: The directory where error logs will be saved.

    Returns:
        A tuple containing the training dataset and a dictionary of testing datasets.
    """

    def _get_dataset_param(config_dict, dataset_name: str, stage: str):
        # Template_featurizer is under development
        # Lig_atom_rename/shuffle_mols/shuffle_sym_ids do not affect the performance very much
        return {
            "name": dataset_name,
            **config_dict["base_info"],
            "cropping_configs": config_dict["cropping_configs"],
            "error_dir": error_dir,
            "msa_featurizer": get_msa_featurizer(configs, dataset_name, stage),
            "template_featurizer": None,
            "lig_atom_rename": config_dict.get("lig_atom_rename", False),
            "shuffle_mols": config_dict.get("shuffle_mols", False),
            "shuffle_sym_ids": config_dict.get("shuffle_sym_ids", False),
            "constraint": config_dict.get("constraint", {}),
        }

    data_config = configs.data
    logger.info(f"Using train sets {data_config.train_sets}")
    assert len(data_config.train_sets) == len(
        data_config.train_sampler.train_sample_weights
    )
    train_datasets = []
    datapoint_weights = []
    
    
    esm_config = configs.get("esm", None)
    
    use_apo_pos = configs.model.diffusion_module.use_apo_pos or configs.model.input_embedder.use_apo_pos
    
    for train_name in data_config.train_sets:
        config_dict = data_config[train_name].to_dict()
        dataset_param = _get_dataset_param(
            config_dict, dataset_name=train_name, stage="train"
        )
        dataset_param["ref_pos_augment"] = data_config.get(
            "train_ref_pos_augment", True
        )
        dataset_param["limits"] = data_config.get("limits", -1)
        dataset_param["esm_config"] = esm_config
        dataset_param["use_apo_pos"] = use_apo_pos
        train_dataset = BaseSingleDataset(**dataset_param)
        # for debug:
        # test_data = train_dataset[0]
        train_datasets.append(train_dataset)
        datapoint_weights.append(
            get_sample_weights(
                **data_config[train_name]["sampler_configs"],
                indices_df=train_dataset.indices_list,
            )
        )
    train_dataset = WeightedMultiDataset(
        datasets=train_datasets,
        dataset_names=data_config.train_sets,
        datapoint_weights=datapoint_weights,
        dataset_sample_weights=data_config.train_sampler.train_sample_weights,
    )

    test_datasets = {}
    test_sets = data_config.test_sets
    for test_name in test_sets:
        config_dict = data_config[test_name].to_dict()
        dataset_param = _get_dataset_param(
            config_dict, dataset_name=test_name, stage="test"
        )
        dataset_param["esm_config"] = esm_config
        dataset_param["ref_pos_augment"] = data_config.get("test_ref_pos_augment", True)
        dataset_param["use_apo_pos"] = use_apo_pos
        test_dataset = BaseSingleDataset(**dataset_param)
        test_datasets[test_name] = test_dataset
    return train_dataset, test_datasets
