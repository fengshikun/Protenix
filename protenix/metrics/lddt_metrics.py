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

from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from protenix.model import sample_confidence


def kabsch(P, Q):
    """最标准的 Kabsch 算法，返回旋转矩阵 R 和平移 t"""
    P_cent = P - P.mean(axis=0, keepdims=True)
    Q_cent = Q - Q.mean(axis=0, keepdims=True)

    H = P_cent.T @ Q_cent
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # reflection fix
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = Q.mean(axis=0) - P.mean(axis=0) @ R
    return R, t


def kabsch_numpy(P, Q, mask=None):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.

    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = np.dot(p.T, q)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Validate right-handed coordinate system
    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] *= -1.0

    # Optimal rotation
    R = np.dot(Vt.T, U.T)

    # RMSD
    if mask is not None:
        p = p[mask]
        q = q[mask]
    
    rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / p.shape[0])

    return R, t, rmsd

def get_complex_level_rankers(scores, keys):
    assert all([k in ["plddt", "gpde", "ranking_score"] for k in keys])
    rankers = {}
    for key in keys:
        if key == "gpde":
            descending = False
        else:
            descending = True
        ranking = scores[key].argsort(dim=0, descending=descending)
        rankers[f"{key}.rank1"] = lambda x, rank1_idx=ranking[0].item(): x[
            ..., rank1_idx
        ]
    return rankers


def add_diff_metrics(scores, ranker_keys):
    diff_metrics = {
        "diff/best_worst": scores["best"] - scores["worst"],
        "diff/best_random": scores["best"] - scores["random"],
        "diff/best_median": scores["best"] - scores["median"],
    }

    for key in ranker_keys:
        diff_metrics.update(
            {
                f"diff/best_{key}": scores["best"] - scores[f"{key}.rank1"],
                f"diff/{key}_median": scores[f"{key}.rank1"] - scores["median"],
            }
        )
    scores.update(diff_metrics)
    return scores


class LDDTMetrics(nn.Module):
    """LDDT: evaluated on chains and interfaces"""

    def __init__(self, configs):
        super(LDDTMetrics, self).__init__()
        self.eps = configs.metrics.lddt.eps
        self.configs = configs
        self.chunk_size = self.configs.infer_setting.lddt_metrics_chunk_size
        self.lddt_base = LDDT(eps=self.eps)

        self.complex_ranker_keys = configs.metrics.get(
            "complex_ranker_keys", ["plddt", "gpde", "ranking_score"]
        )

    def compute_lddt(self, pred_dict: dict, label_dict: dict):
        """compute complex-level and chain/interface-level lddt

        Args:
            pred_dict (Dict): a dictionary containing
                coordinate: [N_sample, N_atom, 3]
            label_dict (Dict): a dictionary containing
                coordinate: [N_sample, N_atom, 3]
                lddt_mask: [N_atom, N_atom]
        """

        out = {}

        # Complex-level
        lddt = self.lddt_base.forward(
            pred_coordinate=pred_dict["coordinate"],
            true_coordinate=label_dict["coordinate"],
            lddt_mask=label_dict["lddt_mask"],
            chunk_size=self.chunk_size,
        )  # [N_sample]
        out["complex"] = lddt

        return out
    
    def compute_ligand_rmsd_with_kabsch(self, gt, preds, lig_mask):
        """
        gt:    (N, 3) ground truth complex coords
        preds: (M, N, 3) predicted complex coords
        lig_mask: (n') ligand atom indices
        """

        M, N, _ = preds.shape
        ligand_rmsds2 = np.zeros(M)
        ligand_rmsds = np.zeros(M)

        for i in range(M):
            pred = preds[i]  # (N, 3)
            R, t, ligand_rmsd = kabsch_numpy(pred, gt, mask=lig_mask)
            ligand_rmsds[i] = ligand_rmsd

            # ---------- Step 1: Kabsch align whole complex ----------
            R, t = kabsch(pred, gt)
            aligned_pred = pred @ R + t  # shape (N, 3)

            # ---------- Step 2: extract ligand after alignment ----------
            aligned_lig = aligned_pred[lig_mask]     # (n', 3)
            gt_lig = gt[lig_mask]                    # (n', 3)
            
            pred_lig = pred[lig_mask]

            # ---------- Step 3: ligand RMSD ----------
            diff = aligned_lig - gt_lig
            ligand_rmsds2[i] = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

        # # ---------- summary ----------
        # ratio_lt_2A = np.mean(ligand_rmsds < 2.0)
        # ratio_lt_5A = np.mean(ligand_rmsds < 5.0)

        return {
            "ligand_rmsds": ligand_rmsds,      
        }

    def aggregate(
        self,
        vals,
        dim: int = -1,
        aggregators: dict = {},
    ):
        N_sample = vals.size(dim)
        median_index = N_sample // 2
        basic_sample_aggregators = {
            "best": lambda x: x.max(dim=dim)[0],
            "worst": lambda x: x.min(dim=dim)[0],
            "random": lambda x: x.select(dim=dim, index=0),
            "mean": lambda x: x.mean(dim=dim),
            "median": lambda x: x.sort(dim=dim, descending=True)[0].select(
                dim=dim, index=median_index
            ),
        }
        sample_aggregators = {**basic_sample_aggregators, **aggregators}

        return {
            agg_name: agg_func(vals)
            for agg_name, agg_func in sample_aggregators.items()
        }

    def aggregate_lddt(self, lddt_dict, per_sample_summary_confidence):

        # Merge summary_confidence results
        confidence_scores = sample_confidence.merge_per_sample_confidence_scores(
            per_sample_summary_confidence
        )

        # Complex-level LDDT
        complex_level_ranker = get_complex_level_rankers(
            confidence_scores, self.complex_ranker_keys
        )

        complex_lddt = self.aggregate(
            lddt_dict["complex"], aggregators=complex_level_ranker
        )
        complex_lddt = add_diff_metrics(complex_lddt, self.complex_ranker_keys)
        # Log metrics
        complex_lddt = {
            f"lddt/complex/{name}": value for name, value in complex_lddt.items()
        }
        return complex_lddt, {}


class LDDT(nn.Module):
    """LDDT base metrics"""

    def __init__(self, eps: float = 1e-10):
        super(LDDT, self).__init__()
        self.eps = eps

    def _chunk_base_forward(self, pred_distance, true_distance) -> torch.Tensor:
        distance_error_l1 = torch.abs(
            pred_distance - true_distance
        )  # [N_sample, N_pair_sparse]
        thresholds = [0.5, 1, 2, 4]
        sparse_pair_lddt = (
            torch.stack([distance_error_l1 < t for t in thresholds], dim=-1)
            .to(dtype=distance_error_l1.dtype)
            .mean(dim=-1)
        )  # [N_sample, N_pair_sparse]
        del distance_error_l1
        # Compute mean
        if sparse_pair_lddt.numel() == 0:  # corespand to all zero in dense mask
            sparse_pair_lddt = torch.zeros_like(sparse_pair_lddt)
        lddt = torch.mean(sparse_pair_lddt, dim=-1)
        return lddt

    def _chunk_forward(
        self, pred_distance, true_distance, chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        if chunk_size is None:
            return self._chunk_base_forward(pred_distance, true_distance)
        else:
            lddt = []
            N_sample = pred_distance.shape[-2]
            no_chunks = N_sample // chunk_size + (N_sample % chunk_size != 0)
            for i in range(no_chunks):
                lddt_i = self._chunk_base_forward(
                    pred_distance[
                        ...,
                        i * chunk_size : (i + 1) * chunk_size,
                        :,
                    ],
                    true_distance,
                )
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)  # [N_sample]
            return lddt

    def _calc_sparse_dist(self, pred_coordinate, true_coordinate, l_index, m_index):
        pred_coords_l = pred_coordinate.index_select(
            -2, l_index
        )  # [N_sample, N_atom_sparse_l, 3]
        pred_coords_m = pred_coordinate.index_select(
            -2, m_index
        )  # [N_sample, N_atom_sparse_m, 3]
        true_coords_l = true_coordinate.index_select(
            -2, l_index
        )  # [N_atom_sparse_l, 3]
        true_coords_m = true_coordinate.index_select(
            -2, m_index
        )  # [N_atom_sparse_m, 3]

        pred_distance_sparse_lm = torch.norm(
            pred_coords_l - pred_coords_m, p=2, dim=-1
        )  # [N_sample, N_pair_sparse]
        true_distance_sparse_lm = torch.norm(
            true_coords_l - true_coords_m, p=2, dim=-1
        )  # [N_sample, N_pair_sparse]
        return pred_distance_sparse_lm, true_distance_sparse_lm

    def forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        lddt_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """LDDT: evaluated on complex, chains and interfaces
        sparse implementation, which largely reduce cuda memory when atom num reaches 10^4 +

        Args:
            pred_coordinate (torch.Tensor): the pred coordinates
                [N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [N_atom, 3]
            lddt_mask (torch.Tensor):
                sparse version of [N_atom, N_atom] atompair mask based on bespoke radius of true distance
                [N_nonzero_mask, 2]

        Returns:
            Dict[str, torch.Tensor]:
                "best": [N_eval]
                "worst": [N_eval]
        """
        lddt_indices = torch.nonzero(lddt_mask, as_tuple=True)
        l_index = lddt_indices[0]
        m_index = lddt_indices[1]
        pred_distance_sparse_lm, true_distance_sparse_lm = self._calc_sparse_dist(
            pred_coordinate, true_coordinate, l_index, m_index
        )
        group_lddt = self._chunk_forward(
            pred_distance_sparse_lm, true_distance_sparse_lm, chunk_size=chunk_size
        )  # [N_sample]
        return group_lddt

    @staticmethod
    def compute_lddt_mask(
        true_coordinate: torch.Tensor,
        true_coordinate_mask: torch.Tensor,
        is_nucleotide: torch.Tensor = None,
        is_nucleotide_threshold: float = 30.0,
        threshold: float = 15.0,
    ):
        # Distance mask
        distance_mask = (
            true_coordinate_mask[..., None] * true_coordinate_mask[..., None, :]
        )
        # Distances for all atom pairs
        # Note: we convert to bf16 for saving cuda memory, if performance drops, do not convert it
        distance = (torch.cdist(true_coordinate, true_coordinate) * distance_mask).to(
            true_coordinate.dtype
        )  # [..., N_atom, N_atom]

        # Local mask
        c_lm = distance < threshold  # [..., N_atom, N_atom]
        if is_nucleotide is not None:
            # Use a different radius for nucleotide
            is_nucleotide_mask = is_nucleotide.bool()[..., None]
            c_lm = (distance < is_nucleotide_threshold) * is_nucleotide_mask + c_lm * (
                ~is_nucleotide_mask
            )

        # Zero-out diagonals of c_lm and cast to float
        c_lm = c_lm * (
            1 - torch.eye(n=c_lm.size(-1), device=c_lm.device, dtype=distance.dtype)
        )
        # Zero-out atom pairs without true coordinates
        c_lm = c_lm * distance_mask  # [..., N_atom, N_atom]
        return c_lm

# Copyright 2025 ByteDance and/or its affiliates.
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

from typing import Optional

import numpy as np


def align_src_to_tar(
    src_pose: np.ndarray,
    tar_pose: np.ndarray,
    atom_mask: Optional[np.ndarray] = None,
    weight: Optional[np.ndarray] = None,
    allow_reflection: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find optimal transformation, rotation (and reflection) of two poses using NumPy.

    Args:
        src_pose (np.ndarray): [N, 3] the pose to perform transformation on
        tar_pose (np.ndarray): [N, 3] the target pose to align src_pose to
        atom_mask (np.ndarray): [N] a mask for atoms
        weight (np.ndarray): [N] a weight vector to be applied
        allow_reflection (bool): whether to allow reflection when finding optimal alignment

    Returns:
        rot: optimal rotation matrix
        translate: optimal translation vector
    """
    if atom_mask is not None:
        atom_mask = atom_mask.astype(float)
        src_pose = src_pose * atom_mask[..., None]
        tar_pose = tar_pose * atom_mask[..., None]
    else:
        atom_mask = np.ones(src_pose.shape[:-1], dtype=float)

    if weight is None:
        weight = atom_mask
    else:
        weight = weight.astype(float)
        weight = weight * atom_mask

    weighted_n_atoms = np.sum(weight, axis=-1, keepdims=True)[..., None]
    src_pose_centroid = (
        np.sum(src_pose * weight[..., None], axis=-2, keepdims=True) / weighted_n_atoms
    )
    src_pose_centered = src_pose - src_pose_centroid
    tar_pose_centroid = (
        np.sum(tar_pose * weight[..., None], axis=-2, keepdims=True) / weighted_n_atoms
    )
    tar_pose_centered = tar_pose - tar_pose_centroid
    H_mat = (src_pose_centered * weight[..., None]).swapaxes(-2, -1) @ (
        tar_pose_centered * atom_mask[..., None]
    )

    u, s, vh = np.linalg.svd(H_mat)
    u = u.swapaxes(-1, -2)
    vh = vh.T

    if not allow_reflection:
        det = np.linalg.det(vh @ u)
        diagonal = np.stack([np.ones_like(det), np.ones_like(det), det], axis=-1)
        rot = np.diagflat(diagonal) @ u
        rot = vh @ rot
    else:
        rot = vh @ u
    translate = tar_pose_centroid - src_pose_centroid @ rot.swapaxes(-1, -2)
    return rot, translate


def apply_transform(pose: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """
    Apply a rotation and translation to a pose.

    Arguments:
        pose: [..., N, 3] the pose to perform transformation on
        rot: optimal rotation matrix
        trans: optimal translation vector

    Returns:
        transformed_pose: [..., N, 3] the transformed pose
    """
    return pose @ rot.swapaxes(-1, -2) + trans


def rmsd(
    pose1: np.ndarray,
    pose2: np.ndarray,
    mask: np.ndarray = None,
    eps: float = 0.0,
    reduce: bool = True,
):
    """
    Compute RMSD between two poses, with the same shape.

    Arguments:
        pred_pose, true_pose: [..., N, 3], two poses with the same shape.
        mask: [..., N], mask to indicate which elements to compute.
        eps: Add a tolerance to avoid floating number issues.
        reduce: Decide the return shape of RMSD.

    Returns:
        rmsd_value: If reduce is True, return the mean of RMSD over batches;
                    else return an array containing each RMSD separately.
    """
    assert pose1.shape == pose2.shape  # [..., N, 3]

    if mask is None:
        mask = np.ones(pose2.shape[:-1], dtype=float)
    else:
        mask = mask.astype(float)

    # Compute squared error.
    err2 = (np.square(pose1 - pose2).sum(axis=-1) * mask).sum(axis=-1) / (
        mask.sum(axis=-1) + eps
    )

    # Calculate RMSD with added epsilon tolerance.
    rmsd_value = np.sqrt(err2 + eps)

    # Option to reduce RMSD to a mean value.
    if reduce:
        rmsd_value = rmsd_value.mean()

    return rmsd_value


def partially_aligned_rmsd(
    src_pose: np.ndarray,
    tar_pose: np.ndarray,
    align_mask: Optional[np.ndarray] = None,
    rmsd_mask: Optional[np.ndarray] = None,
    weight: Optional[np.ndarray] = None,
    eps: float = 0.0,
    reduce: bool = True,
    allow_reflection: bool = False,
):
    """
    RMSD when aligning parts of the complex coordinate,
    does NOT take permutation symmetricity into consideration

    Args:
        src_pose (np.ndarray): [N, 3] Source pose.
        tar_pose (np.ndarray): [N, 3] Target pose.
        align_mask (np.ndarray): [N] A mask representing which coordinates to align.
        rmsd_mask (np.ndarray): [N] A mask representing which coordinates to compute RMSD.
        weight (np.ndarray): [N] A weight tensor assining weights in alignment for each atom.
        eps (float): Add a tolerance to avoid floating number issue in sqrt.
        reduce: Decide the return shape of RMSD.
        allow_reflection (bool): Whether to allow reflection when finding optimal alignment

    Returns:
        aligned_part_rmsd : the RMSD of part being align_masked
        rmsd_value: the RMSD  of part being rmsd_mask
        rot (np.ndarray): optimal rotation matrix
        translate (np.ndarray): optimal translation vector
    """
    rot, translate = align_src_to_tar(
        src_pose,
        tar_pose,
        atom_mask=align_mask,
        weight=weight,
        allow_reflection=allow_reflection,
    )
    transformed_src_pose = apply_transform(src_pose, rot, translate)
    rmsd_value = rmsd(
        transformed_src_pose, tar_pose, mask=rmsd_mask, eps=eps, reduce=reduce
    )
    aligned_part_rmsd = rmsd(
        transformed_src_pose, tar_pose, mask=align_mask, eps=eps, reduce=reduce
    )
    return aligned_part_rmsd, rmsd_value, rot, translate