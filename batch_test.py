# import os
# import sys


# model_lst = [
#     {'model_path':'/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_base_default_v0.5.0.pt',
#      "additional_cmd": ""},     
#      # baseline
#     {'model_path': '/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/bridge_exps/protenix_mini_pdbbind_v3_finetune_diffusion_20251225_224559/99999_ema_0.999.pt', "additional_cmd": ""},
#      # squence sft
#     {'model_path': '/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/bridge_exps/protenix_mini_pdbbind_v3_apo_condition_finetune_diffusion_20251226_090013/99999_ema_0.999.pt', 
#      "additional_cmd": "--model.diffusion_module.use_apo_pos True"}, # apo sft
#     {'model_path': '/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/ve_sigma_min0.4_max40_dbim_align_2_20260116_170603/checkpoints/27999_ema_0.999.pt', 
#      "additional_cmd": "--model.diffusion_module.use_apo_pos True --ddbm True --ddbm_configs.pred_mode ve --ddbm_configs.sigma_max 40.0 --ddbm_configs.sigma_min 0.4 --ddbm_configs.train_sampler RealUniformSamplerSquare --ddbm_configs.infer_sampler dbim --ddbm_configs.align_af3 True "} # bridge
# ]


# N_Steps = {1, 2, 5, 10, 20}


# base_test_cmd = f'cd /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix && export CUTLASS_PATH=/opt/cutlass && export WANDB_DIR=/vepfs-mlp2/mlp-public/shikunfeng/Datas/WANDB_LOGS && export WANDB_API_KEY=a46eaf1ea4fdcf3a2a93022568aa1c730c208b50 && /vepfs-mlp2/mlp-public/shikunfeng/Envs/alphainteract2/bin/torchrun --nproc_per_node=1 --master-port 29501 ./runner/train.py -u runner/train.py --run_name protenix_train --seed 42 --base_dir ./output --dtype bf16 --project protenix --use_wandb false --diffusion_batch_size 48 --eval_interval 200 --log_interval 50 --checkpoint_interval 400 --ema_decay 0.999 --train_crop_size 384 --max_steps 100000 --warmup_steps 2000 --lr 0.001 --sample_diffusion.N_step {N_steps} --triangle_attention triattention --triangle_multiplicative cuequivariance --data.train_sets pdbbind_prot_ligand --model.diffusion_module.use_apo_pos False --data.test_sets posebustersv2 --data.posebusters_0925.base_info.max_n_token 768 --data.num_dl_workers 0 --eval_only True --load_checkpoint_path {model_path} --model_name protenix_mini_esm_online_v0.1.0 {additional_cmd} > {log_name}.log 2>&1 & '



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto-generate Volc MLTask conf YAMLs by cloning a template YAML and replacing:
- EntrypointPath
- TaskName
(and optionally keeps everything else unchanged)

It also prints (and optionally executes) submit commands:
  volc ml_task submit --conf <yaml>

Typical usage:
  python gen_task_confs.py --template template.yaml --outdir ./confs --dry-run
  python gen_task_confs.py --template template.yaml --outdir ./confs --submit

Notes:
- This script assumes your template YAML already contains all required fields
  (ImageUrl / ResourceQueueId / TaskRoleSpecs / Storages / etc).
- Ports are auto-incremented per generated task to avoid conflicts.
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import yaml
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString




def force_plain_str(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="")

yaml.SafeDumper.add_representer(str, force_plain_str)

# -----------------------------
# 1) Define experiment matrix
# -----------------------------
@dataclass(frozen=True)
class ExpCfg:
    name: str
    model_path: str
    additional_cmd: str = ""


DEFAULT_EXPS: List[ExpCfg] = [
     ExpCfg(
        name="baseline_ode",
        model_path="/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt",
        additional_cmd=("--sample_diffusion.gamma0 0 "
                        "--sample_diffusion.step_scale_eta 1.0 "
        ),
    ),
    ExpCfg(
        name="baseline",
        model_path="/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/release_data/checkpoint/protenix_mini_esm_v0.5.0.pt",
        additional_cmd="",
    ),
    ExpCfg(
        name="seqsft",
        model_path="/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/bridge_exps/protenix_mini_pdbbind_v3_finetune_diffusion_20251225_224559/99999_ema_0.999.pt",
        additional_cmd="",
    ),
    ExpCfg(
        name="aposft",
        model_path="/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/bridge_exps/protenix_mini_pdbbind_v3_apo_condition_finetune_diffusion_20251226_090013/99999_ema_0.999.pt",
        additional_cmd="--model.diffusion_module.use_apo_pos True",
    ),
    ExpCfg(
        name="aposft_ode",
        model_path="/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/bridge_exps/protenix_mini_pdbbind_v3_apo_condition_finetune_diffusion_20251226_090013/99999_ema_0.999.pt",
        additional_cmd=("--model.diffusion_module.use_apo_pos True "
                        "--sample_diffusion.gamma0 0 "
                        "--sample_diffusion.step_scale_eta 1.0 "
        ),
    ),
    ExpCfg(
        name="bridge",
        model_path="/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/ve_sigma_min0.4_max40_dbim_align_2_20260116_170603/checkpoints/27999_ema_0.999.pt",
        additional_cmd=(
            "--model.diffusion_module.use_apo_pos True "
            "--ddbm True "
            "--ddbm_configs.pred_mode ve "
            "--ddbm_configs.sigma_max 40.0 "
            "--ddbm_configs.sigma_min 0.4 "
            "--ddbm_configs.train_sampler RealUniformSamplerSquare "
            "--ddbm_configs.infer_sampler dbim "
            "--ddbm_configs.align_af3 True"
        ),
    ),
    ExpCfg(
        name="bridge_31999",
        model_path="/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/ve_sigma_min0.4_max40_dbim_align_2_20260116_170603/checkpoints/31999_ema_0.999.pt",
        additional_cmd=(
            "--model.diffusion_module.use_apo_pos True "
            "--ddbm True "
            "--ddbm_configs.pred_mode ve "
            "--ddbm_configs.sigma_max 40.0 "
            "--ddbm_configs.sigma_min 0.4 "
            "--ddbm_configs.train_sampler RealUniformSamplerSquare "
            "--ddbm_configs.infer_sampler dbim "
            "--ddbm_configs.align_af3 True"
        ),
    ),
    ExpCfg(
        name="bridge_23999",
        model_path="/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/ve_sigma_min0.4_max40_dbim_align_2_20260116_170603/checkpoints/23999_ema_0.999.pt",
        additional_cmd=(
            "--model.diffusion_module.use_apo_pos True "
            "--ddbm True "
            "--ddbm_configs.pred_mode ve "
            "--ddbm_configs.sigma_max 40.0 "
            "--ddbm_configs.sigma_min 0.4 "
            "--ddbm_configs.train_sampler RealUniformSamplerSquare "
            "--ddbm_configs.infer_sampler dbim "
            "--ddbm_configs.align_af3 True"
        ),
    ),
    ExpCfg(
        name="bridge_55999",
        model_path="/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/ve_sigma_min0.4_max40_dbim_align_2_20260116_170603/checkpoints/55999_ema_0.999.pt",
        additional_cmd=(
            "--model.diffusion_module.use_apo_pos True "
            "--ddbm True "
            "--ddbm_configs.pred_mode ve "
            "--ddbm_configs.sigma_max 40.0 "
            "--ddbm_configs.sigma_min 0.4 "
            "--ddbm_configs.train_sampler RealUniformSamplerSquare "
            "--ddbm_configs.infer_sampler dbim "
            "--ddbm_configs.align_af3 True"
        ),
    ),
    ExpCfg(
        name="bridge_4w_apo_95999",
        model_path="/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/ve_sigma_min0.4_max40_dbim_align_4w_apo_20260306_102821/checkpoints/95999_ema_0.999.pt",
        additional_cmd=(
            "--model.diffusion_module.use_apo_pos True "
            "--ddbm True "
            "--ddbm_configs.pred_mode ve "
            "--ddbm_configs.sigma_max 40.0 "
            "--ddbm_configs.sigma_min 0.4 "
            "--ddbm_configs.infer_sampler dbim "
            "--ddbm_configs.align_af3 True"
        ),
    ),
    ExpCfg(
        name="bridge_4w_apo_rmsd_lt_30_99999",
        model_path="/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/ve_sigma_min0.4_max40_dbim_align_4w_apo_rmsd_lt_30_20260325_192109/checkpoints/99999_ema_0.999.pt",
        additional_cmd=(
            "--data.train_sets weightedPDB_4w_prot_lig_apo_rmsd_lt_30 "
            "--model.diffusion_module.use_apo_pos True "
            "--ddbm True "
            "--ddbm_configs.pred_mode ve "
            "--ddbm_configs.sigma_max 40.0 "
            "--ddbm_configs.sigma_min 0.4 "
            "--ddbm_configs.train_sampler RealUniformSampler "
            "--ddbm_configs.infer_sampler dbim "
            "--ddbm_configs.align_af3 True"
        ),
    ),
    ExpCfg(
        name="bridge_4w_apo_rmsd_lt_100_99999",
        model_path="/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/ve_sigma_min0.4_max40_dbim_align_4w_apo_rmsd_lt_100_20260325_164258/checkpoints/99999_ema_0.999.pt",
        additional_cmd=(
            "--data.train_sets weightedPDB_4w_prot_lig_apo_rmsd_lt_100 "
            "--model.diffusion_module.use_apo_pos True "
            "--ddbm True "
            "--ddbm_configs.pred_mode ve "
            "--ddbm_configs.sigma_max 40.0 "
            "--ddbm_configs.sigma_min 0.4 "
            "--ddbm_configs.train_sampler RealUniformSampler "
            "--ddbm_configs.infer_sampler dbim "
            "--ddbm_configs.align_af3 True"
        ),
    ),
    ExpCfg(
        name="bridge_4w_apo_pocketmixed_rmsd_lt_100_83999",
        model_path="/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/ve_sigma_min0.4_max40_dbim_align_4w_apo_pocketmixed_rmsd_lt_100_nosie2a_20260411_225257/checkpoints/83999.pt",
        additional_cmd=(
            "--data.train_sets weightedPDB_4w_prot_lig_apo_rmsd_lt_100 "
            "--model.diffusion_module.use_apo_pos True "
            "--ddbm True "
            "--ddbm_configs.pred_mode ve "
            "--ddbm_configs.sigma_max 40.0 "
            "--ddbm_configs.sigma_min 0.4 "
            "--ddbm_configs.train_sampler RealUniformSampler "
            "--ddbm_configs.infer_sampler dbim "
            "--ddbm_configs.align_af3 True "
            "--ddbm_configs.ligand_init_mode mixed_50_50 "
            "--ddbm_configs.apo_pocket_noise_sigma 2.0"
        ),
    ),
    ExpCfg(
        name="bridge_4w_apo_proteincenter_rmsd_lt_100_83999",
        model_path="/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/ve_sigma_min0.4_max40_dbim_align_4w_apo_pocketmixed_rmsd_lt_100_nosie2a_20260411_225257/checkpoints/83999.pt",
        additional_cmd=(
            "--data.train_sets weightedPDB_4w_prot_lig_apo_rmsd_lt_100 "
            "--model.diffusion_module.use_apo_pos True "
            "--ddbm True "
            "--ddbm_configs.pred_mode ve "
            "--ddbm_configs.sigma_max 40.0 "
            "--ddbm_configs.sigma_min 0.4 "
            "--ddbm_configs.train_sampler RealUniformSampler "
            "--ddbm_configs.infer_sampler dbim "
            "--ddbm_configs.align_af3 True "
            "--ddbm_configs.ligand_init_mode protein_center "
            "--ddbm_configs.apo_pocket_noise_sigma 2.0"
        ),
    ),
    ExpCfg(
        name="bridge_pdbbind_prot_ligand_99999",
        model_path="/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/ve_sigma_min0.4_max40_dbim_align_pdbbind_prot_ligand_20260325_163609/checkpoints/99999_ema_0.999.pt",
        additional_cmd=(
            "--data.train_sets pdbbind_prot_ligand "
            "--model.diffusion_module.use_apo_pos True "
            "--ddbm True "
            "--ddbm_configs.pred_mode ve "
            "--ddbm_configs.sigma_max 40.0 "
            "--ddbm_configs.sigma_min 0.4 "
            "--ddbm_configs.train_sampler RealUniformSampler "
            "--ddbm_configs.infer_sampler dbim "
            "--ddbm_configs.align_af3 True"
        ),
    ),
    ExpCfg(
        name="aposft_4w_apo_condition_99999",
        model_path="/vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix/output/protenix_mini_4w_apo_condition_finetune_diffusion_20260324_154907/checkpoints/99999_ema_0.999.pt",
        additional_cmd=(
            "--data.train_sets weightedPDB_4w_prot_lig_apo "
            "--model.diffusion_module.use_apo_pos True"
        ),
    ),
]

DEFAULT_NSTEPS = [1, 5, 10, 20, 200]
DEFAULT_MODEL_N_SEEDS = [1]

TEST_SET_EXTRA_ARGS = {
    "posebusters_0925": "--data.posebusters_0925.base_info.max_n_token 768",
    "posebustersv2": "--data.posebustersv2.base_info.max_n_token 768",
    "pdbbind_test": "--data.pdbbind_test.base_info.max_n_token 768",
}

TEST_SET_GLOBAL_MAX_N_TOKEN = {
    "pdbbind_test_proteinix_overlap": 2000,
    "gen_apo_pdbbind_test_proteinix_overlap": 2000,
    "gen_apo_check_pdbbind_test_proteinix_overlap": 2000,
}


# -----------------------------
# 2) Entrypoint generator
# -----------------------------
ENTRYPOINT_TEMPLATE = (
    "cd /vepfs-mlp2/mlp-public/shikunfeng/Project/Protenix && "
    "export CUTLASS_PATH=/opt/cutlass && "
    "export WANDB_DIR=/vepfs-mlp2/mlp-public/shikunfeng/Datas/WANDB_LOGS && "
    "export WANDB_API_KEY={wandb_api_key} && "
    "/vepfs-mlp2/mlp-public/shikunfeng/Envs/alphainteract2/bin/torchrun "
    "--nproc_per_node=1 --master-port {master_port} "
    "./runner/train.py "
    "--run_name {run_name} "
    "--seed 42 "
    "--base_dir ./output "
    "--dtype bf16 "
    "--project protenix "
    "--use_wandb {use_wandb} "
    "--diffusion_batch_size 48 "
    "--eval_interval {eval_interval} "
    "--log_interval 50 "
    "--checkpoint_interval 400 "
    "--ema_decay 0.999 "
    "--train_crop_size 384 "
    "--max_steps 100000 "
    "--warmup_steps 2000 "
    "--lr 0.001 "
    "--sample_diffusion.N_step {n_step} "
    "--model.N_model_seed {model_n_seed} "
    "--triangle_attention triattention "
    "--triangle_multiplicative cuequivariance "
    "--data.train_sets pdbbind_prot_ligand "
    "--data.test_sets {test_set} "
    "{test_set_args} "
    "--model_name protenix_mini_esm_online_v0.1.0 "
    "--find_unused_parameters True "
    "--load_checkpoint_path {model_path} "
    "{additional_cmd} "
    "--eval_only True "
    "> {log_name}.log 2>&1"
)


def normalize_ws(s: str) -> str:
    return " ".join(s.strip().split())


def build_test_set_args(test_set: str) -> str:
    test_sets = [item.strip() for item in test_set.split(",") if item.strip()]
    args: list[str] = []

    for ds_name in test_sets:
        extra_arg = TEST_SET_EXTRA_ARGS.get(ds_name)
        if extra_arg:
            args.append(extra_arg)

    global_max_tokens = [
        TEST_SET_GLOBAL_MAX_N_TOKEN[ds_name]
        for ds_name in test_sets
        if ds_name in TEST_SET_GLOBAL_MAX_N_TOKEN
    ]
    if global_max_tokens:
        args.append(f"--test_max_n_token {max(global_max_tokens)}")

    return " ".join(args)


def make_task_name(prefix: str, exp: ExpCfg, n_step: int, model_n_seed: int) -> str:
    task_name = f"{prefix}_{exp.name}_step{n_step}"
    if model_n_seed != 1:
        task_name += f"_mseed{model_n_seed}"
    return task_name


def make_entrypoint(
    *,
    task_name: str,
    exp: ExpCfg,
    n_step: int,
    model_n_seed: int,
    master_port: int,
    wandb_api_key: str,
    use_wandb: bool,
    eval_interval: int,
    test_set: str,
) -> str:
    cmd = ENTRYPOINT_TEMPLATE.format(
        wandb_api_key=wandb_api_key,
        master_port=master_port,
        run_name=task_name,
        use_wandb="true" if use_wandb else "false",
        eval_interval=eval_interval,
        n_step=n_step,
        model_n_seed=model_n_seed,
        test_set=test_set,
        test_set_args=build_test_set_args(test_set),
        model_path=exp.model_path,
        additional_cmd=exp.additional_cmd.strip(),
        log_name=task_name,
    )
    return normalize_ws(cmd)


# -----------------------------
# 3) YAML IO + submission
# -----------------------------
def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f)


# def dump_yaml(data: dict, path: Path) -> None:
#     # Keep it readable; exact formatting may differ from original, but keys/values remain correct.
#     with path.open("w", encoding="utf-8") as f:
#         yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, width=10**9)


yaml = YAML()
yaml.width = 10**9  # 避免自动折行

def dump_yaml_ruamel(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)

def submit_task(yaml_path: Path, dry_run: bool) -> None:
    cmd = ["volc", "ml_task", "submit", "--conf", str(yaml_path)]
    print(f"[SUBMIT] {' '.join(cmd)}")
    if dry_run:
        return
    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] submitted {yaml_path.name}")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] submit failed: {yaml_path.name}")
        print(e)


def select_exps(all_exps: Sequence[ExpCfg], only: Sequence[str]) -> List[ExpCfg]:
    if not only:
        return list(all_exps)
    only_set = {x.strip() for x in only if x.strip()}
    selected = [e for e in all_exps if e.name in only_set]
    missing = sorted(list(only_set - {e.name for e in all_exps}))
    if missing:
        raise SystemExit(f"Unknown exp names: {missing}. Available: {[e.name for e in all_exps]}")
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=str, required=True, help="Template YAML path (existing task conf).")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for generated YAMLs.")
    ap.add_argument("--prefix", type=str, default="protenix_eval", help="TaskName prefix.")
    ap.add_argument("--only", nargs="*", default=[], help="Only generate these exp names, e.g. baseline bridge.")
    ap.add_argument("--nsteps", nargs="*", type=int, default=DEFAULT_NSTEPS, help="N_step list.")
    ap.add_argument(
        "--model-n-seeds",
        nargs="*",
        type=int,
        default=DEFAULT_MODEL_N_SEEDS,
        help="Values for --model.N_model_seed, e.g. 1 5.",
    )
    ap.add_argument("--start-port", type=int, default=29501, help="Starting master-port.")
    ap.add_argument("--port-step", type=int, default=1, help="Port increment per task.")
    ap.add_argument("--use-wandb", action="store_true", help="Set --use_wandb true in entrypoint.")
    ap.add_argument("--eval-interval", type=int, default=400, help="--eval_interval in entrypoint.")
    ap.add_argument("--test-set", type=str, default="posebustersv2,pdbbind_test", help="--data.test_sets value.")
    ap.add_argument(
        "--wandb-api-key",
        type=str,
        default=os.environ.get("WANDB_API_KEY", "a46eaf1ea4fdcf3a2a93022568aa1c730c208b50"),
        help="WANDB API key (default: env WANDB_API_KEY else hardcoded).",
    )
    ap.add_argument("--submit", action="store_true", help="Submit generated YAMLs via volc.")
    ap.add_argument("--dry-run", action="store_true", help="Do not submit; only generate and print commands.")
    args = ap.parse_args()

    template_path = Path(args.template).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    template = load_yaml(template_path)
    exps = select_exps(DEFAULT_EXPS, args.only)

    port = args.start_port
    generated: List[Path] = []

    for exp in exps:
        for n in args.nsteps:
            for model_n_seed in args.model_n_seeds:
                task_name = make_task_name(args.prefix, exp, n, model_n_seed)
                entrypoint = make_entrypoint(
                    task_name=task_name,
                    exp=exp,
                    n_step=n,
                    model_n_seed=model_n_seed,
                    master_port=port,
                    wandb_api_key=args.wandb_api_key,
                    use_wandb=args.use_wandb,
                    eval_interval=args.eval_interval,
                    test_set=args.test_set,
                )

                conf = copy.deepcopy(template)
                conf["TaskName"] = task_name
                entrypoint = normalize_ws(entrypoint)
                # entrypoint = f"\n\t{entrypoint}"
                # import pdb; pdb.set_trace()
                conf["Entrypoint"] =  entrypoint
                # conf["EntrypointPath"] = SingleLineStr(entrypoint)
                # conf["EntrypointPath"] = LiteralScalarString(entrypoint)

                # dump_yaml_ruamel(conf, yaml_path)

                # Optional hygiene: remove fields that usually shouldn't be preset in a "submit conf"
                # (safe even if missing)
                for k in ["Id", "CreateTime", "LaunchTime", "FinishTime", "ServerTime", "UpdateTime", "State", "Reason", "ExitCode", "DiagInfo"]:
                    conf.pop(k, None)

                yaml_path = outdir / f"{task_name}.yaml"
                # dump_yaml(conf, yaml_path)
                dump_yaml_ruamel(conf, yaml_path)
                generated.append(yaml_path)

                print(f"[CONF] {yaml_path}")
                print(f"[ENTRYPOINT] {entrypoint}")
                print(f"[CMD] volc ml_task submit --conf {yaml_path}")
                print()

                port += args.port_step

    if args.submit:
        for p in generated:
            submit_task(p, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
