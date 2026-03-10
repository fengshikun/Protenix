
#!/usr/bin/env python3
import argparse, csv, os
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from protenix.data.data_pipeline import DataPipeline
from protenix.utils.file_io import dump_gzip_pickle

REQ_COLS = ["type","pdb_id","cluster_id","entity_1_id","chain_1_id","mol_1_type","cluster_1_id",
            "entity_2_id","chain_2_id","mol_2_type","cluster_2_id"]

def iter_inputs(input_path: str):
    p = Path(input_path)
    if p.is_dir():
        yield from (str(x) for x in sorted(p.glob("*.cif")))
    else:
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

def process_one(mmcif_path: str, out_dir: str, dataset: str, cluster_file: str | None):
    sid = Path(mmcif_path).stem
    out_pkl = Path(out_dir) / f"{sid}.pkl.gz"
    try:
        sample_indices_list, bioassembly_dict = DataPipeline.get_data_from_mmcif(
            mmcif=mmcif_path,
            pdb_cluster_file=cluster_file,
            dataset=dataset,
        )
        if not sample_indices_list or not bioassembly_dict:
            return ("fail", sid, mmcif_path, "empty_output")

        # 强制唯一，避免覆盖
        bioassembly_dict["pdb_id"] = sid
        for r in sample_indices_list:
            r["pdb_id"] = sid

        dump_gzip_pickle(bioassembly_dict, out_pkl)
        return ("ok", sid, mmcif_path, sample_indices_list)

    except Exception as e:
        return ("fail", sid, mmcif_path, f"{type(e).__name__}: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--input_path", required=True)
    ap.add_argument("-o","--output_csv", required=True)
    ap.add_argument("-b","--bio_dir", required=True)
    ap.add_argument("-n","--num_cpu", type=int, default=8)
    ap.add_argument("-d","--distillation", action="store_true")
    ap.add_argument("-c","--cluster_txt", default=None)
    ap.add_argument("--failed_tsv", default=None)
    args = ap.parse_args()

    os.makedirs(args.bio_dir, exist_ok=True)
    if args.failed_tsv is None:
        args.failed_tsv = str(Path(args.output_csv).with_suffix(".failed.tsv"))

    dataset = "Distillation" if args.distillation else "WeightedPDB"
    inputs = list(iter_inputs(args.input_path))
    print("inputs:", len(inputs), "dataset:", dataset, "cpu:", args.num_cpu)

    # header
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REQ_COLS)
        w.writeheader()
    with open(args.failed_tsv, "w", newline="") as f:
        wf = csv.writer(f, delimiter="\t")
        wf.writerow(["sample_id","mmcif_path","reason"])

    results = Parallel(n_jobs=args.num_cpu, backend="loky")(
        delayed(process_one)(p, args.bio_dir, dataset, args.cluster_txt)
        for p in tqdm(inputs, desc="prepare_safe")
    )

    ok = 0
    fail = 0
    with open(args.output_csv, "a", newline="") as fcsv, open(args.failed_tsv, "a", newline="") as ff:
        w = csv.DictWriter(fcsv, fieldnames=REQ_COLS)
        wf = csv.writer(ff, delimiter="\t")
        for status, sid, path, payload in results:
            if status == "ok":
                ok += 1
                for r in payload:
                    w.writerow({k: r.get(k, "") for k in REQ_COLS})
            else:
                fail += 1
                wf.writerow([sid, path, payload])

    print(f"DONE ok={ok} fail={fail}")
    print("bio_dir:", args.bio_dir)
    print("csv:", args.output_csv)
    print("failed:", args.failed_tsv)

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
    main()
