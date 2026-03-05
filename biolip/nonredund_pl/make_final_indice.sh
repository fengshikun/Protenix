#!/usr/bin/env bash
set -euo pipefail

# ========== Paths (edit if needed) ==========
IDX_IN="/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/all_data/prepared/indices.csv"
BIO_DIR="/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/all_data/prepared/bioassembly"
BAD_LIST="/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/select.txt"

IDX_OUT="/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/all_data/prepared/indices_final.csv"
IDX_DROPPED="/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/all_data/prepared/indices_dropped_debug.csv"

echo "=== Build indices_final.csv from indices.csv + bioassembly atom_array ground truth ==="
echo "[IN ] IDX_IN      : $IDX_IN"
echo "[IN ] BIO_DIR     : $BIO_DIR"
echo "[IN ] BAD_LIST    : $BAD_LIST"
echo "[OUT] IDX_OUT     : $IDX_OUT"
echo "[OUT] IDX_DROPPED : $IDX_DROPPED"
echo ""

python - <<'PY'
import gzip, pickle
from pathlib import Path
import pandas as pd
import numpy as np

IDX_IN = Path("/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/all_data/prepared/indices.csv")
BIO_DIR = Path("/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/all_data/prepared/bioassembly")
BAD_PATH = Path("/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/select.txt")

IDX_OUT = Path("/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/all_data/prepared/indices_final.csv")
IDX_DROPPED = Path("/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/all_data/prepared/indices_dropped_debug.csv")

# ---------- 0) load blacklist ----------
bad = set()
if BAD_PATH.exists():
    for line in BAD_PATH.read_text().splitlines():
        line = line.strip()
        if (not line) or line.startswith("#"):
            continue
        bad.add(line.split()[0].upper())

# common water / unknown
bad |= {"HOH","WAT","DOD","H2O",".","?","UNK"}

# ---------- 1) load indices.csv ----------
df = pd.read_csv(IDX_IN)
print("[INFO] indices.csv rows:", len(df))
print("[INFO] columns:", list(df.columns))

need_cols = ["pdb_id", "mol_1_type", "mol_2_type", "cluster_2_id"]
missing_cols = [c for c in need_cols if c not in df.columns]
if missing_cols:
    raise RuntimeError(f"indices.csv missing required columns: {missing_cols}")

# normalize
df["pdb_u"] = df["pdb_id"].astype(str).str.lower()
df["lig_u"] = df["cluster_2_id"].astype(str).str.upper()

# ---------- 2) type filter: only prot-ligand ----------
# keep only mol_1_type == prot and mol_2_type == ligand (case-insensitive)
m1 = df["mol_1_type"].astype(str).str.lower()
m2 = df["mol_2_type"].astype(str).str.lower()
mask_type = (m1 == "prot") & (m2 == "ligand")
df_type = df[mask_type].copy()
print("[INFO] after type filter (prot-ligand):", len(df_type))

# ---------- 3) blacklist filter on ligand id ----------
mask_badlig = df_type["lig_u"].isin(bad)
df_type["drop_reason"] = ""
df_type.loc[mask_badlig, "drop_reason"] = "blacklisted_ligand_id"
df_ok0 = df_type[~mask_badlig].copy()
print("[INFO] after blacklist filter:", len(df_ok0), f"(removed {mask_badlig.sum()})")

# ---------- 4) load ligand sets from bioassembly atom_array (once per pdb) ----------
def extract_ligs_from_obj(obj):
    """
    Return set of hetero res_name (uppercased) excluding blacklist.
    Try atom_array first, then token_array as fallback.
    """
    aa = obj.get("atom_array", None)
    if aa is None:
        aa = obj.get("token_array", None)

    if aa is None:
        return set(), "no_atom_or_token_array"

    # res_name
    if not hasattr(aa, "res_name"):
        return set(), "no_res_name"

    res = np.asarray(aa.res_name).astype(str)
    # hetero mask
    if hasattr(aa, "hetero"):
        het = np.asarray(aa.hetero).astype(bool)
    else:
        # fallback: assume everything non-empty is hetero (conservative)
        het = np.ones_like(res, dtype=bool)

    ligs = set(res[het])
    ligs = {x.upper() for x in ligs if x and (x.upper() not in bad)}
    return ligs, None

pdbs = df_ok0["pdb_u"].unique().tolist()
pdb_to_ligs = {}
stats = {
    "missing_pkl": 0,
    "load_fail": 0,
    "no_atom_or_token_array": 0,
    "no_res_name": 0,
}

for pdb in pdbs:
    pkl = BIO_DIR / f"{pdb}.pkl.gz"
    if not pkl.exists():
        pdb_to_ligs[pdb] = set()
        stats["missing_pkl"] += 1
        continue
    try:
        with gzip.open(pkl, "rb") as f:
            obj = pickle.load(f)
        ligs, err = extract_ligs_from_obj(obj)
        pdb_to_ligs[pdb] = ligs
        if err:
            stats[err] = stats.get(err, 0) + 1
    except Exception:
        pdb_to_ligs[pdb] = set()
        stats["load_fail"] += 1

print("[INFO] bioassembly stats:", stats)

# ---------- 5) keep only rows whose ligand id appears in bioassembly hetero set ----------
def lig_present(pdb, lig):
    return lig in pdb_to_ligs.get(pdb, set())

mask_present = [lig_present(p, l) for p, l in zip(df_ok0["pdb_u"], df_ok0["lig_u"])]
mask_present = pd.Series(mask_present, index=df_ok0.index)

df_ok0["drop_reason"] = ""
df_ok0.loc[~mask_present, "drop_reason"] = "ligand_not_in_atom_array_hetero"

df_keep = df_ok0[mask_present].copy()
df_drop = df_ok0[~mask_present].copy()

# ---------- 6) final cleanup ----------
for c in ["pdb_u", "lig_u"]:
    if c in df_keep.columns:
        df_keep.drop(columns=[c], inplace=True)

# Save outputs
df_keep.to_csv(IDX_OUT, index=False)
df_drop.to_csv(IDX_DROPPED, index=False)

print("")
print("===== SUMMARY =====")
print("input_rows(indices.csv)          :", len(df))
print("prot-ligand rows                 :", len(df_type))
print("after blacklist                  :", len(df_ok0))
print("kept_rows (lig exists in bioasm) :", len(df_keep))
print("dropped_rows                     :", len(df_drop))
print("wrote:", IDX_OUT)
print("wrote dropped debug:", IDX_DROPPED)

# top dropped ligand ids (for diagnosis)
if len(df_drop) > 0:
    top_bad = df_drop["lig_u"].value_counts().head(20)
    print("\nTop dropped ligands (not found in atom_array hetero):")
    print(top_bad.to_string())
PY

echo ""
echo "=== DONE ==="
echo "[OUT] $IDX_OUT"
echo "[OUT] $IDX_DROPPED"
echo ""
echo "Next steps:"
echo "1) Set configs/configs_data.py -> qbiolip_nonredund.base_info.indices_fpath to:"
echo "   $IDX_OUT"
echo "2) Keep use_reference_chains_only consistent with the bioassembly cache you generated."
echo "3) Re-run training."
