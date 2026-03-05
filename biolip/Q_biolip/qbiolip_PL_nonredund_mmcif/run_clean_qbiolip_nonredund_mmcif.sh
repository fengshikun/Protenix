#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# One-command pipeline:
#   Q-BioLiP nonredund mmCIF -> cleaned complex CIFs -> Protenix prepare_training_data
#
# Put this .sh and the .py in the SAME directory (e.g. your qbiolip_PL_nonredund_mmcif folder),
# then run:
#   bash run_clean_qbiolip_nonredund_mmcif.sh
#
# It will:
#   1) (optional) extract tarballs if nonredund_*_cif dirs are missing
#   2) build merged complex CIFs with atom-name normalization to avoid Protenix KeyError
#   3) run Protenix scripts/prepare_training_data.py to generate indices.csv + bioassembly/*.pkl.gz
#
# Key env vars (override as needed):
#   PROTENIX_ROOT=/home/dataset-local/tmp/zsl/Protenix
#   OUT_ROOT=/path/to/output
#   WORKERS=32
#   PREP_WORKERS=32
#   UPDATE_CCD=0|1
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR}"

# ---- paths ----
PROTENIX_ROOT="${PROTENIX_ROOT:-/home/dataset-local/tmp/zsl/Protenix}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/clean_qbiolip_nonredund}"

REC_TAR="${REC_TAR:-$DATA_ROOT/nonredund_rec_cif.tar.gz}"
LIG_TAR="${LIG_TAR:-$DATA_ROOT/nonredund_lig_cif.tar.gz}"
REC_DIR="${REC_DIR:-$DATA_ROOT/nonredund_rec_cif}"
LIG_DIR="${LIG_DIR:-$DATA_ROOT/nonredund_lig_cif}"
LIG_JSON="${LIG_JSON:-$DATA_ROOT/ligand.json}"

PY="${PY:-$SCRIPT_DIR/clean_qbiolip_nonredund_mmcif.py}"

# ---- compute resources ----
WORKERS="${WORKERS:-32}"
CHUNKSIZE="${CHUNKSIZE:-200}"
PREP_WORKERS="${PREP_WORKERS:-32}"

# ---- filters (tune) ----
MIN_LIG_HEAVY_ATOMS="${MIN_LIG_HEAVY_ATOMS:-5}"
MAX_LIG_HEAVY_ATOMS="${MAX_LIG_HEAVY_ATOMS:-120}"
# If 1: keep only ligids that exist in ligand.json with a non-empty SMILES (stronger "small-molecule" filter)
REQUIRE_SMILES="${REQUIRE_SMILES:-0}"

# ---- Protenix CCD cache update (recommended if your ligands are newer than 2024-06-08) ----
UPDATE_CCD="${UPDATE_CCD:-0}"

# ---- outputs ----
COMPLEX_DIR="${OUT_ROOT}/complexes"
MANIFEST="${OUT_ROOT}/manifest.tsv"
FAILED="${OUT_ROOT}/failed.tsv"
STATS="${OUT_ROOT}/stats.txt"
CIF_LIST="${OUT_ROOT}/complexes_cif_paths.txt"

INDICES_CSV="${OUT_ROOT}/qbiolip_nonredund_indices.csv"
BIOASSEMBLY_DIR="${OUT_ROOT}/qbiolip_nonredund_bioassembly"

mkdir -p "${OUT_ROOT}"

echo "=== [0/3] Check / extract inputs ==="
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] REC_DIR=${REC_DIR}"
echo "[INFO] LIG_DIR=${LIG_DIR}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] PROTENIX_ROOT=${PROTENIX_ROOT}"

if [[ ! -d "${REC_DIR}" ]]; then
  if [[ ! -s "${REC_TAR}" ]]; then
    echo "[FATAL] missing receptor dir and tarball: ${REC_DIR} / ${REC_TAR}" >&2
    exit 2
  fi
  echo "[INFO] extracting receptor tar: ${REC_TAR}"
  tar -xzf "${REC_TAR}" -C "${DATA_ROOT}"
fi

if [[ ! -d "${LIG_DIR}" ]]; then
  if [[ ! -s "${LIG_TAR}" ]]; then
    echo "[FATAL] missing ligand dir and tarball: ${LIG_DIR} / ${LIG_TAR}" >&2
    exit 2
  fi
  echo "[INFO] extracting ligand tar: ${LIG_TAR}"
  tar -xzf "${LIG_TAR}" -C "${DATA_ROOT}"
fi

echo "[INFO] receptor CIF count: $(ls -1 "${REC_DIR}"/*.cif 2>/dev/null | wc -l || true)"
echo "[INFO] ligand   CIF count: $(ls -1 "${LIG_DIR}"/*.cif 2>/dev/null | wc -l || true)"

echo "=== [1/3] Build merged complex CIFs (clean + normalize) ==="
REQ_SMILES_FLAG=""
if [[ "${REQUIRE_SMILES}" == "1" ]]; then
  REQ_SMILES_FLAG="--require_smiles_in_ligand_json"
fi

python3 "${PY}" \
  --rec_root "${REC_DIR}" \
  --lig_root "${LIG_DIR}" \
  --out_root "${OUT_ROOT}" \
  --ligand_json "${LIG_JSON}" \
  --workers "${WORKERS}" \
  --chunksize "${CHUNKSIZE}" \
  --min_lig_heavy_atoms "${MIN_LIG_HEAVY_ATOMS}" \
  --max_lig_heavy_atoms "${MAX_LIG_HEAVY_ATOMS}" \
  ${REQ_SMILES_FLAG}

echo "=== [2/3] Build CIF list + run Protenix prepare_training_data.py ==="
echo "[INFO] generating CIF list: ${CIF_LIST}"
mkdir -p "${OUT_ROOT}"
find "${COMPLEX_DIR}" -name "*.cif" | sort > "${CIF_LIST}"
echo "[INFO] complex CIF count: $(wc -l < "${CIF_LIST}")"

if [[ "${UPDATE_CCD}" == "1" ]]; then
  echo "[INFO] UPDATE_CCD=1 -> updating Protenix CCD cache (may take a while)"
  python3 "${PROTENIX_ROOT}/scripts/gen_ccd_cache.py" -n "${PREP_WORKERS}"
else
  echo "[INFO] UPDATE_CCD=0 -> skip CCD cache update"
fi

# Use -d: these CIFs are not raw RCSB downloads; we already cleaned them.
python3 "${PROTENIX_ROOT}/scripts/prepare_training_data.py" \
  -i "${CIF_LIST}" \
  -o "${INDICES_CSV}" \
  -b "${BIOASSEMBLY_DIR}" \
  -n "${PREP_WORKERS}" \
  -d

echo "=== DONE ==="
echo "[DONE] complexes: ${COMPLEX_DIR}"
echo "[DONE] manifest:  ${MANIFEST}"
echo "[DONE] failed:    ${FAILED}"
echo "[DONE] stats:     ${STATS}"
echo "[DONE] cif_list:  ${CIF_LIST}"
echo "[DONE] indices:   ${INDICES_CSV}"
echo "[DONE] bioasm:    ${BIOASSEMBLY_DIR}"
