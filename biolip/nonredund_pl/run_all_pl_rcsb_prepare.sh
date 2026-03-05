#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Build full Protenix training data from PL_nonredund.txt
# 0) Extract unique PDB IDs from Assembly_ID (first column)
# 1) Download RCSB original mmCIF (.cif.gz) with concurrency=32 (retry loop)
# 2) Validate gzip + decompress -> .cif (delete bad gz)
# 3) Retry missing with lower concurrency=8
# 4) Run Protenix prepare_training_data.py -> indices.csv + bioassembly/
#
# Inputs:
#   F   : CSV file with header, first col Assembly_ID like 4h4e_1
# Outputs:
#   OUT : mmcif_gz/, mmcif/, prepared/ (pdb_ids.txt, cif_list.txt, indices.csv, bioassembly/, prepare.log)
###############################################################################

F="/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/PL_nonredund.txt"
OUT="/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/all_data"
PROTENIX_ROOT="/home/dataset-local/tmp/zsl/Protenix"

DL_P=32
UNZIP_P=32
RETRY_N=8

mkdir -p "$OUT"/{mmcif_gz,mmcif,prepared,logs,tmp}

echo "=== [0/4] Extract unique PDB IDs from PL_nonredund.txt ==="
PDB_LIST="$OUT/prepared/pdb_ids.txt"

# Assembly_ID is col1, like 4h4e_1; take first 4 chars => pdb_id
awk -F',' 'NR>1{print substr($1,1,4)}' "$F" \
  | tr 'A-Z' 'a-z' | grep -E '^[0-9a-z]{4}$' \
  | sort -u > "$PDB_LIST"

echo "[INFO] Unique PDB IDs: $(wc -l < "$PDB_LIST")"
echo "[INFO] Example PDB IDs:"
head -n 5 "$PDB_LIST" || true

echo "=== [1/4] Download RCSB mmCIF (.cif.gz) with concurrency=$DL_P ==="
DL_LOG="$OUT/logs/download_fail.log"
: > "$DL_LOG"

# Download missing only; retry loop (compatible with older curl)
cat "$PDB_LIST" | xargs -P "$DL_P" -I{} bash -lc '
  pid="{}"
  out="'"$OUT"'/mmcif_gz/${pid}.cif.gz"
  if [ -s "$out" ]; then
    exit 0
  fi
  url="https://files.rcsb.org/download/${pid}.cif.gz"
  for i in $(seq 1 '"$RETRY_N"'); do
    curl -L -sS --connect-timeout 10 --max-time 180 "$url" -o "$out" && break
    sleep 1
  done
  if [ ! -s "$out" ]; then
    echo "[FAIL] ${pid}" >> "'"$DL_LOG"'"
    rm -f "$out"
  fi
'

echo "[INFO] downloaded_gz: $(ls -1 "$OUT"/mmcif_gz/*.cif.gz 2>/dev/null | wc -l || true)"
echo "[INFO] zero_size_gz : $(find "$OUT"/mmcif_gz -name "*.cif.gz" -size 0 2>/dev/null | wc -l || true)"
echo "[INFO] download_fail_lines: $(wc -l < "$DL_LOG" || true)"

echo "=== [2/4] Validate gzip + decompress to .cif (concurrency=$UNZIP_P) ==="
UNZIP_LOG="$OUT/logs/bad_gz_deleted.log"
: > "$UNZIP_LOG"

# Validate each gz; if valid -> decompress; if invalid -> delete gz
find "$OUT"/mmcif_gz -name "*.cif.gz" -print0 \
  | xargs -0 -P "$UNZIP_P" -I{} bash -lc '
    gz="{}"
    base=$(basename "$gz" .gz)
    out="'"$OUT"'/mmcif/${base}"
    if [ -s "$out" ]; then
      exit 0
    fi
    if gzip -t "$gz" 2>/dev/null; then
      gzip -dc "$gz" > "$out" || rm -f "$out"
    else
      echo "[BAD_GZ] $gz" >> "'"$UNZIP_LOG"'"
      rm -f "$gz"
    fi
  '

echo "[INFO] unzipped_cif: $(ls -1 "$OUT"/mmcif/*.cif 2>/dev/null | wc -l || true)"
echo "[INFO] bad_gz_deleted_lines: $(wc -l < "$UNZIP_LOG" || true)"

echo "=== [3/4] Retry missing (lower concurrency=8) ==="
MISSING_LIST="$OUT/tmp/missing_pdb.txt"
python3 - <<PY
from pathlib import Path
out = Path("$OUT")
pdbs = [x.strip() for x in (out/"prepared/pdb_ids.txt").read_text().splitlines() if x.strip()]
missing = []
for pid in pdbs:
    cif = out/"mmcif"/f"{pid}.cif"
    if (not cif.exists()) or cif.stat().st_size == 0:
        missing.append(pid)
(out/"tmp/missing_pdb.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
print("missing_after_first_pass:", len(missing))
PY

if [ -s "$MISSING_LIST" ]; then
  echo "[INFO] Retrying missing downloads with concurrency=8 ..."
  cat "$MISSING_LIST" | xargs -P 8 -I{} bash -lc '
    pid="{}"
    gz="'"$OUT"'/mmcif_gz/${pid}.cif.gz"
    cif="'"$OUT"'/mmcif/${pid}.cif"
    url="https://files.rcsb.org/download/${pid}.cif.gz"

    for i in $(seq 1 '"$RETRY_N"'); do
      curl -L -sS --connect-timeout 10 --max-time 180 "$url" -o "$gz" && break
      sleep 1
    done

    if gzip -t "$gz" 2>/dev/null; then
      gzip -dc "$gz" > "$cif" || rm -f "$cif"
    else
      rm -f "$gz"
    fi
  '
else
  echo "[INFO] No missing files to retry."
fi

echo "[INFO] final_unzipped_cif: $(ls -1 "$OUT"/mmcif/*.cif 2>/dev/null | wc -l || true)"

echo "=== [3.1/4] Quick sanity: check _entity_poly.type exists (should show lines) ==="
grep -R -m 3 -n '^_entity_poly\.type' "$OUT"/mmcif | head -n 10 || true

echo "=== [4/4] Protenix prepare_training_data.py ==="
CIF_LIST="$OUT/prepared/cif_list.txt"
find "$OUT"/mmcif -name "*.cif" | sort > "$CIF_LIST"
echo "[INFO] cif_count: $(wc -l < "$CIF_LIST")"

PREP_LOG="$OUT/prepared/prepare.log"
PYTHONWARNINGS="ignore" python3 -u "$PROTENIX_ROOT/scripts/prepare_training_data.py" \
  -i "$CIF_LIST" \
  -o "$OUT/prepared/indices.csv" \
  -b "$OUT/prepared/bioassembly" \
  -n 32 \
  2>&1 | tee "$PREP_LOG"

echo "=== DONE ==="
echo "[OUT] pdb_ids:      $PDB_LIST"
echo "[OUT] mmcif_gz:     $OUT/mmcif_gz"
echo "[OUT] mmcif:        $OUT/mmcif"
echo "[OUT] indices:      $OUT/prepared/indices.csv"
echo "[OUT] bioassembly:  $OUT/prepared/bioassembly"
echo "[LOG] prepare.log:  $PREP_LOG"
