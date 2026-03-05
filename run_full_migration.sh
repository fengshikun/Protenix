#!/usr/bin/env bash
set -euo pipefail
SRC=/home/dataset-local/tmp/zsl/Protenix
DST=/home/dataset-assist-0/tmp/zsl/zsl/Protenix
OLD='/home/dataset-local/tmp/zsl/Protenix'
NEW='/home/dataset-assist-0/tmp/zsl/zsl/Protenix'

rsync -a --info=progress2 "$SRC"/ "$DST"/
# keep migrated repo runnable from new root
if command -v rg >/dev/null 2>&1; then
  rg -l "$OLD" "$DST/configs" | xargs -r sed -i "s|$OLD|$NEW|g"
fi

echo "[DONE] full migration completed at $(date '+%F %T')"
