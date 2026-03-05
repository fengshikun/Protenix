#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${1:-./qbiolip_PL_nonredund_mmcif}"
mkdir -p "${DEST_DIR}"
cd "${DEST_DIR}"

UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
REF='https://yanglab.qd.sdu.edu.cn/Q-BioLiP/'

URLS_pdb=(
  # 分开的pdb结构 只有小分子
  "https://yanglab.qd.sdu.edu.cn/Q-BioLiP/DATA/application/PL/nonredund_rec.tar.gz"
  "https://yanglab.qd.sdu.edu.cn/Q-BioLiP/DATA/application/PL/nonredund_lig.tar.gz"
  "https://yanglab.qd.sdu.edu.cn/Q-BioLiP/DATA/application/PL/nonredund_fasta.tar.gz"
  "https://yanglab.qd.sdu.edu.cn/Q-BioLiP/DATA/application/PL/PL_nonredund.txt"
)
URLS=(
  # mmcif
  "https://yanglab.qd.sdu.edu.cn/Q-BioLiP/DATA/nonredund_rec_cif.tar.gz"
  "https://yanglab.qd.sdu.edu.cn/Q-BioLiP/DATA/nonredund_lig_cif.tar.gz"
  "https://yanglab.qd.sdu.edu.cn/Q-BioLiP/DATA/Nonredund_annotations.csv"
  "https://yanglab.qd.sdu.edu.cn/Q-BioLiP/ligand/ligand.json"
)



has_aria2() { command -v aria2c >/dev/null 2>&1; }

download_one() {
  local url="$1"
  local fname
  fname="$(basename "$url")"

  echo "==> $fname"

  if has_aria2; then
    # aria2 多连接；https 失败就自动尝试 http
    aria2c -c -x 16 -s 16 -k 1M \
      --user-agent="${UA}" \
      --referer="${REF}" \
      "$url" || \
    aria2c -c -x 16 -s 16 -k 1M \
      --user-agent="${UA}" \
      --referer="${REF}" \
      "${url/https:\/\/yanglab.qd.sdu.edu.cn/http:\/\/yanglab.qd.sdu.edu.cn}"
  else
    # wget：先 https（忽略证书），失败回退 http
    wget -c --no-check-certificate --user-agent="${UA}" --referer="${REF}" "$url" || \
    wget -c --user-agent="${UA}" --referer="${REF}" "${url/https:\/\/yanglab.qd.sdu.edu.cn/http:\/\/yanglab.qd.sdu.edu.cn}"
  fi
}

for u in "${URLS[@]}"; do
  download_one "$u"
done

echo "DONE. Files in ${DEST_DIR}"
ls -lh
