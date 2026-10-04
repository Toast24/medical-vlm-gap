#!/bin/bash
# Background-safe, resumable dataset downloads (ReXVal, IU X-ray). Run from the project root.
# Finished steps are skipped (marker files in logs/downloads/). ~/.netrc is used but never modified.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/downloads; mkdir -p "$LOG"
step() {
  local name=$1; shift
  if [ -f "$LOG/$name.done" ]; then echo "[$(date "+%F %T")] $name: already done, skipped"; return; fi
  echo "[$(date "+%F %T")] $name: START"
  if "$@" > "$LOG/$name.log" 2>&1; then touch "$LOG/$name.done"; echo "[$(date "+%F %T")] $name: DONE"
  else echo "[$(date "+%F %T")] $name: FAILED (see $LOG/$name.log)"; fi
}

rexval() (
  [ -f ~/.netrc ] || { echo "no ~/.netrc (PhysioNet credentials)"; exit 1; }
  mkdir -p data/rexval
  wget -r -N -c -np -nH --cut-dirs=3 -R "index.html*" -P data/rexval \
       https://physionet.org/files/rexval-dataset/1.0.0/
  echo "files downloaded:"; find data/rexval -type f | head -20
)

iu_xray() (
  mkdir -p data/iu-xray && cd data/iu-xray || exit 1
  ok=1
  for f in NLMCXR_reports.tgz NLMCXR_png.tgz; do
    wget -c -N "https://openi.nlm.nih.gov/imgs/collections/$f" || ok=0
  done
  if [ $ok = 1 ]; then
    mkdir -p reports png
    tar -xzf NLMCXR_reports.tgz -C reports && tar -xzf NLMCXR_png.tgz -C png
  else
    echo "OpenI download failed; trying the Hugging Face mirror"
    "$HF_CLI" download ykumards/open-i --repo-type dataset --local-dir hf_mirror
  fi
  echo "reports: $(find . -name '*.xml' | wc -l)  images: $(find . -name '*.png' | wc -l)"
)

source slurm/local.env
HF_CLI="$(dirname "$PYTHON_JUDGE")/hf"; [ -x "$HF_CLI" ] || HF_CLI="$(dirname "$PYTHON_JUDGE")/huggingface-cli"
step rexval rexval
step iu_xray iu_xray
echo "[$(date "+%F %T")] dataset downloads finished"
