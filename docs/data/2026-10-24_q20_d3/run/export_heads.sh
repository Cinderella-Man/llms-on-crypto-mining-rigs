#!/usr/bin/env bash
# D3 exactness set (2026-10-24/25, a CUDA machine, not the rig). Env: D = the check dir (README.md); MODELS = Flint GGUFs;
# LOCK = the GPU lock file (default /tmp/gpu4.lock); STORE_SRC / TRAIN where named below.
# Q20/D3: trained heads ($TRAIN/heads/<name>/head.pt + config.json) into the side-file format
# for the engine: q8_0 matrices with an f16 output head (export_gguf.py's default), alpha from <name>/alpha.txt (3 if
# missing), a depth-general head's recorded layers = the 26-layer grid. Output heads/<name>.gguf here plus md5s in
# heads/MD5SUMS. CPU only, low CPU share, ~10 s per head.
#   ./export_heads.sh <head name>...
set -euo pipefail
D=${D:?D=<the D3 check dir, see ../README.md>}
[ $# -ge 1 ] || { echo "usage: $0 <head name>..."; exit 1; }
docker run --rm --cpu-shares=256 --label pascal-quants=1 --name pq-q20-d3-export -v "$D:/d" -v "${TRAIN:?TRAIN=<the drafter training dir: heads/, vocab.npy>}:/t:ro" -e OMP_NUM_THREADS=4 \
    -e PYTHONPATH=/d/src/gguf-py pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime bash -euo pipefail -c '
    cd /d/py
    for h in "$@"; do
        H=/t/heads/$h
        a=$(cat $H/alpha.txt 2>/dev/null || echo 3)
        python export_gguf.py --head $H --vocab /t/vocab.npy --out /d/heads/$h.gguf --alpha $a | cut -c1-400
    done' bash "$@"
cd $D/heads
for h in "$@"; do
    sed -i "/  $h.gguf\$/d" MD5SUMS 2> /dev/null || true
    md5sum $h.gguf | tee -a MD5SUMS
done
