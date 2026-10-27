#!/usr/bin/env bash
# D3 exactness set (2026-10-24/25, a CUDA machine, not the rig). Env: D = the check dir (README.md); MODELS = Flint GGUFs;
# LOCK = the GPU lock file (default /tmp/gpu4.lock); STORE_SRC / TRAIN where named below.
# Q20/D3: every reference case again after window correction (tokens in flight older than the split's window have
# passed the last card: final-state rows like committed ones; for a fused head with the first fused depth only), into
# cases3/: the six-anchor cases (windows longer than the six-card window exercise it), R1 on real Q51 data, and the
# depth-general cases on 3-12-card splits (G-c12's windows of 9-11 against a 7-row staircase). CPU only, low CPU share.
# The real-data case only with REAL_HEAD=<train.py folder of a six-anchor fused head trained on the strict data list>
# (the R1 head it used on 2026-10-25 was trained on excluded data and is deleted).
set -euo pipefail
D=${D:?D=<the D3 check dir, see ../README.md>}
docker run --rm --cpu-shares=256 --label pascal-quants=1 --name pq-q20-d3-cases3 -v "$D:/d" \
    -v "${TRAIN:-$D}:/t:ro" -e OMP_NUM_THREADS=4 -e REAL_HEAD="${REAL_HEAD:-}" \
    -e PYTHONPATH=/d/src/gguf-py pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime bash -euo pipefail -c '
    cd /d/py
    C=/d/cases3
    mkdir -p $C
    run() { echo "$1: $(python check_export.py "$@" 2>&1 | tail -1 | grep -o "\"window\": [^]]*]\|\"older\": \"[a-z0-9L]*\"\|Error.*" | tr "\n" " ")"; }
    run case $C/A-f16
    run case $C/A-fuse-f16 --fuse
    run case $C/A-fuse-q8 --fuse --type q8_0
    run case $C/B-f16 --head B --kind pre --windows 5,5,5,6,1,2,5
    run case $C/C-f16 --head C --windows 1,1,1,1
    run case $C/G-c6  --general --split 9,8,8,8,8,8
    run case $C/G-c9  --general --split 6,6,6,6,5,5,5,5,5 --windows 6,6,6,7,8,1,2,6
    run case $C/G-c9l --general --split 6,6,6,6,6,6,6,5,2 --windows 7,7,8,9,1,2,7
    run case $C/G-c12 --general --split 5,4,4,4,4,4,4,4,4,4,4,4 --windows 9,9,10,11,1,3,9
    run case $C/G-c4  --general --split 13,12,12,12 --windows 3,3,3,4,1,2,3
    run case $C/G-c3  --general --split 17,16,16 --windows 2,2,2,3,1,2
    run case $C/G-c6pre --general --split 9,8,8,8,8,8 --kind pre --windows 5,5,5,6,1,2,5
    run case $C/G-c9-q8 --general --split 6,6,6,6,5,5,5,5,5 --windows 6,6,6,7,8,1,2,6 --type q8_0 --f16 head
    if [ -n "$REAL_HEAD" ]; then
        python check_export.py real $C/R1real-f16 --weights $REAL_HEAD --vocab /t/vocab.npy \
            --flint /t/flint --feats /t/pack/q51.bin --seq 3 --type f16 2>&1 | tail -1 | cut -c1-300
    fi
'
