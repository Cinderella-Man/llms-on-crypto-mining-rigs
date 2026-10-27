#!/usr/bin/env bash
# D3 exactness set (2026-10-24/25, a CUDA machine, not the rig). Env: D = the check dir (README.md); MODELS = Flint GGUFs;
# LOCK = the GPU lock file (default /tmp/gpu4.lock); STORE_SRC / TRAIN where named below.
# Q20/D3: depth-general heads (M2/R2 shape): reference cases on several layer splits (the C++ derives each window
# from the split and must match drafter.split_depths + depth_name), a spare-card case, and random general heads of the
# real shape. CPU only, low CPU share.
set -euo pipefail
D=${D:?D=<the D3 check dir, see ../README.md>}
docker run --rm --cpu-shares=256 --label pascal-quants=1 --name pq-q20-d3-cases2 -v "$D:/d" -e OMP_NUM_THREADS=4 \
    -e PYTHONPATH=/d/src/gguf-py pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime bash -euo pipefail -c '
    cd /d/py
    C=/d/cases
    python check_export.py case $C/G-c6  --general --split 9,8,8,8,8,8 | cut -c1-400
    python check_export.py case $C/G-c9  --general --split 6,6,6,6,5,5,5,5,5 --windows 6,6,6,7,8,1,2,6 | cut -c1-400
    python check_export.py case $C/G-c9l --general --split 6,6,6,6,6,6,6,5,2 --windows 7,7,8,9,1,2,7 | cut -c1-400
    python check_export.py case $C/G-c12 --general --split 5,4,4,4,4,4,4,4,4,4,4,4 --windows 9,9,10,11,1,3,9 | cut -c1-400
    python check_export.py case $C/G-c4  --general --split 13,12,12,12 --windows 3,3,3,4,1,2,3 | cut -c1-400
    python check_export.py case $C/G-c3  --general --split 17,16,16 --windows 2,2,2,3,1,2 | cut -c1-400
    python check_export.py case $C/G-c6pre --general --split 9,8,8,8,8,8 --kind pre --windows 5,5,5,6,1,2,5 | cut -c1-400
    python check_export.py case $C/G-c9-q8 --general --split 6,6,6,6,5,5,5,5,5 --windows 6,6,6,7,8,1,2,6 --type q8_0 --f16 head | cut -c1-400
    python check_export.py head /d/heads/rand-G-h16.gguf --general --f16 head | cut -c1-300
'
