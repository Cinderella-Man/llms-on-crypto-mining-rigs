#!/usr/bin/env bash
# D3 exactness set (2026-10-24/25, a CUDA machine, not the rig). Env: D = the check dir (README.md); MODELS = Flint GGUFs;
# LOCK = the GPU lock file (default /tmp/gpu4.lock); STORE_SRC / TRAIN where named below.
# Q20/D3: random-weight heads and reference test cases for llama-pipe-drafter-check (CPU only, pytorch image, 4 threads),
# plus random heads of the real shape for the executor's exactness checks.
set -euo pipefail
D=${D:?D=<the D3 check dir, see ../README.md>}
docker run --rm --label pascal-quants=1 --name pq-q20-d3-cases -v "$D:/d" -e OMP_NUM_THREADS=4 \
    -e PYTHONPATH=/d/src/gguf-py pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime bash -euo pipefail -c '
    cd /d/py
    python check_export.py case /d/cases/A-f16
    python check_export.py case /d/cases/A-fuse-f16 --fuse
    python check_export.py case /d/cases/A-fuse-q8 --fuse --type q8_0
    python check_export.py case /d/cases/B-f16 --head B --windows 5,5,5,6,1,2,5
    python check_export.py case /d/cases/C-f16 --head C --windows 1,1,1,1
    mkdir -p /d/heads
    python check_export.py head /d/heads/rand-A.gguf
    python check_export.py head /d/heads/rand-A-fuse.gguf --fuse
    python check_export.py head /d/heads/rand-B.gguf --head B
'
