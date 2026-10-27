#!/usr/bin/env bash
# D3 exactness set (2026-10-24/25, a CUDA machine, not the rig). Env: D = the check dir (README.md); MODELS = Flint GGUFs;
# LOCK = the GPU lock file (default /tmp/gpu4.lock); STORE_SRC / TRAIN where named below.
# Q20/D3: every numeric case through the depth-general build in one hold of the GPU 4 lock: the six-anchor cases (the
# window derived from the six-card split must reproduce the heads' staircases), the trained R1 head on real Q51 data,
# and the depth-general cases on 3-12-card splits and a spare card. Device table; correctness only.
set -u
D=${D:?D=<the D3 check dir, see ../README.md>}
flock "${LOCK:-/tmp/gpu4.lock}" docker run --rm --label pascal-quants=1 --name pq-q20-d3-num2 --gpus '"device=4"' \
    -v $D:$D -e LD_LIBRARY_PATH=$D/out/cuda12-libs -e D=$D nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -c '
    B=$D/out/build/bin; C=$D/cases
    run() { # name split kind tol
        r=$($B/llama-pipe-drafter-check $C/$1/drafter.gguf $C/$1/case.bin --split $2 ${3:+--kind $3} --tol $4 2>&1)
        echo "$r" > $D/check/num2-$1.log
        echo "$1 ($2${3:+ $3}): $(echo "$r" | grep -o "DRAFTER_CHECK.*" || echo "$r" | grep -iE "error|fail" | tail -2)"
    }
    run A-f16      9,8,8,8,8,8 "" 0.05
    run A-fuse-f16 9,8,8,8,8,8 "" 0.05
    run A-fuse-q8  9,8,8,8,8,8 "" 0.25
    run B-f16      9,8,8,8,8,8 pre 0.05
    run C-f16      9,8,8,8,8,8 "" 0.05
    run R1real-f16 9,8,8,8,8,8 "" 0.05
    run G-c6       9,8,8,8,8,8 "" 0.05
    run G-c9       6,6,6,6,5,5,5,5,5 "" 0.05
    run G-c9l      6,6,6,6,6,6,6,5,2 "" 0.05
    run G-c12      5,4,4,4,4,4,4,4,4,4,4,4 "" 0.05
    run G-c4       13,12,12,12 "" 0.05
    run G-c3       17,16,16 "" 0.05
    run G-c6pre    9,8,8,8,8,8 pre 0.05
    run G-c9-q8    6,6,6,6,5,5,5,5,5 "" 0.25
'
