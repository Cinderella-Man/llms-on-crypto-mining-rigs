#!/usr/bin/env bash
# Where one guess of Flint's MTP head goes on a P106: llama-pipe-drafter-check --bench on the rig's first card (nothing
# else running), the head's window for the release split (6 tokens in flight + 1 commit, an 8-row graph), at the
# server's default 4,096-position ring (--ctx 2048 gives the same 4,352 cells) and at 8,192 positions. Per ring size:
# the plain timing (300 guesses), a per-kernel trace (GGML_CUDA_PROFILE, 100 guesses) and a per-evaluation host/GPU
# trace (GGML_CUDA_PROFILE_HOST, 300 guesses); then the first guess after a miss (one token in flight).
#   RIG=rig BIN=<rig bin folder of Miner Llama v2.0> HEAD=<rig path of the head GGUF> OUT=<local folder> bash profile.sh
# Summaries: scripts/cuda_profile_summary.py OUT/prof-2048.csv --tokens 80; ... OUT/host-2048.csv --tokens 200
set -uo pipefail
RIG=${RIG:-rig}
BIN=${BIN:?rig bin folder}
HEAD=${HEAD:?rig path of the head GGUF}
O=${OUT:?local folder}
SPLIT=${SPLIT:-6,6,5,5,6,5,5,6,5}   # what -ts 6,6,5,5,6,5,5,6,6 assigns to Flint's 49 layers
R=${RIG_TMP:-/tmp/cheap-guess-profile}
mkdir -p "$O"
ssh "$RIG" "rm -rf $R && mkdir -p $R && nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu \
    --format=csv,noheader" > "$O/gpus-before.txt"
for ctx in 2048 8192; do
    for mode in plain prof host; do
        case $mode in
            plain) ENV=""; N=300 ;;
            prof)  ENV="GGML_CUDA_PROFILE=$R/prof-$ctx.csv"; N=100 ;;
            host)  ENV="GGML_CUDA_PROFILE_HOST=$R/host-$ctx.csv"; N=300 ;;
        esac
        echo "=== ctx $ctx $mode $(date +%T)" | tee -a "$O/bench.log"
        ssh "$RIG" "CUDA_VISIBLE_DEVICES=0 $ENV $BIN/llama-pipe-drafter-check $HEAD --bench $N --split $SPLIT --commits 1 \
            --ctx $ctx 2>&1 | grep -E 'DRAFTER_BENCH|init:|error|failed'" | tee -a "$O/bench.log"
    done
done
echo "=== ctx 2048 window 1 $(date +%T)" | tee -a "$O/bench.log"
ssh "$RIG" "CUDA_VISIBLE_DEVICES=0 $BIN/llama-pipe-drafter-check $HEAD --bench 300 --split $SPLIT --commits 1 --ctx 2048 \
    --window 1 2>&1 | grep DRAFTER_BENCH" | tee -a "$O/bench.log"
for f in prof-2048 prof-8192 host-2048 host-8192; do
    ssh "$RIG" "cat $R/$f.csv" > "$O/$f.csv"
done
ssh "$RIG" "rm -rf $R"
echo "=== done $(date +%T)"
