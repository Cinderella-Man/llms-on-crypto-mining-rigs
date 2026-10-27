#!/bin/bash
# Q38 (2026-10-23), run on the rig: S4-L0 on 6 cards, tg32 with per-launch-group CUDA event timing
# (GGML_CUDA_PROFILE) on the wide-rows fork build bb7390902, for comparison with run2.sh's profiles.
# Env: BIN (fork build), MODEL (S4-L0 GGUF), OUT (work dir).
B=${BIN:-/opt/llama-builds/llama.cpp-20261023-bb7390902-pascal-p106/bin}
M=${MODEL:-/mnt/models2/flint/flint-S4L0.gguf}
OUT=${OUT:-/tmp/q38}
mkdir -p "$OUT"; cd "$OUT"
echo "== prof $(date +%T)"
LLAMA_NO_MOE_DOWN_RESCALE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 GGML_CUDA_PROFILE=$OUT/prof-s4l0-widek.csv $B/llama-bench -m $M -ngl 99 -fa on -p 0 -n 32 -r 1 -o md > $OUT/prof-s4l0-widek.txt 2>&1
echo "   exit $? $(date +%T)"
echo DONE
