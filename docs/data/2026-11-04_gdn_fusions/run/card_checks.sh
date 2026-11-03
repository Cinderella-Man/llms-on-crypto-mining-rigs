#!/usr/bin/env bash
# C19-C20 (Q437, Q438) on one CUDA card (a P106-090 on the rig, or any card): correctness only, minutes.
#   1. test-backend-ops, the card against the CPU: the fusions' cases (1-8 tokens, 33-512 tokens) and the ops they
#      replace, with the fusions on and with GGML_CUDA_GDN_FUSE=0;
#   2. test-llama-archs on a random qwen4exp (Flash-Next's architecture): CPU logits against the card's, one batch and
#      chunked, with the fused graph and with LLAMA_GDN_FUSE=0, without and with rollback snapshots
#      (LLAMA_ARCHS_N_RS_SEQ=3).
# Usage: BIN=<folder with test-backend-ops and test-llama-archs> OUT=<dir> [DEV=0] card_checks.sh
# Prints one line per run; the logs go to $OUT.
set -uo pipefail
: "${BIN:?BIN=<folder with the test binaries>}" "${OUT:?OUT=<dir>}"
DEV=${DEV:-0}
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DEV

OPS=GATED_DELTA_NET_STATE_ROWS,GDN_CONV,GDN_CONV_CACHE_FUSION,GATED_RMS_NORM,GATED_DELTA_NET,GATED_DELTA_NET_CACHE_FUSION,SSM_CONV,SSM_CONV_BIAS_SILU,RMS_NORM,L2_NORM

strip() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }

for arm in fused off; do
    env_arm=""
    [ $arm = off ] && env_arm=GGML_CUDA_GDN_FUSE=0
    env $env_arm "$BIN/test-backend-ops" -b CUDA0 -o "$OPS" > "$OUT/tbo-$arm.log" 2>&1
    echo "test-backend-ops $arm: rc=$? $(grep -E 'tests passed' "$OUT/tbo-$arm.log" | tail -1)" \
         "FAIL lines: $(grep -c FAIL "$OUT/tbo-$arm.log")"
done

for fuse in 1 0; do
    for rs in 0 3; do
        log=$OUT/archs-fuse$fuse-rs$rs.log
        LLAMA_GDN_FUSE=$fuse GGML_CUDA_GDN_FUSE=$fuse LLAMA_ARCHS_N_RS_SEQ=$rs \
            "$BIN/test-llama-archs" -a '^qwen4exp$' -b CUDA0 -s 1 > "$log" 2>&1
        echo "test-llama-archs fuse=$fuse n_rs_seq=$rs: rc=$? $(strip "$log" | grep -E '^\|.*qwen4exp' | tr -s ' ' | tr '\n' ' ')"
    done
done
