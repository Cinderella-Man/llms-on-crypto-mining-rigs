#!/usr/bin/env bash
# C662 (Q1150): which fork default explains each difference from stock upstream on a CPU. Stock b12246's test-llama-archs
# (built by docs/data/2026-11-10_ladder6/run/stock_identity.sh) writes the logits of the named architectures; the
# branch's CPU build compares with one setting flipped.
# Usage (on the build machine): STOCK=<stock bin/> BRANCH=<branch CPU bin/> OUT=<dir> bash stock_flips.sh
set -uo pipefail
: "${STOCK:?}" "${BRANCH:?}" "${OUT:?}"
mkdir -p "$OUT"; rm -f "$OUT/done"; : > "$OUT/summary.txt"
docker run --rm --cpu-shares=256 ${DOCKER_ARGS:-} -v "$STOCK:$STOCK:ro" -v "$BRANCH:$BRANCH:ro" -v "$OUT:/k" \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -c '
    cmp() { # <name> <arch> <env...>
        local n=$1 a=$2; shift 2
        rm -f /k/logits-*.f32
        env LLAMA_ARCHS_LOGITS=/k/logits- nice -n 19 '"$STOCK"'/test-llama-archs -b CPU -s 1 -a "^$a\$" > /k/$n.stock.log 2>&1
        env "$@" LLAMA_ARCHS_LOGITS=/k/logits- nice -n 19 '"$BRANCH"'/test-llama-archs -b CPU -s 1 -a "^$a\$" 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > /k/$n.branch.log
        echo "$n ($*): $(grep "logits against" /k/$n.branch.log | sed "s/^.*f32: //" | tr "\n" ";")" >> /k/summary.txt
    }
    cmp qwen4exp-default      qwen4exp  X=1
    cmp qwen4exp-gdnfuse0     qwen4exp  LLAMA_GDN_FUSE=0
    cmp deepseek4-default     deepseek4 X=1
    cmp deepseek4-headtail0   deepseek4 LLAMA_KV_HEAD_TAIL=0
    cmp deepseek4-kv0         deepseek4 LLAMA_KV_HEAD_EMPTY=0 LLAMA_KV_HEAD_TAIL=0 LLAMA_KV_HEAD_FREE=0 LLAMA_KV_RESTORE_CANONICAL=0
    rm -f /k/logits-*.f32
    chown -R '"$(id -u):$(id -g)"' /k'
date -Is > "$OUT/done"
