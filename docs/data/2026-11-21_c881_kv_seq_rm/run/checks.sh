#!/usr/bin/env bash
# C881 (Q1590): the CPU checks of the rework against the build it came from, after build.sh built two trees:
#   $WORK/t/base   pascal-p106-v5 with only the test's trace commit added (the reference)
#   $WORK/t/c881   pascal-p106-v5-kv-seq-rm
# Usage (on the build machine): WORK=<work dir> [IMAGE=pq-c565-cpu:24.04] bash checks.sh
# Containers: label pascal-quants=1 and c881=1, --cpu-shares=256, --memory=8g, --network none, only $WORK mounted; the
# two trees' --kv-seq-rm arms run at the same time, one container each, the rest after them.
# Output in $WORK/chk/:
#   seqrm/            seqrm_loop.sh's logs, traces and verdicts per tree
#   kvcache-<tree>-<rules>.log   test-kv-cache with the fork's rules (fork) and with LLAMA_KV_HEAD_TAIL=0 (tail0)
#   archs-<tree>.log             test-llama-archs -s 1: every architecture's random model on the CPU
#   verdicts.tsv                 one line per check (check, verdict, detail)
# Done marker: $WORK/chk/DONE
set -uo pipefail
WORK=${WORK:?WORK=<work dir>}
IMAGE=${IMAGE:-pq-c565-cpu:24.04}
RUN=$(cd "$(dirname "$0")" && pwd)
C=$WORK/chk
mkdir -p "$C/seqrm"
rm -f "$C/DONE"
: > "$C/verdicts.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$C/verdicts.tsv"; }
D="--rm --label pascal-quants=1 --label c881=1 --cpu-shares=256 --memory=8g --memory-swap=8g --network none -v $WORK:$WORK -w /tmp"

# the arms of C740's check (docs/data/2026-11-11_c740_seq_rm_bound), on the CPU backend: every architecture with the
# host mask; every architecture with the mask built from the device copies of the cells (their changed-cell list in
# use) and its check; eight architectures at seeds 2 and 3 the same way; the switch's test hook =2, which must fail.
# Known aborts of the architectures themselves, without the switch too (C835): plamo2's two 4-sequence tests, and with
# the device mask minimax-01's MoE test in 4 streams
MASK=LLAMA_GPU_KQ_MASK=1,LLAMA_GPU_KQ_MASK_CHECK=1
EIGHT='^(llama|qwen4exp|qwen3next|qwen35moe|gemma3|mamba2|deepseek2|deepseek4)$'
# (a test's name is "<architecture> <Dense|MoE> <configuration>"; "." stands for its spaces, the arms being split at spaces)
SKIP='^plamo2.Dense.4.(seq|split)$'
SKIPM='^plamo2.Dense.4.(seq|split)$|^minimax-01.MoE.4.split$'
ARMS="cpu@@1@@$SKIP mask@@1@$MASK@$SKIPM mask8s2@$EIGHT@2@$MASK@ mask8s3@$EIGHT@3@$MASK@ control@^(llama|qwen4exp|gemma3)\$@1@$MASK,LLAMA_ARCHS_SEQ_RM_BOUND_ON=2@"

for tree in base c881; do
    docker run $D --name "c881-seqrm-$tree" -e RUN="$RUN" -e B="$WORK/t/$tree/build/bin" -e O="$C/seqrm" -e TREE="$tree" \
        -e ARMS="$ARMS" "$IMAGE" bash -c 'bash "$RUN/seqrm_loop.sh" "$B" "$O" "$TREE" "$ARMS"' > "$C/seqrm/$tree.console" 2>&1 &
done
wait
cat "$C/seqrm/verdicts-base.tsv" "$C/seqrm/verdicts-c881.tsv" >> "$C/verdicts.tsv"

# the same cells, heads, changed-cell lists and logits in both builds: every arm's trace (the control's hook works
# differently in the two builds, so its traces are not compared)
for arm in cpu mask mask8s2 mask8s3; do
    a=$C/seqrm/base-$arm.log.trace b=$C/seqrm/c881-$arm.log.trace
    n=$(wc -l < "$a")
    if cmp -s "$a" "$b" && [ "$n" -gt 0 ]; then
        verdict "trace-$arm" SAME "$n lines, $(grep -c $'\tedit ' "$a") edits and $(grep -c $'\tdecode ' "$a") decodes; skip base $(cat "$C/seqrm/base-$arm.log.skip"), c881 $(cat "$C/seqrm/c881-$arm.log.skip")"
    else
        verdict "trace-$arm" DIFFERENT "base $n lines, c881 $(wc -l < "$b") lines; first difference: $(cmp "$a" "$b" 2>&1 | head -n 1)"
    fi
done

docker run $D --name c881-tests -e W="$WORK" -e C="$C" "$IMAGE" bash -c '
    for tree in base c881; do
        B=$W/t/$tree/build/bin
        timeout 600 "$B/test-kv-cache" > "$C/kvcache-$tree-fork.log" 2>&1; echo "rc=$?" > "$C/kvcache-$tree-fork.rc"
        LLAMA_KV_HEAD_TAIL=0 timeout 600 "$B/test-kv-cache" > "$C/kvcache-$tree-tail0.log" 2>&1; echo "rc=$?" > "$C/kvcache-$tree-tail0.rc"
        s=$(date +%s)
        timeout 3600 nice -n 19 "$B/test-llama-archs" -s 1 > "$C/archs-$tree.log" 2>&1
        echo "rc=$? s=$(( $(date +%s) - s ))" > "$C/archs-$tree.rc"
    done'
for tree in base c881; do
    for r in fork tail0; do
        verdict "kvcache-$tree-$r" "$(cat "$C/kvcache-$tree-$r.rc")" "$(grep -a -m 1 'GGML_ASSERT' "$C/kvcache-$tree-$r.log" | sed 's/^.*tests\///')"
    done
    verdict "archs-$tree" "$(cat "$C/archs-$tree.rc")" "$(grep -a -c ' OK ' "$C/archs-$tree.log") OK lines, $(grep -a -c 'FAIL' "$C/archs-$tree.log") FAIL lines"
done
if cmp -s <(tail -n +2 "$C/archs-base.log") <(tail -n +2 "$C/archs-c881.log"); then
    verdict archs-compare SAME "every line but the first (a timestamp)"
else
    verdict archs-compare DIFFERENT "$(diff <(tail -n +2 "$C/archs-base.log") <(tail -n +2 "$C/archs-c881.log") | head -n 4 | tr '\n' ' ')"
fi
date -Is > "$C/DONE"
