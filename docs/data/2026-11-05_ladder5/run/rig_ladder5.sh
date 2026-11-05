#!/usr/bin/env bash
# C537 (Q1026) on one rig: -ladder5 against -ladder4 with Flash-Next's real files (Swift pmix-q2-fast, S35's line:
# docs/data/2026-11-04_s35_stack). Run from the repository root on a machine that only runs ssh; every rig command runs
# under the RAM watchdog (scripts/ram_watchdog.sh).
#   identity  llama-perplexity on TEXT (CHUNKS chunks of CTX tokens, one micro-batch each) with every build of BUILDS,
#             two settings each: every switch at its default (no variable set), and S35's no-draft line (the k-column
#             settings, the Q8_0 hop cast through the staged hop); each run writes its next-token distributions
#             (--kl-divergence-base) and the files of a setting must be the same bytes for every build
#   tk        llama-bench on S35's no-draft line, -p 1,2,3,4,5 -n 128, five repetitions: T(k) for k-token passes
#             (k = 1-5) and plain decoding, the builds alternated over LOADS loads each (default 2)
# Env: RIG (ssh alias), MODEL (the first shard on the rig), BUILDS ('<folder under /opt/llama-builds> ...': the first is
#   the base), TEXT (a text file on the rig), DIR (a folder on the rig's model disk: the distributions are ~0.5 MB a
#   token), OUT (a local folder), STEPS ('identity tk'), CTX (256), CHUNKS (8), LOADS (2)
# Output: OUT/identity.tsv (setting, build, md5 of the distributions, perplexity line), OUT/tk-<build>-<i>.jsonl,
#   OUT/*.err, OUT/rw_*.tsv (the watchdog's host-memory log)
set -uo pipefail
: "${RIG:?}" "${MODEL:?}" "${BUILDS:?}" "${TEXT:?}" "${DIR:?}" "${OUT:?}"
STEPS=${STEPS:-identity tk}
CTX=${CTX:-256}
CHUNKS=${CHUNKS:-8}
LOADS=${LOADS:-2}
has() { [[ " $STEPS " == *" $1 "* ]]; }
KENV="GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0"
HOP="LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1"
# S35's layer split; llama-bench separates the shares with "/", the other tools with ","
SPLIT="-ngl 99 -fa 1 -ts 6/5/5/6/5/5/6/5/6 -lm dio"
PSPLIT="-ngl 99 -fa on -ts 6,5,5,6,5,5,6,5,6 -lm dio"
mkdir -p "$OUT"
# <name> <tool> <env> <arguments>: one run on the rig under the RAM watchdog, its stdout to OUT/<name>.out
run() {
    ssh "$RIG" "mkdir -p $DIR && cat > /dev/shm/ram_watchdog.sh" < scripts/ram_watchdog.sh
    ssh "$RIG" "LOG=/dev/shm/rw_$1.tsv PROC=$2 MIN_AVAIL_MB=60 setsid nohup chrt -f 10 bash /dev/shm/ram_watchdog.sh > /dev/null 2>&1 < /dev/null & echo \$! > /dev/shm/rw.pid
        cd $DIR && env $3 timeout 3600 $4 > $1.out 2> $1.err; echo \"$1 rc=\$?\"
        kill \$(cat /dev/shm/rw.pid); mv /dev/shm/rw_$1.tsv $DIR/; rm -f /dev/shm/rw.pid /dev/shm/ram_watchdog.sh"
    for f in "$1.out" "$1.err" "rw_$1.tsv"; do ssh "$RIG" "cat $DIR/$f" > "$OUT/$f"; done
}
if has identity; then
    printf 'setting\tbuild\tmd5\tperplexity\n' > "$OUT/identity.tsv"
    for s in defaults s35; do
        e=""; [ $s = s35 ] && e="$KENV $HOP"
        for b in $BUILDS; do
            n=id-$s-${b##*-pascal-p106-}
            run "$n" llama-perplexity "$e" "/opt/llama-builds/$b/bin/llama-perplexity -m $MODEL $PSPLIT -f $TEXT -c $CTX -b $CTX \
                -ub $CTX --chunks $CHUNKS --kl-divergence-base $DIR/$n.kld"
            m=$(ssh "$RIG" "md5sum < $DIR/$n.kld | cut -d' ' -f1; rm -f $DIR/$n.kld")
            printf '%s\t%s\t%s\t%s\n' "$s" "$b" "$m" "$(grep -a 'Final estimate' "$OUT/$n.err" "$OUT/$n.out" | tail -n 1 | cut -d: -f2-)" >> "$OUT/identity.tsv"
        done
        nm=$(awk -F'\t' -v s=$s '$1 == s {print $3}' "$OUT/identity.tsv" | sort -u | wc -l)
        echo "identity $s: $([ "$nm" = 1 ] && echo SAME || echo DIFFERS) ($nm distinct md5 over $(echo $BUILDS | wc -w) builds)"
    done
fi
if has tk; then
    for i in $(seq "$LOADS"); do
        for b in $BUILDS; do
            run "tk-${b##*-pascal-p106-}-$i" llama-bench "$KENV $HOP" "/opt/llama-builds/$b/bin/llama-bench -m $MODEL $SPLIT \
                -b 2048 -ub 512 -p 1,2,3,4,5 -n 128 -r 5 -o jsonl"
        done
    done
fi
ssh "$RIG" "rm -f $DIR/*.out $DIR/*.err $DIR/rw_*.tsv"
