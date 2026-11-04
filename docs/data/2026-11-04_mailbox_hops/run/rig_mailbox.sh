#!/usr/bin/env bash
# rig_mailbox.sh: C143r, the mailbox hop on a rig, ON the rig and detached, from a folder that holds the bundle (the hop
# microbenchmark megakernel_p106 built from bench/megakernel_p106, this script, fn_line.sh and fn_requests.py from
# docs/data/2026-11-03_flash_next_baseline/run/, and optionally scripts/ram_watchdog.sh). Ends with OUT/DONE.
#   1. the hop between two cards, for each pair in HOPS (default "0,1 3,4 4,5 3,5": two cards behind one switch, a
#      switch to the card on its own root port and back, two switches): megakernel_p106 --hop A,B --hop-only at
#      HOP_SIZES (one token's residual in Q8_0, F16, F32: 10,880 / 20,480 / 40,960 B; a 4-token pass: 43,520 / 81,920 /
#      163,840 B; 1 MiB), as host-staged copies, the driver's peer copy, the one-shot mailbox, the fork's manual
#      staged hop in 1, 2 and 4 chunks and the fork's streamed mailbox in 1, 4, 8 and 16 chunks
#   2. unless BENCH=0: llama-bench per arm in BENCH_ARMS, one load each, REPS times: decode (-n 128; the line of the
#      stack run S35, which gave 14.98 tok/s with f32 hops and 17.06 with q8staged on the first rig) and passes of
#      PASS_K tokens (-p PASS_K): the time of a k-token verify pass is k / its tok/s
#   3. unless LINES=0: one server line per arm in ARMS (fn_line.sh, MODE=speed: decode at short context, fresh prompts
#      of ~1k, ~8k and ~32k tokens, two identity prompts at temperature 0 whose output hash compares the arms)
#   Arms (the switches of the build BUILD; the mailbox and staged arms copy the same bytes as their base, so their
#   identity hashes must agree with it):
#        f32       no switch (F32 residual, cudaMemcpyPeerAsync)
#        staged    GGML_CUDA_HOP_STAGED=1
#        mbox      GGML_CUDA_HOP_MAILBOX=1
#        q8staged  LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1           (the release line's hops since S35)
#        q8mbox    LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_MAILBOX=1          (every hop by mailbox; prompts on the copy stream)
#        q8mboxst  LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_MAILBOX=1 GGML_CUDA_HOP_MAILBOX_MAX_KB=256 GGML_CUDA_HOP_STAGED=1
#                  (decode and verify hops by mailbox, prompt micro-batches through the staged hop)
#        q8mbox16  q8mbox in 16 chunks a hop (GGML_CUDA_HOP_MAILBOX_CHUNKS=16)
#        q8mbox2   q8mbox in 2 chunks a hop
# Env: OUT (default ./out), BIN (default ./megakernel_p106), HOPS, HOP_SIZES, BENCH (1), BENCH_ARMS (default "f32
#      q8staged q8mbox q8mboxst mbox"), REPS (3), PASS_K (1,2,4), BENCH_ARGS (default '-ngl 99 -fa 1 -b 2048 -ub 512
#      -ts 6/5/5/6/5/5/6/5/6 -lm dio'), LINES (1), ARMS (default "q8staged q8mbox q8mboxst"), and for steps 2-3 BUILDS
#      (folder of the builds), BUILD (the mailbox build's folder name), MODEL (first shard), CTX (65536), ARGS (default
#      '-b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off'), PORT, SSD_DEV, USB_DEV; ENV_ALL (default none) goes to every
#      line before the arm's switches (e.g. the kernel settings the comparison runs used); WATCHDOG (a ram_watchdog.sh,
#      started under chrt -f 10 around steps 2 and 3 with PROC llama-bench / llama-server and MIN_AVAIL_MB
#      WATCHDOG_MIN_MB, default 60); FORCE=1 starts even with a compute process on the cards
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$here/out}
BIN=${BIN:-$here/megakernel_p106}
HOPS=${HOPS-0,1 3,4 4,5 3,5}
HOP_SIZES=${HOP_SIZES:-10880,20480,40960,43520,81920,163840,1048576}
BENCH=${BENCH:-1}
BENCH_ARMS=${BENCH_ARMS:-f32 q8staged q8mbox q8mboxst mbox}
REPS=${REPS:-3}
PASS_K=${PASS_K:-1,2,4}
BENCH_ARGS=${BENCH_ARGS:--ngl 99 -fa 1 -b 2048 -ub 512 -ts 6/5/5/6/5/5/6/5/6 -lm dio}
LINES=${LINES:-1}
ARMS=${ARMS:-q8staged q8mbox q8mboxst}
CTX=${CTX:-65536}
ARGS=${ARGS:--b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off}
FORCE=${FORCE:-0}
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
log() { echo "$(date -Is) $*" | tee -a "$OUT/run.log"; }

MB="GGML_CUDA_HOP_MAILBOX=1"
declare -A ENV_OF=(
    [f32]="" [staged]="GGML_CUDA_HOP_STAGED=1" [mbox]="$MB"
    [q8staged]="LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1" [q8mbox]="LLAMA_HOP_CAST=q8_0 $MB"
    [q8mboxst]="LLAMA_HOP_CAST=q8_0 $MB GGML_CUDA_HOP_MAILBOX_MAX_KB=256 GGML_CUDA_HOP_STAGED=1"
    [q8mbox16]="LLAMA_HOP_CAST=q8_0 $MB GGML_CUDA_HOP_MAILBOX_CHUNKS=16"
    [q8mbox2]="LLAMA_HOP_CAST=q8_0 $MB GGML_CUDA_HOP_MAILBOX_CHUNKS=2"
)

wd_pid=""
watchdog_start() { # <process name>
    [ -n "${WATCHDOG:-}" ] || return 0
    LOG=/dev/shm/rw_mailbox_$1.tsv PROC=$1 MIN_AVAIL_MB=${WATCHDOG_MIN_MB:-60} chrt -f 10 bash "$WATCHDOG" \
        > /dev/null 2>&1 < /dev/null &
    wd_pid=$!
}
watchdog_stop() { # <process name>
    [ -n "$wd_pid" ] || return 0
    kill "$wd_pid" 2> /dev/null; sleep 1; wd_pid=""
    mv /dev/shm/rw_mailbox_$1.tsv "$OUT/" 2> /dev/null
    grep -h KILL "$OUT/rw_mailbox_$1.tsv" 2> /dev/null | while read -r l; do log "watchdog: $l"; done
}

apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
if [ -n "$apps" ] && [ "$FORCE" != 1 ]; then
    log "compute processes on the cards (FORCE=1 to go on): $apps"
    touch "$OUT/DONE"
    exit 2
fi
{ uptime; free -m; nvidia-smi --query-gpu=index,name,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu --format=csv; } \
    > "$OUT/quiet_check.txt" 2>&1
md5sum "$BIN" "$here"/fn_line.sh "$here"/fn_requests.py > "$OUT/md5.txt" 2>&1

log "1. hops between cards: ${HOPS:-none}"
for p in $HOPS; do
    tag=${p/,/-}
    "$BIN" --hop "$p" --hop-only --hop-sizes "$HOP_SIZES" --csv "$OUT/hop-$tag.csv" > "$OUT/hop-$tag.txt" 2>&1
    log "hop $p: exit $?; $(grep -E 'hop data|mailbox data|timeouts' "$OUT/hop-$tag.txt" | tr '\n' ' ')"
done

if [ "$BENCH" = 1 ] || [ "$LINES" = 1 ]; then
    : "${BUILDS:?BUILDS=<folder of the builds>}" "${BUILD:?BUILD=<the mailbox build>}" "${MODEL:?MODEL=<first shard>}"
fi

if [ "$BENCH" = 1 ]; then
    watchdog_start llama-bench
    for arm in $BENCH_ARMS; do
        [ -n "${ENV_OF[$arm]+x}" ] || { log "unknown arm $arm"; continue; }
        e="${ENV_ALL:+$ENV_ALL }${ENV_OF[$arm]}"
        log "2. llama-bench $arm: ${e:-no switch}"
        # one load: the passes of PASS_K tokens (pp tests) and the decode (tg128)
        # shellcheck disable=SC2086
        env $e timeout 2400 "$BUILDS/$BUILD/bin/llama-bench" -m "$MODEL" $BENCH_ARGS -p "$PASS_K" -n 128 -r "$REPS" \
            -o json > "$OUT/bench-$arm.json" 2> "$OUT/bench-$arm.err"
        rc=$?
        grep -E 'card hops in|by mailbox|staged through|staged hops,' "$OUT/bench-$arm.err" > "$OUT/hops-bench-$arm.txt" 2> /dev/null
        log "llama-bench $arm: exit $rc; $(grep -oE '"n_(prompt|gen)": [0-9]+|"avg_ts": [0-9.]+' "$OUT/bench-$arm.json" | tr '\n' ' ')"
    done
    watchdog_stop llama-bench
fi

if [ "$LINES" = 1 ]; then
    watchdog_start llama-server
    for arm in $ARMS; do
        [ -n "${ENV_OF[$arm]+x}" ] || { log "unknown arm $arm"; continue; }
        log "3. line $arm: ${ENV_OF[$arm]:-no switch}"
        LABEL=$arm BUILD=$BUILD BUILDS=$BUILDS MODEL=$MODEL CTX=$CTX ARGS="$ARGS" MODE=speed OUT=$OUT \
            ENVS="${ENV_ALL:+$ENV_ALL }${ENV_OF[$arm]}" bash "$here/fn_line.sh" > "$OUT/line-$arm.out" 2>&1
        # the switches' own log lines: the boundaries the cast found, each boundary's ring or mailbox and its counts
        grep -E 'card hops in|by mailbox|staged through|staged hops,' "$OUT/server-$arm.log" > "$OUT/hops-$arm.txt" 2>/dev/null
        log "line $arm: $(grep -E '^(decode|load|ident)' "$OUT/line-$arm.out" | cut -c1-120 | tr '\n' ' ')"
    done
    watchdog_stop llama-server
fi
log "done"
touch "$OUT/DONE"
