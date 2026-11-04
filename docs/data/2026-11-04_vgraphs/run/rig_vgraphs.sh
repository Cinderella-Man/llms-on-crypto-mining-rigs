#!/usr/bin/env bash
# rig_vgraphs.sh OUT: C151 (Q701) on the rig. S18's DFlash2 line (Q561: Swift-1.5 Q2_0 with Inco AI's DFlash2 drafter,
# two drafted tokens, -b 512) loads once per arm, with GGML_CUDA_GRAPH_DEBUG=1 and the server's debug log (-lv 5), and
# answers the same requests; the logs then give, per request, how often the CUDA graphs were captured, re-captured and
# re-instantiated, and how often the verify batch changed width. Each card's memory and the host's are recorded once
# loaded and after each request. Arms (ARMS, in this order by default):
#   ladder2   -ladder2's build (BIN_ladder2): one graph arena for every verify width, as before the port
#   vg        the per-size graph build (BIN_vg) at its defaults: one arena per width up to 8 tokens, one scheduler,
#             calloc'd graph metadata
#   vg-gps0   BIN_vg with LLAMA_GRAPH_PER_SIZE=0: -ladder2's arenas on the new build (what #31274 alone changes)
#   vg-sps1   BIN_vg with LLAMA_SCHED_PER_SIZE=1: a scheduler per width (upstream's ~400 MiB more VRAM on GB300)
# DEBUG=0 runs the same arms without the graph debug and with the default log (for memory and speed without the log).
# Before each load the rig must be quiet: no llama-server and every card at QUIET_MIB (default 8) or less.
# Runs only ssh here, like smoke_dflash.sh (Q561), whose guard, server script and requests it uses. Environment:
#   RIG, BENCH (bench/agent_check), MODEL, DFLASH (the drafter), RIG_PORT, LOGDIR (a folder on the rig), GUARD
#   (mem_guard.sh here), BIN_ladder2, BIN_vg (llama-server on the rig), optional ARMS, REQUESTS (chat completion
#   bodies here; default Q561's two smoke requests), CTX (65536), MIN_AVAIL_MB (150, the agent check's), DEBUG (1),
#   ARGS (S18's line, below), QUIET_MIB (8)
# Output: OUT/<arm>/ (server log, VRAM and host per stage, replies, marks.tsv with the server log's line count at each
# stage, the monitors), OUT/run.log. Then: graph_counts.py OUT (where the files are) for the tables.
set -uo pipefail
OUT=$(realpath -m "${1:?usage: rig_vgraphs.sh OUT}")
: "${RIG:?}" "${BENCH:?}" "${MODEL:?}" "${DFLASH:?}" "${RIG_PORT:?}" "${LOGDIR:?}" "${GUARD:?}" "${BIN_ladder2:?}" "${BIN_vg:?}"
HERE=$(dirname "$(realpath "$0")")
S18=$HERE/../../2026-11-03_flash_next_dflash2/run
ARMS=${ARMS:-ladder2 vg vg-gps0 vg-sps1}
REQUESTS=${REQUESTS:-$S18/smoke_request.json $S18/smoke_greedy.json}
CTX=${CTX:-65536}
DEBUG=${DEBUG:-1}
QUIET_MIB=${QUIET_MIB:-8}
COMMON="-ngl 99 -fa on -c $CTX -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek \
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui"
# S18's timed line (dflash2b): two drafted tokens, -b 512, the drafter on card 8 with the output head
ARGS=${ARGS:-"-b 512 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio --no-host -md $DFLASH --spec-type draft-dflash \
--spec-draft-n-max 2 --spec-draft-n-ubatch 64 -devd CUDA8 -ngld 99"}
mkdir -p "$OUT"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/run.log"; }

arm_config() { # sets BIN and ENVS for an arm
    case $1 in
        ladder2) BIN=$BIN_ladder2; ENVS="";;
        vg)      BIN=$BIN_vg; ENVS="";;
        vg-gps0) BIN=$BIN_vg; ENVS="LLAMA_GRAPH_PER_SIZE=0";;
        vg-sps1) BIN=$BIN_vg; ENVS="LLAMA_SCHED_PER_SIZE=1";;
        *) echo "unknown arm $1"; return 1;;
    esac
    if [ "$DEBUG" = 1 ]; then ENVS="GGML_CUDA_GRAPH_DEBUG=1${ENVS:+ $ENVS}"; fi
}

run_arm() { # <arm>
    local a=$1 o=$OUT/$1 d=$LOGDIR/$1 up n log
    arm_config "$a" || return 1
    mkdir -p "$o"
    local sargs="$COMMON $ARGS"
    [ "$DEBUG" = 1 ] && sargs="$sargs -lv 5"
    log=$d/server-$a.log
    # quiet: no server, every card idle
    if ! ssh "$RIG" "! ps -eo comm | grep -qx llama-server && nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            | awk -v q=$QUIET_MIB '\$1 > q { bad = 1 } END { exit bad }'"; then
        say "$a: the rig is not quiet (a llama-server runs or a card holds memory); stopping"; return 1
    fi
    ssh "$RIG" "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader" > "$o/vram_quiet.csv"
    ssh "$RIG" "mkdir -p $d && cat > $d/mem_guard.sh" < "$GUARD"
    ssh "$RIG" "LOG=$d/mem_guard.log setsid nohup chrt -f 10 bash $d/mem_guard.sh > /dev/null 2>&1 < /dev/null & sleep 1; \
        cat $d/mem_guard.log.pid" > "$o/guard.pid" && say "$a: guard up ($(cat "$o/guard.pid"))"
    say "$a: starting ${ENVS:+$ENVS }$BIN $sargs -m $MODEL"
    ssh "$RIG" "LOGDIR=$d MIN_AVAIL_MB=${MIN_AVAIL_MB:-150} PORT=$RIG_PORT ACTION=start ARM=$a BIN=$BIN MODEL=$MODEL \
        SERVER_ENV='$ENVS' SERVER_ARGS='$sargs' bash -s" < "$BENCH/rig_server.sh" 2>&1 | tee -a "$OUT/run.log"
    up=${PIPESTATUS[0]}
    printf 'stage\tlog_lines\tepoch\n' > "$o/marks.tsv"
    mark() { ssh "$RIG" "printf '%s\t%s\t%s\n' $1 \$(wc -l < $log) \$(date +%s)" >> "$o/marks.tsv"; }
    stage() { # <name>: VRAM per card, the host and the server's memory
        ssh "$RIG" "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader" > "$o/vram_$1.csv"
        ssh "$RIG" "cat /proc/meminfo; echo '--- status'; cat /proc/\$(cat $d/server.pid)/status; echo '--- smaps_rollup'; \
            cat /proc/\$(cat $d/server.pid)/smaps_rollup" > "$o/host_$1.txt"
        mark "$1"
    }
    if [ "$up" = 0 ]; then
        stage loaded
        n=0
        for r in $REQUESTS; do
            n=$((n + 1))
            say "$a: request $n: $(basename "$r")"
            ssh "$RIG" "curl -s -m 900 127.0.0.1:$RIG_PORT/v1/chat/completions -H 'Content-Type: application/json' -d @-" \
                < "$r" > "$o/reply_$n.json"
            say "$a: request $n done ($(wc -c < "$o/reply_$n.json") bytes)"
            stage "request_$n"
        done
        ssh "$RIG" "curl -s 127.0.0.1:$RIG_PORT/metrics" > "$o/metrics.txt"
    else
        say "$a: the server did not come up"
    fi
    ssh "$RIG" "LOGDIR=$d PORT=$RIG_PORT ACTION=end bash -s" < "$BENCH/rig_server.sh" 2>&1 | tee -a "$OUT/run.log"
    ssh "$RIG" "kill \$(cat $d/mem_guard.log.pid) 2> /dev/null; rm -f $d/mem_guard.sh"
    # the logs back; the debug log is large, so it travels compressed
    ssh "$RIG" "tar -C $d -czf - ." | tar -C "$o" -xzf - && ssh "$RIG" "rm -rf $d" && say "$a: rig logs in $o, removed on the rig"
    [ "$up" = 0 ]
}

for a in $ARMS; do
    run_arm "$a" || say "$a: FAILED"
done
say "done: $ARMS"
