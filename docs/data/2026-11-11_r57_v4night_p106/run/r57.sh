#!/usr/bin/env bash
# R57 (Q1224) on one rig: a new Miner Llama build (NEW) against the v3.0 release build (REF) on Flash-Next's v3 preset
# (release/flash-next-v3/settings.env), every switch the preset does not set at its default. Run from the repository
# root on a machine that only runs ssh. The steps run in this order, whatever STEPS lists:
#   load    one llama-server per build (NEW, then REF) with the preset, untimed: memory per card (nvidia-smi) 20 s after it
#           answers, the host's MemAvailable, and the load log's buffer lines, among them the MTP draft's hidden rows
#           (upstream #31927: "h_nextn buffer size", or its warning that the rows go through host memory)
#   loadlv  one more load of NEW with -lv 4: llama-server's default verbosity (3) shows the library's warnings but not its
#           INFO lines (the buffers per card, the h_nextn buffer), which print from verbosity 4
#   greedy  greedy identity with docs/data/2026-11-05_ladder5/run/rig_greedy.sh: every default, Flash-Next's no-draft
#           split, REF then NEW, one call per prompt (code: the script's default; prose: a hash-map paragraph)
#   prep    the agent check's bench files and sandbox image on the client, the bugfix tasks validated
#   agent   the agent check of the preset (bench/agent_check, 12 bugfix tasks, a 12-minute cap a task, a 40-minute budget
#           an arm), one arm per build: ARM_NEW on NEW, ARM_REF on REF, in ARM_ORDER; the arms take exactly the server
#           line and switches of v3.0's arm t1swift (docs/data/2026-11-05_release_flashnext/run/rig_t1.sh)
#   report  the agent check's report of both arms; the client's folder collected to OUT/agent
# Env: RIG, CLIENT (ssh aliases), MODEL (Swift Q2_0's first shard on the rig), DRAFT (the MTP draft on the rig), NEW, REF
#   (build folders under /opt/llama-builds), RIGDIR (a folder on the rig for logs, on a drive with room that does not
#   hold the n-gram table), OUT (a local folder), STEPS ("load greedy prep agent report"), ARM_NEW (v4night), ARM_REF
#   (v3rel), ARM_ORDER ("$ARM_NEW $ARM_REF"), and release_check_s12.sh's RIG_PORT RELAY_PORT CLIENT_PORT PROXY_PORT
#   CLIENT_DIR CLIENT_OUT CPU_SHARES (256).
set -uo pipefail
: "${RIG:?}" "${CLIENT:?}" "${MODEL:?}" "${DRAFT:?}" "${NEW:?}" "${REF:?}" "${RIGDIR:?}" "${OUT:?}"
STEPS=${STEPS:-load greedy prep agent report}
ARM_NEW=${ARM_NEW:-v4night}; ARM_REF=${ARM_REF:-v3rel}; ARM_ORDER=${ARM_ORDER:-$ARM_NEW $ARM_REF}
has() { [[ " $STEPS " == *" $1 "* ]]; }
say() { echo "$(date +%H:%M:%S) $*"; }
# shellcheck disable=SC1091
. release/flash-next-v3/settings.env  # PRESET_ENV, PRESET_ARGS
# rig_t1.sh's MT: the preset's server line after its common part (release_check_s12.sh's COMMON adds that)
MT="-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host"
export BENCH=$PWD/bench/agent_check RIG CLIENT MODEL DRAFT CTX=65536 TIME_CAP_MIN=12 MIN_AVAIL_MB=60 \
    CPU_SHARES=${CPU_SHARES:-256} RIG_LOGDIR=$RIGDIR/agent-logs
RC=docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh
mkdir -p "$OUT"

load() {  # <label> <build folder> [more server arguments]
    local d=$RIGDIR/load-$1 o=$OUT/load-$1 up=0
    mkdir -p "$o"
    say "load $1: $2 ${3:-}"
    ssh "$RIG" "LOGDIR=$d MIN_AVAIL_MB=$MIN_AVAIL_MB ACTION=start ARM=$1 BIN=/opt/llama-builds/$2/bin/llama-server MODEL=$MODEL \
        PORT=$RIG_PORT SERVER_ENV='$PRESET_ENV' SERVER_ARGS='$PRESET_ARGS -md $DRAFT --metrics --no-webui ${3:-}' bash -s" \
        < bench/agent_check/rig_server.sh > "$o/console.txt" 2>&1 && up=1
    if [ $up = 1 ]; then
        sleep 20
        ssh "$RIG" "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits" > "$o/mem.csv"
        ssh "$RIG" "grep -E '^(MemTotal|MemAvailable|CmaFree|SwapTotal|SwapFree):' /proc/meminfo" > "$o/meminfo.txt"
    fi
    ssh "$RIG" "LOGDIR=$d ACTION=end PORT=$RIG_PORT bash -s" < bench/agent_check/rig_server.sh >> "$o/console.txt" 2>&1
    ssh "$RIG" "tar -C $d -cf - ." | tar -C "$o" -xf - && ssh "$RIG" "rm -rf $d"
    grep -aE 'h_nextn|hidden rows|buffer size|KV self size|projected|failed|error' "$o/server-$1.log" > "$o/buffers.txt"
    say "load $1: $(tail -n 1 "$o/console.txt" | head -c 200); $(grep -ac h_nextn "$o/buffers.txt") h_nextn lines"
    [ $up = 1 ]
}

if has load; then
    load "$ARM_NEW" "$NEW" || { say "the new build did not load; stopping"; exit 1; }
    load "$ARM_REF" "$REF" || { say "the reference build did not load; stopping"; exit 1; }
fi
if has loadlv; then
    load "${ARM_NEW}lv" "$NEW" "-lv 4" || { say "the new build did not load at -lv 4"; exit 1; }
fi

if has greedy; then
    for p in code prose; do
        case $p in
            code)  PROMPT='Write a Python function that parses an ISO 8601 date string into a datetime object, then three pytest tests for it.' ;;
            prose) PROMPT='Explain in one paragraph how a hash map handles collisions, then give one example in Python.' ;;
        esac
        say "greedy $p"
        RIG=$RIG MODEL=$MODEL TAG=flashnext-$p PROMPT=$PROMPT ARGS='-ngl 99 -fa on -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio' \
            DIR=$RIGDIR/greedy OUT=$OUT/greedy BUILDS="$REF $NEW" bash docs/data/2026-11-05_ladder5/run/rig_greedy.sh \
            > "$OUT/greedy-$p.console" 2>&1
        say "$(tail -n 1 "$OUT/greedy-$p.console")"
    done
    ssh "$RIG" "rmdir $RIGDIR/greedy"
fi

if has prep; then
    say "prep"
    ARMS=$ARM_NEW STEPS=prep bash $RC "$OUT/agent-prep" > "$OUT/agent-prep.console" 2>&1 || { say "prep failed"; exit 1; }
fi

if has agent; then
    for a in $ARM_ORDER; do
        case $a in "$ARM_NEW") b=$NEW;; "$ARM_REF") b=$REF;; *) say "unknown arm $a"; exit 1;; esac
        say "arm $a: $b"
        env REL_BIN="/opt/llama-builds/$b/bin/llama-server" "ARGS_$a=$MT" "BUDGET_MIN_$a=40" REL_ENV="$PRESET_ENV" \
            ARMS="$a" STEPS=run bash $RC "$OUT/agent-$a" > "$OUT/agent-$a.console" 2>&1
        say "arm $a: exit $?"
    done
fi

if has report; then
    say "report"
    ARMS="$ARM_REF $ARM_NEW" STEPS="report collect" bash $RC "$OUT/agent" > "$OUT/agent-report.console" 2>&1
    say "report: exit $?"
fi
say "R57_DONE"
