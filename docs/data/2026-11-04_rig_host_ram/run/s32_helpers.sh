#!/usr/bin/env bash
# s32_helpers.sh: shell functions that make each step of the host-RAM run one command. Source it from the repository root,
# after setting RIG and the values of docs/data/2026-11-03_flash_next_mtp/run/s12.env.example (REL_BIN, MODEL, RIG_PORT, CTX,
# ARGS_nodraft, ARGS_mtp2, MIN_AVAIL_MB):
#
#   set -a; . s12.env; set +a
#   . docs/data/2026-11-04_rig_host_ram/run/s32_helpers.sh
#
# Everything runs on the rig through ssh "$RIG"; nothing runs here. The rig's census files stay on the rig, in
# /var/tmp/rig_host_ram (which a reboot keeps), until `pull` or `pull_all` copies them into $RESULTS.
: "${RIG:?set RIG to the rig's ssh alias}" "${REL_BIN:?set REL_BIN to llama-server's path}" "${MODEL:?}" "${RIG_PORT:?}" "${CTX:?}" "${ARGS_nodraft:?}"

D=${D:-docs/data/2026-11-04_rig_host_ram}
RHR=${RHR:-scripts/rig_host_ram.sh}
RESULTS=${RESULTS:-$D/rig}
mkdir -p "$RESULTS"
AGENT_BENCH=${AGENT_BENCH:-bench/agent_check}
# The load every arm shares: the agent check's server line without a draft (the arm that fits the 4 GB host today).
LOAD_ARGS=${LOAD_ARGS:-"-ngl 99 -fa on -c $CTX -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --no-webui $ARGS_nodraft"}
LOGDIR=${LOGDIR:-/tmp/s32}

# rhr census NAME | delta A B | apply thp | revert thp | boot-howto: the tool, read from this checkout.
rhr() { ssh "$RIG" "bash -s -- $*" < "$RHR"; }
# rhr_env 'BIN=… MODEL=…' contexts: the same with environment variables for the tool.
rhr_env() { local e=$1; shift; ssh "$RIG" "$e bash -s -- $*" < "$RHR"; }
# pull NAME…: copy census files from the rig into $RESULTS; pull_all copies every file the tool wrote; pull_contexts the Q565 tables;
# clean_rig removes what the run left on the rig.
pull() { mkdir -p "$RESULTS"; local n; for n in "$@"; do ssh "$RIG" "cat /var/tmp/rig_host_ram/$n.tsv" > "$RESULTS/$n.tsv"; done; }
pull_all() { local f; for f in $(ssh "$RIG" "cd /var/tmp/rig_host_ram && ls *.tsv"); do pull "${f%.tsv}"; done; }
pull_contexts() { mkdir -p "$RESULTS/contexts"; local f; for f in contexts slope; do ssh "$RIG" "cat /var/tmp/rig_host_ram/contexts/$f.tsv" > "$RESULTS/contexts/$f.tsv"; done; }
clean_rig() { ssh "$RIG" "systemctl stop s32-llama 2> /dev/null; rm -rf /var/tmp/rig_host_ram $LOGDIR"; }

# load NAME 'ENV=VALUE …' 'more server arguments': start the server through the agent check's own start script, wait for it.
load() {
    ssh "$RIG" "ACTION=start ARM=$1 BIN=$REL_BIN MODEL=$MODEL PORT=$RIG_PORT LOGDIR=$LOGDIR MIN_AVAIL_MB=${MIN_AVAIL_MB:-150} \
        SERVER_ENV='${2:-}' SERVER_ARGS='$LOAD_ARGS ${3:-}' bash -s" < "$AGENT_BENCH/rig_server.sh"
}
unload() { ssh "$RIG" "ACTION=end PORT=$RIG_PORT LOGDIR=$LOGDIR bash -s" < "$AGENT_BENCH/rig_server.sh"; }
# measure NAME 'ENV=VALUE …' 'more server arguments': load, census as loaded_NAME, each card's memory as vram_NAME.csv, unload (the server stops even
# if a step fails).
measure() {
    local rc
    load "$1" "${2:-}" "${3:-}" && rhr census "loaded_$1" && ssh "$RIG" "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader" > "$RESULTS/vram_$1.csv"
    rc=$?
    unload
    return $rc
}

# load_slice / unload_slice: the server in its own slice (llama.slice, under the root cgroup) with MemoryLow on the unit and on the
# slice (a cgroup's protection is capped by its parents'), so MemoryHigh on system.slice and user.slice (rhr apply cgroup)
# never throttles it. MEMLOW: a little under llama0.cg.current_kB of the baseline census.
load_slice() {
    ssh "$RIG" "mkdir -p $LOGDIR && systemd-run --quiet --collect --unit=s32-llama --slice=llama.slice -p MemoryLow=${MEMLOW:-1G} \
        -p StandardOutput=file:$LOGDIR/server-slice.log -p StandardError=inherit \
        $REL_BIN $LOAD_ARGS -m $MODEL --host 127.0.0.1 --port $RIG_PORT && \
        systemctl set-property --runtime llama.slice MemoryLow=${MEMLOW:-1G} && \
        for i in \$(seq 900); do curl -sf http://127.0.0.1:$RIG_PORT/health > /dev/null && exit 0; \
        systemctl is-active -q s32-llama || { tail -n 5 $LOGDIR/server-slice.log; exit 1; }; sleep 2; done; exit 1"
}
unload_slice() { ssh "$RIG" "systemctl stop s32-llama; rm -rf $LOGDIR"; }
measure_slice() { load_slice && rhr census "loaded_$1"; local rc=$?; unload_slice; return $rc; }
