#!/usr/bin/env bash
# R203 (Q1592), one Flash-Next half (fnoc0 at stock, fnoc1 with the use profile's offsets): the rig's state and the
# offset kit's read-only state, then C854's run/flash_agent.sh (R156's driver, arm v4r = the released v4.0 preset,
# under timeout 5400), then the state again with the kit's start check. It sets no offset, clock or power limit: the
# offset half is run after the kit's apply. From the repository root on the machine that drives the rig.
# Machine-specific values come from the environment:
#   RIG, CLIENT, MODEL, DRAFT, NEW, REF, RIGDIR, CLIENT_DIR, CLIENT_OUT   as in C854's run/flash_agent.sh
#   OUT          a local folder: start_state.txt, end_state.txt, done, and the driver's out/
#   KIT          the offset kit's folder on the rig (climb.py, env-use.sh)
#   RELAY_PORT, PROXY_PORT   the client's ports (8286 and 18783 by default)
: "${RIG:?}" "${KIT:?}" "${OUT:?}"
mkdir -p "$OUT"
state() {
  timeout 120 ssh "$RIG" "ps -eo comm | grep -x -E 'llama-server|llama-bench|llama-cli' || echo 'no llama process'
    nvidia-smi --query-gpu=index,memory.used,power.limit,clocks.max.sm,clocks.max.mem --format=csv,noheader
    grep MemAvailable /proc/meminfo
    cd $KIT && . ./env-use.sh && python3 climb.py state $1"
}
{ date -Is; state; } > "$OUT/start_state.txt" 2>&1
OUT=$OUT/out bash docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh
echo "exit $?" > "$OUT/done"
{ date -Is; state "&& python3 climb.py start-check"; } > "$OUT/end_state.txt" 2>&1
