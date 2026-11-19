#!/usr/bin/env bash
# R209 (Q1603), one Ornith run on the second rig: the rig's state and the offset kit's read-only state, then R204's
# harness call (docs/data/2026-11-18_r204_ornith_glu_agent/run/ornith_agent_glu.sh: Ornith's v5 preset,
# release/ornith-v5/settings.env, on the v5 engine), the arm and the machine-specific values from the environment.
# Run from the repository root on the machine that drives the rig:
#   RIG, CLIENT, MODEL, BIN, RIG_LOGDIR, CLIENT_DIR, CLIENT_OUT   as in ornith_agent_glu.sh
#   RELAY_PORT, PROXY_PORT   the client's ports (defaults 8286 and 18783; another check on the same client needs others)
#   KIT          the offset kit's folder on the rig, with its env file ENV_FILE (default ./env-use.sh): only its
#                read-only state is read
#   OUT          a local folder: start_state.txt and the run's agent-<ARM>/
#   ARM          o5oc0 (every offset 0) or o5oc1 (the full profile's offsets, after the kit's apply)
# STEPS="report collect" with OUT=<out>/agent writes the report afterwards (no state is read then).
set -u
: "${RIG:?}" "${BIN:?}" "${KIT:?}" "${OUT:?}" "${ARM:?}"
mkdir -p "$OUT"
if [ "${STEPS:-run}" = run ]; then
  { date -Is
    timeout 120 ssh "$RIG" "uptime -s; pgrep -a 'llama|python' || echo 'no llama/python process'
      nvidia-smi --query-gpu=index,memory.used,power.limit,clocks.max.sm,clocks.max.mem --format=csv,noheader
      grep MemAvailable /proc/meminfo; md5sum $BIN
      cd $KIT && . ${ENV_FILE:-./env-use.sh} && python3 climb.py state"
  } > "$OUT/start_state.txt" 2>&1
  OUT="$OUT/agent-$ARM" STEPS=run bash docs/data/2026-11-18_r204_ornith_glu_agent/run/ornith_agent_glu.sh
else
  OUT="$OUT/agent" STEPS="$STEPS" bash docs/data/2026-11-18_r204_ornith_glu_agent/run/ornith_agent_glu.sh
fi
