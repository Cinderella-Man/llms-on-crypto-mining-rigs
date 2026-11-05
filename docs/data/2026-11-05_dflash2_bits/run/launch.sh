#!/usr/bin/env bash
# launch.sh: C528 (Q1017) from the machine that drives the rig (it runs only ssh, tar and cat here). Puts the scripts and
# the requests on the rig, then starts rig_bits.sh there, detached, and returns. Run from the repository root:
#   RIG=rig W=<rig folder> BIN=<-ladder3 build>/bin/llama-server MODEL=<Swift Q2_0 first shard> DRAFT_DIR=<the drafters>
#     TEXT=<the agent text on the rig> bash docs/data/2026-11-05_dflash2_bits/run/launch.sh
# Optional: ARMS (default "q8a=Q8_0 q4=Q4_K_M q5=Q5_K_M q8b=Q8_0": arm=quant of
# DRAFT_DIR/Qwen3.8-Flash-Next-DFlash2-<quant>.gguf, in load order), GAP_S, MIN_AVAIL_MB, PORT (rig_bits.sh's).
# The first-turn request is built on the rig (first_turn_request.sh with BUILD_SSH="ssh $RIG"); the agent text must
# already be on the rig (its md5 is checked against TEXT_MD5, default C271's file). When it ends: W/out/DONE, the tables
# in W/out/tables/; bring W/out back with tar over ssh and remove W.
set -euo pipefail
: "${RIG:?}" "${W:?}" "${BIN:?}" "${MODEL:?}" "${DRAFT_DIR:?}" "${TEXT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
ARMS_IN=${ARMS:-q8a=Q8_0 q4=Q4_K_M q5=Q5_K_M q8b=Q8_0}
TEXT_MD5=${TEXT_MD5:-312fd205dec2fba4879d4bf421f1e80e}
arms=
for s in $ARMS_IN; do arms="$arms ${s%%=*}=$DRAFT_DIR/Qwen3.8-Flash-Next-DFlash2-${s#*=}.gguf"; done
got=$(ssh "$RIG" "md5sum < '$TEXT'" | cut -c1-32)
[ "$got" = "$TEXT_MD5" ] || { echo "agent text md5 $got, want $TEXT_MD5"; exit 1; }
ssh "$RIG" "mkdir -p $W/run $W/req"
for f in rig_bits.sh make_requests.py bits_tables.py; do ssh "$RIG" "cat > $W/run/$f" < "$HERE/$f"; done
ssh "$RIG" "cat > $W/run/rig_server.sh" < "$REPO/bench/agent_check/rig_server.sh"
ssh "$RIG" "cat > $W/run/smoke_greedy.json" < "$REPO/docs/data/2026-11-03_flash_next_dflash2/run/smoke_greedy.json"
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
BUILD_SSH="ssh $RIG" LONG_MAX_TOKENS=512 bash "$REPO/bench/agent_check/first_turn_request.sh" "$tmp"
ssh "$RIG" "cat > $W/run/first_turn.json" < "$tmp"
ssh "$RIG" "python3 -I $W/run/make_requests.py --first $W/run/first_turn.json --smoke $W/run/smoke_greedy.json \
  --text '$TEXT' --out $W/req"
# shellcheck disable=SC2029
timeout 30 ssh "$RIG" "cd $W && BIN='$BIN' MODEL='$MODEL' ARMS='${arms# }' REQ=$W/req OUT=$W/out \
  SERVER_SH=$W/run/rig_server.sh TABLES=$W/run/bits_tables.py ${GAP_S:+GAP_S=$GAP_S} ${MIN_AVAIL_MB:+MIN_AVAIL_MB=$MIN_AVAIL_MB} \
  ${PORT:+PORT=$PORT} setsid nohup bash run/rig_bits.sh > run.console 2>&1 < /dev/null & sleep 2; cat $W/out/run.pid" \
  || true  # the ssh waits on the detached job until the timeout
echo "started; wait for $W/out/DONE"
