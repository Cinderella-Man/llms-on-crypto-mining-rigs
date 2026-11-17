#!/usr/bin/env bash
# r190.sh load|run|end: R190 (Q1557) on one rig, the steps in the order they ran. R183's Flash-Next arm again, with
# two changes: the v4 preset's own file (UkisAI's Swift-1.5 GSQ-RCO Q2_0) and a question that names the six facts'
# kinds (QUESTION=kinds), so the answer needs no list of the messages. One load of llama-server on the v4 preset
# (release/flash-next-v4/settings.env, unchanged; the MTP draft as the preset names it), then recall_depth.py from
# this machine through an ssh tunnel: the control prompt (8,000 tokens), then 56,460 and 60,240 tokens, seed 1546,
# greedy. A greedy reply that hits the cap with no answer is sent once more with the preset's sampling and the seed,
# and the run stops there; an answer that misses facts stops the run at once (ON_MISS=sample).
#   load  copies serve.sh, the preset and the RAM watchdog to the rig, starts the server, waits until it answers
#   run   the prompts (the tunnel must be up: ssh -N -L $LPORT:127.0.0.1:8080 "$RIG")
#   end   stops the server (the idle check after), copies the rig's files to OUT/rig and removes them there
# Env: RIG (ssh alias), BIN (the build's bin folder on the rig), MODEL (the first shard on the rig), DRAFT (the MTP
# draft on the rig), OUT (results here), LPORT (the tunnel's local port, default 18080), TARGETS (default
# "8000 56460 60240").
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TOOL=$HERE/../../../../bench/recall_depth
: "${RIG:?}" "${OUT:?}"
LPORT=${LPORT:-18080}
TARGETS=${TARGETS:-8000 56460 60240}
export RIG BIN MODEL DRAFT
export QUESTION=kinds ON_MISS=sample SAMPLING=1.0,0.95,20,0 SEED=1546 MAX_TOKENS=8192 NAME=kinds
case ${1:?usage: r190.sh load|run|end} in
load) bash "$TOOL/remote.sh" put && bash "$TOOL/remote.sh" start && bash "$TOOL/remote.sh" wait ;;
# shellcheck disable=SC2086
run)  URL=http://127.0.0.1:$LPORT OUT=$OUT/client python3 "$TOOL/recall_depth.py" run $TARGETS ;;
end)  bash "$TOOL/remote.sh" stop; OUT=$OUT/rig bash "$TOOL/remote.sh" collect ;;
*)    echo "usage: r190.sh load|run|end"; exit 1 ;;
esac
