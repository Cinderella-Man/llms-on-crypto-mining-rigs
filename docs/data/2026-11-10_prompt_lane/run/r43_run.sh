#!/usr/bin/env bash
# r43_run.sh: one run of R43 (Q1155), the prompt lane on the v3.0 release line, ON the rig. Run it detached from a
# folder that holds this script, rig_taper.sh and taper_replay.py (docs/data/2026-11-04_prompt_taper/run/), the agent
# check's turns.csv (docs/data/2026-11-03_retype_agent/agent/turns.csv) and the preset's settings.env
# (release/flash-next-v3/settings.env). It exports the preset's environment (PRESET_ENV) and calls rig_taper.sh with
# the preset's line; rig_taper.sh adds -ngl 99 -fa on -c 65536 -np 1 -bs --cache-ram 0 --jinja --reasoning-format
# deepseek and sets --ctx-checkpoints per load (A = 4, B = 0). The taper is set per request, off and on, in every load.
# Env: BUILDS (folder of the builds), BUILD (the release build's folder), MODEL (Swift Q2_0's first shard), DRAFT (the
#      MTP draft), OUT, LOADS ("A B"), ARGS_ADD (appended to the line, e.g. "--ctx-checkpoints-vram 4"),
#      MIN_AVAIL_MB (60: the preset's first long request leaves the 4 GB host at 70 to 96 MiB), and any switch the
#      caller exports (e.g. LLAMA_SERVER_TAIL_CKPT_MIN=4096) reaches the server.
#   ssh $RIG "cd <work> && BUILDS=<builds> BUILD=<release build> MODEL=<first shard> DRAFT=<draft> OUT=<work>/run1 \
#     LOADS='A B' setsid nohup bash <work>/r43_run.sh > <work>/run1.out 2>&1 < /dev/null &"
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
: "${BUILDS:?}" "${BUILD:?}" "${MODEL:?}" "${DRAFT:?}" "${OUT:?}"
# shellcheck disable=SC1091
. "$here/settings.env"
# shellcheck disable=SC2086
export $PRESET_ENV
export BUILDS BUILD MODEL OUT
export LOADS=${LOADS:-A B}
export MIN_AVAIL_MB=${MIN_AVAIL_MB:-60}
export ARGS="-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host${ARGS_ADD:+ $ARGS_ADD}"
mkdir -p "$OUT"
{ echo "PRESET_ENV=$PRESET_ENV"; echo "ARGS=$ARGS"; echo "LOADS=$LOADS MIN_AVAIL_MB=$MIN_AVAIL_MB";
  env | grep -E '^(LLAMA_|GGML_)' | sort; } > "$OUT/env.txt"
exec bash "$here/rig_taper.sh"
