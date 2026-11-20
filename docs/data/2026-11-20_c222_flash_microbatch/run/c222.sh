#!/usr/bin/env bash
# c222.sh: C222 (Q775), Flash-Next's arm: whole prompt calls of up to 2,048 tokens in micro-batches of 128 and of 64,
# against the v5 preset's taper (short=600:64). One detached load of the taper replay
# (docs/data/2026-11-04_prompt_taper/run/rig_taper.sh with taper_replay.py) on the rig, with Flash-Next's v5 preset
# (release/flash-next-v5/settings.env) as it is; only the taper changes, per request ("ubatch_taper"), pass by pass.
# Made from docs/data/2026-11-12_r60_taper_v4/run/r60_loads.sh, which ran v4's line through r43_run.sh; this one
# exports the v5 preset's environment and hands its whole line to rig_taper.sh. Run from a machine that only runs ssh;
# the load runs on the rig, detached, and ends with $WORK/$OUTN/DONE.
#   passes (PASSES): the control first and last and between the arms,
#     short=600:64 short=2048:128 short=2048:64 short=600:64 short=2048:64 short=2048:128 short=600:64
#   A call is "short" when it starts with nothing in flight and is waited for (llama-server's whole prompt call with
#   -b 4096 and in-call checkpoints) and holds N tokens or fewer; the spec holds one short rule, so short=2048:128 reads
#   calls under 600 tokens in micro-batches of 128 as well. No fresh prompts (FRESH=0): rig_taper.sh's fresh prompts
#   compare off and on, neither of them an arm here.
# rig_taper.sh adds -ngl 99 -fa on -c 65536 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format
# deepseek around ARGS; the preset's line holds the same flags with the same values, so the server keeps those values
# (it takes the last of a repeated flag and logs a warning).
# Env: RIG (ssh alias), WORK (the folder on the rig with rig_taper.sh, taper_replay.py, the agent check's turns.csv
#      (docs/data/2026-11-03_retype_agent/agent/turns.csv) and the v5 preset's settings.env), BUILDS, BUILD (the v5
#      engine's folder), MODEL (Swift Q2_0's first shard on the rig), DRAFT (the MTP draft on the rig), OUTN (c222),
#      PASSES (above), MIN_AVAIL_MB (60, as r43_run.sh), DRY=1 prints the rig command only.
set -euo pipefail
: "${RIG:?}" "${WORK:?}" "${BUILDS:?}" "${BUILD:?}" "${MODEL:?}" "${DRAFT:?}"
OUTN=${OUTN:-c222}
PASSES=${PASSES:-short=600:64 short=2048:128 short=2048:64 short=600:64 short=2048:64 short=2048:128 short=600:64}
MIN_AVAIL_MB=${MIN_AVAIL_MB:-60}
O=$WORK/$OUTN
# shellcheck disable=SC2016
cmd="cd $WORK && . ./settings.env && export \$PRESET_ENV && mkdir -p $O && \
{ echo \"PRESET_ENV=\$PRESET_ENV\"; echo \"ARGS=-md $DRAFT \$PRESET_ARGS\"; echo \"PASSES_A=$PASSES\"; \
echo \"MIN_AVAIL_MB=$MIN_AVAIL_MB FRESH=0\"; env | grep -E '^(LLAMA_|GGML_)' | sort; } > $O/env.txt && \
BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL OUT=$O LOADS=A FRESH=0 MIN_AVAIL_MB=$MIN_AVAIL_MB \
ARGS=\"-md $DRAFT \$PRESET_ARGS\" PASSES_A='$PASSES' setsid nohup bash $WORK/rig_taper.sh > $WORK/$OUTN.out 2>&1 < /dev/null &"
echo "$cmd"
[ "${DRY:-0}" = 1 ] && exit 0
# an ssh call that starts a detached job can wait for it: the timeout ends the call, not the job
timeout 20 ssh "$RIG" "$cmd" || true
sleep 5
timeout 60 ssh "$RIG" "cat $O/env.txt; tail -n 3 $O/run.log 2> /dev/null"
