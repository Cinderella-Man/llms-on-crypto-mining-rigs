#!/usr/bin/env bash
# C912's Flash-Next agent run (Q1649): checks a started (or dry) llama-server command against today's
# release/flash-next-v5/settings.env, whose line reads prompt calls of up to 1,400 tokens in micro-batches of 64:
#   the environment before the binary equals the preset's PRESET_ENV byte for byte;
#   the flags after the binary are the preset's PRESET_ARGS plus the harness's own --metrics and --no-webui and
#     -md <the draft>, each flag with its value, in any order (the harness writes the common part first).
# R227's run/preset_check.sh (docs/data/2026-11-20_r227_flash_taper1400_agent/) did the same against the preset as it
# was before it took --ubatch-taper short=1400:64; this one takes the preset's line as it is.
# Usage, from the repository root:
#   bash docs/data/2026-11-24_c912_flash_agent/run/preset_check.sh LINE_FILE
set -euo pipefail
f=${1:?}
# shellcheck disable=SC1091
. release/flash-next-v5/settings.env
line=$(cat "$f")
bin=$(tr ' ' '\n' <<< "$line" | grep -m1 '/bin/llama-server$') || { echo "FAIL: no llama-server in the line"; exit 1; }
env_part=${line%% "$bin" *}
args_part=${line#* "$bin" }
draft=$(sed -n 's/.* -md \([^ ]*\) .*/\1/p' <<< "$line")
[ "$env_part" = "$PRESET_ENV" ] && echo "env: equal to PRESET_ENV, $(wc -w <<< "$env_part") switches" \
  || { echo "FAIL: env differs"; diff <(tr ' ' '\n' <<< "$PRESET_ENV") <(tr ' ' '\n' <<< "$env_part"); exit 1; }
# each flag with the words up to the next flag, sorted
groups() { tr ' ' '\n' | awk '/^-/ { if (g != "") print g; g = $0; next } { g = g " " $0 } END { if (g != "") print g }' | sort; }
if diff <(groups <<< "$PRESET_ARGS --metrics --no-webui -md $draft") <(groups <<< "$args_part"); then
  echo "args: PRESET_ARGS plus --metrics --no-webui -md $draft ($(groups <<< "$args_part" | wc -l) flags)"
else
  echo "FAIL: args differ (< preset, > line)"; exit 1
fi
