#!/usr/bin/env bash
# Checks a started (or dry) llama-server command of run/flash_shortlist.sh against release/flash-next-v5/settings.env:
#   the environment before the binary equals the preset's PRESET_ENV byte for byte, with LLAMA_DRAFT_SHORTLIST's third
#     value (the full-head rounds after a shortlist miss) set to ROUNDS (2, the preset's own, by default);
#   the flags after the binary are the preset's PRESET_ARGS plus the harness's own --metrics and --no-webui and
#     -md <the draft>, each flag with its value, in any order (the harness writes the common part first).
# C912's run/preset_check.sh (docs/data/2026-11-24_c912_flash_agent/) with the shortlist's rounds as an argument.
# Usage, from the repository root: bash docs/data/2026-11-25_r229_flash_shortlist_agent/run/preset_check.sh LINE_FILE [ROUNDS]
set -euo pipefail
f=${1:?}; rounds=${2:-2}
# shellcheck disable=SC1091
. release/flash-next-v5/settings.env
S=LLAMA_DRAFT_SHORTLIST=32768,8192
[[ " $PRESET_ENV " == *" $S,2 "* ]] || { echo "FAIL: settings.env has no $S,2"; exit 1; }
want_env=${PRESET_ENV/"$S,2"/"$S,$rounds"}
line=$(cat "$f")
bin=$(tr ' ' '\n' <<< "$line" | grep -m1 '/bin/llama-server$') || { echo "FAIL: no llama-server in the line"; exit 1; }
env_part=${line%% "$bin" *}
args_part=${line#* "$bin" }
draft=$(sed -n 's/.* -md \([^ ]*\) .*/\1/p' <<< "$line")
[ "$env_part" = "$want_env" ] && echo "env: equal to PRESET_ENV ($S,$rounds), $(wc -w <<< "$env_part") switches" \
  || { echo "FAIL: env differs"; diff <(tr ' ' '\n' <<< "$want_env") <(tr ' ' '\n' <<< "$env_part"); exit 1; }
# each flag with the words up to the next flag, sorted
groups() { tr ' ' '\n' | awk '/^-/ { if (g != "") print g; g = $0; next } { g = g " " $0 } END { if (g != "") print g }' | sort; }
if diff <(groups <<< "$PRESET_ARGS --metrics --no-webui -md $draft") <(groups <<< "$args_part"); then
  echo "args: PRESET_ARGS plus --metrics --no-webui -md $draft ($(groups <<< "$args_part" | wc -l) flags)"
else
  echo "FAIL: args differ (< preset, > line)"; exit 1
fi
