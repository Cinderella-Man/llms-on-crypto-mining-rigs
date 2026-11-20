#!/usr/bin/env bash
# Checks a started (or dry) llama-server command of run/flash_taper.sh against release/flash-next-v5/settings.env:
#   the environment before the binary equals the preset's PRESET_ENV byte for byte;
#   the flags after the binary are the preset's PRESET_ARGS, with --ubatch-taper's value set to TAPER (short=600:64,
#     the preset's own, by default), plus the harness's own --metrics and --no-webui and -md <the draft>, each flag with
#     its value, in any order (the harness writes the common part first).
# Usage, from the repository root:
#   bash docs/data/2026-11-20_r227_flash_taper1400_agent/run/preset_check.sh LINE_FILE [TAPER]
set -euo pipefail
f=${1:?}; taper=${2:-short=600:64}
# shellcheck disable=SC1091
. release/flash-next-v5/settings.env
[[ " $PRESET_ARGS " == *" --ubatch-taper short=600:64 "* ]] || { echo "FAIL: settings.env has no --ubatch-taper short=600:64"; exit 1; }
want_args=${PRESET_ARGS/"--ubatch-taper short=600:64"/"--ubatch-taper $taper"}
line=$(cat "$f")
bin=$(tr ' ' '\n' <<< "$line" | grep -m1 '/bin/llama-server$') || { echo "FAIL: no llama-server in the line"; exit 1; }
env_part=${line%% "$bin" *}
args_part=${line#* "$bin" }
draft=$(sed -n 's/.* -md \([^ ]*\) .*/\1/p' <<< "$line")
[ "$env_part" = "$PRESET_ENV" ] && echo "env: equal to PRESET_ENV, $(wc -w <<< "$env_part") switches" \
  || { echo "FAIL: env differs"; diff <(tr ' ' '\n' <<< "$PRESET_ENV") <(tr ' ' '\n' <<< "$env_part"); exit 1; }
# each flag with the words up to the next flag, sorted
groups() { tr ' ' '\n' | awk '/^-/ { if (g != "") print g; g = $0; next } { g = g " " $0 } END { if (g != "") print g }' | sort; }
if diff <(groups <<< "$want_args --metrics --no-webui -md $draft") <(groups <<< "$args_part"); then
  echo "args: PRESET_ARGS (--ubatch-taper $taper) plus --metrics --no-webui -md $draft ($(groups <<< "$args_part" | wc -l) flags)"
else
  echo "FAIL: args differ (< preset, > line)"; exit 1
fi
