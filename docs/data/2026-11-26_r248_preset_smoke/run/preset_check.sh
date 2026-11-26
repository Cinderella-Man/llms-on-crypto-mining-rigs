#!/usr/bin/env bash
# Checks a started (or dry) llama-server command against a preset file, word by word:
#   the environment before the binary equals the preset's PRESET_ENV byte for byte;
#   the flags after the binary are the preset's PRESET_ARGS plus the harness's own --metrics and --no-webui and
#     -md <the draft>, each flag with its value, in any order (the harness writes its run line first).
# R243's run/preset_check.sh (docs/data/2026-11-26_r243_flash_131k_agent_rig1/) without its -c 65536 / -ub 512 arms:
# the preset as it stands since 2026-11-26 carries -c 131072 -ub 256 and _MIN_RATIO=0.45 itself.
# Usage, from the repository root:
#   [PRESET=release/flash-next-v5/settings.env] bash docs/data/2026-11-26_r248_preset_smoke/run/preset_check.sh LINE_FILE
set -euo pipefail
f=${1:?}
# shellcheck disable=SC1090
. "${PRESET:-release/flash-next-v5/settings.env}"
line=$(cat "$f")
bin=$(tr ' ' '\n' <<< "$line" | grep -m1 '/bin/llama-server$') || { echo "FAIL: no llama-server in the line"; exit 1; }
env_part=${line%% "$bin" *}
args_part=${line#* "$bin" }
draft=$(sed -n 's/.* -md \([^ ]*\) .*/\1/p' <<< "$line")
[ -n "$draft" ] || { echo "FAIL: no -md in the line"; exit 1; }
[ "$env_part" = "$PRESET_ENV" ] && echo "env: equal to PRESET_ENV, $(wc -w <<< "$env_part") switches" \
  || { echo "FAIL: env differs"; diff <(tr ' ' '\n' <<< "$PRESET_ENV") <(tr ' ' '\n' <<< "$env_part"); exit 1; }
# each flag with the words up to the next flag, sorted
groups() { tr ' ' '\n' | awk '/^-/ { if (g != "") print g; g = $0; next } { g = g " " $0 } END { if (g != "") print g }' | sort; }
want=$(groups <<< "$PRESET_ARGS --metrics --no-webui -md $draft")
if diff <(echo "$want") <(groups <<< "$args_part"); then
  echo "args: PRESET_ARGS plus --metrics --no-webui -md $draft ($(groups <<< "$args_part" | wc -l) flags;" \
    "$(grep -E '^-(c|ub) ' <<< "$want" | paste -sd' '))"
else
  echo "FAIL: args differ (< wanted, > line)"; exit 1
fi
