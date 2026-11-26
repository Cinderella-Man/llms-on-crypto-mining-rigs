#!/usr/bin/env bash
# Checks a started (or dry) llama-server command of run/gmix_agent.sh against release/flash-next-v5/settings.env:
#   the environment before the binary equals the preset's PRESET_ENV byte for byte, followed by EXTRA (nothing by
#     default; GGML_CUDA_MTOK_IQ_FAST=2 for the gmix-q2 runs);
#   the flags after the binary are the preset's PRESET_ARGS plus the harness's own --metrics and --no-webui and
#     -md <the draft>, each flag with its value, in any order (the harness writes the common part first).
# C912's run/preset_check.sh (docs/data/2026-11-24_c912_flash_agent/) with switches after the preset's as an argument.
# Usage, from the repository root: bash docs/data/2026-11-26_c1173_gmix_agent_rig1/run/preset_check.sh LINE_FILE [EXTRA]
set -euo pipefail
f=${1:?}; extra=${2:-}
# shellcheck disable=SC1091
. release/flash-next-v5/settings.env
want_env=$PRESET_ENV${extra:+ $extra}
line=$(cat "$f")
bin=$(tr ' ' '\n' <<< "$line" | grep -m1 '/bin/llama-server$') || { echo "FAIL: no llama-server in the line"; exit 1; }
env_part=${line%% "$bin" *}
args_part=${line#* "$bin" }
draft=$(sed -n 's/.* -md \([^ ]*\) .*/\1/p' <<< "$line")
[ "$env_part" = "$want_env" ] && echo "env: equal to PRESET_ENV${extra:+ plus $extra}, $(wc -w <<< "$env_part") switches" \
  || { echo "FAIL: env differs"; diff <(tr ' ' '\n' <<< "$want_env") <(tr ' ' '\n' <<< "$env_part"); exit 1; }
# each flag with the words up to the next flag, sorted
groups() { tr ' ' '\n' | awk '/^-/ { if (g != "") print g; g = $0; next } { g = g " " $0 } END { if (g != "") print g }' | sort; }
if diff <(groups <<< "$PRESET_ARGS --metrics --no-webui -md $draft") <(groups <<< "$args_part"); then
  echo "args: PRESET_ARGS plus --metrics --no-webui -md $draft ($(groups <<< "$args_part" | wc -l) flags)"
else
  echo "FAIL: args differ (< preset, > line)"; exit 1
fi
