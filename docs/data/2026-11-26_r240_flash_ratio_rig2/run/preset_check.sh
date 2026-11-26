#!/usr/bin/env bash
# Checks a started (or dry) llama-server command of run/flash_ratio.sh against release/flash-next-v5/settings.env:
#   the environment before the binary equals the preset's PRESET_ENV byte for byte, with
#     GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=10 (the preset's own) and
#     GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO set to RATIO (0.5, the preset's own, by default);
#   the flags after the binary are the preset's PRESET_ARGS plus the harness's own --metrics and --no-webui and
#     -md <the draft>, each flag with its value, in any order (the harness writes the common part first).
# C1175's run/preset_check.sh (docs/data/2026-11-26_c1175_nsm10_rig2/, from R229's and C912's) for the preset since
# 2026-11-26 15:11, with the switch point as the argument.
# Usage, from the repository root: bash docs/data/2026-11-26_r240_flash_ratio_rig2/run/preset_check.sh LINE_FILE [RATIO]
set -euo pipefail
f=${1:?}; ratio=${2:-0.5}
# shellcheck disable=SC1091
. release/flash-next-v5/settings.env
N=GGML_CUDA_FATTN_SPARSE_PASCAL_NSM
R=GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO
[[ " $PRESET_ENV " == *" $N=10 "* ]] || { echo "FAIL: settings.env has no $N=10"; exit 1; }
[[ " $PRESET_ENV " == *" $R=0.5 "* ]] || { echo "FAIL: settings.env has no $R=0.5"; exit 1; }
want_env=${PRESET_ENV/"$R=0.5"/"$R=$ratio"}
line=$(cat "$f")
bin=$(tr ' ' '\n' <<< "$line" | grep -m1 '/bin/llama-server$') || { echo "FAIL: no llama-server in the line"; exit 1; }
env_part=${line%% "$bin" *}
args_part=${line#* "$bin" }
draft=$(sed -n 's/.* -md \([^ ]*\) .*/\1/p' <<< "$line")
[ "$env_part" = "$want_env" ] && echo "env: equal to PRESET_ENV ($N=10, $R=$ratio), $(wc -w <<< "$env_part") switches" \
  || { echo "FAIL: env differs"; diff <(tr ' ' '\n' <<< "$want_env") <(tr ' ' '\n' <<< "$env_part"); exit 1; }
# each flag with the words up to the next flag, sorted
groups() { tr ' ' '\n' | awk '/^-/ { if (g != "") print g; g = $0; next } { g = g " " $0 } END { if (g != "") print g }' | sort; }
if diff <(groups <<< "$PRESET_ARGS --metrics --no-webui -md $draft") <(groups <<< "$args_part"); then
  echo "args: PRESET_ARGS plus --metrics --no-webui -md $draft ($(groups <<< "$args_part" | wc -l) flags)"
else
  echo "FAIL: args differ (< preset, > line)"; exit 1
fi
