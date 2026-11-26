#!/usr/bin/env bash
# Checks a started (or dry) llama-server command of run/flash_131k.sh against release/flash-next-v5/settings.env:
#   the environment before the binary equals the preset's PRESET_ENV byte for byte;
#   the flags after the binary are the preset's PRESET_ARGS plus the harness's own --metrics and --no-webui and
#     -md <the draft>, each flag with its value, in any order (the harness writes the common part first); for
#     ARM=l131 with -c 131072 in place of -c 65536 and -ub 256 in place of -ub 512 (R124's line).
# C1175's run/preset_check.sh (docs/data/2026-11-26_c1175_nsm10_rig2/) for the preset as it stands since 2026-11-26
# (_NSM=10, _MIN_RATIO=0.5), with R124's two replacements (docs/data/2026-11-26_r124_flash_131k_rig2/run/line_check.sh).
# PRESET (default release/flash-next-v5/settings.env) names the preset file to check against: the runs used the preset at
# d296f5ce (git show d296f5ce:release/flash-next-v5/settings.env > preset_d296f5ce.env; PRESET=preset_d296f5ce.env ...),
# before 602ba0cf added cuts=1 to the taper.
# Usage, from the repository root: bash docs/data/2026-11-26_r243_flash_131k_agent_rig1/run/preset_check.sh LINE_FILE [preset|l131]
set -euo pipefail
f=${1:?}; arm=${2:-preset}
case $arm in preset|l131) ;; *) echo "FAIL: unknown arm $arm (preset or l131)"; exit 1 ;; esac
# shellcheck disable=SC1091
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
grep -qx -- '-c 65536' <<< "$want" && grep -qx -- '-ub 512' <<< "$want" \
  || { echo "FAIL: PRESET_ARGS has no -c 65536 or no -ub 512"; exit 1; }
[ "$arm" = l131 ] && want=$(sed -E 's/^-c 65536$/-c 131072/; s/^-ub 512$/-ub 256/' <<< "$want" | sort)
if diff <(echo "$want") <(groups <<< "$args_part"); then
  echo "args ($arm): PRESET_ARGS$([ "$arm" = l131 ] && echo " with -c 131072 -ub 256") plus --metrics --no-webui -md $draft ($(groups <<< "$args_part" | wc -l) flags)"
else
  echo "FAIL: args differ (< wanted, > line)"; exit 1
fi
