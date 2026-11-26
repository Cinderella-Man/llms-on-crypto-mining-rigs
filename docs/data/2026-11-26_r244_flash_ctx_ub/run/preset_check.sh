#!/usr/bin/env bash
# Checks a started (or dry) llama-server command of run/r244.sh against release/flash-next-v5/settings.env:
#   the environment before the binary equals the preset's PRESET_ENV byte for byte;
#   the flags after the binary are the preset's PRESET_ARGS plus the harness's own --metrics and --no-webui and
#     -md <the draft>, each flag with its value, in any order; for ARM=u256 with -ub 256 in place of -ub 512, for
#     ARM=l131 with -c 131072 in place of -c 65536 and -ub 256 in place of -ub 512.
# R243's run/preset_check.sh (docs/data/2026-11-26_r243_flash_131k_agent_rig1/, arms preset and l131) with the arm u256
# added; for p512 and l131 the two give the same verdict.
# PRESET (default release/flash-next-v5/settings.env) names the preset file to check against.
# Usage, from the repository root: bash docs/data/2026-11-26_r244_flash_ctx_ub/run/preset_check.sh LINE_FILE [p512|u256|l131]
set -euo pipefail
f=${1:?}; arm=${2:-p512}
case $arm in p512|preset|u256|l131) ;; *) echo "FAIL: unknown arm $arm (p512, u256 or l131)"; exit 1 ;; esac
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
grep -qx -- '-c 65536' <<< "$want" && grep -qx -- '-ub 512' <<< "$want" \
  || { echo "FAIL: PRESET_ARGS has no -c 65536 or no -ub 512"; exit 1; }
case $arm in
  u256) want=$(sed -E 's/^-ub 512$/-ub 256/' <<< "$want" | sort); what=" with -ub 256" ;;
  l131) want=$(sed -E 's/^-c 65536$/-c 131072/; s/^-ub 512$/-ub 256/' <<< "$want" | sort); what=" with -c 131072 -ub 256" ;;
  *) what= ;;
esac
if diff <(echo "$want") <(groups <<< "$args_part"); then
  echo "args ($arm): PRESET_ARGS$what plus --metrics --no-webui -md $draft ($(groups <<< "$args_part" | wc -l) flags)"
else
  echo "FAIL: args differ (< wanted, > line)"; exit 1
fi
