#!/usr/bin/env bash
# Checks one server line of run/r124.sh (a "  env ... llama-server -m ... <flags>" line from STEP=dry, or the "=== ..."
# line a load wrote at the top of its server-<load>.log) against release/flash-next-v5/settings.env:
#   the environment equals the preset's PRESET_ENV byte for byte;
#   the flags are PRESET_ARGS with -c 131072 and -ub 256, plus -md <the draft> for ARM=draft, or without the draft's
#     flags (-md and every --spec-* flag with its value) for ARM=nodraft; each flag with its value, in any order.
# From C1175's run/preset_check.sh (docs/data/2026-11-26_c1175_nsm10_rig2/).
# PRESET (default release/flash-next-v5/settings.env) names the preset file to check against: the run used the preset at
# bf7982b1 (git show bf7982b1:release/flash-next-v5/settings.env > preset_bf7982b1.env; PRESET=preset_bf7982b1.env ...).
# Usage, from the repository root: bash docs/data/2026-11-26_r124_flash_131k_rig2/run/line_check.sh LINE_FILE draft|nodraft
set -euo pipefail
f=${1:?}; arm=${2:?draft or nodraft}
# shellcheck disable=SC1091
. "${PRESET:-release/flash-next-v5/settings.env}"
line=$(sed -E 's/^=== [^ ]+ [^ ]+: //; s/^[0-9:]* +//; s/^ *env //' "$f")
line=${line% --host 127.0.0.1 --port *}
bin=$(tr ' ' '\n' <<< "$line" | grep -m1 '/llama-server$') || { echo "FAIL: no llama-server in the line"; exit 1; }
env_part=${line%% "$bin" *}
args_part=${line#* "$bin" }
args_part=$(sed -E 's/^-m [^ ]+ //' <<< "$args_part")
[ "$env_part" = "$PRESET_ENV" ] && echo "env: equal to PRESET_ENV, $(wc -w <<< "$env_part") switches" \
  || { echo "FAIL: env differs"; diff <(tr ' ' '\n' <<< "$PRESET_ENV") <(tr ' ' '\n' <<< "$env_part"); exit 1; }
groups() { tr ' ' '\n' | awk '/^-/ { if (g != "") print g; g = $0; next } { g = g " " $0 } END { if (g != "") print g }' | sort; }
want=$(groups <<< "$PRESET_ARGS" | sed -E 's/^-c 65536$/-c 131072/; s/^-ub 512$/-ub 256/')
if [ "$arm" = draft ]; then
  draft=$(sed -n 's/.* -md \([^ ]*\).*/\1/p' <<< "$args_part")
  [ -n "$draft" ] || { echo "FAIL: no -md in the draft arm's line"; exit 1; }
  want=$(printf '%s\n-md %s\n' "$want" "$draft" | sort)
else
  want=$(grep -v -E '^(--spec-|-md )' <<< "$want")
fi
if diff <(echo "$want") <(groups <<< "$args_part"); then
  echo "args ($arm): PRESET_ARGS with -c 131072 -ub 256$([ "$arm" = draft ] && echo " plus -md" || echo " without the draft's flags") ($(groups <<< "$args_part" | wc -l) flags)"
else
  echo "FAIL: args differ (< wanted, > line)"; exit 1
fi
