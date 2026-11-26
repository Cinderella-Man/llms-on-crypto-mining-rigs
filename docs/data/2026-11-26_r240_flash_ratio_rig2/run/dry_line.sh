#!/usr/bin/env bash
# Prints the server line that run/flash_ratio.sh would start for ARM, without loading anything: the driver chain
# (flash_nsm.sh, C854's flash_agent.sh and R156's r156.sh) runs unchanged on a copy of the files it
# reads, in which the agent check's harness (docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh) is replaced
# by a stub that prints the harness's own "starting <arm> on the rig:" line (its COMMON part taken from the real
# harness) and exits, and ssh only echoes. C1175's run/dry_line.sh (itself R229's and R227's) with this row's script.
# Usage, from the repository root:
#   ARM=r25a WORK=<an empty scratch folder> NEW=<build folder name> [DRAFT=<draft path on the rig>] bash docs/data/2026-11-26_r240_flash_ratio_rig2/run/dry_line.sh
set -euo pipefail
: "${ARM:?}" "${WORK:?}" "${NEW:?}"
HARNESS=docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh
D=$WORK/repo
rm -rf "$D" "$WORK/out"
mkdir -p "$D"
tar -cf - release/flash-next-v3/settings.env docs/data/2026-11-13_r156_agent_v4r/run/r156.sh \
  docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh docs/data/2026-11-26_r240_flash_ratio_rig2/run/flash_ratio.sh \
  | tar -C "$D" -xf -
mkdir -p "$D/$(dirname $HARNESS)"
{ echo '#!/usr/bin/env bash'
  echo 'CTX=${CTX:-65536}'
  sed -n '/^COMMON="/,/--no-webui"$/p' "$HARNESS"
  # shellcheck disable=SC2016
  echo 'case " $STEPS " in *" run "*) v=ARGS_$ARMS; echo "starting $ARMS on the rig: ${REL_ENV:+$REL_ENV }$REL_BIN $COMMON ${!v}" ;; *) echo "stub: $STEPS" ;; esac'
} > "$D/$HARNESS"
grep -q -- '--no-webui"$' "$D/$HARNESS" || { echo "COMMON not found in $HARNESS"; exit 1; }
ssh() { echo "ssh (dry): $1"; }
export -f ssh
cd "$D"
env -u V4R_ENV -u GAINS_ENV -u V4C2_TAPER ARM="$ARM" RIG=dry CLIENT=local MODEL="${MODEL:-model.gguf}" DRAFT="${DRAFT:-/dry/draft.gguf}" NEW="$NEW" REF=ref RIGDIR=/dry/rigdir \
  OUT="$WORK/out" CLIENT_DIR="$WORK/client" CLIENT_OUT="$WORK/client-out" \
  bash docs/data/2026-11-26_r240_flash_ratio_rig2/run/flash_ratio.sh > "$WORK/console.txt" 2>&1 || true
line=$(grep -h -m1 'starting v4r on the rig: ' "$WORK/out/agent-v4r.console") || { cat "$WORK/console.txt"; exit 1; }
printf '%s\n' "${line#starting v4r on the rig: }"
