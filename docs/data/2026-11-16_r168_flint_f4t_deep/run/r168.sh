#!/usr/bin/env bash
# R168 (Q1495): Flint's released v4 preset against v3.0 at depth. Decode and prompt reading at depth on Flint Pascal
# Q8 GGUF (p065) for the v4 preset as released (release/flint-v4/settings.env) on the release candidate v4-rc2
# (pascal-p106-v4 eb82916d3) against the v3.0 release (release/flint-v3/settings.env on
# pascal-p106-v3-b12076-ladder5 1551f5fe1), on one rig. R161's lane and settings
# (docs/data/2026-11-14_r161_flint_release_deep/run/r161.sh: bench/deep_lane with the sampler pass, each arm on its
# own build), with two arms:
#   v3    v3.0's build (ARM_BIN_v3) on release/flint-v3/settings.env as shipped (no switch)
#   v4    v4-rc2's build (BIN) on release/flint-v4/settings.env as it is: its PRESET_ENV and its PRESET_ARGS
# The lane reads one preset file (v3's) and changes it per arm. v4's line is v3.0's line with flags added after it, so
# this script reads both files and stops unless:
#   - v3's PRESET_ENV is empty (the lane would give it to both arms);
#   - v4's PRESET_ARGS is v3's PRESET_ARGS followed by "flag value" pairs, each flag one that v3's line does not hold
#     and none twice.
# v4 then gets those pairs as ARM_ARGS_v4 (the lane adds them at the end of the line, in their order) and v4's whole
# PRESET_ENV as ARM_ENV_v4, so each arm's server line is its own file's, word for word. DRY=1 prints both lines.
# Both arms load the same file (MODEL), as in R161: the presets name the file by their own names.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (v4-rc2's llama-server on the rig), REF_BIN
# (v3.0's llama-server on the rig), MODEL (Flint Pascal Q8 GGUF on the rig), TEXT_LOCAL (the recorded agent text here:
# the release check's proxy.jsonl.bodies.gz), OUT (here, for collect), DEPTHS (default 0 4096 8192 16384 32768 49152).
# STEP=start|wait|status|collect; DRY=1 prints the server lines and starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
: "${WORK:?WORK=<run folder on the rig>}"
export RIG WORK
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
V3_FILE=release/flint-v3/settings.env
V4_FILE=release/flint-v4/settings.env
V3_ARGS=$(. $V3_FILE && echo "$PRESET_ARGS")
V3_ENV=$(. $V3_FILE && echo "$PRESET_ENV")
V4_ARGS=$(. $V4_FILE && echo "$PRESET_ARGS")
V4_ENV=$(. $V4_FILE && echo "$PRESET_ENV")
[ -n "$V3_ARGS" ] && [ -n "$V4_ARGS" ] || { echo "no PRESET_ARGS in $V3_FILE or $V4_FILE"; exit 1; }
[ -z "$V3_ENV" ] || { echo "$V3_FILE sets PRESET_ENV ($V3_ENV): the lane would give it to v4 too"; exit 1; }
# v4's added flags: what follows v3's line in v4's
case "$V4_ARGS" in
  "$V3_ARGS") V4_ADD='' ;;
  "$V3_ARGS "*) V4_ADD=${V4_ARGS#"$V3_ARGS "} ;;
  *) echo "$V4_FILE's PRESET_ARGS does not begin with $V3_FILE's line"; exit 1 ;;
esac
read -ra v3w <<< "$V3_ARGS"
read -ra add <<< "$V4_ADD"
seen=' '
for ((i = 0; i < ${#add[@]}; i += 2)); do
  f=${add[i]} v=${add[i + 1]:-}
  [[ $f =~ ^--?[A-Za-z] ]] || { echo "v4's added '$f' is not a flag"; exit 1; }
  [ -n "$v" ] && ! [[ $v =~ ^--?[A-Za-z] ]] || { echo "v4's added flag $f has no value: the lane adds 'flag value' pairs"; exit 1; }
  for w in "${v3w[@]}"; do [ "$w" = "$f" ] && { echo "v4 changes v3's flag $f: the lane would replace it in place"; exit 1; }; done
  [[ $seen == *" $f "* ]] && { echo "v4 adds $f twice"; exit 1; }
  seen="$seen$f "
done
echo "v3: $V3_FILE, no switch; v4: $V4_FILE, environment: ${V4_ENV:-none}; flags after v3.0's line: ${V4_ADD:-none}"
case ${STEP:-start} in
start)
  : "${BIN:?}" "${REF_BIN:?}" "${MODEL:?}" "${TEXT_LOCAL:?}"
  PRESET=$PWD/$V3_FILE BIN=$BIN ARM_BIN_v3=$REF_BIN MODEL=$MODEL TEXT_LOCAL=$TEXT_LOCAL \
    TAG='^t1swift/' SAMPLER_PASS=1 N_PREDICT=256 DEPTHS="${DEPTHS:-0 4096 8192 16384 32768 49152}" WARM=1 \
    ARMS="v3 v4" \
    ARM_ENV_v4="$V4_ENV" \
    ARM_ARGS_v4="$V4_ADD" \
    LOADS="warm:v3 v31:v3 v41:v4 v32:v3 v42:v4" \
    bash $L start || exit 1
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
esac
