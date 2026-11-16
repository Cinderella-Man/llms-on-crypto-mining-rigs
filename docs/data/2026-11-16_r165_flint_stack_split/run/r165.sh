#!/usr/bin/env bash
# R165 (Q1486): Flint's candidate stack split at depth. R161's deep lane
# (docs/data/2026-11-14_r161_flint_release_deep/run/r161.sh) found the stack of every candidate of release/flint-v4
# slower than the plain v4 preset from 4,096 tokens on; this run times the candidates one an arm against that preset,
# on Flint Pascal Q8 GGUF (p065), one build (BIN, v4-rc2), the sampler pass on.
# The control is v4: release/flint-v4/settings.env as it is, read by the lane from the file (its PRESET_ENV and
# PRESET_ARGS). Every other arm is that line plus its own ARM_ENV_<arm> (variables) and ARM_ARGS_<arm> (flags, as the
# lane's ARGS_SET), nothing else. The arms come from the environment:
#   ARMS            the arms, the control v4 first (e.g. "v4 v4gf v4hs")
#   ARM_ENV_<arm>   variables added to v4's PRESET_ENV (e.g. ARM_ENV_v4gf=GGML_CUDA_GRAPHS_FORCE=1)
#   ARM_ARGS_<arm>  flags added to v4's line (e.g. ARM_ARGS_v4tp="--ubatch-taper short=600:64")
#   LOADS           the load order, "name:arm ..." (default: the lane's, a warm-up of v4, then every arm once, twice)
# The two runs of R165:
#   R165a  ARMS="v4 v4gf v4hs" ARM_ENV_v4gf=GGML_CUDA_GRAPHS_FORCE=1 ARM_ENV_v4hs=GGML_CUDA_HOP_STAGED=1
#          LOADS="warm:v4 v41:v4 gf1:v4gf hs1:v4hs v42:v4 gf2:v4gf hs2:v4hs"
#   R165b  ARMS="v4 v4ka v4ck v4tp" ARM_ENV_v4ka="GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1"
#          ARM_ARGS_v4ck="--ctx-checkpoints-vram 4" ARM_ARGS_v4tp="--ubatch-taper short=600:64"
#          LOADS="warm:v4 v41:v4 ka1:v4ka ck1:v4ck tp1:v4tp v42:v4 ka2:v4ka ck2:v4ck tp2:v4tp"
# The script refuses an arm list whose first arm is not v4, a control with switches of its own, a candidate arm with
# no switch, and a candidate whose variable the preset already sets.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (v4-rc2's llama-server on the rig), MODEL
# (Flint Pascal Q8 GGUF on the rig), TEXT_LOCAL (the recorded agent text here: the release check's
# proxy.jsonl.bodies.gz), OUT (here, for collect), DEPTHS (default 0 4096 8192 16384 32768 49152), PRESET (default
# release/flint-v4/settings.env). STEP=start|wait|status|collect; DRY=1 prints the server lines and starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
: "${WORK:?WORK=<run folder on the rig>}"
export RIG WORK
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
PRESET=${PRESET:-$PWD/release/flint-v4/settings.env}
case ${STEP:-start} in
start)
  : "${BIN:?}" "${MODEL:?}" "${TEXT_LOCAL:?}" "${ARMS:?ARMS=<arms, v4 first>}"
  set -- $ARMS
  [ "$1" = v4 ] || { echo "the first arm (the control) must be v4, not $1"; exit 1; }
  v=ARM_ENV_v4 w=ARM_ARGS_v4
  [ -z "${!v:-}${!w:-}" ] || { echo "the control v4 is the preset as it is: no ARM_ENV_v4 or ARM_ARGS_v4"; exit 1; }
  P_ENV=$(. "$PRESET" && echo "$PRESET_ENV")
  shift
  for a in "$@"; do
    v=ARM_ENV_$a w=ARM_ARGS_$a
    [ -n "${!v:-}${!w:-}" ] || { echo "arm $a has no switch: set ARM_ENV_$a or ARM_ARGS_$a"; exit 1; }
    for e in ${!v:-}; do
      [[ " $P_ENV " != *" ${e%%=*}="* ]] || { echo "arm $a: ${e%%=*} is already in the preset"; exit 1; }
    done
  done
  # ARM_ENV_<arm>, ARM_ARGS_<arm> and LOADS reach the lane from this script's environment as they are
  PRESET=$PRESET BIN=$BIN MODEL=$MODEL TEXT_LOCAL=$TEXT_LOCAL \
    TAG='^t1swift/' SAMPLER_PASS=1 N_PREDICT=256 DEPTHS="${DEPTHS:-0 4096 8192 16384 32768 49152}" WARM=1 \
    ARMS="$ARMS" \
    bash $L start || exit 1
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
esac
