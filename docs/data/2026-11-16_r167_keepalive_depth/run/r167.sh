#!/usr/bin/env bash
# R167 (Q1492): the keep-alive at depth. R165b (docs/data/2026-11-16_r165_flint_stack_split) found that on Flint the
# keep-alive (GGML_CUDA_KEEPALIVE=decode with GGML_CUDA_KEEPALIVE_EAGER=1) reads prompts deep into a context slower
# (-16.5 % over all depths) and decodes slower (-3.0 %). This run times a model's v4 preset against the same preset
# with variables taken out or put in, on the deep lane (bench/deep_lane), one build (BIN), the sampler pass on: R157's
# lane and settings (docs/data/2026-11-13_r157_release_deep), with the arms from the environment.
#   PRESET          the model's preset file (e.g. release/flash-next-v4/settings.env), read by the lane as it is
#   ARMS            the arms, the control first: the control is the preset as it is
#   ARM_DROP_<arm>  names of variables the arm takes out of the preset's PRESET_ENV; each must be set by the preset
#   ARM_ADD_<arm>   "NAME=value ..." variables the arm puts in; none may be set by the preset unless the arm drops it
#   ARM_ARGS_<arm>  flags the arm changes on the preset's line (the lane's ARGS_SET); optional
#   LOADS           the load order, "name:arm ..." (default: the lane's, a warm-up of the control, then every arm once,
#                   twice)
# The lane's ENV_DROP is the same for every arm, so this script drops the union of all arms' ARM_DROP_* and gives each
# arm back, with the preset's own values, the ones it keeps (its ARM_ENV_<arm>, followed by its ARM_ADD_<arm>). Every
# arm's environment is then the preset's, less its own drops, plus its own adds; the variables given back stand at the
# end of the line, and the server prints its switches sorted (its "switches:" line), so the control's switches are the
# preset's.
# The three runs of R167:
#   R167b  Flash-Next (PRESET=release/flash-next-v4/settings.env, MODEL_DRAFT the MTP draft, WD_ENV=MIN_AVAIL_MB=60):
#          ARMS="v4r v4nk v4ne" ARM_DROP_v4nk="GGML_CUDA_KEEPALIVE GGML_CUDA_KEEPALIVE_EAGER"
#          ARM_DROP_v4ne=GGML_CUDA_KEEPALIVE_EAGER LOADS="warm:v4r r1:v4r n1:v4nk e1:v4ne r2:v4r n2:v4nk e2:v4ne"
#   R167a  Ornith (release/ornith-v4/settings.env), the same three arms
#   R167c  Flint (release/flint-v4/settings.env, which sets neither variable): ARM_ADD_<arm>=GGML_CUDA_KEEPALIVE=decode
# The script refuses a control with changes of its own, an arm with none, a drop of a variable the preset does not set,
# an add of one it sets (and the arm keeps), an add that is not NAME=value, and ENV_DROP, ENV_ADD or ARM_ENV_* given
# from outside (it sets them).
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (the build's llama-server on the rig),
# MODEL (the model's first shard on the rig), MODEL_DRAFT (its draft on the rig, when the preset has one), TEXT_LOCAL
# (the recorded agent text here: the release check's proxy.jsonl.bodies.gz), OUT (here, for collect), DEPTHS (default
# 0 4096 8192 16384 32768 49152), WD_ENV (the RAM watchdog's limits, e.g. MIN_AVAIL_MB=60). STEP=start|wait|status|
# collect; DRY=1 prints every arm's server line and starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
: "${WORK:?WORK=<run folder on the rig>}"
export RIG WORK
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
case ${STEP:-start} in
start)
  : "${PRESET:?PRESET=<the preset file of the model>}" "${BIN:?}" "${MODEL:?}" "${TEXT_LOCAL:?}" "${ARMS:?ARMS=<arms, the control first>}"
  [ -r "$PRESET" ] || { echo "no preset at $PRESET"; exit 1; }
  [ -z "${ENV_DROP+x}${ENV_ADD+x}" ] || { echo "ENV_DROP and ENV_ADD are set by this script: use ARM_DROP_<arm> and ARM_ADD_<arm>"; exit 1; }
  for v in $(compgen -v | grep '^ARM_ENV_'); do echo "$v is set by this script: use ARM_DROP_<arm> and ARM_ADD_<arm>"; exit 1; done
  P_ENV=$(. "$PRESET" && echo "$PRESET_ENV")
  p_val() { local w; for w in $P_ENV; do [ "${w%%=*}" = "$1" ] && { echo "$w"; return 0; }; done; return 1; }
  set -- $ARMS
  control=$1
  v=ARM_DROP_$control w=ARM_ADD_$control x=ARM_ARGS_$control
  [ -z "${!v:-}${!w:-}${!x:-}" ] || { echo "the control $control is the preset as it is: no ARM_DROP_, ARM_ADD_ or ARM_ARGS_$control"; exit 1; }
  shift
  union=''
  for a in "$@"; do
    v=ARM_DROP_$a w=ARM_ADD_$a x=ARM_ARGS_$a
    [ -n "${!v:-}${!w:-}${!x:-}" ] || { echo "arm $a changes nothing: set ARM_DROP_$a, ARM_ADD_$a or ARM_ARGS_$a"; exit 1; }
    for n in ${!v:-}; do
      [[ $n =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || { echo "arm $a: '$n' is not a variable name"; exit 1; }
      p_val "$n" > /dev/null || { echo "arm $a: $n is not set by the preset, so it cannot be dropped"; exit 1; }
      [[ " $union " == *" $n "* ]] || union="${union:+$union }$n"
    done
    for e in ${!w:-}; do
      [[ $e =~ ^[A-Za-z_][A-Za-z0-9_]*=.+$ ]] || { echo "arm $a: '$e' is not NAME=value"; exit 1; }
      if p_val "${e%%=*}" > /dev/null && [[ " ${!v:-} " != *" ${e%%=*} "* ]]; then
        echo "arm $a: ${e%%=*} is already set by the preset"; exit 1
      fi
    done
  done
  for a in $ARMS; do
    v=ARM_DROP_$a w=ARM_ADD_$a
    env_a=''
    for n in $union; do [[ " ${!v:-} " == *" $n "* ]] || env_a="${env_a:+$env_a }$(p_val "$n")"; done
    env_a="$env_a${!w:+ ${!w}}"
    export "ARM_ENV_$a=${env_a# }"
    echo "arm $a: drops ${!v:-none}; adds ${!w:-none}; given back from the preset's ENV_DROP: ${env_a:-none}"
  done
  # MODEL_DRAFT, WD_ENV, ARM_ARGS_<arm> and LOADS reach the lane from this script's environment as they are
  PRESET=$PRESET BIN=$BIN MODEL=$MODEL TEXT_LOCAL=$TEXT_LOCAL \
    TAG=${TAG:-'^t1swift/'} SAMPLER_PASS=1 N_PREDICT=256 DEPTHS="${DEPTHS:-0 4096 8192 16384 32768 49152}" WARM=1 \
    ENV_DROP="$union" ARMS="$ARMS" \
    bash $L start || exit 1
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
esac
