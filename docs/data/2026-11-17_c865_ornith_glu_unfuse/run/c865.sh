#!/usr/bin/env bash
# C865 (Q1559), Q350's first arm on a P106 rig: GGML_CUDA_MMID_GLU_UNFUSE unset, 1 and 2 on Ornith-1.5-122B Pascal N2P
# GGUF (valid1) on nine P106-090 cards, with its v4 preset (release/ornith-v4/settings.env), all three arms on one build
# (fork pascal-p106-v5 a112712e3, which merged this switch, 9550c62d6, with two others, each off by default). Arms:
#   off  the switch unset: v4.0's fusions. On this file the routed gate, up and GLU are one launch in every layer
#        (N2P or N3P, gate and up of one type) and the shared expert's are another (Q5_0); the one launch for both
#        (the shared-expert match) never applies, since it needs one weight type (ornith_ffn_types.tsv)
#   u1   GGML_CUDA_MMID_GLU_UNFUSE=1: the routed gate and up products as two launches, then the GLU; the shared expert
#        stays fused
#   u2   GGML_CUDA_MMID_GLU_UNFUSE=2: upstream's #32239, no fused matrix-vector product on Pascal: the shared expert's
#        gate, up and GLU apart too
# Run from the repository's root on a machine that runs only ssh and tar; everything timed runs detached on the rig.
# Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (the build's llama-server on the rig; its
# test-backend-ops is beside it), MODEL (the Ornith file on the rig), TEXT_LOCAL (the recorded agent text here: the
# release check's proxy.jsonl.bodies.gz, as R162), OUT (here, for collect). STEP:
#   tbo      test-backend-ops on card 0, each arm: the gate/up/GLU fusion, shared-expert, expert-product and norm-to-Q8
#            (a routed and a shared pair reading the norm's Q8_1 row) cases of the types this file decodes with (N2P,
#            N3P, Q5_0), then every GLU case. Every case must pass in every arm. Logs to
#            WORK-tbo/tbo-<arm>-{mm,glu}.log on the rig, the pass lines printed; a few minutes, untimed
#   dry      the lane's server lines, nothing started
#   start    bench/deep_lane on the rig, detached: loads warm:off, then off, u1, u2 twice, alternated; depths 0 and
#            8,192; 256 tokens at temperature 0 twice per depth (the second pass with the preset's sampler,
#            SAMPLER_PASS=1, as R162), every reply kept (KEEP_REPLY=1) so a difference can be found by token
#   wait     returns when the lane has ended; status: its last lines
#   tables   on the rig, after the run and before collect: deep_table.py with off as the control, into WORK/out/tables
#            (each arm a gain, negligible or a loss against off at each depth; the same greedy reply in every arm)
#   collect  the lane's results to OUT; WORK is removed on the rig (CLEAN=0 keeps it), WORK-tbo too
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/c865}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
SW=GGML_CUDA_MMID_GLU_UNFUSE
ARMS_ENV="off: u1:$SW=1 u2:$SW=2"
common() {
  export PRESET=release/ornith-v4/settings.env BIN=${BIN:?} MODEL=${MODEL:?} TEXT_LOCAL=${TEXT_LOCAL:?} \
    TAG='^t1swift/' SAMPLER_PASS=1 N_PREDICT=256 KEEP_REPLY=1 DEPTHS="0 8192" \
    ARMS="off u1 u2" ARM_ENV_u1="$SW=1" ARM_ENV_u2="$SW=2" \
    LOADS="warm:off f1:off a1:u1 b1:u2 f2:off a2:u1 b2:u2"
}
case ${STEP:-start} in
tbo)
  : "${BIN:?}"
  TBO=$(dirname "$BIN")/test-backend-ops
  T=$WORK-tbo
  # shellcheck disable=SC2029
  ssh "$RIG" "mkdir -p $T && cd $T && for spec in $ARMS_ENV; do
      arm=\${spec%%:*}; sw=\${spec#*:}
      echo \"== \$arm (\${sw:-switch unset}) \$(date -Is)\"
      env \$sw $TBO -b CUDA0 -o MUL_MAT_VEC_FUSION,MUL_MAT_ID_SHARED,MUL_MAT_ID,ADD_RMS_NORM_MUL_Q8 \
        -p 'type(_a)?=(n2p|n3p|q5_0),' > tbo-\$arm-mm.log 2>&1; echo \"mm exit \$?\"
      env \$sw $TBO -b CUDA0 -o GLU > tbo-\$arm-glu.log 2>&1; echo \"glu exit \$?\"
      grep -hE 'tests passed|FAIL' tbo-\$arm-mm.log tbo-\$arm-glu.log | sort | uniq -c | head -n 12
    done"
  ;;
dry) common; DRY=1 bash $L start ;;
start) common; bash $L start ;;
wait) bash $L wait ;;
status) bash $L status ;;
tables)
  ssh "$RIG" "cd $WORK && mkdir -p out/tables && T=out/tables &&
    python3 -I deep_lane/deep_table.py out --control off --out \$T/table.md --tsv \$T/classes.tsv > /dev/null &&
    ls \$T && cat \$T/table.md | head -n 60"
  ;;
collect)
  OUT=${OUT:?} bash $L collect
  if [ "${CLEAN:-1}" = 1 ]; then
    ssh "$RIG" "[ -d $WORK-tbo ] && tar -C $WORK-tbo -cf - ." | { mkdir -p "$OUT/tbo" && tar -C "$OUT/tbo" -xf -; } &&
      ssh "$RIG" "rm -rf $WORK-tbo" && echo "tbo logs in $OUT/tbo"
  fi
  ;;
*) echo "STEP=tbo|dry|start|wait|status|tables|collect"; exit 1 ;;
esac
