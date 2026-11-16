#!/usr/bin/env bash
# FN93b (Q1487): which of Ornith's stacked candidates makes nine P106-090 cards write another greedy reply at depth.
# In R162's deep lane (docs/data/2026-11-14_r162_ornith_release_deep/run/r162.sh) the candidate stack v4s wrote
# another greedy reply than the v4 preset at 8,192, 16,384 and 32,768 tokens. This run is R162's lane with one
# candidate a load: the v4 preset alone (the control, whose replies must be R162's v4 replies), then the preset with
# each candidate of the stack on its own. It reads only greedy replies: no quiet check and no speed (QUIET=0), no
# warm-up load, no sampler pass, R162's depths up to 16,384, so that every prime reads R162's chunk on the same cached
# tokens. bench/deep_lane runs detached on the rig; this machine runs only ssh and tar.
# Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (v4-rc2's llama-server on the rig), MODEL (the
# Ornith file on the rig), TEXT_LOCAL (the recorded agent text here: the release check's proxy.jsonl.bodies.gz), OUT
# (here, for collect); the arms from outside: ARMS (in load order, the control first) and ARM_ENV_<arm> or
# ARM_ARGS_<arm> for each, exported (run/arms.env holds FN93b's seven).
# STEP=start|wait|status|replies|collect; DRY=1 with start prints each arm's server line and starts none.
#   replies  after the run has ended and before collect: run/replies.py on the rig, against REF (the reference run's
#            requests.csv here, default R162's) and REF_JSONL (its <load>.jsonl files here, if kept), into
#            WORK/out/replies.md and replies.tsv
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/fn93b}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
HERE=$(dirname "$0")
case ${STEP:-start} in
start)
  : "${ARMS:?ARMS=<the arms in load order, the control first> (source run/arms.env)}"
  LOADS=''
  for a in $ARMS; do LOADS="$LOADS $a:$a"; done
  PRESET=release/ornith-v4/settings.env BIN=${BIN:?} MODEL=${MODEL:?} TEXT_LOCAL=${TEXT_LOCAL:?} TAG='^t1swift/' \
    N_PREDICT=256 SAMPLER_PASS=0 QUIET=0 WARM=0 DEPTHS="0 4096 8192 16384" LOADS="${LOADS# }" bash $L start
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
replies)
  ref=${REF:-docs/data/2026-11-14_r162_ornith_release_deep/requests.csv}
  ssh "$RIG" "mkdir -p $WORK/ref && cat > $WORK/deep_lane/replies.py" < "$HERE/replies.py" || exit 1
  ssh "$RIG" "cat > $WORK/ref/requests.csv" < "$ref" || exit 1
  rj=''
  for f in ${REF_JSONL:-}; do
    ssh "$RIG" "cat > $WORK/ref/$(basename "$f")" < "$f" || exit 1
    rj="$rj ref/$(basename "$f")"
  done
  ssh "$RIG" "cd $WORK && python3 -I deep_lane/replies.py --out out --ref ref/requests.csv --ref-arms v4,v4s \
    --ref-jsonl$rj --md out/replies.md --tsv out/replies.tsv"
  ;;
*) echo "STEP=start|wait|status|collect|replies"; exit 1 ;;
esac
