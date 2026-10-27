#!/usr/bin/env bash
# E3: a Flint file within noise of Q8_0 at fewer bytes, as it ran. Everything executes on the quality machine (SSH
# alias QM); the machine that runs this script only copies files over ssh.
#   push      near_q8.py, score.sh and scripts/compose_gguf.py to WORK/run
#   deltas    bytes each switch adds (file and per token), from the headers of BASE and DONOR: WORK/deltas.tsv
#   score     score.sh detached for ITEMS (default: the anchors, then every switch); log WORK/score-<TAG>.out
#   collect   WORK/kld.tsv from WORK/logs
#   solve     WORK/solve.tsv (frontier per token), solve-file.tsv, solve-picks.tsv; SOLVE_ARGS adds options
#   fetch     the TSVs to OUT (default: this folder's parent)
# Env: QM, WORK (scratch on QM, fast disk), BASE (Unsloth's Flint Q8_0 on QM), DONOR (Flint-Pascal-Q5 on QM), REF, BIN,
#   GGUFPY, GPUS, TS, PREFETCH, EXTRA_MOUNTS (see score.sh), ITEMS, TAG, OUT.
set -euo pipefail
QM=${QM:?ssh alias of the quality machine}
WORK=${WORK:?scratch dir on QM}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
OUT=${OUT:-$(dirname "$HERE")}
IMG_PY=pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
SWITCHES="q0 q3 q4 q5 kv4 kv5 o4 o5 sh4 sh5 gu1 gu2 gu3 gu4 gu5 dn0 dn1 dn2 dn3 dn4 dn5 head embd"
py() {  # near_q8.py on QM in Docker, every file folder mounted at its own path
    local m="-v $WORK:$WORK ${EXTRA_MOUNTS:-}" p d
    for p in "${BASE:-}" "${DONOR:-}" "${GGUFPY:-}/x"; do
        [ -n "$p" ] || continue; d=$(dirname "$p"); case " $m " in *" -v $d:$d"*) ;; *) m="$m -v $d:$d:ro" ;; esac
    done
    ssh "$QM" "docker run --rm --label pascal-quants=1 --cpu-shares 256 --network none $m -e PYTHONPATH=${GGUFPY:-} $IMG_PY \
        nice -n 19 python $WORK/run/near_q8.py $*"
}

case "${1:-}" in
push)
    ssh "$QM" "mkdir -p $WORK/run"
    for f in "$HERE/near_q8.py" "$HERE/score.sh" "$REPO/scripts/compose_gguf.py"; do
        ssh "$QM" "cat > $WORK/run/$(basename "$f")" < "$f"
    done
    ;;
deltas)
    py deltas "${BASE:?}" "${DONOR:?}" "$WORK/deltas.tsv"
    ;;
score)
    TAG=${TAG:-switches}
    ITEMS=${ITEMS:-"q8_0@$BASE pascal-q5@$DONOR q8r= $SWITCHES"}
    ENV="WORK=$WORK BASE=$BASE DONOR=$DONOR REF=$REF BIN=$BIN GGUFPY=$GGUFPY GPUS=$GPUS TS='${TS:-}' \
PREFETCH=${PREFETCH:-0} MIN_FREE_GB=${MIN_FREE_GB:-50} EXTRA_MOUNTS='${EXTRA_MOUNTS:-}'"
    ssh "$QM" "cd $WORK; $ENV nohup bash $WORK/run/score.sh $ITEMS > $WORK/score-$TAG.out 2>&1 < /dev/null &"
    echo "started: $WORK/score-$TAG.out"
    ;;
collect)
    items=$(ssh "$QM" "cd $WORK/logs && for f in kld-*.log; do n=\${f#kld-}; echo -n \"\${n%.log}=$WORK/logs/\$f \"; done")
    py collect "$WORK/kld.tsv" "$items"
    ;;
solve)
    py solve "$WORK/deltas.tsv" "$WORK/kld.tsv" "$WORK/solve.tsv" ${SOLVE_ARGS:-}
    ;;
fetch)
    for f in deltas kld solve solve-file solve-picks; do ssh "$QM" "cat $WORK/$f.tsv" > "$OUT/$f.tsv"; done
    ;;
*)
    echo "usage: $0 push | deltas | score | collect | solve | fetch" >&2
    exit 1
    ;;
esac
