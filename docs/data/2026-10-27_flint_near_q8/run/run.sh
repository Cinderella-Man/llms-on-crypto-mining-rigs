#!/usr/bin/env bash
# E3: a Flint file within noise of Q8_0 at fewer bytes, as it ran. Everything executes on the quality machine (SSH
# alias QM); the machine that runs this script only copies files over ssh.
#   push      near_q8.py, score.sh and scripts/compose_gguf.py to WORK/run
#   deltas    bytes each switch adds (file and per token), from the headers of BASE and DONOR: WORK/deltas.tsv
#   agent_base  BASE's (Q8_0's) logits on AGENT_TXT at 12 × 8,192 tokens into AGENT_BASE, detached; log next to it
#   score     score.sh detached for ITEMS (default: the anchors, then every switch); log WORK/score-<TAG>.out;
#             TEXT=agent scores against AGENT_BASE instead of BF16's wikitext logits
#   collect   WORK/kld.tsv from WORK/logs
#   solve     WORK/solve.tsv (frontier per token), solve-file.tsv, solve-picks.tsv; SOLVE_ARGS adds options
#   fetch     the TSVs to OUT (default: this folder's parent)
#   relay     FILE (a composed file on QM) to RIG_FILE on the rig through this machine, with the md5 of the bytes as
#             written on the rig in RIG_FILE.md5; then compared with FILE.md5 on QM (RIG: the rig's SSH alias)
# Env: QM, WORK (scratch on QM, fast disk), BASE (Unsloth's Flint Q8_0 on QM), DONOR (Flint-Pascal-Q5 on QM), REF, BIN,
#   GGUFPY, GPUS, TS, PREFETCH, EXTRA_MOUNTS, TEXT, AGENT_TXT, AGENT_BASE (see score.sh), ITEMS, TAG, OUT.
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
agent_base)
    AGENT_TXT=${AGENT_TXT:?} AGENT_BASE=${AGENT_BASE:?}
    m="-v $(dirname "$AGENT_BASE"):$(dirname "$AGENT_BASE")"
    for p in "$BASE" "$AGENT_TXT" "$BIN"; do
        d=$(dirname "$p"); case " $m " in *" -v $d:$d"*) ;; *) m="$m -v $d:$d:ro" ;; esac; done
    ssh "$QM" "mkdir -p $(dirname "$AGENT_BASE"); cd $(dirname "$AGENT_BASE"); nohup docker run --rm --label pascal-quants=1 \
        --cpu-shares 256 --gpus '\"device=$GPUS\"' -e CUDA_DEVICE_ORDER=PCI_BUS_ID $m nvidia/cuda:12.9.2-devel-ubuntu24.04 \
        $BIN -m $BASE -f $AGENT_TXT -c 8192 -b 8192 -ub 512 --chunks 12 -ngl 99 ${TS:+-ts $TS} -fa on \
        --kl-divergence-base $AGENT_BASE > $AGENT_BASE.log 2>&1 < /dev/null &"
    echo "started: $AGENT_BASE (log $AGENT_BASE.log)"
    ;;
score)
    TAG=${TAG:-switches}
    ITEMS=${ITEMS:-"q8_0@$BASE pascal-q5@$DONOR q8r= $SWITCHES"}
    ENV="WORK=$WORK BASE=$BASE DONOR=$DONOR REF=$REF BIN=$BIN GGUFPY=$GGUFPY GPUS=$GPUS TS='${TS:-}' \
PREFETCH=${PREFETCH:-0} MIN_FREE_GB=${MIN_FREE_GB:-50} EXTRA_MOUNTS='${EXTRA_MOUNTS:-}' TEXT=${TEXT:-wiki} \
AGENT_TXT=${AGENT_TXT:-} AGENT_BASE=${AGENT_BASE:-}"
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
relay)
    RIG=${RIG:-rig} FILE=${FILE:?} RIG_FILE=${RIG_FILE:?}
    ssh "$QM" "cat $FILE" | ssh "$RIG" "mkdir -p $(dirname "$RIG_FILE") && tee $RIG_FILE.part | md5sum > $RIG_FILE.md5 \
        && mv $RIG_FILE.part $RIG_FILE && sed -i 's#-#$(basename "$RIG_FILE")#' $RIG_FILE.md5 && stat -c '%s %n' $RIG_FILE"
    want=$(ssh "$QM" "cut -d' ' -f1 $FILE.md5"); have=$(ssh "$RIG" "cut -d' ' -f1 $RIG_FILE.md5")
    [ "$want" = "$have" ] && echo "md5 ok: $have" || { echo "md5 MISMATCH: $want on QM, $have on the rig"; exit 1; }
    ;;
*)
    echo "usage: $0 push | deltas | agent_base | score | collect | solve | fetch | relay" >&2
    exit 1
    ;;
esac
