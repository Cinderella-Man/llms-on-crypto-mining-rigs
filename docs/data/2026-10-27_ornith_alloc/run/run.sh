#!/usr/bin/env bash
# E1 (docs/research/notes/slq.md): Ornith-1.5-122B's N2P/N3P mix allocated per 8-layer group, as it ran. Everything
# executes on the quality machine (SSH alias QM); the machine that runs this script only copies files over ssh.
# Files c, b and d come from docs/data/2026-10-27_lowbit/run/quantize.sh (FILE=c, b, d; d is c with the expert
# formats swapped, gate/up N3P and down N2P).
#   stage     copy SRC_C (file c on a slow or shared disk) to FILE_C on the fast disk in one pass (O_DIRECT reads, so
#             the page cache of other jobs stays), once / keeps MIN_FREE_GB, and check its md5 against SRC_C.md5
#   push      copy alloc.py, score.sh and scripts/compose_gguf.py to WORK/run
#   deltas    bytes each switch adds on the cards, from the headers of c, b and d: WORK/deltas.tsv
#   score     score.sh detached on QM for ITEMS (default: the 18 switches); log WORK/score-<tag>.out
#   collect   WORK/kld.tsv from WORK/logs and file c's logs (C_LOGS: their prefix, as for alloc.py collect)
#   solve     WORK/solve.tsv and WORK/solve-switches.tsv
#   fetch     the TSVs to OUT (default: this folder's parent)
# Env: QM, WORK (scratch on QM, fast disk), SRC_C, FILE_C, FILE_B, FILE_D, KLD, WIKI, BIN, GGUFPY, GPUS, TEXTS, PREFETCH,
#   MIN_FREE_GB, EXTRA_MOUNTS (see score.sh), C_LOGS, ITEMS, OUT.
set -euo pipefail
QM=${QM:?ssh alias of the quality machine}
WORK=${WORK:?scratch dir on QM}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
OUT=${OUT:-$(dirname "$HERE")}
IMG_PY=pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
ALL="gu0 gu1 gu2 gu3 gu4 gu5 dn0 dn1 dn2 dn3 dn4 dn5 dense0 dense1 dense2 dense3 dense4 dense5"
py() {  # run alloc.py on QM in Docker with every file folder mounted at its own path
    local m="-v $WORK:$WORK ${EXTRA_MOUNTS:-}" p d
    for p in "${FILE_C:-}" "${FILE_B:-}" "${FILE_D:-}" "${GGUFPY:-}/x" "${C_LOGS:-}"; do
        [ -n "$p" ] || continue; d=$(dirname "$p"); case " $m " in *" -v $d:$d"*) ;; *) m="$m -v $d:$d:ro" ;; esac
    done
    ssh "$QM" "docker run --rm --label pascal-quants=1 --cpu-shares 256 --network none $m -e PYTHONPATH=${GGUFPY:-} $IMG_PY \
        nice -n 19 python $WORK/run/alloc.py $*"
}

case "${1:-}" in
stage)
    SRC_C=${SRC_C:?} FILE_C=${FILE_C:?}
    ssh "$QM" "set -e; mkdir -p \$(dirname $FILE_C); size=\$(stat -c %s $SRC_C)
        while [ \$(( \$(df -B1 --output=avail \$(dirname $FILE_C) | tail -1) - size )) -lt \$((${MIN_FREE_GB:-50} * 1000**3)) ]; do
            echo \"\$(date +%T) waiting for room\"; sleep 120; done
        echo \"\$(date +%T) copy \$((size / 1000**3)) GB\"
        dd if=$SRC_C bs=64M iflag=direct status=none | tee $FILE_C.part | md5sum | cut -d' ' -f1 > $FILE_C.md5
        mv $FILE_C.part $FILE_C; echo \"\$(date +%T) done\"
        [ ! -f $SRC_C.md5 ] || [ \"\$(cut -d' ' -f1 $SRC_C.md5)\" = \"\$(cat $FILE_C.md5)\" ] && echo md5 ok || echo md5 MISMATCH"
    ;;
push)
    ssh "$QM" "mkdir -p $WORK/run"
    for f in "$HERE/alloc.py" "$HERE/score.sh" "$REPO/scripts/compose_gguf.py"; do
        ssh "$QM" "cat > $WORK/run/$(basename "$f")" < "$f"
    done
    ;;
deltas)
    py deltas "${FILE_C:?}" "${FILE_B:?}" "${FILE_D:?}" "$WORK/deltas.tsv"
    ;;
score)
    TAG=${TAG:-switches}
    ENV="WORK=$WORK FILE_C=$FILE_C FILE_B=$FILE_B FILE_D=$FILE_D KLD=$KLD WIKI=$WIKI BIN=$BIN GGUFPY=$GGUFPY \
GPUS=${GPUS:-3,4} TEXTS='${TEXTS:-wiki agent code}' PREFETCH=${PREFETCH:-1} MIN_FREE_GB=${MIN_FREE_GB:-50} \
EXTRA_MOUNTS='${EXTRA_MOUNTS:-}'"
    ssh "$QM" "cd $WORK; $ENV nohup bash $WORK/run/score.sh ${ITEMS:-$ALL} > $WORK/score-$TAG.out 2>&1 < /dev/null &"
    echo "started: $WORK/score-$TAG.out"
    ;;
collect)
    items="c=${C_LOGS:?}"
    for n in $ALL ${EXTRA_NAMES:-}; do items="$items $n=$WORK/logs/alloc-$n"; done
    py collect "$WORK/kld.tsv" "$items"
    ;;
solve)
    py solve "$WORK/deltas.tsv" "$WORK/kld.tsv" "$WORK/solve.tsv" ${SOLVE_ARGS:-}
    ;;
fetch)
    for f in deltas kld solve solve-switches; do ssh "$QM" "cat $WORK/$f.tsv" > "$OUT/$f.tsv"; done
    ;;
*)
    echo "usage: $0 stage | push | deltas | score | collect | solve | fetch" >&2
    exit 1
    ;;
esac
