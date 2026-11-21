#!/usr/bin/env bash
# C932 (2026-11-21): when did pipelined decoding stop giving plain decoding's logits on the CPU? Runs check_at.sh
# (beside this script) on fork commits, in steps (STEPS, in this order):
#   points  POINTS, oldest first (the v3.0 and v4.0 release tags and the v5 line's merges), each with ARCHS_POINTS
#           (the Flash-Next and Flint architectures), MISSES_POINTS and llama-pipe-recurrent-check;
#   bisect  git bisect run --first-parent between the last good point before the first bad one and that bad one
#           (or GOOD and BAD), the Flash-Next architecture only, MISSES 0;
#   side    when the first bad commit of the first-parent bisect is a merge: git bisect run between its first parent
#           and the merge, which walks the merged branch's own commits; upstream commits there have no
#           examples/pipespec and are skipped (exit 125), so a merge of an upstream tag ends with a range of candidates.
# Bisects run in their own clone ($W/src, objects shared with SRC) with --no-checkout.
# Usage: W=<work dir> SRC=<llama.cpp clone with the fork's commits> REPO=<folder with check_at.sh, cpu_tests.sh, pipespec/>
#   [POINTS='...'] [GOOD=<commit> BAD=<commit>] [STEPS='points bisect side'] [LOCK=<flock file>] [JOBS=8] [THREADS=6]
#   bash sweep.sh
# Done when $W/DONE exists (it holds the exit code); the outcome in $W/outcome.txt, every check in $W/verdicts.tsv.
# At its end it stops every container labelled c932=1: run nothing else with that label alongside it.
set -uo pipefail
W=${W:?W=<work dir>}
SRC=${SRC:?SRC=<llama.cpp clone>}
REPO=${REPO:?REPO=<scripts folder>}
POINTS=${POINTS:-1551f5fe1 eb82916d3 1f1110c67 602691153 6644c2bc6 3dfad876f}
STEPS=${STEPS:-points bisect side}
ARCHS_POINTS=${ARCHS_POINTS:-qwen4exp flint}
MISSES_POINTS=${MISSES_POINTS:-0 0.5}
export W SRC REPO LOCK="${LOCK:-}" JOBS="${JOBS:-8}" THREADS="${THREADS:-6}"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
out()    { echo "$*" | tee -a "$W/outcome.txt"; }
finish() { docker ps -q --filter label=c932=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
mkdir -p "$W"
rm -f "$W/DONE"
echo "=== start $(date -Is); steps: $STEPS"

GOOD=${GOOD:-} BAD=${BAD:-}
if step points; then
    prev_good="" first_bad=""
    for c in $POINTS; do
        STAGE=points ARCHS="$ARCHS_POINTS" MISSES="$MISSES_POINTS" PRC=1 bash "$REPO/check_at.sh" "$c"
        rc=$?
        out "point $c rc=$rc"
        if [ -z "$first_bad" ]; then
            [ $rc = 0 ] && prev_good=$c
            [ $rc = 1 ] && first_bad=$c
        elif [ $rc = 0 ]; then
            out "point $c is good after the bad point $first_bad: not one change"
        fi
    done
    [ -z "$GOOD" ] && GOOD=$prev_good
    [ -z "$BAD" ] && BAD=$first_bad
    out "points: last good before the first bad ${GOOD:-none}, first bad ${BAD:-none}"
fi

if [ -d "$W/src/.git" ]; then git -C "$W/src" bisect reset -q > /dev/null 2>&1; else
    git clone -q --no-checkout --shared "$SRC" "$W/src" || finish 1
fi
git -C "$W/src" fetch -q "$SRC" 'refs/remotes/*:refs/remotes/*' 2> /dev/null

bisect() { # <tag> <bad> <good> [--first-parent]
    local tag=$1 bad=$2 good=$3; shift 3
    git -C "$W/src" bisect reset -q > /dev/null 2>&1
    git -C "$W/src" bisect start --no-checkout "$@" "$bad" "$good" > "$W/bisect-$tag.start" 2>&1 || { cat "$W/bisect-$tag.start"; return 1; }
    # check_at.sh reads commits from SRC; the clone shares its objects
    (cd "$W/src" && git bisect run env STAGE="$tag" bash -c "bash '$REPO/check_at.sh' \$(git rev-parse BISECT_HEAD)") \
        > "$W/bisect-$tag.out" 2>&1
    git -C "$W/src" bisect log > "$W/bisect-$tag.log"
    grep -aE 'is the first bad commit|only skipped commits|There are only|bisect found' "$W/bisect-$tag.out" | head -n 3
    first=$(grep -a -B0 'is the first bad commit' "$W/bisect-$tag.out" | head -n 1 | cut -d' ' -f1)
    git -C "$W/src" bisect reset -q > /dev/null 2>&1
    echo "first=$first"
}

FIRST=""
if step bisect; then
    [ -n "$GOOD" ] && [ -n "$BAD" ] || { out "bisect: no good/bad pair"; finish 1; }
    r=$(bisect fp "$BAD" "$GOOD" --first-parent)
    echo "$r"
    FIRST=$(echo "$r" | sed -n 's/^first=//p')
    out "first-parent bisect $GOOD..$BAD: first bad ${FIRST:-none}: $(git -C "$SRC" log -1 --format='%h %s' "${FIRST:-HEAD}" 2> /dev/null)"
    out "$(echo "$r" | grep -v '^first=')"
fi

if step side && [ -n "$FIRST" ]; then
    np=$(git -C "$SRC" rev-list --parents -n 1 "$FIRST" | wc -w)
    if [ "$np" -gt 2 ]; then
        r=$(bisect side "$FIRST" "$FIRST^1")
        echo "$r"
        S=$(echo "$r" | sed -n 's/^first=//p')
        out "side bisect $FIRST^1..$FIRST: first bad ${S:-none (see bisect-side.out)}: $(git -C "$SRC" log -1 --format='%h %s' "${S:-$FIRST}")"
        out "$(echo "$r" | grep -v '^first=')"
    else
        out "first bad $FIRST is not a merge: no side bisect"
    fi
fi
finish 0
