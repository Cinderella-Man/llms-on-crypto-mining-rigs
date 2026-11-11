#!/usr/bin/env bash
# C770 (Q1355): the names the switch line counts as read, for each fork branch of the v4 line: SCAN (the branch's
# common/switch-names.cmake) run on each branch's src, common, tools and ggml/src, and the names documented in
# docs/switches.md against them.
# Usage: SRC=<llama.cpp clone with the fork as remote REMOTE, the branches fetched> SCAN=<switch-names.cmake>
#   DOCS=<docs/switches.md> OUT=<dir> BASE=<branch> TIP=<branch> [DAY=<branch>] BRANCHES='<branch> ...' [REMOTE=fork]
#   bash names.sh
# Output: OUT/names/<branch>.txt (one name per line), OUT/names.tsv (branch, commit, names, seconds to scan, the names
#   added and removed against BASE), OUT/docs.tsv (each documented name: in BASE, in TIP, in DAY, the branches that
#   read it), OUT/names_verdict.txt
set -uo pipefail
export LC_ALL=C
SRC=${SRC:?SRC=<llama.cpp clone>}
SCAN=${SCAN:?SCAN=<switch-names.cmake>}
DOCS=${DOCS:?DOCS=<docs/switches.md>}
OUT=${OUT:?OUT=<dir>}
BASE=${BASE:?BASE=<branch>}
TIP=${TIP:?TIP=<branch>}
DAY=${DAY:-}
REMOTE=${REMOTE:-fork}
BRANCHES=${BRANCHES:-}
mkdir -p "$OUT/names"
T=$OUT/trees
short() { echo "${1#pascal-p106-v4-b12246-}"; }

printf 'branch\tcommit\tnames\tscan_s\tadded\tremoved\tadded_names\n' > "$OUT/names.tsv"
all="$BASE $TIP $DAY"
for b in $BRANCHES; do
    [[ " $all " == *" $b "* ]] || all="$all $b"
done
for b in $all; do
    s=$(short "$b")
    rm -rf "$T/$s" && mkdir -p "$T/$s"
    git -C "$SRC" archive "$REMOTE/$b" src common tools ggml/src | tar -x -C "$T/$s"
    t0=$(date +%s.%N)
    cmake -DROOT="$T/$s" -DOUT="$T/$s.cpp" -P "$SCAN"
    t1=$(date +%s.%N)
    grep -oE '^    "[A-Z0-9_]+",$' "$T/$s.cpp" | tr -d ' ",' > "$OUT/names/$s.txt"
    printf '%s\t%s\t%s\t%s\n' "$s" "$(git -C "$SRC" rev-parse --short=9 "$REMOTE/$b")" "$(wc -l < "$OUT/names/$s.txt")" \
        "$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.2f", b - a }')" > "$T/$s.row"
    rm -rf "${T:?}/$s" "$T/$s.cpp"
done
sb=$(short "$BASE")
for b in $all; do
    s=$(short "$b")
    added=$(comm -13 "$OUT/names/$sb.txt" "$OUT/names/$s.txt")
    removed=$(comm -23 "$OUT/names/$sb.txt" "$OUT/names/$s.txt")
    printf '%s\t%s\t%s\t%s\n' "$(cat "$T/$s.row")" "$(echo -n "$added" | grep -c .)" "$(echo -n "$removed" | grep -c .)" \
        "$(echo "$added" | tr '\n' ' ' | sed 's/ $//')" >> "$OUT/names.tsv"
done
rm -rf "$T"

# the documented names: every LLAMA_ or GGML_ name in docs/switches.md
grep -oE '\b(LLAMA|GGML)_[A-Z0-9_]*[A-Z0-9]\b' "$DOCS" | sort -u > "$OUT/docs_names.txt"
printf 'name\tin_base\tin_tip\tin_day\tbranches\n' > "$OUT/docs.tsv"
st=$(short "$TIP") sd=$(short "${DAY:-none}")
while read -r n; do
    has() { [ -f "$OUT/names/$1.txt" ] && grep -qx "$n" "$OUT/names/$1.txt" && echo yes || echo no; }
    br=$(grep -lx "$n" "$OUT"/names/*.txt 2>/dev/null | xargs -r -n 1 basename | sed 's/\.txt$//' | tr '\n' ',' | sed 's/,$//')
    printf '%s\t%s\t%s\t%s\t%s\n' "$n" "$(has "$sb")" "$(has "$st")" "$(has "$sd")" "${br:-none}" >> "$OUT/docs.tsv"
done < "$OUT/docs_names.txt"

{
    if cmp -s "$OUT/names/$sb.txt" "$OUT/names/$st.txt"; then
        echo "tip = base: the same $(wc -l < "$OUT/names/$st.txt") names"
    else
        echo "tip differs from base: $(comm -3 "$OUT/names/$sb.txt" "$OUT/names/$st.txt" | tr -d '\t' | tr '\n' ' ')"
    fi
    if [ -n "$DAY" ]; then
        miss=
        for b in $BRANCHES; do
            for n in $(comm -13 "$OUT/names/$sb.txt" "$OUT/names/$(short "$b").txt"); do
                grep -qx "$n" "$OUT/names/$sd.txt" || miss="$miss $(short "$b"):$n"
            done
        done
        echo "names the branches add that the day merge lacks:${miss:- none}"
    fi
    echo "documented names: $(wc -l < "$OUT/docs_names.txt"); in tip $(awk -F'\t' 'NR > 1 && $3 == "yes"' "$OUT/docs.tsv" | wc -l); in no scanned branch: $(awk -F'\t' 'NR > 1 && $5 == "none" { printf "%s ", $1 }' "$OUT/docs.tsv")"
} | tee "$OUT/names_verdict.txt"
