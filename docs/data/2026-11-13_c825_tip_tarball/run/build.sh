#!/usr/bin/env bash
# C825 (2026-11-13): the P106 rig tarball of pascal-p106-v4's tip, which holds every change of 2026-11-12, each behind its
# switch and off by default. Built on a shared build machine's CPU and checked against the -c813 rig tarball, the last
# rig tarball built on the same base. The method is C813's rig, names and sass steps
# (docs/data/2026-11-12_c813_hrow_half/run/checks.sh). The build waits for LOCK (flock) and runs at a low CPU share
# (--cpu-shares=256, nice 19, JOBS compile jobs), so one build runs at a time on a shared machine.
# Steps (STEPS, in this order by default):
#   rig    scripts/build_llamacpp_docker.sh fork/BRANCH, TESTS=1 with EXTRA_TARGETS (default "test-cuda-keepalive
#          test-llama-archs": the extra tests of the -c811, -c817 and -c813 rig tarballs) into M: sm_61, no AVX,
#          test-backend-ops and the extra tests in bin/
#   info   BUILDINFO.txt's commit against EXPECT; size and md5 of the tarball, md5 of every bin/ file, and which bin/ files
#          are byte for byte REF_TREE's
#   names  the build's switch list (build/common/switch-names.cpp, C770) against REF_SRC's build's: counts, names added
#   sass   cuobjdump -sass and -ptx of bin/llama-server of this build and of REF_TREE's (cuobjdump on the host, CUDA 12.9),
#          anonymous namespace hashes masked, cut into functions (the day merge's sass_attrib.py), compared function by
#          function and as files; REF_DUMPS (optional) holds REF_TREE's dumps as its own checks made them, compared with ours
# Env: M (the work root, WORK of the build), REPO (this repository as run), LOCK, SEED_SRC (a llama.cpp clone with the fork
#   as remote "fork", https), SEED_CCACHE (optional), BRANCH (default pascal-p106-v4), EXPECT (the commit BRANCH must be
#   at, in full or short), REF_TREE (the rig tree to compare with), REF_LABEL (default ref), REF_SRC (optional: REF_TREE's
#   source with its build/), REF_DUMPS (optional), CUOBJDUMP (default cuobjdump), EXTRA_TARGETS, JOBS (8), DOCKER_ARGS,
#   STEPS, ROOM_GB (52)
# Output: M/summary.tsv (step, check, verdict, detail), M/md5.txt, M/sass/, M/logs/; done when M/DONE exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
BRANCH=${BRANCH:-pascal-p106-v4}
EXPECT=${EXPECT:?EXPECT=<the commit BRANCH must be at>}
REF_TREE=${REF_TREE:?REF_TREE=<the rig tree to compare with>}
REF_LABEL=${REF_LABEL:-ref}
EXTRA_TARGETS=${EXTRA_TARGETS:-test-cuda-keepalive test-llama-archs}
CUOBJDUMP=${CUOBJDUMP:-cuobjdump}
JOBS=${JOBS:-8}
STEPS=${STEPS:-rig info names sass}
ROOM_GB=${ROOM_GB:-52}
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
ATTRIB=$REPO/docs/data/2026-11-11_v4_day_merge/run/sass_attrib.py
SRC=$M/src
MASK='s/_GLOBAL__N__[0-9a-f]{8}_/_GLOBAL__N__X_/g;s/_cu_[0-9a-f]{8}/_cu_X/g;s/_INTERNAL_[0-9a-f]{8}/_INTERNAL_X/g'
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'step\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()    { [[ " $STEPS " == *" $1 "* ]]; }
log()     { echo "$(date -Is) $*" | tee -a "$M/logs/c825.log"; }
locked()  { flock -w 43200 "$LOCK" "$@"; }
add()     { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a "$M/summary.tsv"; }
tree()    { ls -td "$M"/llama.cpp-*-"$BRANCH" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
free_gb() { df -BG --output=avail "$M" | tail -n 1 | tr -dc '0-9'; }
md5()     { md5sum < "$1" | cut -d' ' -f1; }
finish()  { log "done"; date -Is > "$M/DONE"; exit "${1:-0}"; }

if [ "$(free_gb)" -lt "$ROOM_GB" ]; then
    add start disk FAIL "under $ROOM_GB GB free ($(free_gb) GB): not started"
    finish 1
fi
if [ ! -d "$SRC/.git" ]; then
    git clone -q --local --no-checkout "${SEED_SRC:?SEED_SRC=<a llama.cpp clone with the fork as remote fork>}" "$SRC" &&
        git -C "$SRC" remote add fork "$(git -C "$SEED_SRC" remote get-url fork)" &&
        git -C "$SRC" remote set-url origin "$(git -C "$SEED_SRC" remote get-url origin)"
fi
git -C "$SRC" fetch -q fork "$BRANCH:refs/remotes/fork/$BRANCH" || { add start "$BRANCH" FAIL "no fork/$BRANCH"; finish 1; }
TIP=$(git -C "$SRC" rev-parse "fork/$BRANCH")
WANT=$(git -C "$SRC" rev-parse "$EXPECT" 2> /dev/null)
if [ "$TIP" != "$WANT" ]; then
    add start "$BRANCH" FAIL "fork/$BRANCH is at ${TIP:0:9}, not $EXPECT: not built"
    finish 1
fi
log "C825: $BRANCH at ${TIP:0:9}, steps: $STEPS, extra targets: $EXTRA_TARGETS, $(free_gb) GB free"
add start "$BRANCH" OK "tip ${TIP:0:9} ($(git -C "$SRC" log -1 --format=%s "$TIP" | cut -c1-120)); $(free_gb) GB free"

if step rig; then
    [ -d "$M/ccache" ] || [ -z "${SEED_CCACHE:-}" ] || cp -al "$SEED_CCACHE" "$M/ccache"
    log "rig build"
    locked env WORK="$M" TESTS=1 EXTRA_TARGETS="$EXTRA_TARGETS" JOBS="$JOBS" DOCKER_ARGS="$DA" nice -n 19 \
        bash "$REPO/scripts/build_llamacpp_docker.sh" "fork/$BRANCH" > "$M/logs/rig.log" 2>&1
    rc=$?
    T=$(tree)
    ok=OK
    [ $rc = 0 ] && [ -n "$T" ] && [ -x "$T/bin/llama-server" ] || ok=FAIL
    for t in $EXTRA_TARGETS; do [ -x "$T/bin/$t" ] || ok=FAIL; done
    add rig build $ok "rc=$rc; $(basename "$T"); TESTS=1 EXTRA_TARGETS=\"$EXTRA_TARGETS\" JOBS=$JOBS; $(grep -ac 'warning' "$M/logs/rig.log") warning lines"
    log "rig build rc=$rc"
    [ $ok = OK ] || finish 1
fi
T=$(tree)

if step info; then
    B=$T/BUILDINFO.txt
    c=$(sed -n 's/^commit: \([0-9a-f]*\).*/\1/p' "$B")
    add info BUILDINFO "$([ "$c" = "$TIP" ] && echo OK || echo FAIL)" \
        "$(grep -E '^(name|commit|image|built|avx|highest)' "$B" | tr '\n' ';')"
    { echo "# $(basename "$T").tar.gz"
      echo "$(stat -c %s "$T.tar.gz") bytes, md5 $(md5 "$T.tar.gz")"
      echo "# bin/ (this build, then $REF_LABEL's where it has the file)"
      for f in "$T"/bin/*; do
          r=$REF_TREE/bin/$(basename "$f")
          if [ -f "$r" ]; then rm5=$(md5 "$r"); else rm5=-; fi
          m5=$(md5 "$f")
          printf '%s\t%s\t%s\t%s\n' "$(basename "$f")" "$m5" "$rm5" "$([ "$m5" = "$rm5" ] && echo same || echo differs)"
      done; } > "$M/md5.txt"
    add info tarball OK "$(basename "$T").tar.gz $(stat -c %s "$T.tar.gz") B md5 $(md5 "$T.tar.gz")"
    for f in llama-server test-backend-ops $EXTRA_TARGETS; do
        add info "bin/$f" "$([ -x "$T/bin/$f" ] && echo OK || echo FAIL)" "md5 $(md5 "$T/bin/$f"); $REF_LABEL's: $([ -f "$REF_TREE/bin/$f" ] && md5 "$REF_TREE/bin/$f" || echo none)"
    done
    add info "bin/ files" OK "$(ls "$T/bin" | wc -l) files; byte for byte $REF_LABEL's: $(awk -F'\t' '$4 == "same" { printf "%s ", $1 }' "$M/md5.txt")"
fi

if step names; then
    L=$SRC/build/common/switch-names.cpp
    n=$(grep -c '^    "' "$L" 2> /dev/null)
    if [ -n "${REF_SRC:-}" ] && [ -f "$REF_SRC/build/common/switch-names.cpp" ]; then
        R=$REF_SRC/build/common/switch-names.cpp
        added=$(comm -13 <(grep '^    "' "$R" | tr -dc 'A-Za-z0-9_\n' | sort) <(grep '^    "' "$L" | tr -dc 'A-Za-z0-9_\n' | sort) | tr '\n' ' ')
        gone=$(comm -23 <(grep '^    "' "$R" | tr -dc 'A-Za-z0-9_\n' | sort) <(grep '^    "' "$L" | tr -dc 'A-Za-z0-9_\n' | sort) | tr '\n' ' ')
        add names "switch list" "$([ -n "$n" ] && [ -z "$gone" ] && echo OK || echo FAIL)" \
            "$n names in the build's list ($(grep -c '^    "' "$R") in $REF_LABEL's); added: ${added:-none}; gone: ${gone:-none}"
    else
        add names "switch list" "$([ -n "$n" ] && echo OK || echo FAIL)" "$n names in the build's list"
    fi
fi

if step sass; then
    S=$M/sass
    rm -rf "$S" && mkdir -p "$S"
    log "sass"
    # <label> <llama-server>: its SASS and PTX cut into functions
    dumpbin() {
        local w
        for w in sass ptx; do
            locked nice -n 19 "$CUOBJDUMP" -$w "$2" | sed -E "$MASK" | nice -n 19 python3 "$ATTRIB" dump $w > "$S/$1.$w.tsv"
        done
    }
    dumpbin tip "$T/bin/llama-server"
    dumpbin "$REF_LABEL" "$REF_TREE/bin/llama-server"
    for w in sass ptx; do
        nice -n 19 python3 "$ATTRIB" attrib "$S/tip.$w.tsv" "$S/$REF_LABEL.$w.tsv" > "$S/attrib_$w.txt" 2>&1
        same=$(cmp -s "$S/tip.$w.tsv" "$S/$REF_LABEL.$w.tsv" && echo "the same file" || echo "files differ")
        own=
        if [ -n "${REF_DUMPS:-}" ] && [ -f "$REF_DUMPS/$REF_LABEL.$w.tsv" ]; then
            own="; $REF_LABEL's own dump ($REF_DUMPS): $(cmp -s "$S/$REF_LABEL.$w.tsv" "$REF_DUMPS/$REF_LABEL.$w.tsv" && echo "the same file" || echo "differs")"
        fi
        add sass "llama-server $w against $REF_LABEL's" \
            "$(tail -n 1 "$S/attrib_$w.txt" | grep -q 'verdict: OK' && [ "$same" = "the same file" ] && echo SAME || echo DIFFERS)" \
            "$(wc -l < "$S/tip.$w.tsv") functions ($(wc -l < "$S/$REF_LABEL.$w.tsv") in $REF_LABEL's); dumps: $same$own; $(head -n 1 "$S/attrib_$w.txt"); $(sed -n '2,$p' "$S/attrib_$w.txt" | grep -v '^verdict' | sed 's/^ *//' | head -n 12 | tr '\n' ';')"
        md5sum "$S/tip.$w.tsv" "$S/$REF_LABEL.$w.tsv" | sed "s|$S/||" >> "$S/md5.txt"
    done
    log "sass done"
fi

finish 0
