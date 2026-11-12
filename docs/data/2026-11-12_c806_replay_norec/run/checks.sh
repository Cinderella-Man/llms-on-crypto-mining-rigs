#!/usr/bin/env bash
# C806 (2026-11-12): test-backend-ops' replay names the cases a record does not hold (verdict NOREC) and
# scripts/replay_checks.sh tells them apart from failing ones, on pascal-p106-v4 c96561876: built once and checked on a
# build machine, CPU only. The method is C805's (docs/data/2026-11-12_c805_kcol_exact/run/checks.sh). Every build and
# test runs in a container without GPUs or network, waits for LOCK (flock) and runs at a low CPU share
# (--cpu-shares=256, nice 19), so one runs at a time on a shared machine. Steps (STEPS, in this order by default):
#   rig     the rig tarball of BRANCH (scripts/build_llamacpp_docker.sh, TESTS=1: sm_61, no AVX, test-backend-ops in
#           bin/) in $M/main/rig, whose src/ and ccache/ it uses
#   diff    the files BRANCH changes against BASE_COMMIT (tests/test-backend-ops.cpp alone), and the CUDA sources
#           (ggml/src/ggml-cuda) against C805_COMMIT, whose checked build's GPU code is d097d6d88's (none)
#   sass    every GPU function of the new tarball's bin/llama-server (cuobjdump -sass and -ptx, anonymous-namespace
#           hashes masked, cut into functions by the day merge's sass_attrib.py) against KERN_TREE's (d097d6d88's checked
#           -kern build) and C805_TREE's (the checked -c805 build): the same file
#   cpu     test-backend-ops alone, in a static native CPU build (GGML_CUDA=OFF) of BASE_COMMIT and then of the tip, in
#           the same tree folder (ccache): $M/cpu/base and $M/cpu/tip
#   tool    run/norec_tests.sh (the new replay against the base's on a CPU, and replay_checks.sh with C806 and before
#           it, OLD_CHECKS, on both through a shim) in $M/tool
#   saved   run/saved_logs.sh (replay_checks.sh on R112's and R73's saved logs from a P106 through a stub) in $M/saved
#   setcpu  the rig row's run list (run/rig_runs.tsv: dmt-on and the two dmt-q3f runs) on the v4day set (V4DAY_TAR,
#           unpacked into $M/v4day and removed after) with the tip's CPU test-backend-ops through a shim: the cases each
#           run replays and the 30 unrecorded, R112's (the CUDA switches do nothing on a CPU, so the fault does not act)
#   c565   2026-11-05_c565_ladder4fix/run/cpu_tests.sh, static and then shared: ctest (70 tests); each build tree removed
#           after its run
#   tables  the public tables in $M/public (the work folder written as $W, the rig's folders under RIG_ROOT, its model
#           disk's mount point, as $SET and $B)
#   merged  MERGE (the --no-ff merge into pascal-p106-v4, fetched from the fork): its tree is the tip's (then this
#           chain's checks and tarball are the merge's), or the files it adds beyond the tip
# Env: M (the work root), REPO (this repository as run), LOCK, OLD_CHECKS (replay_checks.sh before C806), SEED_SRC (a
#   llama.cpp clone with the fork as remote "fork", https), SEED_CCACHE and SEED_CPU_CCACHE (optional), BRANCH (default
#   pascal-p106-v4-b12246-c806), BASE_COMMIT (c96561876 in full), C805_COMMIT (dd755a09c in full), KERN_TREE and
#   C805_TREE (the unpacked rig tarballs), R112_OUT, R112_DAY_OUT, R73_OUT (saved_logs.sh's), V4DAY_TAR (for setcpu),
#   MERGE (for merged), JOBS
#   (8), IMAGE, CPU_IMAGE, DOCKER_ARGS, STEPS
# Output: $M/summary.tsv (row, check, verdict, detail), $M/sass, $M/tool, $M/saved, $M/logs; done when $M/DONE exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
BRANCH=${BRANCH:-pascal-p106-v4-b12246-c806}
BASE_COMMIT=${BASE_COMMIT:-c96561876e1d32548bacf1df687b410214d8bb52}
C805_COMMIT=${C805_COMMIT:-dd755a09ca8b66ceb80e49cc51da98e9d2378f7a}
JOBS=${JOBS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
STEPS=${STEPS:-rig diff sass cpu tool saved c565}
ROOM_GB=${ROOM_GB:-52}
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
D=$REPO/docs/data
RUN=$D/2026-11-12_c806_replay_norec/run
ATTRIB=$D/2026-11-11_v4_day_merge/run/sass_attrib.py
SRC=$M/main/rig/src
MASK='s/_GLOBAL__N__[0-9a-f]{8}_/_GLOBAL__N__X_/g;s/_cu_[0-9a-f]{8}/_cu_X/g;s/_INTERNAL_[0-9a-f]{8}/_INTERNAL_X/g'
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'row\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
log()    { echo "$(date -Is) $*" | tee -a "$M/logs/c806.log"; }
locked() { flock -w 43200 "$LOCK" "$@"; }
add()    { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a "$M/summary.tsv"; }
tree()   { ls -td "$M"/main/rig/llama.cpp-*-"$1" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
free_gb() { df -BG --output=avail "$M" | tail -n 1 | tr -dc '0-9'; }
room() {
    local i
    for i in $(seq 1 24); do
        [ "$(free_gb)" -ge "$ROOM_GB" ] && return 0
        [ "$i" = 1 ] && log "$1 waits for $ROOM_GB GB free ($(free_gb) GB)"
        sleep 300
    done
    add "$1" disk FAIL "under $ROOM_GB GB free for 2 hours ($(free_gb) GB): stopped"
    exit 1
}
# <mount args...> -- <command...>: a CPU container, every path at its own place
# shellcheck disable=SC2086
cpubox() { locked docker run --rm $DA --network none "$@"; }

if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$M/main/rig"
    git clone -q --local --no-checkout "${SEED_SRC:?SEED_SRC=<a llama.cpp clone with the fork as remote fork>}" "$SRC" &&
        git -C "$SRC" remote add fork "$(git -C "$SEED_SRC" remote get-url fork)" &&
        git -C "$SRC" remote set-url origin "$(git -C "$SEED_SRC" remote get-url origin)"
fi
git -C "$SRC" fetch -q fork "$BRANCH:refs/remotes/fork/$BRANCH" || { echo "no fork/$BRANCH"; exit 1; }
TIP=$(git -C "$SRC" rev-parse "fork/$BRANCH")
git -C "$SRC" merge-base --is-ancestor "$BASE_COMMIT" "$TIP" || { echo "$BASE_COMMIT is not in $BRANCH"; exit 1; }
log "C806 checks: $BRANCH at ${TIP:0:9}, base ${BASE_COMMIT:0:9}, steps: $STEPS"
add start "$BRANCH" OK "tip ${TIP:0:9} ($(git -C "$SRC" log -1 --format=%s "$TIP" | cut -c1-120)); files changed against the base: $(git -C "$SRC" diff --stat "$BASE_COMMIT" "$TIP" | tail -n 1)"

if step rig; then
    [ -d "$M/main/rig/ccache" ] || [ -z "${SEED_CCACHE:-}" ] || cp -al "$SEED_CCACHE" "$M/main/rig/ccache"
    locked env WORK="$M/main/rig" TESTS=1 JOBS="$JOBS" DOCKER_ARGS="$DA" nice -n 19 bash "$REPO/scripts/build_llamacpp_docker.sh" \
        "fork/$BRANCH" > "$M/logs/rig.log" 2>&1
    rc=$?
    T=$(tree "$BRANCH")
    add rig build "$([ $rc = 0 ] && [ -n "$T" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(basename "$T").tar.gz md5 $(md5sum < "$T.tar.gz" | cut -d' ' -f1), $(stat -c %s "$T.tar.gz") B; $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';'); test-backend-ops md5 $(md5sum < "$T/bin/test-backend-ops" | cut -d' ' -f1); $(grep -ac 'warning' "$M/logs/rig.log") warning lines"
    [ -n "$T" ] && (cd "$T/bin" && md5sum ./* > "$M/main/rig/md5.txt")
    [ $rc = 0 ] || { date -Is > "$M/DONE"; exit 1; }
fi
T=$(tree "$BRANCH")

if step diff; then
    files=$(git -C "$SRC" diff --name-only "$BASE_COMMIT" "$TIP" | tr '\n' ' ')
    cuda=$(git -C "$SRC" diff --name-only "$C805_COMMIT" "$TIP" -- ggml/src/ggml-cuda | wc -l)
    add diff files "$([ "$files" = "tests/test-backend-ops.cpp " ] && [ "$cuda" = 0 ] && echo OK || echo FAIL)" \
        "against ${BASE_COMMIT:0:9}: $files($(git -C "$SRC" diff --shortstat "$BASE_COMMIT" "$TIP" | sed 's/^ //')); ggml/src/ggml-cuda against -c805 ${C805_COMMIT:0:9}: $cuda files"
fi

if step sass; then
    S=$M/sass
    rm -rf "$S" && mkdir -p "$S"
    # <label> <llama-server>: its SASS and PTX cut into functions
    dumpbin() {
        local d w
        d=$(dirname "$2")
        for w in sass ptx; do
            # shellcheck disable=SC2086
            locked docker run --rm $DA --network none -v "$d:$d:ro" "$IMAGE" nice -n 19 cuobjdump -$w "$2" \
                | sed -E "$MASK" | nice -n 19 python3 "$ATTRIB" dump $w > "$S/$1.$w.tsv"
        done
    }
    dumpbin c806 "$T/bin/llama-server"
    dumpbin kern "${KERN_TREE:?KERN_TREE=<the -kern rig tree>}/bin/llama-server"
    dumpbin c805 "${C805_TREE:?C805_TREE=<the -c805 rig tree>}/bin/llama-server"
    for b in kern c805; do
        for w in sass ptx; do
            python3 "$ATTRIB" attrib "$S/c806.$w.tsv" "$S/$b.$w.tsv" > "$S/attrib_${b}_$w.txt" 2>&1
            same=$(cmp -s "$S/c806.$w.tsv" "$S/$b.$w.tsv" && echo "the same file" || echo "files differ")
            add sass "llama-server $w against -$b" "$(tail -n 1 "$S/attrib_${b}_$w.txt" | grep -q 'verdict: OK' && [ "$same" = "the same file" ] && echo SAME || echo DIFFERS)" \
                "$(wc -l < "$S/c806.$w.tsv") functions ($(wc -l < "$S/$b.$w.tsv") in -$b's); dumps: $same; $(head -n 1 "$S/attrib_${b}_$w.txt")"
        done
    done
    log "sass done"
fi

C=$M/cpu
if step cpu; then
    [ -d "$C/ccache" ] || { if [ -n "${SEED_CPU_CCACHE:-}" ]; then cp -al "$SEED_CPU_CCACHE" "$C/ccache"; else mkdir -p "$C/ccache"; fi; }
    for b in base tip; do
        c=$([ $b = base ] && echo "$BASE_COMMIT" || echo "$TIP")
        rm -rf "$C/tree" "$C/$b" && mkdir -p "$C/tree/src" "$C/$b"
        git -C "$SRC" archive "$c" | tar -x -C "$C/tree/src"
        cpubox -v "$M:$M" -e CCACHE_DIR="$C/ccache" -e CCACHE_BASEDIR="$C/tree" "$CPU_IMAGE" bash -c '
            cmake -S '"$C/tree/src"' -B '"$C/tree/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
                -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$C/cmake-$b.log"' 2>&1 &&
            nice -n 19 cmake --build '"$C/tree/build"' -j '"$JOBS"' --target test-backend-ops > '"$C/build-$b.log"' 2>&1'
        rc=$?
        cp "$C/tree/build/bin/test-backend-ops" "$C/$b/" 2> /dev/null
        add cpu "test-backend-ops $b" "$([ $rc = 0 ] && [ -x "$C/$b/test-backend-ops" ] && echo OK || echo FAIL)" \
            "rc=$rc; commit ${c:0:9}; static, CPU only; md5 $(md5sum < "$C/$b/test-backend-ops" 2> /dev/null | cut -d' ' -f1); $(grep -ac 'warning:' "$C/build-$b.log") warning lines"
    done
    rm -rf "$C/tree"
fi

if step tool; then
    cpubox -v "$M:$M" -v "$REPO:$REPO:ro" "$CPU_IMAGE" nice -n 19 env NEW="$C/tip" OLD="$C/base" \
        CHECKS="$REPO/scripts/replay_checks.sh" OLD_CHECKS="${OLD_CHECKS:?}" \
        C372="$D/2026-11-04_replay_checks/run/cpu_tests.sh" W="$M/tool" bash "$RUN/norec_tests.sh" > "$M/logs/tool.out" 2>&1
    log "tool rc=$?"
    tail -n +2 "$M/tool/norec_tests.tsv" | awk -F'\t' 'BEGIN { OFS = "\t" } { print "tool", $1 " " $2, $3, $4 }' | tee -a "$M/summary.tsv"
fi

if step saved; then
    ro=""
    for d in "${R112_OUT:?}" "${R112_DAY_OUT:?}" "${R73_OUT:?}"; do ro="$ro -v $d:$d:ro"; done
    # shellcheck disable=SC2086
    cpubox -v "$M:$M" -v "$REPO:$REPO:ro" $ro "$CPU_IMAGE" nice -n 19 env CHECKS="$REPO/scripts/replay_checks.sh" \
        OLD_CHECKS="$OLD_CHECKS" W="$M/saved" R112_OUT="$R112_OUT" R112_DAY_OUT="$R112_DAY_OUT" \
        R112_RUNS="$D/2026-11-11_v4_day_merge/set_runs.tsv" R112_CASES="$D/2026-11-12_r112_v4day_replay/cases.tsv.gz" \
        R73_OUT="$R73_OUT" R73_RUNS="$D/2026-11-11_c705_sparse_prompt_rule/runs.tsv" \
        R73_UNREC="$D/2026-11-12_r73_qsa_prompt_replay/unrecorded.tsv" bash "$RUN/saved_logs.sh" > "$M/logs/saved.out" 2>&1
    log "saved rc=$?"
    tail -n +2 "$M/saved/saved_logs.tsv" | awk -F'\t' 'BEGIN { OFS = "\t" } { print "saved", $1 " " $2, $3, $4 }' | tee -a "$M/summary.tsv"
fi

# the rig row's run list on a CPU: the v4day set (R112's) replayed with the tip's CPU test-backend-ops through a shim
# that runs "-b CUDA0" as "-b CPU". The CUDA switches do nothing on a CPU, so dmt-q3f-pass passes and dmt-q3f-fail is
# WRONG here; what it checks is the selection: 315 and 485 cases replayed, the same 30 unrecorded as R112's
if step setcpu; then
    V4=$M/v4day
    room setcpu
    [ -n "$(find "$V4" -maxdepth 2 -name runs.tsv 2> /dev/null)" ] || {
        rm -rf "$V4" && mkdir -p "$V4" && nice -n 19 tar -C "$V4" -xf "${V4DAY_TAR:?V4DAY_TAR=<the tar of the v4day set>}"; }
    S=$(dirname "$(find "$V4" -maxdepth 2 -name runs.tsv | head -n 1)")
    O=$M/setcpu
    rm -rf "$O" && mkdir -p "$O/shim"
    printf '#!/usr/bin/env bash\n# replay_checks.sh calls test-backend-ops -b CUDA0: here the CPU backend stands in\nargs=()\nfor a in "$@"; do [ "$a" = CUDA0 ] && a=CPU; args+=("$a"); done\necho "shim: -b CUDA0 runs as -b CPU"\nexec %s/test-backend-ops "${args[@]}"\n' \
        "$C/tip" > "$O/shim/test-backend-ops"
    chmod +x "$O/shim/test-backend-ops"
    cpubox -v "$M:$M" -v "$REPO:$REPO:ro" "$CPU_IMAGE" nice -n 19 env SET="$S" RUNS_FILE="$RUN/rig_runs.tsv" BIN="$O/shim" \
        OUT="$O/replay" RUNS='dmt-on dmt-q3f-pass dmt-q3f-fail' TIMEOUT=3600 bash "$REPO/scripts/replay_checks.sh" > "$M/logs/setcpu.out" 2>&1
    # per run: cases passed, verdict and unrecorded count, and the cases the CPU backend declines: the large-activation
    # cases (big=250000) are CUDA's alone, so a CPU replays 295 of dmt's 315 and 462 of the joined runs' 485
    got=$(awk -F'\t' 'NR > 1 { print $1 ":" $4 ":" $8 "/" $10 }' "$O/replay/results.tsv" | while read -r g; do
        echo "$g:$(grep -a -c 'big=.*not supported \[CPU\]' "$O/replay/${g%%:*}.log")"; done | tr '\n' ' ')
    cut -f2 "$O/replay/unrecorded.tsv" | tail -n +2 | sort -u > "$O/ids.txt"
    ids=$(wc -l < "$O/ids.txt")
    same=$( [ -f "$M/saved/r112-kern/new/unrecorded.tsv" ] && cut -f2 "$M/saved/r112-kern/new/unrecorded.tsv" | tail -n +2 | sort -u | cmp -s - "$O/ids.txt" && echo yes || echo no)
    add setcpu "rig_runs.tsv on a CPU" "$([ "$got" = "dmt-on:295/295:ok/30:20 dmt-q3f-pass:462/462:ok/30:23 dmt-q3f-fail:462/462:WRONG/30:23 " ] && [ "$ids" = 30 ] && [ "$same" = yes ] && echo OK || echo FAIL)" \
        "run:passed:verdict/unrecorded:large-activation cases the CPU declines: $got; $ids distinct unrecorded ids, R112's 30: $same; set $(basename "$S") ($(ls "$S/cases" | wc -l) files)"
    rm -rf "$V4"
fi

if step c565; then
    for sh in OFF ON; do
        v=$([ $sh = OFF ] && echo static || echo shared)
        room "c565 $v"
        env SRC="$SRC" WORK="$M/c565-$v" TREES="c806=$TIP" STEPS='tree build ctest' SHARED=$sh JOBS="$JOBS" \
            LOCK="$LOCK" DOCKER_ARGS="--label pascal-quants=1 ${DOCKER_ARGS:-}" nice -n 19 \
            bash "$D/2026-11-05_c565_ladder4fix/run/cpu_tests.sh" > "$M/logs/c565-$v.out" 2>&1
        log "c565 $v rc=$?"
        tail -n +2 "$M/c565-$v/summary.tsv" 2> /dev/null | awk -F'\t' -v r="c565 $v" 'BEGIN { OFS = "\t" }
            NF >= 4 { d = $4; for (i = 5; i <= NF; i++) d = d " " $i; print r, $1 " " $2, $3, d; next } { print r, $1, $2, $3 }' \
            | tee -a "$M/summary.tsv"
        rm -rf "$M/c565-$v/tree"
    done
fi

if step merged; then
    git -C "$SRC" fetch -q fork pascal-p106-v4:refs/remotes/fork/pascal-p106-v4
    MC=$(git -C "$SRC" rev-parse "${MERGE:?MERGE=<the merge commit>}")
    same=$([ "$(git -C "$SRC" rev-parse "$MC^{tree}")" = "$(git -C "$SRC" rev-parse "$TIP^{tree}")" ] && echo yes || echo no)
    extra=$(git -C "$SRC" diff --name-only "$TIP" "$MC" | tr '\n' ' ')
    add merged "pascal-p106-v4 ${MC:0:9}" "$([ "$same" = yes ] && echo SAME || echo DIFFERS)" \
        "its tree is the tip's: $same; parents $(git -C "$SRC" log -1 --format=%p "$MC"); files beyond the tip: ${extra:-none}; fork/pascal-p106-v4 at $(git -C "$SRC" rev-parse --short=9 fork/pascal-p106-v4)"
fi

# the public tables in $M/public, the work folder written as $W and the rig's folders as R112's and R73's consoles
# have them ($SET, $B)
if step tables; then
    P=$M/public
    rm -rf "$P" && mkdir -p "$P/tool" "$P/saved"
    RR=${RIG_ROOT:-/nonexistent}
    anon() { sed -e "s#$M#\$W#g" -e "s#$RR/v4day#\$SET#g" -e "s#$RR/r73/qsa-prompt#\$SET#g" \
        -e "s#$RR/[^ ;]*/opt/llama-builds/#\$B/#g" -e 's#/opt/llama-builds/#$B/#g' "$@"; }
    anon "$M/summary.tsv" > "$P/checks.tsv"
    anon "$M/tool/norec_tests.tsv" > "$P/tool/norec_tests.tsv"
    anon "$M/tool/n4/verdicts.tsv" > "$P/tool/verdicts.tsv"
    anon "$M/tool/n4/runlist.tsv" > "$P/tool/runlist.tsv"
    for s in A B; do anon "$M/tool/n4/new-new-$s/console.txt" > "$P/tool/console_$s.txt"; done
    anon "$M/tool/n4/new-old-A/console.txt" > "$P/tool/console_A_old_build.txt"
    anon "$M/tool/n1-new.log" | sed 's/\x1b\[[0-9;]*m//g' | grep -a -E 'NOREC|not recorded|replay:|tests passed' > "$P/tool/n1_new_lines.txt"
    anon "$M/tool/n1-old.log" | sed 's/\x1b\[[0-9;]*m//g' | grep -a -E 'NOREC|not recorded|replay:|tests passed' > "$P/tool/n1_old_lines.txt"
    anon "$M/tool/n1-new.tsv" > "$P/tool/n1_new_replay_log.tsv"
    anon "$M/saved/saved_logs.tsv" > "$P/saved/saved_logs.tsv"
    for l in r112-kern r112-day r73; do
        anon "$M/saved/$l/compare.tsv" > "$P/saved/compare_$l.tsv"
        anon "$M/saved/$l/new/console.txt" > "$P/saved/console_$l.txt"
        anon "$M/saved/$l/new/results.tsv" > "$P/saved/results_$l.tsv"
    done
    anon "$M/saved/r112-kern/new/unrecorded.tsv" | awk -F'\t' 'NR == 1 || $1 == "dmt-on"' > "$P/saved/unrecorded_r112_dmt.tsv"
    anon "$M/saved/fake/new/console.txt" > "$P/saved/console_fake.txt"
    anon "$M/saved/fake/old/console.txt" > "$P/saved/console_fake_old_script.txt"
    if [ -d "$M/setcpu/replay" ]; then
        anon "$M/setcpu/replay/console.txt" > "$P/setcpu_console.txt"
        anon "$M/setcpu/replay/results.tsv" > "$P/setcpu_results.tsv"
    fi
    sed -e "s#$M#\$W#g" "$M/main/rig/md5.txt" > "$P/rig_bin_md5.txt" 2> /dev/null
    add tables public OK "$(find "$P" -type f | wc -l) files, $(du -sh "$P" | cut -f1); files left with a machine path: $(grep -rl -E '/(srv|mnt|home)/' "$P" | wc -l)"
fi

log "done"
date -Is > "$M/DONE"
