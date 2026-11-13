#!/usr/bin/env bash
# C822 (Q1436) on a build machine, CPU only, no GPU: GGML_CUDA_MMVF_F32_MT_ROWS=auto (48 rows at 4 tokens, 64 at 5-8)
# or a list of limits from 4 tokens on sets GGML_CUDA_MMVF_F32_MT's row limit per token count; a single number keeps
# C807's meaning and the default stays 32. Host code only. From C807's chain.sh. The steps, in order:
#   build   the rig tarball of fork/BRANCH with test-backend-ops and test-cuda-mmvf-f32-rows (scripts/build_llamacpp_docker.sh,
#           TESTS=1, EXTRA_TARGETS) in WORK/w
#   code    the branch's GPU code against BASE's, function by function: cuobjdump -sass and -ptx of llama-server and
#           test-backend-ops, compared by C103's funcs.py (every function the same, none new: the change is host code)
#   rows    the rig build's test-cuda-mmvf-f32-rows (the value's limits and the switch's decision: every single number
#           and the value unset against C807's rule, auto, lists, Qwen3.8-Flash-Next's shapes), in a container without GPUs
#   cpu     C565's cpu_tests.sh, static as the rig builds: BASE_REF's test-llama-archs writes every architecture's
#           logits, the branch's tree builds every target, runs ctest (test-cuda-mmvf-f32-rows among them) and compares
#           its logits (archs, ident)
#   shared  C565's cpu_tests.sh on the branch with shared libraries (upstream's default): every target, ctest, archs
#   record  the replay set f32x (scripts/record_checks.sh) with the rig build's test-backend-ops, in WORK/sets
#   select  every run of the set: the cases its filter selects, and of them how many the run's row limit lets the
#           switch take (its _ROWS: unset 32, a number, auto = 48 at 4 tokens and 64 at 5-8, or a list from 4 tokens on;
#           read here independently of the engine's code)
#   replay  the set's cases replayed on the CPU with the rig build (every case passes: the set's files are whole)
#   merged  with MERGED (the merge commit into the line, in WORK/w/src; with MERGE_BASE, the line's tip it merges onto):
#           every switch off on the merged tree, static: MERGE_BASE's test-llama-archs writes every architecture's
#           logits and the merged tree's compares (cpu_tests.sh, test-llama-archs and test-cuda-mmvf-f32-rows only, in
#           WORK/merged), then the merged tree's test-cuda-mmvf-f32-rows runs
#   tables  WORK/summary.tsv: a line per check
# Every step that builds or tests takes LOCK, so one runs at a time on a shared machine, at a low CPU share
# (--cpu-shares=256, nice 19), in containers without GPUs.
# Env: WORK (a folder of its own), PUB (a checkout of this repository), BASE (a rig tarball whose GPU code is the base's),
#   BASE_REF (the base commit), BRANCH (default pascal-p106-v4-b12246-c822), STEPS (default all, in the order above), LOCK
#   (a flock file, optional), JOBS (default 8), DOCKER_ARGS (e.g. labels), IMAGE (default nvidia/cuda:12.9.2-devel-ubuntu24.04),
#   CPU_IMAGE (default pq-c565-cpu:24.04, which cpu_tests.sh makes), SEED_SRC and SEED_CCACHE (optional: a llama.cpp
#   clone with the fork as remote "fork", and a build's ccache, copied into WORK/w before the first build so that it
#   fetches and compiles only what differs), SEED_CPU_CCACHE (optional: a ccache of cpu_tests.sh's trees)
# Output: WORK/chain.log (a line per step), WORK/build.log, WORK/code/, WORK/rows/, WORK/cpu/, WORK/shared/, WORK/sets/,
#   WORK/select.tsv, WORK/replay/, WORK/merged/, WORK/summary.tsv; WORK/DONE when it ends
set -uo pipefail
WORK=$(realpath -m "${WORK:?WORK=<dir>}")
PUB=$(realpath "${PUB:?PUB=<this repository>}")
BRANCH=${BRANCH:-pascal-p106-v4-b12246-c822}
STEPS=${STEPS:-build code rows cpu shared record select replay merged tables}
JOBS=${JOBS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
RT=test-cuda-mmvf-f32-rows
D="--cpu-shares=256 ${DOCKER_ARGS:-}"
mkdir -p "$WORK/w"
rm -f "$WORK/DONE"
step() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date -Is) $*" | tee -a "$WORK/chain.log"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
tarball() { ls -td "$WORK"/w/llama.cpp-*-"$BRANCH" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
me="$(id -u):$(id -g)"
# the rig build links the CUDA driver library, which a container without GPUs lacks: the toolkit's stub stands in
STUB='s=/usr/local/cuda/lib64/stubs/libcuda.so; if [ -e $s ]; then mkdir -p /tmp/stub && ln -sf $s /tmp/stub/libcuda.so.1;
      export LD_LIBRARY_PATH=/tmp/stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; fi'
# F32's 2-D products of 1-8 tokens: the set's filter (scripts/record_checks.sh, F32X)
F32X='type_a=f32,type_b=f32,m=[0-9]+,n=[1-8],k=[0-9]+,bs=\[1,1\],nr=\[1,1\],'

if step build; then
    [ -d "$WORK/w/src/.git" ] || [ -z "${SEED_SRC:-}" ] || cp -a "$SEED_SRC" "$WORK/w/src"
    [ -d "$WORK/w/ccache" ] || [ -z "${SEED_CCACHE:-}" ] || cp -a "$SEED_CCACHE" "$WORK/w/ccache"
    log "build fork/$BRANCH"
    # shellcheck disable=SC2086
    locked env TESTS=1 EXTRA_TARGETS="$RT" JOBS="$JOBS" WORK="$WORK/w" DOCKER_ARGS="$D" nice -n 19 \
        bash "$PUB/scripts/build_llamacpp_docker.sh" "fork/$BRANCH" > "$WORK/build.log" 2>&1
    rc=$?
    T=$(tarball)
    log "build rc=$rc: $(basename "${T:-none}").tar.gz $([ -n "$T" ] && stat -c %s "$T.tar.gz") B md5 $([ -n "$T" ] && md5sum < "$T.tar.gz" | cut -d' ' -f1); $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" 2> /dev/null | tr '\n' ' ')"
    [ $rc = 0 ] || { touch "$WORK/DONE"; exit 1; }
fi
T=$(tarball)
BIN=${T:+$T/bin}
SHA=$(git -C "$WORK/w/src" rev-parse "fork/$BRANCH")

if step code; then
    # every function of the base must be the same, and none new: the change is host code
    : "${BASE:?BASE=<the base rig tarball>}"
    C=$WORK/code
    mkdir -p "$C/base"
    b=$(tar tzf "$BASE" | head -n 1 | cut -d/ -f1)
    [ -x "$C/base/$b/bin/test-backend-ops" ] || tar xzf "$BASE" -C "$C/base" "$b/bin/llama-server" "$b/bin/test-backend-ops"
    # shellcheck disable=SC2086
    locked docker run --rm $D --network none -v "$WORK:$WORK" "$IMAGE" bash -c '
        for x in llama-server test-backend-ops; do
            for w in sass ptx; do
                nice -n 19 cuobjdump -$w "$1/base/$2/bin/$x" > "$1/base.$x.$w"
                nice -n 19 cuobjdump -$w "$3/$x" > "$1/branch.$x.$w"
            done
        done
        chown -R "$4" "$1"' sh "$C" "$b" "$BIN" "$me" > "$C/code.log" 2>&1
    for x in llama-server test-backend-ops; do
        for w in sass ptx; do
            echo "== $x $w" >> "$C/code.log"
            nice -n 19 python3 "$PUB/docs/data/2026-11-11_c103_union_gather/run/funcs.py" "$w" "$C/base.$x.$w" "$C/branch.$x.$w" \
                >> "$C/code.log" 2>&1
        done
    done
    log "code: $(grep -c '^verdict: SAME' "$C/code.log") of 4 listings with every base function the same, $(grep -c '^verdict: DIFFERS' "$C/code.log") with one changed; new: $(grep '^  new: ' "$C/code.log" | sort -u | sed 's/^  new: //' | tr '\n' ' ')"
    rm -f "$C"/*.sass "$C"/*.ptx
fi

run_rows() { # <bin dir> <out file>: test-cuda-mmvf-f32-rows in a container without GPUs
    mkdir -p "$(dirname "$2")"
    # shellcheck disable=SC2086
    locked docker run --rm $D --network none -v "$1:$1:ro" "$IMAGE" sh -c "$STUB"'
        nice -n 19 '"$1/$RT"'; echo "exit $?"' > "$2" 2>&1
}

if step rows; then
    run_rows "$BIN" "$WORK/rows/rig.log"
    log "rows (rig build): $(grep -a '^test-cuda-mmvf-f32-rows:' "$WORK/rows/rig.log"); $(tail -n 1 "$WORK/rows/rig.log")"
fi

if step cpu; then
    : "${BASE_REF:?BASE_REF=<the base commit>}"
    [ -d "$WORK/cpu/ccache" ] || [ -z "${SEED_CPU_CCACHE:-}" ] || { mkdir -p "$WORK/cpu" && cp -a "$SEED_CPU_CCACHE" "$WORK/cpu/ccache"; }
    log "cpu: base $BASE_REF (test-llama-archs), c822 $SHA (every target, ctest, archs, ident), static"
    # the whole script under LOCK (its own LOCK covers only its builds, and a second flock on the file would wait on this one)
    # shellcheck disable=SC2086
    locked env WORK="$WORK/cpu" SRC="$WORK/w/src" TREES="base=$BASE_REF:test-llama-archs c822=$SHA@base" \
        STEPS='tree build ctest archs ident' SHARED=OFF JOBS="$JOBS" LOCK= DOCKER_ARGS="${DOCKER_ARGS:-}" IMAGE="$CPU_IMAGE" \
        nice -n 19 bash "$PUB/docs/data/2026-11-05_c565_ladder4fix/run/cpu_tests.sh" > "$WORK/cpu.out" 2>&1
    log "cpu rc=$?: $(grep -v '^tree	check' "$WORK/cpu/summary.tsv" | cut -f 1-4 | tr '\t\n' ' ;')"
    log "cpu ctest $RT: $(grep -a "$RT" "$WORK/cpu/c822/ctest.log" | sed 's/  */ /g' | tr '\n' ' ')"
fi

if step shared; then
    [ -d "$WORK/shared/ccache" ] || [ -z "${SEED_CPU_CCACHE:-}" ] || { mkdir -p "$WORK/shared" && cp -a "$SEED_CPU_CCACHE" "$WORK/shared/ccache"; }
    log "shared: c822 $SHA (every target, ctest, archs), shared libraries"
    # shellcheck disable=SC2086
    locked env WORK="$WORK/shared" SRC="$WORK/w/src" TREES="c822=$SHA" STEPS='tree build ctest archs' SHARED=ON JOBS="$JOBS" \
        LOCK= DOCKER_ARGS="${DOCKER_ARGS:-}" IMAGE="$CPU_IMAGE" \
        nice -n 19 bash "$PUB/docs/data/2026-11-05_c565_ladder4fix/run/cpu_tests.sh" > "$WORK/shared.out" 2>&1
    log "shared rc=$?: $(grep -v '^tree	check' "$WORK/shared/summary.tsv" | cut -f 1-4 | tr '\t\n' ' ;')"
    log "shared ctest $RT: $(grep -a "$RT" "$WORK/shared/c822/ctest.log" | sed 's/  */ /g' | tr '\n' ' ')"
fi

if step record; then
    mkdir -p "$WORK/sets"
    log "record f32x"
    locked env OUT="$WORK/sets" SUITE=f32x BIN="$BIN" JOBS="$JOBS" DOCKER_ARGS="${DOCKER_ARGS:-}" \
        nice -n 19 bash "$PUB/scripts/record_checks.sh" > "$WORK/sets/record.console.txt" 2>&1
    log "record rc=$?: $(grep -E 'manifest:|tar:' "$WORK/sets/f32x.log" | sed 's/^[^ ]* //' | tr '\n' ' ')"
fi

if step select; then
    # the cases each run's filter selects among the set's (test-backend-ops searches the case string for the filter as a
    # regular expression; the manifest holds one case a line: id, case string), and of them, at 4-8 tokens, those the
    # run's row limit lets the switch take and those it leaves (a run without the switch takes none)
    S=$WORK/sets/f32x
    printf 'run\tswitches\texpect\tcases\ttaken\tleft\trow_counts\n' > "$WORK/select.tsv"
    while IFS=$'\t' read -r run sw _op filt expect _known; do
        [ "$sw" = - ] && sw=
        sel=$(cut -f 2 "$S/manifest.tsv" | grep -E -- "$filt")
        n=$(printf '%s\n' "$sel" | grep -c 'MUL_MAT')
        on=0
        [[ " $sw " =~ \ GGML_CUDA_MMVF_F32_MT=[12]\  ]] && on=1
        rows=$(printf '%s\n' "$sw" | tr ' ' '\n' | sed -n 's/^GGML_CUDA_MMVF_F32_MT_ROWS=//p')
        set=$(printf '%s\n' "$sw" | tr ' ' '\n' | grep -c '^GGML_CUDA_MMVF_F32_MT_ROWS=')
        counts=$(printf '%s\n' "$sel" | sed -n 's/.*,m=\([0-9]*\),n=\([0-9]*\),.*/\1 \2/p' | awk -v on=$on -v r="$rows" -v set="$set" '
            BEGIN {
                # the limit at 4..8 tokens: unset 32; auto 48,64; else a list read as atoi reads each entry, the last
                # one holding after it; 0 or below: any row count
                if (!set) r = "32"
                if (r == "auto") r = "48,64"
                k = split(r, e, ",")
                for (t = 4; t <= 8; t++) { i = t - 3; if (i <= k) last = e[i] + 0; lim[t] = last }
            }
            { n = $2 + 0; m = $1 + 0
              if (on && n >= 4 && n <= 8 && (lim[n] <= 0 || m <= lim[n])) take++; else left++ }
            END { printf "%d\t%d", take, left }')
        rc=$(printf '%s\n' "$sel" | sed -n 's/.*,m=\([0-9]*\),n=\([0-9]*\),.*/\1x\2/p' | sort -t x -k1,1n -k2,2n | uniq -c | awk '{ printf "%s%s:%s", s, $2, $1; s = " " }')
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$run" "$sw" "$expect" "$n" "$counts" "$rc" >> "$WORK/select.tsv"
    done < "$S/runs.tsv"
    log "select: $(tail -n +2 "$WORK/select.tsv" | awk -F'\t' '{ printf "%s %s cases (%s taken, %s left); ", $1, $4, $5, $6 }')"
fi

if step replay; then
    SET=$WORK/sets/f32x
    R=$WORK/replay
    mkdir -p "$R"
    rm -f "$R/cpu.tsv"
    # shellcheck disable=SC2086
    locked docker run --rm $D --network none -v "$BIN:$BIN:ro" -v "$SET:$SET:ro" -v "$R:/k" "$IMAGE" sh -c "$STUB"'
        nice -n 19 '"$BIN"'/test-backend-ops -b CPU -o MUL_MAT -p "'"$F32X"'" --replay '"$SET"' --replay-log /k/cpu.tsv \
            > /k/cpu.log 2>&1
        echo "exit $?" >> /k/cpu.log
        chown '"$me"' /k/cpu.log /k/cpu.tsv' > "$R/cpu.docker" 2>&1
    log "replay cpu: $(tail -n 1 "$R/cpu.log"); cases $(wc -l < "$R/cpu.tsv"), OK $(cut -f 3 "$R/cpu.tsv" | grep -c '^OK$'), FAIL $(cut -f 3 "$R/cpu.tsv" | grep -c '^FAIL$'); $(grep -a -m 1 'tests passed' "$R/cpu.log" | sed 's/\x1b\[[0-9;]*m//g; s/^ *//')"
fi

if step merged && [ -n "${MERGED:-}" ]; then
    : "${MERGE_BASE:?MERGE_BASE=<the tip of the line the merge goes onto>}"
    [ -d "$WORK/merged/ccache" ] || [ ! -d "$WORK/cpu/ccache" ] || { mkdir -p "$WORK/merged" && cp -a "$WORK/cpu/ccache" "$WORK/merged/ccache"; }
    log "merged: $MERGED on $MERGE_BASE, every switch off (test-llama-archs), static"
    # shellcheck disable=SC2086
    locked env WORK="$WORK/merged" SRC="$WORK/w/src" KEEP_BUILD=1 \
        TREES="base=$MERGE_BASE:test-llama-archs merged=$MERGED:test-llama-archs,$RT@base" \
        STEPS='tree build archs ident' SHARED=OFF JOBS="$JOBS" LOCK= DOCKER_ARGS="${DOCKER_ARGS:-}" IMAGE="$CPU_IMAGE" \
        nice -n 19 bash "$PUB/docs/data/2026-11-05_c565_ladder4fix/run/cpu_tests.sh" > "$WORK/merged.out" 2>&1
    log "merged rc=$?: $(grep -v '^tree	check' "$WORK/merged/summary.tsv" | cut -f 1-4 | tr '\t\n' ' ;')"
    MB=$(find "$WORK/merged" -path '*/bin/*' -name "$RT" -type f 2> /dev/null | head -n 1)
    if [ -n "$MB" ]; then
        # shellcheck disable=SC2086
        locked docker run --rm $D --network none -v "$WORK:$WORK:ro" "$CPU_IMAGE" sh -c "nice -n 19 $MB; echo \"exit \$?\"" \
            > "$WORK/merged/rows.log" 2>&1
    else
        echo "no $RT built in $WORK/merged" > "$WORK/merged/rows.log"
    fi
    log "merged rows: $(grep -a '^test-cuda-mmvf-f32-rows:' "$WORK/merged/rows.log"); $(tail -n 1 "$WORK/merged/rows.log")"
fi

if step tables; then
    rows_line() { # <log>
        printf '%s\t%s\n' "$(grep -aq '^test-cuda-mmvf-f32-rows: [0-9]* checks, 0 failed' "$1" 2> /dev/null && tail -n 1 "$1" | grep -q '^exit 0$' && echo OK || echo FAIL)" \
            "$(grep -a '^test-cuda-mmvf-f32-rows:' "$1" 2> /dev/null | sed 's/^test-cuda-mmvf-f32-rows: //'); $(tail -n 1 "$1" 2> /dev/null)"
    }
    {
        printf 'check\tverdict\tdetail\n'
        printf 'build\t%s\t%s\n' "$([ -n "$T" ] && echo OK || echo FAIL)" \
            "$(basename "${T:-none}").tar.gz $([ -n "$T" ] && stat -c %s "$T.tar.gz") B md5 $([ -n "$T" ] && md5sum < "$T.tar.gz" | cut -d' ' -f1); llama-server md5 $(md5sum < "$BIN/llama-server" | cut -d' ' -f1), test-backend-ops md5 $(md5sum < "$BIN/test-backend-ops" | cut -d' ' -f1), $RT md5 $(md5sum < "$BIN/$RT" | cut -d' ' -f1)"
        printf 'code\t%s\t%s\n' "$([ "$(grep -c '^verdict: SAME' "$WORK/code/code.log")" = 4 ] && echo SAME || echo DIFFERS)" \
            "$(grep -c '^verdict: SAME' "$WORK/code/code.log") of 4 listings (llama-server, test-backend-ops; SASS, PTX) with every base function the same; new: $(grep '^  new: ' "$WORK/code/code.log" | sort -u | sed 's/^  new: //' | tr '\n' ' ')"
        printf 'rows-rig\t%s\n' "$(rows_line "$WORK/rows/rig.log")"
        grep -v '^tree	check' "$WORK/cpu/summary.tsv" 2> /dev/null | awk -F'\t' -v OFS='\t' '{ print "cpu-" $1 "-" $2, $3, $4 }'
        grep -v '^tree	check' "$WORK/shared/summary.tsv" 2> /dev/null | awk -F'\t' -v OFS='\t' '{ print "shared-" $1 "-" $2, $3, $4 }'
        grep -v '^tree	check' "$WORK/merged/summary.tsv" 2> /dev/null | awk -F'\t' -v OFS='\t' '{ print "merged-" $1 "-" $2, $3, $4 }'
        [ -f "$WORK/merged/rows.log" ] && printf 'rows-merged\t%s\n' "$(rows_line "$WORK/merged/rows.log")"
        printf 'record\t%s\t%s cases, %s runs; tar %s B md5 %s\n' "$([ -s "$WORK/sets/f32x.tar.md5" ] && echo OK || echo FAIL)" \
            "$(wc -l < "$WORK/sets/f32x/manifest.tsv")" "$(wc -l < "$WORK/sets/f32x/runs.tsv")" \
            "$(stat -c %s "$WORK/sets/f32x.tar" 2> /dev/null)" "$(cat "$WORK/sets/f32x.tar.md5" 2> /dev/null)"
        # every run selects cases; every fault run that must fail selects only cases its limit lets the switch take, and
        # every fault run that must pass with the switch on (-above) only cases its limit leaves to cuBLAS
        printf 'select\t%s\t%s\n' \
            "$(awk -F'\t' 'NR > 1 && $4 == 0 { bad = 1 }
                NR > 1 && $2 ~ /_FAULT=1/ && $3 == "fail" && ($6 != 0 || $5 == 0) { bad = 1 }
                NR > 1 && $1 ~ /-above$/ && ($5 != 0 || $6 == 0) { bad = 1 }
                END { print (NR > 1 && !bad) ? "OK" : "CHECK" }' "$WORK/select.tsv")" \
            "$(tail -n +2 "$WORK/select.tsv" | awk -F'\t' '{ printf "%s %s (%s, %s/%s); ", $1, $4, $3, $5, $6 }')"
        printf 'replay-cpu\t%s\t%s cases, %s OK, %s FAIL; %s\n' \
            "$([ "$(cut -f 3 "$WORK/replay/cpu.tsv" | grep -c '^FAIL$')" = 0 ] && [ -s "$WORK/replay/cpu.tsv" ] && echo OK || echo FAIL)" \
            "$(wc -l < "$WORK/replay/cpu.tsv")" "$(cut -f 3 "$WORK/replay/cpu.tsv" | grep -c '^OK$')" \
            "$(cut -f 3 "$WORK/replay/cpu.tsv" | grep -c '^FAIL$')" "$(tail -n 1 "$WORK/replay/cpu.log")"
    } > "$WORK/summary.tsv"
    log "tables: $(cut -f 1,2 "$WORK/summary.tsv" | tail -n +2 | tr '\t\n' ' ;')"
fi

touch "$WORK/DONE"
