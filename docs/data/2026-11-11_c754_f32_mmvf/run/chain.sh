#!/usr/bin/env bash
# C754 (Q1330) on a build machine, CPU only, no GPU: GGML_CUDA_MMVF_F32_MT=1 sends F32 weight products of 4-8 tokens to
# llama.cpp's mul_mat_vec_f on compute 6.1 instead of cuBLAS (Qwen3.8-Flash-Next's shared-expert gate and router in a
# verify pass), each token with the one-token kernel's sums. The steps, in order:
#   emul    run/order_emul.c: mul_mat_vec_f's order at 2-8 tokens against its one-token launches, every output (and
#           three controls that must differ: a sequential order, another block size, the fault)
#   build   the rig tarball of fork/BRANCH with test-backend-ops (scripts/build_llamacpp_docker.sh, TESTS=1) in WORK/w
#   res     cuobjdump -res-usage and -sass of the build's mmvf.cu.o, read by run/sass_f32.sh: every F32 instantiation of
#           mul_mat_vec_f (1-8 tokens): registers, spills, FFMA, FMUL and FADD, and the loads and FMAs of every loop
#           (OK: no FMUL, the butterflies' FADDs only, every loop one load per pair of weights and per token's pair of
#           activations with two FMAs per token)
#   code    the branch's GPU code against BASE's, kernel by kernel: cuobjdump -sass and -ptx of llama-server and
#           test-backend-ops, compared by C103's funcs.py (every kernel the same, none new: the change is host code)
#   cpu     C565's cpu_tests.sh, static as the rig builds: BASE_REF's test-llama-archs writes every architecture's logits
#           (tree base, that target only), the branch's tree builds every target, runs ctest and compares its logits
#           (archs, ident)
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid (mode f32) linked against the rig build's tree, for the rig
#   record  the replay set f32x (scripts/record_checks.sh) with the rig build's test-backend-ops, in WORK/sets
#   replay  the set's cases replayed on the CPU with the rig build (every case passes: the set's files are whole)
#   tables  WORK/summary.tsv: a line per check
# Every step that builds or tests takes LOCK, so one runs at a time on a shared machine, at a low CPU share
# (--cpu-shares=256, nice 19), in containers without GPUs.
# Env: WORK (a folder of its own), PUB (a checkout of this repository), BASE (the base build's rig tarball), BASE_REF
#   (its commit), BRANCH (default pascal-p106-v4-b12246-c754), STEPS (default all, in the order above), LOCK (a flock
#   file, optional), JOBS (default 8), DOCKER_ARGS (e.g. labels), IMAGE (default nvidia/cuda:12.9.2-devel-ubuntu24.04),
#   CPU_IMAGE (default pq-c565-cpu:24.04, which cpu_tests.sh makes), SEED_SRC and SEED_CCACHE (optional: a llama.cpp
#   clone with the fork as remote "fork", and the base build's ccache, copied into WORK/w before the first build so that
#   it fetches and compiles only what differs), SEED_CPU_CCACHE (optional: a ccache of cpu_tests.sh's static trees)
# Output: WORK/chain.log (a line per step), WORK/order_emul.tsv, WORK/build.log, WORK/sass/, WORK/sass_f32.tsv,
#   WORK/code/, WORK/cpu/, WORK/bitid/, WORK/sets/, WORK/replay/, WORK/summary.tsv; WORK/DONE when it ends
set -uo pipefail
WORK=$(realpath -m "${WORK:?WORK=<dir>}")
PUB=$(realpath "${PUB:?PUB=<this repository>}")
BRANCH=${BRANCH:-pascal-p106-v4-b12246-c754}
STEPS=${STEPS:-emul build res code cpu bitid record replay tables}
JOBS=${JOBS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
D="--cpu-shares=256 ${DOCKER_ARGS:-}"
HERE=$(cd "$(dirname "$0")" && pwd)
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

if step emul; then
    mkdir -p "$WORK/emul"
    cp "$HERE/order_emul.c" "$WORK/emul/"
    # shellcheck disable=SC2086
    locked docker run --rm $D --network none -v "$WORK/emul:/e" -w /e "$IMAGE" sh -c '
        gcc -O2 -ffp-contract=off -o order_emul order_emul.c -lm && nice -n 19 ./order_emul > order_emul.tsv 2> order_emul.err
        rc=$?; chown '"$me"' order_emul order_emul.tsv order_emul.err; exit $rc' > "$WORK/emul/run.log" 2>&1
    rc=$?
    cp "$WORK/emul/order_emul.tsv" "$WORK/order_emul.tsv" 2> /dev/null
    log "emul rc=$rc: $(cat "$WORK/emul/order_emul.err" 2> /dev/null); $(awk -F'\t' 'NR > 1 { printf "%s T=%s %s-%s %s of %s differ; ", $1, $4, $5, $6, $8, $7 }' "$WORK/order_emul.tsv" 2> /dev/null)"
fi

if step build; then
    [ -d "$WORK/w/src/.git" ] || [ -z "${SEED_SRC:-}" ] || cp -a "$SEED_SRC" "$WORK/w/src"
    [ -d "$WORK/w/ccache" ] || [ -z "${SEED_CCACHE:-}" ] || cp -a "$SEED_CCACHE" "$WORK/w/ccache"
    log "build fork/$BRANCH"
    # shellcheck disable=SC2086
    locked env TESTS=1 JOBS="$JOBS" WORK="$WORK/w" DOCKER_ARGS="$D" nice -n 19 bash "$PUB/scripts/build_llamacpp_docker.sh" \
        "fork/$BRANCH" > "$WORK/build.log" 2>&1
    rc=$?
    T=$(tarball)
    log "build rc=$rc: $(basename "${T:-none}").tar.gz $([ -n "$T" ] && stat -c %s "$T.tar.gz") B md5 $([ -n "$T" ] && md5sum < "$T.tar.gz" | cut -d' ' -f1); $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" 2> /dev/null | tr '\n' ' ')"
    [ $rc = 0 ] || { touch "$WORK/DONE"; exit 1; }
fi
T=$(tarball)
BIN=${T:+$T/bin}
SHA=$(git -C "$WORK/w/src" rev-parse "fork/$BRANCH")

if step res; then
    S=$WORK/sass
    O=$WORK/w/src/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvf.cu.o
    mkdir -p "$S"
    # shellcheck disable=SC2086
    locked docker run --rm $D --network none -v "$WORK:$WORK" "$IMAGE" sh -c '
        nice -n 19 cuobjdump -res-usage "$1" > "$2/mmvf_res.txt" && nice -n 19 cuobjdump -sass "$1" > "$2/mmvf.sass"
        rc=$?; chown -R "$3" "$2"; exit $rc' sh "$O" "$S" "$me" > "$S/cuobjdump.log" 2>&1
    md5sum < "$O" | cut -d' ' -f1 > "$S/mmvf.cu.o.md5"
    SASS=$S bash "$HERE/sass_f32.sh" > "$WORK/sass_f32.tsv" 2> "$S/sass_f32.err"
    log "res rc=$?: $(awk -F'\t' 'NR > 1 && $3 == 0 && $4 == 0 && $2 == 256 { printf "T=%s %s regs stack %s FFMA %s FMUL %s FADD %s/%s SHFL %s loops %s ok %s; ", $1, $5, $6, $8, $9, $10, $11, $12, $14, $17 }' "$WORK/sass_f32.tsv")"
fi

if step code; then
    # every kernel of the base must be the same, and none new: the change is host code
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
    log "code: $(grep -c '^verdict: SAME' "$C/code.log") of 4 listings with every base kernel the same, $(grep -c '^verdict: DIFFERS' "$C/code.log") with one changed; new: $(grep '^  new: ' "$C/code.log" | sort -u | sed 's/^  new: //' | tr '\n' ' ')"
    rm -f "$C"/*.sass "$C"/*.ptx
fi

if step cpu; then
    : "${BASE_REF:?BASE_REF=<the base commit>}"
    [ -d "$WORK/cpu/ccache" ] || [ -z "${SEED_CPU_CCACHE:-}" ] || { mkdir -p "$WORK/cpu" && cp -a "$SEED_CPU_CCACHE" "$WORK/cpu/ccache"; }
    log "cpu: base $BASE_REF (test-llama-archs), c754 $SHA (every target, ctest, archs, ident), static"
    # the whole script under LOCK (its own LOCK covers only its builds, and a second flock on the file would wait on this one)
    # shellcheck disable=SC2086
    locked env WORK="$WORK/cpu" SRC="$WORK/w/src" TREES="base=$BASE_REF:test-llama-archs c754=$SHA@base" \
        STEPS='tree build ctest archs ident' SHARED=OFF JOBS="$JOBS" LOCK= DOCKER_ARGS="${DOCKER_ARGS:-}" IMAGE="$CPU_IMAGE" \
        nice -n 19 bash "$PUB/docs/data/2026-11-05_c565_ladder4fix/run/cpu_tests.sh" > "$WORK/cpu.out" 2>&1
    log "cpu rc=$?: $(grep -v '^tree	check' "$WORK/cpu/summary.tsv" | cut -f 1-4 | tr '\t\n' ' ;')"
fi

if step bitid; then
    mkdir -p "$WORK/bitid"
    # shellcheck disable=SC2086
    locked env DOCKER_ARGS="$D" nice -n 19 bash "$PUB/bench/mmvq_pascal/verify/build_multitoken_bitid.sh" "$WORK/w/src" "$WORK/w/src" \
        > "$WORK/bitid/build.log" 2>&1
    rc=$?
    cp "$PUB/bench/mmvq_pascal/build/multitoken_bitid" "$WORK/bitid/" 2> /dev/null
    log "bitid rc=$rc: md5 $(md5sum < "$WORK/bitid/multitoken_bitid" 2> /dev/null | cut -d' ' -f1)"
fi

if step record; then
    mkdir -p "$WORK/sets"
    log "record f32x"
    locked env OUT="$WORK/sets" SUITE=f32x BIN="$BIN" JOBS="$JOBS" DOCKER_ARGS="${DOCKER_ARGS:-}" \
        nice -n 19 bash "$PUB/scripts/record_checks.sh" > "$WORK/sets/record.console.txt" 2>&1
    log "record rc=$?: $(grep -E 'manifest:|tar:' "$WORK/sets/f32x.log" | sed 's/^[^ ]* //' | tr '\n' ' ')"
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

if step tables; then
    {
        printf 'check\tverdict\tdetail\n'
        e=$WORK/order_emul.tsv
        printf 'emul\t%s\t%s\n' \
            "$(awk -F'\t' 'NR > 1 && $6 == "mt" && $8 != 0 { bad = 1 } NR > 1 && $6 != "mt" && $8 == 0 { bad = 1 }
                END { print (NR > 1 && !bad) ? "OK" : "FAIL" }' "$e" 2> /dev/null)" \
            "$(awk -F'\t' 'NR > 1 { printf "%s T=%s %s-%s: %s of %s differ; ", $1, $4, $5, $6, $8, $7 }' "$e" 2> /dev/null)"
        printf 'build\t%s\t%s\n' "$([ -n "$T" ] && echo OK || echo FAIL)" \
            "$(basename "${T:-none}").tar.gz $([ -n "$T" ] && stat -c %s "$T.tar.gz") B md5 $([ -n "$T" ] && md5sum < "$T.tar.gz" | cut -d' ' -f1)"
        # the unfused instantiations: no stack, no FMUL, the butterflies' FADDs only, every loop the sums only; the fused
        # one-token kernel (its epilogue has FMULs and FADDs of its own): no stack, every loop the sums only
        printf 'res\t%s\t%s\n' \
            "$(awk -F'\t' 'NR > 1 { n++; if ($6 != 0 || $13 != 0 || $14 != $17 || ($3 == 0 && ($9 != 0 || $10 != $11))) bad = 1 } END { print (n && !bad) ? "OK" : "CHECK" }' "$WORK/sass_f32.tsv")" \
            "$(awk -F'\t' 'NR > 1 && $3 == 0 { n++; if ($6 == 0 && $9 == 0 && $10 == $11 && $13 == 0 && $14 == $17) k++ }
                NR > 1 && $3 == 1 { m++; if ($6 == 0 && $13 == 0 && $14 == $17) j++ }
                END { printf "%d of %d unfused instantiations: no stack, no FMUL, the butterflies FADDs only, every loop the sums only; %d of %d fused: no stack, every loop the sums only", k, n, j, m }' "$WORK/sass_f32.tsv")"
        printf 'code\t%s\t%s\n' "$([ "$(grep -c '^verdict: SAME' "$WORK/code/code.log")" = 4 ] && echo SAME || echo DIFFERS)" \
            "new: $(grep '^  new: ' "$WORK/code/code.log" | sort -u | sed 's/^  new: //' | tr '\n' ' ')"
        grep -v '^tree	check' "$WORK/cpu/summary.tsv" 2> /dev/null | awk -F'\t' -v OFS='\t' '{ print "cpu-" $1 "-" $2, $3, $4 }'
        printf 'bitid\t%s\tmd5 %s\n' "$([ -x "$WORK/bitid/multitoken_bitid" ] && echo OK || echo FAIL)" \
            "$(md5sum < "$WORK/bitid/multitoken_bitid" 2> /dev/null | cut -d' ' -f1)"
        printf 'record\t%s\t%s cases, %s runs; tar %s B md5 %s\n' "$([ -s "$WORK/sets/f32x.tar.md5" ] && echo OK || echo FAIL)" \
            "$(wc -l < "$WORK/sets/f32x/manifest.tsv")" "$(wc -l < "$WORK/sets/f32x/runs.tsv")" \
            "$(stat -c %s "$WORK/sets/f32x.tar" 2> /dev/null)" "$(cat "$WORK/sets/f32x.tar.md5" 2> /dev/null)"
        printf 'replay-cpu\t%s\t%s cases, %s OK, %s FAIL; %s\n' \
            "$([ "$(cut -f 3 "$WORK/replay/cpu.tsv" | grep -c '^FAIL$')" = 0 ] && [ -s "$WORK/replay/cpu.tsv" ] && echo OK || echo FAIL)" \
            "$(wc -l < "$WORK/replay/cpu.tsv")" "$(cut -f 3 "$WORK/replay/cpu.tsv" | grep -c '^OK$')" \
            "$(cut -f 3 "$WORK/replay/cpu.tsv" | grep -c '^FAIL$')" "$(tail -n 1 "$WORK/replay/cpu.log")"
    } > "$WORK/summary.tsv"
    log "tables: $(cut -f 1,2 "$WORK/summary.tsv" | tail -n +2 | tr '\t\n' ' ;')"
fi

touch "$WORK/DONE"
