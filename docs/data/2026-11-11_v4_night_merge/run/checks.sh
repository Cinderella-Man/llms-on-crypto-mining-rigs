#!/usr/bin/env bash
# The v4 night merge (2026-11-11): pascal-p106-v4-b12246-night, the night's 17 feature branches merged on
# pascal-p106-v4, built once and checked on a build machine, CPU only. Each branch's own check script runs on the merged
# tree, from this repository's docs/data/2026-11-11_<row>_*/run/, with the merged branch as its branch; the builds they
# share are made once. Every build and test waits for LOCK (flock) and runs at a low CPU share (--cpu-shares=256,
# nice 19), so one runs at a time on a shared machine. Steps (STEPS, in this order by default):
#   main    C620's checks.sh in $M/w (the work folder of scripts/build_llamacpp_docker.sh): the rig tarball (TESTS=1,
#           sm_61, no AVX) with test-llama-archs in its bin/, the static native CPU build of test-llama-archs, C620's
#           KV-only catch-up check with both, and "off": test-llama-archs on every architecture with every switch off
#           against the logits of BASE_CPU (pascal-p106-v4's CPU build): the merged tree computes what the base computes
#   combo   the four QSA switches that meet in build_qsa_sel (C100, C720, C719, C704) on together, with both device
#           checks, against the merged tree's logits with them off, on qwen4exp at its top-k and at 8
#   record  scripts/record_checks.sh SUITE=v4night (the night's seven suites in one set) with the rig build's
#           test-backend-ops
#   replay  that set replayed on the CPU, filter by filter: it reads back and passes before a rig sees it
#   c702    C702's tests from the rig build's tree (the CUDA graphs switch's capture model)
#   c703    C703's overlap check with main's two builds
#   c103    the sparse attention's CPU check (bench/fattn_sparse/check.sh: C705's prompt rule, C103's union kernel)
#   sass    C690's and C722's static checks of the sm_61 code, from the rig build's objects, against their tables
#   prep    a folder for each check below: a local clone of $M/w/src with the fork's refs (c217, c405, c691, c704), the
#           rig tree hardlinked (c691, c704), a ccache to start from (main's for c704, else the row's own in ORIG)
#   c704    C704's chain (cpu tbo archs graph); c691 C691's chain (cpu tbo sampler graph)
#   c100 c719 c720  their cpu_check.sh against BASE_CPU; c675 its cpu_check.sh against BASE_REF; c711 its cpu_check.sh
#           against C675's branch (C711's base); c701 its cpu_check.sh against BASE_REF
#   c217    its checks.sh (cpubuild context control server); c405 its cpu_check.sh (cpubuild smoke)
#   c565    2026-11-05_c565_ladder4fix/run/cpu_tests.sh, static and then shared: ctest, the network test, archs, the
#           hand tests
# The checks' own "off" steps that main's "off" repeats (C217's, C405's archs, C703's) and the record steps of single
# suites are left out; C705's and C103's comparisons of their GPU code with their base build's do not apply to a tree
# that holds other branches' kernels.
# Env: M (the work root), REPO (this repository as run), LOCK (a flock file), BASE_CPU (bin/ of BASE_REF's static
#   native CPU build: test-llama-archs, llama-server), TINY (a small GGUF for test-backend-sampler), BRANCH (default
#   pascal-p106-v4-b12246-night), BASE_REF (default 84f7bd3d2, pascal-p106-v4), C675_REF (default b2763cce7), ORIG
#   (optional: the night sessions' work root; ORIG/<row>/ccache or ORIG/<row>/w/ccache seeds the row's ccache),
#   C565_SEED (optional: a folder with c565-static/ccache and c565-shared/ccache), JOBS (8), THREADS (8), DOCKER_ARGS,
#   IMAGE, STEPS
# Output: $M/summary.tsv (row, check, verdict, detail: each script's own verdict lines; the two chains' log lines as
#   LOG), $M/logs/<step>.out, $M/logs/night.log; done when $M/DONE exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
BASE_CPU=$(realpath "${BASE_CPU:?BASE_CPU=<bin/ of the base CPU build>}")
BRANCH=${BRANCH:-pascal-p106-v4-b12246-night}
BASE_REF=${BASE_REF:-84f7bd3d2}
C675_REF=${C675_REF:-b2763cce7}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
STEPS=${STEPS:-main combo record replay c702 c703 c103 sass prep c704 c691 c100 c719 c720 c675 c711 c701 c217 c405 c565}
DA="--label pascal-quants=1 ${DOCKER_ARGS:-}"
D=$REPO/docs/data
W=$M/w
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'row\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
log()    { echo "$(date -Is) $*" | tee -a "$M/logs/night.log"; }
locked() { flock -w 43200 "$LOCK" "$@"; }
lines()  { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }
add()    { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$M/summary.tsv"; }
# a script's verdict lines from line $3 of its summary: (check, verdict, detail) or (tree|arm|stage, check, verdict,
# detail), the header left out
take() {
    [ -f "$2" ] || { add "$1" summary FAIL "no $2"; return; }
    tail -n +"$3" "$2" | awk -F'\t' -v r="$1" 'BEGIN { OFS = "\t" }
        $0 == "" || ($1 == "check" && ($2 == "verdict" || $2 == "arm")) || (($1 == "tree" || $1 == "stage") && $2 == "check") { next }
        NF >= 4 { d = $4; for (i = 5; i <= NF; i++) d = d " " $i; print r, $1 " " $2, $3, d; next }
        { print r, $1, $2, $3 }' >> "$M/summary.tsv"
}
# a chain's log lines ("<time> <check>: <detail>") as LOG rows, judged by reading them
take_log() {
    [ -f "$2" ] || { add "$1" chain FAIL "no $2"; return; }
    while IFS= read -r l; do
        l=${l#* }
        add "$1" "${l%%:*}" LOG "${l#*: }"
    done < "$2"
}
tree() { ls -td "$W"/llama.cpp-*-"$BRANCH" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
TIP=$(git -C "$W/src" rev-parse "fork/$BRANCH") || { echo "no fork/$BRANCH in $W/src"; exit 1; }
log "night merge checks: $BRANCH at ${TIP:0:9}, base $BASE_REF, steps: $STEPS"

if step main; then
    n=$(($(lines "$W/summary.tsv") + 1))
    env WORK="$W" BUILD="$REPO/scripts/build_llamacpp_docker.sh" BRANCH="$BRANCH" BASE_CPU="$BASE_CPU" JOBS="$JOBS" \
        LOCK="$LOCK" DOCKER_ARGS="$DA" STEPS='build cpubuild kvonly off' nice -n 19 \
        bash "$D/2026-11-11_c620_draft_kv_catchup/run/checks.sh" > "$M/logs/main.out" 2>&1
    log "main rc=$?"
    take c620 "$W/summary.tsv" "$n"
fi
T=$(tree)

if step combo; then
    # the four QSA switches that meet in build_qsa_sel, all exact, together: C100's tables and C720's small inputs on the
    # devices, C719's mask built there, C704's indexer reading the pooled keys in place, with both device checks; the
    # merged tree's own logits with every switch off are the reference, at the model's top-k and at 8 (C704's control,
    # where the selection follows the scores); qwen4exp is the one architecture with all four
    K=$M/combo
    CPU=$W/cpu/bin
    ON="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_DEV_CHECK=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_QSA_POOL_MASK_DEV_CHECK=1 LLAMA_QSA_INDEXER_ROWS=1"
    rm -rf "$K" && mkdir -p "$K/a" "$K/b"
    # shellcheck disable=SC2086
    locked docker run --rm --cpu-shares=256 $DA -v "$CPU:$CPU:ro" -v "$K:$K" "$IMAGE" bash -c '
        run() { # <name> <logits prefix> [env...]
            local n=$1 p=$2; shift 2
            env LLAMA_ARCHS_LOGITS=$p "$@" nice -n 19 '"$CPU"'/test-llama-archs -b CPU -s 1 -a qwen4exp 2>&1 \
                | sed "s/\x1b\[[0-9;]*m//g" > '"$K"'/$n.log
            echo "$n rc=${PIPESTATUS[0]}"
        }
        run off '"$K"'/a/logits-
        run on '"$K"'/a/logits- '"$ON"'
        run top8-off '"$K"'/b/logits- LLAMA_ARCHS_INDEXER_TOP_K=8
        run top8-on '"$K"'/b/logits- LLAMA_ARCHS_INDEXER_TOP_K=8 '"$ON"'
        rm -f '"$K"'/a/*.f32 '"$K"'/b/*.f32
        chown -R '"$(id -u):$(id -g)"' '"$K"'' > "$K/rc.txt" 2>&1
    for n in on top8-on; do
        nc=$(grep -a -c 'logits against' "$K/$n.log")
        nd=$(grep -a 'logits against' "$K/$n.log" | grep -v -c ', 0 of ')
        add combo "qsa four $n" "$([ "$nc" -gt 0 ] && [ "$nd" = 0 ] && grep -q "^$n rc=0" "$K/rc.txt" && echo OK || echo FAIL)" \
            "$nc compared, $nd differ; $(grep -a -m 1 '^Summary' "$K/$n.log"); $(grep "^$n rc=" "$K/rc.txt")"
    done
    log "combo done"
fi

if step record; then
    if [ -n "$T" ]; then
        locked env STEPS="record tar" OUT="$M/sets" SUITE=v4night BIN="$T/bin" JOBS="$JOBS" DOCKER_ARGS="$DA" IMAGE="$IMAGE" \
            nice -n 19 bash "$REPO/scripts/record_checks.sh" > "$M/logs/record.out" 2>&1
        rc=$?
        add set record "$([ $rc = 0 ] && [ -s "$M/sets/v4night.tar.md5" ] && echo OK || echo FAIL)" \
            "rc=$rc; $(grep 'manifest:' "$M/sets/v4night.log" | tail -n 1 | cut -d' ' -f2-); $(grep 'tar:' "$M/sets/v4night.log" | tail -n 1 | cut -d' ' -f2-)"
    else
        add set record FAIL "no rig tree"
    fi
    log "record done"
fi

if step replay; then
    S=$M/sets/v4night
    R=$M/replay
    mkdir -p "$R"
    STUB='s=/usr/local/cuda/lib64/stubs/libcuda.so; if [ -e $s ]; then mkdir -p /tmp/stub && ln -sf $s /tmp/stub/libcuda.so.1;
          export LD_LIBRARY_PATH=/tmp/stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; fi'
    # every distinct filter of the set's runs once, on the CPU (the runs differ only in CUDA switches)
    k=0
    while IFS=$'\t' read -r ops params; do
        k=$((k + 1))
        # shellcheck disable=SC2086
        locked docker run --rm --cpu-shares=256 $DA -v "$T/bin:$T/bin:ro" -v "$S:$S:ro" -v "$R:/k" "$IMAGE" sh -c "$STUB"'
            nice -n 19 '"$T"'/bin/test-backend-ops -b CPU -o "$1" -p "$2" --replay '"$S"' --replay-log /k/cases-'"$k"'.log \
                > /k/cpu-'"$k"'.log 2>&1; echo "rc=$?"' sh "$ops" "$params" > "$R/rc-$k.txt" 2>&1
        rc=$(tail -n 1 "$R/rc-$k.txt")
        nf=$(grep -a -c 'FAIL' "$R/cpu-$k.log")
        add set "replay -o $ops -p $params" "$([ "$rc" = rc=0 ] && [ "$nf" = 0 ] && echo OK || echo FAIL)" \
            "$rc; $nf FAIL lines; $(grep -a -m 1 'tests passed' "$R/cpu-$k.log" | sed 's/^ *//'); $(grep -a 'replay:' "$R/cpu-$k.log" | tail -n 1)"
    done < <(cut -f3,4 "$S/runs.tsv" | awk '!seen[$0]++')
    log "replay done: $k filters"
fi

if step c702; then
    n=$(($(lines "$W/summary.tsv") + 1))
    env WORK="$W" BUILD="$REPO/scripts/build_llamacpp_docker.sh" BRANCH="$BRANCH" JOBS="$JOBS" LOCK="$LOCK" \
        DOCKER_ARGS="$DA" STEPS=tests nice -n 19 bash "$D/2026-11-11_c702_prompt_graphs/run/checks.sh" > "$M/logs/c702.out" 2>&1
    log "c702 rc=$?"
    take c702 "$W/summary.tsv" "$n"
fi

if step c703; then
    n=$(($(lines "$W/summary.tsv") + 1))
    env WORK="$W" BUILD="$REPO/scripts/build_llamacpp_docker.sh" BRANCH="$BRANCH" BASE_CPU="$BASE_CPU" JOBS="$JOBS" \
        LOCK="$LOCK" DOCKER_ARGS="$DA" STEPS=overlap nice -n 19 \
        bash "$D/2026-11-11_c703_catchup_overlap/run/checks.sh" > "$M/logs/c703.out" 2>&1
    log "c703 rc=$?"
    take c703 "$W/summary.tsv" "$n"
fi

if step c103; then
    env WORK="$W" REPO="$REPO" BRANCH="$BRANCH" LOCK="$LOCK" JOBS="$JOBS" DOCKER_ARGS="${DOCKER_ARGS:-}" STEPS=check \
        bash "$D/2026-11-11_c103_union_gather/run/night.sh" > "$M/logs/c103.out" 2>&1
    l=$(grep -a '^C103_STEP check' "$M/logs/c103.out" | tail -n 1)
    ok=$(grep -c '^ok ' "$W/logs/check.log") wrong=$(grep -c '^WRONG ' "$W/logs/check.log")
    add c103 "fattn_sparse check" "$([[ "$l" == *"check rc=0:"* ]] && echo OK || echo FAIL)" "${l#C103_STEP }; $ok ok, $wrong WRONG"
    log "c103 done"
fi

if step sass; then
    S=$M/sass
    mkdir -p "$S"
    O=$W/src/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
    # shellcheck disable=SC2086
    locked nice -n 19 docker run --rm --cpu-shares=256 $DA -v "$W:$W:ro" -v "$S:$S" "$IMAGE" sh -c "
        cuobjdump -res-usage $O/mmvq-pascal-q5k.cu.o > $S/q5k_res.txt 2>&1
        cuobjdump -sass $O/mmvq-pascal-q5k.cu.o > $S/q5k.sass 2>&1
        cuobjdump -res-usage $O/mmvq-pascal-dmt.cu.o > $S/dmt_res.txt 2>&1
        cuobjdump -sass $O/mmvq-pascal-dmt.cu.o > $S/dmt.sass 2>&1
        cuobjdump -sass $O/mmvq-pascal.cu.o | gzip > $S/mmvqp.sass.gz
        chown -R $(id -u):$(id -g) $S"
    for r in c690:2026-11-11_c690_head_four_rows c722:2026-11-11_c722_short_batch_products; do
        n=${r%%:*} f=$D/${r#*:}
        SASS=$S bash "$f/run/sass_check.sh" > "$S/${n}_counts.tsv" 2> "$S/$n.err"
        rc=$?
        if [ $rc = 0 ] && cmp -s "$S/${n}_counts.tsv" "$f/sass_counts.tsv"; then
            add "$n" "sass counts" SAME "$(($(wc -l < "$S/${n}_counts.tsv") - 1)) kernels, the table of the branch's own build"
        else
            add "$n" "sass counts" DIFFERS "rc=$rc; $(diff "$S/${n}_counts.tsv" "$f/sass_counts.tsv" | grep -c '^[<>]') lines differ from the branch's table"
        fi
    done
    log "sass done"
fi

# a local clone of $W/src at the merged branch, with the fork's refs and remote
clone() {
    [ -d "$1/.git" ] && return 0
    git clone -q --local --no-checkout "$W/src" "$1" &&
        git -C "$1" fetch -q origin '+refs/remotes/fork/*:refs/remotes/fork/*' &&
        git -C "$1" remote add fork "$(git -C "$W/src" remote get-url fork)" &&
        git -C "$1" checkout -q --detach "fork/$BRANCH"
}
# <dest ccache> <sources...>: the first that exists, copied once
seed() {
    local d=$1 s
    shift
    [ -d "$d" ] && return 0
    for s in "$@"; do
        [ -n "$s" ] && [ -d "$s" ] && { nice -n 19 cp -a "$s" "$d"; return; }
    done
}
if step prep; then
    O=${ORIG:-}
    for r in c704 c691; do
        mkdir -p "$M/$r/w"
        clone "$M/$r/w/src"
        [ -n "$T" ] && [ ! -e "$M/$r/w/$(basename "$T")" ] && cp -al "$T" "$T.tar.gz" "$M/$r/w/"
    done
    seed "$M/c704/w/ccache" "$W/ccache"
    seed "$M/c691/w/ccache" "${O:+$O/c691/w/ccache}" "$W/ccache"
    for r in c217 c405; do
        mkdir -p "$M/$r"
        clone "$M/$r/src"
        seed "$M/$r/ccache" "${O:+$O/$r/ccache}" "$W/ccache"
    done
    for r in c100 c719 c720 c675 c711 c701; do
        mkdir -p "$M/$r"
        seed "$M/$r/ccache" "${O:+$O/$r/ccache}"
    done
    for v in static shared; do
        mkdir -p "$M/c565-$v"
        seed "$M/c565-$v/ccache" "${C565_SEED:+$C565_SEED/c565-$v/ccache}"
    done
    log "prep done: $(du -sh "$M" | cut -f1) in $M"
fi

if step c704; then
    env WORK="$M/c704" PUB="$REPO" BASE_CPU="$BASE_CPU" BRANCH="$BRANCH" STEPS='cpu tbo archs graph' LOCK="$LOCK" \
        JOBS="$JOBS" DOCKER_ARGS="$DA" IMAGE="$IMAGE" nice -n 19 bash "$D/2026-11-11_c704_indexer_gather/run/chain.sh" \
        > "$M/logs/c704.out" 2>&1
    log "c704 rc=$?"
    take_log c704 "$M/c704/chain.log"
fi

if step c691; then
    env WORK="$M/c691" PUB="$REPO" MODEL="${TINY:?TINY=<a small GGUF>}" BRANCH="$BRANCH" BASE="fork/pascal-p106-v4" \
        STEPS='cpu tbo sampler graph' LOCK="$LOCK" JOBS="$JOBS" DOCKER_ARGS="$DA" IMAGE="$IMAGE" nice -n 19 \
        bash "$D/2026-11-11_c691_sampling_launches/run/chain.sh" > "$M/logs/c691.out" 2>&1
    log "c691 rc=$?"
    take_log c691 "$M/c691/chain.log"
fi

# <row> <data folder> <env...>: a cpu_check.sh that builds its own CPU trees from $W/src
cpu_check() {
    local r=$1 f=$2
    shift 2
    env WORK="$M/$r" SRC="$W/src" TIP="$TIP" JOBS="$JOBS" THREADS="$THREADS" LOCK="$LOCK" DOCKER_ARGS="$DA" "$@" nice -n 19 \
        bash "$D/$f/run/cpu_check.sh" > "$M/logs/$r.out" 2>&1
    log "$r rc=$?"
    take "$r" "$M/$r/cpu/summary.tsv" 2
}
step c100 && cpu_check c100 2026-11-11_c100_kpool_on_cards BASE_BIN="$BASE_CPU"
step c719 && cpu_check c719 2026-11-11_c719_pool_mask_on_cards BASE_BIN="$BASE_CPU"
step c720 && cpu_check c720 2026-11-11_c720_kpool_small_inputs BASE_BIN="$BASE_CPU"
step c675 && cpu_check c675 2026-11-11_c675_end_of_turn_stop BASE="$BASE_REF"
step c711 && cpu_check c711 2026-11-11_c711_block_verify_stop BASE="$C675_REF"
step c701 && cpu_check c701 2026-11-11_c701_microbatch_trace BASE="$BASE_REF"

if step c217; then
    env WORK="$M/c217" BUILD="$REPO/scripts/build_llamacpp_docker.sh" BRANCH="$BRANCH" BASE_REF="$BASE_REF" JOBS="$JOBS" \
        THREADS="$THREADS" LOCK="$LOCK" DOCKER_ARGS="$DA" STEPS='cpubuild context control server' nice -n 19 \
        bash "$D/2026-11-11_c217_incall_checkpoints/run/checks.sh" > "$M/logs/c217.out" 2>&1
    log "c217 rc=$?"
    take c217 "$M/c217/summary.tsv" 2
fi

if step c405; then
    env WORK="$M/c405" BRANCH="$BRANCH" BASE_BIN="$BASE_CPU" JOBS="$JOBS" THREADS="$THREADS" LOCK="$LOCK" \
        DOCKER_ARGS="$DA" STEPS='cpubuild smoke' nice -n 19 bash "$D/2026-11-11_c405_server_gap/run/cpu_check.sh" \
        > "$M/logs/c405.out" 2>&1
    log "c405 rc=$?"
    take c405 "$M/c405/cpu/summary.tsv" 2
fi

if step c565; then
    for sh in OFF ON; do
        v=$([ $sh = OFF ] && echo static || echo shared)
        env SRC="$W/src" WORK="$M/c565-$v" TREES="night=$TIP" STEPS='tree build ctest net archs hand' SHARED=$sh JOBS="$JOBS" \
            THREADS="$THREADS" LOCK="$LOCK" DOCKER_ARGS="$DA" nice -n 19 \
            bash "$D/2026-11-05_c565_ladder4fix/run/cpu_tests.sh" > "$M/logs/c565-$v.out" 2>&1
        log "c565 $v rc=$?"
        take "c565 $v" "$M/c565-$v/summary.tsv" 2
    done
fi

log "done"
date -Is > "$M/DONE"
