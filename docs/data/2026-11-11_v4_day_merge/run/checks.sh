#!/usr/bin/env bash
# The v4 day merge (2026-11-11, C773): pascal-p106-v4-b12246-day, the day's feature branches merged on
# pascal-p106-v4-b12246-c748, built once and checked on a build machine, CPU only. Each branch's own check script runs on
# the merged tree, from this repository's docs/data/2026-11-11_<row>_*/run/, with the merged branch as its branch; the
# builds they share are made once and hardlinked into each check's folder. Every build and test waits for LOCK (flock)
# and runs at a low CPU share (--cpu-shares=256, nice 19), so one runs at a time on a shared machine. Steps (STEPS, in
# this order by default):
#   main    C762's checks.sh in $M/main: the static native CPU build of every target and the GGML_RPC=ON build of
#           llama-server and ggml-rpc-server (cpu), C762's server check with its RPC arms (server), "off": test-llama-archs
#           on every architecture with every switch off (and C762's two arms) against the logits of BASE (the base
#           branch's static native CPU build): the merged tree computes what the base computes, ctest, and the rig
#           tarball (TESTS=1, sm_61, no AVX) with test-llama-archs in its bin/ (build); MAIN_STEPS chooses among them
#   prep    a folder for each check below with the main build's CPU binaries (every one in build/bin), the RPC build's
#           (c738) and the rig tree hardlinked in, as each script's own cpu and build steps leave them, and this
#           repository (each script runs from $WORK/repo: its containers mount $WORK, not REPO)
#   c748 c735 c693 c619 c738 c740 c761 c741 c615   each branch's checks.sh without its cpu, ctest and build steps
#           (main's), and without the steps that build a control tree of their own (C735's nohalf) or a test merge
#           (C615's merge); a branch's off step runs where it sets its switch (C693, C738, C761, C741, C615)
#   combo   the combinations the merge makes: test-llama-archs on every architecture with every switch of both merges on
#           together (the CUDA ones change nothing on a CPU) against BASE's logits; C761's --ngram-prefetch with C615's
#           LLAMA_DRAFT_PREFETCH=1 (C615's merge test); run/server_combo.sh: llama-server's draft-mtp with R90's host
#           switches and with every exact host switch of both merges together against BASE's llama-server
#   sass    run/sass_attrib.py on cuobjdump's SASS and PTX of llama-server: every GPU function of the merged rig build is
#           BASE_RIG's (pascal-p106-v4-b12246-c748's rig build) or that of the branch build in SASS_BUILDS that changed
#           it last
#   record  scripts/record_checks.sh SUITE=v4day (the day's seven kernel suites in one set) with the rig build's
#           test-backend-ops, the cases the branches' sets hold hardlinked from them (LINK_FROM)
#   replay  that set replayed on the CPU, filter by filter: it reads back and passes before a rig sees it
#   bitid   bench/mmvq_pascal/verify/build_multitoken_bitid.sh against the rig build's tree: the bit-identity tool of the
#           rig rows (modes dmt, q5k, q5k1, q3k, bf16, f32, glu)
#   c749    C749's chain.sh, steps emu replay tables: a CPU stand-in of its fault built from the merged tree, the
#           v4night set's lid-rows-fault run replayed three times (NIGHT_SET)
#   c565    2026-11-05_c565_ladder4fix/run/cpu_tests.sh, static and then shared: ctest, the network test, archs, the
#           hand tests
# Env: M (the work root), REPO (this repository as run), LOCK (a flock file), BASE (the base branch's static native CPU
#   build/bin: test-llama-archs, llama-server), BASE_RIG (the base branch's rig tarball tree), SEED_SRC (a llama.cpp
#   clone with the fork as remote "fork", https), SEED_CCACHE and SEED_CPU_CCACHE (ccaches to copy in first, optional),
#   C748_BEFORE (bin/ of C748's control build, for its kvonly step), SASS_BUILDS ('<label>=<rig tarball tree>[@<parent
#   label>] ...': a branch build on another listed branch names it),
#   LINK_FROM (record_checks.sh's), NIGHT_SET (the v4night set folder), BRANCH (default pascal-p106-v4-b12246-day),
#   JOBS (8), THREADS (8), DOCKER_ARGS, IMAGE, STEPS, MAIN_STEPS (C762's steps for main, default all five)
# Output: $M/summary.tsv (row, check, verdict, detail: each script's own verdict lines), $M/logs/<step>.out,
#   $M/logs/day.log; done when $M/DONE exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
BASE=$(realpath "${BASE:?BASE=<build/bin of the base CPU build>}")
BRANCH=${BRANCH:-pascal-p106-v4-b12246-day}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
SRV_IMAGE=pq-c735-srv:24.04
STEPS=${STEPS:-main prep c748 c735 c693 c619 c738 c740 c761 c741 c615 combo sass record replay bitid c749 c565}
DA="--label pascal-quants=1 ${DOCKER_ARGS:-}"
D=$REPO/docs/data
RUN=$D/2026-11-11_v4_day_merge/run
SRC=$M/main/rig/src
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'row\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
log()    { echo "$(date -Is) $*" | tee -a "$M/logs/day.log"; }
locked() { flock -w 43200 "$LOCK" "$@"; }
lines()  { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }
add()    { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$M/summary.tsv"; }
# a script's verdict lines from line $3 of its summary: (check, verdict, detail), the header left out
take() {
    [ -f "$2" ] || { add "$1" summary FAIL "no $2"; return; }
    tail -n +"$3" "$2" | awk -F'\t' -v r="$1" 'BEGIN { OFS = "\t" }
        $0 == "" || ($1 == "check" && $2 == "verdict") { next }
        NF >= 4 { d = $4; for (i = 5; i <= NF; i++) d = d " " $i; print r, $1 " " $2, $3, d; next }
        { print r, $1, $2, $3 }' >> "$M/summary.tsv"
}
tree() { ls -td "$M"/main/rig/llama.cpp-*-"$BRANCH" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }

if step main; then
    mkdir -p "$M/main/rig" "$M/main/cpu"
    if [ ! -d "$SRC/.git" ]; then
        git clone -q --local --no-checkout "${SEED_SRC:?SEED_SRC=<a llama.cpp clone with the fork as remote fork>}" "$SRC" &&
            git -C "$SRC" remote add fork "$(git -C "$SEED_SRC" remote get-url fork)" &&
            git -C "$SRC" remote set-url origin "$(git -C "$SEED_SRC" remote get-url origin)"
    fi
    [ -d "$M/main/rig/ccache" ] || [ -z "${SEED_CCACHE:-}" ] || nice -n 19 cp -a "$SEED_CCACHE" "$M/main/rig/ccache"
    [ -d "$M/main/cpu/ccache" ] || [ -z "${SEED_CPU_CCACHE:-}" ] || nice -n 19 cp -a "$SEED_CPU_CCACHE" "$M/main/cpu/ccache"
    rm -rf "$M/main/repo" && cp -al "$REPO" "$M/main/repo"
    n=$(($(lines "$M/main/summary.tsv") + 1))
    env WORK="$M/main" REPO="$M/main/repo" SRC="$SRC" BRANCH="$BRANCH" BASE_CPU="$BASE" BASE_SRV="$BASE" JOBS="$JOBS" \
        THREADS="$THREADS" LOCK="$LOCK" DOCKER_ARGS="$DA" STEPS="${MAIN_STEPS:-cpu server off ctest build}" nice -n 19 \
        bash "$M/main/repo/docs/data/2026-11-11_c762_draft_rows/run/checks.sh" >> "$M/logs/main.out" 2>&1
    log "main rc=$?"
    take main "$M/main/summary.tsv" "$n"
fi
T=$(tree)
TIP=$(git -C "$SRC" rev-parse "fork/$BRANCH") || { echo "no fork/$BRANCH in $SRC"; exit 1; }
log "day merge checks: $BRANCH at ${TIP:0:9}, base $BASE, rig tree ${T:-none}, steps: $STEPS"

ROWS="c748 c735 c693 c619 c738 c740 c761 c741 c615"
if step prep; then
    B=$M/main/cpu/tip/tree/build/bin
    for r in $ROWS; do
        rm -rf "$M/$r/cpu/tip/bin" && mkdir -p "$M/$r/cpu/tip" "$M/$r/rig"
        rm -rf "$M/$r/repo" && cp -al "$REPO" "$M/$r/repo"
        cp -al "$B" "$M/$r/cpu/tip/bin"
        [ -n "$T" ] && [ ! -e "$M/$r/rig/$(basename "$T")" ] && cp -al "$T" "$M/$r/rig/"
    done
    rm -rf "$M/c738/cpu/rpc/bin" && mkdir -p "$M/c738/cpu/rpc" && cp -al "$M/main/cpu/rpc/tree/build/bin" "$M/c738/cpu/rpc/bin"
    rm -rf "$M/c748/cpu/before" && mkdir -p "$M/c748/cpu/before" && cp -a "${C748_BEFORE:?C748_BEFORE=<bin/ of the C748 control build>}" "$M/c748/cpu/before/bin"
    rm -rf "$M/main/repo" && cp -al "$REPO" "$M/main/repo"
    log "prep done: $(du -sh "$M" | cut -f1) in $M"
fi

# <row> <data folder> <steps> [env...]: the branch's checks.sh on the merged tree in $M/<row>, from $M/<row>/repo
row() {
    local r=$1 f=$2 s=$3 n
    shift 3
    n=$(($(lines "$M/$r/summary.tsv") + 1))
    env WORK="$M/$r" REPO="$M/$r/repo" SRC="$SRC" BRANCH="$BRANCH" JOBS="$JOBS" THREADS="$THREADS" LOCK="$LOCK" \
        DOCKER_ARGS="$DA" STEPS="$s" "$@" nice -n 19 bash "$M/$r/repo/docs/data/$f/run/checks.sh" > "$M/logs/$r.out" 2>&1
    log "$r rc=$?"
    take "$r" "$M/$r/summary.tsv" "$n"
}
step c748 && row c748 2026-11-11_c748_kvonly_kpool 'kvonly overlap kvrig' BEFORE=7b9dabf66 BASE_CPU="$BASE"
step c735 && row c735 2026-11-11_c735_c217_c703 'incall overlap kvonly context server incallrig' BASE_CPU="$BASE" BASE_SRV="$BASE"
step c693 && row c693 2026-11-11_c693_upstream_switches 'hdev overlap off server hdevrig' BASE="$BASE"
step c619 && row c619 2026-11-11_c619_draft_graph_reuse 'keep regress server keeprig' BASE_BIN="$BASE"
step c738 && row c738 2026-11-11_c738_verify_prepare 'vprep fault server off vrig table' BASE_CPU="$BASE" BASE_SRV="$BASE"
step c740 && row c740 2026-11-11_c740_seq_rm_bound 'bound control time server trace boundrig' BASE_BIN="$BASE"
step c761 && row c761 2026-11-11_c761_ngram_prefetch 'npf server off vrig table' BASE_CPU="$BASE" BASE_SRV="$BASE"
step c741 && row c741 2026-11-11_c741_server_yields 'unit server off' BASE_CPU="$BASE" BASE_SRV="$BASE"
step c615 && row c615 2026-11-11_c615_draft_prefetch 'server off rsrv trace' BASE_CPU="$BASE" BASE_SRV="$BASE"

if step combo; then
    K=$M/combo
    CPU=$M/main/cpu/tip/tree/build/bin
    rm -rf "$K" && mkdir -p "$K/off" "$K/npf" "$K/models"
    box() { locked docker run --rm --cpu-shares=256 $DA --network none -v "$K:$K" -v "$CPU:$CPU:ro" -v "$BASE:$BASE:ro" \
                -v "$RUN:$RUN:ro" "$SRV_IMAGE" "$@"; }
    # every switch of both merges on together: the host ones that compute what the base computes and the CUDA ones
    ALL="LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_CATCHUP_OVERLAP=1 LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 \
LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_MTP_GRAPH_KEEP=1 \
LLAMA_VERIFY_PREPARE=2 LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_NGRAM_PREFETCH=1 LLAMA_DRAFT_PREFETCH=1 \
LLAMA_DRAFT_EMBD_DEV=layer LLAMA_MTP_H_DEV=2 GGML_SCHED_PREFETCH=0 GGML_CUDA_MMVQ_PASCAL_DMT=1 \
GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 GGML_CUDA_MMVQ_PASCAL_Q3K_MT=1 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1 \
GGML_CUDA_MMVF_PASCAL_EXACT=1 GGML_CUDA_MMVF_F32_MT=1 GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1 \
GGML_CUDA_FATTN_SPARSE_PASCAL_UNION=1 GGML_CUDA_GRAPHS_FORCE=1"
    # shellcheck disable=SC2016
    box bash -c '
        archs() { env LLAMA_ARCHS_LOGITS='"$K"'/off/logits- "${@:3}" nice -n 19 "$2"/test-llama-archs -b CPU -s 1 2>&1 \
                    | sed "s/\x1b\[[0-9;]*m//g" > '"$K"'/off/$1.log; echo "$1 rc=${PIPESTATUS[0]}"; }
        archs base '"$BASE"'
        archs all '"$CPU"' '"$ALL"'
        rm -f '"$K"'/off/*.f32
        for s in 1 2 3; do
            env LLAMA_DRAFT_PREFETCH=1 nice -n 19 '"$CPU"'/test-llama-archs --ngram-prefetch -s $s 2>&1 \
                | sed "s/\x1b\[[0-9;]*m//g" > '"$K"'/npf/s$s.log
            echo "npf s$s rc=${PIPESTATUS[0]}"
        done
        nice -n 19 '"$CPU"'/test-llama-archs -a "^qwen4exp\$" -s 1 -o '"$K"'/models --mtp > '"$K"'/models/save.log 2>&1
        BASE='"$BASE"' BIN='"$CPU"' MODEL='"$K"'/models/qwen4exp-moe-mtp.gguf OUT='"$K"'/server THREADS='"$THREADS"' \
            SARGS="--override-kv qwen4exp.context_length=int:16384" nice -n 19 bash '"$RUN"'/server_combo.sh > '"$K"'/server.console.txt 2>&1
        echo "server rc=$?"
        chown -R '"$(id -u):$(id -g)"' '"$K"'' > "$K/rc.txt" 2>&1
    grep -a 'logits against' "$K/off/base.log" | grep -v ', 0 of ' > "$K/off/base_self.txt"
    nc=$(grep -a -c 'logits against' "$K/off/all.log")
    grep -a 'logits against' "$K/off/all.log" | grep -v ', 0 of ' | grep -vxFf "$K/off/base_self.txt" > "$K/off/all.differ.txt"
    add combo "off all switches" "$([ "$nc" -gt 0 ] && [ ! -s "$K/off/all.differ.txt" ] && grep -q '^all rc=0' "$K/rc.txt" && echo SAME || echo DIFFERS)" \
        "$nc compared, $(wc -l < "$K/off/all.differ.txt") differ against the base's logits: $(cut -d: -f1 "$K/off/all.differ.txt" | tr '\n' ' '); $(grep -a -m 1 '^Summary' "$K/off/all.log"); $(grep '^all rc=' "$K/rc.txt"); $ALL"
    for s in 1 2 3; do
        add combo "ngram-prefetch with LLAMA_DRAFT_PREFETCH=1 s$s" "$(grep -q "^npf s$s rc=0" "$K/rc.txt" && echo PASS || echo FAIL)" \
            "$(grep "^npf s$s rc=" "$K/rc.txt"); $(grep -a -m 1 '^Summary' "$K/npf/s$s.log")"
    done
    add combo "server_combo" "$(grep -q '^server rc=0' "$K/rc.txt" && grep -q '^PASS' "$K/server/verdict.txt" && echo OK || echo FAIL)" \
        "$(grep '^server rc=' "$K/rc.txt"); $(grep -v '^PASS\|^FAIL' "$K/server/verdict.txt" 2> /dev/null | tr '\n' ';')"
    log "combo done"
fi

if step sass; then
    S=$M/sass
    mkdir -p "$S"
    # <label> <tarball tree>: the function tables of its llama-server's SASS and PTX
    dump() {
        local w
        for w in sass ptx; do
            # shellcheck disable=SC2086
            locked docker run --rm --cpu-shares=256 $DA --network none -v "$2:$2:ro" "$IMAGE" \
                nice -n 19 cuobjdump -$w "$2/bin/llama-server" | nice -n 19 python3 "$RUN/sass_attrib.py" dump $w > "$S/$1.$w.tsv"
        done
    }
    dump day "$T"
    dump base "${BASE_RIG:?BASE_RIG=<the base rig tarball tree>}"
    args_sass=() args_ptx=()
    for b in ${SASS_BUILDS:?SASS_BUILDS=<label>=<tree>[@<parent label>] ...}; do
        l=${b%%=*} t=${b#*=} p=
        [[ "$t" == *@* ]] && p=@${t#*@} t=${t%%@*}
        dump "$l" "$t"
        args_sass+=("$l=$S/$l.sass.tsv$p") args_ptx+=("$l=$S/$l.ptx.tsv$p")
    done
    for w in sass ptx; do
        if [ $w = sass ]; then a=("${args_sass[@]}"); else a=("${args_ptx[@]}"); fi
        python3 "$RUN/sass_attrib.py" attrib "$S/day.$w.tsv" "$S/base.$w.tsv" "${a[@]}" > "$S/attrib_$w.txt" 2>&1
        add sass "llama-server $w by build" "$(tail -n 1 "$S/attrib_$w.txt" | sed 's/^verdict: //')" \
            "$(head -n 1 "$S/attrib_$w.txt"); $(sed -n '2,/^  [A-Z]/p' "$S/attrib_$w.txt" | grep -v '^  [A-Z]' | sed 's/^ *//' | tr '\n' ';')"
    done
    log "sass done"
fi

if step record; then
    if [ -n "$T" ]; then
        locked env STEPS="record tar" OUT="$M/sets" SUITE=v4day BIN="$T/bin" JOBS="$JOBS" DOCKER_ARGS="$DA" IMAGE="$IMAGE" \
            LINK_FROM="${LINK_FROM:-}" nice -n 19 bash "$REPO/scripts/record_checks.sh" > "$M/logs/record.out" 2>&1
        rc=$?
        add set record "$([ $rc = 0 ] && [ -s "$M/sets/v4day.tar.md5" ] && echo OK || echo FAIL)" \
            "rc=$rc; $(grep 'linked .* in all' "$M/sets/v4day.log" | tail -n 1 | cut -d' ' -f2-); $(grep 'manifest:' "$M/sets/v4day.log" | tail -n 1 | cut -d' ' -f2-); $(grep 'tar:' "$M/sets/v4day.log" | tail -n 1 | cut -d' ' -f2-)"
    else
        add set record FAIL "no rig tree"
    fi
    log "record done"
fi

if step replay; then
    S=$M/sets/v4day
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

if step bitid; then
    locked env DOCKER_ARGS="$DA" nice -n 19 bash "$REPO/bench/mmvq_pascal/verify/build_multitoken_bitid.sh" "$SRC" "$SRC" \
        > "$M/logs/bitid.out" 2>&1
    rc=$?
    X=$REPO/bench/mmvq_pascal/build/multitoken_bitid
    add bitid build "$([ $rc = 0 ] && [ -x "$X" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(md5sum < "$X" 2> /dev/null | cut -d' ' -f1) $X; linked against $SRC/build ($(git -C "$SRC" rev-parse --short=9 HEAD))"
    log "bitid done"
fi

if step c749; then
    mkdir -p "$M/c749/w"
    [ -e "$M/c749/w/src" ] || git clone -q --local --no-checkout "$SRC" "$M/c749/w/src"
    git -C "$M/c749/w/src" fetch -q origin "+refs/remotes/fork/$BRANCH:refs/remotes/fork/$BRANCH"
    [ -n "$T" ] && [ ! -e "$M/c749/w/$(basename "$T")" ] && cp -al "$T" "$M/c749/w/"
    env WORK="$M/c749" PUB="$REPO" BRANCH="$BRANCH" SET="${NIGHT_SET:?NIGHT_SET=<the v4night set folder>}" STEPS='emu replay tables' \
        LOCK="$LOCK" JOBS="$JOBS" DOCKER_ARGS="$DA" IMAGE="$IMAGE" nice -n 19 \
        bash "$D/2026-11-11_c749_lid_rows_fault/run/chain.sh" > "$M/logs/c749.out" 2>&1
    log "c749 rc=$?"
    if [ -f "$M/c749/results.tsv" ]; then
        tail -n +2 "$M/c749/results.tsv" | while IFS=$'\t' read -r a b c d rest; do add c749 "replay $a" LOG "$b $c $d $rest"; done
    else
        add c749 results FAIL "no $M/c749/results.tsv"
    fi
fi

if step c565; then
    for sh in OFF ON; do
        v=$([ $sh = OFF ] && echo static || echo shared)
        env SRC="$SRC" WORK="$M/c565-$v" TREES="day=$TIP" STEPS='tree build ctest net archs hand' SHARED=$sh JOBS="$JOBS" \
            THREADS="$THREADS" LOCK="$LOCK" DOCKER_ARGS="$DA" nice -n 19 \
            bash "$D/2026-11-05_c565_ladder4fix/run/cpu_tests.sh" > "$M/logs/c565-$v.out" 2>&1
        log "c565 $v rc=$?"
        take "c565 $v" "$M/c565-$v/summary.tsv" 2
    done
fi

log "done"
date -Is > "$M/DONE"
