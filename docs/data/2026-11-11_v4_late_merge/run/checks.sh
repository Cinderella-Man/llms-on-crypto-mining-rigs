#!/usr/bin/env bash
# The v4 late merge (2026-11-11, C780): the four branches that finished after the day merge was built (C771, C770, C744,
# C742, each on pascal-p106-v4-b12246-c748) merged into pascal-p106-v4 (the day merge, 18d0fa309), built once and checked
# on a build machine, CPU only, as the day merge's run/checks.sh (docs/data/2026-11-11_v4_day_merge) did. Each branch's
# own check script runs on the merged tree, from this repository's docs/data/2026-11-11_<row>_*/run/, with the merged
# branch as its branch and the day merge's static CPU build as its base; the builds they share are made once and
# hardlinked into each check's folder. Every build and test waits for LOCK (flock) and runs at a low CPU share
# (--cpu-shares=256, nice 19), so one runs at a time on a shared machine. Steps (STEPS, in this order by default):
#   base    the base: a static native CPU build of BASE_BRANCH (every target; test-llama-archs and llama-server kept) in
#           $M/base/cpu/tip/bin through C744's checks.sh (cpu, trim)
#   baserig llama-server of the base's rig tarball (BASE_TGZ) unpacked in $M/base/rig for the sass step
#   main    C771's checks.sh in $M/main: the static native CPU build of every target and the GGML_RPC=ON build of
#           llama-server and ggml-rpc-server (cpu), C771's server check with its RPC split arms (server), C748's
#           --mtp-kv-only with C771's switch (kvonly), "off": test-llama-archs on every architecture against the base's
#           logits with every switch off, with C771's switch, and with the k-pool tables on the devices (off), and ctest;
#           MAIN_STEPS chooses
#   prep    a folder for each check below with the main build's CPU binaries hardlinked in (c770 also its CPU build
#           tree, read by C770's names step; c771x762 its RPC binaries), and this repository (each script runs from
#           $WORK/repo: its containers mount $WORK, not REPO)
#   c744    C744's checks.sh, steps server off: its llama-server arms (the cap with greedy, probabilistic and n_min 2
#           drafts, one and two slots) and test-llama-archs with its switch unset and set
#   c742    C742's checks.sh, steps server off: its llama-server arms (the hidden-row copies per step) and test-llama-archs
#   c744x742  C744's server check with C742's switch on in every arm (LLAMA_MTP_SKIP_LAST_ROW=1 in each container): the
#           two drafter switches on together
#   c771x762  C771's server check with C762's LLAMA_DRAFT_EMBD_DEV=layer in every arm: the draft's token table on its
#           layer's device, with and without C771's switch (C771 left this pair unrun: C762 was on its own branch)
#   c770    C770's checks.sh, steps names server rescan: the switch-name scan of the merged tree against the base's and
#           each late branch's (the merged tree must read every name each adds), its llama-server line, and a rescan of
#           the tree's list after a new name (in main's own CPU tree, whose build files name its paths; the tree is
#           restored after, and nothing else reads it from then on)
#   trim    the CPU build trees removed (the binaries the later steps run stay): the build machine's disk
#   rig     C771's checks.sh, step build, in $M/main: the rig tarball (TESTS=1, sm_61, no AVX) with test-llama-archs in its
#           bin/; then the clone's build folder removed (rebuildable)
#   combo  test-llama-archs on every architecture with every switch of the three merges on together (the CUDA ones
#           change nothing on a CPU) against the base's logits; run/server_combo.sh: llama-server's draft-mtp with R90's
#           host switches, with every exact host switch of the night's and the day's merges, and with those and the
#           three late switches, against the base's llama-server
#   sass    the day merge's sass_attrib.py on cuobjdump's SASS and PTX of llama-server: every GPU function of the merged
#           rig build is the base's (none of the four branches changes GPU code)
#   line    C770's build-line check on the merged rig tree: the line's text and a name of the list in its binaries
#   c565    2026-11-05_c565_ladder4fix/run/cpu_tests.sh, static and then shared: ctest, the network test, archs, the
#           hand tests; each build tree removed after its run (the build machine's disk)
#   final   the rig tarball of FINAL_BRANCH (the branch the merge is pushed to once every check above passes) through
#           C620's checks.sh in $M/main/rig (the same clone and ccache), and its bin/ compared with the checked tree's
# Before the rig, c565 and final builds the chain waits (up to 2 hours) until the disk has ROOM_GB free (52 by default,
# the build machine keeps 50 free), and stops if it never does.
# Env: M (the work root), REPO (this repository as run), LOCK (a flock file), BASE_BRANCH (default
#   pascal-p106-v4-b12246-day), BASE_TGZ (its rig tarball), SEED_SRC (a llama.cpp clone with the fork as remote "fork",
#   https), SEED_CCACHE and SEED_CPU_CCACHE (ccaches hardlinked in first, optional), MODELS (C762's two qwen4exp model
#   files), BRANCH (default pascal-p106-v4-b12246-late), LATE (the four branches), FINAL_BRANCH (default pascal-p106-v4),
#   JOBS (8), THREADS (8), DOCKER_ARGS, IMAGE, STEPS, MAIN_STEPS (C771's steps for main)
# Output: $M/summary.tsv (row, check, verdict, detail: each script's own verdict lines), $M/logs/<step>.out,
#   $M/logs/late.log; done when $M/DONE exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
BRANCH=${BRANCH:-pascal-p106-v4-b12246-late}
BASE_BRANCH=${BASE_BRANCH:-pascal-p106-v4-b12246-day}
FINAL_BRANCH=${FINAL_BRANCH:-pascal-p106-v4}
LATE=${LATE:-pascal-p106-v4-b12246-c771 pascal-p106-v4-b12246-c770 pascal-p106-v4-b12246-c744 pascal-p106-v4-b12246-c742}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
SRV_IMAGE=pq-c735-srv:24.04
STEPS=${STEPS:-base baserig main prep c744 c742 c744x742 c771x762 c770 trim rig combo sass line c565}
ROOM_GB=${ROOM_GB:-52}
DA="--label pascal-quants=1 ${DOCKER_ARGS:-}"
D=$REPO/docs/data
RUN=$D/2026-11-11_v4_late_merge/run
SRC=$M/main/rig/src
BASE=$M/base/cpu/tip/bin
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'row\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
log()    { echo "$(date -Is) $*" | tee -a "$M/logs/late.log"; }
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
tree() { ls -td "$M"/main/rig/llama.cpp-*-"$1" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
free_gb() { df -BG --output=avail "$M" | tail -n 1 | tr -dc '0-9'; }
# <step>: wait until ROOM_GB are free, or stop the chain
room() {
    local i
    for i in $(seq 1 24); do
        [ "$(free_gb)" -ge "$ROOM_GB" ] && return 0
        [ "$i" = 1 ] && log "$1 waits for $ROOM_GB GB free ($(free_gb) GB)"
        sleep 300
    done
    add "$1" disk FAIL "under $ROOM_GB GB free for 2 hours ($(free_gb) GB): stopped"
    log "stopped: no room for $1"
    exit 1
}

# the clone every script fetches into, with the fork as remote "fork" and upstream as origin (the rig build's container
# fetches both over https)
if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$M/main/rig"
    git clone -q --local --no-checkout "${SEED_SRC:?SEED_SRC=<a llama.cpp clone with the fork as remote fork>}" "$SRC" &&
        git -C "$SRC" remote add fork "$(git -C "$SEED_SRC" remote get-url fork)" &&
        git -C "$SRC" remote set-url origin "$(git -C "$SEED_SRC" remote get-url origin)"
fi
for b in "$BRANCH" "$BASE_BRANCH" $LATE; do
    git -C "$SRC" fetch -q fork "$b:refs/remotes/fork/$b" || { echo "no fork/$b"; exit 1; }
done
TIP=$(git -C "$SRC" rev-parse "fork/$BRANCH")
log "late merge checks: $BRANCH at ${TIP:0:9}, base $BASE_BRANCH at $(git -C "$SRC" rev-parse --short=9 "fork/$BASE_BRANCH"), steps: $STEPS"

if step base; then
    mkdir -p "$M/base/rig"
    rm -rf "$M/base/repo" && cp -al "$REPO" "$M/base/repo"
    [ -d "$M/base/cpu/ccache" ] || [ -z "${SEED_CPU_CCACHE:-}" ] || { mkdir -p "$M/base/cpu" && cp -al "$SEED_CPU_CCACHE" "$M/base/cpu/ccache"; }
    n=$(($(lines "$M/base/summary.tsv") + 1))
    # C744's script needs a base of its own to start: the steps run here read none
    env WORK="$M/base" REPO="$M/base/repo" SRC="$SRC" BRANCH="$BASE_BRANCH" BASE_CPU="$M/base" BASE_SRV="$M/base" \
        JOBS="$JOBS" THREADS="$THREADS" LOCK="$LOCK" DOCKER_ARGS="$DA" STEPS="cpu trim" nice -n 19 \
        bash "$M/base/repo/docs/data/2026-11-11_c744_draft_cap/run/checks.sh" >> "$M/logs/base.out" 2>&1
    log "base rc=$?"
    take base "$M/base/summary.tsv" "$n"
fi

if step baserig; then
    mkdir -p "$M/base/rig"
    rm -rf "$M/base/rig/"llama.cpp-*
    nice -n 19 tar -C "$M/base/rig" -xzf "${BASE_TGZ:?BASE_TGZ=<the rig tarball of the base branch>}" --wildcards '*/bin/llama-server' '*/BUILDINFO.txt'
    BR=$(ls -d "$M"/base/rig/llama.cpp-* | head -n 1)
    add base "rig llama-server" "$([ -x "$BR/bin/llama-server" ] && echo OK || echo FAIL)" \
        "$(basename "$BASE_TGZ"): bin/llama-server unpacked; $(grep '^commit' "$BR/BUILDINFO.txt")"
fi

if step main; then
    mkdir -p "$M/main/cpu"
    [ -d "$M/main/rig/ccache" ] || [ -z "${SEED_CCACHE:-}" ] || cp -al "$SEED_CCACHE" "$M/main/rig/ccache"
    [ -d "$M/main/cpu/ccache" ] || [ -z "${SEED_CPU_CCACHE:-}" ] || cp -al "$SEED_CPU_CCACHE" "$M/main/cpu/ccache"
    rm -rf "$M/main/repo" && cp -al "$REPO" "$M/main/repo"
    n=$(($(lines "$M/main/summary.tsv") + 1))
    env WORK="$M/main" REPO="$M/main/repo" SRC="$SRC" BRANCH="$BRANCH" BASE_CPU="$BASE" BASE_SRV="$BASE" \
        MODELS="${MODELS:?MODELS=<dir with the two qwen4exp-moe-mtp model files of C762>}" JOBS="$JOBS" THREADS="$THREADS" \
        LOCK="$LOCK" DOCKER_ARGS="$DA" STEPS="${MAIN_STEPS:-cpu server kvonly off ctest}" nice -n 19 \
        bash "$M/main/repo/docs/data/2026-11-11_c771_draft_kpool_split/run/checks.sh" >> "$M/logs/main.out" 2>&1
    log "main rc=$?"
    take main "$M/main/summary.tsv" "$n"
fi

ROWS="c744 c742 c744x742 c771x762 c770"
if step prep; then
    B=$M/main/cpu/tip/bin
    for r in $ROWS; do
        rm -rf "$M/$r/cpu/tip/bin" && mkdir -p "$M/$r/cpu/tip"
        rm -rf "$M/$r/repo" && cp -al "$REPO" "$M/$r/repo"
        cp -al "$B" "$M/$r/cpu/tip/bin"
    done
    rm -rf "$M/c771x762/cpu/rpc/bin" && mkdir -p "$M/c771x762/cpu/rpc" && cp -al "$M/main/cpu/rpc/bin" "$M/c771x762/cpu/rpc/bin"
    rm -rf "$M/c770/cpu/tip/tree" && cp -al "$M/main/cpu/tip/tree" "$M/c770/cpu/tip/tree"
    log "prep done: $(du -sh "$M" | cut -f1) in $M; $(df -h --output=avail "$M" | tail -n 1 | tr -d ' ') free"
fi

# <row> <data folder> <steps> [env...]: the branch's checks.sh on the merged tree in $M/<row>, from $M/<row>/repo
row() {
    local r=$1 f=$2 s=$3 n
    shift 3
    n=$(($(lines "$M/$r/summary.tsv") + 1))
    env WORK="$M/$r" REPO="$M/$r/repo" SRC="$SRC" BRANCH="$BRANCH" JOBS="$JOBS" THREADS="$THREADS" LOCK="$LOCK" \
        DOCKER_ARGS="$DA" STEPS="$s" BASE_CPU="$BASE" BASE_SRV="$BASE" "$@" nice -n 19 \
        bash "$M/$r/repo/docs/data/$f/run/checks.sh" > "$M/logs/$r.out" 2>&1
    log "$r rc=$?"
    take "$r" "$M/$r/summary.tsv" "$n"
}
step c744 && row c744 2026-11-11_c744_draft_cap 'server off'
step c742 && row c742 2026-11-11_c742_last_draft_row 'server off'
step c744x742 && row c744x742 2026-11-11_c744_draft_cap 'server' DOCKER_ARGS="$DA -e LLAMA_MTP_SKIP_LAST_ROW=1"
step c771x762 && row c771x762 2026-11-11_c771_draft_kpool_split 'server' MODELS="$MODELS" DOCKER_ARGS="$DA -e LLAMA_DRAFT_EMBD_DEV=layer"
if step c770; then
    row c770 2026-11-11_c770_switch_log 'names server' MODELS="$MODELS" BASE_BRANCH="$BASE_BRANCH" DAY="$BRANCH" BRANCHES="$LATE"
    # C770's names verdict wants the branch's names = its base's; here the merged tree adds the three late switches
    V=$M/c770/checks/names/names_verdict.txt
    want="tip differs from base: LLAMA_MTP_DRAFT_CAP LLAMA_MTP_EMBD_FIRST LLAMA_MTP_SKIP_LAST_ROW "
    got=$(head -n 1 "$V" 2> /dev/null)
    add c770 "names on the merged tree" "$([ "$got" = "$want" ] && grep -q 'branches add that the day merge lacks: none' "$V" && echo OK || echo FAIL)" \
        "expected the three late switches added and no name of a late branch missing; $(tr '\n' ';' < "$V" 2> /dev/null)"
    # the rescan in main's own CPU tree: its build files name that tree's paths, so a copy cannot be rebuilt; the tree is
    # restored after, and nothing else reads it from here on
    n=$(($(lines "$M/main/summary.tsv") + 1))
    env WORK="$M/main" REPO="$M/main/repo" SRC="$SRC" BRANCH="$BRANCH" BASE_BRANCH="$BASE_BRANCH" BASE_CPU="$BASE" BASE_SRV="$BASE" \
        JOBS="$JOBS" THREADS="$THREADS" LOCK="$LOCK" DOCKER_ARGS="$DA" STEPS=rescan nice -n 19 \
        bash "$M/main/repo/docs/data/2026-11-11_c770_switch_log/run/checks.sh" > "$M/logs/c770-rescan.out" 2>&1
    log "c770 rescan rc=$?"
    take c770 "$M/main/summary.tsv" "$n"
fi

if step trim; then
    rm -rf "$M/main/cpu/tip/tree" "$M/main/cpu/rpc/tree" "$M/c770/cpu/tip/tree"
    add trim "CPU trees" OK "removed (the binaries in */cpu/*/bin stay); $(du -sh "$M" | cut -f1) in $M; $(free_gb) GB free"
fi

if step rig; then
    room rig
    n=$(($(lines "$M/main/summary.tsv") + 1))
    env WORK="$M/main" REPO="$M/main/repo" SRC="$SRC" BRANCH="$BRANCH" BASE_CPU="$BASE" BASE_SRV="$BASE" \
        JOBS="$JOBS" THREADS="$THREADS" LOCK="$LOCK" DOCKER_ARGS="$DA" STEPS=build nice -n 19 \
        bash "$M/main/repo/docs/data/2026-11-11_c771_draft_kpool_split/run/checks.sh" >> "$M/logs/main.out" 2>&1
    log "rig rc=$?"
    take rig "$M/main/summary.tsv" "$n"
    rm -rf "$SRC/build" "$M/main/rig/extra"
    add rig trim OK "the clone's build folder removed; $(free_gb) GB free"
fi
T=$(tree "$BRANCH")

if step combo; then
    K=$M/combo
    CPU=$M/main/cpu/tip/bin
    rm -rf "$K" && mkdir -p "$K/off" "$K/models"
    box() { locked docker run --rm --cpu-shares=256 $DA --network none -v "$K:$K" -v "$CPU:$CPU:ro" -v "$BASE:$BASE:ro" \
                -v "$RUN:$RUN:ro" "$SRV_IMAGE" "$@"; }
    # every switch of the three merges on together: the host ones that compute what the base computes and the CUDA ones
    ALL="LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_CATCHUP_OVERLAP=1 LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 \
LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_MTP_GRAPH_KEEP=1 \
LLAMA_VERIFY_PREPARE=2 LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_NGRAM_PREFETCH=1 LLAMA_DRAFT_PREFETCH=1 \
LLAMA_DRAFT_EMBD_DEV=layer LLAMA_MTP_H_DEV=2 GGML_SCHED_PREFETCH=0 GGML_CUDA_MMVQ_PASCAL_DMT=1 \
GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 GGML_CUDA_MMVQ_PASCAL_Q3K_MT=1 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1 \
GGML_CUDA_MMVF_PASCAL_EXACT=1 GGML_CUDA_MMVF_F32_MT=1 GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1 \
GGML_CUDA_FATTN_SPARSE_PASCAL_UNION=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMVQ_PASCAL_Q5K_1R=1 \
LLAMA_MTP_EMBD_FIRST=1 LLAMA_MTP_DRAFT_CAP=1 LLAMA_MTP_SKIP_LAST_ROW=1"
    # shellcheck disable=SC2016
    box bash -c '
        archs() { env LLAMA_ARCHS_LOGITS='"$K"'/off/logits- "${@:3}" nice -n 19 "$2"/test-llama-archs -b CPU -s 1 2>&1 \
                    | sed "s/\x1b\[[0-9;]*m//g" > '"$K"'/off/$1.log; echo "$1 rc=${PIPESTATUS[0]}"; }
        archs base '"$BASE"'
        archs all '"$CPU"' '"$ALL"'
        rm -f '"$K"'/off/*.f32
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
    add combo "server_combo" "$(grep -q '^server rc=0' "$K/rc.txt" && grep -q '^PASS' "$K/server/verdict.txt" && echo OK || echo FAIL)" \
        "$(grep '^server rc=' "$K/rc.txt"); $(grep -v '^PASS\|^FAIL' "$K/server/verdict.txt" 2> /dev/null | tr '\n' ';')"
    log "combo done"
fi

if step sass; then
    S=$M/sass
    mkdir -p "$S"
    BR=$(ls -d "$M"/base/rig/llama.cpp-* | head -n 1)
    # <label> <dir with bin/llama-server>: the function tables of its llama-server's SASS and PTX
    dump() {
        local w
        for w in sass ptx; do
            # shellcheck disable=SC2086
            locked docker run --rm --cpu-shares=256 $DA --network none -v "$2:$2:ro" "$IMAGE" \
                nice -n 19 cuobjdump -$w "$2/bin/llama-server" | nice -n 19 python3 "$D/2026-11-11_v4_day_merge/run/sass_attrib.py" dump $w > "$S/$1.$w.tsv"
        done
    }
    dump late "$T"
    dump base "$BR"
    for w in sass ptx; do
        python3 "$D/2026-11-11_v4_day_merge/run/sass_attrib.py" attrib "$S/late.$w.tsv" "$S/base.$w.tsv" > "$S/attrib_$w.txt" 2>&1
        add sass "llama-server $w by build" "$(tail -n 1 "$S/attrib_$w.txt" | sed 's/^verdict: //')" \
            "$(head -n 1 "$S/attrib_$w.txt"); $(sed -n '2,$p' "$S/attrib_$w.txt" | grep -v '^verdict' | sed 's/^ *//' | head -n 20 | tr '\n' ';')"
    done
    log "sass done"
fi

if step line; then
    hits=$(grep -rla 'this build reads ' "$T" 2> /dev/null | sed "s|$T/||" | tr '\n' ' ')
    names=$(grep -rla 'LLAMA_MTP_SKIP_LAST_ROW' "$T/bin" 2> /dev/null | sed "s|$T/||" | tr '\n' ' ')
    add line "build-line" "$([ -n "$hits" ] && [ -n "$names" ] && echo OK || echo FAIL)" \
        "$(basename "$T"): the line's text in: ${hits:-none}; LLAMA_MTP_SKIP_LAST_ROW (a late switch, in the list) in: ${names:-none}"
fi

if step c565; then
    for sh in OFF ON; do
        v=$([ $sh = OFF ] && echo static || echo shared)
        room "c565 $v"
        env SRC="$SRC" WORK="$M/c565-$v" TREES="late=$TIP" STEPS='tree build ctest net archs hand' SHARED=$sh JOBS="$JOBS" \
            THREADS="$THREADS" LOCK="$LOCK" DOCKER_ARGS="$DA" nice -n 19 \
            bash "$D/2026-11-05_c565_ladder4fix/run/cpu_tests.sh" > "$M/logs/c565-$v.out" 2>&1
        log "c565 $v rc=$?"
        take "c565 $v" "$M/c565-$v/summary.tsv" 2
        rm -rf "$M/c565-$v/tree"
    done
fi

if step final; then
    git -C "$SRC" fetch -q fork "$FINAL_BRANCH:refs/remotes/fork/$FINAL_BRANCH"
    F=$(git -C "$SRC" rev-parse "fork/$FINAL_BRANCH")
    if [ "$F" != "$TIP" ]; then
        add final "$FINAL_BRANCH" FAIL "fork/$FINAL_BRANCH is ${F:0:9}, not the checked ${TIP:0:9}"
    else
        room final
        n=$(($(lines "$M/main/rig/summary.tsv") + 1))
        env WORK="$M/main/rig" BUILD="$REPO/scripts/build_llamacpp_docker.sh" BRANCH="$FINAL_BRANCH" BASE_CPU="$BASE" REMOTE=fork \
            JOBS="$JOBS" LOCK="$LOCK" DOCKER_ARGS="$DA" STEPS=build \
            bash "$D/2026-11-11_c620_draft_kv_catchup/run/checks.sh" > "$M/logs/final.out" 2>&1
        log "final rc=$?"
        take final "$M/main/rig/summary.tsv" "$n"
        FT=$(tree "$FINAL_BRANCH")
        # the same commit built again from the pushed branch: every binary the checked tree's but for its name
        dif=$(cd "$FT/bin" && for f in *; do cmp -s "$f" "$T/bin/$f" || printf '%s ' "$f"; done)
        nb=$(ls "$FT/bin" | wc -l)
        add final "bin/ against the checked tree" "$([ -n "$FT" ] && [ -z "$dif" ] && [ "$nb" = "$(ls "$T/bin" | wc -l)" ] && echo SAME || echo DIFFERS)" \
            "$(basename "$FT") against $(basename "$T"): $nb files, differ: ${dif:-none}"
        rm -rf "$SRC/build" "$M/main/rig/extra"
    fi
fi

log "done"
date -Is > "$M/DONE"
