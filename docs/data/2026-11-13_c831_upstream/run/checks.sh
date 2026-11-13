#!/usr/bin/env bash
# C831 (2026-11-13): upstream llama.cpp b12360 merged into Miner Llama's v4 line (pascal-p106-v4 80c94fe0d) on
# pascal-p106-v4-b12360-up, built once and checked on a build machine, CPU only. Made from C820's checks.sh
# (docs/data/2026-11-13_c820_online_skip_row) and C662's (docs/data/2026-11-10_ladder6). Every build and test runs in a
# container without GPUs, waits for LOCK (flock) and runs at a low CPU share (--cpu-shares=256, nice 19).
# Steps (STEPS, in this order by default):
#   union   git only: the lines the tip changes against the base that are neither upstream's nor OWN's commits' (the
#           merge's resolutions and fixes), per file
#   cpu     a static native CPU build of the branch's tip (GGML_CUDA=OFF, every target) in $M/cpu; every binary in
#           $M/cpu/bin
#   bcpu    the same of BASE_COMMIT (the v4 line before the merge), llama-server and test-llama-archs, in $M/bcpu
#   rcpu    the tip with the upstream commits REVERT undone (git apply -R of each one's diff), llama-server and
#           test-llama-archs, in $M/rcpu: what the base computes, if those commits explain every difference
#   model   the random qwen4exp MoE model with one MTP block (the tip's test-llama-archs -o --mtp)
#   off     test-llama-archs -b CPU -s 1 on every architecture: the base's build writes each model's logits, the tip
#           compares with every switch off (REVIEW: the models upstream's changes reach) and the revert build compares
#           (SAME when REVERT explains every difference)
#   self    each build against its own logits with the line's switches on together (ALL, the CUDA ones included,
#           which change nothing on a CPU): the base and the tip, SAME expected on both
#   modes   the fork's own test-llama-archs checks (MODES: the MTP draft, the caches, the prepared decode, the n-gram
#           prefetch) on the tip, each passing as its row's checks did
#   server  C820's server_check.sh on the random MTP model: every MTP drafting group (C815's online rule, C742's and
#           C820's skipped rows, C744's cap) on the tip's llama-server, dbase the base's; rserver: dbase and doff again
#           with the revert build's llama-server as the branch's
#   draftmd the random model with its MTP block as a separate draft file (-md, Flash-Next's draft settings) in the tip's and
#           the base's llama-server: each context's graph budget (upstream #32078 sizes it by tensor count only) and nodes
#   rig     the rig tarball of the tip (scripts/build_llamacpp_docker.sh, TESTS=1, EXTRA_TARGETS: sm_61, no AVX) in
#           $M/main/rig
#   rigbase the same build of BASE_COMMIT in the same folder
#   names   C770's switch list in both rig builds (build/common/switch-names.cpp): the counts and the names that differ
#   sass    cuobjdump -sass and -ptx of both builds' bin/llama-server, anonymous-namespace hashes masked, cut into
#           functions (sass_attrib.py dump), then run/sass_diff.py: every function same, changed, new or gone, by kernel
#           family (REVIEW: upstream changed kernels)
#   c565    2026-11-05_c565_ladder4fix/run/cpu_tests.sh, static and then shared: ctest (every registered test, the
#           fork's unit tests and test-backend-ops among them), the network test, archs, the hand tests
#   trim    the CPU build trees removed (the binaries stay)
# Before the c565 builds the chain waits (up to 2 hours) until the disk has ROOM_GB free (52 by default).
# Env: M (the work root), REPO (this repository as run), LOCK, BASE_COMMIT (80c94fe0d), UPSTREAM (b12360), REVERT
#   (default 7f24aacde: upstream #29810, the CPU's flash attention accumulating V in F32), SEED_SRC (a llama.cpp clone
#   with the fork as remote "fork", https), SEED_CCACHE and SEED_CPU_CCACHE (optional), BRANCH (default
#   pascal-p106-v4-b12360-up), JOBS (8), THREADS (8), DOCKER_ARGS, IMAGE, CPU_IMAGE, SRV_IMAGE (CPU_IMAGE with curl and
#   jq), EXTRA_TARGETS (default "test-cuda-keepalive test-llama-archs"), STEPS
# Output: $M/summary.tsv (row, check, verdict, detail), $M/off, $M/self, $M/modes, $M/server, $M/sass, $M/logs; done
#   when $M/DONE exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
BRANCH=${BRANCH:-pascal-p106-v4-b12360-up}
BASE_COMMIT=${BASE_COMMIT:-80c94fe0d}
UPSTREAM=${UPSTREAM:-b12360}
REVERT=${REVERT:-7f24aacde}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
SRV_IMAGE=${SRV_IMAGE:-pq-c735-srv:24.04}
EXTRA_TARGETS=${EXTRA_TARGETS:-test-cuda-keepalive test-llama-archs}
STEPS=${STEPS:-union cpu bcpu rcpu model off self modes server draftmd rig rigbase names sass c565 trim}
ROOM_GB=${ROOM_GB:-52}
MODES=${MODES:---kv-shift --ngram-prefetch --mtp-kv-only --mtp-catchup-overlap --mtp-hrow-lazy --kv-seq-rm --mtp-graph-keep --mtp-replan --verify-prepare --mtp-h-dev --mtp-incall-overlap}
# the line's switches on together: the host ones that compute what the defaults compute, and the CUDA ones
ALL="LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_CATCHUP_OVERLAP=1 LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 \
LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_MTP_GRAPH_KEEP=1 \
LLAMA_VERIFY_PREPARE=2 LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_NGRAM_PREFETCH=1 LLAMA_DRAFT_PREFETCH=1 \
LLAMA_DRAFT_EMBD_DEV=layer LLAMA_MTP_H_DEV=2 GGML_SCHED_PREFETCH=0 LLAMA_MTP_EMBD_FIRST=1 LLAMA_MTP_DRAFT_CAP=1 \
LLAMA_MTP_SKIP_LAST_ROW=2 LLAMA_SCHED_PER_SIZE=1 LLAMA_SCHED_PER_SIZE_RESERVE=2 LLAMA_PP_MULTI_SPLIT=1 \
LLAMA_MTP_HROW_LAZY=1 LLAMA_SPEC_ADAPT=1 LLAMA_SPEC_ADAPT_ONLINE=2 GGML_CUDA_MMVQ_PASCAL_DMT=1 \
GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 GGML_CUDA_MMVQ_PASCAL_Q3K_MT=1 GGML_CUDA_MMVQ_PASCAL_GLU_MT=3 \
GGML_CUDA_MMVF_PASCAL_EXACT=1 GGML_CUDA_MMVF_F32_MT=1 GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1 \
GGML_CUDA_FATTN_SPARSE_PASCAL_UNION=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_GRAPHS_PATCH=1 GGML_CUDA_MMVQ_PASCAL_Q5K_1R=1 \
GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1 GGML_CUDA_MMVQ_PASCAL_DMT_KT=1 GGML_CUDA_MMVQ_PASCAL_RO8=1 \
GGML_CUDA_MMVQ_PASCAL_RO8_KT=1 GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT=1 GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1"
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
D=$REPO/docs/data
RUN=$D/2026-11-13_c831_upstream/run
SRVCHK=$D/2026-11-13_c820_online_skip_row/run/server_check.sh
ATTRIB=$D/2026-11-11_v4_day_merge/run/sass_attrib.py
SRC=$M/main/rig/src
MASK='s/_GLOBAL__N__[0-9a-f]{8}_/_GLOBAL__N__X_/g;s/_cu_[0-9a-f]{8}/_cu_X/g;s/_INTERNAL_[0-9a-f]{8}/_INTERNAL_X/g'
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'row\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
log()    { echo "$(date -Is) $*" | tee -a "$M/logs/c831.log"; }
locked() { flock -w 43200 "$LOCK" "$@"; }
add()    { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a "$M/summary.tsv"; }
tree()   { ls -td "$M"/main/rig/llama.cpp-*-"$1" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
free_gb() { df -BG --output=avail "$M" | tail -n 1 | tr -dc '0-9'; }
# shellcheck disable=SC2086
box()    { locked docker run --rm $DA --network none -v "$M:$M" "$@"; }
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
# <dir> <commit> <ccache> <targets or "all"> [<commits to revert>]: a static native CPU build in <dir>/tree, the named
# binaries (all: every one) into <dir>/bin
cpu_build() {
    local B=$1 commit=$2 cc=$3 targets=$4 rev=${5:-} tgt="" r
    rm -rf "$B/tree" "$B/bin" && mkdir -p "$B/tree/src" "$B/bin"
    git -C "$SRC" archive "$commit" | tar -x -C "$B/tree/src"
    echo "commit $commit $(git -C "$SRC" log -1 --format=%s "$commit")" > "$B/tree.txt"
    for r in $rev; do
        git -C "$SRC" diff "$r^" "$r" | git -C "$B/tree/src" apply -R || { echo "revert $r failed" >> "$B/tree.txt"; return 1; }
        echo "reverted $r $(git -C "$SRC" log -1 --format=%s "$r")" >> "$B/tree.txt"
    done
    [ "$targets" = all ] || tgt="--target $targets"
    box -e CCACHE_DIR="$cc" -e CCACHE_BASEDIR="$B/tree" "$CPU_IMAGE" bash -c '
        cmake -S '"$B/tree/src"' -B '"$B/tree/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$B/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$B/tree/build"' -j '"$JOBS"' '"$tgt"' -- -k > '"$B/build.log"' 2>&1'
}
built() { local b; for b in "$@"; do [ -x "$b" ] || return 1; done; }

if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$M/main/rig"
    git clone -q --local --no-checkout "${SEED_SRC:?SEED_SRC=<a llama.cpp clone with the fork as remote fork>}" "$SRC" &&
        git -C "$SRC" remote add fork "$(git -C "$SEED_SRC" remote get-url fork)" &&
        git -C "$SRC" remote set-url origin "$(git -C "$SEED_SRC" remote get-url origin)"
fi
git -C "$SRC" fetch -q fork "$BRANCH:refs/remotes/fork/$BRANCH" || { echo "no fork/$BRANCH"; exit 1; }
git -C "$SRC" fetch -q origin "refs/tags/$UPSTREAM:refs/tags/$UPSTREAM" 2> /dev/null
TIP=$(git -C "$SRC" rev-parse "fork/$BRANCH")
BASE_FULL=$(git -C "$SRC" rev-parse "$BASE_COMMIT^{commit}") || { echo "no $BASE_COMMIT"; exit 1; }
UP_FULL=$(git -C "$SRC" rev-parse "$UPSTREAM^{commit}") || { echo "no $UPSTREAM"; exit 1; }
git -C "$SRC" merge-base --is-ancestor "$BASE_FULL" "$TIP" || { echo "$BASE_COMMIT is not in $BRANCH"; exit 1; }
git -C "$SRC" merge-base --is-ancestor "$UP_FULL" "$TIP" || { echo "$UPSTREAM is not in $BRANCH"; exit 1; }
for img in "$CPU_IMAGE" "$SRV_IMAGE"; do
    docker image inspect "$img" > /dev/null 2>&1 || { echo "no image $img"; exit 1; }
done
log "C831 checks: $BRANCH at ${TIP:0:9}, base ${BASE_FULL:0:9}, upstream $UPSTREAM ${UP_FULL:0:9}, steps: $STEPS"
add start "$BRANCH" OK "tip ${TIP:0:9} ($(git -C "$SRC" log -1 --format=%s "$TIP" | cut -c1-100)); against the base: $(git -C "$SRC" diff --stat "$BASE_FULL" "$TIP" | tail -n 1); upstream commits merged: $(git -C "$SRC" rev-list --count "$BASE_FULL..$UP_FULL")"

C=$M/cpu
BC=$M/bcpu
RC=$M/rcpu
# git only (2026-11-04_ladder4_merge's union step): every line the tip changes against the base is a line upstream
# changed since its merge base with the base, or one of OWN's commits after the merge; the lines of neither are the
# merge's resolutions and fixes, listed per file in $M/union/diff for review
if step union; then
    U=$M/union
    rm -rf "$U" && mkdir -p "$U/diff"
    ulines() { git -C "$SRC" diff -U0 "$1" "$2" -- "$3" | grep -a "^$4" | grep -av "^$4$4$4 " | LC_ALL=C sort -u; }
    printf 'file\tsources\tadded_extra\tadded_missing\tremoved_extra\tremoved_missing\n' > "$U/union.tsv"
    MB=$(git -C "$SRC" merge-base "$BASE_FULL" "$UP_FULL")
    nbad=0
    for f in $(git -C "$SRC" diff --name-only "$BASE_FULL" "$TIP"); do
        bs=""
        : > "$U/u_add" && : > "$U/u_rem"
        if ! git -C "$SRC" diff --quiet "$MB" "$UP_FULL" -- "$f"; then
            bs="upstream"
            ulines "$MB" "$UP_FULL" "$f" + >> "$U/u_add"
            ulines "$MB" "$UP_FULL" "$f" - >> "$U/u_rem"
        fi
        for c in ${OWN:-}; do
            git -C "$SRC" diff --quiet "$c^" "$c" -- "$f" && continue
            bs="$bs own:${c:0:9}"
            ulines "$c^" "$c" "$f" + >> "$U/u_add"
            ulines "$c^" "$c" "$f" - >> "$U/u_rem"
        done
        LC_ALL=C sort -u -o "$U/u_add" "$U/u_add"; LC_ALL=C sort -u -o "$U/u_rem" "$U/u_rem"
        ulines "$BASE_FULL" "$TIP" "$f" + > "$U/m_add"; ulines "$BASE_FULL" "$TIP" "$f" - > "$U/m_rem"
        ae=$(LC_ALL=C comm -23 "$U/m_add" "$U/u_add" | wc -l); am=$(LC_ALL=C comm -13 "$U/m_add" "$U/u_add" | wc -l)
        re=$(LC_ALL=C comm -23 "$U/m_rem" "$U/u_rem" | wc -l); rm_=$(LC_ALL=C comm -13 "$U/m_rem" "$U/u_rem" | wc -l)
        if [ "$ae$re" != 00 ]; then
            nbad=$((nbad + 1))
            { echo "== $f (sources: ${bs# })"; echo "-- added by the merge, by no source:"; LC_ALL=C comm -23 "$U/m_add" "$U/u_add"
              echo "-- removed by the merge, by no source:"; LC_ALL=C comm -23 "$U/m_rem" "$U/u_rem"; } > "$U/diff/$(echo "$f" | tr / _).txt"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$f" "${bs# }" "$ae" "$am" "$re" "$rm_" >> "$U/union.tsv"
    done
    rm -f "$U"/u_add "$U"/u_rem "$U"/m_add "$U"/m_rem
    nf=$(($(wc -l < "$U/union.tsv") - 1))
    add union "lines of neither upstream nor OWN" "$([ $nbad = 0 ] && echo OK || echo REVIEW)" \
        "$nf files changed against ${BASE_FULL:0:9}; $nbad hold lines of neither (added $(awk -F'\t' 'NR > 1 { s += $3 } END { print s + 0 }' "$U/union.tsv"), removed $(awk -F'\t' 'NR > 1 { s += $5 } END { print s + 0 }' "$U/union.tsv")): $(ls "$U/diff" | sed 's/\.txt$//' | tr '\n' ' ')"
fi

if step cpu; then
    [ -d "$C/ccache" ] || { if [ -n "${SEED_CPU_CCACHE:-}" ]; then mkdir -p "$C" && cp -al "$SEED_CPU_CCACHE" "$C/ccache"; else mkdir -p "$C/ccache"; fi; }
    cpu_build "$C" "$TIP" "$C/ccache" all
    rc=$?
    cp "$C"/tree/build/bin/* "$C/bin/" 2> /dev/null
    add cpu build "$([ $rc = 0 ] && built "$C/bin/test-llama-archs" "$C/bin/llama-server" "$C/bin/test-backend-ops" && echo OK || echo FAIL)" \
        "rc=$rc; $(head -n 1 "$C/tree.txt" | cut -c1-120); static, every target ($(ls "$C/bin" | wc -l) binaries); $(grep -ac 'warning:' "$C/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$C/build.log") error lines"
    [ $rc = 0 ] || { date -Is > "$M/DONE"; exit 1; }
fi

if step bcpu; then
    cpu_build "$BC" "$BASE_FULL" "$C/ccache" "llama-server test-llama-archs"
    rc=$?
    cp "$BC/tree/build/bin/llama-server" "$BC/tree/build/bin/test-llama-archs" "$BC/bin/" 2> /dev/null
    add bcpu build "$([ $rc = 0 ] && built "$BC/bin/llama-server" "$BC/bin/test-llama-archs" && echo OK || echo FAIL)" \
        "rc=$rc; $(head -n 1 "$BC/tree.txt" | cut -c1-120); static; $(grep -acE 'error:|undefined reference' "$BC/build.log") error lines"
    rm -rf "$BC/tree"
fi

if step rcpu; then
    cpu_build "$RC" "$TIP" "$C/ccache" "llama-server test-llama-archs" "$REVERT"
    rc=$?
    cp "$RC/tree/build/bin/llama-server" "$RC/tree/build/bin/test-llama-archs" "$RC/bin/" 2> /dev/null
    add rcpu build "$([ $rc = 0 ] && built "$RC/bin/llama-server" "$RC/bin/test-llama-archs" && echo OK || echo FAIL)" \
        "rc=$rc; $(tr '\n' ';' < "$RC/tree.txt" | cut -c1-300); $(grep -acE 'error:|undefined reference' "$RC/build.log" 2> /dev/null) error lines"
    rm -rf "$RC/tree"
fi

S=$M/server
if step model; then
    rm -rf "$S/models" && mkdir -p "$S/models"
    box "$CPU_IMAGE" bash -c "nice -n 19 $C/bin/test-llama-archs -a '^qwen4exp\$' -s 1 -o $S/models --mtp-draft > $S/models/save.log 2>&1"
    rc=$?
    add model "qwen4exp with an MTP block" "$([ $rc = 0 ] && [ -s "$S/models/qwen4exp-moe-mtp.gguf" ] && [ -s "$S/models/qwen4exp-moe-mtp-draft.gguf" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(ls -l "$S/models/qwen4exp-moe-mtp.gguf" 2> /dev/null | awk '{ print $5 " B" }'); md5 $(md5sum < "$S/models/qwen4exp-moe-mtp.gguf" | cut -d' ' -f1); the draft alone $(ls -l "$S/models/qwen4exp-moe-mtp-draft.gguf" 2> /dev/null | awk "{ print \$5 \" B\" }")"
fi

# <dir> <bin> <label>: <bin>'s test-llama-archs writes every architecture's logits into <dir>/ref
archs_ref() {
    local A=$1 B=$2 l=$3 rc nb
    rm -rf "$A/ref" && mkdir -p "$A/ref"
    box "$CPU_IMAGE" bash -c \
        'env LLAMA_ARCHS_LOGITS='"$A"'/ref/logits- nice -n 19 '"$B"'/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A/ref-$l.log"'; exit ${PIPESTATUS[0]}'
    rc=$?
    nb=$(ls "$A"/ref/logits-*.f32 2> /dev/null | wc -l)
    add "$(basename "$A")" "$l writes" "$([ $rc = 0 ] && [ "$nb" -gt 0 ] && echo OK || echo FAIL)" "rc=$rc; $nb logits files; $(grep -a -m 1 '^Summary' "$A/ref-$l.log")"
}
# <dir> <bin> <tag> <expect SAME|REVIEW> [<env>]: <bin>'s test-llama-archs against the logits in <dir>/ref
archs_cmp() {
    local A=$1 B=$2 tag=$3 want=$4 envs=${5:-} rc n bad wrote v
    box "$CPU_IMAGE" bash -c \
        'env LLAMA_ARCHS_LOGITS='"$A"'/ref/logits- '"$envs"' nice -n 19 '"$B"'/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A/$tag.log"'; exit ${PIPESTATUS[0]}'
    rc=$?
    n=$(grep -ac 'logits against' "$A/$tag.log")
    grep -a 'logits against' "$A/$tag.log" | grep -v ', 0 of ' > "$A/$tag.differ.txt"
    bad=$(cut -d: -f1 "$A/$tag.differ.txt" | tr '\n' ' ')
    wrote=$(grep -ac 'logits_dump_or_compare: wrote' "$A/$tag.log")
    if [ "$rc" = 0 ] && [ "$n" -gt 0 ] && [ -z "$bad" ] && [ "$wrote" = 0 ]; then v=SAME
    elif [ "$want" = REVIEW ] && [ "$rc" = 0 ] && [ "$n" -gt 0 ]; then v=REVIEW
    else v=DIFFERS; fi
    add "$(basename "$A")" "$tag" "$v" \
        "rc=$rc; ${envs:-no switch}; $n compared, $(wc -l < "$A/$tag.differ.txt") differ: ${bad:-none}; written by this build only: $wrote; $(grep -a -m 1 '^Summary' "$A/$tag.log")"
}

if step off; then
    archs_ref "$M/off" "$BC/bin" base
    archs_cmp "$M/off" "$C/bin" tip REVIEW
    archs_cmp "$M/off" "$RC/bin" revert REVIEW
    grep -a 'logits against' "$M/off/tip.log" > "$M/off/tip.lines.txt"
    grep -a 'logits against' "$M/off/revert.log" > "$M/off/revert.lines.txt"
    rm -f "$M"/off/ref/logits-*.f32
fi

if step self; then
    archs_ref "$M/self-base" "$BC/bin" base
    archs_cmp "$M/self-base" "$BC/bin" base-all SAME "$ALL"
    rm -f "$M"/self-base/ref/logits-*.f32
    archs_ref "$M/self" "$C/bin" tip
    archs_cmp "$M/self" "$C/bin" tip-all SAME "$ALL"
    rm -f "$M"/self/ref/logits-*.f32
fi

if step modes; then
    mkdir -p "$M/modes"
    for mode in $MODES; do
        tag=${mode#--}
        box "$CPU_IMAGE" bash -c 'nice -n 19 '"$C"'/bin/test-llama-archs '"$mode"' -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$M/modes/$tag.log"'; exit ${PIPESTATUS[0]}'
        rc=$?
        add modes "$mode" "$([ $rc = 0 ] && echo OK || echo FAIL)" \
            "rc=$rc; $(grep -aiE 'summary|passed|checks|same' "$M/modes/$tag.log" | tail -n 2 | cut -c1-200 | tr '\n' ';')"
    done
fi

if step server; then
    rm -rf "$S/out" "$S/rout"
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$M:$M" -v "$REPO:$REPO:ro" "$SRV_IMAGE" bash -c \
        "BASE=$BC/bin BIN=$C/bin MODEL=$S/models/qwen4exp-moe-mtp.gguf OUT=$S/out THREADS=$THREADS \
         SARGS='--override-kv qwen4exp.context_length=int:16384' nice -n 19 bash $SRVCHK" > "$S/console.txt" 2>&1
    rc=$?
    add server "server_check.sh, every group" "$([ $rc = 0 ] && tail -n 1 "$S/out/verdict.txt" | grep -qx PASS && echo PASS || echo FAIL)" \
        "rc=$rc; $(grep -c ' = ' "$S/out/verdict.txt" 2> /dev/null) arms equal to their reference; $(grep -vE ' = |^PASS$' "$S/out/verdict.txt" 2> /dev/null | cut -c1-160 | head -n 8 | tr '\n' ';')"
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$M:$M" -v "$REPO:$REPO:ro" "$SRV_IMAGE" bash -c \
        "BASE=$BC/bin BIN=$RC/bin MODEL=$S/models/qwen4exp-moe-mtp.gguf OUT=$S/rout THREADS=$THREADS ARMS='dbase doff' \
         SARGS='--override-kv qwen4exp.context_length=int:16384' nice -n 19 bash $SRVCHK" > "$S/rconsole.txt" 2>&1
    rc=$?
    add rserver "dbase doff, the revert build" "$([ $rc = 0 ] && tail -n 1 "$S/rout/verdict.txt" | grep -qx PASS && echo PASS || echo FAIL)" \
        "rc=$rc; $(grep -E '^dbase' "$S/rout/verdict.txt" 2> /dev/null | cut -c1-200 | tr '\n' ';')"
    log "server done"
fi

# upstream #32078 sizes a context's graph budget by its tensor count only (no longer by -ub): a separate MTP draft file
# (Flash-Next's preset: -md, --spec-draft-n-ubatch 128) gets far fewer nodes than before. Each build serves the random
# model with its MTP block as a separate draft at the preset's draft settings, one greedy request; every context's
# budget (max_nodes) and its graphs' nodes are logged (-lv 5)
if step draftmd; then
    DM=$M/draftmd
    rm -rf "$DM" && mkdir -p "$DM"
    # shellcheck disable=SC2016
    locked docker run --rm $DA --network none -v "$M:$M" "$SRV_IMAGE" bash -c '
        one() {
            local O='"$DM"'/$1 pid i
            mkdir -p "$O"
            nice -n 19 "$2"/llama-server -m '"$S"'/models/qwen4exp-moe-mtp.gguf -md '"$S"'/models/qwen4exp-moe-mtp-draft.gguf \
                --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0 --spec-draft-n-ubatch 128 -b 2048 -ub 512 \
                -c 8192 -np 1 -t '"$THREADS"' --host 127.0.0.1 --port 18831 -lv 5 \
                --override-kv qwen4exp.context_length=int:16384 > "$O/server.log" 2>&1 &
            pid=$!
            for i in $(seq 1 300); do curl -sf http://127.0.0.1:18831/health > /dev/null && break; kill -0 $pid 2> /dev/null || break; sleep 1; done
            curl -s http://127.0.0.1:18831/completion -H "Content-Type: application/json" \
                -d "{\"prompt\": [$(seq -s, 10 137)], \"n_predict\": 48, \"temperature\": 0, \"cache_prompt\": false}" > "$O/reply.json"
            kill $pid 2> /dev/null; wait $pid
            echo "$1 rc=$?"
        }
        one base '"$BC/bin"'
        one tip '"$C/bin"'' > "$DM/rc.txt" 2>&1
    for l in base tip; do
        L=$DM/$l/server.log
        ntok=$(jq -r '.tokens_predicted // empty' "$DM/$l/reply.json" 2> /dev/null)
        add draftmd "$l: -md draft at -ub 512, draft ubatch 128" "$([ -n "$ntok" ] && [ "$ntok" -gt 0 ] && ! grep -aqE 'GGML_ASSERT|not enough space' "$L" && echo OK || echo FAIL)" \
            "$(grep "^$l rc=" "$DM/rc.txt"); tokens predicted: ${ntok:-none}; drafted/accepted: $(jq -r '[.timings.draft_n, .timings.draft_n_accepted] | map(tostring) | join("/")' "$DM/$l/reply.json" 2> /dev/null); max_nodes lines: $(grep -aoE 'max_nodes = [0-9]+' "$L" | sort | uniq -c | tr -s ' ' | tr '\n' ';'); graphs: $(grep -aoE 'graph( \(pp bs=[0-9]+, tg bs=[0-9]+\))?: nodes = [0-9 /]+' "$L" | tr '\n' ';')"
    done
    cmp -s <(jq -c '.tokens' "$DM/base/reply.json" 2> /dev/null) <(jq -c '.tokens' "$DM/tip/reply.json" 2> /dev/null)
    add draftmd "the replies' tokens" "$([ $? = 0 ] && echo SAME || echo DIFFERS)" "base against tip, greedy, $(jq -r '.content | length' "$DM/tip/reply.json" 2> /dev/null) characters"
fi

rig_build() {   # <ref> <log>
    locked env WORK="$M/main/rig" TESTS=1 EXTRA_TARGETS="$EXTRA_TARGETS" JOBS="$JOBS" DOCKER_ARGS="$DA" nice -n 19 \
        bash "$REPO/scripts/build_llamacpp_docker.sh" "$1" > "$2" 2>&1
}
if step rig; then
    [ -d "$M/main/rig/ccache" ] || [ -z "${SEED_CCACHE:-}" ] || cp -al "$SEED_CCACHE" "$M/main/rig/ccache"
    rig_build "fork/$BRANCH" "$M/logs/rig.log"
    rc=$?
    T=$(tree "$BRANCH")
    add rig build "$([ $rc = 0 ] && [ -n "$T" ] && [ -x "$T/bin/llama-server" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(basename "$T").tar.gz md5 $(md5sum < "$T.tar.gz" | cut -d' ' -f1), $(stat -c %s "$T.tar.gz") B; $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';'); bin/: $(cd "$T/bin" && for f in llama-server test-backend-ops $EXTRA_TARGETS; do echo -n "$f $(md5sum < "$f" | cut -d' ' -f1) "; done); $(grep -ac 'warning' "$M/logs/rig.log") warning lines"
    [ $rc = 0 ] || { date -Is > "$M/DONE"; exit 1; }
fi
T=$(tree "$BRANCH")

if step names; then
    L=$SRC/build/common/switch-names.cpp
    cp "$L" "$M/logs/names-tip.txt" 2> /dev/null
fi

if step rigbase; then
    rig_build "$BASE_FULL" "$M/logs/rigbase.log"
    rc=$?
    TB=$(tree "${BASE_FULL:0:9}")
    add rigbase build "$([ $rc = 0 ] && [ -n "$TB" ] && [ -x "$TB/bin/llama-server" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(basename "$TB").tar.gz md5 $(md5sum < "$TB.tar.gz" | cut -d' ' -f1), $(stat -c %s "$TB.tar.gz") B; $(grep -E '^(commit)' "$TB/BUILDINFO.txt" | tr '\n' ';'); llama-server md5 $(md5sum < "$TB/bin/llama-server" | cut -d' ' -f1)"
fi
TB=$(tree "${BASE_FULL:0:9}")

if step names; then
    cp "$SRC/build/common/switch-names.cpp" "$M/logs/names-base.txt" 2> /dev/null
    nt=$(grep -c '^    "' "$M/logs/names-tip.txt" 2> /dev/null)
    nb=$(grep -c '^    "' "$M/logs/names-base.txt" 2> /dev/null)
    dn=$(diff <(grep '^    "' "$M/logs/names-base.txt" | sort) <(grep '^    "' "$M/logs/names-tip.txt" | sort) | grep '^[<>]' | tr -d ' ",' | tr '\n' ' ')
    add names "switch list" "$([ -n "$nt" ] && [ "$nt" -gt 0 ] && echo OK || echo FAIL)" "the tip's build reads $nt names, the base's $nb; differ: ${dn:-none}"
fi

if step sass; then
    SD=$M/sass
    rm -rf "$SD" && mkdir -p "$SD"
    dumpbin() {   # <label> <llama-server>: its SASS and PTX cut into functions
        local d w
        d=$(dirname "$2")
        for w in sass ptx; do
            # shellcheck disable=SC2086
            locked docker run --rm $DA --network none -v "$d:$d:ro" "$IMAGE" nice -n 19 cuobjdump -$w "$2" \
                | sed -E "$MASK" | nice -n 19 python3 "$ATTRIB" dump $w > "$SD/$1.$w.tsv"
        done
    }
    dumpbin tip "$T/bin/llama-server"
    dumpbin base "$TB/bin/llama-server"
    for w in sass ptx; do
        python3 "$RUN/sass_diff.py" "$SD/tip.$w.tsv" "$SD/base.$w.tsv" "$SD/diff_$w.tsv" > "$SD/diff_$w.txt" 2>&1
        add sass "llama-server $w against the base" "$(head -n 1 "$SD/diff_$w.txt" | grep -q 'changed 0, new 0, gone 0' && echo SAME || echo REVIEW)" \
            "$(head -n 1 "$SD/diff_$w.txt"); $(sed -n '2,13p' "$SD/diff_$w.txt" | sed 's/^ *//' | tr '\n' ';')"
    done
    log "sass done"
fi

if step c565; then
    for sh in OFF ON; do
        v=$([ $sh = OFF ] && echo static || echo shared)
        room "c565 $v"
        env SRC="$SRC" WORK="$M/c565-$v" TREES="c831=$TIP" STEPS='tree build ctest net archs hand' SHARED=$sh JOBS="$JOBS" \
            THREADS="$THREADS" LOCK="$LOCK" DOCKER_ARGS="--label pascal-quants=1 ${DOCKER_ARGS:-}" nice -n 19 \
            bash "$D/2026-11-05_c565_ladder4fix/run/cpu_tests.sh" > "$M/logs/c565-$v.out" 2>&1
        log "c565 $v rc=$?"
        tail -n +2 "$M/c565-$v/summary.tsv" 2> /dev/null | awk -F'\t' -v r="c565 $v" 'BEGIN { OFS = "\t" }
            NF >= 4 { d = $4; for (i = 5; i <= NF; i++) d = d " " $i; print r, $1 " " $2, $3, d; next } { print r, $1, $2, $3 }' \
            | tee -a "$M/summary.tsv"
        rm -rf "$M/c565-$v/tree"
    done
fi

if step trim; then
    rm -rf "$C/tree" "$BC/tree" "$RC/tree"
    add trim "CPU trees" OK "removed (the binaries stay); $(du -sh "$M" | cut -f1) in $M; $(free_gb) GB free"
fi

log "done"
date -Is > "$M/DONE"
