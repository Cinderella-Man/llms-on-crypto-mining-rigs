#!/usr/bin/env bash
# C1024 (Q1757, 2026-11-24): GGML_CUDA_MMQ_J64_I64, the switch that gives the prompt kernel's 64-column tile (J=64) its
# 64 rows back on the DP4A tile table that compute 6.1 runs (upstream #31352, merged with b12674, gave it 128 rows for
# Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ2_XXS, IQ3_XXS, IQ4_NL and IQ4_XS). Fork branch pascal-p106-v5-mmq-j64-tile,
# one commit on the v5 candidate pascal-p106-v5-cand-b12674 (eee615cdb). Built and checked on a build machine with
# newer NVIDIA cards: they run the rig build's compute 6.1 PTX through the driver's JIT, and since the build has no newer
# code, its prompt products take the same DP4A table a P106 takes. Correctness only: nothing here is timed.
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone that holds the candidate), the fork as remote "fork", the branch
#           fetched and checked against HEAD_SHA, its diff against the candidate listed; the compiler cache hard-linked
#           from RIG_CCACHE. The reference patch $W/ref/patches/j64-rows64.diff: the candidate's DP4A table with the 22
#           J=64 rows that #31352 raised to 128 rows set back to 64 (the commit itself no longer reverts cleanly: the
#           fork's types sit next to its lines)
#   rig     the rig tarball: scripts/build_llamacpp_docker.sh REF with TESTS=1 and EXTRA_TARGETS, the web page pinned
#           (HF_UI_VERSION), under LOCK; its md5 and its binaries' md5s; warnings against CAND_RIG_LOG
#   ref     the reference: the candidate (CAND_REF) with the reference patch, TESTS=0, in $W/ref with a hard-linked copy
#           of the rig step's compiler cache: what the candidate's kernels are without #31352
#   sass    cuobjdump -sass and -ptx of three llama-servers: the candidate's (CAND_RIG), the head's and the
#           reference's; one SHA-1 per kernel copy (sass_compare.py hash, from docs/data/2026-11-17_c861a_fp32_scale/run/)
#           and per PTX function (ptx_functions.py, from docs/data/2026-11-24_c985_topk_order/run/); j64_sass.py: every
#           candidate kernel copy in the head, the head's extra copies, those against the reference's J=64 copies of the
#           11 types, and the reference against the candidate
#   bops    test-backend-ops test -b CUDA0 on the head's build with a P106's Pascal paths forced (P106_ENV, as C968 ran
#           them): MUL_MAT (every case but the repeated-expert-id ones, dup=1, as C912 ran them) and MUL_MAT_ID (the 11
#           types' cases, ID_FILTER: the switch reaches no other type) with the switch unset on the first card of GPUS
#           and with GGML_CUDA_MMQ_J64_I64=2 on the second, at the same time; then MUL_MAT's C1024 cases (J64_FILTER)
#           with =1, which on a card other than compute 6.1 changes no launch. BOPS_ARMS picks arms (default all five;
#           an arm's earlier line in rc.txt is replaced). Before it starts it waits until no other container holds one
#           of the cards (up to GPU_WAIT
#           seconds) and runs PREFLIGHT on them
#   tables  run/c1024_tables.py over the outputs, into $W/tables
# Usage: W=<work dir> SEED=<llama.cpp clone with the candidate> CAND_RIG=<the candidate's rig build folder>
#   CAND_RIG_LOG=<its build log> REPO=<a copy of this repository> [RIG_CCACHE=<dir>] [LOCK=<flock file>]
#   [PREFLIGHT=<cuda check, run as "$PREFLIGHT -g <ids>">] [MOUNTS='<folders the GPU containers mount read-only>']
#   [STEPS=...] [JOBS=8] [GPUS=0,1] [JIT_SEED=<a PTX JIT cache to copy>] [NAME=c1024] bash run.sh
# Every container: labels pascal-quants=1 and $NAME=1, --cpu-shares=256; the cuobjdump containers without network.
# Done when $W/DONE exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with the candidate>}
CAND_RIG=${CAND_RIG:?CAND_RIG=<the candidate rig build folder>}
CAND_RIG_LOG=${CAND_RIG_LOG:?CAND_RIG_LOG=<its build log>}
REPO=${REPO:?REPO=<a copy of this repository>}
REF=${REF:-fork/pascal-p106-v5-mmq-j64-tile}
HEAD_SHA=${HEAD_SHA:-ebec6544e3690ff8cbf557ac7e4c3fd46344d15a}
CAND_REF=${CAND_REF:-fork/pascal-p106-v5-cand-b12674}
CAND_SHA=${CAND_SHA:-eee615cdb9d26c917de01e3c49f794c41de31684}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12674}
EXTRA_TARGETS=${EXTRA_TARGETS:-test-cuda-keepalive test-top-k-sort}
STEPS=${STEPS:-src rig ref sass bops tables}
JOBS=${JOBS:-8}
GPUS=${GPUS:-0,1}
GPU_WAIT=${GPU_WAIT:-10800}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
PREFLIGHT=${PREFLIGHT:-}
NAME=${NAME:-c1024}
P106_ENV=${P106_ENV:-GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304}
# the C1024 block of test-backend-ops' MUL_MAT cases: the 11 types at k=512 (and the nonzero-mean cases at k=1024)
ID_FILTER=${ID_FILTER:-^(?!.*dup=1)type_a=(q1_0|q2_0|q4_0|q4_1|q5_0|q5_1|q8_0|iq2_xxs|iq3_xxs|iq4_nl|iq4_xs),}
BOPS_ARMS=${BOPS_ARMS:-mul_mat-unset mul_mat-j64i64 mul_mat_id-unset mul_mat_id-j64i64 mul_mat-j64-mode1}
J64_FILTER=${J64_FILTER:-type_a=(q1_0|q2_0|q4_0|q4_1|q5_0|q5_1|q8_0|iq2_xxs|iq3_xxs|iq4_nl|iq4_xs),type_b=f32,m=(64|128|192|256|1000|4096),n=(49|64|65|128|512),k=(512|1024),}
D=$REPO/docs/data/2026-11-24_c1024_mmq_j64_tile/run
SASS_PY=$REPO/docs/data/2026-11-17_c861a_fp32_scale/run/sass_compare.py
PTX_PY=$REPO/docs/data/2026-11-24_c985_topk_order/run/ptx_functions.py
CLASSIFY_PY=$REPO/docs/data/2026-11-24_c912_upstream_b12674/run/sass_classify.py
TOOLS=$REPO/scripts
LBL="--label pascal-quants=1 --label $NAME=1"
MEM="--memory=16g --memory-swap=16g"
VOL=(-v "$W:$W")
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
IFS=, read -ra GA <<< "$GPUS"
mkdir -p "$W/rig" "$W/ref/patches" "$W/logs" "$W/checks" "$W/jitcache" "$W/tables"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
say()    { echo "$(date +%H:%M:%S) $*"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter "label=$NAME=1" | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
export LC_ALL=C
warnings() { grep -aE 'warning:|warning #[0-9]|CMake Warning' "$1" \
    | sed -E 's/:[0-9]+:[0-9]+: warning/: warning/; s/\([0-9]+\): warning/: warning/; s/CMakeLists.txt:[0-9]+/CMakeLists.txt/' | sort -u; }
build_dir() { ls -d "$1"/llama.cpp-*-"${2:0:9}"-* 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1; }
# a clone of SEED at $1 with upstream as origin and the fork as "fork" (build_llamacpp_docker.sh fetches both)
clone_seed() {
    local S=$1
    if [ ! -d "$S/.git" ]; then
        git clone -q --no-checkout "$SEED" "$S" || return 1
        git -C "$S" fetch -q "$SEED" 'refs/remotes/*:refs/remotes/*'
        git -C "$S" fetch -q "$SEED" 'refs/tags/*:refs/tags/*'
        git -C "$S" remote set-url origin https://github.com/ggml-org/llama.cpp
    fi
    git -C "$S" remote get-url fork > /dev/null 2>&1 || git -C "$S" remote add fork "$FORK_URL"
    git -C "$S" remote set-url fork "$FORK_URL"
}
# running containers, not ours, that hold one of the cards in GPUS (--gpus by index or UUID, or every card)
holders() {
    local want=",$GPUS," i
    for i in ${GPUS//,/ }; do want+="$(nvidia-smi -i "$i" --query-gpu=uuid --format=csv,noheader 2>/dev/null),"; done
    docker ps -q | while read -r c; do
        docker inspect -f '{{.Name}} L{{index .Config.Labels "'"$NAME"'"}} {{range .HostConfig.DeviceRequests}}{{.Count}}:{{join .DeviceIDs ","}} {{end}}' "$c"
    done | awk -v want="$want" '$2 != "L1" { for (i = 3; i <= NF; i++) { split($i, p, ":");
        if (p[1] == "-1") { print $1; next } n = split(p[2], ids, ",");
        for (j = 1; j <= n; j++) if (index(want, "," ids[j] ",")) { print $1; next } } }'
}
echo "=== start $(date -Is): $REF ($HEAD_SHA) on $CAND_REF ($CAND_SHA); steps: $STEPS"

if step src; then
    S=$W/rig/src # where build_llamacpp_docker.sh (WORK=$W/rig) looks for its clone
    clone_seed "$S" || finish 1
    git -C "$S" fetch -q fork "${REF#fork/}:refs/remotes/$REF" "${CAND_REF#fork/}:refs/remotes/$CAND_REF" || finish 1
    got=$(git -C "$S" rev-parse "refs/remotes/$REF"); cand=$(git -C "$S" rev-parse "refs/remotes/$CAND_REF")
    echo "$REF = $got; $CAND_REF = $cand"
    [ "$got" = "$HEAD_SHA" ] || { echo "$REF moved: stop"; finish 1; }
    [ "$cand" = "$CAND_SHA" ] || { echo "$CAND_REF moved: stop"; finish 1; }
    { git -C "$S" log --oneline "$CAND_SHA..$HEAD_SHA"; git -C "$S" diff --stat "$CAND_SHA" "$HEAD_SHA"; } > "$W/commits.txt"
    cat "$W/commits.txt"
    [ -d "$W/rig/ccache" ] || { [ -n "${RIG_CCACHE:-}" ] && cp -al "$RIG_CCACHE" "$W/rig/ccache"; }
    # the reference patch: the 22 rows "CASE(<type>, 256, 2, 128,  64, ...)" of the DP4A table back to 64 rows
    F=ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh
    git -C "$S" show "$CAND_SHA:$F" > "$W/ref/table-cand.cuh"
    sed -E 's/^(    CASE\(GGML_TYPE_[A-Z0-9_]+, 256, 2, )128(,  64, )/\1 64\2/' "$W/ref/table-cand.cuh" > "$W/ref/table-ref.cuh"
    diff -u --label "a/$F" --label "b/$F" "$W/ref/table-cand.cuh" "$W/ref/table-ref.cuh" > "$W/ref/patches/j64-rows64.diff"
    echo "reference patch: $(grep -c '^-    CASE' "$W/ref/patches/j64-rows64.diff") rows changed, types:" \
        "$(grep '^+    CASE' "$W/ref/patches/j64-rows64.diff" | sed -E 's/.*GGML_TYPE_([A-Z0-9_]+),.*/\1/' | sort -u | tr '\n' ' ')"
    echo "=== sources ready $(date -Is)"
fi

if step rig; then
    rm -f "$W/rig/RC"
    locked env WORK="$W/rig" TESTS=1 JOBS="$JOBS" EXTRA_TARGETS="$EXTRA_TARGETS" \
        DOCKER_ARGS="$LBL --cpu-shares=256 $MEM -e HF_UI_VERSION=$HF_UI_VERSION" \
        bash "$TOOLS/build_llamacpp_docker.sh" "$REF" > "$W/logs/rig.log" 2>&1
    echo "rc=$?" | tee "$W/rig/RC"
    T=$(build_dir "$W/rig" "$HEAD_SHA")
    [ -n "$T" ] && [ -f "$T.tar.gz" ] || { echo "no rig tarball"; finish 1; }
    (cd "$(dirname "$T")" && md5sum "$(basename "$T").tar.gz" && stat -c '%s bytes' "$(basename "$T").tar.gz") | tee "$W/rig/md5.txt"
    (cd "$T/bin" && md5sum *) > "$W/rig/bin-md5-head.txt"
    (cd "$CAND_RIG/bin" && md5sum *) > "$W/rig/bin-md5-cand.txt"
    echo "binaries: head $(wc -l < "$W/rig/bin-md5-head.txt"), candidate $(wc -l < "$W/rig/bin-md5-cand.txt");"\
        "names only in one: $(diff <(awk '{print $2}' "$W/rig/bin-md5-cand.txt") <(awk '{print $2}' "$W/rig/bin-md5-head.txt") | grep -c '^[<>]')"
    grep -aE '^(name|commit|web page|avx|highest glibc|check tools):' "$T/BUILDINFO.txt"
    warnings "$CAND_RIG_LOG" > "$W/checks/rig-warnings-cand.txt"
    warnings "$W/logs/rig.log" > "$W/checks/rig-warnings-head.txt"
    echo "rig warnings: candidate $(wc -l < "$W/checks/rig-warnings-cand.txt") lines, head $(wc -l < "$W/checks/rig-warnings-head.txt"),"\
        "only in head: $(comm -13 "$W/checks/rig-warnings-cand.txt" "$W/checks/rig-warnings-head.txt" | wc -l)"
    comm -13 "$W/checks/rig-warnings-cand.txt" "$W/checks/rig-warnings-head.txt"
    echo "=== rig done $(date -Is)"
fi

if step ref; then
    R=$W/ref
    clone_seed "$R/src" || finish 1
    [ -d "$R/ccache" ] || cp -al "$W/rig/ccache" "$R/ccache"
    rm -f "$R/RC"
    locked env WORK="$R" TESTS=0 JOBS="$JOBS" PATCHES=patches/j64-rows64.diff \
        DOCKER_ARGS="$LBL --cpu-shares=256 $MEM -e HF_UI_VERSION=$HF_UI_VERSION" \
        bash "$TOOLS/build_llamacpp_docker.sh" "$CAND_REF" > "$W/logs/ref.log" 2>&1
    echo "rc=$?" | tee "$R/RC"
    T=$(build_dir "$R" "$CAND_SHA")
    [ -n "$T" ] && [ -x "$T/bin/llama-server" ] || { echo "no reference build"; finish 1; }
    grep -aE '^(name|commit|patch):' "$T/BUILDINFO.txt"
    md5sum "$T/bin/llama-server" | tee "$R/md5.txt"
    rm -f "$T.tar.gz" # only its llama-server is needed
    echo "=== ref done $(date -Is)"
fi

if step sass; then
    S=$W/checks/sass
    rm -rf "$S" && mkdir -p "$S/bin"
    H=$(build_dir "$W/rig" "$HEAD_SHA"); R=$(build_dir "$W/ref" "$CAND_SHA")
    # the three llama-servers side by side (hard links where the file system allows), mounted read-only
    for x in C H R; do
        case $x in C) b=$CAND_RIG/bin/llama-server ;; H) b=$H/bin/llama-server ;; R) b=$R/bin/llama-server ;; esac
        ln -f "$b" "$S/bin/$x-llama-server" 2> /dev/null || cp "$b" "$S/bin/$x-llama-server"
        echo "$x $(md5sum < "$S/bin/$x-llama-server" | cut -c1-32) $b" >> "$S/md5.txt"
    done
    cuo() { docker run --rm $LBL --cpu-shares=256 --network none -v "$S/bin:$S/bin:ro" "$IMAGE" cuobjdump "$1" "$S/bin/$2-llama-server"; }
    for x in C H R; do
        cuo -sass "$x" | nice -n 19 c++filt > "$S/dump-$x.txt"
        cuo -ptx "$x" > "$S/ptx-$x.txt"
        nice -n 19 python3 -I "$SASS_PY" hash < "$S/dump-$x.txt" > "$S/h-$x.tsv"
        # the J=64 kernels of the 11 types, whole, for a line-by-line look where two hashes differ
        nice -n 19 python3 -I "$D/j64_sass.py" extract < "$S/dump-$x.txt" > "$S/j64-$x.txt"
        say "sass $x: $(($(wc -l < "$S/h-$x.tsv") - 1)) kernel copies"
    done
    # every kernel of the candidate against the head's and the reference's, as C912's sass_check.sh compares two builds:
    # names whose copies differ, and for those whether only MOV32I constants differ (a printed source line moves when
    # lines are added above it in mmq.cuh) or the code (sass_classify.py, from docs/data/2026-11-24_c912_upstream_b12674/run/)
    for p in H R; do
        O=$S/C$p; mkdir -p "$O"
        awk -F'\t' 'NR == 1 || $2 != "true"' "$S/h-C.tsv" > "$O/h-B-off.tsv"
        nice -n 19 python3 -I "$SASS_PY" compare "$O/h-B-off.tsv" "$S/h-$p.tsv" "$O/cmp" > "$O/summary.txt" 2>&1
        awk -F'\t' '$4 == "NO" {print $1}' "$O/cmp.identity.tsv" > "$O/differ.list"
        awk -F'\t' '$4 == "missing in new" {print $1}' "$O/cmp.identity.tsv" > "$O/missing.list"
        awk -F'\t' 'FNR == NR {if (FNR > 1 && $2 != "true") b[$1] = 1; next} FNR > 1 && $2 != "true" && !($1 in b) {print $1}' \
            "$S/h-C.tsv" "$S/h-$p.tsv" | sort -u > "$O/only_new.list"
        nice -n 19 python3 -I "$SASS_PY" extract "$O/differ.list" < "$S/dump-C.txt" > "$O/ext-B.txt"
        nice -n 19 python3 -I "$SASS_PY" extract "$O/differ.list" < "$S/dump-$p.txt" > "$O/ext-N.txt"
        nice -n 19 python3 -I "$SASS_PY" linediff "$O/ext-B.txt" "$O/ext-N.txt" > "$O/linediff.txt" 2>&1
        nice -n 19 python3 -I "$CLASSIFY_PY" "$O/h-B-off.tsv" "$S/h-$p.tsv" "$O/ext-B.txt" "$O/ext-N.txt" \
            > "$O/classes.tsv" 2> "$O/classes_summary.txt"
        echo "--- candidate against $p: $(cat "$O/summary.txt" | head -n 1)"
        echo "    names whose copies differ $(wc -l < "$O/differ.list"), missing $(wc -l < "$O/missing.list"), only in $p $(wc -l < "$O/only_new.list")"
        head -n 3 "$O/linediff.txt" | sed 's/^/    /'; sed 's/^/    /' "$O/classes_summary.txt" | head -n 12
        rm -f "$O/ext-B.txt" "$O/ext-N.txt"
    done | tee "$S/pairs.txt"
    nice -n 19 python3 -I "$D/j64_sass.py" compare "$S" | tee "$S/summary.txt"
    {
        echo "--- PTX, candidate against head"; nice -n 19 python3 -I "$PTX_PY" "$S/ptx-C.txt" "$S/ptx-H.txt"
        echo "--- PTX, candidate against reference"; nice -n 19 python3 -I "$PTX_PY" "$S/ptx-C.txt" "$S/ptx-R.txt"
    } 2>&1 | cut -c1-300 | tee "$S/ptx.txt"
    rm -f "$S"/dump-?.txt "$S"/ptx-?.txt "$S"/bin/?-llama-server
    echo "=== sass done $(date -Is)"
fi

if step bops; then
    T=$(build_dir "$W/rig" "$HEAD_SHA")
    o=$W/bops; mkdir -p "$o"; touch "$o/rc.txt"
    [ -n "$T" ] || { echo "no head build"; finish 1; }
    if [ -n "${JIT_SEED:-}" ] && [ -d "$JIT_SEED" ] && [ -z "$(ls -A "$W/jitcache")" ]; then cp -a "$JIT_SEED/." "$W/jitcache/"; fi
    md5sum "$T/bin/test-backend-ops" > "$o/md5.txt"
    t0=$(date +%s)
    while h=$(holders) && [ -n "$h" ]; do
        [ $(( $(date +%s) - t0 )) -lt "$GPU_WAIT" ] || { say "cards $GPUS still held by: $h"; finish 1; }
        sleep 30
    done
    say "cards $GPUS free of other containers (waited $(( $(date +%s) - t0 )) s)"
    if [ -n "$PREFLIGHT" ]; then $PREFLIGHT -g "$GPUS" > "$o/preflight.txt" 2>&1; rc=$?; say "preflight on cards $GPUS rc=$rc"
        [ $rc = 0 ] || finish 1; fi
    DR=(docker run --rm $LBL --cpu-shares=256 "${VOL[@]}" -e "CUDA_CACHE_PATH=$W/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)
    run_bops() {  # <card> <tag> <operators> <params filter> <environment>
        [[ " $BOPS_ARMS " == *" $2 "* ]] || return 0
        sed -i "/^$2 rc=/d" "$o/rc.txt"
        "${DR[@]}" --name "$NAME-$2" --gpus "\"device=$1\"" "$IMAGE" \
            bash -c "env $5 nice -n 19 timeout 10800 $T/bin/test-backend-ops test -b CUDA0 -o '$3' -p '$4' 2>&1 \
                | sed 's/\x1b\[[0-9;]*m//g' > $o/$2.log; echo $2 rc=\${PIPESTATUS[0]} >> $o/rc.txt" > /dev/null 2>&1
        say "bops $2 done: $(grep "^$2 " "$o/rc.txt")"
    }
    F='^(?!.*dup=1)'
    say "bops: $T on cards $GPUS"
    ( run_bops "${GA[0]}" mul_mat-unset MUL_MAT "$F" "$P106_ENV"
      run_bops "${GA[0]}" mul_mat_id-unset MUL_MAT_ID "$ID_FILTER" "$P106_ENV" ) &
    ( run_bops "${GA[1]}" mul_mat-j64i64 MUL_MAT "$F" "$P106_ENV GGML_CUDA_MMQ_J64_I64=2"
      run_bops "${GA[1]}" mul_mat_id-j64i64 MUL_MAT_ID "$ID_FILTER" "$P106_ENV GGML_CUDA_MMQ_J64_I64=2" ) &
    wait
    run_bops "${GA[0]}" mul_mat-j64-mode1 MUL_MAT "$J64_FILTER" "$P106_ENV GGML_CUDA_MMQ_J64_I64=1"
    echo "=== bops done $(date -Is)"
fi

if step tables; then
    nice -n 19 python3 -I "$D/c1024_tables.py" "$W" "$W/tables" | tee "$W/tables/summary.txt"
    echo "=== tables done $(date -Is)"
fi
finish 0
