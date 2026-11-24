#!/usr/bin/env bash
# C992 (Q1725, 2026-11-24): test-backend-ops' TOPK_MOE_TAIL check, fixed for router ties. Fork branch
# pascal-p106-v5-topk-tail-tie (one commit on pascal-p106-v5, tests/test-backend-ops.cpp only) compares the cut ids of
# a batch past n_skip tokens only where a backend's cut weight is not 0, and draws each token's logits again until its
# n_expert_used + 1 best probabilities are a relative 2^-16 apart. Built and checked on a build machine with newer
# NVIDIA cards (compute 8.6), which run the rig build's compute 6.1 PTX through the driver's JIT. Correctness only:
# nothing here is timed.
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone that holds the base), the fork as remote "fork", the branch fetched
#           and checked against HEAD_SHA, its diff against the base listed; the compiler cache hard-linked from
#           RIG_CCACHE
#   rig     the rig tarball: scripts/build_llamacpp_docker.sh REF with TESTS=1 and EXTRA_TARGETS, the base's web page
#           (HF_UI_VERSION), under LOCK; its md5 and its binaries' md5s against the base's; warnings against BASE_LOG
#   check   on one card (GPU, after PREFLIGHT), test-backend-ops of the base (BASE/bin) and of the head, each with no
#           switch (unset) and with the fork's Pascal matrix-vector kernels forced as a P106 runs them (p106, as
#           docs/data/2026-11-23_c968_pascal_mmvq_suite did):
#             kept  C968's kept record of the failing case (KEPT, a folder holding cases/) replayed alone, with a live
#                   check on the restored inputs first (--cpu-check 1)
#             rec   every TOPK_MOE_TAIL and MOE_TAIL case recorded once with each binary (its inputs and the CPU
#                   reference; the record step's live check runs unset), then each record replayed with both binaries
#                   in both settings (--replay-log: per case the verdict and each compared node's error)
#             live  LIVE_REPS live runs of every TOPK_MOE_TAIL case (new random inputs in each run) per binary and
#                   setting, one container each
#   soak    SOAK_REPS more live runs per arm of SOAK_ARMS (<binary>-<setting>), into $W/check/soak-<arm>
#   tables  run/c992_tables.py over the outputs and the records, into $W/tables
# Usage: W=<work dir> SEED=<llama.cpp clone with the base> BASE=<the base's rig build folder> BASE_LOG=<its build log>
#   REPO=<a copy of this repository> [KEPT=<C968's topk_moe_tail folder>] [RIG_CCACHE=<dir>] [LOCK=<flock file>]
#   [PREFLIGHT=<cuda check, run as "$PREFLIGHT -g <id>">] [MOUNTS='<folders the GPU containers mount read-only>']
#   [STEPS=...] [JOBS=8] [GPU=2] [LIVE_REPS=40] [SOAK_REPS=300] [SOAK_ARMS='base-unset head-unset']
#   [JIT_SEED=<a PTX JIT cache to copy>] [NAME=c992] bash run.sh
# Every container: labels pascal-quants=1 and $NAME=1, --cpu-shares=256. Done when $W/DONE exists (it holds the exit
# code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with the base>}
BASE=${BASE:?BASE=<the base rig build folder>}
BASE_LOG=${BASE_LOG:?BASE_LOG=<its build log>}
REPO=${REPO:?REPO=<a copy of this repository>}
REF=${REF:-fork/pascal-p106-v5-topk-tail-tie}
HEAD_SHA=${HEAD_SHA:-b24c9a6751a2d45b4c0c8325522206bbb198471d}
BASE_REF=${BASE_REF:-fork/pascal-p106-v5}
BASE_SHA=${BASE_SHA:-af00bb13b0fd59612bcab545964b61978f04b8d6}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12483}
EXTRA_TARGETS=${EXTRA_TARGETS:-test-cuda-keepalive test-top-k-sort}
KEPT=${KEPT:-$REPO/docs/data/2026-11-23_c968_pascal_mmvq_suite/topk_moe_tail}
STEPS=${STEPS:-src rig check soak tables}
JOBS=${JOBS:-8}
GPU=${GPU:-2}
LIVE_REPS=${LIVE_REPS:-40}
SOAK_REPS=${SOAK_REPS:-300}
SOAK_ARMS=${SOAK_ARMS:-base-unset head-unset}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
PREFLIGHT=${PREFLIGHT:-}
NAME=${NAME:-c992}
D=$REPO/docs/data/2026-11-24_c992_topk_tail_tie/run
TOOLS=$REPO/scripts
LBL="--label pascal-quants=1 --label $NAME=1"
MEM="--memory=16g --memory-swap=16g"
# the failing case alone (C968's filter)
KEPT_F='^n_expert=256,n_tokens=9,n_expert_used=8,with_norm=1,gating_func=1,scale_w=2\.500000,p=0\.800000'
# the fork's Pascal matrix-vector kernels on with a P106's launch shapes (5 SMs, 96 KB of shared memory per SM)
ON="GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 \
GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304"
mkdir -p "$W/rig" "$W/logs" "$W/checks" "$W/jitcache" "$W/tables"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
say()    { echo "$(date +%H:%M:%S) $*"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter "label=$NAME=1" | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
strip()  { sed 's/\x1b\[[0-9;]*m//g' "$1"; }
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
echo "=== start $(date -Is): $REF ($HEAD_SHA) on $BASE_REF ($BASE_SHA); steps: $STEPS"

if step src; then
    S=$W/rig/src # where build_llamacpp_docker.sh (WORK=$W/rig) looks for its clone
    clone_seed "$S" || finish 1
    git -C "$S" fetch -q fork "${REF#fork/}:refs/remotes/$REF" || finish 1
    got=$(git -C "$S" rev-parse "refs/remotes/$REF")
    echo "$REF = $got"
    [ "$got" = "$HEAD_SHA" ] || { echo "$REF moved: stop"; finish 1; }
    git -C "$S" cat-file -e "$BASE_SHA^{commit}" || { echo "no $BASE_SHA in the clone: stop"; finish 1; }
    { git -C "$S" log --oneline "$BASE_SHA..$HEAD_SHA"; git -C "$S" diff --stat "$BASE_SHA" "$HEAD_SHA"; } > "$W/commits.txt"
    cat "$W/commits.txt"
    [ -d "$W/rig/ccache" ] || { [ -n "${RIG_CCACHE:-}" ] && cp -al "$RIG_CCACHE" "$W/rig/ccache"; }
    [ -n "${JIT_SEED:-}" ] && [ -z "$(ls -A "$W/jitcache")" ] && cp -a "$JIT_SEED/." "$W/jitcache/"
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
    (cd "$BASE/bin" && md5sum *) > "$W/rig/bin-md5-base.txt"
    echo "binaries: head $(wc -l < "$W/rig/bin-md5-head.txt"), base $(wc -l < "$W/rig/bin-md5-base.txt");"\
        "the same bytes: $(comm -12 <(sort "$W/rig/bin-md5-base.txt") <(sort "$W/rig/bin-md5-head.txt") | wc -l);"\
        "names only in one: $(diff <(awk '{print $2}' "$W/rig/bin-md5-base.txt") <(awk '{print $2}' "$W/rig/bin-md5-head.txt") | grep -c '^[<>]')"
    echo "binaries that differ: $(join -j 2 <(sort -k2 "$W/rig/bin-md5-base.txt") <(sort -k2 "$W/rig/bin-md5-head.txt") | awk '$2 != $3 {print $1}' | paste -sd' ')"
    grep -aE '^(name|commit|web page|avx|highest glibc|check tools):' "$T/BUILDINFO.txt"
    warnings "$BASE_LOG" > "$W/checks/rig-warnings-base.txt"
    warnings "$W/logs/rig.log" > "$W/checks/rig-warnings-head.txt"
    echo "rig warnings: base $(wc -l < "$W/checks/rig-warnings-base.txt") lines, head $(wc -l < "$W/checks/rig-warnings-head.txt"),"\
        "only in head: $(comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt" | wc -l)"
    comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt"
    echo "=== rig done $(date -Is)"
fi

C=$W/check
H=
VOL=(-v "$W:$W" -v "$BASE:$BASE:ro")
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
bin() { case $1 in base) echo "$BASE/bin/test-backend-ops" ;; head) echo "$H/bin/test-backend-ops" ;; esac; }
sw()  { case $1 in unset) echo "" ;; p106) echo "$ON" ;; esac; }
card_setup() {  # <log name>: the head build found, the card checked
    H=$(build_dir "$W/rig" "$HEAD_SHA")
    [ -n "$H" ] || { echo "no head build"; finish 1; }
    mkdir -p "$C"
    if [ -n "$PREFLIGHT" ]; then
        $PREFLIGHT -g "$GPU" > "$C/$1" 2>&1 || { echo "CUDA preflight failed on GPU $GPU"; cat "$C/$1"; finish 1; }
        echo "preflight GPU $GPU: ok"
    fi
}
# <tag> <binary> <setting> <runs>: live runs of every TOPK_MOE_TAIL case in one container, a log per run in $C/<tag>/
live_arm() {
    local tag=$1 b=$2 s=$3 n=$4
    rm -rf "${C:?}/$tag" && mkdir -p "$C/$tag"
    printf 'binary: %s\nenv: %s\nargs: test -b CUDA0 -o TOPK_MOE_TAIL, %s times\n' "$(bin "$b")" "$(sw "$s")" "$n" > "$C/$tag.cmd"
    timeout 7200 docker run --rm $LBL --cpu-shares=256 $MEM --name "$NAME-$tag" --gpus "\"device=$GPU\"" "${VOL[@]}" \
        -e "CUDA_CACHE_PATH=$W/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296 "$IMAGE" \
        bash -c "for i in \$(seq 1 $n); do env $(sw "$s") X=1 timeout 600 $(bin "$b") test -b CUDA0 -o TOPK_MOE_TAIL > $C/$tag/\$i.log 2>&1; echo \$i rc=\$? >> $C/$tag/rc.txt; done" \
        > /dev/null 2>&1
    say "  $tag: runs $(wc -l < "$C/$tag/rc.txt"), rc!=0 $(grep -vc 'rc=0$' "$C/$tag/rc.txt"), FAIL lines $(cat "$C/$tag"/*.log | sed 's/\x1b\[[0-9;]*m//g' | grep -a ': FAIL' | grep -avc Backend)"
}

if step check; then
    card_setup preflight.txt
    mkdir -p "$W/rec"
    rm -rf "$W/kept" && mkdir -p "$W/kept" && cp -a "$KEPT/." "$W/kept/"
    md5sum "$BASE/bin/test-backend-ops" "$H/bin/test-backend-ops" "$W"/kept/cases/* | tee "$C/md5.txt"
    # <tag> <binary: base|head> <setting: unset|p106> <shell command after "test-backend-ops test -b CUDA0">;
    # its output in $C/<tag>.log, its exit code in $C/rc.txt
    tbo() {
        local tag=$1 b=$2 s=$3 args=$4
        printf 'binary: %s\nenv: %s\nargs: test -b CUDA0 %s\n' "$(bin "$b")" "$(sw "$s")" "$args" > "$C/$tag.cmd"
        timeout 3600 docker run --rm $LBL --cpu-shares=256 $MEM --name "$NAME-$tag" --gpus "\"device=$GPU\"" "${VOL[@]}" \
            -e "CUDA_CACHE_PATH=$W/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296 "$IMAGE" \
            bash -c "env $(sw "$s") X=1 timeout 3000 $(bin "$b") test -b CUDA0 $args > $C/$tag.log 2>&1; echo $tag rc=\$? >> $C/rc.txt" \
            > /dev/null 2>&1
        say "  $tag: $(grep "^$tag " "$C/rc.txt" | tail -n 1); $(strip "$C/$tag.log" | grep -a 'tests passed' | tail -n 1 | sed 's/^ *//'); FAIL $(strip "$C/$tag.log" | grep -a -c ': FAIL')"
    }
    say "kept: C968's record of the failing case, replayed"
    for b in base head; do
        for s in unset p106; do
            tbo "kept-$b-$s" "$b" "$s" "-o TOPK_MOE_TAIL -p '$KEPT_F' --replay $W/kept --replay-allow-missing --cpu-check 1 --replay-log $C/kept-$b-$s.tsv"
        done
    done
    say "rec: every TOPK_MOE_TAIL and MOE_TAIL case recorded with each binary, replayed with both"
    for r in base head; do
        rm -rf "$W/rec/$r"
        tbo "record-$r" "$r" unset "-o TOPK_MOE_TAIL,MOE_TAIL --record $W/rec/$r"
    done
    for r in base head; do
        for b in base head; do
            for s in unset p106; do
                tbo "replay-$r-$b-$s" "$b" "$s" "-o TOPK_MOE_TAIL,MOE_TAIL --replay $W/rec/$r --replay-log $C/replay-$r-$b-$s.tsv"
            done
        done
    done
    say "live: $LIVE_REPS runs of every TOPK_MOE_TAIL case per binary and setting"
    for b in base head; do
        for s in unset p106; do
            live_arm "live-$b-$s" "$b" "$s" "$LIVE_REPS"
        done
    done
    echo "=== check done $(date -Is)"
fi

if step soak; then
    card_setup preflight-soak.txt
    say "soak: $SOAK_REPS more live runs of every TOPK_MOE_TAIL case for each of $SOAK_ARMS"
    for a in $SOAK_ARMS; do
        live_arm "soak-$a" "${a%-*}" "${a#*-}" "$SOAK_REPS"
    done
    echo "=== soak done $(date -Is)"
fi

if step tables; then
    python3 -I "$D/c992_tables.py" "$W" "$W/tables" || finish 1
    echo "=== tables done $(date -Is)"
fi
finish 0
