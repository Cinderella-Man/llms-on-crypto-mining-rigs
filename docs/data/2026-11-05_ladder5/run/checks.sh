#!/usr/bin/env bash
# C537 (Q1026): -ladder5, one fork branch holding every feature (the -ladder4 merge, C565's test fixes, C536's switches,
# C80's and C524's ports, C533's keep-alive), checked on a build machine, CPU only: every merged branch's published
# checks, run on the merged tree, one work folder per branch (the b12076 line against -ladder4, the upstream line against
# -b12127-ladder4). Steps, each the named script's with this tree:
#   flags     2026-11-05_switches/run/checks.sh: the base's static native CPU build (BASE_TARGETS; skipped when
#             $WORK/base-cpu/bin holds them), the rig tarball and the extra tests (EXTRA), the merged CPU build
#             (CPU_TARGETS), test-llama-archs identity with the base for the rig and CPU builds, every C536 switch flipped
#             against the base's logits, test-llama-context with each cache switch flipped, llama-server's checkpoint arms
#   portflip  the ports' switches that steer CUDA code (PORT_FLIPS) set together on the merged CPU build, against the base's
#             logits: no CUDA code runs, so nothing may change
#   union     2026-11-04_ladder4_merge/run/checks.sh: every line the merge changes against BASE is a line of a merged
#             branch (HEADS) or of the merge's own commits (MERGE_OWN); conflict resolutions show up for review
#   ladder4   the same script: -ladder4's branches' checks on the merged CPU build (moeskip coldtier combo adraft units)
#   keepalive 2026-11-05_keepalive/run/checks.sh: GGML_CUDA_KEEPALIVE=decode on the rig build without a device, and
#             test-cuda-keepalive three times
#   c80       2026-11-05_c80_port/run/checks.sh with BUILD_OF: an AVX2 CPU build of the merge (in $WORK/c80), then
#             -promptpath's checks (text reuse, with the tail checkpoint spacing, the decode mask bound; they include
#             -textreuse's and -decodemask's) and -cheapguess's (the staircase drafter, every option)
#   c524      2026-11-05_v2_ports/run/checks.sh with ONE: the sm_61 SASS of every kernel against the base's rig build,
#             test-quantize-fns -v (Q4_1_64), the screened head with a grammar on a tool-calling model (MODEL, TASKS)
# The c80 and c524 servers use fixed ports and container names: SERVER_LOCK keeps two runs from overlapping there.
# Usage: WORK=<dir> REPO=<this repository> BRANCH=<fork branch> BASE=<base commit> BASE_BUILD=<the base's unpacked rig
#   tarball, extras in bin/> TEXT=<text file> TINY=<tiny Flint gguf> CASE=<drafter check case> MODEL=<gguf with
#   output_screen> TASKS=<BFCL tasks.jsonl> HEADS='<name>=<tip> ...' [MERGE_OWN='<commit> ...'] [REMOTE=fork]
#   [JOBS=8] [THREADS=8] [LOCK=<flock file>] [SERVER_LOCK=<flock file>] [DOCKER_ARGS=...] [MOUNTS='<dirs>']
#   [PY_IMAGE=python:3.12-slim] [STEPS='flags portflip union ladder4 keepalive c80 c524'] bash checks.sh
#   WORK      the work folder of scripts/build_llamacpp_docker.sh: its llama.cpp clone in $WORK/src has the fork as remote
#             REMOTE with an https fetch URL, BRANCH and the HEADS' tips fetched; $WORK/ccache
# Output: as the scripts above, in $WORK ($WORK/summary.tsv: check, verdict, detail; $WORK/c80/summary.tsv and
#   $WORK/c524/summary.tsv: branch, check, verdict, detail); done when $WORK/LADDER5_DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${BRANCH:?BRANCH=<fork branch>}" "${BASE:?BASE=<base commit>}" "${BASE_BUILD:?BASE_BUILD=<base rig tree>}"
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
STEPS=${STEPS:-flags portflip union ladder4 keepalive c80 c524}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
BASE_TARGETS=${BASE_TARGETS:-llama-server test-llama-archs test-llama-context test-backend-ops}
CPU_TARGETS=${CPU_TARGETS:-llama-server llama-perplexity llama-quantize test-backend-ops test-llama-archs test-llama-context \
test-bytes-switches test-moe-ids test-spec-adapt test-spec-block test-spec-dump test-draft-mem test-quantize-fns test-hop-cast \
test-route-once test-cuda-keepalive}
export EXTRA=${EXTRA:-test-llama-archs test-llama-context test-draft-mem test-hop-cast test-bytes-switches test-moe-ids \
test-spec-adapt test-hop-mailbox test-route-once test-cuda-keepalive test-quantize-fns}
PORT_FLIPS=${PORT_FLIPS:-GGML_CUDA_MMVQ_PASCAL_MT=1 GGML_CUDA_MMVQ_PASCAL_F16=1 GGML_CUDA_MMQ_MOE_LEGACY=2 \
GGML_CUDA_MMVQ_MAX_COLS=4 GGML_CUDA_KEEPALIVE=decode}
DATA=$REPO/docs/data
D="--cpu-shares=256 ${DOCKER_ARGS:-}"
mkdir -p "$WORK/logs" "$WORK/checks"
rm -f "$WORK/LADDER5_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
serverlocked() { if [ -n "${SERVER_LOCK:-}" ]; then flock -w 21600 "$SERVER_LOCK" "$@"; else "$@"; fi; }
step() { [[ " $STEPS " == *" $1 "* ]]; }
tarball_tree() { local t; t=$(ls -t "$WORK"/llama.cpp-*-"$BRANCH".tar.gz 2>/dev/null | head -n 1); [ -n "$t" ] && echo "$WORK/$(basename "$t" .tar.gz)"; }
common=(WORK="$WORK" REPO="$REPO" BRANCH="$BRANCH" REMOTE="$REMOTE" BASE_BUILD="$BASE_BUILD" BASE_CPU="$WORK/base-cpu/bin"
        JOBS="$JOBS" THREADS="$THREADS" LOCK="${LOCK:-}" DOCKER_ARGS="${DOCKER_ARGS:-}" MOUNTS="${MOUNTS:-}")
git -C "$WORK/src" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$WORK/src" rev-parse "$REMOTE/$BRANCH")
echo "=== ladder5 checks started $(date -Is), $BRANCH at ${TIP:0:9} on ${BASE:0:9}, steps: $STEPS"

if step flags; then
    echo "=== flags $(date -Is)"
    have=1
    for t in $BASE_TARGETS; do [ -x "$WORK/base-cpu/bin/$t" ] || have=0; done
    if [ $have = 0 ]; then
        env "${common[@]}" BASE="$BASE" CPU_TARGETS="$BASE_TARGETS" STEPS=basecpu \
            bash "$DATA/2026-11-05_switches/run/checks.sh" > "$WORK/logs/flags-basecpu.out" 2>&1
    else
        verdict basecpu OK "kept: $WORK/base-cpu/bin ($BASE_TARGETS)"
    fi
    env "${common[@]}" BASE="$BASE" CPU_TARGETS="$CPU_TARGETS" STEPS='build cpubuild identity flips context server' \
        bash "$DATA/2026-11-05_switches/run/checks.sh" > "$WORK/logs/flags.out" 2>&1
fi

if step portflip; then
    echo "=== portflip $(date -Is)"
    K=$WORK/checks/flips
    if [ ! -s "$K/base/n_base.txt" ]; then
        verdict portflip FAIL "no base logits in $K/base (run the flags step first)"
    else
        e=""
        for a in $PORT_FLIPS; do e="$e -e $a"; done
        # shellcheck disable=SC2086
        docker run --rm $D -v "$WORK:$WORK" $e -e LLAMA_ARCHS_LOGITS=/k/logits- -v "$K/base:/k" "$IMAGE" \
            nice -n 19 "$WORK/cpu/bin/test-llama-archs" -b CPU -s 1 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$K/ports.log"
        nc=$(grep -c 'logits against' "$K/ports.log")
        bad=$(grep 'logits against' "$K/ports.log" | grep -v ', 0 of ' | grep -vxFf "$K/base/base_self.txt" | cut -d: -f1 | tr '\n' ' ')
        sb=$(grep -m 1 '^Summary' "$K/base/base.log") sm=$(grep -m 1 '^Summary' "$K/ports.log")
        verdict "flip ports" "$([ "$nc" -gt 0 ] && [ -z "$bad" ] && { [ "$sb" = "$sm" ] || [[ "$sm" == 'Summary: all '* ]]; } && echo SAME || echo DIFFERS)" \
            "$PORT_FLIPS: $nc compared, differ: ${bad:-none}; base: $sb; flipped: $sm"
    fi
fi

for s in union ladder4; do
    step $s || continue
    echo "=== $s $(date -Is)"
    st=$([ $s = union ] && echo union || echo 'moeskip coldtier combo adraft units')
    env "${common[@]}" BASE="$BASE" BRANCHES="${HEADS:?HEADS=<name>=<tip> ...}" MERGE_OWN="${MERGE_OWN:- }" TEXT="${TEXT:-}" \
        CPU_TARGETS="$CPU_TARGETS" STEPS="$st" \
        bash "$DATA/2026-11-04_ladder4_merge/run/checks.sh" > "$WORK/logs/ladder4-$s.out" 2>&1
done

if step keepalive; then
    echo "=== keepalive $(date -Is)"
    env "${common[@]}" STEPS='switchon unit' bash "$DATA/2026-11-05_keepalive/run/checks.sh" > "$WORK/logs/keepalive.out" 2>&1
fi

if step c80; then
    echo "=== c80 $(date -Is)"
    C=$WORK/c80
    rm -rf "$C/cpu-merge" "$C/checks" && mkdir -p "$C"
    [ -d "$C/src/.git" ] || git clone -q --local --no-checkout "$WORK/src" "$C/src"
    git -C "$C/src" checkout -q --force "$TIP" && git -C "$C/src" clean -qfdx
    c80=(WORK="$C" REPO="$REPO" BASE_BUILD="$BASE_BUILD" TINY="${TINY:?TINY=<tiny Flint gguf>}" CASE="${CASE:?CASE=<drafter case>}"
         BRANCHES='promptpath cheapguess' BUILD_OF=merge RIG_TREE="$(tarball_tree)" JOBS="$JOBS" LOCK="${LOCK:-}"
         DOCKER_ARGS="${DOCKER_ARGS:-}" PY_IMAGE="${PY_IMAGE:-python:3.12-slim}")
    env "${c80[@]}" STEPS=cpubuild bash "$DATA/2026-11-05_c80_port/run/checks.sh" > "$WORK/logs/c80-cpubuild.out" 2>&1
    serverlocked env "${c80[@]}" STEPS=own bash "$DATA/2026-11-05_c80_port/run/checks.sh" > "$WORK/logs/c80-own.out" 2>&1
    while IFS=$'\t' read -r b c v d; do [ "$b" = branch ] || verdict "c80 $b $c" "$v" "$d"; done < "$C/summary.tsv"
fi

if step c524; then
    echo "=== c524 $(date -Is)"
    R=$WORK/c524
    rm -rf "$R" && mkdir -p "$R"
    c524=(ROOT="$R" REPO="$REPO" BASE_TREE="$BASE_BUILD" ONE="$WORK" ONE_BRANCH="$BRANCH" JOBS="$JOBS" THREADS="$THREADS"
          DOCKER_ARGS="${DOCKER_ARGS:-}")
    env "${c524[@]}" BRANCHES=q41g64 STEPS='sass units' bash "$DATA/2026-11-05_v2_ports/run/checks.sh" > "$WORK/logs/c524.out" 2>&1
    serverlocked env "${c524[@]}" BRANCHES=toolsample-screen STEPS=screen MODEL="${MODEL:?MODEL=<gguf with output_screen>}" \
        TASKS="${TASKS:?TASKS=<BFCL tasks.jsonl>}" bash "$DATA/2026-11-05_v2_ports/run/checks.sh" > "$WORK/logs/c524-screen.out" 2>&1
    while IFS=$'\t' read -r b c v d; do [ "$b" = branch ] || verdict "c524 $b $c" "$v" "$d"; done < "$R/summary.tsv"
fi

date -Is > "$WORK/LADDER5_DONE"
echo "=== done $(date -Is)"
cat "$WORK/summary.tsv"
