#!/usr/bin/env bash
# C662 (Q1150): -ladder6, the v3.0 release tree of Miner Llama with a newer upstream llama.cpp release merged in,
# checked on a build machine, CPU only. The checks are the merged features' own published scripts, run on this tree.
# Two bases: the release's builds, for what the merge changes (the logits of every architecture, the kernels' code),
# and the branch's own build at its defaults, for what each switch changes. Steps:
#   union     2026-11-04_ladder4_merge/run/checks.sh: every line the branch changes against the release (BASE) is
#             upstream's (UPSTREAM against its merge base) or a commit of the branch after the merge (OWN); the lines of
#             neither are the merge's resolutions, listed for review
#   build     the same script: the rig tarball (sm_61 code and compute_61 PTX, no AVX, TESTS=1) and the extra tests
#   cpubuild  the same script: the branch's static native CPU build (CPU_TARGETS) into $WORK/cpu/bin
#   release   test-llama-archs -b CPU on every architecture, the logits written by a base build and compared by the
#             branch's: the release's rig build (BASE_BUILD) against this tarball's tree, the release's CPU build
#             (BASE_CPU) and the last upstream line's (PREV_CPU, optional) against $WORK/cpu/bin. The architectures that
#             differ are the ones upstream changed: REVIEW lists them; OK when none differs
#   selfbase  $WORK/base-cpu/bin becomes a copy of $WORK/cpu/bin, so that the switch checks below compare a flipped
#             switch with the branch's own defaults
#   flips context server   2026-11-05_switches/run/checks.sh, those steps
#   portflip ladder4 keepalive   2026-11-05_ladder5/run/checks.sh, those steps, against the branch's own rig tree
#   c80 c524  the same script, against the release's rig tree (the staircase drafter's step lines; the sm_61 code of
#             every kernel)
#   c565      2026-11-05_c565_ladder4fix/run/cpu_tests.sh on the branch's tip, static then shared: ctest, the test with
#             network, the archs runs and the hand tests
# Usage: WORK=<dir> REPO=<this repository> BRANCH=<fork branch> BASE=<the release's commit> UPSTREAM=<the merged
#   upstream commit> BASE_BUILD=<the release's unpacked rig tarball, extras in bin/> BASE_CPU=<bin/ of the release's
#   static native CPU build> [PREV_CPU=<bin/ of the last upstream line's CPU build>] [OWN='<commit> ...'] TEXT=<text>
#   TINY=<tiny Flint gguf> CASE=<drafter case> MODEL=<gguf with output_screen> TASKS=<BFCL tasks.jsonl> [REMOTE=fork]
#   [JOBS=8] [THREADS=8] [LOCK=<flock file>] [SERVER_LOCK=<flock file>] [DOCKER_ARGS=...] [PY_IMAGE=python:3.12-slim]
#   [STEPS='union build cpubuild release selfbase flips context server portflip ladder4 keepalive c80 c524 c565'] bash checks.sh
#   WORK      the work folder of scripts/build_llamacpp_docker.sh: its llama.cpp clone in $WORK/src has the fork as
#             remote REMOTE with an https fetch URL and upstream's tags; $WORK/ccache
# Output: as the scripts above, in $WORK ($WORK/summary.tsv: check, verdict, detail; $WORK/c565-{static,shared}/
#   summary.tsv); done when $WORK/LADDER6_DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${BRANCH:?BRANCH=<fork branch>}" "${BASE:?BASE=<release commit>}" "${UPSTREAM:?UPSTREAM=<merged upstream commit>}"
: "${BASE_BUILD:?BASE_BUILD=<release rig tree>}" "${BASE_CPU:?BASE_CPU=<release CPU bin/>}"
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
STEPS=${STEPS:-union build cpubuild release selfbase flips context server portflip ladder4 keepalive c80 c524 c565}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_TARGETS=${CPU_TARGETS:-llama-server llama-perplexity llama-quantize test-backend-ops test-llama-archs test-llama-context \
test-bytes-switches test-moe-ids test-spec-adapt test-spec-block test-spec-dump test-draft-mem test-quantize-fns test-hop-cast \
test-route-once test-cuda-keepalive}
export EXTRA=${EXTRA:-test-llama-archs test-llama-context test-draft-mem test-hop-cast test-bytes-switches test-moe-ids \
test-spec-adapt test-hop-mailbox test-route-once test-cuda-keepalive test-quantize-fns}
DATA=$REPO/docs/data
D="--cpu-shares=256 ${DOCKER_ARGS:-}"
mkdir -p "$WORK/logs" "$WORK/checks"
rm -f "$WORK/LADDER6_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
step() { [[ " $STEPS " == *" $1 "* ]]; }
steps_of() { local o="" s; for s in "$@"; do step "$s" && o="$o $s"; done; echo "${o# }"; }
tarball_tree() { local t; t=$(ls -t "$WORK"/llama.cpp-*-"$BRANCH".tar.gz 2>/dev/null | head -n 1); [ -n "$t" ] && echo "$WORK/$(basename "$t" .tar.gz)"; }
git -C "$WORK/src" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$WORK/src" rev-parse "$REMOTE/$BRANCH")
common=(WORK="$WORK" REPO="$REPO" BRANCH="$BRANCH" REMOTE="$REMOTE" JOBS="$JOBS" THREADS="$THREADS" LOCK="${LOCK:-}"
        DOCKER_ARGS="${DOCKER_ARGS:-}" MOUNTS="${MOUNTS:-}")
echo "=== ladder6 checks started $(date -Is), $BRANCH at ${TIP:0:9} on ${BASE:0:9} + ${UPSTREAM:0:9}, steps: $STEPS"

s=$(steps_of union build cpubuild)
if [ -n "$s" ]; then
    echo "=== $s $(date -Is)"
    env "${common[@]}" BASE="$BASE" BASE_BUILD="$BASE_BUILD" BASE_CPU="$BASE_CPU" BRANCHES="upstream=$UPSTREAM" \
        MERGE_OWN="${OWN:- }" CPU_TARGETS="$CPU_TARGETS" STEPS="$s" \
        bash "$DATA/2026-11-04_ladder4_merge/run/checks.sh" > "$WORK/logs/merge-build.out" 2>&1
fi

if step release; then
    echo "=== release $(date -Is)"
    K=$WORK/checks/release
    rm -rf "$K" && mkdir -p "$K"
    T=$(tarball_tree)
    # a container without a GPU whose CUDA backend loads against the stub driver (the rig builds' binaries)
    STUB='mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub'
    # <name> <base bin dir> <branch bin dir>: the base writes the logits, the branch compares
    ident() {
        local n=$1 bb=$2 mb=$3
        mkdir -p "$K/$n"
        # shellcheck disable=SC2086
        docker run --rm $D -v "$WORK:$WORK" -v "$bb:$bb:ro" -v "$K/$n:/k" "$IMAGE" bash -c "$STUB"'
            run() { env LLAMA_ARCHS_LOGITS=/k/logits- nice -n 19 $2/test-llama-archs -b CPU -s 1 2>&1 \
                        | sed "s/\x1b\[[0-9;]*m//g" > /k/$1.log; echo "$1 rc=${PIPESTATUS[0]}"; }
            run base '"$bb"'
            ls /k/logits-*.f32 | wc -l > /k/n_base.txt
            run merged '"$mb"'
            rm -f /k/*.f32
            chown -R '"$(id -u):$(id -g)"' /k' > "$K/$n/rc.txt" 2>&1
        grep 'logits against' "$K/$n/base.log" | grep -v ', 0 of ' > "$K/$n/base_self.txt"
        local nb nc nw bad sb sm
        nb=$(cat "$K/$n/n_base.txt" 2>/dev/null || echo 0)
        nc=$(grep -c 'logits against' "$K/$n/merged.log")
        nw=$(grep -c 'logits_dump_or_compare: wrote' "$K/$n/merged.log")
        grep 'logits against' "$K/$n/merged.log" | grep -v ', 0 of ' | grep -vxFf "$K/$n/base_self.txt" > "$K/$n/differ.txt"
        bad=$(cut -d: -f1 "$K/$n/differ.txt" | tr '\n' ' ')
        sb=$(grep -m 1 '^Summary' "$K/$n/base.log") sm=$(grep -m 1 '^Summary' "$K/$n/merged.log")
        # a compared file that differs fails its test case, so the branch's summary counts the differing models
        verdict "release-$n" "$([ "$nb" -gt 0 ] && [ "$nc" -gt 0 ] && { [ -z "$bad" ] && echo OK || echo REVIEW; } || echo FAIL)" \
            "base wrote $nb logits files; $nc compared, $(wc -l < "$K/$n/differ.txt") differ: ${bad:-none}; written by the branch only: $nw; base: $sb; branch: $sm; $(tr '\n' ' ' < "$K/$n/rc.txt")"
    }
    if [ -n "$T" ]; then ident rig "$BASE_BUILD/bin" "$T/bin"; else verdict release-rig FAIL "no tarball tree"; fi
    ident cpu "$BASE_CPU" "$WORK/cpu/bin"
    [ -n "${PREV_CPU:-}" ] && ident prev-cpu "$PREV_CPU" "$WORK/cpu/bin"
fi

if step selfbase; then
    rm -rf "$WORK/base-cpu" "$WORK/checks/flips/base" && mkdir -p "$WORK/base-cpu" && cp -a "$WORK/cpu/bin" "$WORK/base-cpu/bin"
    verdict selfbase OK "base-cpu/bin = cpu/bin ($(git -C "$WORK/src" rev-parse --short=9 HEAD))"
fi

s=$(steps_of flips context server)
if [ -n "$s" ]; then
    echo "=== $s $(date -Is)"
    env "${common[@]}" BASE="$BASE" BASE_BUILD="$(tarball_tree)" CPU_TARGETS="$CPU_TARGETS" STEPS="$s" \
        bash "$DATA/2026-11-05_switches/run/checks.sh" > "$WORK/logs/flags.out" 2>&1
fi

l5=(BASE="$BASE" HEADS="upstream=$UPSTREAM" MERGE_OWN="${OWN:- }" TEXT="${TEXT:-}" TINY="${TINY:-}" CASE="${CASE:-}"
    MODEL="${MODEL:-}" TASKS="${TASKS:-}" SERVER_LOCK="${SERVER_LOCK:-}" PY_IMAGE="${PY_IMAGE:-python:3.12-slim}"
    CPU_TARGETS="$CPU_TARGETS")
s=$(steps_of portflip ladder4 keepalive)
if [ -n "$s" ]; then
    echo "=== $s $(date -Is)"
    env "${common[@]}" "${l5[@]}" BASE_BUILD="$(tarball_tree)" STEPS="$s" \
        bash "$DATA/2026-11-05_ladder5/run/checks.sh" > "$WORK/logs/ladder5-self.out" 2>&1
fi
s=$(steps_of c80 c524)
if [ -n "$s" ]; then
    echo "=== $s $(date -Is)"
    env "${common[@]}" "${l5[@]}" BASE_BUILD="$BASE_BUILD" STEPS="$s" \
        bash "$DATA/2026-11-05_ladder5/run/checks.sh" > "$WORK/logs/ladder5-release.out" 2>&1
fi

if step c565; then
    for sh in OFF ON; do
        w=$WORK/c565-$([ $sh = OFF ] && echo static || echo shared)
        echo "=== c565 SHARED=$sh $(date -Is)"
        env SRC="$WORK/src" WORK="$w" TREES="l6=$TIP" STEPS='tree build ctest net archs hand' SHARED=$sh JOBS="$JOBS" \
            THREADS="$THREADS" LOCK="${LOCK:-}" DOCKER_ARGS="${DOCKER_ARGS:-}" \
            bash "$DATA/2026-11-05_c565_ladder4fix/run/cpu_tests.sh" > "$WORK/logs/c565-$sh.out" 2>&1
        while IFS=$'\t' read -r t c v d; do [ "$t" = tree ] || verdict "c565 $([ $sh = OFF ] && echo static || echo shared) $c" "$v" "$d"; done < "$w/summary.tsv"
    done
fi

rm -f "$WORK/LADDER5_DONE"
date -Is > "$WORK/LADDER6_DONE"
echo "=== done $(date -Is)"
cat "$WORK/summary.tsv"
