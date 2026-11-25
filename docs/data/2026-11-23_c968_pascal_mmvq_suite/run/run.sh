#!/usr/bin/env bash
# C968 (Q1698): the fork's Pascal matrix-vector kernels in the card suite, on a build machine with three newer NVIDIA
# cards that run the rig build's compute 6.1 PTX. Correctness only, nothing timed.
#
# The fork's kernels for small batches (mul_mat_vec_q_pascal and its multi-token mc kernel, the k-column kernels, the
# multi-token expert kernels, the BF16 kernels, and the opt-in read-once kernels) turn themselves on only on compute 6.1
# cards: GGML_CUDA_MMVQ_PASCAL and GGML_CUDA_MMVF_PASCAL at their default 1 need compute 6.1, and =2 turns them on for any
# NVIDIA card. So a suite run on newer cards without =2 checks llama.cpp's own kernels in their place. This driver runs
# the card suite's three shares that reach these kernels (as docs/data/2026-11-21_c947_v5_merged_gpu split it) with them
# forced on and a P106's launch shapes (5 SMs, 96 KB of shared memory per SM), from scripts/pascal_forced.sh (sourced
# below; $ON): FORCED=all (the default) is PASCAL_FORCED, every compute-6.1-only path of the fork (also the attention
# block, the persistent MoE kernel and the Pascal flash-attention tile kernel, as a P106 runs them); FORCED=mmv is
# PASCAL_FORCED_MMV, the matrix-vector kernels alone, the set C968's run used.
# Settings (arms):
#   unset   no switch: llama.cpp's own matrix-vector kernels on these cards (what C947's suite ran)
#   p106    $ON: the fork's kernels as a P106 runs them with every switch at its default (the Flint, Ornith and Qwen3.6
#           v5 presets set no kernel switch)
#   fn      $ON plus every GGML_CUDA_MMV* switch of release/flash-next-v5/settings.env (the read-once kernels for Q3_K,
#           Q4_K, IQ4_XS, Q5_K and Q6_K, the fused gate/up launch, the k-column kernel's exact mode, ...): the kernels a
#           P106 runs under the Flash-Next preset
#   p106-kcol0  p106 with GGML_CUDA_MMVQ_PASCAL_KCOL=0 (no k-column kernel), so that a case whose result this changes
#           was computed by a k-column kernel under p106
#   fn-f32  fn with GGML_CUDA_MMVF_F32_MT=2 after the preset's 1 (pascal_raise_gates): that switch has its own compute
#           6.1 gate (=1 acts only on compute 6.1, =2 on any NVIDIA card), so under fn its F32 products of 4 to 8 tokens
#           stay on llama.cpp's path on newer cards; replayed in a second invocation (STEPS=replay REPLAYS=fn-f32)
# Shares (one per card): mul_mat (MUL_MAT), mul_mat_id (MUL_MAT_ID), fused (the 43 fused-case operators test mode runs
# that the support listing does not name, run/ops-testmode-only.txt), each every case but the repeated-expert-id ones
# (-p "^(?!.*dup=1)").
# Steps (STEPS, default "live record replay"), each share's steps in order on its own card, the three cards at once:
#   live    test-backend-ops test -b CUDA0, live, with p106 and with fn: each case on the card against the CPU
#   record  the same cases with --record: inputs and CPU reference written to REC (cases over RMAX MB left out, and not
#           run; RMAX=0: none left out), and the live check run with RECORD_SETTING (unset by default; p106 makes this
#           step the p106 live check as well)
#   replay  the record replayed with unset, unset again (whether a case's error repeats), p106, fn and p106-kcol0
#           (--replay-log: per case the verdict and the normalized squared error against the CPU), so every setting
#           computes the same inputs and a case whose error differs from unset's was computed by another kernel
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env:
#   WORK      a folder for the outputs (mounted read-write into every container)
#   NEW       bin/ folder of the rig build
#   REPO      a copy of this repository (release/flash-next-v5/settings.env)
#   REC       a folder for the records (~53 GB at RMAX 64; mounted read-write)
#   MOUNTS    folders the containers mount read-only at the same paths (the build)
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT    a CUDA check run before the first step as "$PREFLIGHT -g <ids>" (optional)
#   GPUS (0,1,2: the cards of mul_mat, mul_mat_id, fused), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04), STEPS,
#   SHARES ("mul_mat mul_mat_id fused"), RMAX (64), NAME (c968: the containers' name prefix),
#   REPLAYS (the replay step's settings, "unset unset2 p106 fn p106-kcol0"), RECORD_SETTING (unset), FORCED (all)
# Output: WORK/bops/<share>-<step>.log, .cmd, <share>-replay-<setting>.tsv, rc.txt, summary.txt; WORK/DONE at the end.
set -uo pipefail
: "${WORK:?}" "${NEW:?}" "${REPO:?}" "${REC:?}"
# shellcheck source=../../../../scripts/pascal_forced.sh
. "$REPO/scripts/pascal_forced.sh"
case ${FORCED:-all} in
    all) ON=$PASCAL_FORCED ;;
    mmv) ON=$PASCAL_FORCED_MMV ;;
    *) echo "FORCED must be all or mmv"; exit 1 ;;
esac
[ -n "$ON" ] || { echo "no forced set in $REPO/scripts/pascal_forced.sh"; exit 1; }
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPUS=${GPUS:-0,1,2}
STEPS=${STEPS:-live record replay}
SHARES=${SHARES:-mul_mat mul_mat_id fused}
RMAX=${RMAX:-64}
NAME=${NAME:-c968}
REPLAYS=${REPLAYS:-unset unset2 p106 fn p106-kcol0}
RECORD_SETTING=${RECORD_SETTING:-unset}
PREFLIGHT=${PREFLIGHT:-}
D=$(cd "$(dirname "$0")" && pwd)
F='^(?!.*dup=1)'
IFS=, read -ra G <<< "$GPUS"
[ ${#G[@]} = 3 ] || { echo "GPUS must name three cards"; exit 1; }

FN_SW=$(unset PRESET_ENV; . "$REPO/release/flash-next-v5/settings.env" > /dev/null 2>&1
        for kv in $PRESET_ENV; do case $kv in GGML_CUDA_MMV*) printf '%s ' "$kv" ;; esac; done)
[ -n "$FN_SW" ] || { echo "no GGML_CUDA_MMV* switch in $REPO/release/flash-next-v5/settings.env"; exit 1; }

sw() {  # <setting>: its environment
    case $1 in
        unset | unset2) echo "" ;;
        p106) echo "$ON" ;;
        fn) echo "$ON $FN_SW" ;;
        p106-kcol0) echo "$ON GGML_CUDA_MMVQ_PASCAL_KCOL=0" ;;
        fn-f32) echo "$ON $FN_SW $(pascal_raise_gates "$FN_SW")" ;;
        *) echo "BAD_SETTING=$1" ;;
    esac
}
ops() {  # <share>: its operators
    case $1 in
        mul_mat) echo MUL_MAT ;;
        mul_mat_id) echo MUL_MAT_ID ;;
        fused) paste -sd, "$D/ops-testmode-only.txt" ;;
    esac
}

say() { echo "$(date +%H:%M:%S) $*"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
strip() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$WORK:$WORK" -v "$REC:$REC")
# the PTX JIT's cache, shared by every container (the cards compile the build's compute 6.1 PTX once)
JIT=(-e "CUDA_CACHE_PATH=$WORK/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)
# shellcheck disable=SC2206
DR=(docker run --rm ${DOCKER_ARGS:-} "${VOL[@]}" "${JIT[@]}")
o=$WORK/bops
mkdir -p "$WORK/jitcache" "$o" "$REC"
rm -f "$WORK/DONE"
md5sum "$NEW/test-backend-ops" "$D"/* > "$WORK/md5.txt" 2>&1
printf 'ON: %s\nFN_SW: %s\n' "$ON" "$FN_SW" > "$WORK/settings.txt"
say "start: steps '$STEPS', shares '$SHARES' on cards $GPUS, NEW=$NEW"
say "fn adds: $FN_SW"

tbo() {  # <card> <share> <run tag> <setting> <test-backend-ops arguments after the operators and the filter>
    local card=$1 share=$2 tag=$3 env
    env=$(sw "$4")
    shift 4
    local args="-o '$(ops "$share")' -p '$F' $*"
    printf 'env: %s\nargs: test -b CUDA0 %s\n' "$env" "$args" > "$o/$share-$tag.cmd"
    # shellcheck disable=SC2086
    "${DR[@]}" --name "$NAME-$share-$tag" --gpus "\"device=$card\"" "$IMG" \
        bash -c "env $env X=1 nice -n 19 timeout 10800 $NEW/test-backend-ops test -b CUDA0 $args > $o/$share-$tag.log 2>&1; echo $share-$tag rc=\$? >> $o/rc.txt" \
        > /dev/null 2>&1
    say "  $share $tag (card $card): $(grep "^$share-$tag " "$o/rc.txt" | tail -n 1); $(strip "$o/$share-$tag.log" | grep -a -E 'tests passed' | tail -n 1 | sed 's/^ *//'); FAIL $(strip "$o/$share-$tag.log" | grep -a -c ': FAIL')"
}

chain() {  # <card> <share>
    local card=$1 share=$2 s
    if has live; then
        for s in p106 fn; do tbo "$card" "$share" "live-$s" "$s" ""; done
    fi
    if has record; then
        tbo "$card" "$share" record "$RECORD_SETTING" "--record $REC/$share --record-max-case-mb $RMAX"
    fi
    if has replay; then
        for s in $REPLAYS; do
            tbo "$card" "$share" "replay-$s" "$s" "--replay $REC/$share --replay-log $o/$share-replay-$s.tsv --replay-allow-missing"
        done
    fi
    say "$share done"
}

if [ -n "$PREFLIGHT" ]; then
    $PREFLIGHT -g "$GPUS" > "$o/preflight.txt" 2>&1
    rc=$?
    say "preflight rc=$rc"
    [ $rc = 0 ] || { say "preflight failed: nothing run"; date -Is > "$WORK/DONE"; exit 1; }
fi
i=0
for share in $SHARES; do
    chain "${G[$i]}" "$share" &
    i=$((i + 1))
done
wait

: > "$o/summary.txt"
for f in "$o"/*.log; do
    t=$(basename "$f" .log)
    echo "$t: $(grep "^$t " "$o/rc.txt" | tail -n 1); $(strip "$f" | grep -a -m1 'tests passed' | sed 's/^ *//'); OK $(strip "$f" | grep -a -c ': OK$'); FAIL $(strip "$f" | grep -a -c ': FAIL'); NOREC $(strip "$f" | grep -a -c ': NOREC'); not supported $(strip "$f" | grep -a -c 'not supported'); illegal $(strip "$f" | grep -a -c 'illegal memory access')" >> "$o/summary.txt"
    strip "$f" | grep -a ': FAIL' | grep -av 'Backend' | sed 's/^ */    /' | cut -c1-260 >> "$o/summary.txt"
done
du -sh "$REC"/* > "$o/rec-size.txt" 2>&1
say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | paste -sd' ')"
date -Is > "$WORK/DONE"
