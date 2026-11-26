#!/usr/bin/env bash
# C1023 (Q1756): a kept record of the card suite's two matrix-product shares, test-backend-ops MUL_MAT and MUL_MAT_ID,
# so that a landing or a branch check replays them in minutes instead of computing every case's processor reference
# again. Correctness only, nothing timed but the wall time of each step. Runs on a build machine with newer NVIDIA cards
# that run the rig build's compute 6.1 PTX.
#
# Why: test-backend-ops checks each case on the card against the processor. The processor's side is the slow part: a
# live MUL_MAT_ID check of the fork's Pascal kernels took 38 minutes on one card while the references used 12 cores
# (docs/data/2026-11-24_land_v5_batch2). test-backend-ops --record writes each case's inputs and the processor's answer
# to a folder; --replay restores them, the card alone computes, and the same comparison gives the same verdicts. The
# record holds no card result, so one record serves any build and any kernel setting, as long as the case list and the
# processor's reference code stay the same.
#
# The fork's kernels for small batches turn themselves on only on compute 6.1 cards unless forced: the record and every
# replay force them with a P106's launch shapes (5 SMs, 96 KB of shared memory per SM), as
# docs/data/2026-11-23_c968_pascal_mmvq_suite/run/run.sh did (C968):
# and, since 2026-11-26 (C1108), cuBLAS as on a P106: in F32, without TF32 (scripts/pascal_forced.sh's PASCAL_CUBLAS).
# C1023's own record and replays ran without it; C1108 replayed MUL_MAT both ways with the same verdicts.
ON=${ON:-"NVIDIA_TF32_OVERRIDE=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 \
GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304"}
# Recording with the kernels forced matters: 278 large-activation cases need the backend feature MMVQ_PASCAL, so
# without it they are "not supported" and left out of the record.
#
# Cases: every case of each share but the repeated-expert-id ones (-p "^(?!.*dup=1)"), the card suite's filter since C947.
# Steps (STEPS, default "record replay manifest"), each share in turn on one card:
#   record    test-backend-ops test -b CUDA0 -o <op> -p <filter> --record REC/<share>, with ON: the live check of the
#             recording build (its verdicts in WORK/bops/<share>-record.log) and the record, uncapped. Refuses to start
#             when REC/<share>/cases already holds files, unless RESUME=1 (test-backend-ops then keeps the cases there
#             and records the rest).
#   replay    for each build in BUILDS (tag=bin-folder ...), each share: --replay REC/<share> --replay-log
#             WORK/bops/<share>-replay-<tag>.tsv, with ON, and --replay-allow-missing when ALLOW_MISSING=1 (the default
#             here is 0: a case the record lacks fails the run). Then, when CPU_CHECK is not 0, the first build again
#             with --cpu-check CPU_CHECK (tag <first>-cpucheck): about one case in CPU_CHECK with inputs under 32 MB also
#             runs the live check on this host's processor, and its reference is compared with the stored one bit for
#             bit, the sign that the record still matches the processor code.
#   manifest  md5 of every file of the record (WORK/bops/manifest-<share>.md5, paths relative to REC), its case count
#             (.meta files) and size, and one read of the largest case file after dropping its cached pages (the disk's own read speed, for
#             the estimate of a replay from a cold disk).
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env:
#   WORK      a folder for the outputs (mounted read-write into every container)
#   NEW       bin/ folder of the rig build that records
#   REC       the record's folder (REC/mul_mat, REC/mul_mat_id; ~45 GB; mounted read-write)
#   BUILDS    the replay step's builds, "tag=bin-folder ..." (default "rec=$NEW")
#   MOUNTS    folders the containers mount read-only at the same paths (the builds)
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT    a CUDA check run before the first step as "$PREFLIGHT -g <card>" (optional)
#   GPU (0), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04), STEPS, SHARES ("mul_mat_id mul_mat"), NAME (c1023: the
#   containers' name prefix), ALLOW_MISSING (0), CPU_CHECK (25), RESUME (0), ON (above)
# Output: WORK/bops/<share>-<tag>.log, .cmd, .sec (wall seconds), <share>-replay-<tag>.tsv, rc.txt, summary.txt,
# manifest-<share>.md5, record.txt; WORK/DONE at the end.
set -uo pipefail
: "${WORK:?}" "${NEW:?}" "${REC:?}"
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPU=${GPU:-0}
STEPS=${STEPS:-record replay manifest}
SHARES=${SHARES:-mul_mat_id mul_mat}
NAME=${NAME:-c1023}
BUILDS=${BUILDS:-rec=$NEW}
ALLOW_MISSING=${ALLOW_MISSING:-0}
CPU_CHECK=${CPU_CHECK:-25}
RESUME=${RESUME:-0}
PREFLIGHT=${PREFLIGHT:-}
F='^(?!.*dup=1)'

say() { echo "$(date +%H:%M:%S) $*"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
strip() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }
op() { case $1 in mul_mat) echo MUL_MAT ;; mul_mat_id) echo MUL_MAT_ID ;; *) echo "BAD_SHARE_$1" ;; esac; }
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$WORK:$WORK" -v "$REC:$REC")
# the PTX JIT's cache, shared by every container (the cards compile the build's compute 6.1 PTX once per build)
JIT=(-e "CUDA_CACHE_PATH=$WORK/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)
# shellcheck disable=SC2206
DR=(docker run --rm ${DOCKER_ARGS:-} "${VOL[@]}" "${JIT[@]}")
o=$WORK/bops
mkdir -p "$WORK/jitcache" "$o" "$REC"
rm -f "$WORK/DONE"
{ for b in $BUILDS; do md5sum "${b#*=}/test-backend-ops"; done; md5sum "$(cd "$(dirname "$0")" && pwd)"/*; } > "$WORK/md5.txt" 2>&1
printf 'ON: %s\nBUILDS: %s\nREC: %s\nfilter: %s\n' "$ON" "$BUILDS" "$REC" "$F" > "$WORK/settings.txt"
say "start: steps '$STEPS', shares '$SHARES' on card $GPU; record $REC; builds $BUILDS"

tbo() {  # <bin folder> <share> <run tag> <test-backend-ops arguments after the operator and the filter>
    local bin=$1 share=$2 tag=$3
    shift 3
    local args="-o '$(op "$share")' -p '$F' $*" t0 t1
    printf 'env: %s\nargs: test -b CUDA0 %s\nbin: %s\n' "$ON" "$args" "$bin" > "$o/$share-$tag.cmd"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    "${DR[@]}" --name "$NAME-$share-$tag" --gpus "\"device=$GPU\"" "$IMG" \
        bash -c "env $ON nice -n 19 timeout 10800 $bin/test-backend-ops test -b CUDA0 $args > $o/$share-$tag.log 2>&1; echo $share-$tag rc=\$? >> $o/rc.txt" \
        > /dev/null 2>&1
    t1=$(date +%s)
    echo $((t1 - t0)) > "$o/$share-$tag.sec"
    say "  $share $tag: $(grep "^$share-$tag " "$o/rc.txt" | tail -n 1); $(strip "$o/$share-$tag.log" | grep -a -E 'tests passed' | tail -n 1 | sed 's/^ *//'); FAIL $(strip "$o/$share-$tag.log" | grep -a -c ': FAIL'); $((t1 - t0)) s"
}

if [ -n "$PREFLIGHT" ]; then
    $PREFLIGHT -g "$GPU" > "$o/preflight.txt" 2>&1
    rc=$?
    say "preflight rc=$rc"
    [ $rc = 0 ] || { say "preflight failed: nothing run"; date -Is > "$WORK/DONE"; exit 1; }
fi

if has record; then
    for share in $SHARES; do
        if [ "$RESUME" != 1 ] && [ -n "$(ls -A "$REC/$share/cases" 2>/dev/null | head -n 1)" ]; then
            say "  $share: $REC/$share/cases already holds files (RESUME=1 keeps them): not recorded"
            continue
        fi
        tbo "$NEW" "$share" record "--record $REC/$share"
    done
fi

if has replay; then
    am=""
    [ "$ALLOW_MISSING" = 1 ] && am="--replay-allow-missing"
    first=""
    for b in $BUILDS; do
        tag=${b%%=*}
        [ -n "$first" ] || first=$b
        for share in $SHARES; do
            tbo "${b#*=}" "$share" "replay-$tag" "--replay $REC/$share --replay-log $o/$share-replay-$tag.tsv $am"
        done
    done
    if [ "$CPU_CHECK" != 0 ] && [ -n "$first" ]; then
        tag=${first%%=*}-cpucheck
        for share in $SHARES; do
            tbo "${first#*=}" "$share" "replay-$tag" "--replay $REC/$share --replay-log $o/$share-replay-$tag.tsv $am --cpu-check $CPU_CHECK"
        done
    fi
fi

if has manifest; then
    : > "$o/record.txt"
    for share in $SHARES; do
        t0=$(date +%s)
        (cd "$REC" && find "$share" -type f -print0 | sort -z | nice -n 19 ionice -c 3 xargs -0 md5sum) > "$o/manifest-$share.md5"
        t1=$(date +%s)
        printf '%s\tcases %s\tfiles %s\tbytes %s\tmanifest md5 %s\tmanifest %s s\n' "$share" \
            "$(find "$REC/$share" -name '*.meta' | wc -l)" "$(wc -l < "$o/manifest-$share.md5")" \
            "$(du -sb "$REC/$share" | cut -f1)" "$(md5sum < "$o/manifest-$share.md5" | cut -c1-32)" $((t1 - t0)) >> "$o/record.txt"
    done
    big=$(find "$REC" -type f -printf '%s %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
    # the file's pages dropped from the page cache first (dd iflag=nocache count=0: this file only), so the disk
    # answers the read; O_DIRECT (iflag=direct) is refused on this ext4 disk with EINVAL
    printf 'largest file %s, read after dropping its cached pages: ' "${big#"$REC"/}" >> "$o/record.txt"
    dd if="$big" iflag=nocache count=0 2> /dev/null
    dd if="$big" of=/dev/null bs=16M 2>&1 | tail -n 1 >> "$o/record.txt"
    say "manifest: $(cut -f1-4 "$o/record.txt" | paste -sd';')"
fi

: > "$o/summary.txt"
for f in "$o"/*.log; do
    t=$(basename "$f" .log)
    echo "$t: $(grep "^$t " "$o/rc.txt" | tail -n 1); $(strip "$f" | grep -a -m1 'tests passed' | sed 's/^ *//'); OK $(strip "$f" | grep -a -c ': OK$'); FAIL $(strip "$f" | grep -a -c ': FAIL'); NOREC $(strip "$f" | grep -a -c ': NOREC'); not supported $(strip "$f" | grep -a -c 'not supported'); illegal $(strip "$f" | grep -a -c 'illegal memory access'); $(cat "$o/$t.sec" 2>/dev/null) s" >> "$o/summary.txt"
    strip "$f" | grep -a -E '^  (record|replay)' | sed 's/^ */    /' >> "$o/summary.txt"
    strip "$f" | grep -a ': FAIL' | grep -av 'Backend' | sed 's/^ */    /' | cut -c1-260 >> "$o/summary.txt"
done
say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")"
date -Is > "$WORK/DONE"
