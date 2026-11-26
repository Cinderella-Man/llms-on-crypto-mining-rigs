#!/usr/bin/env bash
# C1042 (Q1775): kept records of the card suite's other two slow shares, test-backend-ops FLASH_ATTN_EXT and the fused
# cases, so that a landing or a branch check replays its whole card suite instead of computing every case's processor
# reference again. Correctness only, nothing timed but the wall time of each step. Runs on a build machine with newer
# NVIDIA cards that run the rig build's compute 6.1 PTX. docs/data/2026-11-24_c1023_suite_record/run/run.sh (C1023, the
# MUL_MAT and MUL_MAT_ID record) with four changes: the shares here, each share's own settings (arms), the shares side
# by side on their own cards, and a dry listing step.
#
# Why: test-backend-ops checks each case on the card against the processor, and the processor's half is the slow one.
# Every landing runs FLASH_ATTN_EXT live three times (unset, with the Pascal tile kernel, with every Pascal attention
# switch), and the fuller checks run the fused cases live; each run computes every reference again.
# test-backend-ops --record writes each case's inputs and the processor's answer to a folder; --replay restores them,
# the card alone computes, and the same comparison gives the same verdicts. The record holds no card result, so one
# record serves any build and any setting, as long as the case list and the processor's reference code stay the same.
#
# Shares (SHARES, default "flash_attn_ext fused"; mul_mat and mul_mat_id also work, as in C1023):
#   flash_attn_ext  -o FLASH_ATTN_EXT
#   fused           -o <the operators in FUSED_OPS, one per line>: the fused-case operators, every name test mode runs
#                   that is not a ggml operation (MUL_MAT_VEC_FUSION, TOPK_MOE, MOE_LAYER ...). run/fused-ops.txt holds
#                   the 55 of the recording build, the list step's fused-now.txt: C947's 43
#                   (docs/data/2026-11-21_c947_v5_merged_gpu/run/ops-testmode-only.txt) and 12 added since
# Arms (the settings a step runs with; a share's record arm is the first of its arms):
#   all     every Pascal attention switch, as the landings' third FLASH_ATTN_EXT run: the tile kernel on any card
#           (GGML_CUDA_FATTN_TILE_PASCAL=2), its raw Q8_0 loads, int8 K scores and lowest token count, the one-token
#           decode kernel up to 4 rows
#   tile    GGML_CUDA_FATTN_TILE_PASCAL=2 alone, the landings' second run
#   plain   no switch, the landings' first run
#   forced  PASCAL_FORCED (scripts/pascal_forced.sh): every compute-6.1-only path of the fork at a P106's shapes
#   mmv     PASCAL_FORCED_MMV: the fork's matrix-vector kernels alone at a P106's shapes (C968's and C1023's set)
#   union   forced and all together (the list step)
# Support of a FLASH_ATTN_EXT case does not depend on any Pascal switch (ggml_cuda_flash_attn_ext_supported asks
# only for llama.cpp's best kernel), so a record made under one arm holds every case the others run. The fused share is
# recorded under forced, the widest set, so that no case a forced kernel declares supported is left out.
# Steps (STEPS, default "record replay manifest"; list runs only when named):
#   list      on the first card, every operator: test-backend-ops test -b CUDA0 -p <filter> --record-dry under union:
#             each case the card would check, with its record size, nothing computed or written; then --list-ops (the
#             ggml operation names). Writes list-ops.tsv (per operator name: cases the card checks, their record bytes,
#             cases it does not support, and whether the name is a ggml operation) and fused-now.txt (the names that
#             are not, the fused-case operators of this build); the summary line names those FUSED_OPS lacks
#   record    per share on its card: test-backend-ops test -b CUDA0 -o <ops> -p <filter> --record REC/<share> under the
#             share's record arm: the live check of the recording build (its verdicts in WORK/bops/<share>-record.log)
#             and the record, uncapped. Refuses to start when REC/<share>/cases already holds files, unless RESUME=1
#             (test-backend-ops then keeps the cases there and records the rest)
#   replay    per share on its card, after its record: every build in BUILDS (tag=bin-folder ...) under every arm of
#             the share (tag replay-<build>-<arm>), --replay REC/<share> --replay-log, and --replay-allow-missing when
#             ALLOW_MISSING=1 (default 0: a case the record lacks fails the run). Then, when CPU_CHECK is not 0, the first
#             build under the record arm again with --cpu-check CPU_CHECK (tag replay-<build>-cpucheck): about one case
#             in CPU_CHECK with inputs under 32 MB also runs the live check on this host's processor, and its reference
#             is compared with the stored one bit for bit, the sign that the record still matches the processor code
#   manifest  once every share's chain is done: md5 of every file of the record (paths relative to REC), its case
#             count (.meta files) and size, and one read of the largest case file after dropping its cached pages (the
#             disk's own read speed, for the estimate of a replay from a cold disk)
# The shares run side by side, share i on card i of GPUS (wrapping), so two processor-heavy record steps at most run at
# once with two cards.
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env:
#   WORK      a folder for the outputs (mounted read-write into every container)
#   NEW       bin/ folder of the rig build that records (and lists)
#   REC       the record's folder (REC/flash_attn_ext, REC/fused; mounted read-write)
#   BUILDS    the replay step's builds, "tag=bin-folder ..." (default "rec=$NEW")
#   MOUNTS    folders the containers mount read-only at the same paths (the builds)
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT    a CUDA check run before the first step as "$PREFLIGHT -g <cards>" (optional)
#   GPUS (0), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04), STEPS, SHARES, NAME (c1042: the containers' name prefix),
#   ALLOW_MISSING (0), CPU_CHECK (25), RESUME (0), FUSED_OPS (run/fused-ops.txt), PASCAL_FORCED_SH (run/pascal_forced.sh,
#   else this repository's scripts/pascal_forced.sh), ARMS_FA ("all tile plain"), ARMS_FUSED ("forced mmv"),
#   TMO (10800: each test-backend-ops run's timeout in seconds)
# Output: WORK/bops/<share>-<tag>.log, .cmd, .sec (wall seconds), <share>-replay-<tag>.tsv, rc.txt, summary.txt,
# manifest-<share>.md5, record.txt, list-ops.tsv, fused-now.txt; WORK/DONE at the end.
set -uo pipefail
: "${WORK:?}" "${NEW:?}" "${REC:?}"
export LC_ALL=C # sort and comm in one collation
D=$(cd "$(dirname "$0")" && pwd)
PF=${PASCAL_FORCED_SH:-}
if [ -z "$PF" ]; then
    for c in "$D/pascal_forced.sh" "$D/../../../../scripts/pascal_forced.sh"; do
        [ -f "$c" ] && { PF=$c; break; }
    done
fi
# shellcheck source=../../../../scripts/pascal_forced.sh
. "$PF" || { echo "no pascal_forced.sh (PASCAL_FORCED_SH)"; exit 1; }
[ -n "${PASCAL_FORCED:-}" ] && [ -n "${PASCAL_FORCED_MMV:-}" ] || { echo "$PF sets no PASCAL_FORCED"; exit 1; }
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPUS=${GPUS:-0}
STEPS=${STEPS:-record replay manifest}
SHARES=${SHARES:-flash_attn_ext fused}
NAME=${NAME:-c1042}
BUILDS=${BUILDS:-rec=$NEW}
ALLOW_MISSING=${ALLOW_MISSING:-0}
CPU_CHECK=${CPU_CHECK:-25}
RESUME=${RESUME:-0}
PREFLIGHT=${PREFLIGHT:-}
FUSED_OPS=${FUSED_OPS:-$D/fused-ops.txt}
ARMS_FA=${ARMS_FA:-all tile plain}
ARMS_FUSED=${ARMS_FUSED:-forced mmv}
TMO=${TMO:-10800}
F='^(?!.*dup=1)'
FA_ALL="GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_FATTN_TILE_PASCAL_LDS=1 GGML_CUDA_FATTN_TILE_PASCAL_KDP=1 \
GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4"
IFS=, read -ra G <<< "$GPUS"

say() { echo "$(date +%H:%M:%S) $*"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
strip() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }
arm_env() {  # <arm>: its environment
    case $1 in
        plain) echo "" ;;
        tile) echo "GGML_CUDA_FATTN_TILE_PASCAL=2" ;;
        all) echo "$FA_ALL" ;;
        mmv) echo "$PASCAL_FORCED_MMV" ;;
        forced) echo "$PASCAL_FORCED" ;;
        union) echo "$PASCAL_FORCED $FA_ALL" ;;
        *) echo "BAD_ARM=$1" ;;
    esac
}
op() {  # <share>: its -o list (empty: every operator)
    case $1 in
        flash_attn_ext) echo FLASH_ATTN_EXT ;;
        fused) grep -v '^ *$' "$FUSED_OPS" | paste -sd, ;;
        mul_mat) echo MUL_MAT ;;
        mul_mat_id) echo MUL_MAT_ID ;;
        all) echo "" ;;
        *) echo "BAD_SHARE_$1" ;;
    esac
}
arms() {  # <share>: its arms, the record arm first
    case $1 in
        flash_attn_ext) echo "$ARMS_FA" ;;
        fused) echo "$ARMS_FUSED" ;;
        *) echo mmv ;;
    esac
}
rec_arm() { local a; a=$(arms "$1"); echo "${a%% *}"; }

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
{ for b in $BUILDS; do md5sum "${b#*=}/test-backend-ops"; done; md5sum "$D"/* "$PF"; } > "$WORK/md5.txt" 2>&1
{
    printf 'BUILDS: %s\nREC: %s\nfilter: %s\nFUSED_OPS: %s (%s operators)\n' "$BUILDS" "$REC" "$F" "$FUSED_OPS" \
        "$(grep -vc '^ *$' "$FUSED_OPS")"
    for a in all tile plain forced mmv union; do printf 'arm %s: %s\n' "$a" "$(arm_env "$a")"; done
    for s in $SHARES; do printf 'share %s: arms %s (record arm %s)\n' "$s" "$(arms "$s")" "$(rec_arm "$s")"; done
} > "$WORK/settings.txt"
say "start: steps '$STEPS', shares '$SHARES' on cards $GPUS; record $REC; builds $BUILDS"

tbo() {  # <card> <bin folder> <share> <run tag> <arm> <test-backend-ops arguments after the operators and the filter>
    local card=$1 bin=$2 share=$3 tag=$4 arm=$5 env ops args t0 t1
    shift 5
    env=$(arm_env "$arm")
    ops=$(op "$share")
    args="-p '$F' $*"
    [ -n "$ops" ] && args="-o '$ops' $args"
    printf 'arm: %s\nenv: %s\nargs: test -b CUDA0 %s\nbin: %s\ncard: %s\n' "$arm" "$env" "$args" "$bin" "$card" > "$o/$share-$tag.cmd"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    "${DR[@]}" --name "$NAME-$share-$tag" --gpus "\"device=$card\"" "$IMG" \
        bash -c "env $env X=1 nice -n 19 timeout $TMO $bin/test-backend-ops test -b CUDA0 $args > $o/$share-$tag.log 2>&1; echo $share-$tag rc=\$? >> $o/rc.txt" \
        > /dev/null 2>&1
    t1=$(date +%s)
    echo $((t1 - t0)) > "$o/$share-$tag.sec"
    say "  $share $tag (card $card, $arm): $(grep "^$share-$tag " "$o/rc.txt" | tail -n 1); $(strip "$o/$share-$tag.log" | grep -a -E 'tests passed|  record: ' | tail -n 1 | sed 's/^ *//'); FAIL $(strip "$o/$share-$tag.log" | grep -a -c ': FAIL'); $((t1 - t0)) s"
}

if [ -n "$PREFLIGHT" ]; then
    $PREFLIGHT -g "$GPUS" > "$o/preflight.txt" 2>&1
    rc=$?
    say "preflight rc=$rc"
    [ $rc = 0 ] || { say "preflight failed: nothing run"; date -Is > "$WORK/DONE"; exit 1; }
fi

if has list; then
    tbo "${G[0]}" "$NEW" all list union "--record-dry"
    # shellcheck disable=SC2086
    "${DR[@]}" --name "$NAME-list-ops" --gpus "\"device=${G[0]}\"" "$IMG" bash -c "$NEW/test-backend-ops --list-ops" \
        > "$o/ggml-ops.txt" 2>&1
    sed -n 's/^  \([A-Z0-9_]*\)$/\1/p' "$o/ggml-ops.txt" | sort -u > "$o/ggml-ops.names"
    # per operator name: cases the card checks and their record bytes, cases it does not support, and the kind
    strip "$o/all-list.log" | awk -v g="$o/ggml-ops.names" 'BEGIN { while ((getline n < g) > 0) op[n] = 1 }
        match($0, /^  [A-Za-z0-9_]+\(/) {
            n = substr($0, 3, RLENGTH - 3)
            if (match($0, /: in [0-9]+ B, ref [0-9]+ B$/)) {
                s = substr($0, RSTART); gsub(/[^0-9 ]/, "", s); split(s, a, " ")
                c[n]++; b[n] += a[1] + a[2]; seen[n] = 1
            } else if ($0 ~ /: not supported/) {
                ns[n]++; seen[n] = 1
            }
        }
        END { for (k in seen) printf "%s\t%d\t%.0f\t%d\t%s\n", k, c[k], b[k], ns[k], (k in op) ? "ggml op" : "test mode only" }' \
        | sort > "$o/list-ops.tsv"
    # the fused-case operators: every name that is not a ggml operation, with cases the card checks or not
    awk -F'\t' '$5 == "test mode only" { print $1 }' "$o/list-ops.tsv" | sort > "$o/fused-now.txt"
    say "list: $(strip "$o/all-list.log" | grep -a '  record: ' | sed 's/^ *//'); operator names $(wc -l < "$o/list-ops.tsv"), test mode only $(wc -l < "$o/fused-now.txt"); against FUSED_OPS: new '$(comm -13 <(sort "$FUSED_OPS") "$o/fused-now.txt" | paste -sd' ')', gone '$(comm -23 <(sort "$FUSED_OPS") "$o/fused-now.txt" | paste -sd' ')'"
fi

chain() {  # <card> <share>: the share's record and replays, in order, on its card
    local card=$1 share=$2 ra am="" first="" b tag a
    ra=$(rec_arm "$share")
    if has record; then
        if [ "$RESUME" != 1 ] && [ -n "$(ls -A "$REC/$share/cases" 2>/dev/null | head -n 1)" ]; then
            say "  $share: $REC/$share/cases already holds files (RESUME=1 keeps them): not recorded"
        else
            tbo "$card" "$NEW" "$share" record "$ra" "--record $REC/$share"
        fi
    fi
    if has replay; then
        [ "$ALLOW_MISSING" = 1 ] && am="--replay-allow-missing"
        for b in $BUILDS; do
            tag=${b%%=*}
            [ -n "$first" ] || first=$b
            for a in $(arms "$share"); do
                tbo "$card" "${b#*=}" "$share" "replay-$tag-$a" "$a" \
                    "--replay $REC/$share --replay-log $o/$share-replay-$tag-$a.tsv $am"
            done
        done
        if [ "$CPU_CHECK" != 0 ] && [ -n "$first" ]; then
            tag=${first%%=*}-cpucheck
            tbo "$card" "${first#*=}" "$share" "replay-$tag" "$ra" \
                "--replay $REC/$share --replay-log $o/$share-replay-$tag.tsv $am --cpu-check $CPU_CHECK"
        fi
    fi
    say "$share done"
}

if has record || has replay; then
    i=0
    for share in $SHARES; do
        chain "${G[$((i % ${#G[@]}))]}" "$share" &
        i=$((i + 1))
    done
    wait
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
    big=$(for share in $SHARES; do find "$REC/$share" -type f -printf '%s %p\n'; done | sort -n | tail -n 1 | cut -d' ' -f2-)
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
say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | paste -sd' ')"
date -Is > "$WORK/DONE"
