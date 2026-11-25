#!/usr/bin/env bash
# C991 (Q1724): the fork's other compute-6.1-only card paths in the card suite, on a build machine with three newer
# NVIDIA cards that run the rig build's compute 6.1 PTX. Correctness only, nothing timed.
#
# Besides the matrix-vector kernels (checked forced on in docs/data/2026-11-23_c968_pascal_mmvq_suite), three paths of
# the fork turn themselves on only on compute 6.1 cards: the attention block (GGML_CUDA_ATTN_BLOCK), the persistent MoE
# kernel (GGML_CUDA_PERSISTENT_MOE) and the Pascal flash-attention tile kernel (GGML_CUDA_FATTN_TILE_PASCAL). So the
# card suite of docs/data/2026-11-21_c947_v5_merged_gpu checked llama.cpp's kernels in their place in its `rest` and
# fused shares. This driver runs those two shares again with every one of the fork's paths forced on
# (scripts/pascal_forced.sh, PASCAL_FORCED), one pass each, and finds which cases reach each path.
#
# Steps (STEPS, default "list live probe"):
#   list   on the first card: the support listing (operators the support mode names) and a dry record listing
#          (--record-dry: every case test mode would run, nothing computed), both with PASCAL_FORCED. The shares:
#            fa     FLASH_ATTN_EXT (taken out of C947's rest share, so that it gets a card of its own)
#            rest   every other operator the support listing names but MUL_MAT and MUL_MAT_ID (C947's rest share)
#            fused  every operator name of the dry listing that the support listing lacks, but MUL_MAT and MUL_MAT_ID
#                   (C947's fused share: the operators only test mode runs)
#          Every case but the repeated-expert-id ones (-p "^(?!.*dup=1)"), as C947.
#   live   the three shares at once, one card each, with PASCAL_FORCED: each case on the card against the processor.
#          With NSYS set, each runs under Nsight Systems (cuda trace, CUDA graphs traced by node) for the names and launch
#          counts of the kernels it ran; the reports and their times are deleted unread (no timing on this machine).
#   probe  which cases reach a path, after each share's live pass on the same card:
#            fa     the share again with GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1 (the Pascal prompt and decode attention
#                   kernels scale their output by 1.05): a case that fails here was computed by the Pascal kernel
#            fused  MOE_LAYER with GGML_CUDA_PERSISTENT_MOE_CHECK=1 (the kernel's own comparison with the per-op path,
#                   logged when the card's context is freed), then with GGML_CUDA_PERSISTENT_MOE_FAULT=3 (the shared
#                   expert left out of the layer's output): a case that fails there was computed by the persistent kernel
#          The attention block has no such hook; its kernels' names in the live pass's trace say whether a case ran it.
#   ornith (not in the default STEPS: a second invocation with STEPS=ornith) the fa share again on the first card with
#          the GGML_CUDA_FATTN_* switches of release/ornith-v5/settings.env added (GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 and
#          GGML_CUDA_FATTN_DEC_PASCAL=4: calls of one and two query tokens with a Q8_0 cache go to the Pascal kernels, those
#          at head 256 and up to 4 tokens to pull request 9's decode kernel), live (traced with NSYS) and with
#          GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1 (which scales the decode kernel's output too)
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env:
#   WORK      a folder for the outputs (mounted read-write into every container)
#   NEW       bin/ folder of the rig build (with test-backend-ops)
#   REPO      a copy of this repository (scripts/pascal_forced.sh; release/ornith-v5/settings.env for STEPS=ornith)
#   MOUNTS    folders the containers mount read-only at the same paths (the build)
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT    a CUDA check run before the first step as "$PREFLIGHT -g <ids>" (optional)
#   NSYS      an nsys binary on the host, e.g. /opt/nvidia/nsight-systems/<version>/bin/nsys (optional; its install is
#             mounted read-only)
#   GPUS (0,1,2: the cards of fa, rest, fused), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04), STEPS, NAME (c991: the
#   containers' name prefix), TIMEOUT (10800 s a run)
# Output: WORK/bops/<share>-<step>.log, .cmd, rc.txt, summary.txt, kern-<share>.tsv (launches, kernel name);
#   (kern-<share>-<step>.tsv for a step other than live); WORK/list/ (support.txt, dry.txt, ops-*.txt); WORK/DONE at the end.
set -uo pipefail
: "${WORK:?}" "${NEW:?}" "${REPO:?}"
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPUS=${GPUS:-0,1,2}
STEPS=${STEPS:-list live probe}
NAME=${NAME:-c991}
TIMEOUT=${TIMEOUT:-10800}
PREFLIGHT=${PREFLIGHT:-}
NSYS=${NSYS:-}
D=$(cd "$(dirname "$0")" && pwd)
F='^(?!.*dup=1)'
IFS=, read -ra G <<< "$GPUS"
[ ${#G[@]} = 3 ] || { echo "GPUS must name three cards"; exit 1; }
# shellcheck source=../../../../scripts/pascal_forced.sh
. "$REPO/scripts/pascal_forced.sh"
[ -n "${PASCAL_FORCED:-}" ] || { echo "no PASCAL_FORCED in $REPO/scripts/pascal_forced.sh"; exit 1; }

say() { echo "$(date +%H:%M:%S) $*"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
strip() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$WORK:$WORK")
if [ -n "$NSYS" ]; then
    NSYS=$(readlink -f "$NSYS")
    VOL+=(-v "$(dirname "$(dirname "$NSYS")"):$(dirname "$(dirname "$NSYS")"):ro")
fi
# the PTX JIT's cache, shared by every container (the cards compile the build's compute 6.1 PTX once)
JIT=(-e "CUDA_CACHE_PATH=$WORK/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)
# shellcheck disable=SC2206
DR=(docker run --rm ${DOCKER_ARGS:-} "${VOL[@]}" "${JIT[@]}")
o=$WORK/bops
l=$WORK/list
mkdir -p "$WORK/jitcache" "$o" "$l"
rm -f "$WORK/DONE"
md5sum "$NEW/test-backend-ops" "$D"/* "$REPO/scripts/pascal_forced.sh" > "$WORK/md5.txt" 2>&1
printf 'PASCAL_FORCED: %s\n' "$PASCAL_FORCED" > "$WORK/settings.txt"
say "start: steps '$STEPS' on cards $GPUS, NEW=$NEW, NSYS=${NSYS:-none}"
say "forced: $PASCAL_FORCED"

ops() {  # <share>: its operators, comma-separated
    case $1 in
        fa) echo FLASH_ATTN_EXT ;;
        rest) grep -vx -e MUL_MAT -e MUL_MAT_ID -e FLASH_ATTN_EXT "$l/ops-support.txt" | paste -sd, ;;
        fused) paste -sd, "$l/ops-fused.txt" ;;
        moe) echo MOE_LAYER ;;
    esac
}

tbo() {  # <card> <share> <run tag> <extra environment> [nsys]: one test-backend-ops run of the share's operators
    local card=$1 share=$2 tag=$3 env="$PASCAL_FORCED $4" prof=${5:-} pre=""
    local args="-o '$(ops "$share")' -p '$F'"
    printf 'env: %s\nargs: test -b CUDA0 %s\n' "$env" "$args" > "$o/$share-$tag.cmd"
    if [ -n "$prof" ] && [ -n "$NSYS" ]; then
        pre="$NSYS profile -t cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true -o $o/rep-$share-$tag"
        echo "nsys: $pre" >> "$o/$share-$tag.cmd"
    fi
    # shellcheck disable=SC2086
    "${DR[@]}" --name "$NAME-$share-$tag" --gpus "\"device=$card\"" "$IMG" \
        bash -c "env $env X=1 nice -n 19 timeout $TIMEOUT $pre $NEW/test-backend-ops test -b CUDA0 $args > $o/$share-$tag.log 2>&1; echo $share-$tag rc=\$? >> $o/rc.txt" \
        > /dev/null 2>&1
    say "  $share $tag (card $card): $(grep "^$share-$tag " "$o/rc.txt" | tail -n 1); $(strip "$o/$share-$tag.log" | grep -a -E 'tests passed' | tail -n 1 | sed 's/^ *//'); FAIL $(strip "$o/$share-$tag.log" | grep -a -c ': FAIL')"
    if [ -n "$pre" ]; then
        # the kernels' names and launch counts (cuda_gpu_kern_sum: column 3 the launches, the last the demangled name)
        "${DR[@]}" --name "$NAME-$share-$tag-stats" "$IMG" bash -c \
            "$NSYS stats -q -r cuda_gpu_kern_sum -f csv -o $o/kern-$share-$tag $o/rep-$share-$tag.nsys-rep > $o/stats-$share-$tag.log 2>&1" \
            > /dev/null 2>&1
        awk -F'"' 'function last(s, t,   i, j) { i = 0; while ((j = index(substr(s, i + 1), t)) > 0) i += j; return i }
            NR > 1 && NF > 0 {
                if (NF >= 3) { split($1, f, ","); n = $2 } else { split($0, f, ","); n = f[9] }
                k = last(n, ">(")
                if (k > 0) n = substr(n, 1, k); else if ((k = index(n, "(")) > 0) n = substr(n, 1, k - 1)
                sub(/^void /, "", n); gsub(/<unnamed>::/, "", n)
                print f[3] "\t" n
            }' "$o/kern-$share-$tag"_cuda_gpu_kern_sum.csv 2> /dev/null | sort -t$'\t' -k2 > "$o/kern-$share$([ "$tag" = live ] || echo "-$tag").tsv"
        # no timing on this machine: the report and its times go unread
        rm -f "$o/rep-$share-$tag".* "$o/kern-$share-$tag"_*.csv
        say "  $share $tag: $(wc -l < "$o/kern-$share$([ "$tag" = live ] || echo "-$tag").tsv") kernels in the trace"
    fi
}

if [ -n "$PREFLIGHT" ]; then
    $PREFLIGHT -g "$GPUS" > "$o/preflight.txt" 2>&1
    rc=$?
    say "preflight rc=$rc"
    [ $rc = 0 ] || { say "preflight failed: nothing run"; date -Is > "$WORK/DONE"; exit 1; }
fi

if has list; then
    # shellcheck disable=SC2086
    "${DR[@]}" --name "$NAME-list" --gpus "\"device=${G[0]}\"" "$IMG" bash -c \
        "env $PASCAL_FORCED timeout 1800 $NEW/test-backend-ops support -b CUDA0 -p '$F' > $l/support.txt 2>&1;
         env $PASCAL_FORCED timeout 1800 $NEW/test-backend-ops test -b CUDA0 -p '$F' --record $l/dryrec --record-dry > $l/dry.txt 2>&1;
         echo list rc=\$? >> $o/rc.txt" > /dev/null 2>&1
    strip "$l/support.txt" | sed -n 's/^  \([A-Za-z0-9_]*\)(.*/\1/p' | sort -u > "$l/ops-support.txt"
    strip "$l/support.txt" | grep -a -c ': SUPPORTED$' > "$l/n-supported.txt"
    strip "$l/dry.txt" | grep -a -E '^  [A-Za-z0-9_]+\(.*\): in [0-9]+ B, ref [0-9]+ B$' > "$l/dry-cases.txt"
    sed -E 's/^  ([A-Za-z0-9_]+)\(.*/\1/' "$l/dry-cases.txt" | sort | uniq -c | awk '{print $2 "\t" $1}' > "$l/dry-ops.tsv"
    cut -f1 "$l/dry-ops.tsv" | grep -vx -e MUL_MAT -e MUL_MAT_ID | grep -vxF -f "$l/ops-support.txt" > "$l/ops-fused.txt"
    say "list: $(wc -l < "$l/ops-support.txt") operators in the support listing ($(cat "$l/n-supported.txt") supported cases);" \
        "dry listing $(wc -l < "$l/dry-cases.txt") cases of $(wc -l < "$l/dry-ops.tsv") operators; fused share $(wc -l < "$l/ops-fused.txt") operators"
    [ -s "$l/ops-support.txt" ] && [ -s "$l/ops-fused.txt" ] || { say "list failed: nothing run"; date -Is > "$WORK/DONE"; exit 1; }
fi

chain() {  # <card> <share>
    local card=$1 share=$2
    has live && tbo "$card" "$share" live "" nsys
    if has probe; then
        case $share in
            fa) tbo "$card" fa fault "GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1" ;;
            fused)
                tbo "$card" moe check "GGML_CUDA_PERSISTENT_MOE_CHECK=1"
                tbo "$card" moe fault "GGML_CUDA_PERSISTENT_MOE_FAULT=3" ;;
        esac
    fi
    say "$share done"
}
if has live || has probe; then
    i=0
    for share in fa rest fused; do
        chain "${G[$i]}" "$share" &
        i=$((i + 1))
    done
    wait
fi

if has ornith; then
    ORN_SW=$(unset PRESET_ENV; . "$REPO/release/ornith-v5/settings.env" > /dev/null 2>&1
             for kv in $PRESET_ENV; do case $kv in GGML_CUDA_FATTN_*) printf '%s ' "$kv" ;; esac; done)
    [ -n "$ORN_SW" ] || { say "no GGML_CUDA_FATTN_* switch in $REPO/release/ornith-v5/settings.env"; date -Is > "$WORK/DONE"; exit 1; }
    say "ornith adds: $ORN_SW"
    tbo "${G[0]}" fa ornith "$ORN_SW" nsys
    tbo "${G[0]}" fa ornith-fault "$ORN_SW GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1"
fi

: > "$o/summary.txt"
for f in "$o"/*.log; do
    t=$(basename "$f" .log)
    case $t in stats-*) continue ;; esac
    echo "$t: $(grep "^$t " "$o/rc.txt" | tail -n 1); $(strip "$f" | grep -a -m1 'tests passed' | sed 's/^ *//'); OK $(strip "$f" | grep -a -c ': OK$'); FAIL $(strip "$f" | grep -a -c ': FAIL'); not supported $(strip "$f" | grep -a -c 'not supported'); illegal $(strip "$f" | grep -a -c 'illegal memory access')" >> "$o/summary.txt"
    strip "$f" | grep -a ': FAIL' | grep -av 'Backend' | sed 's/^ */    /' | cut -c1-260 >> "$o/summary.txt"
done
say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | paste -sd' ')"
date -Is > "$WORK/DONE"
