#!/usr/bin/env bash
# C755 (2026-11-13): multitoken_bitid's mode shdown (Q4_0's and Q5_0's products of 2-4 tokens against their one-token
# products) built for a P106 rig and checked on a build machine, CPU only. No engine change: the mode is linked against
# the static ggml libraries of a rig build of the fork's pascal-p106-v4 (scripts/build_llamacpp_docker.sh). Every build
# and run waits for LOCK (flock), runs in a container without GPUs or network at a low CPU share
# (--cpu-shares=256, nice 19). Steps (STEPS, in this order by default):
#   build  verify/build_multitoken_bitid.sh against SRC (its build/ kept): the binary for the rig and its md5
#   cpu    BITID_CPU=1 multitoken_bitid shdown: the mode on ggml's CPU backend (no GPU; the CUDA driver library is the
#          toolkit's stub, never called): every comparison must be bit-identical; the lines also give each input's
#          blocks with |I| >= 2^13 (est.)
#   route  which products GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_0,q5_0 moves: every tensor of the served files with its
#          type (run/gguf_types.py, the files' headers only: MODEL_FILES, DRAFT), the Q4_0 and Q5_0 ones listed, and the
#          code that reads the switch in SRC
# Env: M (work root), REPO (this repository), LOCK, SRC (the rig build's llama.cpp tree), MODEL_FILES (the main model's
# split files, space-separated), DRAFT (the MTP draft's file), IMAGE, DOCKER_ARGS, STEPS. Output: $M/summary.tsv,
# $M/cpu_shdown.txt, $M/types.tsv, $M/route.tsv, $M/skip_readers.txt, logs in $M/logs.
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
SRC=$(realpath "${SRC:?SRC=<a rig build tree of pascal-p106-v4, build/ kept>}")
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
STEPS=${STEPS:-build cpu route}
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
RUN=$REPO/docs/data/2026-11-13_c755_shexp_downs/run
X=$REPO/bench/mmvq_pascal/build/multitoken_bitid
mkdir -p "$M/logs"
[ -f "$M/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
locked() { flock -w 43200 "$LOCK" "$@"; }
add()    { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$M/summary.tsv"; }
TIP=$(git -C "$SRC" rev-parse HEAD)

if step build; then
    rm -f "$X"
    locked env DOCKER_ARGS="$DA --network none" nice -n 19 bash "$REPO/bench/mmvq_pascal/verify/build_multitoken_bitid.sh" \
        "$SRC" "$SRC" > "$M/logs/build.log" 2>&1
    rc=$?
    add build "$([ $rc = 0 ] && [ -x "$X" ] && grep -aqw shdown "$X" && echo OK || echo FAIL)" \
        "rc=$rc; multitoken_bitid md5 $(md5sum < "$X" 2> /dev/null | cut -d' ' -f1), $(stat -c %s "$X" 2> /dev/null) B; linked against ${TIP:0:9} ($(git -C "$SRC" log -1 --format=%s "$TIP" | cut -c1-70)); $(grep -ac warning "$M/logs/build.log") warning lines"
fi

if step cpu; then
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$REPO:$REPO:ro" "$IMAGE" bash -c "
        mkdir -p /tmp/stub && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1
        BITID_CPU=1 LD_LIBRARY_PATH=/tmp/stub:/usr/local/cuda/lib64 nice -n 19 $X shdown" > "$M/cpu_shdown.txt" 2>&1
    rc=$?
    n=$(grep -c ': bit-identical\|: DIFFERENT' "$M/cpu_shdown.txt")
    add cpu "$([ $rc = 0 ] && [ "$n" = 72 ] && grep -q '^backend: CPU' "$M/cpu_shdown.txt" && echo OK || echo FAIL)" \
        "rc=$rc; $n product lines (expected 72: 2 types x 4 shapes x 2-4 tokens x 3 inputs); $(tail -n 1 "$M/cpu_shdown.txt")"
fi

if step route; then
    # shellcheck disable=SC2086
    nice -n 19 python3 "$RUN/gguf_types.py" "$SRC/ggml/include/ggml.h" ${MODEL_FILES:?MODEL_FILES=<split files>} ${DRAFT:-} \
        > "$M/types.tsv" 2> "$M/logs/route.log"
    rc=$?
    awk -F'\t' 'NR == 1 { print "file\ttype\ttensors\tnames (blk.N. left out)\tne0 x ne1"; next }
        $2 == "Q4_0" || $2 == "Q5_0" {
            nm = $1; sub(/^blk\.[0-9]+\./, "", nm); k = $6 "\t" $2 "\t" nm "\t" $3 " x " $4; c[k]++
        }
        END { for (k in c) { split(k, a, "\t"); print a[1] "\t" a[2] "\t" c[k] "\t" a[3] "\t" a[4] } }' "$M/types.tsv" \
        | sort > "$M/route.tsv"
    git -C "$SRC" grep -n 'GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP' "$TIP" -- ggml/src > "$M/skip_readers.txt"
    other=$(awk -F'\t' 'NR > 1 && $4 != "ffn_down_shexp.weight"' "$M/route.tsv" | wc -l)
    add route "$([ $rc = 0 ] && [ -s "$M/route.tsv" ] && [ "$other" = 0 ] && echo OK || echo FAIL)" \
        "rc=$rc; $(($(wc -l < "$M/types.tsv") - 1)) tensors; Q4_0/Q5_0: $(awk -F'\t' 'NR > 1 { printf "%s %s %s x %s; ", $3, $2, $4, $5 }' "$M/route.tsv")other Q4_0/Q5_0 tensors: $other; the switch is read in $(cut -d: -f2 "$M/skip_readers.txt" | sort -u | tr '\n' ' ')"
fi
echo done > "$M/DONE"
