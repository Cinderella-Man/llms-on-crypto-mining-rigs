#!/usr/bin/env bash
# C933 (2026-11-24): why two runs of the same hand-run test on the same model wrote different logs. Runs three of the
# hand step's tests N times in each of three ways of catching their output:
#   attach  docker run ... <test> > <host file> 2>&1: docker carries stdout and stderr as two streams and its client
#           writes both into the file (the hand step's way until C933's fix)
#   inside  docker run ... bash -c '<test> > <file> 2>&1': one file opened inside the container
#   split   docker run ... bash -c '<test> > <file>.out 2> <file>.err': stdout and stderr in two files
# Tests: llama-eval-callback and test-llama-context -m on DENSE, test-bytes-switches on MOE (the hand step's arguments).
# Each output file is reduced as compare_hand.py does (time stamps dropped, numbers in timing lines masked) and listed
# in $OUT/streams.tsv: test, way, run, file, lines, md5 of the reduced file, sorted md5 (the md5 of its sorted lines),
# and for eval-callback the number of "sum = " lines.
# Usage: BIN=<build bin/> DENSE=<llama-dense.gguf> MOE=<qwen4exp-moe.gguf> OUT=<dir> [N=4] [THREADS=6]
#   [IMAGE=pq-c565-cpu:24.04] [MOUNTS='-v <dir>:<dir> ...'] [DOCKER_ARGS=...] bash streams.sh
set -uo pipefail
: "${BIN:?BIN=<build bin/>}" "${DENSE:?DENSE=<llama-dense.gguf>}" "${MOE:?MOE=<qwen4exp-moe.gguf>}" "${OUT:?OUT=<dir>}"
N=${N:-4}
THREADS=${THREADS:-6}
IMAGE=${IMAGE:-pq-c565-cpu:24.04}
mkdir -p "$OUT"
# shellcheck disable=SC2206
D=(--rm --cpu-shares=256 --network none ${DOCKER_ARGS:-} ${MOUNTS:-} -v "$OUT:$OUT")
reduce() { sed -E 's/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ [A-Z] ?//; /( ms\b| ms$|ms per token|tokens per second| t\/s|time =|seconds)/ s/[0-9]+(\.[0-9]+)?/N/g' "$1"; }
row() { # <test> <way> <run> <file>
    local f=$4 sums="-"
    [ "$1" = eval-callback ] && sums=$(grep -ac 'sum = ' "$f")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(basename "$f")" "$(reduce "$f" | wc -l)" \
        "$(reduce "$f" | md5sum | cut -c1-12)" "$(reduce "$f" | LC_ALL=C sort | md5sum | cut -c1-12)" "$sums"
}
printf 'test\tway\trun\tfile\tlines\tmd5\tsorted_md5\tsums\n' > "$OUT/streams.tsv"
for t in eval-callback context-model bytes-switches; do
    case $t in
        eval-callback)  cmd=("$BIN/llama-eval-callback" -m "$DENSE" --prompt hello --seed 42 -ngl 0) ;;
        context-model)  cmd=("$BIN/test-llama-context" -m "$DENSE") ;;
        bytes-switches) cmd=("$BIN/test-bytes-switches" -m "$MOE" -t "$THREADS" -n 24 -g 24 -k 1024 --router-scale 30) ;;
    esac
    for i in $(seq 1 "$N"); do
        docker run "${D[@]}" "$IMAGE" "${cmd[@]}" > "$OUT/$t-attach-$i.log" 2>&1
        row "$t" attach "$i" "$OUT/$t-attach-$i.log" >> "$OUT/streams.tsv"
        docker run "${D[@]}" "$IMAGE" bash -c '"$@" > '"$OUT/$t-inside-$i.log"' 2>&1' _ "${cmd[@]}"
        row "$t" inside "$i" "$OUT/$t-inside-$i.log" >> "$OUT/streams.tsv"
        docker run "${D[@]}" "$IMAGE" bash -c '"$@" > '"$OUT/$t-split-$i.out"' 2> '"$OUT/$t-split-$i.err" _ "${cmd[@]}"
        row "$t" split "$i" "$OUT/$t-split-$i.out" >> "$OUT/streams.tsv"
        row "$t" split "$i" "$OUT/$t-split-$i.err" >> "$OUT/streams.tsv"
    done
    echo "$t done $(date -Is)"
done
column -t -s $'\t' "$OUT/streams.tsv"
