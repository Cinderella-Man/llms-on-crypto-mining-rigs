#!/usr/bin/env bash
# One Pascal mix from its plan (bench/pascal_mix/pmix.py): the base file's tensors byte for byte except those the plan
# changes, which come from llama-quantize of the Swift-1.5 Q8_0 of the same model (the router from gguf_retype.py).
#   1. r/  the base with its router F32 -> BF16 (scripts/gguf_retype.py, the plan's retype.txt), checked by `verify`
#   2. i/  r with every tensor of take.txt taken from the Q8_0 (scripts/gguf_compose.py)
#   3. q/  llama-quantize --allow-requantize --keep-split --imatrix IMATRIX --tensor-type-file types.txt of i: the
#          tensors of take.txt to their new types, every other matrix to its own type (copied); the log must show
#          exactly take.txt's tensors converted
#   4. OUT the final file: r with take.txt's tensors from q (gguf_compose.py, `verify`ed), so every tensor the plan
#          keeps is the base's own bytes whatever llama-quantize did with it; md5 of every shard
# Unchanged shards (the n-gram table) are symbolic links at every step; TABLE, when set, is where OUT's table shard
# finally links. r/, i/ and q/ are removed at the end. Steps done before are skipped (WORK/<name>/<step>.done).
# CPU only, pinned to CPUS at nice 19 and idle I/O priority; llama-quantize runs in IMAGE (a CUDA build, with the
# driver stubbed).
#   REPO=<this repository> PLAN=<pmix.py --out> NAME=<prefix of the new shards> BASE=<first shard of the base>
#   DONOR=<first shard of the Q8_0> IMATRIX=<imatrix.gguf> QUANTIZE=<llama-quantize> WORK=<scratch> OUT=<dir>
#   [TABLE=<n-gram table shard>] [CPUS=10,11,22,23] [THREADS=4] [IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04]
#   [MOUNTS="<dirs mounted at the same path>"] bash build.sh
set -euo pipefail
: "${REPO:?}" "${PLAN:?}" "${NAME:?}" "${BASE:?}" "${DONOR:?}" "${IMATRIX:?}" "${QUANTIZE:?}" "${WORK:?}" "${OUT:?}"
CPUS=${CPUS:-10,11,22,23} THREADS=${THREADS:-4} IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
W=$WORK/$NAME
mkdir -p "$W" "$OUT"
log() { printf '%s %s %s\n' "$(date '+%F %T')" "$NAME" "$*"; }
run() { taskset -c "$CPUS" nice -n 19 ionice -c3 "$@"; }
shards() { ls "${1%-00001-of-*}"-0000?-of-0000?.gguf; }
in_dir() { local s out=(); for s in "${@:2}"; do out+=("$1/$(basename "$s")"); done; printf '%s\n' "${out[@]}"; }
if [ -f "$W/out.done" ]; then log "built before: $(tr '\n' ' ' < "$OUT/$NAME.md5")"; exit 0; fi
mapfile -t B < <(shards "$BASE")
mapfile -t D < <(shards "$DONOR")
mapfile -t R < <(in_dir "$W/r" "${B[@]}")
mapfile -t I < <(in_dir "$W/i" "${B[@]}")
TAKE=()
while IFS= read -r t; do [ -n "$t" ] && TAKE+=(--take "$t"); done < "$PLAN/take.txt"
n=$((${#TAKE[@]} / 2))
compose=(python3 -I "$REPO/scripts/gguf_compose.py")
retype=(python3 -I "$REPO/scripts/gguf_retype.py")

if [ ! -f "$W/r.done" ]; then
    log "1: router to BF16 ($(cat "$PLAN/retype.txt"))"
    rm -rf "$W/r"
    run "${retype[@]}" retype --rule "$(cat "$PLAN/retype.txt")" --jobs "$THREADS" --link-unchanged --out-dir "$W/r" "${B[@]}"
    run "${retype[@]}" verify "${B[@]}" --new "${R[@]}" | tee "$W/r.verify"
    date -Is > "$W/r.done"
fi
if [ ! -f "$W/i.done" ]; then
    log "2: $n tensors from the Q8_0"
    rm -rf "$W/i"
    run "${compose[@]}" compose --base "${R[@]}" --donor "${D[@]}" "${TAKE[@]}" --out-dir "$W/i" --link-unchanged
    date -Is > "$W/i.done"
fi
if [ ! -f "$W/q.done" ]; then
    log "3: llama-quantize ($(wc -l < "$PLAN/types.txt") type rules, imatrix $(basename "$IMATRIX"))"
    rm -rf "$W/q"
    mkdir -p "$W/q"
    margs=""
    for d in $MOUNTS; do margs="$margs -v $d:$d"; done
    # shellcheck disable=SC2016,SC2086
    docker run --rm --label pascal-quants=1 --cpu-shares 256 --cpuset-cpus "$CPUS" --name "pq-pmix-quantize-$NAME" \
        $margs "$IMAGE" bash -c 'mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 &&
            LD_LIBRARY_PATH=/tmp/stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} exec nice -n 19 "$@"' quantize \
        "$QUANTIZE" --allow-requantize --keep-split --imatrix "$IMATRIX" --tensor-type-file "$PLAN/types.txt" \
        "${I[0]}" "$W/q/$NAME" Q8_0 "$THREADS" > "$W/quantize.log" 2>&1
    # the converted tensors must be exactly take.txt's
    awk '/^\[ *[0-9]+\/ *[0-9]+\]/ { t = $0; sub(/^\[[^]]*\] */, "", t); sub(/ .*/, "", t) }
         /converting to/ && t != "" { print t; t = "" }' "$W/quantize.log" | sort > "$W/converted.txt"
    if ! cmp -s "$W/converted.txt" <(sort "$PLAN/take.txt"); then
        log "llama-quantize converted other tensors than take.txt:"; diff "$W/converted.txt" <(sort "$PLAN/take.txt") | head
        exit 1
    fi
    date -Is > "$W/q.done"
fi
mapfile -t Q < <(shards "$W/q/$NAME-00001-of-0000${#B[@]}.gguf")
if [ ! -f "$W/out.done" ]; then
    log "4: the final file in $OUT"
    run "${compose[@]}" compose --base "${R[@]}" --donor "${Q[@]}" "${TAKE[@]}" --out-dir "$OUT" --prefix "$NAME" \
        --link-unchanged
    mapfile -t F < <(shards "$OUT/$NAME-00001-of-0000${#B[@]}.gguf")
    run "${compose[@]}" verify --base "${R[@]}" --donor "${Q[@]}" "${TAKE[@]}" --new "${F[@]}" | tee "$W/out.verify"
    if [ -n "${TABLE:-}" ]; then
        for f in "${F[@]}"; do [ -L "$f" ] && ln -sfn "$TABLE" "$f"; done
    fi
    (cd "$OUT" && for f in "${F[@]}"; do [ -L "$f" ] || run md5sum "$(basename "$f")"; done) > "$OUT/$NAME.md5"
    date -Is > "$W/out.done"
fi
rm -rf "$W/r" "$W/i" "$W/q"
log "done: $(tr '\n' ' ' < "$OUT/$NAME.md5")"
