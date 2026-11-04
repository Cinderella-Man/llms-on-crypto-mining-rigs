#!/usr/bin/env bash
# Q928: Swift-1.5 Qwen3.8-Flash-Next with N2P experts, on a CPU. The routed experts are requantized from the Swift-1.5
# Q8_0 with an importance matrix; every other tensor is the pmix-q2-fast file's, byte for byte (its dense retypes, the
# makers' hyper-connections, output layer and n-gram table; docs/data/2026-11-04_pascal_mix/).
#   a/  llama-quantize --pure of the Q8_0: ffn_gate_exps and ffn_up_exps N2P (2.6875 bpw), ffn_down_exps N2P-128 (2.75;
#       the 640-wide rows), every other tensor left at Q8_0 (copied); the log must show exactly the expert tensors converted
#   1.  OUT1: pmix-q2-fast with every routed expert tensor from a/ (scripts/gguf_compose.py, `verify`ed)
#   b/  the same with ffn_down_exps N3P-128 (3.375 bpw) and the gate/up experts left at Q8_0 (copied)
#   2.  OUT2: OUT1 with ffn_down_exps from b/ (`verify`ed)
# Then md5 of every new shard; a/ and b/ are removed at the end. Steps done before are skipped (WORK/<step>.done).
# llama-quantize runs in IMAGE (a CUDA build, the driver stubbed) at nice 19 with --cpu-shares 256; the compose steps on
# the host at nice 19 and idle I/O priority. Unchanged shards (the n-gram table) are symbolic links.
#   REPO=<this repository> Q8=<first shard of the Swift-1.5 Q8_0> IMATRIX=<imatrix.gguf> BASE=<first shard of
#   pmix-q2-fast> QUANTIZE=<llama-quantize of the n2p branch> WORK=<scratch> OUT1=<dir> OUT2=<dir> NAME1=<prefix>
#   NAME2=<prefix> [THREADS=8] [IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04] [MOUNTS="<dirs mounted at the same path>"]
#   bash quantize.sh
set -uo pipefail
: "${REPO:?}" "${Q8:?}" "${IMATRIX:?}" "${BASE:?}" "${QUANTIZE:?}" "${WORK:?}" "${OUT1:?}" "${OUT2:?}" "${NAME1:?}" "${NAME2:?}"
THREADS=${THREADS:-8} IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p "$WORK/logs" "$OUT1" "$OUT2"
log() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$WORK/logs/chain.log"; }
run() { nice -n 19 ionice -c3 "$@"; }
shards() { ls "${1%-00001-of-*}"-0000?-of-0000?.gguf; }
compose=(python3 -I "$REPO/scripts/gguf_compose.py")
EXPERTS=(--take 're:^blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight$')
DOWN=(--take 're:^blk\.[0-9]+\.ffn_down_exps\.weight$')
mapfile -t B < <(shards "$BASE")

# llama-quantize of the Q8_0 into dir $1 with the tensor types $3..; checks that every tensor whose name matches regex
# $2 was converted (other tensors that --pure converts, e.g. BF16 ones, are not taken from it)
quantize() {
    local dir=$1 want=$2; shift 2
    rm -rf "${WORK:?}/$dir" && mkdir -p "$WORK/$dir"
    local margs=""
    for d in ${MOUNTS:-}; do margs="$margs -v $d:$d"; done
    # shellcheck disable=SC2016,SC2086
    docker run --rm --label pascal-quants=1 --cpu-shares 256 --name "pq-n2p-quantize-$dir" $margs "$IMAGE" \
        bash -c 'mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 &&
            LD_LIBRARY_PATH=/tmp/stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} exec nice -n 19 "$@"' quantize \
        "$QUANTIZE" --allow-requantize --pure --keep-split --imatrix "$IMATRIX" "$@" \
        "$Q8" "$WORK/$dir/q" Q8_0 "$THREADS" > "$WORK/logs/quantize-$dir.log" 2>&1
    local rc=$?
    if [ $rc != 0 ]; then
        log "$dir: llama-quantize failed (rc=$rc)"; tail -n 20 "$WORK/logs/quantize-$dir.log"; return 1
    fi
    awk '/^\[ *[0-9]+\/ *[0-9]+\]/ { t = $0; sub(/^\[[^]]*\] */, "", t); sub(/ .*/, "", t); print t }' "$WORK/logs/quantize-$dir.log" |
        grep -E "$want" | sort > "$WORK/$dir.listed"
    awk '/^\[ *[0-9]+\/ *[0-9]+\]/ { t = $0; sub(/^\[[^]]*\] */, "", t); sub(/ .*/, "", t) }
         /converting to/ && t != "" { print t; t = "" }' "$WORK/logs/quantize-$dir.log" | grep -E "$want" | sort > "$WORK/$dir.converted"
    if [ ! -s "$WORK/$dir.listed" ] || ! cmp -s "$WORK/$dir.listed" "$WORK/$dir.converted"; then
        log "$dir: not every tensor matching $want was converted:"; diff "$WORK/$dir.listed" "$WORK/$dir.converted" | head
        return 1
    fi
    log "$dir: $(wc -l < "$WORK/$dir.converted") tensors converted"
}

if [ ! -f "$WORK/a.done" ]; then
    log "a: gate/up N2P, down N2P-128 ($THREADS threads, imatrix $(basename "$IMATRIX"))"
    quantize a 'ffn_(gate|up|down)_exps' --tensor-type ffn_gate_exps=n2p --tensor-type ffn_up_exps=n2p \
        --tensor-type ffn_down_exps=n2p_128 || exit 1
    date -Is > "$WORK/a.done"
fi
mapfile -t A < <(shards "$(ls "$WORK"/a/q-00001-of-*.gguf)")
if [ ! -f "$WORK/out1.done" ]; then
    log "1: $NAME1 in $OUT1"
    run "${compose[@]}" compose --base "${B[@]}" --donor "${A[@]}" "${EXPERTS[@]}" --out-dir "$OUT1" --prefix "$NAME1" \
        --link-unchanged || exit 1
    mapfile -t F1 < <(shards "$(ls "$OUT1/$NAME1"-00001-of-*.gguf)")
    run "${compose[@]}" verify --base "${B[@]}" --donor "${A[@]}" "${EXPERTS[@]}" --new "${F1[@]}" | tee "$WORK/out1.verify"
    [ "${PIPESTATUS[0]}" = 0 ] || exit 1
    (cd "$OUT1" && for f in "${F1[@]}"; do [ -L "$f" ] || run md5sum "$(basename "$f")"; done) > "$OUT1/$NAME1.md5"
    date -Is > "$WORK/out1.done"
fi
mapfile -t F1 < <(shards "$(ls "$OUT1/$NAME1"-00001-of-*.gguf)")
if [ ! -f "$WORK/b.done" ]; then
    log "b: down N3P-128 ($THREADS threads)"
    quantize b 'ffn_down_exps' --tensor-type ffn_down_exps=n3p_128 --tensor-type ffn_gate_exps=q8_0 \
        --tensor-type ffn_up_exps=q8_0 || exit 1
    date -Is > "$WORK/b.done"
fi
mapfile -t Bq < <(shards "$(ls "$WORK"/b/q-00001-of-*.gguf)")
if [ ! -f "$WORK/out2.done" ]; then
    log "2: $NAME2 in $OUT2"
    run "${compose[@]}" compose --base "${F1[@]}" --donor "${Bq[@]}" "${DOWN[@]}" --out-dir "$OUT2" --prefix "$NAME2" \
        --link-unchanged || exit 1
    mapfile -t F2 < <(shards "$(ls "$OUT2/$NAME2"-00001-of-*.gguf)")
    run "${compose[@]}" verify --base "${F1[@]}" --donor "${Bq[@]}" "${DOWN[@]}" --new "${F2[@]}" | tee "$WORK/out2.verify"
    [ "${PIPESTATUS[0]}" = 0 ] || exit 1
    (cd "$OUT2" && for f in "${F2[@]}"; do [ -L "$f" ] || run md5sum "$(basename "$f")"; done) > "$OUT2/$NAME2.md5"
    date -Is > "$WORK/out2.done"
fi
rm -rf "${WORK:?}/a" "${WORK:?}/b"
log "done: $(tr '\n' ' ' < "$OUT1/$NAME1.md5") $(tr '\n' ' ' < "$OUT2/$NAME2.md5")"
date -Is > "$WORK/DONE"
