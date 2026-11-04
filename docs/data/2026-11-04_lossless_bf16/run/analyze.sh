#!/usr/bin/env bash
# The entropy table of Q621: the census of each file's unquantized tensors (census.py), then lossless_entropy over
# each file (entropies, x12 and x11 sizes, the round-trip check of every weight) and the bench's weight sample from the
# first file. CPU only, at nice 19.
# Env: WORK: a checkout of this repository with bench/mmvq_pascal/build/lossless_entropy built (build_lossless.sh);
#      FILES: "name=first shard ..." (each a split GGUF's -00001-of- shard; the sample comes from the first);
#      OUT: where the tables go (default $WORK/lossless_out).
set -euo pipefail
: "${WORK:?}" "${FILES:?}"
OUT=${OUT:-$WORK/lossless_out}
RUN=$WORK/docs/data/2026-11-04_lossless_bf16/run
BIN=$WORK/bench/mmvq_pascal/build/lossless_entropy
mkdir -p "$OUT"

# 4 copies of each hyper-connection projection from across the layers, 8 of the injects, 2 of the indexer and router
# matrices, 4 of the recurrent layers' alpha and beta, the PLE value projection; class:calls per token as in the
# BF16 bench (bench/mmvq_pascal/README.md, "Flash-Next's BF16 products")
TAKE=""
add() { local cls=$1 calls=$2; shift 2; for t in "$@"; do TAKE+="${TAKE:+,}$t=$cls:$calls"; done; }
add hc_up 97 blk.0.hc_attn_up.weight blk.16.hc_ffn_up.weight blk.31.hc_attn_up.weight blk.47.hc_ffn_up.weight
add hc_down 97 blk.0.hc_attn_down.weight blk.16.hc_ffn_down.weight blk.31.hc_attn_down.weight blk.47.hc_ffn_down.weight
add hc_inject 96 blk.0.hc_attn_inject.weight blk.8.hc_attn_inject.weight blk.16.hc_attn_inject.weight \
    blk.24.hc_attn_inject.weight blk.31.hc_ffn_inject.weight blk.39.hc_ffn_inject.weight blk.44.hc_ffn_inject.weight \
    blk.47.hc_ffn_inject.weight
add indexer_q 12 blk.3.indexer.q_proj.weight blk.27.indexer.q_proj.weight
add indexer_k 12 blk.3.indexer.k_proj.weight blk.27.indexer.k_proj.weight
add router 48 blk.0.ffn_gate_inp.weight blk.30.ffn_gate_inp.weight
add ssm_ab 72 blk.0.ssm_alpha.weight blk.0.ssm_beta.weight blk.20.ssm_alpha.weight blk.46.ssm_beta.weight
add ple_value 1 blk.1.ple_value.weight

first=1
for f in $FILES; do
    name=${f%%=*} shard=${f#*=}
    dir=$(dirname "$shard")
    # shellcheck disable=SC2086 # the glob picks the shards
    nice -n 19 python3 -I "$RUN/census.py" ${shard/-00001-of-/-0000?-of-} > "$OUT/census_$name.tsv"
    sample=()
    if [ $first = 1 ]; then
        sample=(--sample "$OUT/sample.lbb" --take "$TAKE")
        first=0
    fi
    nice -n 19 "$BIN" --census "$OUT/census_$name.tsv" --models "$dir" --out "$OUT/tensors_$name.tsv" "${sample[@]}" \
        2> "$OUT/entropy_$name.log"
    tail -n 1 "$OUT/entropy_$name.log"
done
[ -f "$OUT/sample.lbb" ] && md5sum "$OUT/sample.lbb" | sed "s|$OUT/||" > "$OUT/sample.lbb.md5"
python3 -I "$RUN/table.py" "$OUT"/tensors_*.tsv > "$OUT/entropy.md"
echo "done: $OUT/entropy.md"
