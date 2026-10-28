#!/usr/bin/env bash
# Q396's cheaper draft paths, each scored with run/mtp_variant.cpp (FAST: the makers'-chain acceptance and top-1) on the
# first CODE_CHUNKS (40) x 512 tokens of the code text and AGENT_CHUNKS (2) x 8,192 of the agent text: the target is
# the model file itself, the draft a file from run/draft_file.sh (or the same file
# with a draft-only override). A variant's draft file is made when first needed and deleted after the variant unless
# another listed variant reads it. Variants:
#   ref                       the MTP block, embedding and head in Q8_0 (the makers' types)
#   m-q4k, m-q2k              the whole MTP block in Q4_K / Q2_K
#   e-q2k                     only its routed experts in Q2_K
#   top4, top2                ref with 4 / 2 routed experts per draft (expert_used_count, draft model only)
#   shared                    ref with the routed experts' down projection zeroed: the shared expert alone
#   h-q5_0, h-q4_0            the output head in Q5_0 / Q4_0
#   sl-8192 ... sl-65536      ref, the draft head restricted to the shortlist run/shortlist.py wrote (SL=<prefix>)
#   screen                    h-q4_0's head ranks every token, the model's own Q8_0 rows rescore its top 256
#   QUANT=<llama-quantize> SRC=<BF16 .gguf> BIN=<mtp-variant> MODEL=<target .gguf> BLOCK=<40 or 48> WORK=<folder>
#   CODE=<code text> AGENT=<agent text> SL=<shortlist prefix> GPUS=<ids> NAME=<prefix of the tables>
#   [ALLOW_REQUANT=1] [ARCH=qwen35moe] [HEAD_REF=<the reference head type, q8_0>] [EMBD=<embedding type, q8_0>] \
#   bash variants.sh <variant> ...
set -euo pipefail
: "${QUANT:?}" "${SRC:?}" "${BIN:?}" "${MODEL:?}" "${BLOCK:?}" "${WORK:?}" "${CODE:?}" "${AGENT:?}" "${GPUS:?}" "${NAME:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
ARCH=${ARCH:-qwen35moe}
TRUNK_RE="^blk\\.([0-9]|[1-$(( (BLOCK - 1) / 10 ))][0-9])\\."
[ $(( (BLOCK - 1) % 10 )) -ne 9 ] && TRUNK_RE="^blk\\.([0-9]|[1-$(( (BLOCK - 1) / 10 - 1 ))][0-9]|$(( (BLOCK - 1) / 10 ))[0-$(( (BLOCK - 1) % 10 ))])\\."
mkdir -p "$WORK/draft" "$WORK/logs"
file_of() {  # the draft file a variant reads
    case $1 in
        m-q4k|m-q2k|e-q2k|h-q5_0|h-q4_0|shared) echo "$WORK/draft/$NAME-$1.gguf" ;;
        screen) echo "$WORK/draft/$NAME-h-q4_0.gguf" ;;
        *) echo "$WORK/draft/$NAME-ref.gguf" ;;
    esac
}
make_file() {  # make_file VARIANT
    local f; f=$(file_of "$1")
    [ -f "$f" ] && return
    local mtp=q8_0 head=${HEAD_REF:-q8_0}
    case $1 in
        m-q4k) mtp=q4_k ;; m-q2k) mtp=q2_k ;; e-q2k) mtp=e:q2_k ;;
        h-q5_0) head=q5_0 ;; h-q4_0|screen) head=q4_0 ;;
    esac
    if [ "$1" = shared ]; then
        make_file ref
        cp "$(file_of ref)" "$f"
        python3 -I "$HERE/tensor_zero.py" "$f" "blk.$BLOCK.ffn_down_exps.weight"
        return
    fi
    QUANT=$QUANT SRC=$SRC OUT=$f BLOCK=$BLOCK MTP=$mtp HEAD=$head bash "$HERE/draft_file.sh"
}
for v in "$@"; do
    echo "== $(date +%T) $v"
    make_file "$v"
    unset DRAFT_KV SHORTLIST SCREEN
    case $v in
        top4) export DRAFT_KV=$ARCH.expert_used_count=int:4 ;;
        top2) export DRAFT_KV=$ARCH.expert_used_count=int:2 ;;
        sl-*) export SHORTLIST=${SL:?}-${v#sl-}.txt ;;
        screen) export SCREEN=256 ;;
    esac
    export DRAFT=$(file_of "$v") DRAFT_CPU=$TRUNK_RE
    FAST=1 BIN=$BIN MODEL=$MODEL TEXT=$CODE CHUNK=512 CHUNKS=${CODE_CHUNKS:-40} GPUS=$GPUS OUT=$WORK/logs/var-$NAME-$v-code.tsv \
        bash "$HERE/run_accept.sh" | sed -n 2p
    sleep 10
    FAST=1 BIN=$BIN MODEL=$MODEL TEXT=$AGENT CHUNK=8192 CHUNKS=${AGENT_CHUNKS:-2} GPUS=$GPUS OUT=$WORK/logs/var-$NAME-$v-agent.tsv \
        bash "$HERE/run_accept.sh" | sed -n 2p
    sleep 10
    # delete the file unless a later variant reads it
    f=$(file_of "$v"); keep=0; later=0
    for w in "$@"; do
        [ "$w" = "$v" ] && { later=1; continue; }
        [ $later = 1 ] && [ "$(file_of "$w")" = "$f" ] && keep=1
        [ $later = 1 ] && [ "$w" = shared ] && [ "$f" = "$(file_of ref)" ] && keep=1
    done
    [ $keep = 0 ] && rm -f "$f"
done
echo "== $(date +%T) done"
