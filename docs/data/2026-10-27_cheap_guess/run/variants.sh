#!/usr/bin/env bash
# The cheap-guess build on the rig's first card (nothing else running): (1) correctness of the C++ head against
# drafter.Drafter.forward on a random-weight head of the real shape with the release split's windows
# (check_export.py case, made by sim.sh case), per option; (2) the cost of one guess of the head per set of options
# (LLAMA_DRAFT_* and GGML_CUDA_MMVQ_MAX_COLS in the environment) at the server's 4,096-position ring (--ctx 2048 gives
# the same 4,352 cells); (3) per-kernel traces of three sets. Relays the build and the case first; the build stays in
# RIG_DIR for agent.sh, the case is removed.
#   RIG=rig BUILD_HOST=<ssh alias> BUILD_WORK=<folder there, as for build.sh and sim.sh> NAME=<tarball name> \
#   RIG_DIR=<rig folder for the build> HEAD=<rig path of the head GGUF> OUT=<local folder> bash variants.sh
set -uo pipefail
RIG=${RIG:-rig}
H=${BUILD_HOST:?ssh alias of the build machine}
QD=${BUILD_WORK:?folder on the build machine}
NAME=${NAME:?llama.cpp-...-pascal-p106-v3-cheapguess}
RB=${RIG_DIR:?rig folder for the build}
HEAD=${HEAD:?rig path of the head GGUF}
O=${OUT:?local folder}
BIN=$RB/$NAME/bin
SPLIT=${SPLIT:-6,6,5,5,6,5,5,6,5}
R=${RIG_TMP:-/tmp/cheap-guess-variants}
mkdir -p "$O"
if ! ssh "$RIG" "test -x $BIN/llama-pipe-drafter-check"; then
    want=$(ssh "$H" "md5sum $QD/out/$NAME.tar.gz" | cut -d' ' -f1)
    ssh "$H" "cat $QD/out/$NAME.tar.gz" | ssh "$RIG" "mkdir -p $RB && cat > $RB/$NAME.tar.gz"
    have=$(ssh "$RIG" "md5sum $RB/$NAME.tar.gz" | cut -d' ' -f1)
    [ "$want" = "$have" ] || { echo "tarball relay failed ($want / $have)"; exit 1; }
    ssh "$RIG" "tar xzf $RB/$NAME.tar.gz -C $RB && rm $RB/$NAME.tar.gz"
    echo "tarball $want on the rig"
fi
ssh "$RIG" "rm -rf $R && mkdir -p $R"
ssh "$H" "cat $QD/case/G9/drafter.gguf" | ssh "$RIG" "cat > $R/case.gguf"
ssh "$H" "cat $QD/case/G9/case.bin" | ssh "$RIG" "cat > $R/case.bin"

echo "=== correctness $(date +%T)" | tee "$O/check.log"
for env in "LLAMA_DRAFT_PRUNE=0" "LLAMA_DRAFT_PRUNE=1" "LLAMA_DRAFT_ANC=dense" "LLAMA_DRAFT_SPLIT=4" \
           "GGML_CUDA_MMVQ_MAX_COLS=4" "LLAMA_DRAFT_HEAD=q8_0"; do
    echo "--- $env" | tee -a "$O/check.log"
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=0 $env $BIN/llama-pipe-drafter-check $R/case.gguf $R/case.bin --split $SPLIT 2>&1 \
        | grep -E '^step|DRAFTER_CHECK|error'" | tee -a "$O/check.log"
done

bench() {   # tag, environment, ctx (and more arguments)
    echo "--- $1 ($2) ctx $3 $(date +%T)" | tee -a "$O/bench.log"
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=0 $2 $BIN/llama-pipe-drafter-check $HEAD --bench 300 --split $SPLIT --commits 1 \
        --ctx $3 2>&1 | grep -E 'DRAFTER_BENCH|init:|error|failed'" | tee -a "$O/bench.log"
}
Q8="LLAMA_DRAFT_HEAD=q8_0"
MMQ="GGML_CUDA_MMVQ_MAX_COLS=4"
echo "=== cost of one guess $(date +%T)" | tee "$O/bench.log"
bench full       "LLAMA_DRAFT_PRUNE=0" 2048
bench prune      "" 2048
bench q8         "$Q8" 2048
bench q8mmq      "$Q8 $MMQ" 2048
bench q8split    "$Q8 LLAMA_DRAFT_SPLIT=4" 2048
bench q8dense    "$Q8 LLAMA_DRAFT_ANC=dense" 2048
bench q8densemmq "$Q8 LLAMA_DRAFT_ANC=dense $MMQ" 2048
bench v8kq8mmq   "LLAMA_DRAFT_VOCAB=8192 $Q8 $MMQ" 2048
bench q8mmq8k    "$Q8 $MMQ" 8192
bench q8mmq-w1   "$Q8 $MMQ" "2048 --window 1"
bench q8split-w1 "$Q8 LLAMA_DRAFT_SPLIT=4" "2048 --window 1"
echo "=== traces $(date +%T)" | tee -a "$O/bench.log"
for t in "q8:$Q8" "q8mmq:$Q8 $MMQ" "q8split:$Q8 LLAMA_DRAFT_SPLIT=4"; do
    tag=${t%%:*}; env=${t#*:}
    echo "--- trace-$tag ($env) ctx 2048 $(date +%T)" | tee -a "$O/bench.log"
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=0 $env GGML_CUDA_PROFILE=$R/prof-$tag.csv $BIN/llama-pipe-drafter-check $HEAD \
        --bench 100 --split $SPLIT --commits 1 --ctx 2048 2>&1 | grep DRAFTER_BENCH" | tee -a "$O/bench.log"
    ssh "$RIG" "cat $R/prof-$tag.csv" > "$O/prof-$tag.csv"
done
ssh "$RIG" "rm -rf $R"
echo "=== done $(date +%T)"
