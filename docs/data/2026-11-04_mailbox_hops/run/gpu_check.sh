#!/usr/bin/env bash
# C143 on two CUDA cards (any two of a machine with several, none of them busy): the mailbox hop where a layer split
# really crosses between cards. The rig build runs here through the driver's JIT of its compute 6.1 PTX, so the
# checks run the exact code the rig runs. Every step runs in a container that sees only those two cards.
#   1. mailbox   test-hop-mailbox: bytes exact over the two cards for every ring setting (the default; one chunk; a
#                ring of 4 slots of 1 KiB, so hops wrap it, on the copy stream and on the compute stream; the copy
#                stream always; 64 and 1,024 threads; no poll pause and a 50 us one; a 16 KiB limit beside the staged
#                hop), with held streams so writers wait for acknowledgements and readers for flags; the logged mailbox
#                hop counts must match the copies it takes
#   2. model     test-hop-cast on the random qwen4exp (MODEL, from test-llama-archs) split over the two cards with
#                -ts 1,3 (one boundary, before layer 1), every position's logits per arm against the first arm: a
#                prompt of 48 tokens in three micro-batches in flight (pipeline parallelism), a 4-token pass, six
#                single steps.
#                  run 1, reference off (cudaMemcpyPeerAsync, F32): the staged hop and every mailbox arm bit for bit
#                  (GGML_CUDA_HOP_MAILBOX=2, so cards with peer access take it too): default, 1 chunk, the tiny ring,
#                  the tiny ring with the reader on the compute stream, the copy stream always, 64 threads without
#                  a poll pause, decode-sized hops only (the rest staged)
#                  run 2, reference the Q8_0 cast through the staged hop (S35's line): the cast through the mailbox,
#                  default and tiny ring, bit for bit (checked from the arm table: test-hop-cast's "exact" is for arms
#                  without a cast); every hop within the Q8_0 bound
#   3. archs     test-llama-archs on every architecture over both cards and the CPU, three runs: the default twice
#                (the device rows repeat to the last digit: the runs are deterministic) and with
#                GGML_CUDA_HOP_MAILBOX=2: the same rows, numbers included. Not in the default steps: through the
#                compute 6.1 PTX the sweep stops at its first Meta row, before the two-card layer-split rows, because
#                llama.cpp's all-reduce has no compute 6.1 code (Q879); it runs once test-llama-archs can leave the
#                Meta config out (C319)
# Env: TREE (an unpacked build: bin/ with test-hop-mailbox, test-hop-cast, test-llama-archs), MODEL, OUT, GPUS (the
#      two cards, e.g. 5,6), [IMAGE] [GPU_REQ (default --gpus "device=$GPUS")] [DOCKER_ARGS]
#      [STEPS='mailbox model'] [FREE_MIB=1000] [ROUNDS=96]
# Output: OUT/<step>.log, OUT/model-*.tsv, OUT/summary.tsv (step, verdict, detail), OUT/DONE.
set -uo pipefail
TREE=$(realpath "${TREE:?TREE=<unpacked build>}")
MODEL=$(realpath "${MODEL:?MODEL=<random qwen4exp gguf>}")
OUT=$(realpath -m "${OUT:?OUT=<dir>}")
GPUS=${GPUS:?GPUS=<two card ids, e.g. 5,6>}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
STEPS=${STEPS:-mailbox model}
# the cards' request: docker's --gpus by default; GPU_REQ (words without quotes) for another, e.g. CDI's
# "--device nvidia.com/gpu=5 --device nvidia.com/gpu=6"
if [ -n "${GPU_REQ:-}" ]; then read -ra req <<< "$GPU_REQ"; else req=(--gpus "\"device=$GPUS\""); fi
mkdir -p "$OUT/jit-cache"
rm -f "$OUT/DONE"
printf 'step\tverdict\tdetail\n' > "$OUT/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$OUT/summary.tsv"; }
step() { [[ " $STEPS " == *" $1 "* ]]; }

for g in ${GPUS//,/ }; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "${used:-99999}" -ge "${FREE_MIB:-1000}" ]; then
        verdict start FAIL "card $g has $used MiB in use"; date -Is > "$OUT/DONE"; exit 1
    fi
done

mounts="-v $TREE:$TREE:ro -v $(dirname "$MODEL"):$(dirname "$MODEL"):ro -v $OUT:$OUT"
gpu_run() { # <script>
    # shellcheck disable=SC2086
    docker run --rm ${DOCKER_ARGS:-} "${req[@]}" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_CACHE_MAXSIZE=4294967296 \
        -v "$OUT/jit-cache:/root/.nv" $mounts -e B="$TREE/bin" -e M="$MODEL" -e O="$OUT" \
        -e ROUNDS="${ROUNDS:-96}" "$IMAGE" bash -c "$1"
}
echo "=== gpu check $(date -Is) on cards $GPUS, steps: $STEPS"

if step mailbox; then
    gpu_run '$B/test-hop-mailbox --rounds $ROUNDS > $O/mailbox.log 2>&1'
    rc=$?
    nok=$(grep -c ' ok  : ' "$OUT/mailbox.log")
    verdict mailbox "$([ $rc = 0 ] && [ "$nok" -ge 9 ] && echo OK || echo FAIL)" \
        "rc $rc; $nok settings ok; $(tail -n 1 "$OUT/mailbox.log")"
fi

if step model; then
    gpu_run '$B/test-hop-cast -m $M -ngl 99 -ts 1,3 --tsv $O/model-f32.tsv \
        --arm off \
        --arm "staged exact GGML_CUDA_HOP_STAGED=2" \
        --arm "mbox exact GGML_CUDA_HOP_MAILBOX=2" \
        --arm "mbox-c1 exact GGML_CUDA_HOP_MAILBOX=2 GGML_CUDA_HOP_MAILBOX_CHUNKS=1" \
        --arm "mbox-tiny exact GGML_CUDA_HOP_MAILBOX=2 GGML_CUDA_HOP_MAILBOX_CHUNKS=64 GGML_CUDA_HOP_MAILBOX_SLOTS=4 GGML_CUDA_HOP_MAILBOX_SLOT_KB=1" \
        --arm "mbox-tiny-s0 exact GGML_CUDA_HOP_MAILBOX=2 GGML_CUDA_HOP_MAILBOX_CHUNKS=64 GGML_CUDA_HOP_MAILBOX_SLOTS=4 GGML_CUDA_HOP_MAILBOX_SLOT_KB=1 GGML_CUDA_HOP_MAILBOX_STREAM=0" \
        --arm "mbox-s2 exact GGML_CUDA_HOP_MAILBOX=2 GGML_CUDA_HOP_MAILBOX_STREAM=2" \
        --arm "mbox-t64-p0 exact GGML_CUDA_HOP_MAILBOX=2 GGML_CUDA_HOP_MAILBOX_THREADS=64 GGML_CUDA_HOP_MAILBOX_POLL_US=0" \
        --arm "mbox-max256-staged exact GGML_CUDA_HOP_MAILBOX=2 GGML_CUDA_HOP_MAILBOX_MAX_KB=256 GGML_CUDA_HOP_STAGED=2" \
        > $O/model-f32.log 2>&1; echo "f32 rc=$?" > $O/model.rc
        $B/test-hop-cast -m $M -ngl 99 -ts 1,3 --tsv $O/model-q8.tsv \
        --arm "q8_0-staged LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=2" \
        --arm "q8_0-mbox LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_MAILBOX=2" \
        --arm "q8_0-mbox-tiny LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_MAILBOX=2 GGML_CUDA_HOP_MAILBOX_CHUNKS=64 GGML_CUDA_HOP_MAILBOX_SLOTS=4 GGML_CUDA_HOP_MAILBOX_SLOT_KB=1" \
        > $O/model-q8.log 2>&1; echo "q8 rc=$?" >> $O/model.rc'
    # test-hop-cast's "exact" is for arms without a cast, so the cast arms through the mailbox are checked here:
    # identical to the reference (the cast through the staged hop), no logit difference
    nq=$(grep -cE '^  q8_0-mbox(-tiny)? +yes +0 +0 +0 +1\.0000 ' "$OUT/model-q8.log")
    # the mailboxes each context made and used (from the library log), and the copy stream for prompt hops
    nm=$(cat "$OUT"/model-*.log | grep -c 'by mailbox: [0-9]* slots')
    nu=$(cat "$OUT"/model-*.log | grep -o 'by mailbox: [0-9]* hops' | grep -vc 'by mailbox: 0 hops')
    nc=$(cat "$OUT"/model-*.log | grep -o 'by mailbox: [0-9]* hops,[^)]*)' | grep -vc ' 0 on the copy stream')
    ok=1; grep -q 'rc=[1-9]' "$OUT/model.rc" && ok=0; [ "$nu" -gt 0 ] && [ "$nc" -gt 0 ] && [ "$nq" = 2 ] || ok=0
    verdict model "$([ $ok = 1 ] && echo OK || echo FAIL)" \
        "$(tr '\n' ' ' < "$OUT/model.rc")$nm mailboxes made, $nu used, $nc with prompt hops on the copy stream; q8_0 through the mailbox identical to q8_0 staged: $nq of 2 arms; f32: $(tail -n 1 "$OUT/model-f32.log"); q8: $(tail -n 1 "$OUT/model-q8.log")"
fi

if step archs; then
    gpu_run 'for a in default default2 mailbox; do
                 e=""; [ $a = mailbox ] && e="GGML_CUDA_HOP_MAILBOX=2"
                 env $e $B/test-llama-archs -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > $O/archs-$a.log
                 echo "$a rc=${PIPESTATUS[0]}"
             done' > "$OUT/archs.log" 2>&1
    rows() { grep '^|' "$OUT/archs-$1.log"; }
    rep=$(cmp -s <(rows default) <(rows default2) && echo yes || echo no)
    same=$(cmp -s <(rows default) <(rows mailbox) && echo yes || echo no)
    samev=$(cmp -s <(rows default | sed 's/([0-9.e+-]*)//g') <(rows mailbox | sed 's/([0-9.e+-]*)//g') && echo yes || echo no)
    nmb=$(grep -o 'by mailbox: [0-9]* hops' "$OUT/archs-mailbox.log" | grep -vc 'by mailbox: 0 hops')
    v=REVIEW; { [ "$same" = yes ] || { [ "$rep" = no ] && [ "$samev" = yes ]; }; } && [ "$nmb" -gt 0 ] && v=OK
    verdict archs "$v" "$(tr '\n' ' ' < "$OUT/archs.log")default repeats to the digit: $rep; mailbox rows the same: $same (verdicts: $samev); $nmb contexts used the mailbox; $(grep -m 1 '^Summary' "$OUT/archs-mailbox.log")"
fi

date -Is > "$OUT/DONE"
echo "=== gpu check ended $(date -Is)"
