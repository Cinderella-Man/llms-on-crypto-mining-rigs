#!/usr/bin/env bash
# C200/C201 on two CUDA cards (any two of a machine with several, none of them busy): the card-hop switches where a
# layer split really crosses between cards. Runs every step in a container that sees only those two cards.
#   1. ops       test-backend-ops -o CPY on each card against the CPU: every copy case, the new ones at the hop
#                shapes ([2560, 4, 1 / 4 / 512] to and from F16, BF16 and Q8_0) among them
#   2. unit      test-hop-cast --unit-only: the round trips on the CPU and on both cards
#   3. model     test-hop-cast on the random qwen4exp (MODEL, from test-llama-archs) split over the two cards with
#                -ts 1,3 (layer 0 on the first card, layer 1 on the second: one boundary, before layer 1), every
#                position's logits per arm against the first arm (off: the driver's cudaMemcpyPeerAsync, F32):
#                  the staged hop (GGML_CUDA_HOP_STAGED=2, so even cards with peer access take it) in 2 chunks, in 1,
#                  in 4, with 4 KiB slots (a 16-token micro-batch's 64 KiB hop then takes 16 chunks through 2 slots: the
#                  ring wraps and the copy stream runs behind the scheduler's guard), with the copy stream for every hop
#                  and with none: all bit for bit the reference's logits (the bytes are copied exactly);
#                  the casts across the real boundary (LLAMA_HOP_CAST=f16, bf16, q8_0, and q8_0 with the staged hop):
#                  each hop within the round-trip bound, layer 0 unchanged
#                The prompt (48 tokens) runs as three micro-batches in flight (pipeline parallelism), then a 4-token
#                pass and six single steps.
#   4. archs     test-llama-archs on every architecture, the cards and the CPU, at the default and with
#                GGML_CUDA_HOP_STAGED=2: each device setting's logits against the CPU's (the multi-card layer split
#                among them), the same verdicts in both runs
#   5. real      only with REAL_MODEL and TEXT: llama-perplexity on a real model split over both cards (-ts 1,1), 8
#                chunks of 512, with the switches off and with GGML_CUDA_HOP_STAGED=2: the same perplexity to the last
#                printed digit
# Env: TREE (an unpacked build: bin/ with test-backend-ops, test-hop-cast, test-llama-archs, llama-perplexity; the rig
#      tarball runs here through the driver's JIT of its compute 6.1 PTX), MODEL, OUT, GPUS (the two cards, e.g. 3,4),
#      [IMAGE] [DOCKER_ARGS] [REAL_MODEL] [TEXT] [STEPS='ops unit model archs real'] [FREE_MIB=1000]
# Output: OUT/<step>.log, OUT/model.tsv, OUT/summary.tsv (step, verdict, detail), OUT/DONE.
set -uo pipefail
TREE=$(realpath "${TREE:?TREE=<unpacked build>}")
MODEL=$(realpath "${MODEL:?MODEL=<random qwen4exp gguf>}")
OUT=$(realpath -m "${OUT:?OUT=<dir>}")
GPUS=${GPUS:?GPUS=<two card ids, e.g. 3,4>}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
STEPS=${STEPS:-ops unit model archs real}
mkdir -p "$OUT"
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
[ -n "${REAL_MODEL:-}" ] && mounts="$mounts -v $(dirname "$REAL_MODEL"):$(dirname "$REAL_MODEL"):ro"
[ -n "${TEXT:-}" ] && mounts="$mounts -v $(dirname "$TEXT"):$(dirname "$TEXT"):ro"
gpu_run() { # <script>
    # shellcheck disable=SC2086
    docker run --rm ${DOCKER_ARGS:-} --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_CACHE_MAXSIZE=4294967296 \
        -v "$OUT/jit-cache:/root/.nv" $mounts -e B="$TREE/bin" -e M="$MODEL" -e O="$OUT" -e R="${REAL_MODEL:-}" \
        -e T="${TEXT:-}" "$IMAGE" bash -c "$1"
}
mkdir -p "$OUT/jit-cache"

if step ops; then
    gpu_run 'for d in CUDA0 CUDA1; do $B/test-backend-ops -b $d -o CPY > $O/ops-$d.log 2>&1; echo "$d rc=$?"; done' > "$OUT/ops.log" 2>&1
    d=""; ok=1
    for c in CUDA0 CUDA1; do
        nh=$(grep -c 'CPY(type_src=[a-z0-9_]*,type_dst=[a-z0-9_]*,ne_src=\[2560,4,.*OK' "$OUT/ops-$c.log")
        nf=$(grep -c 'FAIL' "$OUT/ops-$c.log")
        d="$d$c: $(grep -E 'tests passed' "$OUT/ops-$c.log" | tail -n 1 | tr -s ' '), $nh hop-shape cases OK, $nf FAIL lines; "
        [ "$nf" = 0 ] && [ "$nh" = 18 ] || ok=0
    done
    grep -q 'rc=[1-9]' "$OUT/ops.log" && ok=0
    verdict ops "$([ $ok = 1 ] && echo OK || echo FAIL)" "$d"
fi

if step unit; then
    gpu_run '$B/test-hop-cast --unit-only > $O/unit.log 2>&1'
    rc=$?
    verdict unit "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(grep -c '^  CUDA' "$OUT/unit.log") card rows; $(tail -n 1 "$OUT/unit.log")"
fi

if step model; then
    gpu_run '$B/test-hop-cast -m $M -ngl 99 -ts 1,3 --tsv $O/model.tsv \
        --arm off \
        --arm "staged exact GGML_CUDA_HOP_STAGED=2" \
        --arm "staged-c1 exact GGML_CUDA_HOP_STAGED=2 GGML_CUDA_HOP_CHUNKS=1" \
        --arm "staged-c4 exact GGML_CUDA_HOP_STAGED=2 GGML_CUDA_HOP_CHUNKS=4" \
        --arm "staged-4k exact GGML_CUDA_HOP_STAGED=2 GGML_CUDA_HOP_SLOT_KB=4 GGML_CUDA_HOP_SLOTS=2" \
        --arm "staged-s2 exact GGML_CUDA_HOP_STAGED=2 GGML_CUDA_HOP_STREAM=2" \
        --arm "staged-s0-4k exact GGML_CUDA_HOP_STAGED=2 GGML_CUDA_HOP_STREAM=0 GGML_CUDA_HOP_SLOT_KB=4 GGML_CUDA_HOP_SLOTS=2" \
        --arm "f16 LLAMA_HOP_CAST=f16" --arm "bf16 LLAMA_HOP_CAST=bf16" --arm "q8_0 LLAMA_HOP_CAST=q8_0" \
        --arm "q8_0-staged LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=2" > $O/model.log 2>&1'
    rc=$?
    # the boundary the cast found and the staged rings, from the library log
    nb=$(grep -c 'card hops in [a-z0-9_]* before layers 1 (the layer split)' "$OUT/model.log")
    nr=$(grep -c 'staged through' "$OUT/model.log")
    nc=$(grep -o 'staged hops,[^)]*)' "$OUT/model.log" | grep -vc ' 0 on the copy stream')
    verdict model "$([ $rc = 0 ] && [ "$nb" -gt 0 ] && [ "$nr" -gt 0 ] && [ "$nc" -gt 0 ] && echo OK || echo FAIL)" \
        "rc $rc; $nb contexts cast before layer 1, $nr staged rings, $nc rings used the copy stream; $(tail -n 1 "$OUT/model.log")"
fi

if step archs; then
    gpu_run '$B/test-llama-archs -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > $O/archs-default.log; echo "default rc=${PIPESTATUS[0]}"
             GGML_CUDA_HOP_STAGED=2 $B/test-llama-archs -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > $O/archs-staged.log; echo "staged rc=${PIPESTATUS[0]}"' \
        > "$OUT/archs.log" 2>&1
    fd=$(grep -c '^|.*FAIL' "$OUT/archs-default.log"); fs=$(grep -c '^|.*FAIL' "$OUT/archs-staged.log")
    same=$(cmp -s <(grep '^|' "$OUT/archs-default.log" | sed 's/([0-9.e+-]*)//g') <(grep '^|' "$OUT/archs-staged.log" | sed 's/([0-9.e+-]*)//g') && echo yes || echo no)
    verdict archs "$([ "$fs" -le "$fd" ] && [ "$same" = yes ] && echo OK || echo REVIEW)" \
        "$(tr '\n' ' ' < "$OUT/archs.log")default $(grep -m 1 '^Summary' "$OUT/archs-default.log"), $fd FAIL rows; staged $(grep -m 1 '^Summary' "$OUT/archs-staged.log"), $fs FAIL rows; same verdicts: $same"
fi

if step real && [ -n "${REAL_MODEL:-}" ] && [ -n "${TEXT:-}" ]; then
    gpu_run 'for a in off staged; do
                 e=""; [ $a = staged ] && e="GGML_CUDA_HOP_STAGED=2"
                 env $e $B/llama-perplexity -m $R -f $T -c 512 --chunks 8 -ngl 99 -ts 1,1 -fa on > $O/real-$a.log 2>&1
                 echo "$a rc=$?"
             done' > "$OUT/real.log" 2>&1
    po=$(grep -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' "$OUT/real-off.log")
    ps=$(grep -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' "$OUT/real-staged.log")
    verdict real "$([ -n "$po" ] && [ "$po" = "$ps" ] && grep -q 'staged through' "$OUT/real-staged.log" && echo OK || echo FAIL)" \
        "off: $po; staged: $ps; $(grep -c 'staged through' "$OUT/real-staged.log") staged rings"
fi

date -Is > "$OUT/DONE"
