#!/usr/bin/env bash
# Q384: the heads scored offline on the agent check's own replayed traces, the text the served acceptance (0.248) was
# measured on, to split the gap to the held-out sessions' offline 0.314 into domain (the check's Exercism tasks, three
# in JavaScript) and engine. The traces' 36 requests become rows (agent_tokens.py --traces --id-base: prompt and reply
# tokens with Flint's chat template), are dumped like the training data (llama-staircase-dump with the 26-layer grid)
# and scored with score.py at the nine-card windows (g9, g9miss); source_split.py splits them by what the server's
# lookups would guess, breakdown.py by segment, Flint's top probability and position. On the GPU machine, in containers; run from a machine that reaches it over ssh.
# Env: HOST, WORK (folder there), BODIES (the replayed arm's proxy.jsonl.bodies.gz there), FLINT_HF (Flint's tokenizer
# and chat template), MODEL (Flint Q8_0; MODEL_DIRS: folders to mount so it resolves), DUMP_BIN (llama-staircase-dump),
# HEADS (head folders, train.py's output), VOCAB (vocab.npy), FLINT_DIR (flint_weights.py's output), OUT (local);
# optional TRACES (the six of replay.sh), GPUS_DUMP (2,6), GPU_SCORE (0), ALPHA (1; outputs get -a<ALPHA> otherwise),
# STEPS (rows dump score split), DOCKER_ARGS,
# TOKENIZER_IMAGE, TORCH_IMAGE (pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime), CUDA_IMAGE
# (nvidia/cuda:12.9.2-devel-ubuntu24.04), GRID (the 26 layers), DCTX (66048).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
H=${HOST:?}; W=${WORK:?}/q384; O=${OUT:?}
: "${BODIES:?}" "${FLINT_HF:?}" "${MODEL:?}" "${DUMP_BIN:?}" "${HEADS:?}" "${VOCAB:?}" "${FLINT_DIR:?}"
TRACES=${TRACES:-rel/bugfix-javascript-space-age/a1/t1,rel/bugfix-javascript-phone-number/a1/t1,rel/bugfix-javascript-complex-numbers/a1/t1,rel/bugfix-python-affine-cipher/a1/t1,rel/bugfix-python-list-ops/a1/t1,rel/bugfix-python-scale-generator/a1/t1}
GPUS_DUMP=${GPUS_DUMP:-2,6}; GPU_SCORE=${GPU_SCORE:-0}; STEPS=${STEPS:-rows dump score split}
A=${ALPHA:-1}; X=$([ "$A" = 1 ] || echo "-a$A")   # score.py --alpha: the served head sharpens by its GGUF's alpha (3)
TORCH_IMAGE=${TORCH_IMAGE:-pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime}
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
TOKENIZER_IMAGE=${TOKENIZER_IMAGE:?an image with transformers}
GRID=${GRID:-4,6,8,9,10,12,14,16,17,18,20,22,24,25,27,29,30,33,34,36,37,39,41,42,44,47}
DCTX=${DCTX:-66048}
DA=${DOCKER_ARGS:-}
mkdir -p "$O"
ssh "$H" "mkdir -p $W/bench"
tar cf - -C "$REPO/bench" --exclude=__pycache__ staircase_drafter ngram_replay | ssh "$H" "tar xf - -C $W/bench"
S=$W/bench/staircase_drafter
m="-v $W:$W"
for d in $(dirname "$BODIES") "$FLINT_HF" $(dirname "$MODEL") ${MODEL_DIRS:-} $(dirname "$DUMP_BIN") $(dirname "$(dirname "$DUMP_BIN")") \
         $(dirname "$VOCAB") "$FLINT_DIR"; do m="$m -v $d:$d:ro"; done
for h in $HEADS; do m="$m -v $h:$h:ro"; done
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
if has rows; then
    ssh "$H" "docker run --rm $DA --cpu-shares 256 --network none $m --entrypoint python3 $TOKENIZER_IMAGE \
        $S/agent_tokens.py $BODIES --model $FLINT_HF --traces $TRACES --id-base 900000 --out $W/rows.jsonl" | tail -1 | tee "$O/rows.txt"
    ssh "$H" "docker run --rm $DA --cpu-shares 256 --network none $m -w $S --entrypoint bash $TORCH_IMAGE -c \
        'python -I feats.py seqs $W/seqs.txt $W/map.jsonl $W/rows.jsonl --max-tok 1000000'" | tail -1 | tee -a "$O/rows.txt"
fi
if has dump; then
    ssh "$H" "docker run --rm $DA --gpus '\"device=$GPUS_DUMP\"' -e CUDA_DEVICE_ORDER=PCI_BUS_ID $m $CUDA_IMAGE bash -c \
        'DUMP_IN=$W/seqs.txt DUMP_OUT=$W/agentcheck.bin DUMP_LAYERS=$GRID DUMP_ALLPOS=4,6,9 DUMP_THREADS=12 DUMP_GPU_TOPK=1 \
        $DUMP_BIN -m $MODEL -ngl 99 -fa on -c $DCTX -b 512 -ub 512 > $W/dump.log 2>&1; tail -3 $W/dump.log'" | tee "$O/dump.txt"
fi
# one head per call: score.py names its output after the head's profile, which the heads share
if has score; then
    for h in $HEADS; do
        n=$(basename "$h")
        ssh "$H" "docker run --rm $DA --gpus '\"device=$GPU_SCORE\"' -e CUDA_DEVICE_ORDER=PCI_BUS_ID --network none $m -w $S \
            $TORCH_IMAGE python -I score.py --heads $h --feats $W/agentcheck.bin --gen $W/rows.jsonl --vocab $VOCAB \
            --flint $FLINT_DIR --only-windows g9,g9miss --alpha $A --out $W/scores-$n$X > $W/scores-$n$X.md 2>&1; \
            grep -E '^## head|\| g9' $W/scores-$n$X.md" | sed "s/^/$n$X: /" | tee -a "$O/scores.txt"
        ssh "$H" "cat $W/scores-$n$X.md" > "$O/scores-$n$X.md"
    done
fi
if has split; then
    for h in $HEADS; do
        n=$(basename "$h")
        ssh "$H" "docker run --rm $DA --cpu-shares 256 --network none $m -w $S --entrypoint bash $TORCH_IMAGE -c \
            'for w in g9 g9miss; do python -I source_split.py $W/rows.jsonl $W/scores-$n$X/scores-A.npz --window \$w; done'" \
            > "$O/split-$n$X.md"
        ssh "$H" "docker run --rm $DA --cpu-shares 256 --network none $m -w $S $TORCH_IMAGE \
            python -I breakdown.py $W/rows.jsonl $W/scores-$n$X/scores-A.npz --windows g9,g9miss" > "$O/breakdown-$n$X.md"
    done
fi
