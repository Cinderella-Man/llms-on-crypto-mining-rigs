#!/usr/bin/env bash
# Q385: where the head's acceptance comes from on the held-out agent sessions. score.py scores each head at the
# nine-card windows (g9, g9miss), saving Flint's top probability and the overlap sum of min(p, q) per position;
# breakdown.py splits the scores by segment, guess source, Flint's top probability, context length and request number,
# and by source x top probability. On the GPU machine, in containers; run from a machine that reaches it over ssh.
# Env: HOST, WORK (folder there), HEADS (head folders, train.py's output), FEATS (the held-out dump, e.g. seqs/test.bin),
# ROWS (its rows, seqs/test-rows.jsonl), VOCAB (vocab.npy), FLINT_DIR (flint_weights.py's output), OUT (local);
# optional GPU (0), ALPHA (1; outputs get -a<ALPHA> otherwise), STEPS (score breakdown), DOCKER_ARGS (e.g. labels,
# more -v mounts), TORCH_IMAGE
# (pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
H=${HOST:?}; W=${WORK:?}/q385; O=${OUT:?}
: "${HEADS:?}" "${FEATS:?}" "${ROWS:?}" "${VOCAB:?}" "${FLINT_DIR:?}"
GPU=${GPU:-0}; STEPS=${STEPS:-score breakdown}
A=${ALPHA:-1}; X=$([ "$A" = 1 ] || echo "-a$A")   # score.py --alpha: the served head sharpens by its GGUF's alpha (3)
TORCH_IMAGE=${TORCH_IMAGE:-pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime}
DA=${DOCKER_ARGS:-}
mkdir -p "$O"
ssh "$H" "mkdir -p $W/bench"
tar cf - -C "$REPO/bench" --exclude=__pycache__ staircase_drafter ngram_replay | ssh "$H" "tar xf - -C $W/bench"
S=$W/bench/staircase_drafter
m="-v $W:$W"
for d in $(dirname "$FEATS") $(dirname "$ROWS") $(dirname "$VOCAB") "$FLINT_DIR"; do m="$m -v $d:$d:ro"; done
for h in $HEADS; do m="$m -v $h:$h:ro"; done
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
# one head per call: score.py names its output after the head's profile, which the heads share
for h in $HEADS; do
    n=$(basename "$h")
    if has score; then
        ssh "$H" "docker run --rm $DA --cpu-shares 256 --gpus '\"device=$GPU\"' -e CUDA_DEVICE_ORDER=PCI_BUS_ID --network none \
            $m -w $S $TORCH_IMAGE python -I score.py --heads $h --feats $FEATS --gen $ROWS --spans $ROWS --vocab $VOCAB \
            --flint $FLINT_DIR --only-windows g9,g9miss --alpha $A --out $W/scores-$n$X > $W/scores-$n$X.md 2>&1; \
            grep -E '^## head|\| g9' $W/scores-$n$X.md" | sed "s/^/$n$X: /" | tee -a "$O/scores.txt"
        ssh "$H" "cat $W/scores-$n$X.md" > "$O/scores-$n$X.md"
    fi
    if has breakdown; then
        ssh "$H" "docker run --rm $DA --cpu-shares 256 --network none $m -w $S $TORCH_IMAGE \
            python -I breakdown.py $ROWS $W/scores-$n$X/scores-A.npz --windows g9,g9miss" > "$O/breakdown-$n$X.md"
    fi
done
