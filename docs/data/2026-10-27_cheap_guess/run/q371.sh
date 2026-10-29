#!/usr/bin/env bash
# Q371 + Q372: two training changes for the MTP head as one arm against a control, both warm-started from the same head
# and trained for the same optimizer steps on the same data. The experiment draws the after-miss window more often
# (train.py --kinds 0.4,0.1,0.5: staircase, pre-card, after-miss; the default is 0.6,0.15,0.25, Q371) and weights the
# loss by what llama-server's lookups would guess at each generated position (source_labels.py's labels,
# --source-weights 0.6,0.6,3,3,1: copy lookup, 2-token continuation, 1-token continuation, none, other; Q372). Then both
# arms are scored like Q384/Q385: the held-out sessions and the agent check's own text at g9 and g9miss, α 3 (served)
# and α 1, split by breakdown.py.
# On the GPU machine, in containers; run from a machine that reaches it over ssh, one step at a time:
#   STEPS=labels, STEPS=smoke (10 steps of the experiment arm: the new options load), STEPS=start (both arms, detached),
#   STEPS=wait, STEPS=score.
# Env: HOST, WORK (folder there for the tools and labels), ARMS (folder there for the two arms' heads), START (head
# folder to warm-start from: its head.pt and cmd.txt), SESSIONS (the run's export/train/sessions_seq.jsonl), OUT (local);
# for score: TEST_FEATS, TEST_ROWS (the held-out dump and rows), AGENT_FEATS, AGENT_ROWS (Q384's dump and rows), VOCAB,
# FLINT_DIR; optional GPUS ("2 5": control, experiment), GPU_SCORE (2), MAX_STEPS (1100), MINUTES (360, a cap),
# DOCKER_ARGS (labels; -v mounts of every folder START's cmd.txt reads: features, vocabulary, Flint's weights), TORCH_IMAGE
# (pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
H=${HOST:?}; W=${WORK:?}/q371; A=${ARMS:?}; O=${OUT:?}
: "${START:?}" "${SESSIONS:?}"
read -r GPU_C GPU_E <<< "${GPUS:-2 5}"
GPU_SCORE=${GPU_SCORE:-2}; MAX_STEPS=${MAX_STEPS:-1100}; MINUTES=${MINUTES:-360}; STEPS=${STEPS:?labels, smoke, start, wait or score}
TORCH_IMAGE=${TORCH_IMAGE:-pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime}
DA=${DOCKER_ARGS:-}
EXP_OPTS="--kinds 0.4,0.1,0.5 --source-labels $W/labels.npz --source-weights 0.6,0.6,3,3,1"
mkdir -p "$O"
ssh "$H" "mkdir -p $W/bench $A/control $A/exp"
tar cf - -C "$REPO/bench" --exclude=__pycache__ staircase_drafter ngram_replay | ssh "$H" "tar xf - -C $W/bench"
S=$W/bench/staircase_drafter
m="-v $W:$W -v $A:$A -v $START:$START:ro -v $(dirname "$SESSIONS"):$(dirname "$SESSIONS"):ro"
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
# the start head's training command with this run's output, budget and warm start (its --resume, if any, dropped)
arm_cmd() {   # arm folder, extra options
    ssh "$H" "cat $START/cmd.txt" | sed -e "s# --out [^ ]*# --out $1#" -e 's# --minutes [^ ]*##' -e 's# --resume##' \
        -e "s#\$# --init-from $1/start.pt --max-steps $MAX_STEPS --minutes $MINUTES $2#"
}
if has labels; then
    ssh "$H" "docker run --rm $DA --cpu-shares 256 --network none $m -w $S $TORCH_IMAGE \
        python -I source_labels.py $SESSIONS $W/labels.npz --jobs 8" | tail -2 | tee "$O/labels.txt"
fi
if has smoke; then
    c=$(arm_cmd $A/smoke "$EXP_OPTS" | sed -e "s# --max-steps $MAX_STEPS# --max-steps 10#" -e "s# --minutes $MINUTES# --minutes 30#")
    ssh "$H" "mkdir -p $A/smoke && cp $START/head.pt $A/smoke/start.pt && docker run --rm $DA --cpu-shares 256 \
        --gpus '\"device=$GPU_E\"' -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        --shm-size 8g --network none $m -w $S $TORCH_IMAGE bash -c '$c' > $A/smoke/train.log 2>&1; \
        grep -E 'init_from|source_weights|kinds|\"step\": 10|final_val|Error|error' $A/smoke/train.log | cut -c1-300" \
        | tee "$O/smoke.txt"
fi
if has start; then
    for arm in control exp; do
        g=$GPU_C; opts=""
        [ $arm = exp ] && { g=$GPU_E; opts=$EXP_OPTS; }
        c=$(arm_cmd $A/$arm "$opts")
        echo "$c" | tee "$O/cmd-$arm.txt"
        ssh "$H" "cp $START/head.pt $A/$arm/start.pt && echo '$c' > $A/$arm/cmd.txt && docker rm -f pq-q371-$arm >/dev/null 2>&1; \
            docker run -d --name pq-q371-$arm $DA --cpu-shares 256 --gpus '\"device=$g\"' -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --shm-size 8g --network none $m -w $S $TORCH_IMAGE \
            bash -c '$c >> $A/$arm/train.log 2>&1'"
    done
fi
if has wait; then
    for arm in control exp; do
        ssh "$H" "docker wait pq-q371-$arm; tail -1 $A/$arm/train.log | cut -c1-400" | tee -a "$O/end.txt"
    done
fi
if has score; then
    : "${TEST_FEATS:?}" "${TEST_ROWS:?}" "${AGENT_FEATS:?}" "${AGENT_ROWS:?}" "${VOCAB:?}" "${FLINT_DIR:?}"
    for d in $(dirname "$TEST_FEATS") $(dirname "$TEST_ROWS") $(dirname "$AGENT_FEATS") $(dirname "$AGENT_ROWS") \
             $(dirname "$VOCAB") "$FLINT_DIR"; do m="$m -v $d:$d:ro"; done
    for arm in control exp; do
        for t in heldout agent; do
            if [ $t = heldout ]; then f=$TEST_FEATS; r=$TEST_ROWS; sp="--spans $r"; else f=$AGENT_FEATS; r=$AGENT_ROWS; sp=""; fi
            for a in 3 1; do
                n=$arm-$t-a$a
                ssh "$H" "docker run --rm $DA --cpu-shares 256 --gpus '\"device=$GPU_SCORE\"' -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
                    --network none $m -w $S $TORCH_IMAGE python -I score.py --heads $A/$arm --feats $f --gen $r $sp \
                    --vocab $VOCAB --flint $FLINT_DIR --only-windows g9,g9miss --alpha $a --out $W/scores-$n > $W/scores-$n.md 2>&1; \
                    grep -E '\| g9' $W/scores-$n.md" | sed "s/^/$n: /" | tee -a "$O/scores.txt"
                ssh "$H" "cat $W/scores-$n.md" > "$O/scores-$n.md"
                ssh "$H" "docker run --rm $DA --cpu-shares 256 --network none $m -w $S $TORCH_IMAGE \
                    python -I breakdown.py $r $W/scores-$n/scores-A.npz --windows g9,g9miss" > "$O/breakdown-$n.md"
            done
        done
    done
fi
