#!/usr/bin/env bash
# The offline part, on the build machine (CPU only, Docker): the correctness case for variants.sh, the shortlist
# coverage, the agent check's requests as token sequences, the break-even sweep and the head's acceptance by guess
# source. Tools in bench/staircase_drafter/; this repo's copy of bench/ is sent to BUILD_WORK/bench first.
#   BUILD_HOST=<ssh alias> BUILD_WORK=<folder there> bash sim.sh STEP...
# STEP:
#   case      random-weight head of the real shape + reference logits (check_export.py case); GGUF_PY=<gguf-py folder>
#   coverage  shortlist_coverage.py; VOCAB=<vocab.npy> SESSIONS=<sessions_seq.jsonl> IDS=<held-out id list>
#   tokens    agent_tokens.py over the agent check's proxy logs; BODIES="<proxy.jsonl.bodies.gz> ..." FLINT_HF=<folder
#             with Flint's tokenizer.json and chat template>
#   cost      cost_sim.py on the tokens (card 1, then a spare card)
#   split     source_split.py; ROWS=<the rows score.py scored> SCORES=<score.py's scores-A.npz>
# Images: TORCH_IMAGE (numpy, torch), TOKENIZER_IMAGE (transformers); plain python3 for coverage and cost.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
H=${BUILD_HOST:?ssh alias of the build machine}
D=${BUILD_WORK:?folder on the build machine}
TORCH_IMAGE=${TORCH_IMAGE:-pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime}
TOKENIZER_IMAGE=${TOKENIZER_IMAGE:?an image with transformers, e.g. the vLLM image}
DOCKER_ARGS=${DOCKER_ARGS:-}
ssh "$H" "mkdir -p $D/bench $D/case $D/sim"
tar cf - -C "$REPO/bench" --exclude=__pycache__ staircase_drafter ngram_replay | ssh "$H" "tar xf - -C $D/bench"
S=$D/bench/staircase_drafter
dk() {   # image, mounts, command
    ssh "$H" "docker run --rm --cpu-shares 256 $DOCKER_ARGS --network none -v $D:$D $2 -w $S --entrypoint bash $1 -c '$3'"
}
for step in "$@"; do
    echo "=== $step $(date +%T)"
    case $step in
        case)
            dk "$TORCH_IMAGE" "-v ${GGUF_PY:?}:/gguf-py:ro -e PYTHONPATH=/gguf-py" \
                "nice -n 19 python check_export.py case $D/case/G9 --general --split 6,6,5,5,6,5,5,6,5 --windows 6,6,6,7,1,2,6,6" ;;
        coverage)
            ssh "$H" "cd $S && nice -n 19 python3 -I shortlist_coverage.py ${VOCAB:?} ${SESSIONS:?} --ids ${IDS:?}" ;;
        tokens)
            mounts="-v ${FLINT_HF:?}:${FLINT_HF}:ro"
            for b in ${BODIES:?}; do mounts="$mounts -v $(dirname "$b"):$(dirname "$b"):ro"; done
            dk "$TOKENIZER_IMAGE" "$mounts" "python3 agent_tokens.py $BODIES --model $FLINT_HF --out $D/sim/agent-mtp.jsonl" ;;
        cost)
            ssh "$H" "cd $S && nice -n 19 python3 -I cost_sim.py $D/sim/agent-mtp.jsonl --head-ms 0,0.5,1,1.5,2,2.5,3,4,5.3 \
                --accept 0.1,0.15,0.2,0.25,0.3,0.4 --jobs 8 --out $D/sim/cost-agent.json"
            ssh "$H" "cd $S && nice -n 19 python3 -I cost_sim.py $D/sim/agent-mtp.jsonl --orders new --head-ms 0,2,3,4,5.3,8 \
                --accept 0.1,0.15,0.2,0.3 --head-card -1 --tap-ms 0.5 --jobs 8 --out $D/sim/cost-agent-spare.json" ;;
        split)
            mounts="-v $(dirname "${ROWS:?}"):$(dirname "$ROWS"):ro -v $(dirname "${SCORES:?}"):$(dirname "$SCORES"):ro"
            dk "$TORCH_IMAGE" "$mounts" "for w in g9 g9miss; do nice -n 19 python -I source_split.py $ROWS $SCORES --window \$w; done" ;;
        *) echo "unknown step $step"; exit 2 ;;
    esac
done
