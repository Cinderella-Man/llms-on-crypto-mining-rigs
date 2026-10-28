#!/usr/bin/env bash
# Q261 on the rig: the card-built attention mask for decode steps only (LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_MAX=16)
# against the host mask, as a paired replay: six recorded agent-check traces (the answer store's set, picked across
# the control run's range of task times) re-sent to the fix branch's build with and without the switch, through
# bench/agent_check/release_check.sh's replay mode. Same prompts, seeds and reply lengths, and the card mask is exact,
# so both servers generate the same tokens: the ratio of ms per generated token is the mask's effect alone, with a 95 %
# interval over traces; the fixed cost per request shows the one full upload of the cell copies after each prompt.
# Run from the repository root with nothing else on the rig.
#   base  the release line on the fix branch's build, no variables
#   mask  + LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_MAX=16
# Env: ENV_FILE (the machine values, as in bench/agent_check/release.env.example, with MODEL = Flint Q8_0 on the rig),
# BIN (the fix branch's llama-server on the rig), REPLAY_LOG (the recorded run's proxy.jsonl on the client),
# CLIENT_ROOT (results folder on the client), OUT, ARMS_LIST (default "base mask").
set -uo pipefail
: "${ENV_FILE:?}" "${BIN:?}" "${REPLAY_LOG:?}" "${CLIENT_ROOT:?}"
LIST=${ARMS_LIST:-base mask}
O=${OUT:-replay}
REL="-b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4"
set -a; . "$ENV_FILE"; set +a
export REPLAY_LOG REPLAY_REF=base
export REPLAY_TRACES=rel/bugfix-javascript-space-age/a1/t1,rel/bugfix-javascript-phone-number/a1/t1,rel/bugfix-javascript-complex-numbers/a1/t1,rel/bugfix-python-affine-cipher/a1/t1,rel/bugfix-python-list-ops/a1/t1,rel/bugfix-python-scale-generator/a1/t1
steps='prep run report collect'
for c in $LIST; do
    case $c in
        base) ENVS="" ;;
        mask) ENVS="LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_MAX=16" ;;
        *) echo "unknown arm $c"; continue ;;
    esac
    echo "=== $c: REL_ENV='$ENVS' $(date +%T)"
    REPLAY_LABEL=$c ARMS=rel CTX=262144 REL_BIN="$BIN" REL_ENV="$ENVS" REL_ARGS="$REL" \
        CLIENT_OUT=$CLIENT_ROOT/out-mask-replay STEPS=$steps bench/agent_check/release_check.sh "$O" \
        || echo "=== $c failed ($?)"
    steps='run report collect'
done
echo "=== done $(date +%T)"
