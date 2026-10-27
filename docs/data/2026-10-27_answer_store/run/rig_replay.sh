#!/usr/bin/env bash
# The answer store's paired test on the rig: six recorded traces of the control arm (agent/base; picked across its range
# of task times with `replay.py pick --task-set bugfix --n 6`) replayed to the release line with and without
# --pipe-store, through bench/agent_check/release_check.sh's replay mode. Same prompts, seeds and reply lengths, and
# pipelined decoding is exact, so both servers generate the same tokens: the ratio of ms per generated token is the
# store's effect alone, with a 95 % interval over traces. Run from the repository root with nothing else on the rig,
# after rig_test.sh (which put the store on the rig).
# Env: ENV_FILE (the machine values, as in bench/agent_check/release.env.example, with REL_BIN = the v2.0 build and
# MODEL = Flint Q8_0 on the rig), STORE_DIR (the store's folder on the rig), REPLAY_LOG (the control arm's proxy.jsonl
# on the client), CLIENT_ROOT (results folder on the client), OUT, ARMS_LIST (default "store base").
set -uo pipefail
: "${ENV_FILE:?}" "${STORE_DIR:?}" "${REPLAY_LOG:?}" "${CLIENT_ROOT:?}"
LIST=${ARMS_LIST:-store base}
O=${OUT:-replay}
F=$STORE_DIR/agentic-run1.pstore
REL="-b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4"
set -a; . "$ENV_FILE"; set +a
export REPLAY_LOG REPLAY_REF=base
export REPLAY_TRACES=rel/bugfix-javascript-space-age/a1/t1,rel/bugfix-javascript-phone-number/a1/t1,rel/bugfix-javascript-complex-numbers/a1/t1,rel/bugfix-python-affine-cipher/a1/t1,rel/bugfix-python-list-ops/a1/t1,rel/bugfix-python-scale-generator/a1/t1
[ "$(ssh "$RIG" "sha256sum $F" | cut -d' ' -f1)" = 86c5dedb2bc692c14999dc68c1a837624729ca982486286fceeab88a3ed8c13c ] \
  || { echo "the store on the rig is missing or changed"; exit 1; }
steps='prep run report collect'
for c in $LIST; do
    case $c in
        base)  REL_ARGS="$REL" ;;
        store) REL_ARGS="$REL --pipe-store $F" ;;
        *) echo "unknown arm $c"; continue ;;
    esac
    echo "=== $c: REL_ARGS='$REL_ARGS' $(date +%T)"
    REPLAY_LABEL=$c ARMS=rel CTX=262144 REL_ARGS="$REL_ARGS" CLIENT_OUT=$CLIENT_ROOT/out-store-replay \
        STEPS=$steps bench/agent_check/release_check.sh "$O" || echo "=== $c failed ($?)"
    steps='run report collect'
done
echo "=== done $(date +%T)"
