#!/usr/bin/env bash
# Q168 on the rig: Flint p065 (docs/data/2026-10-27_flint_near_q8, md5 8387b2e585a7b9cc10f40cb6b4531694) against
# Unsloth's Q8_0 on Miner Llama v2.0's release line, nine cards, 262,144 tokens of context, through
# bench/agent_check/release_check.sh. Run from the repository root with nothing else on the rig.
#   replay  six recorded agent-check traces (the answer store's control arm, as in
#           docs/data/2026-10-27_answer_store/run/rig_replay.sh) replayed to Q8_0 and then to p065: same prompts, seeds and
#           reply lengths (at most), so the ratio of ms per generated token is the file's speed, with a 95 % interval
#           over traces (replay.md, reference q8)
#   live    the agent check itself on p065 (12 bug-fix tasks with real tool runs): tasks solved and tok/s, to set beside
#           Q8_0's earlier runs on the same build
# Env: ENV_FILE (the machine values, as in bench/agent_check/release.env.example, with REL_BIN = the v2.0 build),
# MODEL_Q8 and MODEL_P065 (the two files on the rig), REPLAY_LOG (the control arm's proxy.jsonl on the client),
# CLIENT_ROOT (results folder on the client), OUT, PARTS (default "replay live").
set -uo pipefail
: "${ENV_FILE:?}" "${MODEL_Q8:?}" "${MODEL_P065:?}" "${REPLAY_LOG:?}" "${CLIENT_ROOT:?}"
O=${OUT:-rig}
REL="-b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4"
set -a; . "$ENV_FILE"; set +a
# MODEL_P065.md5 is the md5 of the bytes as they were written on the rig (run.sh relay)
[ "$(ssh "$RIG" "cut -d' ' -f1 $MODEL_P065.md5 && stat -c %s $MODEL_P065" | tr '\n' ' ')" = \
  "8387b2e585a7b9cc10f40cb6b4531694 40431236544 " ] || { echo "p065 on the rig is missing or changed"; exit 1; }
steps='prep run report collect'
for part in ${PARTS:-replay live}; do
    case $part in
    replay)
        export REPLAY_LOG REPLAY_REF=q8
        export REPLAY_TRACES=rel/bugfix-javascript-space-age/a1/t1,rel/bugfix-javascript-phone-number/a1/t1,rel/bugfix-javascript-complex-numbers/a1/t1,rel/bugfix-python-affine-cipher/a1/t1,rel/bugfix-python-list-ops/a1/t1,rel/bugfix-python-scale-generator/a1/t1
        for arm in q8 p065; do
            m=$MODEL_Q8; [ "$arm" = p065 ] && m=$MODEL_P065
            echo "=== replay $arm: MODEL=$m $(date +%T)"
            MODEL=$m REPLAY_LABEL=$arm ARMS=rel CTX=262144 REL_ARGS="$REL" CLIENT_OUT=$CLIENT_ROOT/out-near-q8-replay \
                STEPS=$steps bench/agent_check/release_check.sh "$O/replay" || echo "=== replay $arm failed ($?)"
            steps='run report collect'
        done
        unset REPLAY_LOG REPLAY_REF REPLAY_TRACES
        ;;
    live)
        echo "=== live p065 $(date +%T)"
        MODEL=$MODEL_P065 ARMS=rel CTX=262144 REL_ARGS="$REL" CLIENT_OUT=$CLIENT_ROOT/out-near-q8-live \
            STEPS=$steps bench/agent_check/release_check.sh "$O/live" || echo "=== live p065 failed ($?)"
        steps='run report collect'
        [ -r "$O/live/table.md" ] && grep -E "^\| rel " "$O/live/table.md" | sed "s/^/p065 /"
        ;;
    *) echo "unknown part $part" ;;
    esac
done
echo "=== done $(date +%T)"
