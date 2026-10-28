#!/usr/bin/env bash
# Q260 and Q263 on the rig: the agent check (bench/agent_check/release_check.sh: a coding agent fixing 12 small bugs
# with real tool runs) with Flint Q8_0 on nine cards at 262,144 tokens, the release line on the fix branch's build
# (pascal-p106-v3-promptpath: v2.0 plus three switches, all off by default). It has to be the live check: a replay
# sends the recorded replies back, never the server's own, so every replayed turn re-reads its reply whatever the
# server does. Run from the repository root with nothing else on the rig.
#   tr      LLAMA_SERVER_TEXT_REUSE=1: the cached ids wherever the prompt spells the same text (Q260)
#   trtail  tr + LLAMA_SERVER_TAIL_CKPT_MIN=4096: an end-of-prompt checkpoint only every 4,096 tokens (Q263)
#   base    the release line on the same build, no variables (a same-slot control, optional)
# The release line: -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4 (release_check.sh adds -c, -np 1, -bs,
# --ctx-checkpoints 4, --cache-ram 0 and the rest).
# Env: ENV_FILE (the machine values, as in bench/agent_check/release.env.example, with MODEL = Flint Q8_0 on the rig),
# BIN (the fix branch's llama-server on the rig), CLIENT_ROOT (results folder on the client), OUT, ARMS_LIST
# (default "tr trtail").
# Read the result from each arm's server log (rig/server-rel.log) with rereads.py: restores of the previous prompt's
# checkpoint, the "prompt ids differ from the cache ... same text" lines, prompt seconds per task and first-token times.
set -uo pipefail
: "${ENV_FILE:?}" "${BIN:?}" "${CLIENT_ROOT:?}"
LIST=${ARMS_LIST:-tr trtail}
O=${OUT:-agent}
mkdir -p "$O"
REL="-b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4"
set -a; . "$ENV_FILE"; set +a
for c in $LIST; do
    case $c in
        base)   ENVS="" ;;
        tr)     ENVS="LLAMA_SERVER_TEXT_REUSE=1" ;;
        trtail) ENVS="LLAMA_SERVER_TEXT_REUSE=1 LLAMA_SERVER_TAIL_CKPT_MIN=4096" ;;
        *) echo "unknown arm $c"; continue ;;
    esac
    echo "=== $c: REL_ENV='$ENVS' $(date +%T)"
    ARMS=rel CTX=262144 REL_BIN="$BIN" REL_ENV="$ENVS" REL_ARGS="$REL" CLIENT_OUT=$CLIENT_ROOT/out-rereads-$c \
        STEPS='prep run report collect' bench/agent_check/release_check.sh "$O/$c" || echo "=== $c failed ($?)"
    [ -r "$O/$c/table.md" ] && grep -E "^\| rel " "$O/$c/table.md" | sed "s/^/$c /"
    [ -r "$O/$c/rig/server-rel.log" ] && echo "$c: $(grep -c 'restored context checkpoint' "$O/$c/rig/server-rel.log") checkpoint restores," \
        "$(grep -c 'same text:' "$O/$c/rig/server-rel.log") requests kept cached ids past a token difference"
done
echo "=== done $(date +%T)"
