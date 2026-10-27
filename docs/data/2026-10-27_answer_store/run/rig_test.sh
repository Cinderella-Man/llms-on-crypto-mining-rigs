#!/usr/bin/env bash
# The answer store on the rig: the agent check (bench/agent_check/release_check.sh: a coding agent fixing 12 small
# bugs with real tool runs) on Miner Llama v2.0 (37e717ff6) with Flint Q8_0 on nine cards at 262,144 tokens, the
# release line with --pipe-store, then the release line again as a control in the same slot. Run from the repository
# root with nothing else on the rig.
#   store  -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4 --pipe-store STORE_DIR/agentic-run1.pstore
#   base   the release line: -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4 (release_check.sh adds the rest)
# Env: ENV_FILE (the machine values, as in bench/agent_check/release.env.example, with REL_BIN = the v2.0 build and
# MODEL = Flint Q8_0 on the rig), STORE_SRC (the store on the client machine, build_stores.sh's WORK/full/agentic.pstore),
# STORE_DIR (a folder on the rig's internal disk), CLIENT_ROOT (results folder on the client), OUT, ARMS_LIST.
set -uo pipefail
: "${ENV_FILE:?}" "${STORE_SRC:?}" "${STORE_DIR:?}" "${CLIENT_ROOT:?}"
LIST=${ARMS_LIST:-store base}
O=${OUT:-agent}
F=$STORE_DIR/agentic-run1.pstore
mkdir -p "$O"
REL="-b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4"
set -a; . "$ENV_FILE"; set +a
ssh "$RIG" "mkdir -p $STORE_DIR"
want=$(ssh "$CLIENT" "sha256sum $STORE_SRC" | cut -d' ' -f1)   # the store, copied once and checked by SHA-256
have=$(ssh "$RIG" "sha256sum $F 2>/dev/null" | cut -d' ' -f1)
[ "$want" = "$have" ] || ssh "$CLIENT" "cat $STORE_SRC" | ssh "$RIG" "cat > $F"
[ "$want" = "$(ssh "$RIG" "sha256sum $F" | cut -d' ' -f1)" ] || { echo "copy of the store failed"; exit 1; }
echo "store: $want on the rig"
for c in $LIST; do
    case $c in
        base)  REL_ARGS="$REL" ;;
        store) REL_ARGS="$REL --pipe-store $F" ;;
        *) echo "unknown arm $c"; continue ;;
    esac
    echo "=== $c: REL_ARGS='$REL_ARGS' $(date +%T)"
    ARMS=rel CTX=262144 REL_ARGS="$REL_ARGS" CLIENT_OUT=$CLIENT_ROOT/out-store-$c \
        STEPS='prep run report collect' bench/agent_check/release_check.sh "$O/$c" || echo "=== $c failed ($?)"
    [ -r "$O/$c/table.md" ] && grep -E "^\| rel " "$O/$c/table.md" | sed "s/^/$c /"
done
echo "=== done $(date +%T)"
