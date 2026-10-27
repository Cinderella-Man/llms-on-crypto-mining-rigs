#!/usr/bin/env bash
# The 240-minute MTP heads on the rig: the agent check (bench/agent_check/release_check.sh: a coding agent fixing 12
# small bugs with real tool runs) on the v2.0 engine (rc4 37e717ff6) with Flint Q8_0 on nine cards at 262k, the
# release line without and with each head. Run from the repository root with nothing else on the rig.
#   ARMS_LIST="check base nine gen" OUT=<results folder> HEADS_SRC=<quant machine folder with the GGUFs> \
#     RIG_HEADS=<folder on the rig> bash docs/data/2026-10-27_mtp_head/run/rig_test.sh
#   check  the cost of one guess on one P106 (llama-pipe-drafter-check --bench), a diagnostic
#   base   the release line: -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4 (release_check.sh's COMMON adds the rest)
#   nine   base + --pipe-guess staircase --pipe-draft flint-mtp-nine.gguf (the nine-card head)
#   gen    base + the 3-12-card head
# The agent check's machine values (RIG, CLIENT, ports, CLIENT_DIR, STOCK_BIN, REL_BIN, MODEL, budgets) come from an
# env file as in bench/agent_check/release.env.example: set -a; . release.env; set +a. QUANT is the ssh alias of the
# machine that holds the head files.
set -uo pipefail
: "${OUT:?}" "${HEADS_SRC:?}" "${RIG_HEADS:?}" "${RIG:?}" "${QUANT:?}" "${REL_BIN:?}" "${CLIENT_DIR:?}"
LIST=${ARMS_LIST:-check base nine gen}
mkdir -p "$OUT"
TS="-ts 6,6,5,5,6,5,5,6,6"
REL="-b 512 -ub 128 $TS --pipe-depth 4"
ssh "$RIG" "mkdir -p $RIG_HEADS"
for h in nine gen; do   # the side files, copied once and checked by SHA-256
    want=$(ssh "$QUANT" "sha256sum $HEADS_SRC/flint-mtp-$h.gguf" | cut -d' ' -f1)
    have=$(ssh "$RIG" "sha256sum $RIG_HEADS/flint-mtp-$h.gguf 2>/dev/null" | cut -d' ' -f1)
    [ "$want" = "$have" ] || ssh "$QUANT" "cat $HEADS_SRC/flint-mtp-$h.gguf" | ssh "$RIG" "cat > $RIG_HEADS/flint-mtp-$h.gguf"
    [ "$want" = "$(ssh "$RIG" "sha256sum $RIG_HEADS/flint-mtp-$h.gguf" | cut -d' ' -f1)" ] \
        || { echo "copy of $h failed"; exit 1; }
    echo "$h: $want on the rig"
done
for c in $LIST; do
    if [ "$c" = check ]; then
        for h in nine gen; do
            echo "=== cost of one guess, $h ($(date +%T))"
            ssh "$RIG" "$(dirname "$REL_BIN")/llama-pipe-drafter-check $RIG_HEADS/flint-mtp-$h.gguf --bench 300 \
                --split 6,6,5,5,6,5,5,6,5 --commits 1 --ctx 8192 2>&1 | tail -4" | tee "$OUT/check-$h.log"
        done
        continue
    fi
    case $c in
        base)     REL_ARGS="$REL" ;;
        nine|gen) REL_ARGS="$REL --pipe-guess staircase --pipe-draft $RIG_HEADS/flint-mtp-$c.gguf" ;;
        *) echo "unknown arm $c"; continue ;;
    esac
    echo "=== $c: REL_ARGS='$REL_ARGS' $(date +%T)"
    ARMS=rel CTX=262144 REL_ARGS="$REL_ARGS" CLIENT_OUT="$(dirname "$CLIENT_DIR")/out-mtp-$c" \
        STEPS='prep run report collect' bench/agent_check/release_check.sh "$OUT/$c" || echo "=== $c failed ($?)"
    [ -r "$OUT/$c/table.md" ] && grep -E "^\| rel " "$OUT/$c/table.md" | sed "s/^/$c /"
done
echo "=== done $(date +%T)"
