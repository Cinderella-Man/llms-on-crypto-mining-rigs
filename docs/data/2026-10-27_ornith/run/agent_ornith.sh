#!/usr/bin/env bash
# Ornith-1.5-122B on the rig (2026-10-27): the agent check (bench/agent_check/release_check.sh: the pi agent fixing 12
# Exercism bugs, one attempt each, 12-minute cap per attempt) per candidate, nine cards, at Ornith's makers' sampling
# (T 0.6, top-p 0.95, top-k 20, min-p 0; the harness default is Flint's T 1.0). Every arm gets a 35-minute budget, so
# stock and Miner Llama compare like for like. Run from a machine that reaches the rig and the client over ssh, with
# nothing else on the rig; results in OUT/<candidate>/. CANDS picks the candidates below, in order.
#   RIG=rig CLIENT=<client ssh alias> CLIENT_WORK=<folder on the client> AD=<rig path of the AD GGUF> \
#   STOCK=<rig path of stock llama-server> V2=<rig path of Miner Llama v2.0 llama-server> OUT=<local folder> \
#   CANDS="stock v2 stock-mtp v2-mtp" bash agent_ornith.sh        (UD=<rig path of the UD GGUF> for ud-stock, ud-v2;
#   also V3=<rig path of a llama-server with the IQ2_S/IQ3_XXS Pascal kernels> for ud-v3; LOWBIT=<rig path of a
#   llama-server of fork branch pascal-p106-v3-lowbit>, VALID1=<rig path of the N2P/N3P file of
#   docs/data/2026-10-28_ornith_valid1> and VALID1_LINE=<its server line> for valid1)
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
CANDS=${CANDS:?candidates, e.g. "stock"}
AD=${AD:?rig path of Ornith-1.5-122B-AD-IQ3_XXS-IQ2_S.gguf}
UD=${UD:-}
STOCK=${STOCK:?rig path of stock llama-server (d77256999)}
V2=${V2:?rig path of Miner Llama v2.0 llama-server (37e717ff6)}
O=${OUT:?local results folder}
mkdir -p "$O"
# the harness's machine values (bench/agent_check/release_check.sh)
export RIG=${RIG:-rig} CLIENT=${CLIENT:?ssh alias of the client machine}
export RIG_PORT=${RIG_PORT:-8091} RELAY_PORT=${RELAY_PORT:-8098} CLIENT_PORT=${CLIENT_PORT:-18391} \
  PROXY_PORT=${PROXY_PORT:-18392}
CW=${CLIENT_WORK:?folder on the client}
export CLIENT_DIR=$CW/bench RIG_LOGDIR=/tmp/agent-check-rel TIME_CAP_MIN=12
SAMPLING='{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0}'
BUDGET_MIN_stock=35; BUDGET_MIN_rel=35
export SAMPLING BUDGET_MIN_stock BUDGET_MIN_rel
# the lines the diagnostics chose (docs/data/2026-10-27_ornith/fit_projection.csv, serve_probe.csv)
STOCK_CTX=${STOCK_CTX:-49152}; V2_CTX=${V2_CTX:-49152}
STOCK_LINE=${STOCK_LINE:--b 2048 -ub 512 -ts 5,6,5,6,5,5,6,5,7 -fit off}
STOCK_MTP_LINE=${STOCK_MTP_LINE:--b 2048 -ub 256 -ts 6,6,5,6,6,5,6,5,5 -fit off --spec-type draft-mtp --spec-draft-n-max 3}
V2_LINE=${V2_LINE:-$STOCK_LINE}; V2_MTP_LINE=${V2_MTP_LINE:-$STOCK_MTP_LINE}
for c in $CANDS; do
  # per candidate: M (model), CTX, ARMS (stock = STOCK_BIN with STOCK_ARGS, rel = REL_BIN with REL_ARGS), the args
  M=$AD; STOCK_BIN=$STOCK; REL_BIN=$V2; STOCK_ARGS=""; REL_ARGS=""
  case $c in
    stock)       ARMS=stock; CTX=$STOCK_CTX; STOCK_ARGS="$STOCK_LINE" ;;
    stock-mtp)   ARMS=stock; CTX=${STOCK_MTP_CTX:-$STOCK_CTX}; STOCK_ARGS="$STOCK_MTP_LINE" ;;
    v2)          ARMS=rel;   CTX=$V2_CTX; REL_ARGS="$V2_LINE" ;;
    v2-mtp)      ARMS=rel;   CTX=${V2_MTP_CTX:-$V2_CTX}; REL_ARGS="$V2_MTP_LINE" ;;
    ud-stock)    M=${UD:?}; ARMS=stock; CTX=${UD_STOCK_CTX:-$STOCK_CTX}; STOCK_ARGS="${UD_STOCK_LINE:-$STOCK_LINE}" ;;
    ud-v2)       M=${UD:?}; ARMS=rel;   CTX=${UD_V2_CTX:-$V2_CTX}; REL_ARGS="${UD_V2_LINE:-$V2_LINE}" ;;
    ud-v3)       M=${UD:?}; ARMS=rel;   CTX=${UD_V2_CTX:-$V2_CTX}; REL_BIN=${V3:?}; REL_ARGS="${UD_V2_LINE:-$V2_LINE}" ;;
    valid1)      M=${VALID1:?}; ARMS=rel; CTX=${VALID1_CTX:-$V2_CTX}; REL_BIN=${LOWBIT:?}; REL_ARGS="${VALID1_LINE:?}" ;;
    *) echo "unknown candidate $c"; continue ;;
  esac
  echo "=== $c: MODEL=$M ARMS=$ARMS CTX=$CTX STOCK_ARGS='$STOCK_ARGS' REL_ARGS='$REL_ARGS' $(date +%T)"
  MODEL=$M ARMS=$ARMS CTX=$CTX STOCK_BIN=$STOCK_BIN REL_BIN=$REL_BIN STOCK_ARGS="$STOCK_ARGS" REL_ARGS="$REL_ARGS" \
    CLIENT_OUT=$CW/out-ornith-$c STEPS=${STEPS:-prep run report collect} \
    "$REPO/bench/agent_check/release_check.sh" "$O/$c" || echo "=== $c failed ($?)"
  [ -r "$O/$c/table.md" ] && grep -E "^\| (rel|stock) " "$O/$c/table.md" | sed "s/^/$c /"
  [ -r "$O/$c/compare.md" ] && grep -E "^\| (rel|stock) \| [0-9]+ / " "$O/$c/compare.md" | sed "s/^/$c speeds /"
done
echo "=== agent runs done $(date +%T)"
