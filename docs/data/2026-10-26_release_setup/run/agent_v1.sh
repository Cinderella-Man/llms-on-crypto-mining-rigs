#!/usr/bin/env bash
# The v1 engine on the coding-agent check (2026-10-26): v1 = fork commit eeaa59660 (branch pascal-p106) with Unsloth
# Q8_0, run exactly like stock llama.cpp's row (agent_search.sh's stock): -c 65536 -b 2048 -ub 512
# -ts 6,6,5,5,6,5,5,6,6 and stock's 35-minute budget. v1 builds its attention mask on the host like stock. It goes
# through release_check.sh's "stock" arm with STOCK_BIN pointing at v1, so its tables label the arm "stock". Run from
# the repository root with nothing else on the rig. Env: ENV_FILE (as in agent_search.sh), Q8_MODEL, V1_BIN,
# CLIENT_ROOT, OUT. Result in OUT/v1-q8/.
set -uo pipefail
: "${ENV_FILE:?the machine values for release_check.sh}" "${CLIENT_ROOT:?results folder on the client}"
: "${Q8_MODEL:?Flint Q8_0 on the rig}"
O=${OUT:-agent}
mkdir -p "$O"
set -a; . "$ENV_FILE"; set +a
V1_BIN=${V1_BIN:-/opt/llama-builds/llama.cpp-20261023-eeaa59660-pascal-p106/bin/llama-server}
c=v1-q8
echo "=== $c: STOCK_BIN=$V1_BIN MODEL=$Q8_MODEL $(date +%T)"
MODEL=$Q8_MODEL STOCK_BIN=$V1_BIN ARMS=stock CTX=65536 STOCK_ARGS="-b 2048 -ub 512 -ts 6,6,5,5,6,5,5,6,6" \
  CLIENT_OUT=$CLIENT_ROOT/out-$c STEPS='prep run report collect' \
  bench/agent_check/release_check.sh "$O/$c" || echo "=== $c failed ($?)"
[ -r "$O/$c/table.md" ] && grep -E "^\| stock " "$O/$c/table.md" | sed "s/^/$c /"
[ -r "$O/$c/compare.md" ] && grep -E "^\| stock \| [0-9]+ / " "$O/$c/compare.md" | sed "s/^/$c speeds /"
echo "=== v1 run done $(date +%T)"
