#!/usr/bin/env bash
# tables.sh: R142's tables (Q1446) from the agent check's client folders, on the client's CPU at nice 19. Run it in an
# empty folder with REPO (this repository's root, or a copy holding the four scripts below at their paths), R90 (R90's
# client results folder: docs/data/2026-11-11_r90_v4_vs_v3) and R72 (R72's, which R142's runs share: run/r142.sh's
# CLIENT_OUT). NEWARMS names R142's arms in R72's folder (v4c3; R142b adds its own arm). R72's res/turns.csv must hold
# every arm named here: the report step rewrites it with the arms it is given, so the last report over that folder names
# gains v4c2 v4c22 v4c2g and NEWARMS (README, "Commands"). It writes, as R72's tables:
#   merged/agent.tsv, merged/agent_11.tsv, merged/agent_nocx.tsv  each arm against R90's v3 (agent_tables.py): all 12
#                                                                 tasks, R90's 11, and the 12 without
#                                                                 bugfix-javascript-complex-numbers
#   prompt_buckets.tsv                                            prompt speed and first token by a turn's new prompt
#   task_acceptance.tsv                                           the draft's acceptance by task
#   host.tsv                                                      each new arm's lowest MemAvailable and swap, from the
#                                                                 rig's RAM watchdog (rig/ram_watchdog-<arm>.tsv)
set -euo pipefail
: "${REPO:?}" "${R90:?}" "${R72:?}"
NEWARMS=${NEWARMS:-v4c3}
D=$REPO/docs/data
A="r90v3 r90gains gains v4c2 v4c22 v4c2g $NEWARMS"
SRC="$R90:v3:r90v3 $R90:gains:r90gains $R72:gains:gains $R72:v4c2:v4c2 $R72:v4c22:v4c22 $R72:v4c2g:v4c2g"
for a in $NEWARMS; do SRC+=" $R72:$a:$a"; done
py() { nice -n 19 python3 -I "$@"; }
# shellcheck disable=SC2086
{
  py "$D/2026-11-12_r72_agent_v4c2/run/merge_named.py" merged $SRC > merge.log
  py "$D/2026-11-05_keepalive/run/agent_tables.py" merged $A > t12.log
  SKIP_TASKS=bugfix-javascript-variable-length-quantity OUT_TAG=_11 py "$D/2026-11-05_keepalive/run/agent_tables.py" merged $A > t11.log
  SKIP_TASKS=bugfix-javascript-complex-numbers OUT_TAG=_nocx py "$D/2026-11-05_keepalive/run/agent_tables.py" merged $A > tnocx.log
  py "$D/2026-11-11_r90_v4_vs_v3/run/prompt_buckets.py" merged/res/turns.csv $A > prompt_buckets.tsv
  SKIP_TASKS=bugfix-javascript-complex-numbers py "$D/2026-11-12_r72_agent_v4c2/run/task_acceptance.py" "$R72/res" \
    gains v4c2 v4c22 v4c2g $NEWARMS > task_acceptance.tsv
}
printf 'arm\tlowest_memavailable_mib\tat_unix_s\tswap_start_mib\tswap_max_mib\n' > host.tsv
for a in $NEWARMS; do
  nice -n 19 awk -F'\t' -v arm="$a" '!/^#/ && NF >= 5 && $2 ~ /^[0-9]/ { if (m == "" || $2 < m) { m = $2; t = $1 }
    if (s0 == "") s0 = $5; if ($5 > s) s = $5 } END { print arm "\t" m "\t" t "\t" s0 "\t" s }' \
    "$R72/rig/ram_watchdog-$a.tsv" >> host.tsv
done
