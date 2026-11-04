#!/usr/bin/env bash
# S33 (Q692, Q558): the MTP draft on the agent check again after the rig's kho=off boot (CMA 0).
#   1. a smoke of arm mtpk (S12's draft line without S15's workarounds) through smoke_dflash.sh: the agent's first-turn
#      request, then S12's smoke request, under rig_server.sh's RAM watchdog;
#   2. the timed arm mtpk if the smoke loaded and answered, else skipped;
#   3. the timed arm mtp2k (S15's mtp2 line);
#   4. the report of the earlier arms and these two on the client, and the client's folder collected to OUT.
# Run from the repository root. ENVFILE is the machine file with S12's values (run/s12.env.example of
# docs/data/2026-11-03_flash_next_mtp, filled in, with ARGS_mtp and ARGS_mtp2), plus s33.env.example's lines. OUT is a
# local folder. MIN_AVAIL_MB is the watchdog's MemAvailable floor (this run: 150, the agent check's old value).
set -uo pipefail
: "${ENVFILE:?}" "${OUT:?}"
set -a; . "$ENVFILE"; set +a
RC=docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh
SMOKE=docs/data/2026-11-03_flash_next_dflash2/run/smoke_dflash.sh
mkdir -p "$OUT"
ARMS_RUN=${ARMS_S33:-mtpk mtp2k}
if [ "${SKIP_SMOKE:-0}" != 1 ]; then
  REQUESTS=docs/data/2026-11-03_flash_next_mtp/run/smoke_request.json ARGS="$ARGS_mtpk" SMOKE_LOGDIR=$RIG_LOGDIR-smoke \
    bash "$SMOKE" "$OUT/smoke-mtpk" > "$OUT/smoke-mtpk.console" 2>&1
  rc=$?
  echo "smoke mtpk exit $rc"
  [ "$rc" = 0 ] || { echo "arm mtpk's smoke did not load or answer: its timed arm is skipped"; ARMS_RUN=mtp2k; }
fi
for a in $ARMS_RUN; do
  ARMS=$a STEPS=run bash "$RC" "$OUT/agent-$a" > "$OUT/agent-$a.console" 2>&1
  echo "arm $a: exit $?"
done
ARMS="nodraft mtp2 dflash2b mtpk mtp2k" STEPS="report collect" bash "$RC" "$OUT/agent-report" > "$OUT/agent-report.console" 2>&1
echo "report: exit $?"
