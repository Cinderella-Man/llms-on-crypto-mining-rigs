#!/usr/bin/env bash
# agent_check.sh OUT: Q550 served on the rig (the first step of Q551's speed ladder): the agent check of pipelined
# speculation against the same build without it. Each arm in ARMS (default "pipe0 pipe3", set by ARGS_<arm>) runs on
# its own through release_check_s12.sh (bench/agent_check/release_check.sh with named arms), after a quiet check of the
# rig (no llama, python, copy, checksum or apt process, the idle unattended-upgrades shutdown helper aside; no GPU
# process; nine cards at their idle few MiB), with
# mem_guard.sh watching the host's memory; then one report over all arms, the results copied to OUT, and the
# pipelined line's guess counts summed from the server log (guesses.csv).
#
# Runs on the machine that drives the check (it only runs ssh), with release_check_s12.sh's environment (RIG, CLIENT,
# ports, CLIENT_DIR, CLIENT_OUT, REL_BIN, MODEL, ARGS_<arm>, BUDGET_MIN_<arm>, ...; see that script) plus:
#   RC      release_check_s12.sh (default: docs/data/2026-11-03_flash_next_mtp/run/ of this repository)
#   GUARD   mem_guard.sh (default: next to RC)
#   GUARD_LOG  the guard's log on the rig (default /dev/shm/mem_guard.log: tmpfs, so writing it never waits on a disk)
#   STEPS   any of "prep arms report" (default all three; arms already run are skipped by the check's resume)
set -uo pipefail
OUT=$(realpath -m "${1:?usage: agent_check.sh OUT}")
HERE=$(cd "$(dirname "$0")" && pwd)
RC=${RC:-$HERE/../../2026-11-03_flash_next_mtp/run/release_check_s12.sh}
GUARD=${GUARD:-$(dirname "$RC")/mem_guard.sh}
GUARD_LOG=${GUARD_LOG:-/dev/shm/mem_guard.log}
ARMS=${ARMS:-pipe0 pipe3}
STEPS=${STEPS:-prep arms report}
: "${RIG:?}" "${CLIENT:?}" "${CLIENT_OUT:?}"
mkdir -p "$OUT"
has() { [[ " $STEPS " == *" $1 "* ]]; }
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/driver.log"; }

quiet() {  # quiet LABEL: appends the rig's state to quiet_check.txt, fails unless it is quiet
  timeout 60 ssh "$RIG" 'echo "=== $(date -Is) '"$1"'"; cat /proc/loadavg
    busy=$(ps -eo comm,args | grep -v "unattended-upgrade-shutdown --wait-for-signal" | awk "{print \$1}" \
      | grep -xE "llama-.*|python3?|cp|rsync|md5sum|sha256sum|dd|apt|apt-get|dpkg|unattended-upgr" | sort | uniq -c)
    [ -z "$busy" ] && echo "quiet: no llama, python, copy, checksum or apt processes" || { echo "BUSY: $busy"; q=1; }
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    [ -z "$apps" ] && echo "no GPU processes" || { echo "GPU processes: $apps"; q=1; }
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | sort | uniq -c
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1 > 16 {b = 1} END {exit b}" || q=1
    echo "MemAvailable $(awk "/^MemAvailable/ {print int(\$2 / 1024)}" /proc/meminfo) MiB"
    exit ${q:-0}' >> "$OUT/quiet_check.txt" 2>&1
}

guard_start() {
  ssh "$RIG" "cat > /dev/shm/mem_guard.sh" < "$GUARD" || return 1
  timeout 20 ssh "$RIG" "LOG=$GUARD_LOG setsid -f nohup chrt -f 10 bash /dev/shm/mem_guard.sh > /dev/null 2>&1 < /dev/null"
  sleep 3
  timeout 20 ssh "$RIG" "kill -0 \$(cat $GUARD_LOG.pid)" && say "mem_guard running on the rig ($GUARD_LOG)"
}

guard_stop() {  # the guard stopped, its log copied to OUT and removed on the rig
  timeout 20 ssh "$RIG" "kill \$(cat $GUARD_LOG.pid) 2> /dev/null"
  timeout 60 ssh "$RIG" "cat $GUARD_LOG" > "$OUT/mem_guard-$1.log" \
    && timeout 20 ssh "$RIG" "rm -f $GUARD_LOG $GUARD_LOG.pid /dev/shm/mem_guard.sh"
  grep "KILL" "$OUT/mem_guard-$1.log" | tee -a "$OUT/driver.log"
}

if has prep; then
  say "prep"
  STEPS=prep ARMS="$ARMS" bash "$RC" "$OUT" || { say "prep failed"; exit 1; }
fi

if has arms; then
  for arm in $ARMS; do
    if ssh "$CLIENT" "test -f $CLIENT_OUT/$arm.rc"; then say "arm $arm already run"; continue; fi
    quiet "before arm $arm (timed)" || { say "the rig is not quiet (quiet_check.txt); stopping"; exit 1; }
    say "arm $arm: the rig is quiet"
    guard_start || { say "mem_guard did not start; stopping"; exit 1; }
    STEPS=run ARMS=$arm bash "$RC" "$OUT"
    rc=$?
    guard_stop "$arm"
    [ "$rc" = 0 ] || { say "arm $arm failed (exit $rc); stopping"; exit 1; }
  done
fi

if has report; then
  say "report and collect"
  STEPS="report collect" ARMS="$ARMS" bash "$RC" "$OUT"
  # the pipelined line's per-request counts: "pipelined = G guesses, R right, W wrong, S from the store; ..."
  {
    echo "arm,requests,requests_pipelined,generated_tokens,guesses,right,wrong,right_share,right_per_generated_token"
    for arm in $ARMS; do
      f=$OUT/rig/server-$arm.log
      [ -f "$f" ] || continue
      awk -v arm="$arm" '
        / eval time = / && !/prompt eval/ { n++; for (i = 1; i <= NF; i++) if ($i == "tokens" && $(i - 1) ~ /^[0-9]+$/) { gen += $(i - 1); break } }
        /pipelined = / { p++; for (i = 1; i <= NF; i++) { if ($i == "guesses,") g += $(i - 1); if ($i == "right,") r += $(i - 1); if ($i == "wrong,") w += $(i - 1) } }
        END { printf "%s,%d,%d,%d,%d,%d,%d,%s,%s\n", arm, n, p, gen, g, r, w, g ? sprintf("%.3f", r / g) : "", gen ? sprintf("%.3f", r / gen) : "" }' "$f"
    done
  } > "$OUT/guesses.csv"
  cat "$OUT/guesses.csv" | tee -a "$OUT/driver.log"
fi
say "done"
