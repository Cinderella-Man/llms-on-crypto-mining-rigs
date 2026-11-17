#!/usr/bin/env bash
# card_log_test.sh: self-test of scripts/card_log.py (the read-only card log), of the drivers that embed or start it,
# and of scripts/card_table.py. Sections (SECTIONS, default "sync static fake"):
#   sync    the copies embedded in bench/agent_check/rig_server.sh and scripts/bench.sh are byte for byte
#           scripts/card_log.py, and bench/deep_lane/remote.sh copies it beside deep_lane.sh
#   static  card_log.py names no NVML function that sets anything, and none that reads a card's PCI fields, UUID or
#           serial number
#   fake    card_log.py against a stand-in for NVML written in Python (no card and no NVIDIA library is touched): the
#           columns, the start and stop lines, every NVML call it makes is a Get from the list below, it stops on
#           --duration, on SIGTERM at once, on the stop file and when its --parent ends; then card_table.py on that log
#           finds the slow stretch the stand-in put in and none on the other card
#   real    card_log.py on this machine's own cards for REAL_S seconds (default 5): a row about every 0.5 s with a
#           value in every field. Only on a machine with NVIDIA cards that is not running a timed load.
#   bash scripts/card_log_test.sh                      # sync, static, fake: no GPU needed
#   SECTIONS=real REAL_S=10 bash scripts/card_log_test.sh
# Prints one line per check and "card_log_test: N passed, M failed"; the exit status is 1 when anything failed.
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
L=$REPO/scripts/card_log.py
TAB=$REPO/scripts/card_table.py
SECTIONS=${SECTIONS:-"sync static fake"}
REAL_S=${REAL_S:-5}
T=$(mktemp -d /tmp/card_log_test.XXXXXX)
trap 'rm -rf "$T"' EXIT
pass=0 fail=0
has() { [[ " $SECTIONS " == *" $1 "* ]]; }
ok() { pass=$((pass + 1)); echo "ok    $*"; }
bad() { fail=$((fail + 1)); echo "FAIL  $*"; }
check() { local d=$1; shift; if "$@"; then ok "$d"; else bad "$d"; fi; }

if has sync; then
  echo "== sync: the embedded copies"
  for f in bench/agent_check/rig_server.sh scripts/bench.sh; do
    sed -n "/<< 'CARD_LOG'\$/,/^CARD_LOG\$/p" "$REPO/$f" | sed '1d;$d' > "$T/embedded.py"
    check "$f embeds scripts/card_log.py unchanged" cmp -s "$T/embedded.py" "$L"
  done
  check "bench/deep_lane/remote.sh copies scripts/card_log.py to the rig" \
    grep -q 'scripts/card_log.py' "$REPO/bench/deep_lane/remote.sh"
fi

if has static; then
  echo "== static: what card_log.py may call"
  check "no NVML function that sets anything" bash -c "! grep -nE 'nvml[A-Za-z_]*Set|nvml[A-Za-z_]*(Reset|Clear)' '$L'"
  check "no PCI, UUID or serial query" bash -c "! grep -nE 'nvml[A-Za-z_]*(Pci|PCI|Uuid|UUID|Serial|Pcie)' '$L'"
fi

if has fake; then
  echo "== fake: card_log.py on a stand-in for NVML, then card_table.py"
  cat > "$T/fake_nvml.py" << 'FAKE'
"""Runs card_log.main() with ctypes.CDLL replaced by a stand-in: 2 cards; card 0 busy at 1,911 MHz throughout, card 1
busy at 1,354 MHz from its 5th to its 16th sample (a slow stretch), at 1,911 otherwise. Records every call."""
import ctypes, importlib.util, json, os, signal, subprocess, sys, threading, time
mode, logpath, calls_out = sys.argv[1], sys.argv[2], sys.argv[3]
calls = set()
n = {0: 0, 1: 0}

class Lib:
    def __getattr__(self, name):
        raise AttributeError(name)  # a function the stand-in lacks: card_log.py must treat it as missing
    def _rec(self, name):
        calls.add(name)
    @staticmethod
    def nvmlErrorString(r): return b"stand-in error"  # a plain function: card_log.py sets its restype
    def nvmlInit_v2(self): self._rec("nvmlInit_v2"); return 0
    def nvmlShutdown(self): self._rec("nvmlShutdown"); return 0
    def nvmlDeviceGetCount_v2(self, ref): self._rec("nvmlDeviceGetCount_v2"); ref._obj.value = 2; return 0
    def nvmlDeviceGetHandleByIndex_v2(self, i, ref):
        self._rec("nvmlDeviceGetHandleByIndex_v2"); ref._obj.value = i.value + 1; return 0
    def nvmlSystemGetDriverVersion(self, buf, size):
        self._rec("nvmlSystemGetDriverVersion"); buf.value = b"580.178.04"; return 0
    def nvmlDeviceGetName(self, h, buf, size): self._rec("nvmlDeviceGetName"); buf.value = b"P106-090"; return 0
    def nvmlDeviceGetPerformanceState(self, h, ref):
        self._rec("nvmlDeviceGetPerformanceState"); ref._obj.value = 0; return 0
    def nvmlDeviceGetClockInfo(self, h, clk, ref):
        self._rec("nvmlDeviceGetClockInfo")
        g = h.value - 1
        if clk.value == 1:  # the SM clock is asked first: one sample of this card
            n[g] += 1
            ref._obj.value = 1354 if g == 1 and 5 <= n[g] <= 16 else 1911
        else:
            ref._obj.value = 4006
        return 0
    def nvmlDeviceGetPowerUsage(self, h, ref): self._rec("nvmlDeviceGetPowerUsage"); ref._obj.value = 31640; return 0
    def nvmlDeviceGetTemperature(self, h, which, ref): self._rec("nvmlDeviceGetTemperature"); ref._obj.value = 33; return 0
    def nvmlDeviceGetMemoryInfo(self, h, ref): self._rec("nvmlDeviceGetMemoryInfo"); ref._obj.used = 4692 << 20; return 0
    def nvmlDeviceGetUtilizationRates(self, h, ref): self._rec("nvmlDeviceGetUtilizationRates"); ref._obj.gpu = 37; return 0
    def nvmlDeviceGetCurrentClocksEventReasons(self, h, ref):
        self._rec("nvmlDeviceGetCurrentClocksEventReasons"); ref._obj.value = 0x4 if h.value == 2 else 0; return 0
    def nvmlDeviceGetEnforcedPowerLimit(self, h, ref): self._rec("nvmlDeviceGetEnforcedPowerLimit"); ref._obj.value = 75000; return 0
    def nvmlDeviceGetMemClkVfOffset(self, h, ref): self._rec("nvmlDeviceGetMemClkVfOffset"); ref._obj.value = 0; return 0
    def nvmlDeviceGetGpcClkVfOffset(self, h, ref): self._rec("nvmlDeviceGetGpcClkVfOffset"); ref._obj.value = 0; return 0

ctypes.CDLL = lambda *a, **k: Lib()
spec = importlib.util.spec_from_file_location("card_log", os.environ["CARD_LOG"])
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)
args = ["card_log.py", "--log", logpath, "--interval", "0.05"]
if mode == "duration":
    args += ["--duration", "1.2"]
elif mode == "signal":
    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
elif mode == "stopfile":
    args += ["--stop", logpath + ".stop"]
    threading.Timer(0.5, lambda: open(logpath + ".stop", "w").close()).start()
elif mode == "parent":
    p = subprocess.Popen(["sleep", "0.5"])
    threading.Thread(target=p.wait, daemon=True).start()  # reaped when it ends, as init reaps a gone driver
    args += ["--parent", str(p.pid)]
sys.argv = args
t0 = time.monotonic()
rc = cl.main()
json.dump({"rc": rc, "s": time.monotonic() - t0, "calls": sorted(calls)}, open(calls_out, "w"))
FAKE
  allowed="nvmlInit_v2 nvmlShutdown nvmlDeviceGetCount_v2 nvmlDeviceGetHandleByIndex_v2 nvmlSystemGetDriverVersion \
nvmlDeviceGetName nvmlDeviceGetPerformanceState nvmlDeviceGetClockInfo nvmlDeviceGetPowerUsage nvmlDeviceGetTemperature \
nvmlDeviceGetMemoryInfo nvmlDeviceGetUtilizationRates nvmlDeviceGetCurrentClocksEventReasons \
nvmlDeviceGetEnforcedPowerLimit nvmlDeviceGetMemClkVfOffset nvmlDeviceGetGpcClkVfOffset"
  for mode in duration signal stopfile parent; do
    log=$T/$mode.csv
    CARD_LOG=$L python3 -I "$T/fake_nvml.py" "$mode" "$log" "$T/$mode.json" > "$T/$mode.out" 2>&1
    check "$mode: the stand-in run ends with status 0" python3 -I -c "import json,sys; sys.exit(json.load(open('$T/$mode.json'))['rc'])"
    case $mode in
      duration) why='after 1.2 s' ;; signal) why=signal ;; stopfile) why='stop file' ;; parent) why='parent [0-9]* is gone' ;;
    esac
    check "$mode: the stop line names why ($why)" grep -q "card_log stop ($why): [0-9]* samples" "$log"
  done
  log=$T/duration.csv
  check "the header: epoch, time, took_ms and 8 fields for each of 2 cards" test "$(grep -v '^#' "$log" | head -n 1)" = \
    "epoch,time,took_ms,g0_pstate,g0_sm_mhz,g0_mem_mhz,g0_w,g0_c,g0_used_mib,g0_util,g0_reasons,g1_pstate,g1_sm_mhz,g1_mem_mhz,g1_w,g1_c,g1_used_mib,g1_util,g1_reasons"
  check "every row has 19 fields, none empty" bash -c "grep -v '^#' '$log' | tail -n +2 | awk -F, 'NF != 19 { exit 1 } { for (i = 1; i <= NF; i++) if (\$i == \"\") exit 1 }'"
  check "a sample row as the stand-in gave it" bash -c "grep -v '^#' '$log' | sed -n 2p | grep -qE ',P0,1911,4006,31.64,33,4692,37,0x0,P0,1911,4006,31.64,33,4692,37,0x4\$'"
  rows=$(grep -vc '^#' "$log")
  check "--duration 1.2 at 0.05 s: 20 to 27 rows plus the header ($((rows - 1)))" test "$rows" -ge 21 -a "$rows" -le 28
  check "the start line: driver, cards, limits and offsets" grep -q "card_log start: pid [0-9]*, driver 580.178.04, 2 cards (P106-090), interval 0.05 s; g0 limit 75 W, memory offset 0, core offset 0; g1 limit 75 W" "$log"
  check "the stop line: CPU seconds and sampling time" grep -qE "s CPU \(user\+sys\), sampling [0-9.]+ ms median, [0-9.]+ ms max, [0-9]+ samples late" "$log"
  check "SIGTERM stops it within 0.3 s of the signal" python3 -I -c "import json,sys; sys.exit(json.load(open('$T/signal.json'))['s'] > 0.8)"
  check "every NVML call is a Get from the list" python3 -I -c "
import json, sys
calls = set(json.load(open('$T/duration.json'))['calls']) | set(json.load(open('$T/signal.json'))['calls'])
extra = calls - set('$allowed'.split())
print('  calls:', len(calls), 'outside the list:', sorted(extra)) if extra else None
sys.exit(1 if extra else 0)"
  python3 -I "$TAB" "$log" --stretch-s 0.3 --tsv "$T/t.tsv" > "$T/t.md" 2> "$T/t.err"
  check "card_table.py reads the log" test -s "$T/t.md"
  check "card_table.py: busy by utilization" grep -q "^Busy: utilization above 0 %" "$T/t.md"
  check "card_table.py: card 1's slow stretch, 12 samples at 1,354 MHz" grep -qE "^\| 1 \| .* \| 0\.[5-7] \| 12 \| 1,354 \| 1,354 \| P0 \| power cap \|" "$T/t.md"
  check "card_table.py: no stretch on card 0" bash -c "! grep -qE '^\| 0 \| 20[0-9]{2}-' '$T/t.md'"
  check "card_table.py: card 1 has 12 busy samples below 1,800 MHz and power cap in every sample" \
    bash -c "awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) c[\$i] = i } NR == 3 { exit !(\$c[\"busy_below_1800\"] == 12 && \$c[\"reason_power_cap\"] == \$c[\"samples\"] && \$c[\"sm_busy_min\"] == 1354) }' '$T/t.tsv'"
  [ "$fail" = 0 ] || { echo "-- logs:"; tail -n 3 "$T"/*.out "$T/t.err" 2> /dev/null; }
fi

if has real; then
  echo "== real: card_log.py on this machine's cards for $REAL_S s"
  python3 -I "$L" --log "$T/real.csv" --duration "$REAL_S" > "$T/real.out" 2>&1
  rc=$?
  check "card_log.py ends with status 0 ($(tr '\n' ' ' < "$T/real.out"))" test "$rc" = 0
  rows=$(grep -vc '^#' "$T/real.csv" 2> /dev/null || echo 0)
  want=$(python3 -I -c "print(int($REAL_S / 0.5 * 0.9))")
  check "about two rows a second ($((rows - 1)) rows, want $want or more)" test "$((rows - 1))" -ge "$want"
  check "a value in every field" bash -c "grep -v '^#' '$T/real.csv' | tail -n +2 | awk -F, '{ for (i = 1; i <= NF; i++) if (\$i == \"\") exit 1 }'"
  grep '^#' "$T/real.csv"
  python3 -I "$TAB" "$T/real.csv" | head -n 8
fi

echo "card_log_test: $pass passed, $fail failed"
[ "$fail" = 0 ]
