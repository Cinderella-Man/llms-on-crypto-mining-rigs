#!/usr/bin/env bash
# card_log_test.sh: self-test of scripts/card_log.py (the read-only card log), of the drivers that embed or start it,
# of scripts/card_table.py and of scripts/nvml_watchdog.py's clocks. Sections (SECTIONS, default
# "sync static fake fast watchdog drivers"; about 90 s, most of it the drivers' own sleeps):
#   sync    the copies embedded in bench/agent_check/rig_server.sh, scripts/bench.sh, scripts/server_bench.sh and R43's
#           replay driver (docs/data/2026-11-04_prompt_taper/run/rig_taper.sh), and the copies beside
#           bench/pipespec/rig_*.sh and bench/release_matrix/run_matrix.sh, are byte for byte scripts/card_log.py, and
#           bench/deep_lane/remote.sh copies it beside deep_lane.sh
#   static  card_log.py names no NVML function that sets anything, and none that reads a card's PCI fields, UUID or
#           serial number; no driver that has the card log still starts an nvidia-smi poll
#   fake    card_log.py against a stand-in for NVML written in Python (no card and no NVIDIA library is touched): the
#           columns, the start and stop lines, every NVML call it makes is a Get from the list below, it stops on
#           --duration, on SIGTERM at once, on the stop file and when its --parent ends; then card_table.py on that log
#           finds the slow stretch the stand-in put in and none on the other card
#   fast    --fast, --slow-every and --profile (C942) on scripts/card_log_standin.py's stand-in: --fast makes the same
#           NVML calls in the same order and the same rows as today's sample, over 40 samples of 3 cards with a slow
#           stretch and a memory offset set halfway; --slow-every 4 reads the memory clock, temperature and memory used
#           on every 4th sample and at each P-state change and nothing else differs; from the command line the
#           environment variables turn each on (and only they), the first row reaches the file at once, SIGTERM keeps
#           every row, the profile line comes just before the stop line, and a bad CARD_LOG_SLOW_EVERY stops it
#   watchdog nvml_watchdog.py on the stand-in (scripts/card_log_standin.py's) with its wall clock set back 2 s halfway:
#           each row's time from one clock reading, mono_s rising through the step, a log started without mono_s
#           appended to without it; card_table.py orders that log by mono_s
#   drivers each driver that starts the card log, dry-run in a scratch folder with stub programs (no model, no GPU, no
#           nvidia-smi: a stub records its calls) and card_log.py on the stand-in: R43's replay driver (a normal run, a
#           run killed with SIGKILL, NVML missing, CARD_LOG=0), the four bench/pipespec/rig_*.sh from a copy of their
#           folder, and scripts/server_bench.sh (a normal run and a server that dies while loading): the log starts,
#           rows are written, it stops with the run and no logger is left (bench/release_matrix/test/dryrun.sh covers
#           run_matrix.sh)
#   real    card_log.py on this machine's own cards for REAL_S seconds (default 5): a row about every 0.5 s with a
#           value in every field. Only on a machine with NVIDIA cards that is not running a timed load.
#   bash scripts/card_log_test.sh                      # everything but real: no GPU needed
#   SECTIONS=real REAL_S=10 bash scripts/card_log_test.sh
#   KEEP=1 SECTIONS=drivers bash scripts/card_log_test.sh   # keeps the scratch folder (the dry runs' logs) and names it
# Prints one line per check and "card_log_test: N passed, M failed"; the exit status is 1 when anything failed.
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
L=$REPO/scripts/card_log.py
TAB=$REPO/scripts/card_table.py
SECTIONS=${SECTIONS:-"sync static fake fast watchdog drivers"}
STANDIN=$REPO/scripts/card_log_standin.py
REAL_S=${REAL_S:-5}
T=$(mktemp -d /tmp/card_log_test.XXXXXX)
if [ "${KEEP:-0}" = 1 ]; then trap 'echo "outputs kept in $T"' EXIT; else trap 'rm -rf "$T"' EXIT; fi
pass=0 fail=0
has() { [[ " $SECTIONS " == *" $1 "* ]]; }
ok() { pass=$((pass + 1)); echo "ok    $*"; }
bad() { fail=$((fail + 1)); echo "FAIL  $*"; }
check() { local d=$1; shift; if "$@"; then ok "$d"; else bad "$d"; fi; }

if has sync; then
  echo "== sync: the embedded copies"
  for f in bench/agent_check/rig_server.sh scripts/bench.sh scripts/server_bench.sh \
    docs/data/2026-11-04_prompt_taper/run/rig_taper.sh; do
    sed -n "/<< 'CARD_LOG'\$/,/^CARD_LOG\$/p" "$REPO/$f" | sed '1d;$d' > "$T/embedded.py"
    check "$f embeds scripts/card_log.py unchanged" cmp -s "$T/embedded.py" "$L"
  done
  for f in bench/pipespec/card_log.py bench/release_matrix/card_log.py; do
    check "$f is scripts/card_log.py unchanged" cmp -s "$REPO/$f" "$L"
  done
  check "bench/deep_lane/remote.sh copies scripts/card_log.py to the rig" \
    grep -q 'scripts/card_log.py' "$REPO/bench/deep_lane/remote.sh"
fi

if has static; then
  echo "== static: what card_log.py may call"
  check "no NVML function that sets anything" bash -c "! grep -nE 'nvml[A-Za-z_]*Set|nvml[A-Za-z_]*(Reset|Clear)' '$L'"
  check "no PCI, UUID or serial query" bash -c "! grep -nE 'nvml[A-Za-z_]*(Pci|PCI|Uuid|UUID|Serial|Pcie)' '$L'"
  for f in bench/agent_check/rig_server.sh scripts/bench.sh scripts/server_bench.sh bench/deep_lane/deep_lane.sh \
    docs/data/2026-11-04_prompt_taper/run/rig_taper.sh bench/release_matrix/run_matrix.sh bench/pipespec/rig_prefill.sh \
    bench/pipespec/rig_hostloop.sh bench/pipespec/rig_run.sh bench/pipespec/rig_streams.sh; do
    check "$f starts no nvidia-smi poll" bash -c "! grep -nE 'nvidia-smi.* -l(ms)? ' '$REPO/$f'"
  done
fi

if has fake; then
  echo "== fake: card_log.py on a stand-in for NVML, then card_table.py"
  cat > "$T/fake_nvml.py" << 'FAKE'
"""Runs card_log.main() with ctypes.CDLL replaced by a stand-in: 2 cards; card 0 busy at 1,911 MHz throughout, card 1
busy at 1,354 MHz from its 5th to its 16th sample (a slow stretch), at 1,911 otherwise. Records every call."""
import ctypes, importlib.util, json, os, signal, subprocess, sys, threading, time
sys.dont_write_bytecode = True
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
  ev=2 fast=0 prof=0  # the # lines: start and stop, plus the fast line and the profile line when they are asked for
  { [ "${CARD_LOG_FAST:-0}" != 0 ] || [ "${CARD_LOG_SLOW_EVERY:-1}" -gt 1 ] 2> /dev/null; } && fast=1 && ev=$((ev + 1))
  [ "${CARD_LOG_PROFILE:-0}" != 0 ] && prof=1 && ev=$((ev + 1))
  if [ $fast = 1 ]; then
    check "CARD_LOG_FAST or CARD_LOG_SLOW_EVERY is set: the log says the fast sample made it" grep -q "card_log fast: " "$log"
  fi
  if [ $prof = 1 ]; then
    check "CARD_LOG_PROFILE is set: the profile line before the stop line" bash -c "tail -n 2 '$log' | head -n 1 | grep -q 'card_log profile: '"
  fi
  check "$ev # lines: start$([ $fast = 1 ] && echo ', fast')$([ $prof = 1 ] && echo ', profile'), stop" test "$(grep -c '^#' "$log")" = $ev
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

if has fast; then
  echo "== fast: --fast, --slow-every and --profile against today's sample, on the stand-in"
  cat > "$T/fast_check.py" << 'FASTCHECK'
"""Today's sample (Nvml.sample) against --fast (FastSample) on card_log_standin.py's stand-in, 3 cards, 40 samples,
the memory offset set from sample 20 on: the NVML calls, in order, and every row. Then --slow-every 4 with a stand-in
whose temperature counts its reads and whose card 0 sleeps (P8) in samples 10 to 14. Args: card_log.py standin.py."""
import ctypes, importlib.util, os, sys
sys.dont_write_bytecode = True
clpath, stpath = sys.argv[1:3]
os.environ["STANDIN_CARDS"] = "3"
os.environ["STANDIN_OFFSET_FILE"] = off = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "fast.offset")
N, fails = 40, []
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
def check(what, ok, detail=""):
    print(f"  {'ok' if ok else 'FAIL'}: {what}{' (' + detail + ')' if detail and not ok else ''}")
    ok or fails.append(what)
class Recorder:  # the stand-in, with the order of the NVML calls kept
    def __init__(self, lib):
        self.lib, self.seq = lib, []
    def __getattr__(self, name):
        f = getattr(self.lib, name)
        def rec(*a):
            self.seq.append(name)
            return f(*a)
        return rec
def run(st, cl, mode, slow_every=1):
    st.samples.clear()
    if os.path.exists(off):
        os.remove(off)
    ctypes.CDLL = lambda *a, **k: st.Lib()
    nv = cl.Nvml()
    hs = [nv.handle(i) for i in range(nv.count())]
    nv.lib = rec = Recorder(nv.lib)
    fast = cl.FastSample(nv, hs, slow_every).row if mode == "fast" else None
    rows, per = [], []
    for k in range(N):
        if k == 20:
            open(off, "w").close()
        n0 = len(rec.seq)
        rows.append(fast() if fast else sum((nv.sample(h) for h in hs), []))
        per.append(rec.seq[n0:])
    os.remove(off)
    return rows, rec.seq, per
st, cl = load("standin", stpath), load("card_log", clpath)
r_old, s_old, _ = run(st, cl, "today")
r_new, s_new, _ = run(st, cl, "fast")
check("the same NVML calls in the same order (%d calls, 24 a sample)" % len(s_old), s_new == s_old and len(s_old) == 24 * N)
check("the same rows, field for field (%d rows of 24 fields)" % N, r_new == r_old and all(len(r) == 24 for r in r_old))
check("the rows show the stand-in's slow stretch and the memory offset",
      r_old[6][9] == "1354" and r_old[3][9] == "1911" and r_old[19][2] == "4006" and r_old[20][2] == "4304")
# --slow-every 4: temperature = how often it was read; card 0 in P8 for samples 10 to 14
reads = {}
def temp(self, h, which, ref):
    g = h.value - 1
    reads[g] = reads.get(g, 0) + 1
    ref._obj.value = reads[g]
    return 0
def pstate(self, h, ref):
    g = h.value - 1
    ref._obj.value = 8 if g == 0 and 10 <= st.samples.get(0, 0) <= 14 else 0  # SM clock counts samples; read after
    return 0
st.Lib.nvmlDeviceGetTemperature, st.Lib.nvmlDeviceGetPerformanceState = temp, pstate
r_slow, _, per = run(st, cl, "fast", 4)
reads.clear()
r_all, _, _ = run(st, cl, "fast", 1)
due0 = [k for k in range(N) if k % 4 == 0 or k in (10, 15)]  # card 0: every 4th sample and its two P-state changes
due1 = [k for k in range(N) if k % 4 == 0]
want = [[sum(1 for d in due if d <= k) for due in (due0, due1, due1)] for k in range(N)]
got = [[int(r[8 * g + 4]) for g in range(3)] for r in r_slow]
check("temperature read on every 4th sample and at card 0's P-state changes (samples 10 and 15)", got == want,
      f"got {got[:17]}")
check("P-state, SM clock, power, utilization and reasons as with every read, on every sample",
      all([r[8 * g + j] for g in range(3) for j in (0, 1, 3, 6, 7)] == [a[8 * g + j] for g in range(3) for j in (0, 1, 3, 6, 7)]
          for r, a in zip(r_slow, r_all)))
check("card 0 is P8 in samples 10 to 14 only", [r[0] for r in r_slow[9:16]] == ["P0"] + ["P8"] * 5 + ["P0"])
check("memory clock and memory used: read on the due samples, repeated in between (the offset seen at sample 20)",
      all(r_slow[k][8 * g + j] == r_all[max(d for d in (due0 if g == 0 else due1) if d <= k)][8 * g + j]
          for k in range(N) for g in range(3) for j in (2, 5)) and r_slow[20][2] == "4304" and r_slow[19][2] == "4006")
calls = [len(p) for p in per]
want_calls = [sum(5 + 3 * (k in (due0 if g == 0 else due1)) for g in range(3)) for k in range(N)]
check("NVML calls a sample: 5 a card, 8 on a due sample (%d in %d samples, against %d)" % (sum(calls), N, 24 * N),
      calls == want_calls, f"{calls}")
sys.exit(1 if fails else 0)
FASTCHECK
  python3 -I "$T/fast_check.py" "$L" "$STANDIN" > "$T/fast_check.out" 2>&1
  rc=$?
  sed 's/^/ /' "$T/fast_check.out"
  check "today's sample and --fast, call for call and row for row; --slow-every 4's schedule" test "$rc" = 0
  # the command line, on the stand-in the drivers' dry runs use
  S=(python3 -I "$STANDIN" "$L" --interval 0.05)
  env -u CARD_LOG_FAST -u CARD_LOG_SLOW_EVERY -u CARD_LOG_PROFILE "${S[@]}" --log "$T/c_def.csv" --duration 1.2 &
  env -u CARD_LOG_SLOW_EVERY -u CARD_LOG_PROFILE CARD_LOG_FAST=1 "${S[@]}" --log "$T/c_fast.csv" --duration 1.2 &
  env -u CARD_LOG_FAST -u CARD_LOG_SLOW_EVERY CARD_LOG_PROFILE=1 "${S[@]}" --log "$T/c_prof.csv" --duration 1.2 &
  env -u CARD_LOG_FAST -u CARD_LOG_PROFILE CARD_LOG_SLOW_EVERY=4 "${S[@]}" --log "$T/c_slow.csv" --duration 1.2 &
  env -u CARD_LOG_SLOW_EVERY -u CARD_LOG_PROFILE CARD_LOG_FAST=0 "${S[@]}" --log "$T/c_zero.csv" --duration 0.3 &
  wait
  same_rows() {  # the header and the first 19 rows of two logs, apart from epoch, time and took_ms
    [ "$(grep -vc '^#' "$2")" -ge 21 ] && cmp -s <(grep -v '^#' "$1" | cut -d, -f4- | head -n 20) \
      <(grep -v '^#' "$2" | cut -d, -f4- | head -n 20)
  }
  check "CARD_LOG_FAST=1: the header and first 19 rows as today's, apart from epoch, time and took_ms" \
    same_rows "$T/c_def.csv" "$T/c_fast.csv"
  check "CARD_LOG_FAST=1: start line, then the fast line, then rows; the stop line last" bash -c \
    "grep '^#' '$T/c_fast.csv' | sed -n 2p | grep -q 'card_log fast: .* reads every field on every sample\$' && grep -n . '$T/c_fast.csv' | sed -n 2p | grep -q 'card_log start' && tail -n 1 '$T/c_fast.csv' | grep -q 'card_log stop (after 1.2 s): [0-9]* samples'"
  check "today's log (no option set, and CARD_LOG_FAST=0): no fast line, no profile line" bash -c \
    "[ \$(grep -c '^#' '$T/c_def.csv') = 2 ] && [ \$(grep -c '^#' '$T/c_zero.csv') = 2 ]"
  check "CARD_LOG_SLOW_EVERY=4 alone turns on the fast sample and says what it reads" \
    grep -q "card_log fast: .* reads memory clock, temperature and memory used on one sample in 4 and at a P-state change" "$T/c_slow.csv"
  check "CARD_LOG_PROFILE=1: one profile line just before the stop line, with each of the 7 sampled functions" bash -c \
    "[ \$(grep -c 'card_log profile: ' '$T/c_prof.csv') = 1 ] && tail -n 2 '$T/c_prof.csv' | head -n 1 | grep -q 'card_log profile: [0-9]* samples, CPU [0-9.]* ms a sample (user [0-9.]*, system [0-9.]*), in NVML calls [0-9.]* ms a sample' && for f in PerformanceState ClockInfo PowerUsage Temperature MemoryInfo UtilizationRates CurrentClocksEventReasons; do tail -n 2 '$T/c_prof.csv' | head -n 1 | grep -q \"nvmlDeviceGet\$f [0-9.]* a sample, [0-9.]* ms a call\" || exit 1; done"
  prof_rows() { grep -q 'nvmlDeviceGetClockInfo 4 a sample' "$T/c_prof.csv" && same_rows "$T/c_def.csv" "$T/c_prof.csv"; }
  check "CARD_LOG_PROFILE=1: the clock calls 4 a sample (2 cards, SM and memory) and rows as today's" prof_rows
  check "CARD_LOG_SLOW_EVERY=4: rows as today's on the stand-in (its memory clock, temperature and memory stay put)" \
    same_rows "$T/c_def.csv" "$T/c_slow.csv"
  # the first row goes to the file at once (the drivers wait for it), the rest at the stop; SIGTERM keeps every row
  env CARD_LOG_FAST=1 "${S[@]}" --log "$T/c_buf.csv" --duration 3 & pb=$!
  sleep 0.6
  first=$(grep -sc '^[0-9]' "$T/c_buf.csv")
  first=${first:-0}
  wait $pb
  rows=$(grep -c '^[0-9]' "$T/c_buf.csv")
  check "CARD_LOG_FAST=1: the first row in the file at once ($first rows at 0.6 s, before the 5 s write), all at the stop ($rows)" \
    test "$first" -ge 1 -a "$first" -lt 11 -a "$rows" -ge 50
  env CARD_LOG_FAST=1 "${S[@]}" --log "$T/c_term.csv" & pt=$!
  sleep 1.2
  kill -TERM $pt
  wait $pt
  rc=$?
  n=$(grep -c '^[0-9]' "$T/c_term.csv") s=$(sed -n 's/.*card_log stop (signal): \([0-9]*\) samples.*/\1/p' "$T/c_term.csv")
  check "CARD_LOG_FAST=1 and SIGTERM: status $rc, every sampled row written ($n rows, stop line says ${s:-none})" \
    test "$rc" = 0 -a -n "$s" -a "$n" = "$s" -a "$n" -ge 15
  for v in x 0 -2; do
    env CARD_LOG_SLOW_EVERY=$v "${S[@]}" --log "$T/c_bad.csv" --duration 0.2 > "$T/c_bad.out" 2>&1
    rc=$?
    check "CARD_LOG_SLOW_EVERY=$v: status 2 and a message, no log ($(tail -n 1 "$T/c_bad.out" | head -c 100))" \
      bash -c "[ $rc = 2 ] && grep -q 'slow-every' '$T/c_bad.out' && [ ! -e '$T/c_bad.csv' ]"
  done
fi

if has watchdog; then
  echo "== watchdog: nvml_watchdog.py's clocks on the stand-in, then card_table.py"
  W=$REPO/scripts/nvml_watchdog.py
  cat > "$T/wd_run.py" << 'WDRUN'
"""Runs nvml_watchdog.main() on card_log_standin.py's stand-in for NVML (as root, with no kernel log), its wall clock
set back STEP_S seconds once RUN_S / 2 has passed. Args: standin.py nvml_watchdog.py log marker run_s step_s."""
import ctypes, importlib.util, os, sys, time, types
sys.dont_write_bytecode = True
standin, wdpath, log, marker, run_s, step_s = sys.argv[1:7]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
st = load("standin", standin)
ctypes.CDLL = lambda *a, **k: st.Lib()
wd = load("nvml_watchdog", wdpath)
class Kmsg:
    def new_lines(self):
        return []
wd.Kmsg = Kmsg
os.geteuid = lambda: 0
real, start = time.time, time.time()
clock = types.SimpleNamespace(**{k: getattr(time, k) for k in dir(time) if not k.startswith("_")})
clock.time = lambda: real() - (float(step_s) if real() - start > float(run_s) / 2 else 0.0)
wd.time = clock
sys.argv = ["nvml_watchdog.py", "--gpu", "0", "--log", log, "--marker", marker, "--interval", "0.1", "--duration", run_s]
sys.exit(wd.main())
WDRUN
  python3 -I "$T/wd_run.py" "$STANDIN" "$W" "$T/wd.csv" "$T/wd.TRIPPED" 3 2 > "$T/wd.out" 2>&1
  check "the watchdog on the stand-in ends with status 0 ($(tr '\n' ' ' < "$T/wd.out"))" test -s "$T/wd.csv" -a ! -s "$T/wd.out"
  check "its header ends with mono_s" bash -c "grep -v '^#' '$T/wd.csv' | head -n 1 | grep -q ',g1_reasons,mono_s\$'"
  check "mono_s rises in every row" bash -c "grep -v '^#' '$T/wd.csv' | awk -F, 'NR > 1 { if (NR > 2 && \$NF <= m) bad = 1; m = \$NF; n++ } END { exit (bad || n < 20) }'"
  check "the wall clock steps back once, by 2 s, and mono_s does not" python3 -I -c "
import csv, sys, time
rows = [r for r in csv.DictReader(l for l in open('$T/wd.csv') if not l.startswith('#'))]
wall = [time.mktime(time.strptime(r['time'][:19], '%Y-%m-%d %H:%M:%S')) + int(r['time'][20:23]) / 1000 for r in rows]
mono = [float(r['mono_s']) for r in rows]
back = [(wall[i] - wall[i - 1]) - (mono[i] - mono[i - 1]) for i in range(1, len(rows)) if wall[i] < wall[i - 1]]
off = [wall[i] - mono[i] for i in range(len(rows))]
same = [o for o in off if abs(o - off[0]) < 0.05] + [o for o in off if abs(o - off[0] + 2) < 0.05]
print('  steps back:', ['%.3f' % b for b in back], 'rows:', len(rows))
sys.exit(not (len(back) == 1 and abs(back[0] + 2) < 0.05 and len(same) == len(rows)))"
  check "each time stamp comes from one clock reading" python3 -I -c "
import importlib.util, sys, time
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('w', '$W'); w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
t = 1795240799.9996  # the last millisecond of a second: a second reading would land in the next one
sys.exit(w.now(t) != time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t)) + '.999')"
  python3 -I "$TAB" "$T/wd.csv" --tsv "$T/wd.tsv" > "$T/wd.md" 2> "$T/wd.err"
  check "card_table.py orders it by mono_s: card 0 at P0 for the whole 3 s ($(awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i } NR == 2 { print $c["P0_s"] }' "$T/wd.tsv") s)" \
    bash -c "awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) c[\$i] = i } NR == 2 { exit !(\$c[\"P0_s\"] >= 2.7 && \$c[\"P0_s\"] <= 3.3) }' '$T/wd.tsv'"
  grep -v '^#' "$T/wd.csv" | head -n 1 | sed 's/,mono_s$//' > "$T/wd_old.csv"
  grep -v '^#' "$T/wd.csv" | sed -n 2p | sed 's/,[^,]*$//' >> "$T/wd_old.csv"
  oldn=$(head -n 1 "$T/wd_old.csv" | awk -F, '{ print NF }')
  python3 -I "$T/wd_run.py" "$STANDIN" "$W" "$T/wd_old.csv" "$T/wd.TRIPPED" 0.5 0 > "$T/wd_old.out" 2>&1
  check "a log started without mono_s is appended to without it ($oldn fields, one header)" bash -c \
    "[ \$(grep -vc '^#' '$T/wd_old.csv') -gt 3 ] && grep -v '^#' '$T/wd_old.csv' | awk -F, -v n=$oldn 'NF != n { exit 1 }' && ! grep -q mono_s '$T/wd_old.csv'"
fi

if has drivers; then
  echo "== drivers: dry runs with stub programs and card_log.py on the stand-in"
  D=$T/drv
  mkdir -p "$D/stub" "$D/work" "$D/builds/b/bin" "$D/sb/bin"
  python3 -I "$STANDIN" --shim "$D/stub"
  cat > "$D/stub/nvidia-smi" << 'STUB'
#!/usr/bin/env bash
echo "$*" >> "$STUB_LOG/smi_calls.log"
case " $* " in
  *query-compute-apps*) exit 0 ;;
  *memory.used*) printf '0, 4692 MiB, 6144 MiB\n1, 4692 MiB, 6144 MiB\n' ;;
  *) echo "stub nvidia-smi" ;;
esac
STUB
  printf '#!/usr/bin/env bash\necho "pkill $*" >> "$STUB_LOG/pkill_calls.log"\nexit 1\n' > "$D/stub/pkill"
  printf '#!/usr/bin/env bash\n[ "$1" = -f ] && shift 2\nexec "$@"\n' > "$D/stub/chrt"
  printf '#!/usr/bin/env bash\necho "stub llama-server $*"\nsleep "${STUB_UP_S:-1}"\nexit 1\n' > "$D/builds/b/bin/llama-server"
  printf '#!/usr/bin/env bash\necho "stub llama-pipespec $*"\nsleep 2\necho "PIPESPEC name=warm mode=plain tokens=1 tok_s=1"\n' \
    > "$D/builds/b/bin/llama-pipespec"
  cat > "$D/sb/server.py" << 'STUB'
import os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"{}")
time.sleep(float(os.environ.get("STUB_LOAD_S", "1")))
if os.environ.get("STUB_DIE") == "1":
    sys.exit("stub: died while loading")
HTTPServer(("127.0.0.1", port), H).serve_forever()
STUB
  printf '#!/usr/bin/env bash\nexec python3 %s "$@"\n' "$D/sb/server.py" > "$D/sb/bin/llama-server"
  chmod +x "$D"/stub/* "$D"/builds/b/bin/* "$D/sb/bin/llama-server"
  cp "$REPO/docs/data/2026-11-04_prompt_taper/run/rig_taper.sh" "$D/work/"  # a launcher copies this one script
  cp -r "$REPO/bench/pipespec" "$D/pipespec"                                 # as the README's scp -r
  rm -rf "$D/pipespec/out"
  cp "$REPO/scripts/server_bench.sh" "$D/sb/"
  E=(env PATH="$D/stub:$PATH" STUB_LOG="$D")
  logger_left() { pgrep -f "card_log_standi[n].py.*$1" > /dev/null; }
  # R43's replay driver: a normal run, NVML missing, CARD_LOG=0 (in parallel, about 15 s), then one killed with SIGKILL
  rt() { "${E[@]}" BUILDS="$D/builds" BUILD=b MODEL="$D/m.gguf" OUT="$D/work/$1" LOADS=A PORT=18795 "${@:2}" \
    bash "$D/work/rig_taper.sh" > "$D/work/$1.console" 2>&1; }
  rt rt_ok & p1=$!
  rt rt_fail STANDIN_FAIL=1 & p2=$!
  rt rt_off CARD_LOG=0 & p3=$!
  wait $p1 $p2 $p3
  o=$D/work/rt_ok
  check "rig_taper: run.log says the card log runs, before the load" bash -c \
    "grep -n 'card log: every card every 0.5 s to cards.csv' '$o/run.log' | grep -q . && [ \$(grep -n 'card log: every' '$o/run.log' | cut -d: -f1) -lt \$(grep -n 'load A:' '$o/run.log' | head -n 1 | cut -d: -f1) ]"
  rows=$(grep -c '^[0-9]' "$o/cards.csv" 2> /dev/null || echo 0)
  check "rig_taper: OUT/cards.csv has rows for the whole run ($rows rows)" test "$rows" -ge 15
  check "rig_taper: the log stopped with the run (SIGTERM stop line), DONE written, nothing left" bash -c \
    "grep -q 'card_log stop (signal): [0-9]* samples' '$o/cards.csv' && [ -e '$o/DONE' ] && [ ! -e '$o/card_log.py' ] && [ ! -e '$o/card_log.out' ] && ! pgrep -f 'card_log_standi[n].py.*$o/' > /dev/null"
  check "rig_taper: its loads ran as before (load lines, memory files, md5 of the driver)" bash -c \
    "grep -q 'load A: up after' '$o/run.log' && [ -s '$o/load-A/memory_loaded.txt' ] && grep -q 'rig_taper.sh' '$o/md5.txt' && tail -n 1 '$o/run.log' | grep -q ' done\$'"
  check "rig_taper: NVML missing is reported and the run goes on" bash -c \
    "grep -q 'card log: not running: card_log: no NVML' '$D/work/rt_fail/run.log' && [ -e '$D/work/rt_fail/DONE' ] && grep -q 'load A: up after' '$D/work/rt_fail/run.log'"
  check "rig_taper: CARD_LOG=0 starts no card log" bash -c \
    "[ ! -e '$D/work/rt_off/cards.csv' ] && ! grep -q 'card log' '$D/work/rt_off/run.log' && [ -e '$D/work/rt_off/DONE' ]"
  rt rt_kill STUB_UP_S=20 & pk=$!
  for _ in $(seq 40); do grep -qs '^[0-9]' "$D/work/rt_kill/cards.csv" && break; sleep 0.25; done
  pd=$(pgrep -o -f "bash $D/work/rig_taper.sh")
  echo "   (killing that run's driver with SIGKILL on purpose: the shell reports it as Killed)"
  kill -9 "$pd" 2> /dev/null
  wait $pk 2> /dev/null
  for _ in $(seq 12); do logger_left "$D/work/rt_kill/" || break; sleep 0.25; done
  check "rig_taper killed with SIGKILL: the log stops by itself (parent gone)" bash -c \
    "grep -q 'card_log stop (parent $pd is gone)' '$D/work/rt_kill/cards.csv' && ! pgrep -f 'card_log_standi[n].py.*$D/work/rt_kill/' > /dev/null"
  pkill -f "$D/builds/b/bin/llama-server" 2> /dev/null
  check "rig_taper: no nvidia-smi poll in any run" bash -c "! grep -qE -- '(^| )-l(ms)?( |\$)' '$D/smi_calls.log'"
  # the four pipespec drivers from a copy of their folder, in parallel (about 5 s), and rig_run.sh with GPU_LOG=0
  ps_run() { "${E[@]}" BIN="$D/builds/b/bin" MODEL="$D/m.gguf" TAG="$1" "${@:3}" bash "$D/pipespec/$2" \
    > "$D/pipespec/$1.console" 2>&1; }
  ps_run streams rig_streams.sh NS=1 & q1=$!
  ps_run run rig_run.sh DEPTHS=1 PROMPTS=chat & q2=$!
  ps_run prefill rig_prefill.sh CONFIGS=k2 & q3=$!
  ps_run hostloop rig_hostloop.sh NS=1 CONFIGS=u0 & q4=$!
  ps_run run_nolog rig_run.sh DEPTHS=1 PROMPTS=chat GPU_LOG=0 & q5=$!
  wait $q1 $q2 $q3 $q4 $q5
  for t in streams run prefill hostloop; do
    o=$D/pipespec/out/$t
    check "pipespec $t: the log started, rows, stopped with the run, nothing left, summary written" bash -c \
      "grep -q 'card log: every card every 0.5 s to cards.csv' '$D/pipespec/$t.console' && [ \$(grep -c '^[0-9]' '$o/cards.csv') -ge 3 ] && grep -q 'card_log stop (signal)' '$o/cards.csv' && [ ! -e '$o/card_log.out' ] && [ -e '$o/summary.tsv' ] && ! pgrep -f 'card_log_standi[n].py.*$o/' > /dev/null"
  done
  check "pipespec rig_run.sh with GPU_LOG=0: no card log" bash -c \
    "[ ! -e '$D/pipespec/out/run_nolog/cards.csv' ] && [ -e '$D/pipespec/out/run_nolog/summary.tsv' ]"
  # server_bench.sh (its OUT is /tmp/server_bench/<TAG>): a normal run with PROMPTS=none, then a server that dies
  for k in ok die; do
    tag=card_log_test_$$_$k
    rm -rf "/tmp/server_bench/$tag"
    "${E[@]}" BIN="$D/sb/bin" MODEL="$D/m.gguf" TAG="$tag" PROMPTS=none PORT=18796 MIN_AVAIL_MB=0 \
      $([ $k = die ] && echo STUB_DIE=1) bash "$D/sb/server_bench.sh" > "$D/sb/$k.console" 2>&1
    echo $? > "$D/sb/$k.rc"
    cp -r "/tmp/server_bench/$tag" "$D/sb/out_$k" 2> /dev/null
    rm -rf "/tmp/server_bench/$tag"
    pkill -f "$D/sb/server.py" 2> /dev/null
  done
  o=$D/sb/out_ok
  check "server_bench: the log started before the load, rows, stopped at the end, nothing left (exit $(cat "$D/sb/ok.rc"))" bash -c \
    "[ \$(cat '$D/sb/ok.rc') = 0 ] && head -n 1 '$D/sb/ok.console' | grep -q 'card log: every card every 0.5 s to cards.csv' && grep -q '^load time' '$o/summary.txt' && [ \$(grep -c '^[0-9]' '$o/cards.csv') -ge 2 ] && grep -q 'card_log stop (signal)' '$o/cards.csv' && [ ! -e '$o/card_log.py' ] && ! pgrep -f 'card_log_standi[n].py.*card_log_test_$$' > /dev/null"
  check "server_bench: a server that dies while loading still ends the log (exit $(cat "$D/sb/die.rc"))" bash -c \
    "[ \$(cat '$D/sb/die.rc') = 1 ] && grep -q 'server exited during load' '$D/sb/die.console' && grep -q 'card_log stop (signal)' '$D/sb/out_die/cards.csv'"
  check "drivers: no nvidia-smi poll, no logger left" bash -c \
    "! grep -qE -- '(^| )-l(ms)?( |\$)' '$D/smi_calls.log' && ! pgrep -f 'card_log_standi[n].py.*$T/' > /dev/null"
  if [ "${CARD_LOG_FAST:-0}" != 0 ] || [ "${CARD_LOG_SLOW_EVERY:-1}" -gt 1 ] 2> /dev/null; then
    check "drivers: CARD_LOG_FAST or CARD_LOG_SLOW_EVERY reached the logger through each of the 6 drivers' runs" bash -c \
      "for f in '$D/work/rt_ok/cards.csv' '$D'/pipespec/out/{streams,run,prefill,hostloop}/cards.csv '$D/sb/out_ok/cards.csv'; do grep -q 'card_log fast: ' \"\$f\" || { echo \"  no fast line: \$f\"; exit 1; }; done"
  else
    check "drivers: no CARD_LOG_FAST, so no fast line in any driver's log" bash -c \
      "! grep -qs 'card_log fast: ' '$D/work/rt_ok/cards.csv' '$D'/pipespec/out/*/cards.csv '$D/sb/out_ok/cards.csv'"
  fi
  [ "$fail" = 0 ] || { echo "-- consoles:"; tail -n 4 "$D"/work/*.console "$D"/work/*/run.log "$D"/pipespec/*.console "$D"/sb/*.console 2> /dev/null; }
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
