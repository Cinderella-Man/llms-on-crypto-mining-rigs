#!/usr/bin/env python3
"""Tests of the boot-time clock service (bench/mem_offset/boot/) on any Linux machine: no GPU, no NVML, no root, no
systemd; nothing is set or run on a card.

  python3 -I bench/mem_offset/test/test_boot.py [-v]      # about 3 minutes

The real boot/clocks_at_boot.py runs as the unit would run it (its own process, a stop signal, a notify socket for
READY=1), and it runs the real kit: climb.py's start-check, watchdog start and stop, run-apply, state and reset-all,
greedy_check.py and model_check.sh. They run against a stand-in rig in a temporary folder, as in test_soak.py: a
`llama-server` that answers /health and /completion (wrongly, or not at all, from a chosen memory offset on), an
`nvml_memoc.py` that keeps each card's offsets in a JSON file and logs every set, an `nvidia-smi` that lists the
cards, a `journalctl` with no Xid, and an `nvml_watchdog.py` that holds until its stop file and trips, trips one card
or ends when the test says so. climb.py runs through a three-line stand-in that only answers "root" to its root check.

- the profile applied: READY=1, "overclocked: yes" with the profile's md5, every card at the profile's values, the
  watchdog held; the watchdog's log kept under its size; the stop sets every card to 0, stops the watchdog and says
  "overclocked: no"; a second start in the same boot is refused (the step down), and the work folders rotate;
- a card whose check fails at the profile's offsets: every card to 0, no READY;
- refusals with no offset touched and no watchdog started: no profile, a config line that does not read, more cards
  than the profile, another PCI address, another VBIOS, another card set (GPU_SET_ID: an identical rig), another
  driver, an offset already set, a step down this boot, CARDS against the profile, no driver within DRIVER_WAIT, the
  model check without its stored hash;
- a hung check past APPLY_LIMIT: the command ended, every card to 0; a stop while apply's checks run: the same;
- held: a watchdog trip, and the watchdog's end, set every card to 0 and end the service; one card's trip leaves
  "overclocked: partly" with the other card still set, and the stop then resets both;
- the main process killed: stop-post sets every card to 0 and stops the watchdog;
- the optional model check: passes against a hash made at stock by model_check.sh, and a model that answers otherwise
  with the offsets set sends every card to 0;
- the unit file passes `systemd-analyze verify` (when the tool is there), and install.sh installs into a test root,
  keeps an existing config, copies mmvq_pascal when given, never enables the service and runs no systemctl there; it
  also installs from the kit tarball's layout (build.sh's own file list) and refuses to run from its installed copy;
- the pure helpers: the config reader, the card match, the card-set id, the kit's state lines.
"""
import hashlib
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
BOOT = os.path.join(KIT, "boot")
WRAPPER = os.path.join(BOOT, "clocks_at_boot.py")
sys.path.insert(0, BOOT)
import clocks_at_boot as cab  # noqa: E402

BUSES = ["03", "04", "05", "06", "07", "0B", "0C", "0D", "0E"]
VERBOSE = bool(os.environ.get("BOOT_TEST_CONSOLE"))   # print every service console

FAKE_SERVER = r'''#!/usr/bin/env python3
import http.server, json, os, sys, time
g = os.environ.get("CUDA_VISIBLE_DEVICES")          # unset: the served model on every card (model_check.sh)
port = int(sys.argv[sys.argv.index("--port") + 1])
fault = json.loads(os.environ.get("FAKE_FAULTS", "{}")).get(g) if g is not None else None
def offsets():
    with open(os.environ["FAKE_STATE"]) as f:
        return json.load(f)
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        self.send({"status": "ok"})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = "answer to " + req["prompt"][:24]
        if fault and offsets().get(g, [0, 0])[0] >= fault["mem_min"]:
            if fault.get("hang"):
                time.sleep(600)
            text += " (wrong)"
        if g is None and os.environ.get("FAKE_MODEL_WRONG") and any(v[0] or v[1] for v in offsets().values()):
            text += " (model wrong)"
        self.send({"content": text, "completion_probabilities": [{"token": "a", "logprob": -0.05}],
                   "timings": {"prompt_per_second": 50.0, "predicted_per_second": 30.0, "predicted_n": req["n_predict"]}})
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
'''

FAKE_MEMOC = r'''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
core = "--core" in args
for flag in ("--state", "--cap"):
    if flag in args:
        i = args.index(flag)
        del args[i:i + 2]
args = [a for a in args if a != "--core"]
cmd, g = args[0], args[1]
path = os.environ["FAKE_STATE"]
with open(path) as f:
    st = json.load(f)
v = st.setdefault(g, [0, 0])
d = 1 if core else 0
name = "GpcClk" if core else "MemClk"
if cmd == "get":
    print(f"gpu {g} {name} vf offset: {v[d]} (allowed (-2000, 2000))")
    sys.exit(0)
v[d] = int(args[2]) if cmd == "set" else 0
if cmd == "set":
    with open(os.environ["FAKE_CALLS"], "a") as f:
        f.write(f"set {g} {name} {v[d]}\n")
with open(path + ".tmp", "w") as f:
    json.dump(st, f)
os.replace(path + ".tmp", path)
print(f"gpu {g} {name}: " + (f"set {v[d]}" if cmd == "set" else "offset 0"))
'''

FAKE_SMI = r'''#!/bin/sh
[ -n "${FAKE_SMI_FAIL:-}" ] && { echo "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver"; exit 9; }
i=0
case "$*" in *uuid*)
  for bus in $FAKE_BUSES; do echo "$i, 00000000:$bus:00.0, GPU-$i$bus-stand-in-${FAKE_UUID_SET:-a}"; i=$((i + 1)); done
  exit 0 ;;
esac
for bus in $FAKE_BUSES; do
  vb=86.06.5B.00.0C
  [ "$i" = "${FAKE_VBIOS_CARD:-x}" ] && vb=86.06.00.00.00
  echo "$i, 00000000:$bus:00.0, NVIDIA P106-090, $vb, 0, 75.00, 75.00, P8, ${FAKE_DRIVER:-580.178.04}"
  i=$((i + 1))
done
'''

FAKE_JOURNALCTL = '''#!/bin/sh
[ -n "${FAKE_KMSG:-}" ] && [ -f "$FAKE_KMSG" ] && cat "$FAKE_KMSG"
exit 0
'''

FAKE_WATCHDOG = r'''#!/usr/bin/env python3
# stands in for scripts/nvml_watchdog.py: holds until --stop exists; FAKE_CTL/TRIP sets every card to 0 and writes
# the marker, FAKE_CTL/CARD_TRIP (a card number) sets that card to 0 and writes the card marker, FAKE_CTL/WD_DIE ends it
import argparse, hashlib, json, os, sys, time
p = argparse.ArgumentParser()
for o in ("--log", "--marker", "--stop", "--card-marker", "--ledger", "--gpus", "--card-pids", "--temp-max",
          "--power-max", "--power-secs"):
    p.add_argument(o)
p.add_argument("--per-card-xid", action="store_true")
a = p.parse_args()
ctl, state = os.environ["FAKE_CTL"], os.environ["FAKE_STATE"]
tag = hashlib.sha256(open("/proc/sys/kernel/random/boot_id").read().strip().encode()).hexdigest()[:12]
log = open(a.log, "a", buffering=1)
log.write("# fake watchdog start\ntime,g0_sm_mhz,g0_mem_mhz,g0_pstate,pad\n")
def zero(cards, why):
    with open(state) as f:
        st = json.load(f)
    for g in cards:
        v = st.get(g, [0, 0])
        if any(v) and a.ledger:
            with open(a.ledger, "a") as f:
                f.write(f"{tag} {time.strftime('%Y-%m-%d %H:%M:%S')} card {g} mem {v[0]}->0 core {v[1]}->0 {why}\n")
        st[g] = [0, 0]
    with open(state + ".tmp", "w") as f:
        json.dump(st, f)
    os.replace(state + ".tmp", state)
while True:
    if a.stop and os.path.exists(a.stop):
        log.write("# watchdog stop\n")
        sys.exit(0)
    if os.path.exists(os.path.join(ctl, "WD_DIE")):
        sys.exit(1)
    if os.path.exists(os.path.join(ctl, "TRIP")):
        os.remove(os.path.join(ctl, "TRIP"))
        with open(state) as f:
            zero(list(json.load(f)), "watchdog trip")
        with open(a.marker, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} TRIP: card 0 temperature 71 C above 70 C\n")
    cp = os.path.join(ctl, "CARD_TRIP")
    if os.path.exists(cp):
        g = open(cp).read().strip()
        os.remove(cp)
        zero([g], "card trip")
        with open(a.card_marker, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} CARD {g} TRIP: NVRM: Xid (PCI:0000:04:00): 31\n")
    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},1898,4006,P0,{'x' * 200}\n")
    time.sleep(0.2)
'''

ROOT_SHIM = '''import os, runpy, sys
os.geteuid = lambda: 0   # the stand-in rig's only stand-in for root: climb.py's root check
sys.argv[0] = {real!r}
runpy.run_path({real!r}, run_name="__main__")
'''

SETTINGS = 'PRESET_ENV=""\nPRESET_ARGS="-ngl 99 -fa on"\n'


def free_base(n):
    """A port base with n free ports above it (checked, not assumed)."""
    for _ in range(200):
        base = random.randint(21000, 59000)
        socks = []
        try:
            for i in range(n):
                s = socket.socket()
                s.bind(("127.0.0.1", base + i))
                socks.append(s)
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError("no free ports")


class Rig:
    """The stand-in rig, its kit, a config and a profile; start() runs the service as the unit would."""

    def __init__(self, cards=2, buses=None, profile=None, **cfg):
        self.tmp = tempfile.mkdtemp(prefix="boot-test-")
        t = self.tmp
        for d in ("kit", "bin", "path", "ctl", "run", "lib", "ref"):
            os.makedirs(os.path.join(t, d))
        self.kit = os.path.join(t, "kit")
        files = {"kit/climb.py": ROOT_SHIM.format(real=os.path.join(KIT, "climb.py")),
                 "kit/nvml_memoc.py": FAKE_MEMOC, "kit/nvml_watchdog.py": FAKE_WATCHDOG,
                 "bin/llama-server": FAKE_SERVER, "path/nvidia-smi": FAKE_SMI, "path/journalctl": FAKE_JOURNALCTL,
                 "settings.env": SETTINGS, "check.gguf": "stand-in", "model.gguf": "stand-in"}
        for name, text in files.items():
            with open(os.path.join(t, name), "w") as f:
                f.write(text)
            os.chmod(os.path.join(t, name), 0o755)
        for name in ("greedy_check.py", "memoc_units.json", "model_check.sh"):
            shutil.copy(os.path.join(KIT, name), self.kit)
        self.state = os.path.join(t, "offsets.json")
        self.ledger = os.path.join(t, "ledger.log")
        self.calls_path = os.path.join(t, "calls.log")
        self.n = cards
        self.buses = buses or BUSES[:cards]
        self.set_offsets({})
        self.port = free_base(12)
        self.profile = os.path.join(t, "profile.json")
        self.write_profile(profile if profile is not None else {0: (1300, 256, 20), 1: (1000, 231, 18)})
        self.cfg = {"PROFILE": self.profile, "KIT": self.kit, "BIN": os.path.join(t, "bin"),
                    "CHECK_MODEL": os.path.join(t, "check.gguf"), "WORK": os.path.join(t, "lib", "clocks", "work"),
                    "STATE": os.path.join(t, "run", "clocks.state"), "LEDGER": self.ledger, "MMVQ_JOBS": "0",
                    "PORT_BASE": str(self.port), "STOCK_PEAK_SECS": "0", "GREEDY_N": "8", "POLL": "0.5",
                    "DRIVER_WAIT": "2", "APPLY_LIMIT": "150", "SERVER_LIMIT": "60", "KEEP_WORKS": "2",
                    "DRIVER": "580.178.04", "MODEL": os.path.join(t, "model.gguf"),
                    "MODEL_BIN": os.path.join(t, "bin"), "SETTINGS": os.path.join(t, "settings.env"),
                    "MODEL_REF": os.path.join(t, "ref"), "MODEL_PORT": str(self.port + 10)}
        self.cfg.update(cfg)
        self.config = os.path.join(t, "clocks.env")
        self.write_config()
        self.name = f"miner-llama-test-{os.getpid()}-{random.randint(0, 1 << 30)}"
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.bind("\0" + self.name)
        self.sock.setblocking(False)
        self.msgs = []
        self.env = dict(os.environ, FAKE_STATE=self.state, FAKE_CALLS=self.calls_path, FAKE_CTL=os.path.join(t, "ctl"),
                        FAKE_BUSES=" ".join(self.buses), FAKE_FAULTS="{}", NOTIFY_SOCKET="@" + self.name,
                        PATH=os.path.join(t, "path") + ":" + os.environ["PATH"])
        self.procs = []

    # -- files
    def write_profile(self, values, **extra):
        prof = {"what": "test", "complete": True, "climb_boot": "other-boot",
                "cards": [{"card": g, "pci": f"00000000:{BUSES[g]}:00.0", "name": "NVIDIA P106-090",
                           "vbios": "86.06.5B.00.0C", "stock_peak_mhz": 1898, "memory_mts": v[0], "core_mhz": v[1],
                           "core_steps": v[2]} for g, v in sorted(values.items())]}
        prof.update(extra)
        with open(self.profile, "w") as f:
            json.dump(prof, f, indent=1)

    def write_config(self, **over):
        self.cfg.update(over)
        with open(self.config, "w") as f:
            f.write("# test config\n" + "".join(f"{k}={v}\n" for k, v in self.cfg.items() if v is not None))

    def set_offsets(self, values):
        st = {str(g): [0, 0] for g in range(len(self.buses))}
        for g, v in values.items():
            st[str(g)] = list(v)
        with open(self.state, "w") as f:
            json.dump(st, f)

    def offsets(self):
        with open(self.state) as f:
            return {int(k): tuple(v) for k, v in json.load(f).items()}

    def calls(self):
        try:
            with open(self.calls_path) as f:
                return f.read().splitlines()
        except OSError:
            return []

    def state_file(self):
        try:
            with open(self.cfg["STATE"]) as f:
                return dict(ln.split(": ", 1) for ln in f.read().splitlines() if ": " in ln)
        except OSError:
            return {}

    def ledger_lines(self):
        try:
            with open(self.ledger) as f:
                return f.read().splitlines()
        except OSError:
            return []

    def ctl(self, name, text=""):
        with open(os.path.join(self.tmp, "ctl", name), "w") as f:
            f.write(text)

    def watchdogs(self):
        return cab.watchdog_pids(os.path.join(self.cfg["WORK"], "TRIPPED"))

    # -- the service
    def start(self, cmd="start", **env):
        e = dict(self.env, **env)
        with open(os.path.join(self.tmp, f"console-{cmd}.log"), "a") as out:
            p = subprocess.Popen([sys.executable, "-I", WRAPPER, "--config", self.config, cmd], env=e, stdout=out,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        self.procs.append(p)
        return p

    def run(self, cmd, timeout=300, **env):
        p = self.start(cmd, **env)
        return p.wait(timeout=timeout)

    def read_msgs(self):
        while True:
            try:
                self.msgs.append(self.sock.recv(4096).decode())
            except BlockingIOError:
                return self.msgs

    def ready(self):
        return any("READY=1" in m for m in self.read_msgs())

    def wait_for(self, what, timeout=150):
        t1 = time.time() + timeout
        while time.time() < t1:
            if what():
                return True
            time.sleep(0.2)
        return False

    def wait_ready(self, p, timeout=150):
        self.wait_for(lambda: self.ready() or p.poll() is not None, timeout)
        return self.ready()

    def console(self):
        out = []
        for name in sorted(os.listdir(self.tmp)):
            if name.startswith("console-"):
                with open(os.path.join(self.tmp, name)) as f:
                    out.append(f"== {name}\n{f.read()}")
        return "\n".join(out)

    def close(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
                p.wait()
        stop = os.path.join(self.cfg["WORK"], "WD_STOP")
        if os.path.isdir(self.cfg["WORK"]):
            open(stop, "a").close()
        for pid in self.watchdogs():
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        self.sock.close()
        if VERBOSE:
            print(self.console())
        shutil.rmtree(self.tmp, ignore_errors=True)


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


class BootCase(unittest.TestCase):
    def rig(self, **kw):
        r = Rig(**kw)
        self.addCleanup(r.close)
        return r

    def at_stock(self, r, msg=""):
        self.assertEqual(set(r.offsets().values()), {(0, 0)}, msg + "\n" + r.console())

    def refused(self, r, why_re, **env):
        before = r.offsets()
        rc = r.run("start", **env)
        st = r.state_file()
        self.assertEqual(rc, 1, r.console())
        self.assertEqual(st.get("overclocked"), "no", st)
        self.assertEqual(st.get("phase"), "refused", st)
        self.assertRegex(st.get("reason", ""), why_re)
        self.assertIn("no offset was touched", st.get("reason", ""))
        self.assertEqual(r.calls(), [], "an offset was set")
        self.assertEqual(r.offsets(), before, "an offset changed")
        self.assertEqual(r.watchdogs(), [], "a watchdog runs")
        self.assertFalse(r.ready())
        self.assertFalse(os.path.exists(r.cfg["STATE"] + ".owned"))
        return st


class TestApplied(BootCase):
    def test_applied_held_then_stopped(self):
        r = self.rig(WD_LOG_MAX_MB="0.002")
        g = subprocess.run([sys.executable, "-I", WRAPPER, "--config", r.config, "gpu-set-id"], env=r.env,
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(g.returncode, 0, g.stderr)
        self.assertRegex(g.stdout.strip(), r"^[0-9a-f]{16}$")
        r.write_config(GPU_SET_ID=g.stdout.strip())
        p = r.start()
        self.assertTrue(r.wait_ready(p), r.console())
        st = r.state_file()
        self.assertEqual(st["overclocked"], "yes", st)
        self.assertEqual(st["phase"], "overclocked")
        self.assertEqual(st["profile_md5"], md5(r.profile))
        self.assertEqual(st["cards"], "0:+1300/+256 1:+1000/+231")
        self.assertEqual(st["model_check"], "off")
        self.assertEqual(r.offsets(), {0: (1300, 256), 1: (1000, 231)})
        self.assertEqual(sorted(r.calls()), sorted(["set 0 MemClk 1300", "set 0 GpcClk 256", "set 1 MemClk 1000",
                                                    "set 1 GpcClk 231"]))
        self.assertEqual(len(r.watchdogs()), 1)
        work = r.cfg["WORK"]
        for name in ("boot.log", "clocks.state", "profile.json", "APPLY_DONE", "apply.console", "out/climb.log"):
            self.assertTrue(os.path.exists(os.path.join(work, name)), name)
        with open(os.path.join(work, "APPLY_DONE")) as f:
            self.assertEqual(f.read().strip(), "exit 0")
        # held: the watchdog's log is kept under WD_LOG_MAX_MB (2 KiB here; the stand-in writes 1 KiB a second)
        wd = os.path.join(work, "wd.csv")
        self.assertTrue(r.wait_for(lambda: os.path.exists(wd + ".1"), 30), "wd.csv was not rotated")
        self.assertTrue(r.wait_for(lambda: os.path.getsize(wd) < 3 * 2097, 10))
        with open(wd) as f:
            self.assertTrue(f.readline().startswith("#"))
        self.assertIsNone(p.poll())
        # the stop: every card to 0 while the watchdog still holds them, then the watchdog stopped
        p.send_signal(signal.SIGTERM)
        self.assertEqual(p.wait(timeout=120), 0, r.console())
        self.at_stock(r)
        st = r.state_file()
        self.assertEqual((st["overclocked"], st["phase"], st["watchdog"]), ("no", "stock", "stopped"), st)
        self.assertIn("the unit was stopped", st["reason"])
        self.assertEqual(r.watchdogs(), [])
        self.assertEqual(len([ln for ln in r.ledger_lines() if "reset-all" in ln]), 2, r.ledger_lines())
        self.assertFalse(os.path.exists(r.cfg["STATE"] + ".owned"))
        self.assertTrue(any("STOPPING=1" in m for m in r.read_msgs()))
        # stop-post after a clean stop: nothing to do
        self.assertEqual(r.run("stop-post"), 0)
        self.at_stock(r)
        # the same boot again: refused by the kit's start check (the step down), nothing set; the work folder rotates
        n = len(r.calls())
        rc = r.run("start")
        self.assertEqual(rc, 1)
        st = r.state_file()
        self.assertEqual((st["overclocked"], st["phase"]), ("no", "refused"))
        self.assertIn("stepped down this boot", st["reason"])
        self.assertEqual(len(r.calls()), n)
        parent = os.path.dirname(work)
        olds = [d for d in os.listdir(parent) if d.startswith("work.")]
        self.assertEqual(len(olds), 1, olds)
        for _ in range(3):
            time.sleep(1.1)   # a new stamp for each folder
            r.run("start")
        olds = [d for d in os.listdir(parent) if d.startswith("work.")]
        self.assertEqual(len(olds), 2, olds)   # KEEP_WORKS=2

    def test_card_check_fails(self):
        r = self.rig()
        p = r.start(FAKE_FAULTS=json.dumps({"1": {"mem_min": 500}}))
        self.assertFalse(r.wait_ready(p), r.console())
        self.assertEqual(p.wait(timeout=120), 1, r.console())
        self.at_stock(r)
        st = r.state_file()
        self.assertEqual((st["overclocked"], st["phase"]), ("no", "stock"), st)
        self.assertRegex(st["reason"], r"apply exit 1.*card 1 failed its check")
        self.assertEqual(r.watchdogs(), [])
        self.assertTrue(r.calls(), "apply set no offset before its check")

    def test_hung_check_past_the_limit(self):
        r = self.rig(APPLY_LIMIT="20")
        p = r.start(FAKE_FAULTS=json.dumps({"0": {"mem_min": 500, "hang": True}}))
        self.assertFalse(r.wait_ready(p, 60), r.console())
        self.assertEqual(p.wait(timeout=200), 1, r.console())
        self.at_stock(r)
        st = r.state_file()
        self.assertIn("apply had no result after 20 s (APPLY_LIMIT) and was ended", st["reason"])
        self.assertEqual(r.watchdogs(), [])
        self.assertIn("set 0 MemClk 1300", r.calls())


    def test_stopped_during_apply(self):
        r = self.rig()
        p = r.start(FAKE_FAULTS=json.dumps({"1": {"mem_min": 500, "hang": True}}))
        self.assertTrue(r.wait_for(lambda: len(r.calls()) == 4, 90), r.console())   # every offset set, checks running
        time.sleep(1)
        p.send_signal(signal.SIGTERM)
        self.assertEqual(p.wait(timeout=200), 0, r.console())
        self.at_stock(r)
        st = r.state_file()
        self.assertEqual((st["overclocked"], st["phase"]), ("no", "stock"), st)
        self.assertIn("stopped during apply", st["reason"])
        self.assertEqual(r.watchdogs(), [])
        self.assertFalse(r.ready())


class TestRefused(BootCase):
    def test_no_profile(self):
        r = self.rig()
        os.remove(r.profile)
        self.refused(r, r"PROFILE \(the rig's soaked profile\)")

    def test_config_line(self):
        r = self.rig()
        with open(r.config, "a") as f:
            f.write("this is not a setting\n")
        rc = r.run("start")
        self.assertEqual(rc, 1)
        self.assertIn("is not KEY=value", r.state_file()["reason"])
        self.assertEqual(r.calls(), [])

    def test_more_cards_than_the_profile(self):
        r = self.rig(cards=3)
        self.refused(r, r"the rig has 3 cards \(0 1 2\), the profile 2 \(0 1\)")

    def test_another_pci_address(self):
        r = self.rig(buses=["03", "05"])
        self.refused(r, r"card 1 is at 00000000:05:00.0, the profile has 00000000:04:00.0")

    def test_another_vbios(self):
        r = self.rig()
        self.refused(r, r"card 0 has VBIOS 86.06.00.00.00, the profile 86.06.5B.00.0C", FAKE_VBIOS_CARD="0")

    def test_another_card_set(self):
        r = self.rig()
        g = subprocess.run([sys.executable, "-I", WRAPPER, "--config", r.config, "gpu-set-id"], env=r.env,
                           capture_output=True, text=True, timeout=60)
        r.write_config(GPU_SET_ID=g.stdout.strip())
        # an identical rig: the same PCI addresses, names and VBIOS, other cards
        self.refused(r, r"these cards are not the set the config was made on: GPU_SET_ID " + g.stdout.strip(),
                     FAKE_UUID_SET="b")

    def test_another_driver(self):
        r = self.rig()
        self.refused(r, r"the driver is 590.1, DRIVER in the config is 580.178.04", FAKE_DRIVER="590.1")
        r.write_config(DRIVER="")
        r.write_profile({0: (1300, 256, 20), 1: (1000, 231, 18)}, driver="580.178.04")
        self.refused(r, r"the profile's driver is 580.178.04", FAKE_DRIVER="590.1")

    def test_an_offset_already_set(self):
        r = self.rig()
        r.set_offsets({0: (200, 0)})
        self.refused(r, r"start check: .*card 0 offsets \(mem core\) 200 0")
        self.assertEqual(r.offsets()[0], (200, 0))   # not the service's to reset

    def test_a_step_down_this_boot(self):
        r = self.rig()
        with open(r.ledger, "w") as f:
            f.write(f"{cab.boot_tag()} 2026-11-19 09:00:00 card 3 mem 1100->0 core 0->0 a soak\n")
        self.refused(r, r"an offset stepped down this boot")

    def test_cards_against_the_profile(self):
        r = self.rig(CARDS="0 1 2")
        self.refused(r, r"CARDS=0 1 2 in the config is not the profile's cards \[0, 1\]")

    def test_no_driver(self):
        r = self.rig(DRIVER_WAIT="1")
        self.refused(r, r"after 1 s: nvidia-smi gave no cards", FAKE_SMI_FAIL="1")

    def test_model_check_without_its_hash(self):
        r = self.rig(MODEL_CHECK="1")
        self.refused(r, r"MODEL_REF/greedy-stock1.sha")


class TestHeld(BootCase):
    def held(self, **kw):
        r = self.rig(**kw)
        p = r.start()
        self.assertTrue(r.wait_ready(p), r.console())
        self.assertEqual(r.state_file()["overclocked"], "yes")
        return r, p

    def test_watchdog_trip(self):
        r, p = self.held()
        r.ctl("TRIP")
        self.assertEqual(p.wait(timeout=120), 1, r.console())
        self.at_stock(r)
        st = r.state_file()
        self.assertEqual((st["overclocked"], st["phase"]), ("no", "stock"), st)
        self.assertIn("the watchdog tripped: ", st["reason"])
        self.assertEqual(r.watchdogs(), [])

    def test_watchdog_ends(self):
        r, p = self.held()
        r.ctl("WD_DIE")
        self.assertEqual(p.wait(timeout=120), 1, r.console())
        self.at_stock(r)
        self.assertIn("the watchdog ended", r.state_file()["reason"])

    def test_one_card_trips(self):
        r, p = self.held()
        r.ctl("CARD_TRIP", "1")
        self.assertTrue(r.wait_for(lambda: r.state_file().get("overclocked") == "partly", 30), r.console())
        st = r.state_file()
        self.assertEqual(st["card_trips"], "1")
        self.assertIn("card(s) 1 to 0", st["reason"])
        self.assertEqual(r.offsets(), {0: (1300, 256), 1: (0, 0)})
        self.assertIsNone(p.poll())
        p.send_signal(signal.SIGTERM)
        self.assertEqual(p.wait(timeout=120), 0)
        self.at_stock(r)
        self.assertEqual(r.state_file()["overclocked"], "no")

    def test_main_killed_then_stop_post(self):
        r, p = self.held()
        p.kill()
        p.wait()
        self.assertEqual(r.offsets(), {0: (1300, 256), 1: (1000, 231)})
        self.assertEqual(len(r.watchdogs()), 1)
        self.assertEqual(r.run("stop-post"), 1, r.console())
        self.at_stock(r)
        st = r.state_file()
        self.assertEqual((st["overclocked"], st["phase"]), ("no", "stock"), st)
        self.assertIn("ended without its reset", st["reason"])
        self.assertEqual(r.watchdogs(), [])
        self.assertFalse(os.path.exists(r.cfg["STATE"] + ".owned"))


class TestModelCheck(BootCase):
    def with_reference(self):
        r = self.rig(MODEL_CHECK="1")
        e = dict(r.env, BIN=r.cfg["MODEL_BIN"], MODEL=r.cfg["MODEL"], SETTINGS=r.cfg["SETTINGS"], OUT=r.cfg["MODEL_REF"],
                 TAG="stock", ROUNDS="2", MINUTES="0", PORT=r.cfg["MODEL_PORT"], WORK=os.path.join(r.tmp, "refwork"),
                 N="8")
        p = subprocess.run(["bash", os.path.join(r.kit, "model_check.sh")], env=e, capture_output=True, text=True,
                           timeout=120)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertTrue(os.path.exists(os.path.join(r.cfg["MODEL_REF"], "greedy-stock1.sha")))
        return r

    def test_passes(self):
        r = self.with_reference()
        p = r.start()
        self.assertTrue(r.wait_ready(p, 200), r.console())
        st = r.state_file()
        self.assertEqual(st["overclocked"], "yes", st)
        with open(os.path.join(r.cfg["MODEL_REF"], "greedy-stock1.sha")) as f:
            ref = f.read().strip()
        self.assertEqual(st["model_check"], f"passed (2 rounds equal to the stock hash {ref})")
        self.assertIn("the model check passed", st["reason"])
        p.send_signal(signal.SIGTERM)
        self.assertEqual(p.wait(timeout=120), 0)
        self.at_stock(r)

    def test_differs(self):
        r = self.with_reference()
        p = r.start(FAKE_MODEL_WRONG="1")
        self.assertFalse(r.wait_ready(p, 200), r.console())
        self.assertEqual(p.wait(timeout=120), 1)
        self.at_stock(r)
        st = r.state_file()
        self.assertIn("the model check failed (exit 1)", st["reason"])
        self.assertEqual(st["model_check"], "failed (exit 1)")
        self.assertEqual(r.watchdogs(), [])


class TestFiles(unittest.TestCase):
    def test_unit_verifies(self):
        if not shutil.which("systemd-analyze"):
            self.skipTest("no systemd-analyze")
        t = tempfile.mkdtemp(prefix="boot-unit-")
        self.addCleanup(shutil.rmtree, t, True)
        shutil.copy(os.path.join(BOOT, "miner-llama-clocks.service"), t)
        p = subprocess.run(["systemd-analyze", "verify", "./miner-llama-clocks.service"], cwd=t, capture_output=True,
                           text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertNotIn("miner-llama-clocks", p.stdout + p.stderr)

    def test_unit_runs_what_install_puts_there(self):
        with open(os.path.join(BOOT, "miner-llama-clocks.service")) as f:
            unit = f.read()
        for line in ("Type=notify", "KillMode=mixed", "Restart=no", "WantedBy=multi-user.target",
                     "ExecStart=/usr/bin/python3 -I /opt/miner-llama/mem_offset/boot/clocks_at_boot.py --config "
                     "/etc/miner-llama/clocks.env start"):
            self.assertIn(line, unit)

    def test_install(self):
        t = tempfile.mkdtemp(prefix="boot-install-")
        self.addCleanup(shutil.rmtree, t, True)
        root, path = os.path.join(t, "root"), os.path.join(t, "path")
        os.makedirs(path)
        with open(os.path.join(path, "systemctl"), "w") as f:
            f.write(f"#!/bin/sh\necho \"$@\" >> {t}/systemctl.calls\n")
        os.chmod(os.path.join(path, "systemctl"), 0o755)
        env = dict(os.environ, ROOT=root, PATH=path + ":" + os.environ["PATH"])
        p = subprocess.run(["bash", os.path.join(BOOT, "install.sh")], env=env, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        dest = os.path.join(root, "opt", "miner-llama", "mem_offset")
        for name in ("climb.py", "nvml_memoc.py", "greedy_check.py", "memoc_units.json", "model_check.sh",
                     "nvml_watchdog.py", "boot/clocks_at_boot.py", "boot/miner-llama-clocks.service"):
            self.assertTrue(os.path.exists(os.path.join(dest, name)), name)
        self.assertEqual(md5(os.path.join(dest, "boot", "clocks_at_boot.py")), md5(WRAPPER))
        self.assertEqual(md5(os.path.join(dest, "nvml_watchdog.py")),
                         md5(os.path.join(KIT, "..", "..", "scripts", "nvml_watchdog.py")))
        self.assertTrue(os.access(os.path.join(dest, "boot", "clocks_at_boot.py"), os.X_OK))
        self.assertTrue(os.path.exists(os.path.join(root, "etc", "systemd", "system", "miner-llama-clocks.service")))
        cfg = os.path.join(root, "etc", "miner-llama", "clocks.env")
        self.assertEqual(md5(cfg), md5(os.path.join(BOOT, "clocks.env.example")))
        self.assertIn("mmvq     NONE", p.stdout)
        self.assertIn("not enabled", p.stdout)
        self.assertFalse(os.path.exists(os.path.join(t, "systemctl.calls")), "systemctl ran under ROOT")
        # again, with an edited config and an mmvq_pascal: the config is kept, the binary copied
        with open(cfg, "a") as f:
            f.write("BIN=/some/engine/bin\n")
        mm = os.path.join(t, "mmvq_pascal")
        with open(mm, "w") as f:
            f.write("stand-in binary\n")
        p = subprocess.run(["bash", os.path.join(BOOT, "install.sh")], env=dict(env, MMVQ_BIN=mm),
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("kept as it was", p.stdout)
        with open(cfg) as f:
            self.assertIn("BIN=/some/engine/bin", f.read())
        self.assertEqual(md5(os.path.join(dest, "bin", "mmvq_pascal")), md5(mm))
        # the example config reads, and the installer never enables anything
        cfg_d, bad = cab.read_config(os.path.join(BOOT, "clocks.env.example"))
        self.assertEqual(bad, [])
        self.assertEqual(cfg_d["MODEL_CHECK"], "0")
        with open(os.path.join(BOOT, "install.sh")) as f:
            code = [ln for ln in f.read().splitlines() if not ln.lstrip().startswith("#") and "echo" not in ln]
        self.assertFalse([ln for ln in code if re.search(r"systemctl\s+(enable|start|restart)", ln)])

    def test_install_from_a_kit_tarball(self):
        # the layout build.sh packs (its own file list, read from build.sh): install.sh runs from it
        with open(os.path.join(KIT, "build.sh")) as f:
            text = f.read()
        part = text[text.index('cp "$here/nvml_memoc.py"'):text.index("md5sum bin/*")]
        names = re.findall(r'"\$here/([^"]+)"', part)
        self.assertIn("boot/install.sh", names)
        self.assertIn("model_check.sh", names)
        t = tempfile.mkdtemp(prefix="boot-tarball-")
        self.addCleanup(shutil.rmtree, t, True)
        kit = os.path.join(t, "mem_offset-kit-test")
        os.makedirs(os.path.join(kit, "boot"))
        os.makedirs(os.path.join(kit, "bin"))
        for n in names:
            sub = "boot" if n.startswith("boot/") else ""
            shutil.copy(os.path.join(KIT, n), os.path.join(kit, sub, os.path.basename(n)))
        with open(os.path.join(kit, "bin", "mmvq_pascal"), "w") as f:
            f.write("stand-in binary\n")
        root = os.path.join(t, "root")
        p = subprocess.run(["bash", os.path.join(kit, "boot", "install.sh")], env=dict(os.environ, ROOT=root),
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn(f"mmvq     from {kit}/bin", p.stdout)
        dest = os.path.join(root, "opt", "miner-llama", "mem_offset")
        self.assertEqual(md5(os.path.join(dest, "nvml_watchdog.py")), md5(os.path.join(kit, "nvml_watchdog.py")))
        # and not from the installed copy itself
        p = subprocess.run(["bash", os.path.join(dest, "boot", "install.sh")], env=dict(os.environ, ROOT=root),
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)

    def test_nothing_rig_specific(self):
        pat = re.compile(r"/srv/|/mnt/|/home/|GPU-[0-9a-f]{8}-")
        for name in os.listdir(BOOT):
            if name.endswith((".py", ".sh", ".service", ".example")):
                with open(os.path.join(BOOT, name)) as f:
                    self.assertFalse(pat.findall(f.read()), name)


class TestHelpers(unittest.TestCase):
    def test_read_config(self):
        t = tempfile.mkdtemp(prefix="boot-cfg-")
        self.addCleanup(shutil.rmtree, t, True)
        path = os.path.join(t, "c.env")
        with open(path, "w") as f:
            f.write("# a comment\n\nexport BIN=/x/bin\nCHECK_MODEL='/m/a b.gguf'\nWORK=/var/lib/w   # trailing\n"
                    'MODEL="q#1"\nEMPTY=\n')
        cfg, bad = cab.read_config(path)
        self.assertEqual(bad, [])
        self.assertEqual(cfg, {"BIN": "/x/bin", "CHECK_MODEL": "/m/a b.gguf", "WORK": "/var/lib/w", "MODEL": "q#1",
                               "EMPTY": ""})
        with open(path, "w") as f:
            f.write("bin=/x\nA B=1\n. ./env.sh\n")
        self.assertEqual(len(cab.read_config(path)[1]), 3)
        self.assertIn("No such file", cab.read_config(os.path.join(t, "none"))[1][0])

    def test_match_cards(self):
        smi = cab.parse_smi("0, 00000000:03:00.0, NVIDIA P106-090, 86.06.5B.00.0C, 0, 75.00, 75.00, P8, 580.178.04\n"
                            "1, 00000000:04:00.0, NVIDIA P106-090, 86.06.5B.00.0C, 0, 75.00, 75.00, P8, 580.178.04\n")
        prof = {"cards": [{"card": 0, "pci": "0000:03:00", "name": "NVIDIA P106-090", "vbios": "86.06.5B.00.0C"},
                          {"card": 1, "pci": "00000000:04:00.0", "name": "NVIDIA P106-090"}]}
        self.assertEqual(cab.match_cards(prof, smi, "580.178.04"), [])
        prof["cards"][1]["name"] = "NVIDIA P106-100"
        self.assertEqual(cab.match_cards(prof, smi), ["card 1 is a NVIDIA P106-090, the profile has NVIDIA P106-100"])
        del prof["cards"][1]["pci"]
        self.assertIn("no pci or name", cab.match_cards(prof, smi)[0])
        with self.assertRaises(cab.Refused):
            cab.match_cards({"cards": [{"card": 0}, {"card": 0}]}, smi)
        with self.assertRaises(cab.Refused):
            cab.match_cards({"cards": []}, smi)

    def test_state_lines(self):
        text = ("boot ca8d5b60310f, driver 580.178.04\ncard,pci,memory_used_mib,power_limit_w,default_limit_w,pstate,"
                "mem_offset,core_offset\n0,00000000:03:00.0,0,75.00,75.00,P0,1300,256\n"
                "1,00000000:04:00.0,0,75.00,75.00,P0,None,None\nload processes: none\nwatchdog running: yes\n")
        self.assertEqual(cab.parse_state(text), ({0: (1300, 256), 1: (None, None)}, True))
        self.assertEqual(cab.offsets_wanted({"cards": [{"card": 0, "memory_mts": 1300, "core_mhz": 256}]}, 0.5),
                         {0: (650, 256)})

    def test_gpu_set_id(self):
        a = "0, 00000000:03:00.0, GPU-aaaa\n1, 00000000:04:00.0, GPU-bbbb\n"
        b = "1, 00000000:04:00.0, GPU-bbbb\n0, 00000000:03:00.0, GPU-aaaa\n"
        self.assertEqual(cab.gpu_set_id(a), cab.gpu_set_id(b))   # the order nvidia-smi prints does not matter
        self.assertNotEqual(cab.gpu_set_id(a), cab.gpu_set_id(a.replace("bbbb", "cccc")))
        self.assertEqual(cab.gpu_set_id("No devices were found\n"), "")


if __name__ == "__main__":
    unittest.main()
