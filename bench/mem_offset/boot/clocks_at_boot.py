#!/usr/bin/env python3
"""clocks_at_boot.py: a rig's soaked clock-offset profile set at boot, checked, and held while the watchdog runs.

The main process of miner-llama-clocks.service (bench/mem_offset/boot/; the kit README's section "Overclocked at every
boot"). It runs the kit's own commands as a person would on a fresh boot, each one under a time limit:

  1. reads the rig's config (--config, default /etc/miner-llama/clocks.env) and waits for the NVIDIA driver and the
     files it names (DRIVER_WAIT seconds);
  2. refuses, with every card left at stock and no offset touched, when: the config, the kit or the profile is missing
     or does not read; the cards present do not match the profile (the same card numbers, and each card's PCI address,
     name and VBIOS as the profile records them) or, with GPU_SET_ID in the config, are not the set the config was
     made on (identical rigs share PCI addresses); the driver differs from DRIVER or from the profile's; the model
     check is on and its files are missing; a watchdog already runs on the work folder; or the kit's `start-check`
     fails (a load runs, a card's memory is in use, a power limit was changed, an offset is set, or an offset stepped
     down since the boot);
  3. starts the kit's watchdog, runs the kit's `apply` of the profile (every card set and read back, then every card's
     greedy check against its own stock result), and with MODEL_CHECK=1 the served model's whole-memory check
     (model_check.sh, two rounds against the stock hash stored in MODEL_REF); then reads every offset back against
     the profile;
  4. on any failure, and when stopped: every offset of every card to 0 and read back, then the watchdog stopped;
  5. on success: tells systemd it is ready and stays, the unit's main process, while the watchdog holds the cards.
     A watchdog trip, or the watchdog's end, sets every card to 0 and ends the unit; a trip of one card (an Xid that
     names it) is recorded ("overclocked: partly") and the other cards keep their offsets. Stopping the unit sets
     every card to 0.

Every step goes to the journal (stdout) and to WORK/boot.log. The state file (STATE, default
/run/miner-llama/clocks.state; /run is empty again at every boot) holds "overclocked: yes", "no", "partly" or
"unknown", the phase, the reason, the profile's md5 and the time; a copy goes to WORK/clocks.state.

  clocks_at_boot.py [--config FILE] start        the unit's ExecStart (Type=notify)
  clocks_at_boot.py [--config FILE] stop-post    the unit's ExecStopPost: if the main process ended without its own
                                                 reset (killed, or a bug), every card to 0 and the watchdog stopped
  clocks_at_boot.py [--config FILE] status       prints the state file
  clocks_at_boot.py gpu-set-id                   read-only: this rig's card-set id for GPU_SET_ID in the config (a
                                                 hash of the cards' PCI addresses and UUIDs; no serial is printed)

Exit status: 0 done (a clean stop, or nothing to do), 1 a refusal or a failure (every card at 0), 2 usage, 3 a reset
that did not read back 0 (power the rig off at the console).
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

DEFAULTS = {
    "KIT": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "PROFILE": "",
    "WORK": "/var/lib/miner-llama/clocks/work",
    "KEEP_WORKS": "5",
    "STATE": "/run/miner-llama/clocks.state",
    "DRIVER": "",
    "GPU_SET_ID": "",
    "DRIVER_WAIT": "180",
    "APPLY_LIMIT": "2400",
    "APPLY_ROUNDS": "1",
    "MODEL_CHECK": "0",
    "MODEL": "",
    "MODEL_BIN": "",
    "SETTINGS": "",
    "MODEL_REF": "/var/lib/miner-llama/model-ref",
    "MODEL_ROUNDS": "2",
    "MODEL_LIMIT": "3600",
    "MODEL_PORT": "8300",
    "POLL": "10",
    "WD_LOG_MAX_MB": "64",
}
INTS = ("KEEP_WORKS", "DRIVER_WAIT", "APPLY_LIMIT", "APPLY_ROUNDS", "MODEL_ROUNDS", "MODEL_LIMIT", "MODEL_PORT")
FLOATS = ("POLL", "WD_LOG_MAX_MB")
# the kit's own query (climb.py Kit.smi), so the card list is read the same way apply reads it
SMI = ["nvidia-smi", "--query-gpu=index,pci.bus_id,name,vbios_version,memory.used,power.limit,power.default_limit,"
       "pstate,driver_version", "--format=csv,noheader,nounits"]
# the cards' own ids, read only to hash them: which rig this is, without a serial in any file
SMI_UUID = ["nvidia-smi", "--query-gpu=index,pci.bus_id,uuid", "--format=csv,noheader"]
KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
PCI_RE = re.compile(r"([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})")
LIMITS = {"smi": 30, "start-check": 120, "watchdog start": 60, "watchdog stop": 60, "state": 120, "reset-all": 180}
GRACE = 90   # seconds a command gets to end after TERM (apply stops its check servers and resets) before KILL


class Refused(Exception):
    """A check before any offset failed: every card stays as it is (at stock on a fresh boot)."""


class Failed(Exception):
    """A step after the watchdog started failed: every card goes to 0."""


# ---------------------------------------------------------------- small helpers (pure; test/test_boot.py)

def read_config(path):
    """KEY=value lines: an optional 'export ', '#' comments (also after an unquoted value, as ' # ...'), a value
    optionally in matching quotes, no expansion. Returns ({key: value}, [why a line does not read])."""
    cfg, bad = {}, []
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError as e:
        return cfg, [f"config {path}: {e.strerror or e}"]
    for n, raw in enumerate(lines, 1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export "):].lstrip()
        k, eq, v = s.partition("=")
        k = k.strip()
        if not eq or not KEY_RE.match(k):
            bad.append(f"config {path} line {n} is not KEY=value: {s[:80]}")
            continue
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        elif " #" in v or "\t#" in v:
            v = re.split(r"\s#", v, maxsplit=1)[0].rstrip()
        cfg[k] = v
    return cfg, bad


def pci_key(text):
    """'00000000:05:00.0' or '0000:05:00' -> (domain, bus, device) as ints; None if it does not read."""
    m = PCI_RE.search(text or "")
    return tuple(int(v, 16) for v in m.groups()) if m else None


def parse_smi(text):
    """The kit's nvidia-smi query -> {index: {"pci", "name", "vbios", "used", "driver"}}."""
    cards = {}
    for ln in (text or "").strip().splitlines():
        r = [v.strip() for v in ln.split(",")]
        if len(r) < 9 or not r[0].isdigit():
            continue
        cards[int(r[0])] = {"pci": r[1], "name": r[2], "vbios": r[3], "used": r[4], "driver": r[8]}
    return cards


def profile_cards(prof):
    """{card: entry} of a profile's cards, or Refused when it has none or an entry without a card number."""
    cards = prof.get("cards") if isinstance(prof, dict) else None
    if not isinstance(cards, list) or not cards:
        raise Refused("the profile has no cards")
    by = {}
    for c in cards:
        if not isinstance(c, dict) or not isinstance(c.get("card"), int) or isinstance(c.get("card"), bool):
            raise Refused(f"the profile has a card entry without a card number: {str(c)[:80]}")
        if c["card"] in by:
            raise Refused(f"the profile lists card {c['card']} twice")
        by[c["card"]] = c
    return by


def match_cards(prof, smi, driver=""):
    """Why the cards present do not match the profile ([] when they do): the same card numbers, and each card's PCI
    address, name and VBIOS as the profile records them (a profile holds no serials or UUIDs: it is public-ready),
    and the driver against DRIVER and the profile's `driver`, when given."""
    by = profile_cards(prof)
    bad = []
    if sorted(smi) != sorted(by):
        bad.append(f"the rig has {len(smi)} cards ({' '.join(str(g) for g in sorted(smi))}), the profile "
                   f"{len(by)} ({' '.join(str(g) for g in sorted(by))})")
    for g in sorted(set(smi) & set(by)):
        p, c = by[g], smi[g]
        if not p.get("pci") or not p.get("name"):
            bad.append(f"card {g}: the profile has no pci or name to check the card against")
            continue
        if pci_key(p["pci"]) != pci_key(c["pci"]):
            bad.append(f"card {g} is at {c['pci']}, the profile has {p['pci']}")
        if p["name"] != c["name"]:
            bad.append(f"card {g} is a {c['name']}, the profile has {p['name']}")
        if p.get("vbios") and p["vbios"] != c["vbios"]:
            bad.append(f"card {g} has VBIOS {c['vbios']}, the profile {p['vbios']}")
    drivers = sorted({c["driver"] for c in smi.values()})
    for want, src in ((driver, "DRIVER in the config"), (prof.get("driver") or "", "the profile's driver")):
        if want and drivers != [want]:
            bad.append(f"the driver is {' '.join(drivers)}, {src} is {want} (the offsets' units were measured on "
                       "it; measure them again with nvml_memoc.py units, then climb and soak on the new driver)")
    return bad


def gpu_set_id(text):
    """This rig's card set as one id: a hash of every card's index, PCI address and UUID (nvidia-smi SMI_UUID), so
    that identical rigs, whose cards share PCI addresses, names and VBIOS, still differ. '' if nothing reads."""
    rows = sorted(",".join(v.strip() for v in ln.split(",")) for ln in (text or "").strip().splitlines()
                  if ln.count(",") >= 2 and ln.split(",")[0].strip().isdigit())
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:16] if rows else ""


def parse_state(text):
    """`climb.py state` -> ({card: (mem raw, core MHz)}, watchdog running?)."""
    offs = {}
    for ln in (text or "").splitlines():
        r = ln.strip().split(",")
        if len(r) == 8 and r[0].isdigit():
            try:
                offs[int(r[0])] = (int(r[6]), int(r[7]))
            except ValueError:
                offs[int(r[0])] = (None, None)
    return offs, "watchdog running: yes" in (text or "")


def offsets_wanted(prof, raw_per_mts):
    """{card: (mem raw, core MHz)} the profile gives, in nvml_memoc.py's raw units."""
    return {g: (int(round((c.get("memory_mts") or 0) * raw_per_mts)), int(c.get("core_mhz") or 0))
            for g, c in profile_cards(prof).items()}


def boot_tag():
    """The boot's tag, as the kit's ledger and profiles write it: a hash of the kernel's boot id, not the id."""
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            return hashlib.sha256(f.read().strip().encode()).hexdigest()[:12]
    except OSError:
        return "unknown"


def watchdog_pids(marker):
    """The pids of nvml_watchdog.py processes started with --marker MARKER (how climb.py finds its watchdog)."""
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as f:
                args = f.read().decode(errors="replace").split("\0")
        except OSError:
            continue
        if any(a.endswith("nvml_watchdog.py") for a in args) and \
                any(a == "--marker" and i + 1 < len(args) and args[i + 1] == marker for i, a in enumerate(args)):
            out.append(int(d))
    return out


def notify(msg):
    """One message to systemd (Type=notify); nothing when not run by systemd."""
    path = os.environ.get("NOTIFY_SOCKET")
    if not path:
        return
    if path.startswith("@"):
        path = "\0" + path[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(path)
            s.sendall(msg.encode())
    except OSError:
        pass


def md5_of(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ---------------------------------------------------------------- the service

class Service:
    def __init__(self, config):
        self.config = config
        self.c = dict(DEFAULTS)
        self.extra = {}            # config keys this file does not read: passed to the kit as they are
        self.env = {k: v for k, v in os.environ.items() if k != "NOTIFY_SOCKET"}
        self.stopping = False      # a stop signal came
        self.cleaning = False      # resetting: a signal then changes nothing
        self.work = None
        self.pending = []          # log lines before WORK exists
        self.st = {"overclocked": "no", "phase": "starting", "reason": "", "since": "", "boot": boot_tag(),
                   "profile": "", "profile_md5": "", "cards": "", "card_trips": "0", "model_check": "off",
                   "watchdog": "not started", "work": ""}
        self.prof = None
        self.cards = []
        self.trips_seen = 0

    # -- logging and state
    def log(self, msg):
        print(msg, flush=True)   # the journal stamps the time
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        if self.work and os.path.isdir(self.work):
            with open(os.path.join(self.work, "boot.log"), "a") as f:
                for ln in self.pending + [line]:
                    f.write(ln + "\n")
            self.pending = []
        else:
            self.pending.append(line)

    def state(self, **kw):
        self.st.update({k: str(v) for k, v in kw.items()})
        self.st["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        order = ("overclocked", "phase", "reason", "time", "since", "boot", "profile", "profile_md5", "cards",
                 "card_trips", "model_check", "watchdog", "work")
        text = "".join(f"{k}: {self.st.get(k, '')}\n" for k in order)
        paths = [self.c["STATE"]] + ([os.path.join(self.work, "clocks.state")] if self.work and
                                     os.path.isdir(self.work) else [])
        for p in paths:
            try:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p + ".tmp", "w") as f:
                    f.write(text)
                os.replace(p + ".tmp", p)
            except OSError as e:
                print(f"state file {p}: {e}", flush=True)
        notify(f"STATUS=overclocked: {self.st['overclocked']}; {self.st['phase']}: {self.st['reason'][:200]}")

    # -- the config
    def load(self):
        cfg, bad = read_config(self.config)
        for k, v in cfg.items():   # the lines that read, so that a refusal still goes to the config's STATE
            (self.c if k in DEFAULTS else self.extra)[k] = v
        if bad:
            raise Refused("; ".join(bad))
        for k in INTS + FLOATS:
            try:
                v = int(self.c[k]) if k in INTS else float(self.c[k])
            except ValueError:
                raise Refused(f"config {self.config}: {k}={self.c[k]} is not a number")
            if v < 0:
                raise Refused(f"config {self.config}: {k}={self.c[k]} is below 0")
            self.c[k] = v
        self.c["MODEL_CHECK"] = self.c["MODEL_CHECK"].strip().lower() in ("1", "yes", "true", "on")
        w = self.c["WORK"]
        if not os.path.isabs(w) or len([p for p in w.split("/") if p]) < 2 or w.rstrip("/") != os.path.normpath(w):
            raise Refused(f"config {self.config}: WORK={w} must be an absolute path two folders deep or more")
        self.c["WORK"] = os.path.normpath(w)
        self.resolve_kit()

    def resolve_kit(self):
        kit = self.c["KIT"]
        wd = os.path.join(kit, "nvml_watchdog.py")
        if not os.path.exists(wd):
            wd = os.path.normpath(os.path.join(kit, "..", "..", "scripts", "nvml_watchdog.py"))
        self.kit_files = {"MEMOC": os.path.join(kit, "nvml_memoc.py"), "GREEDY": os.path.join(kit, "greedy_check.py"),
                          "UNITS": os.path.join(kit, "memoc_units.json"), "WATCHDOG": wd,
                          "MMVQ": os.path.join(kit, "bin", "mmvq_pascal")}
        for k in self.kit_files:
            if k in self.extra:
                self.kit_files[k] = self.extra.pop(k)

    def kit_env(self):
        e = dict(self.env)
        e.update(self.extra)
        e.update(self.kit_files)
        e["WORK"] = self.c["WORK"]
        if self.cards:
            e["CARDS"] = " ".join(str(g) for g in self.cards)
        return e

    def climb(self, *args):
        return [sys.executable, os.path.join(self.c["KIT"], "climb.py")] + [str(a) for a in args]

    # -- commands under a time limit
    def run(self, cmd, limit, what, env=None, out=None, grace=10):
        """(exit status, output). Past `limit` seconds: TERM to the command's process group, KILL `grace` s later, and
        status 124. A stop signal ends it the same way (unless resetting) and the caller sees self.stopping."""
        fh = open(out, "a") if out else None
        try:
            p = subprocess.Popen(cmd, env=env or self.kit_env(), stdin=subprocess.DEVNULL,
                                 stdout=fh or subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 start_new_session=True)
        except OSError as e:
            if fh:
                fh.close()
            return 127, f"{what}: {e}"
        t0 = time.monotonic()
        try:
            while True:
                try:
                    o, _ = p.communicate(timeout=0.5)
                    return p.returncode, o or ""
                except subprocess.TimeoutExpired:
                    pass
                late = time.monotonic() - t0 > limit
                if late or (self.stopping and not self.cleaning):
                    self.end(p, grace)
                    try:
                        o, _ = p.communicate(timeout=5)
                    except (subprocess.TimeoutExpired, ValueError):
                        o = ""
                    why = f"no result after {limit:.0f} s" if late else "stopped (the unit is stopping)"
                    return 124, (o or "") + f"\n{what}: {why}"
        finally:
            if fh:
                fh.close()

    @staticmethod
    def end(p, grace):
        for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10)):
            try:
                os.killpg(p.pid, sig)
            except OSError:
                pass
            try:
                p.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue

    def kit(self, what, *args):
        rc, o = self.run(self.climb(*args), LIMITS[what], what)
        for ln in o.strip().splitlines():
            self.log(f"  {what}: {ln}")
        return rc, o

    def extend(self, seconds):
        notify(f"EXTEND_TIMEOUT_USEC={int((seconds + 300) * 1e6)}")

    # -- the work folder: a fresh one every start, the last KEEP_WORKS kept beside it
    def fresh_work(self):
        w = self.c["WORK"]
        parent, base = os.path.dirname(w), os.path.basename(w)
        os.makedirs(parent, exist_ok=True)
        if os.path.lexists(w):
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.lstat(w).st_mtime))
            dst, n = f"{w}.{stamp}", 1
            while os.path.lexists(dst):
                dst, n = f"{w}.{stamp}-{n}", n + 1
            os.rename(w, dst)
        os.makedirs(os.path.join(w, "pids"))
        old_re = re.compile(re.escape(base) + r"\.\d{8}-\d{6}(-\d+)?$")
        old = sorted((d for d in os.listdir(parent) if old_re.match(d) and os.path.isdir(os.path.join(parent, d))
                      and not os.path.islink(os.path.join(parent, d))),
                     key=lambda d: os.path.getmtime(os.path.join(parent, d)))
        for d in old[:max(0, len(old) - self.c["KEEP_WORKS"])]:
            shutil.rmtree(os.path.join(parent, d), ignore_errors=True)
        self.work = w
        self.st["work"] = w

    def marker(self, name="TRIPPED"):
        return os.path.join(self.c["WORK"], name)

    # -- step 1 and 2: the checks before any offset
    def needed_files(self):
        c, k = self.c, self.c["KIT"]
        need = [(os.path.join(k, "climb.py"), "the kit's climb.py"), (self.kit_files["MEMOC"], "nvml_memoc.py"),
                (self.kit_files["GREEDY"], "greedy_check.py"), (self.kit_files["UNITS"], "memoc_units.json"),
                (self.kit_files["WATCHDOG"], "nvml_watchdog.py")]
        if str(self.extra.get("MMVQ_JOBS", "2")).strip() != "0":
            need.append((self.kit_files["MMVQ"], "mmvq_pascal (or MMVQ_JOBS=0 in the config)"))
        bin_ = self.extra.get("BIN", "")
        need += [(os.path.join(bin_, "llama-server") if bin_ else "", "BIN/llama-server (apply's check servers)"),
                 (self.extra.get("CHECK_MODEL", ""), "CHECK_MODEL (apply's check model)"),
                 (c["PROFILE"], "PROFILE (the rig's soaked profile)")]
        if c["MODEL_CHECK"]:
            mb = c["MODEL_BIN"]
            need += [(os.path.join(k, "model_check.sh"), "the kit's model_check.sh"),
                     (os.path.join(mb, "llama-server") if mb else "", "MODEL_BIN/llama-server (the model check)"),
                     (c["MODEL"], "MODEL (the served model)"), (c["SETTINGS"], "SETTINGS (its release settings)"),
                     (os.path.join(c["MODEL_REF"], "greedy-stock1.sha"),
                      "MODEL_REF/greedy-stock1.sha (the model's stock hash)")]
        return need

    def wait_ready(self):
        """The driver answers and every file is there, within DRIVER_WAIT s; Refused with what is missing."""
        t0 = time.monotonic()
        while True:
            rc, o = self.run(SMI, LIMITS["smi"], "nvidia-smi", env=self.env)
            smi = parse_smi(o) if rc == 0 else {}
            missing = [f"{what} ({path or 'not set'})" for path, what in self.needed_files()
                       if not path or not os.path.exists(path)]
            if smi and not missing:
                return smi
            if self.stopping or time.monotonic() - t0 >= self.c["DRIVER_WAIT"]:
                why = []
                if not smi:
                    why.append(f"nvidia-smi gave no cards (exit {rc}: {o.strip()[:160]})")
                if missing:
                    why.append("missing: " + "; ".join(missing))
                raise Refused(("stopped while waiting: " if self.stopping else
                               f"after {self.c['DRIVER_WAIT']} s: ") + "; ".join(why))
            self.sleep(min(5.0, max(0.5, self.c["DRIVER_WAIT"] - (time.monotonic() - t0))))

    def checks(self):
        self.log(f"1. the driver and the files (up to {self.c['DRIVER_WAIT']} s)")
        self.extend(self.c["DRIVER_WAIT"] + 300)
        smi = self.wait_ready()
        self.log(f"   {len(smi)} cards, driver {' '.join(sorted({c['driver'] for c in smi.values()}))}")
        self.log(f"2. the profile {self.c['PROFILE']} against the cards present")
        try:
            with open(self.c["PROFILE"]) as f:
                self.prof = json.load(f)
        except (OSError, ValueError) as e:
            raise Refused(f"the profile {self.c['PROFILE']} does not read: {e}")
        self.st["profile_md5"] = md5_of(self.c["PROFILE"])
        bad = match_cards(self.prof, smi, self.c["DRIVER"])
        if bad:
            raise Refused("the cards present do not match the profile: " + "; ".join(bad))
        if self.c["GPU_SET_ID"]:
            rc, o = self.run(SMI_UUID, LIMITS["smi"], "nvidia-smi", env=self.env)
            have = gpu_set_id(o) if rc == 0 else ""
            if have != self.c["GPU_SET_ID"]:
                raise Refused(f"these cards are not the set the config was made on: GPU_SET_ID {self.c['GPU_SET_ID']}"
                              f", this rig's {have or 'unreadable'} (another rig, or a card changed: the profile is "
                              "that rig's)")
            self.log(f"   GPU_SET_ID {have}: the cards the config was made on")
        else:
            self.log("   GPU_SET_ID is not set: the profile is checked by PCI address, name and VBIOS, which "
                     "identical rigs share")
        self.cards = sorted(profile_cards(self.prof))
        given = self.extra.get("CARDS", "").split()
        if given and sorted(int(g) for g in given if g.isdigit()) != self.cards:
            raise Refused(f"CARDS={self.extra['CARDS']} in the config is not the profile's cards {self.cards}")
        shutil.copy(self.c["PROFILE"], os.path.join(self.work, "profile.json"))
        self.log(f"   cards {self.cards} match (PCI address, name, VBIOS); profile md5 {self.st['profile_md5']}")
        if watchdog_pids(self.marker()):
            raise Refused(f"a watchdog with this work folder's marker already runs ({self.marker()})")
        self.log("3. the kit's start check")
        rc, o = self.kit("start-check", "start-check")
        if rc != 0:
            raise Refused("start check: " + " ".join(o.split())[:400])

    # -- step 3: the watchdog, apply, the model check, the read-back
    def own(self, on):
        flag = self.c["STATE"] + ".owned"
        if on:
            with open(flag, "w") as f:
                json.dump({"work": self.c["WORK"], "cards": self.cards}, f)
        elif os.path.exists(flag):
            os.remove(flag)

    def watchdog_start(self):
        self.log("4. the watchdog")
        rc, o = self.kit("watchdog start", "watchdog", "start")
        pids = watchdog_pids(self.marker())
        if rc != 0 or not pids:
            raise Failed(f"the watchdog did not start (exit {rc}): {' '.join(o.split())[:200]}")
        self.st["watchdog"] = f"running (pid {pids[0]})"
        self.log(f"   running, pid {pids[0]}")

    def apply(self):
        lim = self.c["APPLY_LIMIT"]
        self.log(f"5. apply {self.c['PROFILE']} (at most {lim} s)")
        self.state(phase="applying", reason="the kit's apply: every card set, read back and checked")
        self.extend(lim + GRACE)
        rc, o = self.run(self.climb("run-apply", os.path.join(self.work, "profile.json"),
                                    "--rounds", self.c["APPLY_ROUNDS"]), lim, "apply",
                         out=os.path.join(self.work, "apply.console"), grace=GRACE)
        done = ""
        try:
            with open(self.marker("APPLY_DONE")) as f:
                done = f.read().strip()
        except OSError:
            pass
        try:
            with open(os.path.join(self.work, "out", "climb.log")) as f:
                kit_log = f.read().splitlines()
        except OSError:
            kit_log = []
        for ln in kit_log:
            self.log(f"  apply: {ln}")
        if rc == 124:
            self.log(f"  apply: {o.strip().splitlines()[-1] if o.strip() else 'stopped'}")
        if self.stopping:
            raise Failed("stopped during apply")
        if rc == 124:
            raise Failed(f"apply had no result after {lim} s (APPLY_LIMIT) and was ended; APPLY_DONE "
                         f"'{done or 'none'}'")
        if rc != 0 or done != "exit 0":
            last = [ln for ln in kit_log if re.search(r"\b(FAIL|REFUSED|RESET FAILED|STOP)\b", ln)]
            raise Failed(f"apply exit {rc}, APPLY_DONE '{done or 'none'}'" + (f": {last[-1][20:]}" if last else ""))

    def model_check(self):
        c = self.c
        lim = c["MODEL_LIMIT"]
        out = os.path.join(self.work, "model")
        os.makedirs(out, exist_ok=True)
        shutil.copy(os.path.join(c["MODEL_REF"], "greedy-stock1.sha"), os.path.join(out, "greedy-stock1.sha"))
        with open(os.path.join(out, "greedy-stock1.sha")) as f:
            ref = f.read().strip()
        self.log(f"6. the model check: {c['MODEL_ROUNDS']} rounds of {os.path.basename(c['MODEL'])} against the "
                 f"stored stock hash {ref} (at most {lim} s)")
        self.state(phase="model check", reason=f"the served model's whole-memory check against {ref}")
        self.extend(lim + GRACE)
        e = self.kit_env()
        e.update({"TAG": "profile", "ROUNDS": str(c["MODEL_ROUNDS"]), "MINUTES": "0", "BIN": c["MODEL_BIN"],
                  "MODEL": c["MODEL"], "SETTINGS": c["SETTINGS"], "OUT": out, "PORT": str(c["MODEL_PORT"]),
                  "GREEDY": self.kit_files["GREEDY"]})
        rc, o = self.run(["bash", os.path.join(c["KIT"], "model_check.sh")], lim, "model check", env=e,
                         out=os.path.join(self.work, "model.console"), grace=GRACE)
        try:
            with open(os.path.join(out, "model.log")) as f:
                for ln in f.read().splitlines():
                    self.log(f"  model check: {ln}")
        except OSError:
            pass
        if self.stopping:
            raise Failed("stopped during the model check")
        if rc != 0:
            self.st["model_check"] = f"failed (exit {rc})"
            raise Failed(f"the model check failed (exit {rc}" + (f", no result after {lim} s)" if rc == 124 else ")"))
        self.st["model_check"] = f"passed ({c['MODEL_ROUNDS']} rounds equal to the stock hash {ref})"

    def read_back(self):
        self.log("7. every offset read back against the profile")
        rc, o = self.kit("state", "state")
        offs, wd = parse_state(o)
        try:
            with open(self.kit_files["UNITS"]) as f:
                raw = json.load(f).get("raw_per_mts", 1.0)
        except (OSError, ValueError):
            raw = None
        if rc != 0 or raw not in (1.0, 0.5):
            raise Failed(f"the kit's state did not read (exit {rc}, units {raw})")
        want = offsets_wanted(self.prof, raw)
        bad = [f"card {g} reads {offs.get(g)}, the profile gives {want[g]}" for g in self.cards
               if offs.get(g) != want[g]]
        if not wd:
            bad.append("the kit's state does not see the watchdog")
        if bad:
            raise Failed("read back: " + "; ".join(bad))
        by = profile_cards(self.prof)
        self.st["cards"] = " ".join(f"{g}:+{by[g].get('memory_mts') or 0}/+{by[g].get('core_mhz') or 0}"
                                    for g in self.cards)

    # -- step 4: back to stock
    def to_stock(self, reason, rc):
        self.cleaning = True
        notify("STOPPING=1")
        self.log(f"to stock: {reason}")
        self.state(phase="resetting", reason=reason)
        rr, _ = self.kit("reset-all", "reset-all")
        self.kit("watchdog stop", "watchdog", "stop")
        for pid in watchdog_pids(self.marker()):
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.kill(pid, sig)
                except OSError:
                    break
                self.sleep(3, check=False)
        left = watchdog_pids(self.marker())
        self.st["watchdog"] = f"still running (pids {left})" if left else "stopped"
        self.own(False)
        if rr != 0:
            self.state(overclocked="unknown", phase="reset failed",
                       reason=f"{reason}; RESET FAILED: a card does not read back 0 (WORK/out/climb.log). Power the "
                              "rig off at the console")
            self.log("RESET FAILED: power the rig off at the console")
            return 3
        self.state(overclocked="no", phase="stock", since="", cards="",
                   reason=f"{reason}; every offset of every card is 0 (a card whose offset went down runs slower "
                          "until the next boot)")
        self.log("every offset of every card is 0; the watchdog is stopped")
        return rc

    # -- step 5: held
    def hold(self):
        poll, cap = self.c["POLL"], self.c["WD_LOG_MAX_MB"] * 1024 * 1024
        while True:
            self.sleep(poll)
            if self.stopping:
                return self.to_stock("the unit was stopped", 0)
            trip = self.read_lines(self.marker())
            if trip:
                return self.to_stock(f"the watchdog tripped: {trip[-1][:200]}", 1)
            if not watchdog_pids(self.marker()):
                return self.to_stock("the watchdog ended", 1)
            trips = self.read_lines(self.marker("CARD_TRIPS"))
            if len(trips) > self.trips_seen:
                for ln in trips[self.trips_seen:]:
                    self.log(f"card trip: {ln[:200]}")
                self.trips_seen = len(trips)
                at0 = sorted({int(m) for ln in trips for m in re.findall(r"CARD (\d+) TRIP", ln)})
                self.state(overclocked="partly", phase="overclocked, a card tripped", card_trips=len(trips),
                           reason=f"the watchdog set card(s) {' '.join(str(g) for g in at0)} to 0 on an Xid that "
                                  "named them; the other cards keep the profile's offsets")
            self.rotate(os.path.join(self.work, "wd.csv"), cap)

    def rotate(self, path, cap):
        """The watchdog's sample log (two rows a second) kept under WD_LOG_MAX_MB: the older half goes to wd.csv.1.
        The watchdog appends (O_APPEND), so truncating in place is safe."""
        try:
            if cap <= 0 or os.path.getsize(path) <= cap:
                return
            with open(path) as f:
                head = [ln for ln in (f.readline() for _ in range(4)) if ln.startswith("#") or ln.startswith("time")]
            shutil.copy(path, path + ".1")
            os.truncate(path, 0)
            with open(path, "a") as f:
                f.write("".join(head))
            self.log(f"the watchdog's log passed {cap / 1048576:.0f} MiB: moved to wd.csv.1")
        except OSError as e:
            self.log(f"the watchdog's log: {e}")

    @staticmethod
    def read_lines(path):
        try:
            with open(path, errors="replace") as f:
                return [ln for ln in f.read().splitlines() if ln.strip()]
        except OSError:
            return []

    def sleep(self, secs, check=True):
        t1 = time.monotonic() + secs
        while time.monotonic() < t1 and not (check and self.stopping):
            time.sleep(min(0.25, max(0.0, t1 - time.monotonic())))

    def on_signal(self, signum, frame):
        if not self.cleaning:
            self.stopping = True

    # -- the commands
    def lock(self):
        os.makedirs(os.path.dirname(self.c["STATE"]), exist_ok=True)
        self.lockf = open(self.c["STATE"] + ".lock", "w")
        try:
            fcntl.flock(self.lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Refused("another clocks_at_boot.py runs (its lock is held)")

    def start(self):
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, self.on_signal)
        try:
            self.load()
        except Refused as e:
            self.state(overclocked="no", phase="refused", reason=str(e))
            self.log(f"REFUSED: {e}")
            return 1
        try:
            self.lock()
        except Refused as e:
            self.log(f"REFUSED: {e}")
            return 1
        self.st["profile"] = self.c["PROFILE"]
        self.log(f"miner-llama clocks: config {self.config}, kit {self.c['KIT']}, boot {self.st['boot']}")
        try:
            if os.path.exists(self.c["STATE"] + ".owned"):
                raise Refused("an earlier start this boot set offsets and did not reset them: run stop-post first")
            if watchdog_pids(self.marker()):
                raise Refused(f"a watchdog with the work folder's marker already runs ({self.marker()})")
            self.fresh_work()
            self.state(overclocked="no", phase="checking", reason="checks before any offset")
            self.checks()
        except Refused as e:
            self.state(overclocked="no", phase="refused", reason=f"{e}; no offset was touched")
            self.log(f"REFUSED: {e}. No offset was touched.")
            return 1
        except Exception as e:   # noqa: BLE001  (OSError, or a bug: nothing was set, so nothing is reset)
            self.state(overclocked="no", phase="refused", reason=f"{e!r}; no offset was touched")
            self.log(f"REFUSED: {e!r}. No offset was touched.")
            return 1
        if self.stopping:
            self.state(overclocked="no", phase="stock", reason="stopped before any offset")
            return 0
        self.own(True)
        try:
            self.watchdog_start()
            self.apply()
            if self.c["MODEL_CHECK"]:
                self.model_check()
            self.read_back()
        except Failed as e:
            return self.to_stock(str(e), 0 if self.stopping else 1)
        except Exception as e:   # noqa: BLE001  (a bug must still end at stock)
            return self.to_stock(f"an error in clocks_at_boot.py: {e!r}", 1)
        if self.stopping:
            return self.to_stock("the unit was stopped", 0)
        self.state(overclocked="yes", phase="overclocked", since=time.strftime("%Y-%m-%d %H:%M:%S"),
                   reason="apply passed" + (" and the model check passed" if self.c["MODEL_CHECK"] else
                                            "; the model check is off") + "; the watchdog holds the cards")
        self.log(f"overclocked: cards {self.st['cards']} (memory MT/s / core MHz); the watchdog holds them")
        notify("READY=1")
        try:
            return self.hold()
        except Exception as e:   # noqa: BLE001
            return self.to_stock(f"an error in clocks_at_boot.py while holding: {e!r}", 1)

    def stop_post(self):
        try:
            self.load()
        except Refused as e:
            self.log(f"stop-post: {e}")
            self.resolve_kit()
        flag = self.c["STATE"] + ".owned"
        if not os.path.exists(flag):
            self.log("stop-post: nothing to do (the main process reset the cards, or set none)")
            return 0
        try:
            with open(flag) as f:
                own = json.load(f)
            self.c["WORK"], self.cards = own["work"], own["cards"]
        except (OSError, ValueError, KeyError) as e:
            self.log(f"stop-post: the owned flag does not read ({e}); resetting with the config's WORK and cards")
        self.work = self.c["WORK"] if os.path.isdir(self.c["WORK"]) else None
        try:
            with open(self.c["STATE"]) as f:
                for ln in f.read().splitlines():
                    k, _, v = ln.partition(": ")
                    if k in self.st:
                        self.st[k] = v
        except OSError:
            pass
        return self.to_stock("the service's main process ended without its reset", 1)

    def gpu_set(self):
        rc, o = self.run(SMI_UUID, LIMITS["smi"], "nvidia-smi", env=self.env)
        have = gpu_set_id(o) if rc == 0 else ""
        if not have:
            print(f"nvidia-smi gave no cards (exit {rc}): {o.strip()[:200]}", file=sys.stderr)
            return 1
        print(have)
        return 0

    def status(self):
        self.c["STATE"] = read_config(self.config)[0].get("STATE", DEFAULTS["STATE"])
        try:
            with open(self.c["STATE"]) as f:
                sys.stdout.write(f.read())
            return 0
        except OSError:
            print(f"no state file {self.c['STATE']}: the service has not run since this boot")
            return 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="/etc/miner-llama/clocks.env")
    p.add_argument("cmd", choices=("start", "stop-post", "status", "gpu-set-id"))
    a = p.parse_args(argv)
    s = Service(a.config)
    return {"start": s.start, "stop-post": s.stop_post, "status": s.status, "gpu-set-id": s.gpu_set}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
