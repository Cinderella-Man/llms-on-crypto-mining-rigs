#!/usr/bin/env python3
"""soak.py: the offsets `climb.py apply` set, held with every card loaded at once, every card checked every round.

  soak.py [--minutes 30] [--limit 45] [--keep-going] [--profile P1.json] [--ladder P2.json [P3.json ...]]
          (run detached, as root, in the kit folder, after `apply` passed)

Needs: `apply` passed on this boot (WORK/APPLY_DONE says "exit 0"), the climb kit's watchdog running, each card's
stock hash from `apply`'s stock round (WORK/out/cardN/greedy-stock.sha).

Starts one llama-server per card (the kit's server settings, one after another), then runs rounds until MINUTES have
passed: in each round every card's greedy check (greedy_check.py: two prompts, 256 tokens each, temperature 0, every
token's top-10 probabilities hashed) at once, each against that card's own stock hash and under LIMIT seconds.

Three ways to run it:

  plain          the first check that differs, fails or runs past LIMIT sets every offset of every card to 0
                 (climb.py reset-all, read back), stops the servers and ends: SOAK_DONE "exit 1 ...".
  --keep-going   every card runs the full MINUTES and each card's differing checks are counted. An error near a
                 card's edge is rare and passes (the request after it is clean again), so a count over the whole time
                 says more than the first failure, and one card's error does not end the other cards' soak. A card
                 whose check fails or runs past LIMIT, or that differs --max-differs times at an offset, is set to 0
                 alone (climb.py reset-card, read back) and leaves; the others go on. The offsets stay as they are at
                 the end: SOAK_DONE "exit 0 rounds N differs 0" only if no check differed and no card left.
  --ladder P..   implies --keep-going. Each profile is one more rung of MINUTES. Offsets only go up in a boot, so the
                 search goes up too: `apply` sets the lowest rung, and after each rung every card that has not
                 differed yet is raised to the next profile's values (read back); a card that differed stays where it
                 is. One boot gives every card its highest rung that held the full MINUTES.

Writes WORK/soak/soak.csv (one row per card and round), rungs.csv (one row per card and rung: the offsets, rounds,
passes, differing checks), soak.log, soaked.json and, at the end, WORK/SOAK_DONE. soaked.json is a profile
`climb.py apply` reads: each card at the highest rung it held clean from the first rung on, 0 for a card with none.

What still stops everything (every offset of every card to 0): a watchdog trip, the watchdog no longer running, a
card that does not read back 0 after its reset, and a crash of this script. A STOP file in WORK ends the soak after
the running round with the offsets left as they are.
"""
import argparse
import csv
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = [sys.executable] + (["-I"] if sys.flags.isolated else [])


def log(work, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(work, "soak", "soak.log"), "a") as f:
        f.write(line + "\n")


def watchdog_running(work):
    out = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True).stdout
    return any("nvml_watchdog.py" in ln and f"--marker {os.path.join(work, 'TRIPPED')}" in ln for ln in out.splitlines())


def health(port):
    return subprocess.run(["curl", "-sf", "-m", "5", f"http://127.0.0.1:{port}/health"],
                          capture_output=True).returncode == 0


def write_pids(work, g, pids):
    with open(os.path.join(work, "pids", f"{g}.pids"), "w") as f:
        f.write("".join(f"{p}\n" for p in pids if p))


def stop(procs):
    for p in procs:
        if p and p.poll() is None:
            try:
                os.killpg(p.pid, 15)
            except OSError:
                pass
    for p in procs:
        if p:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, 9)
                except OSError:
                    pass


def parse_get(text):
    """nvml_memoc.py get: 'gpu 0 MemClk vf offset: 1200 (allowed ...)' -> 1200 (None if absent)."""
    m = re.search(r"vf offset: (-?\d+)", text or "")
    return int(m.group(1)) if m else None


def card_values(profile, g):
    """(memory MT/s, core MHz, core steps) of card g in a profile, None if the profile lacks the card."""
    for c in profile.get("cards", []):
        if c.get("card") == g:
            return int(c.get("memory_mts") or 0), int(c.get("core_mhz") or 0), int(c.get("core_steps") or 0)
    return None


def ladder_errors(start, rungs, cards, mem_cap, core_cap):
    """Why a ladder cannot run: a rung that lacks a card, goes above a cap, or would step a card down.
    start: {card: (mem, core)} as set now; rungs: the profiles of rung 2 on."""
    bad, cur = [], dict(start)
    for i, prof in enumerate(rungs, 2):
        for g in cards:
            v = card_values(prof, g)
            if v is None:
                bad.append(f"rung {i} has no card {g}")
                continue
            if v[0] > mem_cap or v[1] > core_cap:
                bad.append(f"rung {i} card {g}: +{v[0]} MT/s, +{v[1]} MHz is above the caps {mem_cap}, {core_cap}")
            if v[0] < cur[g][0] or v[1] < cur[g][1]:
                bad.append(f"rung {i} card {g}: +{v[0]} MT/s, +{v[1]} MHz would step down from +{cur[g][0]}, "
                           f"+{cur[g][1]}")
            cur[g] = (max(v[0], cur[g][0]), max(v[1], cur[g][1]))
    return bad


def soaked_values(history):
    """A card's rungs, in order, as (mem, core, steps, complete, differs, left) -> the (mem, core, steps) of the
    highest rung it held clean from the first rung on; (0, 0, 0) if the first rung was not clean."""
    best = (0, 0, 0)
    for mem, core, steps, complete, differs, left in history:
        if not complete or differs or left:
            break
        best = (mem, core, steps)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=30, help="minutes a rung")
    ap.add_argument("--limit", type=float, default=45, help="seconds a check may take")
    ap.add_argument("--work", default=os.environ.get("WORK", "/var/tmp/climb"))
    ap.add_argument("--bin", default=os.environ.get("BIN", "/opt/llama-builds/bin"))
    ap.add_argument("--check-model", default=os.environ.get("CHECK_MODEL", ""))
    ap.add_argument("--cards", default=os.environ.get("CARDS", "0 1 2 3 4 5 6 7 8"))
    ap.add_argument("--port-base", type=int, default=int(os.environ.get("PORT_BASE", "8200")))
    ap.add_argument("--server-args", default=os.environ.get("SERVER_ARGS", "-ngl 99 -fa on -c 1024 -np 1 -t 1 --no-warmup"))
    ap.add_argument("--greedy-n", type=int, default=256)
    ap.add_argument("--keep-going", action="store_true",
                    help="count each card's differing checks over the full time instead of stopping at the first")
    ap.add_argument("--max-differs", type=int, default=3,
                    help="--keep-going: a card that differs this often at an offset is set to 0 alone and leaves")
    ap.add_argument("--profile", help="the profile `apply` set (the cards' names for soaked.json)")
    ap.add_argument("--ladder", nargs="+", default=[], metavar="PROFILE",
                    help="the next rungs' profiles, lowest first: clean cards go up after each rung")
    ap.add_argument("--memoc", default=os.environ.get("MEMOC", os.path.join(HERE, "nvml_memoc.py")))
    ap.add_argument("--units", default=os.environ.get("UNITS", os.path.join(HERE, "memoc_units.json")))
    ap.add_argument("--climb", default=os.environ.get("CLIMB", os.path.join(HERE, "climb.py")))
    ap.add_argument("--greedy", default=os.environ.get("GREEDY", os.path.join(HERE, "greedy_check.py")))
    ap.add_argument("--mem-cap", type=int, default=int(os.environ.get("MEM_CAP", "1400")))
    ap.add_argument("--core-cap", type=int, default=int(os.environ.get("CORE_CAP", "400")))
    a = ap.parse_args(argv)
    keep = a.keep_going or bool(a.ladder)
    work, cards = a.work, [int(v) for v in a.cards.split()]
    sd = os.path.join(work, "soak")
    os.makedirs(sd, exist_ok=True)
    os.makedirs(os.path.join(work, "pids"), exist_ok=True)
    done = os.path.join(work, "SOAK_DONE")
    if os.path.exists(done):
        os.remove(done)
    servers, result = {}, "exit 1 not started"
    left = {}                                   # card -> why it left the soak (set to 0 alone)
    offs = {}                                   # card -> (memory MT/s, core MHz, core steps) as set now
    hist = {g: [] for g in cards}               # card -> its rungs, for soaked_values()
    raw_per_mts = 1.0

    def size(name):
        p = os.path.join(work, name)
        return os.path.getsize(p) if os.path.exists(p) else 0
    base = {n: size(n) for n in ("TRIPPED", "CARD_TRIPS")}   # trips before the soak do not count

    def on_signal(signum, frame):
        raise RuntimeError(f"signal {signum}")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def run(cmd, timeout=60):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return p.returncode, p.stdout + p.stderr
        except subprocess.TimeoutExpired:
            return 124, f"timed out after {timeout} s"

    def readback(g):
        """(memory MT/s, core MHz) as the driver reports them, None for a value it does not give."""
        _, o1 = run(PY + [a.memoc, "get", str(g)])
        _, o2 = run(PY + [a.memoc, "--core", "get", str(g)])
        mem, core = parse_get(o1), parse_get(o2)
        return (None if mem is None else round(mem / raw_per_mts)), core

    def fail_all(why):
        log(work, f"FAIL: {why}; every offset of every card to 0")
        stop(list(servers.values()))
        rc = run(PY + [a.climb, "--work", work, "reset-all"], timeout=300)
        log(work, "reset-all: " + rc[1].strip().replace("\n", " | ")[-600:])
        return f"exit 1 {why}"

    def leave(g, why, reset=True):
        """One card out of the soak: its offsets to 0 and read back first, then its server stopped."""
        log(work, f"card {g} leaves the soak: {why}" + ("; its offsets to 0, the other cards go on" if reset else ""))
        rc = run(PY + [a.climb, "--work", work, "reset-card", str(g)], timeout=120) if reset else (0, "")
        if reset:
            log(work, "reset-card: " + rc[1].strip().replace("\n", " | ")[-300:])
        stop([servers.pop(g, None)])
        write_pids(work, g, [])
        left[g] = why
        if rc[0] != 0:
            raise RuntimeError(f"card {g} did not read back 0 after its reset")
        offs[g] = (0, 0, 0)

    def raise_card(g, mem, core, steps):
        """One card's offsets up to (mem, core), read back from a new process; '' or why it failed."""
        cur = offs[g]
        if mem > cur[0]:
            rc, o = run(PY + [a.memoc, "--state", a.units, "--cap", str(a.mem_cap), "set", str(g), str(mem)])
            if rc != 0:
                return f"the driver refused memory +{mem} ({o.strip()[-120:]})"
        if core > cur[1]:
            rc, o = run(PY + [a.memoc, "--core", "set", str(g), str(core)])
            if rc != 0:
                return f"the driver refused core +{core} ({o.strip()[-120:]})"
        want = (max(mem, cur[0]), max(core, cur[1]))
        rb = readback(g)
        if rb != want:
            return f"offsets read back as {rb[0]}, {rb[1]} instead of {want[0]}, {want[1]}"
        offs[g] = (want[0], want[1], steps if core > cur[1] else cur[2])
        return ""

    try:
        with open(os.path.join(work, "APPLY_DONE")) as f:
            if f.read().strip() != "exit 0":
                raise SystemExit("apply did not pass on this boot: no soak")
        if not watchdog_running(work):
            raise SystemExit("the watchdog is not running: no soak")
        try:
            with open(a.units) as f:
                raw_per_mts = float(json.load(f).get("raw_per_mts", 1.0))
        except (OSError, ValueError):
            if a.ladder:
                raise SystemExit(f"units file {a.units} not readable: no ladder")
        rungs = []
        for path in a.ladder:
            with open(path) as f:
                rungs.append(json.load(f))
        ident = None
        if a.profile or a.ladder:
            with open(a.profile or a.ladder[0]) as f:
                ident = json.load(f)
        for g in cards:
            rb = readback(g)
            if None in rb:
                if a.ladder:
                    raise SystemExit(f"card {g}: its offsets cannot be read ({a.memoc}): no ladder")
                rb = (rb[0] or 0, rb[1] or 0)
            first = card_values(ident, g) if ident and a.profile else None
            offs[g] = (rb[0], rb[1], first[2] if first and first[1] == rb[1] else round(rb[1] / 12.66))
        bad = ladder_errors({g: offs[g][:2] for g in cards}, rungs, cards, a.mem_cap, a.core_cap)
        if bad:
            raise SystemExit("the ladder cannot run: " + "; ".join(bad))
        for g in cards:
            d = os.path.join(sd, f"card{g}")
            os.makedirs(d, exist_ok=True)
            shutil.copy(os.path.join(work, "out", f"card{g}", "greedy-stock.sha"), os.path.join(d, "greedy-stock.sha"))
        refs = {g: open(os.path.join(sd, f"card{g}", "greedy-stock.sha")).read().strip() for g in cards}
        log(work, f"soak: cards {cards}, {1 + len(rungs)} rung(s) of {a.minutes} min, limit {a.limit} s a check, "
                  + ("every differing check counted" if keep else "the first failure ends it")
                  + ", stock hashes " + ", ".join(f"{g}:{h}" for g, h in refs.items()))
        for g in cards:   # one server per card, one after another, at the offsets `apply` set
            cmd = [os.path.join(a.bin, "llama-server"), "-m", a.check_model] + shlex.split(a.server_args) + \
                ["--host", "127.0.0.1", "--port", str(a.port_base + g)]
            env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(g))
            t0 = time.time()
            servers[g] = subprocess.Popen(cmd, env=env, stdout=open(os.path.join(sd, f"card{g}", "server.log"), "w"),
                                          stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            write_pids(work, g, [servers[g].pid])
            while not health(a.port_base + g):
                time.sleep(1)
                if servers[g].poll() is not None or time.time() - t0 > 600:
                    result = fail_all(f"the server on card {g} did not come up")
                    return 1
            log(work, f"server: card {g} ready in {time.time() - t0:.0f} s")
        passes = {g: 0 for g in cards}
        differs = {g: 0 for g in cards}             # over the whole soak
        first_diff = {}                             # card -> (rung, round, mem, core)
        rnd, stopped = 0, False
        new_csv = not os.path.exists(os.path.join(sd, "soak.csv"))
        with open(os.path.join(sd, "soak.csv"), "a", newline="") as fcsv, \
                open(os.path.join(sd, "rungs.csv"), "a", newline="") as frung:
            w, wr = csv.writer(fcsv), csv.writer(frung)
            if new_csv:
                w.writerow(["time", "round", "card", "hash", "seconds", "verdict", "reason", "rung", "mem_mts",
                            "core_mhz"])
                wr.writerow(["rung", "card", "mem_mts", "core_mhz", "minutes", "rounds", "passes", "differs",
                             "first_differing_round", "complete", "left"])
            for rung in range(1, len(rungs) + 2):
                if rung > 1:   # the clean cards go up; a card that differed stays where it is
                    up, held = [], []
                    for g in cards:
                        if g in left:
                            continue
                        if differs[g]:
                            held.append(g)
                            continue
                        mem, core, steps = card_values(rungs[rung - 2], g)
                        why = raise_card(g, mem, core, steps)
                        if why:
                            leave(g, f"rung {rung}: {why}")
                        else:
                            up.append(f"{g}:+{offs[g][0]}/+{offs[g][1]}")
                    log(work, f"rung {rung}: raised (card:memory MT/s/core MHz) {' '.join(up) or 'none'}"
                              + (f"; held where they differed: {held}" if held else ""))
                act = [g for g in cards if g not in left]
                if not act:
                    log(work, f"rung {rung}: no card left")
                    break
                log(work, f"rung {rung}: {a.minutes} min on cards {act} at "
                          + " ".join(f"{g}:+{offs[g][0]}/+{offs[g][1]}" for g in act))
                r_rounds = {g: 0 for g in cards}
                r_pass = {g: 0 for g in cards}
                r_diff = {g: 0 for g in cards}
                r_first = {}
                r_offs = {g: offs[g] for g in cards}
                t_end = time.time() + a.minutes * 60
                while time.time() < t_end:
                    act = [g for g in cards if g not in left]
                    if not act:
                        break
                    rnd += 1
                    if size("TRIPPED") > base["TRIPPED"]:
                        result = fail_all("the watchdog tripped")
                        return 1
                    if size("CARD_TRIPS") > base["CARD_TRIPS"]:
                        with open(os.path.join(work, "CARD_TRIPS"), "rb") as f:
                            f.seek(base["CARD_TRIPS"])
                            text = f.read().decode(errors="replace")   # only the lines since the last look
                        base["CARD_TRIPS"] += len(text.encode())
                        if not keep:
                            result = fail_all("a one-card trip: " + text.strip()[-200:])
                            return 1
                        for m in re.finditer(r"CARD (\d+) TRIP", text):   # the watchdog set that card to 0 itself
                            if int(m.group(1)) in act:
                                leave(int(m.group(1)), "the watchdog tripped on this card")
                        act = [g for g in cards if g not in left]
                        if not act:
                            break
                    if not watchdog_running(work):
                        result = fail_all("the watchdog is no longer running")
                        return 1
                    if os.path.exists(os.path.join(work, "STOP")):
                        log(work, f"STOP file: the soak ends before round {rnd}; offsets left as they are")
                        rnd -= 1
                        stopped = True
                        break
                    procs, t0 = {}, time.time()
                    for g in act:
                        d = os.path.join(sd, f"card{g}")
                        procs[g] = subprocess.Popen(
                            PY + [a.greedy, "--url", f"http://127.0.0.1:{a.port_base + g}", "--out", d, "--tag",
                                  f"soak{rnd}", "--ref", "stock", "--n", str(a.greedy_n)],
                            stdout=open(os.path.join(d, f"greedy-soak{rnd}.out"), "w"),
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                        write_pids(work, g, [servers[g].pid, procs[g].pid])
                    bad, out = [], []
                    for g in act:
                        remaining = max(0.1, a.limit - (time.time() - t0))
                        try:
                            rc = procs[g].wait(timeout=remaining)
                            why = {0: "", 1: "differs from stock", 3: "request failed"}.get(rc, f"exit {rc}")
                        except subprocess.TimeoutExpired:
                            rc, why = -1, f"longer than {a.limit:.0f} s"   # stopped below, after the card's reset
                        el = time.time() - t0
                        d = os.path.join(sd, f"card{g}")
                        sha = os.path.join(d, f"greedy-soak{rnd}.sha")
                        h = open(sha).read().strip() if os.path.exists(sha) else ""
                        js = os.path.join(d, f"greedy-soak{rnd}.json")
                        r_rounds[g] += 1
                        if rc == 0:
                            passes[g] += 1
                            r_pass[g] += 1
                            if os.path.exists(js):
                                os.remove(js)   # a pass is its hash; keep only a differing check's output
                        elif keep and rc == 1:
                            differs[g] += 1
                            r_diff[g] += 1
                            r_first.setdefault(g, rnd)
                            first_diff.setdefault(g, (rung, rnd, offs[g][0], offs[g][1]))
                            log(work, f"card {g} round {rnd} (rung {rung}, +{offs[g][0]} MT/s, +{offs[g][1]} MHz): "
                                      f"differs from stock ({differs[g]} so far)")
                            if differs[g] > a.max_differs and os.path.exists(js):
                                os.remove(js)   # the first few differing outputs are kept, not every one
                            if any(offs[g][:2]) and differs[g] >= a.max_differs:
                                out.append((g, f"{differs[g]} checks differed"))
                        elif keep:
                            out.append((g, f"round {rnd}: {why}"))
                        else:
                            bad.append(f"card {g} round {rnd}: {why}")
                        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), rnd, g, h, f"{el:.1f}",
                                    "PASS" if rc == 0 else "FAIL", why, rung, offs[g][0], offs[g][1]])
                        if rc == 0 or rc == 1:
                            write_pids(work, g, [servers[g].pid])
                    fcsv.flush()
                    if bad:
                        stop([p for p in procs.values() if p.poll() is None])
                        result = fail_all("; ".join(bad))
                        return 1
                    for g, why in out:   # the card to 0 first (a hung kernel may then finish), then its processes
                        leave(g, why)
                        stop([procs[g]])
                    if rnd % 5 == 0:
                        log(work, f"round {rnd}: passes per card " + " ".join(f"{g}:{passes[g]}" for g in cards)
                            + (" | differs per card " + " ".join(f"{g}:{differs[g]}" for g in cards) if keep else ""))
                complete = not stopped and time.time() >= t_end
                for g in cards:
                    if r_rounds[g] == 0 and g in left:
                        continue
                    mem, core, steps = r_offs[g]
                    hist[g].append((mem, core, steps, complete, r_diff[g], g in left))
                    wr.writerow([rung, g, mem, core, a.minutes, r_rounds[g], r_pass[g], r_diff[g], r_first.get(g, ""),
                                 "yes" if complete and g not in left else "no", left.get(g, "")])
                frung.flush()
                log(work, f"rung {rung} done: differs per card " + " ".join(f"{g}:{r_diff[g]}" for g in cards)
                    + (f"; left: {sorted(left)}" if left else ""))
                if stopped:
                    break
        stop(list(servers.values()))
        for g in cards:
            write_pids(work, g, [])
        nd = sum(differs.values())
        if keep:
            held = {g: soaked_values(hist[g]) for g in cards}
            for g in cards:
                fd = first_diff.get(g)
                log(work, f"card {g}: held +{held[g][0]} MT/s, +{held[g][1]} MHz clean"
                    + (f"; first differed in rung {fd[0]}, round {fd[1]}, at +{fd[2]} MT/s, +{fd[3]} MHz "
                       f"({differs[g]} checks differed)" if fd else "; never differed")
                    + (f"; left: {left[g]}" if g in left else ""))
            if ident:
                by = {c.get("card"): c for c in ident.get("cards", [])}
                soaked = {"what": "P106 clock offsets that held a soak, from bench/mem_offset/soak.py; "
                                  "`climb.py apply` reads it",
                          "date": time.strftime("%Y-%m-%d"), "complete": True, "climb_boot": None,
                          "rungs": 1 + len(rungs), "minutes_a_rung": a.minutes,
                          "each_card": "memory_mts and core_mhz are the highest rung the card held for the full "
                                       "minutes with no differing check, from the first rung on; 0 if none",
                          "cards": []}
                for g in cards:
                    c = dict(by.get(g, {"card": g}))
                    fd = first_diff.get(g)
                    c.update({"memory_mts": held[g][0], "core_mhz": held[g][1], "core_steps": held[g][2],
                              "soak": {"rungs": [{"memory_mts": h[0], "core_mhz": h[1], "complete": h[3],
                                                  "differing_checks": h[4]} for h in hist[g]],
                                       "first_differed": {"rung": fd[0], "round": fd[1], "memory_mts": fd[2],
                                                          "core_mhz": fd[3]} if fd else None,
                                       "left": left.get(g)}})
                    for k in ("expected_peak_mhz", "memory", "core", "failure", "needs_reboot_before_use"):
                        c.pop(k, None)
                    soaked["cards"].append(c)
                with open(os.path.join(sd, "soaked.json"), "w") as f:
                    json.dump(soaked, f, indent=1)
                log(work, f"soaked: {os.path.join(sd, 'soaked.json')}")
        log(work, f"soak done: {rnd} rounds; passes per card " + " ".join(f"{g}:{passes[g]}" for g in cards)
            + (f"; differs per card " + " ".join(f"{g}:{differs[g]}" for g in cards)
               + (f"; left: {sorted(left)}" if left else "") + "; the offsets stay as they are while the watchdog runs"
               if keep else ", every card passed every round; the offsets stay set while the watchdog runs"))
        ok = not nd and not left
        result = f"exit {0 if ok else 1} rounds {rnd}" + (f" differs {nd}" if keep else "") \
            + (f" left {','.join(str(g) for g in sorted(left))}" if left else "")
        return 0 if ok else 1
    except SystemExit as e:
        log(work, f"REFUSED: {e}")
        result = f"exit 1 refused: {e}"
        return 1
    except Exception as e:  # noqa: BLE001 -- any crash of the driver: every card to 0
        result = fail_all(f"the soak driver crashed: {e!r}")
        return 1
    finally:
        with open(done, "w") as f:
            f.write(result + "\n")


if __name__ == "__main__":
    sys.exit(main())
