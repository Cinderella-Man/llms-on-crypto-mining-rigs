#!/usr/bin/env python3
"""R200's soak: the profile `climb.py apply` set, held for MINUTES with all cards loaded at once.

  soak.py [--minutes 30] [--limit 45]          (run detached, as root, in the kit folder, after `apply` passed)

Needs: `apply` passed on this boot (WORK/APPLY_DONE says "exit 0"), the climb kit's watchdog running, each card's
stock hash from `apply`'s stock round (WORK/out/cardN/greedy-stock.sha). The offsets are left as `apply` set them.

Starts one llama-server per card (the kit's server settings, one after another), then runs rounds until MINUTES have
passed: in each round every card's greedy check (greedy_check.py: two prompts, 256 tokens each, temperature 0, every
token's top-10 probabilities hashed) at once, each against that card's own stock hash and under LIMIT seconds.
Writes WORK/soak/soak.csv (one row per card and round), WORK/soak/soak.log and, at the end, WORK/SOAK_DONE.

A check that differs, fails, or runs past LIMIT, a watchdog trip or a one-card trip, or the watchdog no longer running:
every offset of every card is set to 0 (climb.py reset-all, read back), the servers stopped, SOAK_DONE "exit 1 ...".
A STOP file in WORK ends the soak after the running round with the offsets left as they are. At the end of a clean
soak the servers are stopped and the offsets stay set while the watchdog runs: SOAK_DONE "exit 0 ...".
With --keep-going (a stock control, every offset 0) a check that differs is counted and the soak goes on; SOAK_DONE
says "exit 0 rounds N differs 0" only if no check differed.
"""
import argparse
import csv
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--limit", type=float, default=45)
    ap.add_argument("--work", default=os.environ.get("WORK", "/var/tmp/climb"))
    ap.add_argument("--bin", default=os.environ.get("BIN", "/opt/llama-builds/bin"))
    ap.add_argument("--check-model", default=os.environ.get("CHECK_MODEL", ""))
    ap.add_argument("--cards", default=os.environ.get("CARDS", "0 1 2 3 4 5 6 7 8"))
    ap.add_argument("--port-base", type=int, default=8200)
    ap.add_argument("--server-args", default="-ngl 99 -fa on -c 1024 -np 1 -t 1 --no-warmup")
    ap.add_argument("--greedy-n", type=int, default=256)
    ap.add_argument("--keep-going", action="store_true",
                    help="a stock control (every offset 0): a check that differs is counted and the soak goes on")
    a = ap.parse_args()
    work, cards = a.work, [int(v) for v in a.cards.split()]
    sd = os.path.join(work, "soak")
    os.makedirs(sd, exist_ok=True)
    done = os.path.join(work, "SOAK_DONE")
    if os.path.exists(done):
        os.remove(done)
    servers, result = {}, "exit 1 not started"

    def size(name):
        p = os.path.join(work, name)
        return os.path.getsize(p) if os.path.exists(p) else 0
    base = {n: size(n) for n in ("TRIPPED", "CARD_TRIPS")}   # trips before the soak (none in R200) do not count

    def on_signal(signum, frame):
        raise RuntimeError(f"signal {signum}")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def fail_all(why):
        log(work, f"FAIL: {why}; every offset of every card to 0")
        stop(list(servers.values()))
        rc = subprocess.run(["python3", os.path.join(HERE, "climb.py"), "--work", work, "reset-all"],
                            capture_output=True, text=True)
        log(work, "reset-all: " + (rc.stdout + rc.stderr).strip().replace("\n", " | ")[-600:])
        return f"exit 1 {why}"

    try:
        with open(os.path.join(work, "APPLY_DONE")) as f:
            if f.read().strip() != "exit 0":
                raise SystemExit("apply did not pass on this boot: no soak")
        if not watchdog_running(work):
            raise SystemExit("the watchdog is not running: no soak")
        for g in cards:
            d = os.path.join(sd, f"card{g}")
            os.makedirs(d, exist_ok=True)
            shutil.copy(os.path.join(work, "out", f"card{g}", "greedy-stock.sha"), os.path.join(d, "greedy-stock.sha"))
        refs = {g: open(os.path.join(sd, f"card{g}", "greedy-stock.sha")).read().strip() for g in cards}
        log(work, f"soak: cards {cards}, {a.minutes} min, limit {a.limit} s a check, stock hashes "
                  + ", ".join(f"{g}:{h}" for g, h in refs.items()))
        for g in cards:   # one server per card, one after another, at the profile's offsets
            cmd = [os.path.join(a.bin, "llama-server"), "-m", a.check_model] + shlex.split(a.server_args) + \
                ["--host", "127.0.0.1", "--port", str(a.port_base + g)]
            env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(g))
            t0 = time.time()
            servers[g] = subprocess.Popen(cmd, env=env, stdout=open(os.path.join(sd, f"card{g}", "server.log"), "w"),
                                          stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            write_pids(work, g, [servers[g].pid])
            while not health(a.port_base + g):
                time.sleep(3)
                if servers[g].poll() is not None or time.time() - t0 > 600:
                    result = fail_all(f"the server on card {g} did not come up")
                    return 1
            log(work, f"server: card {g} ready in {time.time() - t0:.0f} s")
        passes = {g: 0 for g in cards}
        differs = {g: 0 for g in cards}
        t_end = time.time() + a.minutes * 60
        rnd = 0
        with open(os.path.join(sd, "soak.csv"), "a", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["time", "round", "card", "hash", "seconds", "verdict", "reason"])
            while time.time() < t_end:
                rnd += 1
                if size("TRIPPED") > base["TRIPPED"]:
                    result = fail_all("the watchdog tripped")
                    return 1
                if size("CARD_TRIPS") > base["CARD_TRIPS"]:
                    result = fail_all("a one-card trip: " + open(os.path.join(work, "CARD_TRIPS")).read().strip()[-200:])
                    return 1
                if not watchdog_running(work):
                    result = fail_all("the watchdog is no longer running")
                    return 1
                if os.path.exists(os.path.join(work, "STOP")):
                    log(work, f"STOP file: the soak ends before round {rnd}; offsets left as they are")
                    rnd -= 1
                    break
                procs, t0 = {}, time.time()
                for g in cards:
                    d = os.path.join(sd, f"card{g}")
                    procs[g] = subprocess.Popen(
                        ["python3", os.path.join(HERE, "greedy_check.py"), "--url",
                         f"http://127.0.0.1:{a.port_base + g}", "--out", d, "--tag", f"soak{rnd}", "--ref", "stock",
                         "--n", str(a.greedy_n)], stdout=open(os.path.join(d, f"greedy-soak{rnd}.out"), "w"),
                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                    write_pids(work, g, [servers[g].pid, procs[g].pid])
                bad = []
                for g in cards:
                    left = max(0.1, a.limit - (time.time() - t0))
                    try:
                        rc = procs[g].wait(timeout=left)
                        why = {0: "", 1: "differs from stock", 3: "request failed"}.get(rc, f"exit {rc}")
                    except subprocess.TimeoutExpired:
                        stop([procs[g]])
                        rc, why = -1, f"longer than {a.limit:.0f} s"
                    el = time.time() - t0
                    d = os.path.join(sd, f"card{g}")
                    sha = os.path.join(d, f"greedy-soak{rnd}.sha")
                    h = open(sha).read().strip() if os.path.exists(sha) else ""
                    js = os.path.join(d, f"greedy-soak{rnd}.json")
                    if rc == 0:
                        passes[g] += 1
                        if os.path.exists(js):
                            os.remove(js)   # a pass is its hash; keep only a differing check's output
                    elif a.keep_going and rc == 1:
                        differs[g] += 1
                        log(work, f"card {g} round {rnd}: differs from stock (counted; --keep-going)")
                    else:
                        bad.append(f"card {g} round {rnd}: {why}")
                    w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), rnd, g, h, f"{el:.1f}",
                                "PASS" if rc == 0 else "FAIL", why])
                    write_pids(work, g, [servers[g].pid])
                fcsv.flush()
                if bad:
                    result = fail_all("; ".join(bad))
                    return 1
                if rnd % 5 == 0:
                    log(work, f"round {rnd}: passes per card " + " ".join(f"{g}:{passes[g]}" for g in cards)
                        + (" | differs per card " + " ".join(f"{g}:{differs[g]}" for g in cards) if a.keep_going else ""))
        stop(list(servers.values()))
        for g in cards:
            write_pids(work, g, [])
        nd = sum(differs.values())
        log(work, f"soak done: {rnd} rounds; passes per card " + " ".join(f"{g}:{passes[g]}" for g in cards)
            + (f"; differs per card " + " ".join(f"{g}:{differs[g]}" for g in cards) if a.keep_going
               else ", every card passed every round; the offsets stay set while the watchdog runs"))
        result = f"exit {1 if nd else 0} rounds {rnd}" + (f" differs {nd}" if a.keep_going else "")
        return 1 if nd else 0
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
