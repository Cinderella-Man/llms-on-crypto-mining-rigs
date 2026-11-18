#!/usr/bin/env python3
"""Tests of soak.py on any Linux machine: no GPU, no NVML, no root; nothing is set or run on a card.

  python3 -I bench/mem_offset/test/test_soak.py [-v]

soak.py runs for real against a stand-in rig made in a temporary folder: a `llama-server` that answers /health and
/completion (and, from a chosen memory offset on, answers wrongly once, every time, or exits), an `nvml_memoc.py` that
keeps each card's offsets in a JSON file, an `nvidia-smi` that lists the cards, and a process named like the watchdog.
The real greedy_check.py and the real `climb.py reset-all` and `reset-card` run against them.

- the ladder: clean cards go up a rung at a time (memory, then the core); a card that differs once stays where it is;
  a card that differs three times, and a card whose server dies, are set to 0 alone while the others go on;
  soaked.json holds each card's highest clean rung, `climb.py apply` reads it, and so does
  `climb.py --from-profile` (memory straight to the soaked values, the core climb for the cards without one);
- plain mode: the first differing check sets every card to 0;
- a stock control with --keep-going: no difference, exit 0;
- a ladder that would step a card down, or go above a cap, is refused before any server starts.
"""
import contextlib
import csv
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
sys.path.insert(0, KIT)
import climb  # noqa: E402
import soak  # noqa: E402

CARDS = [0, 2, 5, 7, 8]

FAKE_SERVER = r'''#!/usr/bin/env python3
import http.server, json, os, sys
g = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
port = int(sys.argv[sys.argv.index("--port") + 1])
fault = json.loads(os.environ.get("FAKE_FAULTS", "{}")).get(g)
seen = {"n": 0}
def mem():
    with open(os.environ["FAKE_STATE"]) as f:
        return json.load(f).get(g, [0, 0])[0]
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
        if fault and mem() >= fault["mem_min"]:
            seen["n"] += 1
            if fault.get("die"):
                os._exit(1)
            if fault.get("every") or seen["n"] == 1:
                text += " (wrong)"
        self.send({"content": text, "completion_probabilities": [{"token": "a", "logprob": -0.05}],
                   "timings": {"prompt_per_second": 50.0, "predicted_per_second": 30.0, "predicted_n": req["n_predict"]}})
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
'''

FAKE_MEMOC = r'''#!/usr/bin/env python3
import json, os, sys
args = [a for a in sys.argv[1:]]
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
with open(path + ".tmp", "w") as f:
    json.dump(st, f)
os.replace(path + ".tmp", path)
print(f"gpu {g} {name}: " + (f"set {v[d]}" if cmd == "set" else "offset 0"))
'''

FAKE_SMI = '''#!/bin/sh
i=0
for bus in 03 04 05 06 07 0B 0C 0D 0E; do
  echo "$i, 00000000:$bus:00.0, NVIDIA P106-090, 86.06.5B.00.0C, 0, 75.00, 75.00, P8, 580.178.04"
  i=$((i + 1))
done
'''


def profile(values):
    """A profile as climb.py writes it, with the simulated rig's cards: {card: (memory, core, core steps)}."""
    return {"what": "test", "complete": True, "climb_boot": "other-boot",
            "cards": [{"card": g, "pci": climb.SCENARIO["pci"][g], "name": "NVIDIA P106-090", "vbios": "86.06.5B.00.0C",
                       "stock_peak_mhz": 1898, "memory_mts": v[0], "core_mhz": v[1], "core_steps": v[2]}
                      for g, v in sorted(values.items())]}


class Rig:
    """The stand-in rig in a temporary folder, as it is after `climb.py apply` passed."""

    def __init__(self, start, faults=None):
        self.tmp = tempfile.mkdtemp(prefix="soak-test-")
        self.work = os.path.join(self.tmp, "work")
        self.state = os.path.join(self.tmp, "state.json")
        self.ledger = os.path.join(self.tmp, "ledger.log")
        self.port = random.randint(20000, 50000)
        for d in ("bin", "path", os.path.join("work", "out"), os.path.join("work", "pids")):
            os.makedirs(os.path.join(self.tmp, d))
        for name, text in (("bin/llama-server", FAKE_SERVER), ("nvml_memoc.py", FAKE_MEMOC),
                           ("path/nvidia-smi", FAKE_SMI), ("nvml_watchdog.py", "import time\ntime.sleep(600)\n")):
            with open(os.path.join(self.tmp, name), "w") as f:
                f.write(text)
            os.chmod(os.path.join(self.tmp, name), 0o755)
        with open(self.state, "w") as f:
            json.dump({str(g): [0, 0] for g in range(9)}, f)
        self.env = dict(os.environ, WORK=self.work, BIN=os.path.join(self.tmp, "bin"), CHECK_MODEL="check.gguf",
                        MEMOC=os.path.join(self.tmp, "nvml_memoc.py"), LEDGER=self.ledger, FAKE_STATE=self.state,
                        FAKE_FAULTS=json.dumps({str(g): f for g, f in (faults or {}).items()}),
                        CARDS=" ".join(str(g) for g in CARDS), PATH=os.path.join(self.tmp, "path") + ":" + os.environ["PATH"])
        self.procs = [subprocess.Popen([sys.executable, os.path.join(self.tmp, "nvml_watchdog.py"), "--marker",
                                        os.path.join(self.work, "TRIPPED")])]
        # each card's stock hash, from the real greedy check against a stand-in server at offset 0
        srv = subprocess.Popen([os.path.join(self.tmp, "bin", "llama-server"), "--port", str(self.port + 20)],
                               env=dict(self.env, FAKE_FAULTS="{}"))
        self.procs.append(srv)
        d0 = os.path.join(self.work, "out", "stock")
        os.makedirs(d0)
        for _ in range(50):
            if soak.health(self.port + 20):
                break
            subprocess.run(["sleep", "0.1"])
        rc = subprocess.run([sys.executable, "-I", os.path.join(KIT, "greedy_check.py"), "--url",
                             f"http://127.0.0.1:{self.port + 20}", "--out", d0, "--tag", "stock", "--n", "8"],
                            capture_output=True, text=True)
        assert rc.returncode == 0, rc.stdout + rc.stderr
        srv.kill()
        for g in CARDS:
            os.makedirs(os.path.join(self.work, "out", f"card{g}"))
            shutil.copy(os.path.join(d0, "greedy-stock.sha"), os.path.join(self.work, "out", f"card{g}"))
        with open(os.path.join(self.work, "APPLY_DONE"), "w") as f:   # what `apply` leaves: the offsets set
            f.write("exit 0\n")
        st = self.offsets()
        for g, v in start.items():
            st[str(g)] = [v[0], v[1]]
        with open(self.state, "w") as f:
            json.dump(st, f)

    def offsets(self):
        with open(self.state) as f:
            return json.load(f)

    def write(self, name, values):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as f:
            json.dump(profile(values), f)
        return path

    def soak(self, *args):
        p = subprocess.run([sys.executable, "-I", os.path.join(KIT, "soak.py"), "--port-base", str(self.port),
                            "--greedy-n", "8", "--limit", "20"] + list(args), env=self.env, capture_output=True,
                           text=True, timeout=300)
        with open(os.path.join(self.work, "SOAK_DONE")) as f:
            return p.returncode, f.read().strip(), p.stdout + p.stderr

    def ledger_lines(self):
        if not os.path.exists(self.ledger):
            return []
        with open(self.ledger) as f:
            return f.read().splitlines()

    def close(self):
        for p in self.procs:
            p.kill()
            p.wait()
        shutil.rmtree(self.tmp, ignore_errors=True)


RUNG1 = {0: (1100, 0, 0), 2: (1000, 0, 0), 5: (1000, 0, 0), 7: (800, 0, 0), 8: (1100, 0, 0)}
RUNG2 = {0: (1200, 0, 0), 2: (1100, 0, 0), 5: (1100, 0, 0), 7: (900, 0, 0), 8: (1200, 0, 0)}
RUNG3 = {0: (1300, 0, 0), 2: (1200, 0, 0), 5: (1200, 0, 0), 7: (1000, 0, 0), 8: (1300, 0, 0)}
RUNG4 = {0: (1300, 256, 20), 2: (1200, 0, 0), 5: (1200, 0, 0), 7: (1000, 0, 0), 8: (1300, 231, 18)}


class Pure(unittest.TestCase):
    def test_ladder_errors(self):
        start = {g: v[:2] for g, v in RUNG1.items()}
        self.assertEqual(soak.ladder_errors(start, [profile(RUNG2), profile(RUNG3), profile(RUNG4)], CARDS, 1400, 400), [])
        down = soak.ladder_errors(start, [profile(RUNG3), profile(RUNG2)], CARDS, 1400, 400)
        self.assertEqual(len(down), 5)
        self.assertIn("would step down", down[0])
        self.assertIn("above the caps", soak.ladder_errors(start, [profile(RUNG3)], CARDS, 1200, 400)[0])
        self.assertIn("has no card 8", soak.ladder_errors(start, [profile({g: RUNG2[g] for g in (0, 2, 5, 7)})], CARDS,
                                                         1400, 400)[0])

    def test_soaked_values(self):
        clean = (1000, 0, 0, True, 0, False)
        self.assertEqual(soak.soaked_values([clean, (1100, 0, 0, True, 0, False), (1200, 54, 4, True, 0, False)]),
                         (1200, 54, 4))
        self.assertEqual(soak.soaked_values([clean, (1100, 0, 0, True, 1, False), (1100, 0, 0, True, 0, False)]),
                         (1000, 0, 0))          # a rung with a differing check ends it, a later clean rung does not count
        self.assertEqual(soak.soaked_values([clean, (1100, 0, 0, False, 0, False)]), (1000, 0, 0))   # a rung cut short
        self.assertEqual(soak.soaked_values([(1000, 0, 0, True, 0, True)]), (0, 0, 0))               # the card left
        self.assertEqual(soak.soaked_values([]), (0, 0, 0))


class Ladder(unittest.TestCase):
    def test_clean_cards_go_up_a_differing_card_stays_a_failing_card_leaves_alone(self):
        rig = Rig(RUNG1, {7: {"mem_min": 900}, 2: {"mem_min": 1200, "every": True}, 5: {"mem_min": 1200, "die": True}})
        try:
            p1 = rig.write("p1.json", RUNG1)
            rc, done, out = rig.soak("--minutes", "0.1", "--profile", p1, "--ladder", rig.write("p2.json", RUNG2),
                                     rig.write("p3.json", RUNG3), rig.write("p4.json", RUNG4))
            self.assertEqual(rc, 1, out)
            self.assertRegex(done, r"^exit 1 rounds \d+ differs 4 left 2,5$")
            # the clean cards at the top rung, memory then core; card 7 where it differed; cards 2 and 5 at 0
            st = rig.offsets()
            self.assertEqual([st[str(g)] for g in CARDS], [[1300, 256], [0, 0], [0, 0], [900, 0], [1300, 231]])
            led = rig.ledger_lines()
            self.assertEqual(sorted(ln.split()[4] for ln in led), ["2", "5"])       # only those two stepped down
            self.assertTrue(any("card 2 mem 1200->0" in ln for ln in led))
            with open(os.path.join(rig.work, "soak", "soaked.json")) as f:
                soaked = json.load(f)
            got = {c["card"]: (c["memory_mts"], c["core_mhz"], c["core_steps"]) for c in soaked["cards"]}
            self.assertEqual(got, {0: (1300, 256, 20), 2: (1100, 0, 0), 5: (1100, 0, 0), 7: (800, 0, 0),
                                   8: (1300, 231, 18)})
            c7 = next(c for c in soaked["cards"] if c["card"] == 7)
            self.assertEqual(c7["soak"]["first_differed"]["memory_mts"], 900)
            self.assertEqual(c7["soak"]["first_differed"]["rung"], 2)
            self.assertEqual([r["memory_mts"] for r in c7["soak"]["rungs"]], [800, 900, 900, 900])   # never raised again
            with open(os.path.join(rig.work, "soak", "rungs.csv")) as f:
                rows = list(csv.DictReader(f))
            r7 = [r for r in rows if r["card"] == "7"]
            self.assertEqual([(r["rung"], r["mem_mts"], r["differs"]) for r in r7],
                             [("1", "800", "0"), ("2", "900", "1"), ("3", "900", "0"), ("4", "900", "0")])
            r2 = [r for r in rows if r["card"] == "2"]
            self.assertEqual([(r["rung"], r["mem_mts"], r["differs"], r["complete"]) for r in r2],
                             [("1", "1000", "0", "yes"), ("2", "1100", "0", "yes"), ("3", "1200", "3", "no")])
            self.assertIn("request failed", next(r for r in rows if r["card"] == "5" and r["rung"] == "3")["left"])
            self.assertTrue(all(int(r["rounds"]) >= 2 for r in rows if r["complete"] == "yes"))
            # `climb.py apply` reads soaked.json (the simulated rig: nothing is set)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = climb.main(["--dry-run", "--work", os.path.join(rig.tmp, "apply"), "--check-model", "check.gguf",
                                 "apply", os.path.join(rig.work, "soak", "soaked.json")])
            self.assertEqual(rc, 0, buf.getvalue()[-2000:])
            self.assertIn("set 0 1300", buf.getvalue())
            self.assertIn("--core set 8 231", buf.getvalue())
            self.assertNotIn(" set 2 1200", buf.getvalue())
            # and `climb.py --from-profile` reads it: memory straight to each card's soaked value, then the core
            # climb for the cards it gives no core value (2, 5 and 7), their memory held
            buf = io.StringIO()
            fp = os.path.join(rig.tmp, "from-profile")
            with contextlib.redirect_stdout(buf):
                rc = climb.main(["--dry-run", "--work", fp, "--check-model", "check.gguf", "--ledger",
                                 os.path.join(fp, "ledger"), "--from-profile",
                                 os.path.join(rig.work, "soak", "soaked.json"), "climb"])
            self.assertEqual(rc, 0, buf.getvalue()[-2000:])
            mems = sorted(set(int(v) for v in re.findall(r"--state \S+ --cap \d+ set \d+ (\d+)", buf.getvalue())))
            self.assertEqual(mems, [800, 1100, 1300])          # each soaked value, nothing climbed
            with open(os.path.join(fp, "out", "profile.json")) as f:
                out = json.load(f)
            self.assertEqual(out["core_climb_cards"], [2, 5, 7])
            got = {c["card"]: (c["memory_mts"], c["core"]["took_part"]) for c in out["cards"]}
            self.assertEqual(got, {0: (1300, False), 2: (1100, True), 5: (1100, True), 7: (800, True), 8: (1300, False)})
            self.assertEqual([c["core_mhz"] for c in out["cards"] if c["card"] in (0, 8)], [256, 231])
            self.assertTrue(all(c["core_mhz"] > 0 for c in out["cards"] if c["card"] in (2, 5, 7)))
        finally:
            rig.close()

    def test_a_ladder_that_steps_down_is_refused_before_any_server(self):
        rig = Rig(RUNG2)
        try:
            rc, done, out = rig.soak("--minutes", "0.05", "--ladder", rig.write("low.json", RUNG1))
            self.assertEqual(rc, 1)
            self.assertIn("refused: the ladder cannot run", done)
            self.assertIn("would step down", done)
            self.assertNotIn("server: card", out)
            self.assertEqual(rig.offsets()["0"], [1200, 0])
            self.assertEqual(rig.ledger_lines(), [])
        finally:
            rig.close()


class PlainAndControl(unittest.TestCase):
    def test_plain_mode_the_first_differing_check_sets_every_card_to_0(self):
        rig = Rig(RUNG1, {7: {"mem_min": 800}})
        try:
            rc, done, out = rig.soak("--minutes", "0.2")
            self.assertEqual(rc, 1, out)
            self.assertRegex(done, r"^exit 1 card 7 round 1: differs from stock$")
            self.assertTrue(all(rig.offsets()[str(g)] == [0, 0] for g in CARDS))
            self.assertEqual(len(rig.ledger_lines()), 5)
            self.assertFalse(os.path.exists(os.path.join(rig.work, "soak", "soaked.json")))
        finally:
            rig.close()

    def test_a_stock_control_counts_and_passes(self):
        rig = Rig({})
        try:
            rc, done, out = rig.soak("--minutes", "0.1", "--keep-going")
            self.assertEqual(rc, 0, out)
            self.assertRegex(done, r"^exit 0 rounds \d+ differs 0$")
            self.assertEqual(rig.ledger_lines(), [])
        finally:
            rig.close()

    def test_keep_going_at_an_offset_a_card_that_differs_once_stays_set(self):
        rig = Rig(RUNG1, {7: {"mem_min": 800}})
        try:
            rc, done, out = rig.soak("--minutes", "0.1", "--keep-going", "--profile", rig.write("p1.json", RUNG1))
            self.assertEqual(rc, 1, out)
            self.assertRegex(done, r"^exit 1 rounds \d+ differs 1$")
            self.assertEqual(rig.offsets()["7"], [800, 0])       # counted, not reset: the others' soak went on
            self.assertEqual(rig.ledger_lines(), [])
            with open(os.path.join(rig.work, "soak", "soaked.json")) as f:
                got = {c["card"]: c["memory_mts"] for c in json.load(f)["cards"]}
            self.assertEqual(got, {0: 1100, 2: 1000, 5: 1000, 7: 0, 8: 1100})
        finally:
            rig.close()


class ResetCard(unittest.TestCase):
    def test_reset_card_sets_one_card_to_0_and_writes_the_ledger(self):
        rig = Rig(RUNG4)
        try:
            p = subprocess.run([sys.executable, "-I", os.path.join(KIT, "climb.py"), "--work", rig.work, "reset-card",
                                "8"], env=rig.env, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            st = rig.offsets()
            self.assertEqual(st["8"], [0, 0])
            self.assertEqual(st["0"], [1300, 256])
            self.assertEqual(len(rig.ledger_lines()), 1)
            self.assertIn("card 8 mem 1300->0 core 231->0 reset-card", rig.ledger_lines()[0])
            p = subprocess.run([sys.executable, "-I", os.path.join(KIT, "climb.py"), "--work", rig.work, "reset-card",
                                "8"], env=rig.env, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0)
            self.assertEqual(len(rig.ledger_lines()), 1)          # already 0: no second ledger line
        finally:
            rig.close()


class ModelCheck(unittest.TestCase):
    def run_check(self, rig, tag, **env):
        e = dict(rig.env, TAG=tag, MODEL="model.gguf", SETTINGS=os.path.join(rig.tmp, "settings.env"),
                 OUT=os.path.join(rig.tmp, "model"), PORT=str(rig.port + 30), N="8", GREEDY=os.path.join(KIT, "greedy_check.py"),
                 **env)
        e.pop("CUDA_VISIBLE_DEVICES", None)
        p = subprocess.run(["bash", os.path.join(KIT, "model_check.sh")], env=e, capture_output=True, text=True,
                           timeout=200)
        return p.returncode, p.stdout + p.stderr

    def test_the_reference_then_the_same_output_passes_and_a_difference_fails(self):
        rig = Rig({}, {0: {"mem_min": 1000, "every": True}})    # the served model answers wrongly from +1,000 on
        try:
            with open(os.path.join(rig.tmp, "settings.env"), "w") as f:
                f.write('PRESET_ENV="LLAMA_TEST=1"\nPRESET_ARGS="-c 1024"\n')
            rc, out = self.run_check(rig, "profile")
            self.assertEqual(rc, 2, out)                       # no stock reference yet
            rc, out = self.run_check(rig, "stock", ROUNDS="2")
            self.assertEqual(rc, 0, out)
            self.assertIn("stock round 2", out)
            self.assertIn("SAME as stock1", out)
            rc, out = self.run_check(rig, "profile", ROUNDS="2", MINUTES="0.05")
            self.assertEqual(rc, 0, out)
            self.assertIn("generation tok/s, median of", out)
            self.assertIn("prompt 1 30", out)
            with open(os.path.join(rig.work, "pids", "8.pids")) as f:
                self.assertEqual(f.read(), "")                  # the server's pid is cleared at the end
            st = rig.offsets()
            st["0"] = [1000, 0]
            with open(rig.state, "w") as f:
                json.dump(st, f)
            rc, out = self.run_check(rig, "profile", ROUNDS="5")
            self.assertEqual(rc, 1, out)
            self.assertIn("profile round 1", out)
            self.assertIn("DIFFERS from stock1", out)
            self.assertNotIn("profile round 2", out)            # it stops at the first round that differs
        finally:
            rig.close()


if __name__ == "__main__":
    unittest.main()
