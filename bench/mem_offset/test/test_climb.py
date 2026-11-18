#!/usr/bin/env python3
"""Tests of climb.py's logic on any Linux machine: no GPU, no NVML, no root; nothing is set or run on a card.

  python3 -I bench/mem_offset/test/test_climb.py [-v]

- the parsers on the recorded output of R193 and R194 (2026-11-17), and the watchdog's Xid parsing;
- replays of R193 and R194 through the climb's own set, check and failure code (fixtures/, built from the raw output
  by make_replay_fixtures.py): they must fail the same cards for the same reasons, and leave the others set;
- a full simulated climb whose cards fail as the recorded ones did (card 0's memory at +1,400 MT/s, Q133; card 2's core
  at a 2,202 MHz peak with an Xid, R193; card 4's memory at +1,200 MT/s with wrong results and a hang, R194): the
  profile, the rule that offsets only go up, and the cards that go on;
- the stops that end everything (the watchdog's trip, an Xid naming no card, a check that does not stop);
- apply: refused on the climb's boot or after a step down, the profile set and checked, everything to 0 on a failure;
- --from-profile: no memory climb (each card's memory set once, to the profile's value), one check round for every
  card, the core climb only for the cards without a core value (or --core-cards) with the memory held, a card that
  fails the check round (memory or a kept core) set to 0 alone, the profile `apply` reads, its refusals before any
  offset, and a stop in the core climb;
- the --dry-run command line starts no process.
"""
import contextlib
import gzip
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
FIX = os.path.join(HERE, "fixtures")
sys.path.insert(0, KIT)
import climb  # noqa: E402

spec = importlib.util.spec_from_file_location("nvml_watchdog", os.path.join(KIT, "..", "..", "scripts", "nvml_watchdog.py"))
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)

R193_XID = ("NVRM: Xid (PCI:0000:05:00): 31, pid=4603, name=mmvq_pascal, channel 0x00000018, intr 10000000. MMU "
            "Fault: ENGINE GRAPHICS GPC0 GPCCLIENT_T1_0 faulted @ 0x7a72_f0548000. Fault is of type FAULT_PDE "
            "ACCESS_TYPE_READ")
RIG1_BUS = {0: (0, 3, 0), 1: (0, 4, 0), 2: (0, 5, 0), 3: (0, 6, 0), 4: (0, 7, 0), 5: (0, 0xb, 0), 6: (0, 0xc, 0),
            7: (0, 0xd, 0), 8: (0, 0xe, 0)}
FULL_SWEEP = "--types q8_0,q5_0,q4_1,q5_K,iq4_xs"
# the three recorded failures, as simulated cards
RECORDED_FAULTS = [
    {"card": 0, "mem_min": 1400, "check": "greedy", "kind": ["differs"]},    # Q133: only the probabilities changed
    {"card": 2, "peak_min": 2202, "check": "mmvq", "kind": ["xid"]},         # R193: Xid 31 at a 2,202 MHz peak
    {"card": 4, "mem_min": 1200, "check": "mmvq", "kind": ["wrong", "hang"]},  # R194: wrong results, then a hang
]


def fixture(name):
    path = os.path.join(FIX, name)
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return f.read()


def slurp(path):
    with open(path) as f:
        return f.read()


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


class Rig:
    """A simulated rig and the kit on it."""

    def __init__(self, test, extra=(), scenario=None, fixtures=FIX, x=None):
        self.work = tempfile.mkdtemp(prefix="climb-test-")
        test.addCleanup(shutil.rmtree, self.work, True)
        self.a = climb.parse(["--dry-run", "--work", self.work, "--check-model", "/models/check.gguf",
                              "--ledger", os.path.join(self.work, "ledger"), *extra, "climb"])
        self.x = x or climb.SimExec(lambda s: None, scenario, fixtures)
        self.k = climb.Kit(self.a, self.x)
        self.c = climb.Climb(self.k)

    def sets(self):
        """[(card, 'mem'|'core'|'reset', value)] in the order the commands were printed."""
        out = []
        for cmd in self.x.commands:
            m = re.search(r"nvml_memoc\.py (--core )?(?:--state \S+ --cap \d+ )?(set|reset) (\d+)(?: (\d+))?", cmd)
            if m:
                kind = "reset" if m.group(2) == "reset" else ("core" if m.group(1) else "mem")
                out.append((int(m.group(3)), kind, int(m.group(4)) if m.group(4) else 0))
        return out

    def profile(self):
        with open(os.path.join(self.work, "out", "profile.json")) as f:
            return json.load(f)


def replay(test, name):
    """R193's or R194's offsets set on its cards at once, then one round of the climb's checks (the recorded order:
    the benchmark, then the greedy check), through the climb's own code."""
    with open(os.path.join(FIX, f"{name}_replay.json")) as f:
        sc = json.load(f)
    cards = [p["card"] for p in sc["points"]]
    r = Rig(test, ["--cards", " ".join(map(str, cards)), "--order", "mmvq,greedy", "--mmvq-jobs", "9",
                   "--mmvq-args", FULL_SWEEP], sc)
    with quiet():
        smi, size = r.c.preflight()
        r.c.new_state(smi, size, "climb")
        r.k.start_servers(cards)
        r.c.stock(cards)
        for p in sc["points"]:
            test.assertEqual(r.k.set_offsets(p["card"], mem=p["mem"], core=p["core"]), "")
        t0 = r.x.now()
        res = r.k.run_round(cards, name, "stock", r.c.limits(cards), lambda g, why: r.c.fail_card(g, "replay", why))
    return r, sc, res, t0


class Parsers(unittest.TestCase):
    def test_xid_names_its_card(self):
        self.assertEqual(climb.xid_card(R193_XID, RIG1_BUS), 2)
        self.assertEqual(climb.xid_code(R193_XID), "31")
        self.assertIsNone(climb.xid_card("NVRM: Xid 79 without an address", RIG1_BUS))
        self.assertEqual(climb.pci_key("00000000:0B:00.0"), (0, 0xb, 0))

    def test_watchdog_parses_the_same(self):
        self.assertEqual(watchdog.xid_card(R193_XID, RIG1_BUS), 2)
        self.assertEqual(watchdog.xid_pid(R193_XID), 4603)
        self.assertIsNone(watchdog.xid_card(climb.TEST_LINE, RIG1_BUS))
        self.assertEqual(watchdog.xid_card(climb.TEST_LINE + " (PCI:0000:0e:00)", RIG1_BUS), 8)

    def test_watchdog_kills_one_cards_pids(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        p1 = subprocess.Popen(["sleep", "30"])
        p2 = subprocess.Popen(["sleep", "30"])
        self.addCleanup(p2.wait)
        self.addCleanup(p2.kill)
        with open(os.path.join(d, "3.pids"), "w") as f:
            f.write(f"{p1.pid}\n")
        killed = watchdog.kill_pids(watchdog.card_pids(d, 3))
        self.assertEqual([pid for pid, _ in killed], [p1.pid])
        self.assertIsNotNone(p1.wait(timeout=5))
        self.assertIsNone(p2.poll())
        self.assertEqual(watchdog.card_pids(d, 4), [])

    def test_peaks_from_the_watchdog_log(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "wd.csv")
        cols = ["time", "tripped", "gpu_mem_offset", "gpu_gpc_offset"] + [
            f"g{i}_{k}" for i in range(2) for k in ("pstate", "sm_mhz", "mem_mhz", "w", "limit_w", "c", "used_mib", "reasons")]
        t0 = float(int(time.time()))

        def row(t, sm, ps="P0"):
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + ".250"
            return ",".join([stamp, "0", "0", "0", ps, str(sm), "4604", "60", "75", "40", "900", "0x0",
                             "P8", "139", "405", "5", "75", "30", "0", "0x1"]) + "\n"
        with open(path, "w") as f:
            f.write("# watchdog start\n" + ",".join(cols) + "\n" + row(t0, 1860) + row(t0 + 1, 1873)
                    + row(t0 + 2, 1999, "P2"))
        x = climb.RealExec(lambda s: None)
        self.assertEqual(x.peaks(path, [0, 1], t0, t0 + 3), {0: [1873, 4604], 1: [None, None]})
        with open(path, "a") as f:
            f.write(row(t0 + 10, 2151) + row(t0 + 11, 2138) + "2026-11-")   # a line still being written
        self.assertEqual(x.peaks(path, [0, 1], t0 + 9, t0 + 12), {0: [2151, 4604], 1: [None, None]})

    def test_offsets_read_back(self):
        self.assertEqual(climb.parse_get("gpu 0 MemClk vf offset: 1200 (allowed (-2000, 2000))"), 1200)
        self.assertIsNone(climb.parse_get("gpu 0 MemClk vf offset: Not Supported"))

    def test_core_offset_lands_on_whole_steps(self):
        for n in range(1, 60):
            off = climb.core_offset(n, 12.66)
            for driver_bin in (12.65, 12.66):
                self.assertEqual(int(off / driver_bin), n)        # the driver drops the remainder (R194)
                self.assertEqual(round(off / driver_bin), n)      # or rounds it
        # R194's offsets and the steps the driver gave them: 300 -> 23, 275 -> 21, 262 -> 20, 249 -> 19
        for off, steps in ((300, 23), (275, 21), (262, 20), (249, 19)):
            self.assertEqual(int(off / 12.65), steps)

    def test_greedy_lines_from_r193(self):
        ok, h, _ = climb.parse_greedy("greedy card0: hash 200fe90f478c1933, 512 tokens, prompt tok/s 55.66 60.48, "
                                      "generation tok/s 32.25 32.15 | SAME as ref", 0, False, 60, "ref")
        self.assertEqual((ok, h), (True, "200fe90f478c1933"))
        ok, _, why = climb.parse_greedy("greedy card2: request failed: Remote end closed connection without response",
                                        3, False, 60, "ref")
        self.assertFalse(ok)
        self.assertIn("request failed", why)
        ok, _, why = climb.parse_greedy("greedy x: hash d1ffd1ffd1ffd1ff, 512 tokens | DIFFERS from stock (5a1e)",
                                        1, False, 60, "stock")
        self.assertEqual((ok, why), (False, "greedy output differs from stock"))
        ok, _, why = climb.parse_greedy("", None, True, 60, "stock")
        self.assertEqual((ok, why), (False, "greedy longer than its limit (60 s)"))

    def test_mmvq_output_from_r193_and_r194(self):
        ok, passes, fails, gbs, _ = climb.parse_mmvq(fixture("r194/mmvq_card0.csv.gz"), fixture("r194/mmvq_card0.log.gz"),
                                                     0, False, 210)
        self.assertEqual((ok, passes, fails, gbs), (True, 516, 0, 204.76))
        ok, passes, fails, gbs, why = climb.parse_mmvq("", fixture("r193/mmvq_card2.log.gz"), 1, False, 210)
        self.assertFalse(ok)
        self.assertEqual((passes, fails, gbs), (285, 0, None))
        self.assertIn("mmvq exit 1 (CUDA error", why)
        self.assertIn("illegal memory access", why)
        ok, passes, fails, gbs, why = climb.parse_mmvq("", fixture("r194/mmvq_card4.log.gz"), None, True, 210)
        self.assertFalse(ok)
        self.assertEqual(fails, 2)
        self.assertIn("2 wrong kernel results", why)
        self.assertIn("longer than its limit (210 s)", why)


class Replays(unittest.TestCase):
    def test_r193_card2_fails_on_its_xid_cards_0_1_pass(self):
        r, sc, res, _ = replay(self, "r193")
        self.assertTrue(res[0]["ok"] and res[1]["ok"])
        self.assertEqual([res[g]["mmvq"]["gbs"] for g in (0, 1)], [204.70, 204.80])
        self.assertEqual([res[g]["mmvq"]["passes"] for g in (0, 1)], [516, 516])
        self.assertFalse(res[2]["ok"])
        why = "; ".join(res[2]["why"])
        self.assertIn("Xid 31", why)   # the scan finds the Xid (at 36 s) and stops the benchmark at once
        self.assertEqual(res[2]["xid"], 1)
        # only card 2 went to 0; cards 0 and 1 kept their offsets and the climb would go on with them
        self.assertEqual(r.x.off[2], [0, 0])
        self.assertEqual([r.x.off[0], r.x.off[1]], [[1200, 300], [1200, 300]])
        self.assertEqual(r.k.card(2)["status"], "failed")
        self.assertTrue(r.k.card(2)["stepped_down_mid_climb"])
        self.assertEqual(len(climb.stepped_down_this_boot(r.x.read(r.a.ledger), r.k.boot())), 1)

    def test_r194_card4_fails_on_wrong_results_and_the_time_limit(self):
        r, sc, res, t0 = replay(self, "r194")
        for g, gbs in zip(range(4), (204.76, 204.60, 204.76, 204.62)):
            self.assertTrue(res[g]["ok"], res[g])
            self.assertEqual((res[g]["mmvq"]["passes"], res[g]["mmvq"]["gbs"]), (516, gbs))
            self.assertTrue(res[g]["greedy"]["ok"])
        self.assertFalse(res[4]["ok"])
        why = "; ".join(res[4]["why"])
        self.assertIn("2 wrong kernel results", why)
        self.assertIn("mmvq longer than its limit (210 s)", why)   # 3 x the 70 s sweep
        self.assertNotIn("greedy", res[4])                         # dropped once the card failed
        # the limit ends the hang 210 s after its start: the recorded hang was stopped by hand 383 s after card 4 was
        # set (15:30:53 to 15:37:16), with five cards offset meanwhile
        failed_at = time.mktime(time.strptime(r.k.card(4)["failure"]["time"], "%Y-%m-%dT%H:%M:%S"))
        self.assertLessEqual(failed_at - t0, 211)
        self.assertLess(failed_at - t0, 383)
        self.assertEqual(r.x.off[4], [0, 0])
        self.assertEqual([r.x.off[g] for g in range(4)], [[1200, 300], [1200, 300], [1200, 249], [1200, 249]])


class FullClimb(unittest.TestCase):
    def run_climb(self, scenario=None, extra=(), x=None):
        r = Rig(self, extra, scenario, x=x)
        with quiet():
            rc = r.c.run()
        return r, rc

    def test_recorded_failures(self):
        r, rc = self.run_climb({"faults": RECORDED_FAULTS})
        self.assertEqual(rc, 0)
        p = r.profile()
        self.assertTrue(p["complete"])
        cards = {c["card"]: c for c in p["cards"]}
        c0, c2, c4 = cards[0], cards[2], cards[4]
        # card 0: memory clean to +1,300, fails at +1,400 on the greedy check alone; no core climb
        self.assertEqual((c0["memory"]["last_clean_mts"], c0["memory"]["failed_at_mts"]), (1300, 1400))
        self.assertEqual((c0["memory_mts"], c0["core_mhz"]), (1200, 0))
        self.assertEqual(c0["failure"]["reason"], "greedy output differs from stock")
        self.assertFalse(c0["core"]["took_part"])
        self.assertTrue(c0["needs_reboot_before_use"])
        # card 4: memory fails at +1,200 (wrong results, then the hang cut by the limit)
        self.assertEqual((c4["memory"]["last_clean_mts"], c4["memory_mts"]), (1100, 1000))
        self.assertIn("2 wrong kernel results", c4["failure"]["reason"])
        self.assertIn("longer than its limit", c4["failure"]["reason"])
        self.assertTrue(c4["needs_reboot_before_use"])
        # card 2: memory clean to +1,400, core fails at a 2,202 MHz peak with an Xid; 24 steps fail, 22 pass, 20 kept
        self.assertEqual((c2["memory"]["last_clean_mts"], c2["memory_mts"]), (1400, 1300))
        self.assertEqual(c2["failure"]["phase"], "core")
        self.assertIn("Xid 31", c2["failure"]["reason"])
        self.assertEqual(c2["core"]["failed_at_mhz"], climb.core_offset(24, 12.66))
        self.assertEqual(c2["core"]["last_clean_mhz"], climb.core_offset(22, 12.66))
        self.assertEqual((c2["core_mhz"], c2["core_steps"]), (climb.core_offset(20, 12.66), 20))
        self.assertTrue(c2["needs_reboot_before_use"])
        # the others go on to the top of both climbs: memory +1,400 and a 2,250 MHz peak; kept one step under
        for g in (1, 3, 5, 6, 7, 8):
            c = cards[g]
            self.assertIsNone(c["failure"])
            self.assertFalse(c["needs_reboot_before_use"])
            self.assertEqual((c["memory"]["last_clean_mts"], c["memory_mts"]), (1400, 1300))
            self.assertLessEqual(c["core"]["last_clean_peak_mhz"], 2251)
            self.assertGreater(c["core"]["last_clean_peak_mhz"], 2225)
            self.assertEqual(c["core_mhz"], climb.core_offset(c["core_steps"], 12.66))
            self.assertLess(c["core_mhz"], c["core"]["last_clean_mhz"])
        self.assertEqual(cards[1]["core_mhz"], climb.core_offset(28, 12.66))
        # every card at 0 at the end, read back
        self.assertTrue(all(v == [0, 0] for v in r.x.off.values()))

    def test_offsets_only_go_up_and_a_failed_card_stays_out(self):
        r, _ = self.run_climb({"faults": RECORDED_FAULTS})
        last = {}
        ended = set()
        end_resets = {}
        for g, kind, v in r.sets():
            if kind == "reset":
                end_resets[g] = end_resets.get(g, 0) + 1
                ended.add(g)
                continue
            self.assertNotIn(g, ended, f"card {g} set again after its reset")
            self.assertGreater(v, last.get((g, kind), 0), f"card {g} {kind} did not go up")
            last[(g, kind)] = v
        # each card is reset twice (memory, core) at its failure or at the end, and twice more at the end if it failed
        self.assertEqual({g: n for g, n in end_resets.items()},
                         {g: (4 if g in (0, 2, 4) else 2) for g in range(9)})
        import csv
        with open(os.path.join(r.work, "out", "steps.csv")) as f:
            rows = list(csv.DictReader(f))
        row = next(x for x in rows if (x["phase"], x["step"], x["card"]) == ("memory", "+1200", "4"))
        self.assertEqual((row["mem_mts"], row["verdict"]), ("1200", "FAIL"))
        self.assertIn("wrong kernel results", row["reason"])

    def test_the_watchdogs_trip_ends_everything(self):
        class Tripping(climb.SimExec):
            def sleep(self, s):
                super().sleep(s)
                if self.t > self.trip_at and self.marker not in self.files:
                    self.files[self.marker] = "2026-11-18 10:00:00 TRIP 1: gpu 3 at 71 C > 70.0 C\n"
        x = Tripping(lambda s: None, None, FIX)
        r = Rig(self, x=x)
        x.marker = r.k.marker()
        x.trip_at = x.t + 13 * 60   # during the memory climb
        with quiet():
            rc = r.c.run()
        self.assertEqual(rc, 1)
        p = r.profile()
        self.assertFalse(p["complete"])
        self.assertIn("the watchdog tripped", p["stop_reason"])
        self.assertTrue(all(v == [0, 0] for v in x.off.values()))
        self.assertTrue(all(c["failure"] is None for c in p["cards"]))
        self.assertTrue(0 < p["cards"][0]["memory"]["last_clean_mts"] < 1400)

    def test_an_xid_naming_no_card_ends_everything(self):
        class Anonymous(climb.SimExec):
            def sleep(self, s):
                super().sleep(s)
                if self.t > self.at and not self.xids:
                    self.xids.append((self.t, "NVRM: Xid 79, GPU has fallen off the bus"))
        x = Anonymous(lambda s: None, None, FIX)
        x.at = x.t + 12 * 60
        r = Rig(self, x=x)
        with quiet():
            rc = r.c.run()
        self.assertEqual(rc, 1)
        self.assertIn("an Xid that names no card", r.profile()["stop_reason"])
        self.assertTrue(all(v == [0, 0] for v in x.off.values()))

    def test_a_check_that_does_not_stop_ends_everything(self):
        class Stuck(climb.SimExec):
            def _mmvq(self, g, tag, csv_path, out):
                p = super()._mmvq(g, tag, csv_path, out)
                if g == 3 and tag == "mem800":
                    p.done_at = None
                    p.stop = lambda grace=5.0: False
                return p
        x = Stuck(lambda s: None, None, FIX)
        r = Rig(self, x=x)
        with quiet():
            rc = r.c.run()
        self.assertEqual(rc, 1)
        self.assertIn("card 3: its mmvq check does not stop", r.profile()["stop_reason"])
        self.assertEqual(r.profile()["cards"][3]["failure"]["mem"], 800)
        self.assertTrue(all(v == [0, 0] for v in x.off.values()))

    def test_refuses_without_the_watchdog_or_after_a_step_down(self):
        r = Rig(self, scenario={"watchdog": False})
        with quiet():
            with self.assertRaises(climb.Refused) as e:
                r.c.run()
        self.assertIn("watchdog is not running", str(e.exception))
        r = Rig(self)
        r.x.ledger = f"{r.k.boot()} 2026-11-18 09:00:00 card 4 mem 1200->0 core 275->0 by hand\n"
        with quiet():
            with self.assertRaises(climb.Refused) as e:
                r.c.run()
        self.assertIn("stepped down this boot", str(e.exception))
        r = Rig(self)
        r.x.off[5] = [0, 249]
        with quiet():
            with self.assertRaises(climb.Refused) as e:
                r.c.run()
        self.assertIn("card 5 offsets (mem core) 0 249", str(e.exception))
        self.assertEqual(r.sets(), [])


class ApplyProfile(unittest.TestCase):
    def profile(self):
        r = Rig(self, scenario={"faults": RECORDED_FAULTS})
        with quiet():
            r.c.run()
        path = os.path.join(r.work, "out", "profile.json")
        return path, r.profile()

    def apply(self, path, scenario=None, ledger=None):
        r = Rig(self, scenario=scenario)
        if ledger:
            r.x.ledger = ledger.replace("BOOT", r.k.boot())
        a = climb.parse(["--dry-run", "--work", r.work, "--check-model", "/models/check.gguf", "--ledger",
                         r.a.ledger, "apply", path])
        r.k.a = r.a = a
        with quiet():
            rc = climb.Apply(r.k).run(path)
        return r, rc

    def test_refused_on_the_climbs_boot(self):
        path, _ = self.profile()
        r, rc = self.apply(path)
        self.assertEqual(rc, 1)
        self.assertIn("has not rebooted since the climb", slurp(r.k.logf))
        self.assertEqual(r.sets(), [])

    def test_refused_after_a_step_down(self):
        path, _ = self.profile()
        r, rc = self.apply(path, {"boot_id": "after-a-reboot"}, ledger="BOOT 2026-11-18 card 1 mem 1300->0 core 0->0 x\n")
        self.assertEqual(rc, 1)
        self.assertIn("stepped down this boot", slurp(r.k.logf))
        self.assertEqual(r.sets(), [])

    def test_sets_the_profile_and_keeps_it(self):
        path, prof = self.profile()
        r, rc = self.apply(path, {"boot_id": "after-a-reboot"})
        self.assertEqual(rc, 0)
        for c in prof["cards"]:
            self.assertEqual(r.x.off[c["card"]], [c["memory_mts"], c["core_mhz"]])
        self.assertEqual([s for s in r.sets() if s[1] == "reset"], [])
        with open(os.path.join(r.work, "out", "apply.json")) as f:
            self.assertTrue(json.load(f)["complete"])

    def test_one_failure_sets_every_card_to_0(self):
        path, prof = self.profile()
        bad = [{"card": 6, "mem_min": 1300, "check": "greedy", "kind": ["differs"]}]
        r, rc = self.apply(path, {"boot_id": "after-a-reboot", "faults": bad})
        self.assertEqual(rc, 1)
        self.assertTrue(all(v == [0, 0] for v in r.x.off.values()))
        log = slurp(r.k.logf)
        self.assertIn("card 6 failed its check at the profile's offsets: greedy output differs from stock", log)
        self.assertIn("needs a reboot before the next try", log)


# a soaked.json as soak.py writes it for rig 2 (R201's rung 4): cores on cards 0, 3, 4 and 8, none on the others
SOAKED = {0: (1300, 256, 20), 1: (1000, 0, 0), 2: (1200, 0, 0), 3: (1300, 54, 4), 4: (1200, 206, 16),
          5: (1200, 0, 0), 6: (1100, 0, 0), 7: (1000, 0, 0), 8: (1300, 231, 18)}
CLIMBERS = [1, 2, 5, 6, 7]
KEPT = [0, 3, 4, 8]


def soaked(test, values=SOAKED, pci=None, **top):
    """A source profile in a temporary folder, shaped like soak.py's soaked.json; returns its path."""
    d = tempfile.mkdtemp(prefix="climb-src-")
    test.addCleanup(shutil.rmtree, d, True)
    prof = {"what": "P106 clock offsets that held a soak (test)", "complete": True, "climb_boot": None,
            "cards": [{"card": g, "pci": (pci or {}).get(g, climb.SCENARIO["pci"][g]), "name": "NVIDIA P106-090",
                       "vbios": "86.06.5B.00.0C", "stock_peak_mhz": 1898, "memory_mts": v[0], "core_mhz": v[1],
                       "core_steps": v[2], "soak": {"left": None}} for g, v in sorted(values.items())]}
    prof.update(top)
    path = os.path.join(d, "soaked.json")
    with open(path, "w") as f:
        json.dump(prof, f)
    return path


class FromProfile(unittest.TestCase):
    def run_fp(self, path, extra=(), scenario=None, x=None):
        r = Rig(self, ["--from-profile", path, *extra], scenario, x=x)
        with quiet():
            rc = climb.FromProfile(r.k).run(r.a.from_profile)
        return r, rc

    def test_the_plan_from_a_profile(self):
        prof = json.loads(slurp(soaked(self)))
        plan, bad = climb.profile_plan(prof, list(range(9)), None, 1400, 400, 12.66)
        self.assertEqual(bad, [])
        self.assertEqual([g for g in range(9) if plan[g]["core_climb"]], CLIMBERS)
        self.assertEqual((plan[3]["memory_mts"], plan[3]["core_mhz"], plan[3]["core_steps"]), (1300, 54, 4))
        plan, bad = climb.profile_plan(prof, list(range(9)), [1, 2, 3, 5, 6, 7], 1400, 400, 12.66)
        self.assertEqual(bad, [])
        self.assertEqual((plan[3]["core_climb"], plan[3]["core_mhz"], plan[3]["profile_core_mhz"]), (True, 0, 54))
        self.assertEqual((plan[0]["core_climb"], plan[0]["core_mhz"]), (False, 256))
        # a core without its steps (a hand-made profile): the steps from the core
        p2 = {"cards": [dict(prof["cards"][0], core_steps=None)]}
        self.assertEqual(climb.profile_plan(p2, [0], None, 1400, 400, 12.66)[0][0]["core_steps"], 20)
        # what cannot run, all named at once
        broken = json.loads(json.dumps(prof))
        broken["cards"] = [c for c in broken["cards"] if c["card"] != 8]
        broken["cards"][1]["memory_mts"] = "x"
        broken["cards"][2]["pci"] = ""
        broken["cards"][4]["core_mhz"] = 450
        _, bad = climb.profile_plan(broken, list(range(9)), None, 1200, 400, 12.66)
        text = "; ".join(bad)
        self.assertIn("card 8 is not in the profile", text)
        self.assertIn("card 1: memory_mts, core_mhz or core_steps is not a whole number", text)
        self.assertIn("card 2: the profile has no pci or name", text)
        self.assertIn("card 4: core +450 MHz is above --core-cap 400", text)
        self.assertIn("card 0: memory +1300 MT/s is above the cap +1200", text)
        self.assertIn("--core-cards [9] are not among", "; ".join(
            climb.profile_plan(prof, [0, 1], [1, 9], 1400, 400, 12.66)[1]))
        self.assertEqual(climb.profile_plan({"cards": []}, [0], None, 1400, 400, 12.66)[1], ["the profile has no cards"])

    def test_memory_straight_to_the_profile_then_the_core_climb_for_the_cards_without_one(self):
        faults = [
            {"card": 5, "mem_min": 1200, "check": "greedy", "kind": ["differs"]},   # its soaked memory fails here
            {"card": 4, "peak_min": 2080, "check": "greedy", "kind": ["differs"]},  # its kept core fails here
            {"card": 2, "peak_min": 2100, "check": "mmvq", "kind": ["xid"]},        # an Xid in the core climb
        ]
        path = soaked(self)
        r, rc = self.run_fp(path, scenario={"faults": faults})
        self.assertEqual(rc, 0)
        p = r.profile()
        self.assertTrue(p["complete"])
        self.assertEqual((p["from_profile"], p["core_climb_cards"], p["memory_steps_mts"]),
                         ("soaked.json", CLIMBERS, []))
        cards = {c["card"]: c for c in p["cards"]}
        sets = r.sets()
        # no memory climb: each card's memory is set once, to the profile's value
        mem_sets = [(g, v) for g, kind, v in sets if kind == "mem"]
        self.assertEqual(sorted(mem_sets), sorted((g, v[0]) for g, v in SOAKED.items()))
        # the kept cores are set with their memory, before the check round; the climbers' cores only after it
        first_climb = next(i for i, (g, kind, v) in enumerate(sets) if kind == "core" and g in CLIMBERS)
        last_mem = max(i for i, (g, kind, v) in enumerate(sets) if kind == "mem")
        self.assertLess(last_mem, first_climb)
        self.assertEqual([(g, v) for g, kind, v in sets[:first_climb] if kind == "core"],
                         [(g, SOAKED[g][1]) for g in KEPT])
        # offsets only go up, and a card is never set again after its reset
        last, ended = {}, set()
        for g, kind, v in sets:
            if kind == "reset":
                ended.add(g)
                continue
            self.assertNotIn(g, ended, f"card {g} set again after its reset")
            self.assertGreater(v, last.get((g, kind), 0))
            last[(g, kind)] = v
        # card 5: its soaked memory fails the check round; to 0 alone, no core climb, nothing in the profile
        c5 = cards[5]
        self.assertEqual((c5["memory_mts"], c5["core_mhz"]), (0, 0))
        self.assertEqual((c5["failure"]["phase"], c5["failure"]["reason"]), ("profile", "greedy output differs from stock"))
        self.assertEqual((c5["memory"]["failed_at_mts"], c5["memory"]["limit_found"]), (1200, True))
        self.assertFalse(c5["core"]["took_part"])
        self.assertTrue(c5["needs_reboot_before_use"])
        # card 4: its kept core fails the check round; to 0 alone (the limit is not named: memory and core were set)
        c4 = cards[4]
        self.assertEqual((c4["memory_mts"], c4["core_mhz"], c4["failure"]["phase"]), (0, 0, "profile"))
        self.assertEqual((c4["failure"]["mem"], c4["failure"]["core"]), (1200, 206))
        self.assertFalse(c4["memory"]["limit_found"])
        self.assertTrue(c4["needs_reboot_before_use"])
        # card 2: memory held at +1,200 through the core climb, which ends at its Xid; one step under the last clean
        c2 = cards[2]
        self.assertEqual(c2["memory_mts"], 1200)
        self.assertEqual(c2["failure"]["phase"], "core")
        self.assertEqual(c2["failure"]["mem"], 1200)
        self.assertIn("Xid 31", c2["failure"]["reason"])
        self.assertLess(c2["core_mhz"], c2["core"]["last_clean_mhz"])
        self.assertLess(c2["core"]["last_clean_mhz"], c2["core"]["failed_at_mhz"])
        self.assertGreaterEqual(c2["core"]["failed_at_peak_mhz"], 2100)
        # the kept cards: their profile values, checked once, then to 0 at once
        for g in (0, 3, 8):
            c = cards[g]
            self.assertEqual((c["memory_mts"], c["core_mhz"], c["core_steps"]), SOAKED[g])
            self.assertEqual(c["memory"]["profile_check"], "PASS")
            self.assertFalse(c["core"]["took_part"])
            self.assertIsNone(c["failure"])
            self.assertFalse(c["needs_reboot_before_use"])
            self.assertEqual(r.k.card(g)["status"], "done")
        # the climbers that never failed: their memory, and a core one step under the top of the climb
        for g in (1, 6, 7):
            c = cards[g]
            self.assertEqual(c["memory_mts"], SOAKED[g][0])
            self.assertTrue(c["core"]["took_part"])
            self.assertIsNone(c["failure"])
            self.assertGreater(c["core"]["last_clean_peak_mhz"], 2225)
            self.assertEqual(c["core_mhz"], climb.core_offset(c["core_steps"], 12.66))
            self.assertLess(c["core_mhz"], c["core"]["last_clean_mhz"])
        # steps.csv: no memory step; one check row for every card; core rows only for the climbers
        import csv
        with open(os.path.join(r.work, "out", "steps.csv")) as f:
            rows = list(csv.DictReader(f))
        self.assertFalse([x for x in rows if x["phase"] == "memory"])
        check = {int(x["card"]): x for x in rows if x["phase"] == "profile"}
        self.assertEqual(sorted(check), list(range(9)))
        self.assertEqual([g for g in range(9) if check[g]["verdict"] == "FAIL"], [4, 5])
        self.assertEqual(check[0]["core_mhz"], "256")
        self.assertEqual(sorted({int(x["card"]) for x in rows if x["phase"] == "core"}), [1, 2, 6, 7])   # not 5
        self.assertTrue(all(x["mem_mts"] == str(SOAKED[int(x["card"])][0]) for x in rows if x["phase"] == "core"))
        self.assertTrue(all(v == [0, 0] for v in r.x.off.values()))
        # `climb.py profile` writes the same profile again from climb.json
        with open(os.path.join(r.work, "out", "climb.json")) as f:
            self.assertEqual(climb.build_profile(json.load(f)), p)
        # apply reads it after a reboot: the cards that failed stay at 0
        out = os.path.join(r.work, "out", "profile.json")
        a = Rig(self, scenario={"boot_id": "after-a-reboot"})
        a.k.a = a.a = climb.parse(["--dry-run", "--work", a.work, "--check-model", "/m/c.gguf", "--ledger",
                                   a.a.ledger, "apply", out])
        with quiet():
            self.assertEqual(climb.Apply(a.k).run(out), 0)
        for g, c in cards.items():
            self.assertEqual(a.x.off[g], [c["memory_mts"], c["core_mhz"]])

    def test_core_cards_climbs_a_card_from_0_whatever_core_the_profile_gives(self):
        r, rc = self.run_fp(soaked(self), ["--core-cards", "1 2 3 5 6 7"])
        self.assertEqual(rc, 0)
        card3 = [v for g, kind, v in r.sets() if g == 3 and kind == "core"]
        self.assertEqual(card3[0], 28)            # the climb's first step above its stock peak, not the profile's +54
        self.assertEqual(card3, sorted(card3))
        self.assertIn(" 3:+1300/+0 ", slurp(r.k.logf))   # the check round ran with its core at 0
        c3 = next(c for c in r.profile()["cards"] if c["card"] == 3)
        self.assertEqual((c3["memory_mts"], c3["core"]["took_part"], c3["core"]["from_profile_mhz"]), (1300, True, 54))
        self.assertGreater(c3["core_mhz"], 54)
        self.assertIn("card 3's core climbs from 0 (the profile's +54 MHz is not used)", slurp(r.k.logf))
        self.assertEqual(r.profile()["core_climb_cards"], [1, 2, 3, 5, 6, 7])

    def test_refusals_before_any_offset(self):
        cases = [
            (soaked(self, pci={2: "00000000:05:00.0", 3: "00000000:09:00.0"}), (), None, "card 3 is NVIDIA P106-090 at"),
            (soaked(self, {g: SOAKED[g] for g in range(8)}), ("--cards", "0 1 2 3 4 5 6 7 8"), None,
             "card 8 is not in the profile"),
            (soaked(self), ("--mem-steps", "200 400 600 800 1000 1200"), None, "memory +1300 MT/s is above the cap"),
            (soaked(self, climb_boot=climb.boot_tag("5f0c4b8e-0000-4000-8000-000000000000")), (), None,
             "has not rebooted since the climb"),
            (soaked(self), (), {"watchdog": False}, "the watchdog is not running"),
        ]
        for path, extra, sc, want in cases:
            r = Rig(self, ["--from-profile", path, *extra], sc)
            with quiet():
                with self.assertRaises(climb.Refused) as e:
                    climb.FromProfile(r.k).run(r.a.from_profile)
            self.assertIn(want, str(e.exception))
            self.assertEqual(r.sets(), [], want)
        r = Rig(self, ["--from-profile", soaked(self)])
        r.x.off[6] = [1100, 0]   # an offset left from before: the start check
        with quiet():
            with self.assertRaises(climb.Refused) as e:
                climb.FromProfile(r.k).run(r.a.from_profile)
        self.assertIn("card 6 offsets (mem core) 1100 0", str(e.exception))
        self.assertEqual(r.sets(), [])

    def test_a_trip_in_the_core_climb_ends_everything_and_keeps_what_was_found(self):
        class Tripping(climb.SimExec):
            def sleep(self, s):
                super().sleep(s)
                if self.t > self.trip_at and self.marker not in self.files:
                    self.files[self.marker] = "2026-11-18 10:00:00 TRIP 1: gpu 6 at 71 C > 70.0 C\n"
        x = Tripping(lambda s: None, None, FIX)
        x.trip_at = x.t + 13 * 60   # servers 9 x 45 s, two stock rounds and the check round take ~10 min
        r = Rig(self, ["--from-profile", soaked(self)], x=x)
        x.marker = r.k.marker()
        with quiet():
            rc = climb.FromProfile(r.k).run(r.a.from_profile)
        self.assertEqual(rc, 1)
        p = r.profile()
        self.assertFalse(p["complete"])
        self.assertIn("the watchdog tripped", p["stop_reason"])
        self.assertTrue(all(v == [0, 0] for v in x.off.values()))
        cards = {c["card"]: c for c in p["cards"]}
        for g in CLIMBERS:
            self.assertEqual(cards[g]["memory_mts"], SOAKED[g][0])   # its check round passed
            self.assertTrue(0 < cards[g]["core"]["last_clean_mhz"] < 300)
        for g in KEPT:
            self.assertEqual((cards[g]["memory_mts"], cards[g]["core_mhz"]), SOAKED[g][:2])

    def test_an_xid_on_a_card_that_is_done_is_logged_not_a_stop(self):
        class LateXid(climb.SimExec):
            def sleep(self, s):
                super().sleep(s)
                if self.t > self.at and not self.xids:
                    self.xids.append((self.t, self._xid_line(8)))
        x = LateXid(lambda s: None, None, FIX)
        x.at = x.t + 12 * 60
        r = Rig(self, ["--from-profile", soaked(self)], x=x)
        with quiet():
            rc = climb.FromProfile(r.k).run(r.a.from_profile)
        self.assertEqual(rc, 0)
        self.assertIn("xid: on card 8, which already left the climb", slurp(r.k.logf))


class CommandLine(unittest.TestCase):
    def test_from_profile_dry_run_and_plan(self):
        src = soaked(self)
        work = tempfile.mkdtemp(prefix="climb-cli-")
        self.addCleanup(shutil.rmtree, work, True)

        def no(*a, **k):
            raise AssertionError("a process was started in --dry-run")
        saved = (subprocess.Popen, subprocess.run)
        subprocess.Popen = subprocess.run = no
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                base = ["--dry-run", "--work", work, "--check-model", "/m/check.gguf", "--ledger",
                        os.path.join(work, "ledger"), "--from-profile", src]
                rcs = [climb.main(base + ["plan"]), climb.main(base + ["climb"]),
                       climb.main(base[:-1] + [soaked(self, {0: (1600, 0, 0)}), "climb"]),
                       climb.main(base + ["--core-cards", "0 9", "plan"])]
        finally:
            subprocess.Popen, subprocess.run = saved
        self.assertEqual(rcs, [0, 0, 1, 1])
        text = out.getvalue()
        self.assertIn("then the core climb for cards [1, 2, 5, 6, 7], memory held", text)
        self.assertIn("3,1300,54,no,54", text)
        self.assertIn("target_peak_mhz,card1_core_mhz,card2_core_mhz,card5_core_mhz,card6_core_mhz,card7_core_mhz", text)
        self.assertIn("profile: check round at", text)
        self.assertIn("--core-cards [9] are not among", text)
        self.assertIn("REFUSED: profile", err.getvalue())
        self.assertIn("memory +1600 MT/s is above the cap +1400", err.getvalue())
        self.assertNotIn("Traceback", text + err.getvalue())

    def test_core_cards_needs_from_profile(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                climb.parse(["--core-cards", "1 2", "climb"])

    def test_dry_run_starts_no_process(self):
        work = tempfile.mkdtemp(prefix="climb-cli-")
        self.addCleanup(shutil.rmtree, work, True)

        def no(*a, **k):
            raise AssertionError("a process was started in --dry-run")
        saved = (subprocess.Popen, subprocess.run)
        subprocess.Popen = subprocess.run = no
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                rcs = [climb.main(["--dry-run", "--work", work, "--check-model", "/m/check.gguf",
                                   "--ledger", os.path.join(work, "ledger"), *cmd])
                       for cmd in (["start-check"], ["trip-test"], ["watchdog", "start"], ["climb"], ["plan"])]
        finally:
            subprocess.Popen, subprocess.run = saved
        self.assertEqual(rcs, [0, 0, 0, 0, 0])
        text = out.getvalue()
        self.assertIn("nvml_memoc.py --state", text)
        self.assertIn("set 0 200", text)
        self.assertIn("--per-card-xid", text)
        self.assertIn("/dev/kmsg", text)
        self.assertNotIn("Traceback", text)
        self.assertTrue(os.path.exists(os.path.join(work, "out", "profile.json")))

    def test_plan_matches_r194s_steps_near_2151(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            climb.main(["--dry-run", "--work", tempfile.mkdtemp(prefix="climb-plan-"), "plan"])
        rows = [ln.split(",") for ln in out.getvalue().splitlines() if re.match(r"^\d{4},", ln)]
        peaks = [int(r[0]) for r in rows]
        self.assertEqual(peaks[0], 1885)
        self.assertTrue(all(25 <= b - a <= 26 for a, b in zip(peaks, peaks[1:])))
        row = next(r for r in rows if r[0] == "2139")   # the target nearest R194's 2,138-2,151 MHz peaks
        steps = [int(int(v) / 12.65) for v in row[1:]]
        self.assertEqual(steps, [22, 22, 18, 18, 20, 18, 20, 18, 19])   # R194: 23, 23, 19, 19, 21, 19, 21, 19, 20


if __name__ == "__main__":
    unittest.main()
