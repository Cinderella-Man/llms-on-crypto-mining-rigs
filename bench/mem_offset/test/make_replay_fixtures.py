#!/usr/bin/env python3
"""Builds the replay fixtures of test/test_climb.py from the raw output folders of R193 and R194 (2026-11-17), and the
stock-round samples of R200's five boots.

  make_replay_fixtures.py R193_OUT R194_OUT [FIXTURES]
  make_replay_fixtures.py --stock-windows R200_RUN [FIXTURES]   # R200_RUN: one folder per boot (climb, try1 ... try4),
                                                                # each with the watchdog's wd*.csv.gz and out/climb.log

R193_OUT and R194_OUT are the runs' `out/` folders (the files the data READMEs list: r193.log, greedy.log,
mmvq_card*.csv/.log); FIXTURES defaults to test/fixtures next to this script. It copies, gzipped, the benchmark output
of every card the runs checked, and writes one replay scenario per run: for each card, the offsets it was checked at,
the benchmark's exit status and output, the greedy check's line, the Xid line, the measured peak clock and the
durations, read from the driver's log. Data: docs/data/2026-11-17_r193_nine_cards_soak/ and
docs/data/2026-11-17_r194_nine_cards_soak/.
"""
import gzip
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# R193's only Xid: card 2 (PCI 05:00), from the watchdog's trip line (docs/data/2026-11-17_r193_nine_cards_soak/tables/
# watchdog.txt) and the kernel log (14:51:08)
R193_XID = ("NVRM: Xid (PCI:0000:05:00): 31, pid=4603, name=mmvq_pascal, channel 0x00000018, intr 10000000. MMU "
            "Fault: ENGINE GRAPHICS GPC0 GPCCLIENT_T1_0 faulted @ 0x7a72_f0548000. Fault is of type FAULT_PDE "
            "ACCESS_TYPE_READ")
R193_XID_TIME = "2026-11-17 14:51:08"
PEAKS = {  # measured peak core clocks at the offsets (the runs' READMEs): R193 cards 0-2, R194 cards 0-4
    "r193": {0: 2151, 1: 2151, 2: 2202},
    "r194": {0: 2151, 1: 2151, 2: 2138, 3: 2151, 4: 2138},
}


def t(stamp):
    return time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))


def gz(src, dst):
    with open(src, "rb") as f, open(dst, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as g:
        g.write(f.read())


def build(run, out, fixtures):
    log = open(os.path.join(out, f"{run}.log")).read().splitlines()
    greedy = open(os.path.join(out, "greedy.log")).read().splitlines()
    d = os.path.join(fixtures, run)
    os.makedirs(d, exist_ok=True)
    sets, verdict = {}, {}
    for ln in log:
        m = re.match(r"(\S+ \S+) card (\d+): set.*\(mem core\): (\d+) (\d+)", ln)
        if m:
            sets[int(m.group(2))] = (m.group(1), int(m.group(3)), int(m.group(4)))
        m = re.match(r"(\S+ \S+) card (\d+): (PASS|FAIL)", ln)
        if m:
            verdict[int(m.group(2))] = (m.group(1), m.group(3))
    points = []
    for g, (stamp, mem, core) in sorted(sets.items()):
        pt = {"card": g, "mem": mem, "core": core, "peak": PEAKS[run].get(g), "set_at": stamp}
        gl = [ln for ln in greedy if ln.startswith(f"greedy card{g}:")]
        pt["greedy"] = ({"line": gl[0].replace(f"greedy card{g}:", "greedy {tag}:"), "s": 20,
                         "rc": 0 if "SAME as" in gl[0] else 3 if "request failed" in gl[0] else 1}
                        if gl else {"not_recorded": True, "s": 20})
        csvp = os.path.join(out, f"mmvq_card{g}.csv")
        logp = os.path.join(out, f"mmvq_card{g}.log")
        timep = os.path.join(out, f"mmvq_card{g}.time")
        tm = open(timep).read() if os.path.exists(timep) else ""
        mm = {}
        if os.path.exists(csvp):
            gz(csvp, os.path.join(d, f"mmvq_card{g}.csv.gz"))
            mm["csv"] = f"{run}/mmvq_card{g}.csv.gz"
        gz(logp, os.path.join(d, f"mmvq_card{g}.log.gz"))
        mm["log"] = f"{run}/mmvq_card{g}.log.gz"
        if "terminated by signal" in tm:
            mm.update(rc=None, s=None, note="hung: killed by hand; its log is all it printed")
        else:
            m = re.search(r"non-zero status (\d+)", tm)
            mm["rc"] = int(m.group(1)) if m else 0
            # the run checked a card in this order: set, mmvq, greedy, verdict; a passing card took ~90 s in all, the
            # greedy check ~20 s of it (512 tokens at ~32 tok/s)
            mm["s"] = (t(verdict[g][0]) - t(stamp) - 20) if g in verdict and verdict[g][1] == "PASS" else None
        if run == "r193" and g == 2:
            mm["s"] = t(R193_XID_TIME) - t(stamp) + 1
            pt["xid"] = [{"after_s": t(R193_XID_TIME) - t(stamp), "line": R193_XID}]
        pt["mmvq"] = mm
        points.append(pt)
    sc = {
        "note": f"replay of {run.upper()} (2026-11-17, rig 1) from its raw output by test/make_replay_fixtures.py; "
                "the stock round of the replay is simulated: its check times (mmvq 70 s, the full 516-check sweep; "
                "greedy 20 s) are those of the passing cards at the offsets",
        "cards": 9, "stock_peaks": [1860, 1860, 1911, 1911, 1885, 1911, 1885, 1911, 1898],
        "greedy_s": 20, "mmvq_s": 70, "points": points,
    }
    with open(os.path.join(fixtures, f"{run}_replay.json"), "w") as f:
        json.dump(sc, f, indent=1)
    print(f"{run}: {len(points)} cards -> {os.path.join(fixtures, run + '_replay.json')}")


def stock_windows(run_dir, fixtures, boots=("climb", "try1", "try2", "try3", "try4")):
    """R200's stock rounds on its five boots (the climb and the four applies): every watchdog sample of the two stock
    rounds (each card's P-state, SM and memory clock; nothing else), and each boot's round times and the stock peaks
    the kit read then (round 2 alone), from the boot's climb.log. run_dir holds one folder per boot, each with the
    watchdog's wd*.csv.gz and out/climb.log. Writes r200_stock.csv.gz and r200_stock.json."""
    import glob
    cols = ["time"] + [f"g{g}_{k}" for g in range(9) for k in ("pstate", "sm_mhz", "mem_mhz")]
    rows, meta = [], []
    for b in boots:
        log = open(os.path.join(run_dir, b, "out", "climb.log")).read().splitlines()
        i1 = max(i for i, ln in enumerate(log) if "stock: round 1 " in ln)
        i2 = next(i for i in range(i1, len(log)) if "stock: round 2 " in log[i])
        ends = [i for i in range(i2, len(log)) if re.search(r"stock: card \d+ hash", log[i])]
        peaks = {int(re.search(r"card (\d+) hash", log[i]).group(1)): int(re.search(r"stock peak (\d+) MHz", log[i]).group(1))
                 for i in ends[:9]}
        win = {"boot": b, "round1": log[i1][:19], "round2": log[i2][:19], "end": log[ends[0]][:19],
               "kit_read_mhz": [peaks[g] for g in range(9)]}
        meta.append(win)
        t0, t1 = t(win["round1"]) - 2, t(win["end"]) + 2
        wd = sorted(glob.glob(os.path.join(run_dir, b, "wd*.csv.gz")))[0]
        idx = None
        with gzip.open(wd, "rt") as f:
            for ln in f:
                if ln.startswith("#"):
                    continue
                v = ln.rstrip("\n").split(",")
                if idx is None:
                    idx = {k: i for i, k in enumerate(v)}
                    continue
                try:
                    ts = t(v[0][:19])
                except ValueError:
                    continue
                if t0 <= ts <= t1:
                    rows.append(",".join([v[0]] + [v[idx[c]] for c in cols[1:]]))
    path = os.path.join(fixtures, "r200_stock.csv.gz")
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as g:
        g.write(("\n".join([",".join(cols)] + rows) + "\n").encode())
    note = ("R200's stock rounds on its five boots (2026-11-17, rig 2), from the watchdog's samples and the runs' "
            "climb.log by test/make_replay_fixtures.py --stock-windows; kit_read_mhz is the stock peak the kit read "
            "then, from round 2 alone")
    with open(os.path.join(fixtures, "r200_stock.json"), "w") as f:
        json.dump({"note": note, "boots": meta}, f, indent=1)
    print(f"r200 stock windows: {len(meta)} boots, {len(rows)} samples -> {path}")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--stock-windows":
        stock_windows(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "fixtures"))
        return 0
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    fixtures = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "fixtures")
    build("r193", sys.argv[1], fixtures)
    build("r194", sys.argv[2], fixtures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
