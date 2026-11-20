#!/usr/bin/env python3
"""R209 run 3 (Q1603): Ornith's whole-memory check at stock and with the offsets, round by round, and the cards'
heat, power and clocks from the watchdog's samples while each check's rounds ran.

    python3 -I model_rounds.py rounds MODEL_LOG > model_rounds.tsv
    python3 -I model_rounds.py summary MODEL_LOG > model_check.md
    python3 -I model_rounds.py cards MODEL_LOG WD_CSV > model_cards.md
    python3 -I model_rounds.py again EARLIER_LOG LATER_LOG > r211_model_check.md
    python3 -I model_rounds.py again EARLIER_LOG LATER_LOG PREVIOUS_LOG > r219_model_check.md
    python3 -I model_rounds.py whole WD_CSV > r211_watchdog.md

MODEL_LOG is model_check.sh's OUT/model.log (both tags in one file), WD_CSV the kit watchdog's WORK/wd.csv (a row
every 0.5 s). A check's window runs from its "ready" line to its "done" line; `cards` skips a tag the log lacks.
`again` (R211, Q1621) sets a later boot's check with the offsets (LATER_LOG, profile rounds only, compared with the
earlier check's stock reference) beside the earlier boot's stock and profile rounds (EARLIER_LOG); with PREVIOUS_LOG
(R219, Q1634), another later boot's check of the same kind goes beside it too. `whole` gives each card's highest
temperature and power over every watchdog sample, `apply`'s check servers included. Standard library only.
"""
import csv
import re
import statistics
import sys

ROUND = re.compile(r"^(\S+ \S+) (stock|profile) round (\d+): greedy \S+: hash (\w+), (\d+) tokens, prompt tok/s "
                   r"([\d.]+) ([\d.]+), generation tok/s ([\d.]+) ([\d.]+)(.*)\(exit (\d+)\)$")
READY = re.compile(r"^(\S+ \S+) (stock|profile): ready in (\d+) s; ([\d ]+) MiB used per card; (\d+) MiB host")
DONE = re.compile(r"^(\S+ \S+) (stock|profile): done, (\d+) rounds in (\d+) s, exit (\d+)$")


def parse(path):
    rounds, ready, done = [], {}, {}
    for ln in open(path):
        ln = ln.rstrip("\n")
        m = ROUND.match(ln)
        if m:
            t, tag, r, h, n, p1, p2, g1, g2, rest, ex = m.groups()
            same = "reference" if (tag == "stock" and r == "1") else ("same" if "SAME as stock1" in rest else "DIFFERS")
            rounds.append(dict(time=t, tag=tag, round=int(r), hash=h, tokens=int(n), prompt1=float(p1),
                               prompt2=float(p2), gen1=float(g1), gen2=float(g2), vs_stock1=same, exit=int(ex)))
            continue
        m = READY.match(ln)
        if m:
            ready[m[2]] = dict(time=m[1], load_s=int(m[3]), mib=m[4].split(), host=int(m[5]))
            continue
        m = DONE.match(ln)
        if m:
            done[m[2]] = dict(time=m[1], rounds=int(m[3]), secs=int(m[4]), exit=int(m[5]))
    return rounds, ready, done


def pct(a, b):
    return f"{(b / a - 1) * 100:+.2f} %"


def cmd_rounds(path):
    rounds, _, _ = parse(path)
    cols = ["tag", "round", "time", "hash", "tokens", "vs_stock1", "prompt1", "prompt2", "gen1", "gen2", "exit"]
    print("\t".join(cols))
    for r in rounds:
        print("\t".join(str(r[c]) for c in cols))


def cmd_summary(path):
    rounds, ready, done = parse(path)
    by = {t: [r for r in rounds if r["tag"] == t] for t in ("stock", "profile")}
    s, p = by["stock"], by["profile"]
    s2 = next(r for r in s if r["round"] == 2)
    p2 = next(r for r in p if r["round"] == 2)
    later = [r for r in p if r["round"] >= 2]

    def med(rs, k):
        return statistics.median(r[k] for r in rs)
    print("| | stock | with the offsets | change |")
    print("|---|---|---|---|")
    print(f"| load, s | {ready['stock']['load_s']} | {ready['profile']['load_s']} | |")
    print(f"| host memory available when loaded, MiB | {ready['stock']['host']:,} | {ready['profile']['host']:,} | |")
    print(f"| rounds; the same hash as stock's round 1 | {len(s)}; {sum(r['vs_stock1'] == 'same' for r in s)} "
          f"(round 1 is the reference) | {len(p)}; {sum(r['vs_stock1'] == 'same' for r in p)} | |")
    print(f"| the rounds' time, s | {done['stock']['secs']} | {done['profile']['secs']} | |")
    print(f"| hash | {s[0]['hash']} | {', '.join(sorted({r['hash'] for r in p}))} | |")
    for k, name in (("gen", "generation"), ("prompt", "prompt processing")):
        a1, b1 = s[0][k + "1"], p[0][k + "1"]
        a2, b2 = s[0][k + "2"], p[0][k + "2"]
        print(f"| {name} tok/s in round 1 (prompts 1, 2) | {a1:.2f}, {a2:.2f} | {b1:.2f}, {b2:.2f} | "
              f"{pct(a1, b1)}, {pct(a2, b2)} |")
        a1, a2, b1, b2 = s2[k + "1"], s2[k + "2"], p2[k + "1"], p2[k + "2"]
        print(f"| {name} tok/s in round 2 (prompts 1, 2) | {a1:.2f}, {a2:.2f} | {b1:.2f}, {b2:.2f} | "
              f"{pct(a1, b1)}, {pct(a2, b2)} |")
        m1, m2 = med(later, k + "1"), med(later, k + "2")
        lo1, hi1 = min(r[k + "1"] for r in later), max(r[k + "1"] for r in later)
        lo2, hi2 = min(r[k + "2"] for r in later), max(r[k + "2"] for r in later)
        print(f"| {name} tok/s, median of rounds 2 on (prompts 1, 2) | {a1:.2f}, {a2:.2f} (round 2 only) | "
              f"{m1:.2f}, {m2:.2f} ({len(later)} rounds; lowest {lo1:.2f}, {lo2:.2f}; highest {hi1:.2f}, {hi2:.2f}) | "
              f"{pct(a1, m1)}, {pct(a2, m2)} |")


def cmd_cards(path, wd):
    _, ready, done = parse(path)
    rows = list(csv.DictReader(ln for ln in open(wd) if not ln.startswith("#")))
    n = sum(1 for k in rows[0] if re.fullmatch(r"g\d+_c", k))
    print("| check | card | highest °C | highest W | samples above 80 W, of | SM clock median, MHz | highest SM MHz | "
          "memory clock median, MHz | offsets (memory, core) |")
    print("|---|---:|---:|---:|---|---:|---:|---:|---|")
    for tag in ("stock", "profile"):
        if tag not in ready or tag not in done:
            continue
        t0, t1 = ready[tag]["time"], done[tag]["time"]
        w = [r for r in rows if t0 <= r["time"][:19] <= t1]
        if not w:
            continue
        for g in range(n):
            c = [float(r[f"g{g}_c"]) for r in w]
            p = [float(r[f"g{g}_w"]) for r in w]
            sm = [int(r[f"g{g}_sm_mhz"]) for r in w]
            mem = [int(r[f"g{g}_mem_mhz"]) for r in w]
            offs = sorted({(r[f"g{g}_mem_off"], r[f"g{g}_gpc_off"]) for r in w})
            print(f"| {tag} | {g} | {max(c):.0f} | {max(p):.2f} | {sum(v > 80 for v in p)} of {len(p)} | "
                  f"{statistics.median(sm):,.0f} | {max(sm):,} | {statistics.median(mem):,.0f} | "
                  f"{'; '.join(f'+{a}, +{b}' for a, b in offs)} |")
    for tag in ("stock", "profile"):
        if tag not in ready or tag not in done:
            continue
        t0, t1 = ready[tag]["time"], done[tag]["time"]
        w = [r for r in rows if t0 <= r["time"][:19] <= t1]
        print(f"\n{tag}: {len(w)} samples, {t0} to {t1}" + (
            f"; tripped in any: {any(r['tripped'] not in ('0', '') for r in w)}" if w else
            " (the watchdog was not running yet)"))


def cmd_again(earlier, later, previous=None):
    e, e_ready, _ = parse(earlier)
    l, l_ready, l_done = parse(later)
    es = {r["round"]: r for r in e if r["tag"] == "stock"}
    ep = {r["round"]: r for r in e if r["tag"] == "profile"}
    lp = {r["round"]: r for r in l if r["tag"] == "profile"}
    if previous:
        v, v_ready, _ = parse(previous)
        vp = {r["round"]: r for r in v if r["tag"] == "profile"}

    def cells(prev_cell, delta=""):
        # the previous later boot's column, and its comparison column, only when PREVIOUS_LOG is given
        return (f" {prev_cell} |", f" {delta} |") if previous else ("", "")
    pc, pd = cells("with the offsets, previous later boot", "this boot against the previous later boot")
    print(f"| | stock, earlier boot | with the offsets, earlier boot |{pc} **with the offsets, this boot** | "
          f"this boot against the earlier boot's offsets |{pd} this boot against stock |")
    print("|---|---|---|---|---|---|" + ("---|---|" if previous else ""))
    pc, pd = cells(v_ready['profile']['load_s'] if previous else "")
    print(f"| load, s | {e_ready['stock']['load_s']} | {e_ready['profile']['load_s']} |{pc} "
          f"**{l_ready['profile']['load_s']}** | |{pd} |")
    pc, pd = cells(f"{v_ready['profile']['host']:,}" if previous else "")
    print(f"| host memory available when loaded, MiB | {e_ready['stock']['host']:,} | {e_ready['profile']['host']:,} |{pc} "
          f"**{l_ready['profile']['host']:,}** | |{pd} |")
    pc, pd = cells(' '.join(v_ready['profile']['mib']) if previous else "")
    print(f"| MiB used, cards 0 to 8 | {' '.join(e_ready['stock']['mib'])} | {' '.join(e_ready['profile']['mib'])} |{pc} "
          f"**{' '.join(l_ready['profile']['mib'])}** | |{pd} |")
    pc, pd = cells(f"{len(vp)}; {sum(r['vs_stock1'] == 'same' for r in vp.values())} of {len(vp)}" if previous else "")
    print(f"| rounds; the stock hash | {len(es)}; `{es[1]['hash']}` | {len(ep)}; "
          f"{sum(r['vs_stock1'] == 'same' for r in ep.values())} of {len(ep)} |{pc} "
          f"**{len(lp)}; {sum(r['vs_stock1'] == 'same' for r in lp.values())} of {len(lp)}** | |{pd} |")
    pc, pd = cells(', '.join('`' + vp[k]['hash'] + '`' for k in sorted(vp)) if previous else "")
    print(f"| hash in each round | {', '.join('`' + es[k]['hash'] + '`' for k in sorted(es))} | "
          f"{', '.join('`' + h + '`' for h in sorted({r['hash'] for r in ep.values()}))} in every round |{pc} "
          f"**{', '.join('`' + lp[k]['hash'] + '`' for k in sorted(lp))}** | |{pd} |")
    for k, name in (("gen", "generation"), ("prompt", "prompt processing")):
        for rnd in sorted(lp):
            a = [es[rnd][k + "1"], es[rnd][k + "2"]] if rnd in es else None
            b = [ep[rnd][k + "1"], ep[rnd][k + "2"]]
            c = [lp[rnd][k + "1"], lp[rnd][k + "2"]]
            d = [vp[rnd][k + "1"], vp[rnd][k + "2"]] if previous and rnd in vp else None
            pc, pd = cells(f"{d[0]:.2f}, {d[1]:.2f}" if d else "", f"{pct(d[0], c[0])}, {pct(d[1], c[1])}" if d else "")
            print(f"| {name} tok/s in round {rnd} (prompts 1, 2) | "
                  f"{(f'{a[0]:.2f}, {a[1]:.2f}') if a else ''} | {b[0]:.2f}, {b[1]:.2f} |{pc} **{c[0]:.2f}, {c[1]:.2f}** | "
                  f"{pct(b[0], c[0])}, {pct(b[1], c[1])} |{pd} "
                  f"{(f'{pct(a[0], c[0])}, {pct(a[1], c[1])}') if a else ''} |")
    print(f"\nThis boot's check: {len(lp)} rounds in {l_done['profile']['secs']} s, exit {l_done['profile']['exit']}.")


def cmd_whole(wd):
    rows = list(csv.DictReader(ln for ln in open(wd) if not ln.startswith("#")))
    n = sum(1 for k in rows[0] if re.fullmatch(r"g\d+_c", k))
    print(f"{len(rows)} samples a card, {rows[0]['time'][:19]} to {rows[-1]['time'][:19]}; "
          f"tripped in any: {any(r['tripped'] not in ('0', '') for r in rows)}\n")
    print("| card | highest °C | highest W | samples above 80 W | most samples above 80 W in a row |")
    print("|---:|---:|---:|---:|---:|")
    for g in range(n):
        c = [float(r[f"g{g}_c"]) for r in rows]
        w = [float(r[f"g{g}_w"]) for r in rows]
        run = best = 0
        for v in w:
            run = run + 1 if v > 80 else 0
            best = max(best, run)
        print(f"| {g} | {max(c):.0f} | {max(w):.2f} | {sum(v > 80 for v in w)} | {best} |")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "rounds":
        cmd_rounds(sys.argv[2])
    elif cmd == "summary":
        cmd_summary(sys.argv[2])
    elif cmd == "cards" and len(sys.argv) == 4:
        cmd_cards(sys.argv[2], sys.argv[3])
    elif cmd == "again" and len(sys.argv) in (4, 5):
        cmd_again(*sys.argv[2:])
    elif cmd == "whole":
        cmd_whole(sys.argv[2])
    else:
        sys.exit(__doc__)
