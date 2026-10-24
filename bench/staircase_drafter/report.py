"""Markdown tables and gate verdicts from score.py's summary.json and sim.py's results (Q20/D1).

usage: python report.py SUMMARY.json [SIM.json]
Gates (pipeline_drafter.md section 6): head A or B with coupled p >= 0.55 on fresh text (>= 0.6 overall) goes to the
rig (both conditions); the best head below 0.5 on fresh text kills learned guessers; C is kept if within 0.05 of A; U - A is what
staleness costs, A - T what Flint's features buy.
"""
import json
import sys

ROWS = [("A", "stair", "A: staircase on card 1 (six cards)"), ("A", "miss", "A: after a miss"),
        ("A", "pre", "A on pre-step inputs"), ("A", "nine", "A: nine-card staircase"),
        ("A", "nine_light", "A: nine-card staircase, light last card"),
        ("B", "pre", "B: pre-step (spare card)"), ("B", "late", "B: late pre-step"), ("B", "miss", "B: after a miss"),
        ("B", "stair", "B on card-1 inputs"), ("B", "nine", "B: nine-card staircase"),
        ("B", "nine_light", "B: nine-card staircase, light last card"),
        ("C", "plain", "C: card 1's output only"), ("U", "miss", "U: older tokens complete"),
        ("T", "plain", "T: tokens only")]


def fmt(x):
    return f"{x[0]:.3f} ({x[1]:.3f}–{x[2]:.3f})"


def main():
    s = json.load(open(sys.argv[1]))
    print("| Head and inputs | Coupled, fresh | Coupled, all | Coupled, copied | Top guess, fresh | "
          "Fresh: chat / code / edit / agent | Reasoning / answer |")
    print("|---|---|---|---|---|---|---|")
    get = {}
    for head, win, label in ROWS:
        if head not in s or win not in s[head]["windows"]:
            continue
        r = s[head]["windows"][win]
        get[(head, win)] = r
        cats = " / ".join(f"{r[f'coupled_fresh_{c}'][0]:.3f}" for c in ("chat", "code", "edit", "agent"))
        segs = " / ".join(f"{r[k][0]:.3f}" if k in r else "–" for k in ("coupled_think", "coupled_answer"))
        print(f"| {label} | {fmt(r['coupled_fresh'])} | {fmt(r['coupled'])} | {r['coupled_copy'][0]:.3f} | "
              f"{r['top_fresh'][0]:.3f} | {cats} | {segs} |")
    print()
    best_ab = max(((h, w) for h, w in get if h in "AB" and w in ("stair", "pre", "late")),
                  key=lambda k: get[k]["coupled_fresh"][0], default=None)
    if best_ab:
        r = get[best_ab]
        go = r["coupled_fresh"][0] >= 0.55 and r["coupled"][0] >= 0.6
        print(f"- Gate to the rig: best of A/B is {best_ab[0]} ({best_ab[1]}) at {r['coupled_fresh'][0]:.3f} fresh, "
              f"{r['coupled'][0]:.3f} overall: {'GO' if go else 'no go'} (needs >= 0.55 fresh and >= 0.6 overall)")
    best = max(get, key=lambda k: get[k]["coupled_fresh"][0], default=None)
    if best:
        v = get[best]["coupled_fresh"][0]
        print(f"- Kill check: best head {best[0]} ({best[1]}) {v:.3f} fresh: {'kill' if v < 0.5 else 'no kill'}")
    pairs = [(("C", "plain"), ("A", "stair"), "C - A (cross-card features)"),
             (("U", "miss"), ("A", "stair"), "U - A (staleness)"),
             (("A", "stair"), ("T", "plain"), "A - T (Flint's features)"),
             (("A", "miss"), ("U", "miss"), "A after a miss - U (cost of one head for both cases)"),
             (("B", "pre"), ("A", "stair"), "B pre-step - A")]
    for a, b, label in pairs:
        if a in get and b in get:
            d = get[a]["coupled_fresh"][0] - get[b]["coupled_fresh"][0]
            d_all = get[a]["coupled"][0] - get[b]["coupled"][0]
            print(f"- {label}: {d:+.3f} fresh, {d_all:+.3f} overall")
    if len(sys.argv) > 2:
        sim = json.load(open(sys.argv[2]))
        keys = ("chat", "code", "edit", "agent", "all", "edit/answer")
        print("\n| Setup | Guesser | Chat | Code | Edits | Agent | All | File rewrites | tok/s, all |")
        print("|---|---|---|---|---|---|---|---|---|")
        for cfg, r in sim.items():
            setup, guesser = cfg.split(" | ", 1)
            print(f"| {setup} | {guesser} | " + " | ".join(f"{r['speedup'][k]:.2f}" for k in keys) +
                  f" | {r['tok_s_all']} |")


if __name__ == "__main__":
    main()
