#!/usr/bin/env python3
"""Q219: a Pascal mix of Qwen3.8-Flash-Next whose routed experts are allocated per 8-layer group across the published
GSQ builds, with Q167's method (docs/data/2026-10-27_ornith_alloc: one switch at a time scored for KLD, the costs
added, the best set within a byte budget).

The start is pmix-q2-fast (Swift Q2_0's experts, the dense weights in the P106's fastest types). A switch moves one
class of one 8-layer group to a donor file's tensors, byte for byte (scripts/gguf_compose.py):
  gu<d><g>     the routed experts' gate and up of group g from donor d
  dn<d><g>     the routed experts' down of group g from donor d
  dense<d><g>  every other tensor of group g's layers (attention, recurrent, shared expert, router, hyper-connections,
               norms) from donor d
The donors (--donor KEY=DUMP_NAME; defaults): iq3 = pmix-iq3-fast (Swift IQ3_XXS's routed experts byte for byte, its
dense weights in the P106's types), iq2 = swift-iq2_xs. Swift Q2_0's experts are the start's own. A group's options
for one class exclude each other (gu from iq3 or iq2, not both). Other switch sets (e.g. one dense class at a time,
C326) come in a --spec TSV of the same columns as `specs` prints.

  galloc.py specs [--donors iq3,iq2] [--dense iq3]                    the switches: name, class, group, layers, donor,
                                                                       regex (tab separated)
  galloc.py switches --dumps DIR [--base pmix-q2-fast] [--spec TSV] [--donor KEY=NAME ...] [--eff TSV] > switches.tsv
        per switch from bench/file_choice/dump.py's header dumps: tensors, tensors whose type changes, the bytes it
        adds on the cards and the est. ms a decoded token it adds (bench/file_choice's model, timed and expected
        kernels); the last row (name `base`) holds the start's bytes on the cards and est. ms
  galloc.py same --regex RE --base SHARD... --donor SHARD...          the tensors the regex matches that are byte for
                                                                       byte the same in both files (a switch whose
                                                                       tensors all are is a no-op)
  galloc.py collect OUT.tsv NAME=LOGPREFIX...                         mean KLD, its error and same top token per text
                                                                       from llama-perplexity logs LOGPREFIX.<text>.log
  galloc.py solve --switches TSV --kld TSV --out TSV [--budgets 44.3e9,47e9] [--texts agent,wiki,code]
        [--base pmix-q2-fast] [--ms-budget MS] [--weights 1,1,1]
        per budget (bytes on the cards, absolute), the set of switches with the least predicted damage: the sum over
        texts of each switch's KLD change relative to the start's KLD on that text (Q167's score, a weight per text),
        exactly, over every combination of each group's options (a Pareto front of bytes, damage and ms merged group
        by group); a switch without all texts scored is left out. Writes OUT and OUT's -switches.tsv (cost per switch)
  galloc.py final --switches TSV --picks NAME,...                     the gguf_compose.py steps that build a chosen
                                                                       set, one compose per donor in turn: step,
                                                                       donor file, its --take patterns (tab separated)
Standard library only.
"""
import argparse, collections, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "file_choice"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
from model import gbps, load_eff, role  # noqa: E402

GROUPS = [(0, 7), (8, 15), (16, 23), (24, 31), (32, 39), (40, 47)]
CLASSES = {
    "gu": r"ffn_(?:gate|up)_exps\.weight$",
    "dn": r"ffn_down_exps\.weight$",
    "dense": r"(?!ffn_(?:gate|up|down)_exps\.)",
}
DONORS = {"iq3": "pmix-iq3-fast", "iq2": "swift-iq2_xs"}
COLS = ("name", "class", "group", "layers", "donor", "regex")


def group_regex(lo, hi, pat):
    layers = "|".join(str(i) for i in range(lo, hi + 1))
    return rf"^blk\.(?:{layers})\.{pat}"


def builtin_specs(donors=("iq3", "iq2"), dense=("iq3",)):
    out = []
    for d in donors:
        for cls in ("gu", "dn"):
            for g, (lo, hi) in enumerate(GROUPS):
                out.append((f"{cls}{d}{g}", cls, str(g), f"{lo}-{hi}", d, group_regex(lo, hi, CLASSES[cls])))
    for d in dense:
        for g, (lo, hi) in enumerate(GROUPS):
            out.append((f"dense{d}{g}", "dense", str(g), f"{lo}-{hi}", d, group_regex(lo, hi, CLASSES["dense"])))
    return out


def read_tsv(path):
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in f if line.strip()]


def write_tsv(path, header, rows):
    f = sys.stdout if path == "-" else open(path, "w")
    f.write("\t".join(header) + "\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
    if f is not sys.stdout:
        f.close()


def load_specs(path):
    if not path:
        return [dict(zip(COLS, s)) for s in builtin_specs()]
    return read_tsv(path)


def dump(dumps, name):
    return json.load(open(os.path.join(dumps, f"{name}.json")))


def token_ms(eff, d):
    """name -> (bytes on the cards, est. timed ms, est. expected ms) a decoded token, bench/file_choice's model."""
    ne, nu = d["keys"]["expert_count"], d["keys"]["expert_used_count"]
    out = {}
    for name, typ, _dims, size, _shard in d["tensors"]:
        rl = role(name)
        if rl == "host":
            out[name] = (0, 0.0, 0.0)
            continue
        b = size * nu / ne if rl.startswith("routed") else size
        g = gbps(eff, typ, rl)
        out[name] = (size, b / 1e6 / g[0], b / 1e6 / g[1])
    return out


def cmd_specs(a):
    write_tsv("-", COLS, builtin_specs(a.donors.split(","), [x for x in a.dense.split(",") if x]))


def cmd_switches(a):
    eff = load_eff(a.eff)
    donors = dict(DONORS)
    donors.update(dict(x.split("=", 1) for x in a.donor))
    base = dump(a.dumps, a.base)
    bt = {n: t for n, t, _d, _s, _sh in base["tensors"]}
    bc = token_ms(eff, base)
    rows = []
    cache = {}
    for s in load_specs(a.spec):
        dn = donors.get(s["donor"], s["donor"])
        if dn not in cache:
            d = dump(a.dumps, dn)
            cache[dn] = ({n: t for n, t, _d, _s, _sh in d["tensors"]}, token_ms(eff, d))
        dt, dc = cache[dn]
        pat = re.compile(s["regex"])
        names = [n for n in bt if pat.search(n)]
        assert names, f"{s['name']}: {s['regex']} matches no tensor of {a.base}"
        missing = [n for n in names if n not in dt]
        assert not missing, f"{s['name']}: {dn} has no {missing[0]}"
        changed = [n for n in names if dt[n] != bt[n]]
        db = sum(dc[n][0] - bc[n][0] for n in names)
        dms = [sum(dc[n][i] - bc[n][i] for n in names) for i in (1, 2)]
        types = collections.Counter(f"{bt[n]}>{dt[n]}" for n in changed)
        rows.append((s["name"], s["class"], s["group"], s["layers"], s["donor"], dn, len(names), len(changed), db,
                     f"{dms[0]:.3f}", f"{dms[1]:.3f}", " ".join(f"{k}:{v}" for k, v in sorted(types.items())) or "-",
                     s["regex"]))
    on_card = sum(v[0] for v in bc.values())
    ms = [a.nonweight + sum(v[i] for v in bc.values()) for i in (1, 2)]
    rows.append(("base", "base", "", "", "", a.base, len(bt), 0, on_card, f"{ms[0]:.3f}", f"{ms[1]:.3f}", "-", ""))
    write_tsv("-", ("name", "class", "group", "layers", "donor", "donor_file", "tensors", "type_changes",
                    "delta_bytes", "delta_ms_timed", "delta_ms_expected", "types", "regex"), rows)


def cmd_same(a):
    import hashlib
    from gguf_compose import Model
    pat = re.compile(a.regex)
    base, donor = Model(a.base), Model(a.donor)
    same = diff = 0
    for name, (h, dims, typ, off) in base.where.items():
        if not pat.search(name):
            continue
        h2, dims2, typ2, off2 = donor.where[name]
        if (dims, typ) != (dims2, typ2):
            diff += 1
            print(f"{name}\tdiffers (type {typ} / {typ2})")
            continue
        sums = []
        for hh, o in ((h, off), (h2, off2)):
            m = hashlib.md5()
            with open(hh.path, "rb") as f:
                f.seek(hh.data0 + o)
                left = min(hh.slot[name], h.slot[name], h2.slot[name])
                while left:
                    buf = f.read(min(left, 1 << 24))
                    m.update(buf)
                    left -= len(buf)
            sums.append(m.hexdigest())
        if sums[0] == sums[1]:
            same += 1
        else:
            diff += 1
        print(f"{name}\t{'same' if sums[0] == sums[1] else 'differs'}")
    print(f"# {same} the same, {diff} differ")


def parse_log(path):
    try:
        text = open(path, errors="replace").read()
    except OSError:
        return None
    k = re.search(r"Mean\s+KLD:\s+(-?[0-9.]+)\s+±\s+([0-9.]+)", text)
    s = re.search(r"Same top p:\s+([0-9.]+)\s+±\s+([0-9.]+)\s*%", text)
    if not k:
        return None
    return (k.group(1), k.group(2), s.group(1) if s else "", s.group(2) if s else "")


def cmd_collect(a):
    rows = []
    for item in a.items:
        name, prefix = item.split("=", 1)
        for text in ("wiki", "agent", "code"):
            r = parse_log(f"{prefix}.{text}.log")
            if r:
                rows.append((name, text, *r))
    write_tsv(a.out, ("name", "text", "kld", "kld_se", "same_top_pct", "same_top_se"), rows)


def pareto(states):
    """States (bytes, damage, ms, picks) no other state beats in all three (ties keep the first)."""
    states.sort(key=lambda s: (s[0], s[1], s[2]))
    keep = []
    for s in states:
        if any(k[0] <= s[0] and k[1] <= s[1] and k[2] <= s[2] for k in keep):
            continue
        keep.append(s)
    return keep


def cmd_solve(a):
    texts = a.texts.split(",")
    weights = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(texts)
    sw = read_tsv(a.switches)
    base = [r for r in sw if r["name"] == "base"][0]
    on_card, base_ms = int(base["delta_bytes"]), float(base["delta_ms_timed"])
    k = {(r["name"], r["text"]): (float(r["kld"]), float(r["kld_se"])) for r in read_tsv(a.kld)}
    ref = {t: k[(a.base, t)][0] for t in texts}
    cost = {}
    for r in sw:
        if r["name"] == "base" or not all((r["name"], t) in k for t in texts):
            continue
        dk = [k[(r["name"], t)][0] - ref[t] for t in texts]
        cost[r["name"]] = (int(r["delta_bytes"]), sum(w * d / ref[t] for w, d, t in zip(weights, dk, texts)),
                           float(r["delta_ms_timed"]), dk, r)
    # each group's options: one switch or none per class
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for n, c in cost.items():
        groups[c[4]["group"]][c[4]["class"]].append(n)
    states = [(0, 0.0, 0.0, ())]
    for g in sorted(groups):
        opts = [(0, 0.0, 0.0, ())]
        for names in groups[g].values():
            choices = [(0, 0.0, 0.0, ())] + [(*cost[n][:3], (n,)) for n in names]
            opts = [(o[0] + c[0], o[1] + c[1], o[2] + c[2], o[3] + c[3]) for o in opts for c in choices]
        states = pareto([(s[0] + o[0], s[1] + o[1], s[2] + o[2], s[3] + o[3]) for s in states for o in opts])
    rows = []
    budgets = [float(x) for x in a.budgets.split(",")]
    for budget in budgets:
        ok = [s for s in states if on_card + s[0] <= budget and (a.ms_budget is None or base_ms + s[2] <= a.ms_budget)]
        if not ok:
            print(f"budget {budget:.4g}: no set of switches fits", file=sys.stderr)
            continue
        b, d, m, picks = min(ok, key=lambda s: (s[1], s[0]))
        pred = [ref[t] + sum(cost[n][3][i] for n in picks) for i, t in enumerate(texts)]
        se = [sum(k[(n, t)][1] ** 2 for n in picks) ** 0.5 for t in texts]
        rows.append((f"{budget:.0f}", ",".join(sorted(picks, key=lambda n: (cost[n][4]["group"], n))) or "-", b,
                     on_card + b, f"{d:.4f}", f"{m:.2f}", f"{base_ms + m:.2f}",
                     *(f"{p:.5f}" for p in pred), *(f"{x:.5f}" for x in se)))
    write_tsv(a.out, ("budget_bytes", "switches", "delta_bytes", "on_card_bytes", "damage", "delta_ms", "est_ms",
                      *(f"pred_{t}" for t in texts), *(f"switch_se_{t}" for t in texts)), rows)
    per = os.path.splitext(a.out)[0] + "-switches.tsv"
    write_tsv(per, ("name", "class", "group", "delta_bytes", "delta_ms", "damage", *(f"dkld_{t}" for t in texts),
                    "damage_per_gb"),
              [(n, c[4]["class"], c[4]["group"], c[0], f"{c[2]:.3f}", f"{c[1]:.4f}", *(f"{x:.5f}" for x in c[3]),
                f"{c[1] / (c[0] / 1e9):.4f}" if c[0] else "") for n, c in sorted(cost.items())])
    print(open(a.out).read())


def cmd_final(a):
    sw = {r["name"]: r for r in read_tsv(a.switches)}
    picks = [p for p in a.picks.split(",") if p]
    by = collections.defaultdict(list)
    for p in picks:
        by[sw[p]["donor_file"]].append(sw[p]["regex"])
    for i, (dn, regexes) in enumerate(by.items()):
        print(f"{i + 1}\t{dn}\t" + "\t".join(f"re:{r}" for r in regexes))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("specs")
    p.add_argument("--donors", default="iq3,iq2")
    p.add_argument("--dense", default="iq3")
    p = sub.add_parser("switches")
    p.add_argument("--dumps", required=True)
    p.add_argument("--base", default="pmix-q2-fast")
    p.add_argument("--spec")
    p.add_argument("--donor", action="append", default=[])
    p.add_argument("--eff", default=os.path.join(HERE, "..", "file_choice", "efficiency.tsv"))
    p.add_argument("--nonweight", type=float, default=33.48)
    p = sub.add_parser("same")
    p.add_argument("--regex", required=True)
    p.add_argument("--base", nargs="+", required=True)
    p.add_argument("--donor", nargs="+", required=True)
    p = sub.add_parser("collect")
    p.add_argument("out")
    p.add_argument("items", nargs="+")
    p = sub.add_parser("solve")
    p.add_argument("--switches", required=True)
    p.add_argument("--kld", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budgets", default="44.3e9,47e9")
    p.add_argument("--texts", default="agent,wiki,code")
    p.add_argument("--weights")
    p.add_argument("--base", default="pmix-q2-fast")
    p.add_argument("--ms-budget", type=float)
    p = sub.add_parser("final")
    p.add_argument("--switches", required=True)
    p.add_argument("--picks", required=True)
    a = ap.parse_args()
    {"specs": cmd_specs, "switches": cmd_switches, "same": cmd_same, "collect": cmd_collect, "solve": cmd_solve,
     "final": cmd_final}[a.cmd](a)


if __name__ == "__main__":
    main()
