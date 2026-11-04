#!/usr/bin/env python3
# split_plan_test.py DIR: a self-test of split_plan.py (beside it). Writes header-only GGUF files (the header, then the
# file extended to its full size, as split_plan.py reads only headers and offsets) of random qwen3next-like models in
# whole MiB: per layer a norm, an attention or recurrent core and three routed-expert tensors, token embeddings (host
# only), an output norm and head, over two shards. Every plan the tool prints is re-added from its own -dev, -ts and -ot
# (each card's MiB and free) and compared with brute force over every split (every cut of the layers, or of the units
# with --split-experts). Checks:
#   (1) the default search and --ts scoring find the brute force's best smallest free on 3, 4 and 5 cards of mixed
#       6, 8 and 10 GB, with compute buffers per card, --last-extra and --extra-any (the piece on the best card);
#   (2) --fewest scores every card count as brute force does, frees the cards after the first k of --devices (in that
#       order), checks --own-card on the first free card, picks the fewest cards that keep --margin and prints their
#       -dev and -ts, or, when only -ot moves keep it, the fewest moves that do;
#   (3) 12 cards of three sizes (the any-card-count rule), whole layers, and the same output with the shards given in
#       reverse order;
#   (4) bench/file_choice/fit.py still parses the default output (its plan() on the same model).
# Negative controls: planted bugs in copies of split_plan.py must each fail a check: the output layer left off the
# last card, the margin ignored, the free cards taken from the front of --devices, the --extra-any piece fixed on the
# first card, the -ts cut one layer early, the model's keys taken from the first shard given. Prints "ok" or fails an assertion. Uses DIR for its files and removes them.
# Standard library only.
import itertools, os, random, re, shutil, struct, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(here, "split_plan.py")
MiB, ARCH, KINDS = 1 << 20, "qwen3next", ("gate", "up", "down")
PLAN = re.compile(r"^#(\d+): (?:-dev (\S+) )?-ts (\S+)(?: -ot (\S+))?  smallest free (-?\d+) MiB"
                  r"(?:; the (\d+) MiB piece on CUDA(\d+))?$")
ROW = re.compile(r"^  card (\d+): layers (\d+)-(\d+), (\d+) attention.*, (-?\d+) MiB, (-?\d+) free$")
OT = re.compile(r"blk\\\.(\d+)\\\.ffn_(gate|up|down)_exps=CUDA(\d+)")
BUGS = [  # (what, text in split_plan.py, its replacement)
    ("output left off", "extra[-1] += out + a.last_extra", "extra[-1] += a.last_extra"),
    ("margin ignored", "if ok_own and ((w and w[0][0] >= a.margin)", "if ok_own and ((w and w[0][0] >= 0)"),
    ("free cards from the front", "used, freed = devices[:k], devices[k:]", "used, freed = devices[-k:], devices[:-k]"),
    ("piece on the first card", "extra[at] += a.extra_any", "extra[0] += a.extra_any"),
    ("cut a layer early", "self.core_at[sum(counts[:i + 1])]", "self.core_at[sum(counts[:i + 1]) - 1]"),
    ("keys from the first shard given", 'kv = kv if "general.architecture" in kv else k', "kv = kv or k"),
]


def gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_shard(path, n_layer, interval, tensors, no):
    """A header-only GGUF of F32 tensors of whole MiB: [(name, MiB)], laid out in order; as in llama.cpp's split files,
    only the first shard (no 0) holds the model's keys, the others their split number."""
    kvs = [gstr("general.architecture") + struct.pack("<I", 8) + gstr(ARCH),
           gstr(f"{ARCH}.block_count") + struct.pack("<II", 4, n_layer),
           gstr(f"{ARCH}.full_attention_interval") + struct.pack("<II", 4, interval)] if no == 0 else \
        [gstr("split.no") + struct.pack("<IH", 2, no)]
    infos, off = b"", 0
    for name, mib in tensors:
        infos += gstr(name) + struct.pack("<IQQIQ", 2, 262144, mib, 0, off)
        off += mib * MiB
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs) + infos
    with open(path, "wb") as f:
        f.write(head)
        f.truncate((len(head) + 31) // 32 * 32 + off)


def make_model(d, rng, n_layer, interval, big=1, layers=None, out=None):
    m = {"n_layer": n_layer, "interval": interval, "out": out or rng.randint(100, 600) * big,
         "layers": layers or [{"core": rng.randint(20, 150) * big, **{k: rng.randint(40, 300) * big for k in KINDS}}
                              for _ in range(n_layer)]}
    t = [[("token_embd.weight", rng.randint(100, 700))], []]
    for i, L in enumerate(m["layers"]):
        s = t[0] if i < n_layer // 2 else t[1]
        s += [(f"blk.{i}.attn_norm.weight", 1), (f"blk.{i}.attn_qkv.weight", L["core"] - 1)]
        s += [(f"blk.{i}.ffn_{k}_exps.weight", L[k]) for k in KINDS]
    t[1] += [("output_norm.weight", 1), ("output.weight", m["out"] - 1)]
    os.makedirs(d, exist_ok=True)
    m["files"] = [os.path.join(d, f"m-0000{j + 1}-of-00002.gguf") for j in range(2)]
    for no, (path, tensors) in enumerate(zip(m["files"], t)):
        write_shard(path, n_layer, interval, tensors, no)
    return m


def attn(m, i):
    return (i + 1) % m["interval"] == 0


def layer_mib(m, i, c):
    L = m["layers"][i]
    return L["core"] + sum(L[k] for k in KINDS) + (c["kv"] if attn(m, i) else c["state"])


def units(m, c, split):
    out = []
    for i, L in enumerate(m["layers"]):
        if split:
            out += [(i, None, L["core"] + (c["kv"] if attn(m, i) else c["state"]))] + [(i, k, L[k]) for k in KINDS]
        else:
            out.append((i, None, layer_mib(m, i, c)))
    return out


def room(m, c, used, at):
    comp = [c["compute"][min(j, len(c["compute"]) - 1)] for j in range(len(used))]
    return [c["free"][d] - comp[j] - (m["out"] + c["last"] if j == len(used) - 1 else 0)
            - (c["any"] if j == at else 0) for j, d in enumerate(used)]


def splits(us, k):
    """Every cut of the units into k runs that each hold a layer core, as (runs, moved tensors)."""
    for cuts in itertools.combinations(range(1, len(us)), k - 1):
        b = (0,) + cuts + (len(us),)
        runs = [us[b[j]:b[j + 1]] for j in range(k)]
        if all(any(u[1] is None for u in r) for r in runs):
            yield runs, sum(next(p for p, u in enumerate(r) if u[1] is None) for r in runs)


def ref_best(m, c, used, split, target=None):
    """Brute force: the largest smallest free over every split (and every card for the --extra-any piece), or with
    target the fewest moved tensors among the splits that leave every card target MiB (None if none)."""
    best = None
    for at in range(len(used)) if c["any"] else [None]:
        r = room(m, c, used, at)
        for runs, mv in splits(units(m, c, split), len(used)):
            w = min(r[j] - sum(u[2] for u in run) for j, run in enumerate(runs))
            if target is None:
                best = w if best is None else max(best, w)
            elif w >= target:
                best = mv if best is None else min(best, mv)
    return best


def tool(t, m, c, *more):
    args = ["--cards", len(c["free"]), "--free", ",".join(map(str, c["free"])), "--kv-mib", c["kv"],
            "--state-mib", c["state"], "--compute", ",".join(map(str, c["compute"])), "--last-extra", c["last"]]
    args += ["--extra-any", c["any"]] if c["any"] else []
    p = subprocess.run([sys.executable, "-I", t] + [str(x) for x in args + list(more)] + m["files"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout.splitlines()


def check_plans(m, c, lines):
    """Re-add every printed plan from its -dev, -ts and -ot; returns [(dev list, smallest free, moves)]."""
    plans = []
    for n, line in enumerate(lines):
        p = PLAN.match(line)
        if not p:
            continue
        cards = [int(x[4:]) for x in p.group(2).split(",")] if p.group(2) else list(range(len(c["free"])))
        counts = [int(x) for x in p.group(3).split(",")]
        counts[-1] -= 1
        layer_card = [j for j, k in enumerate(counts) for _ in range(k)]
        assert len(layer_card) == m["n_layer"] and min(counts) >= 1, line
        used = [0] * len(cards)
        for i in range(m["n_layer"]):
            used[layer_card[i]] += layer_mib(m, i, c)
        ot = OT.findall(p.group(4) or "")
        for L, kind, d in ot:
            used[layer_card[int(L)]] -= m["layers"][int(L)][kind]
            used[cards.index(int(d))] += m["layers"][int(L)][kind]
        for j in range(len(cards)):
            used[j] += c["compute"][min(j, len(c["compute"]) - 1)]
        used[-1] += m["out"] + c["last"]
        if p.group(7):
            assert int(p.group(6)) == c["any"], line
            used[cards.index(int(p.group(7)))] += c["any"]
        else:
            assert not c["any"], line
        frees = [c["free"][d] - u for d, u in zip(cards, used)]
        assert int(p.group(5)) == min(frees), (line, frees)
        for j, d in enumerate(cards):
            r = ROW.match(lines[n + 1 + j])
            assert r and (int(r.group(1)), int(r.group(5)), int(r.group(6))) == (d, used[j], frees[j]), (lines[n + 1 + j], used[j])
        plans.append((cards, min(frees), len(ot)))
    return plans


def costs(rng, cards):
    return {"free": [rng.choice([6023, 8000, 10100]) - rng.randint(0, 40) for _ in range(cards)], "kv": rng.randint(20, 160),
            "state": rng.randint(1, 15), "compute": [rng.randint(50, 300), rng.randint(50, 400)],
            "last": rng.choice([0, 0, rng.randint(50, 900)]), "any": rng.choice([0, rng.randint(30, 700)])}


def check_default(t, d, rng):
    for case in range(6):
        cards = 3 + case % 3
        m = make_model(f"{d}/def{case}", rng, rng.randint(cards + 1, 7), rng.choice([2, 3, 4]))
        for split in (False, True):
            c = costs(rng, cards)
            lines = tool(t, m, c, *(["--split-experts"] if split else []), "--top", 2)
            plans = check_plans(m, c, lines)
            assert len(plans) == 2 and plans[0][1] == ref_best(m, c, list(range(cards)), split), lines
            ts = PLAN.match(next(x for x in lines if x.startswith("#1:"))).group(3)
            scored = check_plans(m, c, tool(t, m, c, "--ts", ts))
            assert len(scored) == 1 and scored[0][1] <= plans[0][1], (ts, scored)
            if not split:
                assert scored[0][1] == plans[0][1], (ts, scored)


def check_ot_case(t, d, rng):
    """Three 400 MiB layers on cards of 802 MiB with a 150 MiB margin: whole layers need three cards, two hold them
    only with layer 1's up and down tensors moved to the second card (exactly two moves)."""
    m = make_model(f"{d}/ot", rng, 3, 4, layers=[{"core": 100, "gate": 100, "up": 100, "down": 100}] * 3, out=2)
    c = {"free": [802] * 3, "kv": 0, "state": 0, "compute": [0], "last": 0, "any": 0}
    lines = tool(t, m, c, "--fewest", "--margin", 150, "--split-experts")
    assert any(x.startswith("fewest: 2 cards (CUDA0,CUDA1), 1 free (CUDA2)") for x in lines), lines
    plans = check_plans(m, c, lines)
    assert [(p[0], p[1], p[2]) for p in plans] == [([0, 1], 200, 2)], (plans, lines)
    lines = tool(t, m, c, "--fewest", "--margin", 150)  # whole layers only: two cards keep 2 MiB, so three
    assert any(x.startswith("fewest: 3 cards (CUDA0,CUDA1,CUDA2), 0 free") for x in lines), lines
    assert [p[:3] for p in check_plans(m, c, lines)] == [([0, 1, 2], 400, 0)], lines


def check_fewest(t, d, rng, n_cases=10):
    seen_ot = seen_own = 0
    check_ot_case(t, d, rng)
    seen_ot += 1
    for case in range(n_cases):
        cards = 3 + case % 3
        m = make_model(f"{d}/few{case}", rng, rng.randint(cards + 1, 6), rng.choice([2, 3, 4]))
        c = costs(rng, cards)
        devices = list(range(cards))
        rng.shuffle(devices)
        split, margin = case % 2 == 1, rng.choice([100, 300, 500])
        own = rng.choice([0, rng.randint(500, 4000), 99999])
        more = ["--fewest", "--margin", margin, "--devices", ",".join(map(str, devices))] + \
            (["--split-experts"] if split else []) + (["--own-card", own] if own else [])
        lines = tool(t, m, c, *more)
        h = lines.index("cards\tfree_cards\town_free\twhole_free\twhole_ts\tot_free\tot_moved")
        rows = [x.split("\t") for x in lines[h + 1:] if re.match(r"^\d+\t", x)]
        assert [int(r[0]) for r in rows] == list(range(cards, 0, -1)), rows
        chosen = None
        for r in rows:
            k = int(r[0])
            used, freed = devices[:k], devices[k:]
            assert r[1] == (",".join(f"CUDA{x}" for x in freed) or "-"), r
            own_free = (c["free"][freed[0]] - own if freed else None) if own else None
            assert r[2] == ("-" if not own else "none" if own_free is None else str(own_free)), r
            w = ref_best(m, c, used, False)
            assert int(r[3]) == w, (r, w)
            o = ref_best(m, c, used, True) if split else None
            assert r[5] == ("-" if not split else str(o)), (r, o)
            if (not own or (own_free is not None and own_free >= margin)) and (w >= margin or (split and o >= margin)):
                chosen = k
        if chosen is None:
            assert lines[-1] == f"no card count keeps {margin} MiB free on every card", lines
            continue
        head = next(x for x in lines if x.startswith("fewest: "))
        assert head.startswith(f"fewest: {chosen} cards ({','.join(f'CUDA{x}' for x in devices[:chosen])})"), head
        seen_own += bool(own)
        plans = check_plans(m, c, lines)
        assert plans and all(p[0] == devices[:chosen] and p[1] >= margin for p in plans), (plans, lines)
        whole_ok = ref_best(m, c, devices[:chosen], False) >= margin
        if whole_ok:
            assert all(p[2] == 0 for p in plans), lines
        else:
            seen_ot += 1
            assert plans[0][2] == ref_best(m, c, devices[:chosen], True, margin), (plans, lines)
    return seen_ot, seen_own


def check_twelve(t, d, rng):
    m = make_model(f"{d}/twelve", rng, 16, 4, big=3)
    c = {"free": [6023] * 4 + [8000] * 4 + [10100] * 4, "kv": 161, "state": 12, "compute": [344, 419], "last": 0, "any": 0}
    devices = list(range(11, -1, -1))  # the big cards first
    lines = tool(t, m, c, "--fewest", "--margin", 300, "--devices", ",".join(map(str, devices)))
    head = next(x for x in lines if x.startswith("fewest: "))
    k = int(head.split()[1])
    assert ref_best(m, c, devices[:k], False) >= 300 and (k == 1 or ref_best(m, c, devices[:k - 1], False) < 300), lines
    plans = check_plans(m, c, lines)
    assert plans and plans[0][0] == devices[:k], plans
    m["files"] = m["files"][::-1]  # the shards in any order: the model's keys come from the shard that holds them
    assert tool(t, m, c, "--fewest", "--margin", 300, "--devices", ",".join(map(str, devices))) == lines


def check_fit_parse(d, rng):
    sys.path.insert(0, os.path.join(here, "..", "bench", "file_choice"))
    import fit  # noqa: E402  (imports this folder's split_plan.py)
    m = make_model(f"{d}/fit", rng, 6, 4)
    c = {"free": [6023] * 4, "kv": 81, "state": 3, "compute": [60, 70], "last": 300, "any": 0}
    base = ["--cards", "4", "--free", "6023", "--kv-mib", "81", "--state-mib", "3", "--compute", "60,70",
            "--last-extra", "300"] + m["files"]
    for more in ([], ["--split-experts"]):
        free, ts, ot = fit.plan(base + more)
        p = PLAN.match(next(x for x in tool(TOOL, m, c, *more) if x.startswith("#1:")))
        assert (free, ts, ot) == (int(p.group(5)), p.group(3), p.group(4) or ""), (free, ts, ot)


def run_checks(t, d):
    rng = random.Random(20261104)
    check_default(t, d, rng)
    seen_ot, seen_own = check_fewest(t, d, rng)
    check_twelve(t, d, rng)
    return seen_ot, seen_own


def main():
    d = os.path.abspath(sys.argv[1])
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    seen_ot, seen_own = run_checks(TOOL, f"{d}/real")
    assert seen_ot and seen_own, f"the cases never needed -ot moves ({seen_ot}) or an own card ({seen_own})"
    check_fit_parse(f"{d}/real", random.Random(1))
    for what, old, new in BUGS:
        src = open(TOOL).read()
        assert src.count(old) == 1, what
        bug = f"{d}/bug/split_plan.py"
        os.makedirs(os.path.dirname(bug), exist_ok=True)
        open(bug, "w").write(src.replace(old, new))
        try:
            run_checks(bug, f"{d}/bugrun")
        except AssertionError:
            print(f"planted bug caught: {what}")
        else:
            raise AssertionError(f"planted bug not caught: {what}")
    shutil.rmtree(d)
    print("ok")


if __name__ == "__main__":
    main()
