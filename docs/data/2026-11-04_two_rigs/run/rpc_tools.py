#!/usr/bin/env python3
"""Helpers of cpu_rpc_check.sh (standard library only):
  rpc_tools.py compare DIR REF "ARM ARM ..."   tokens and top-5 probabilities of every request DIR/<arm>.<req>.json
                                               against DIR/<REF>.<req>.json: SAME (identical tokens), DIFFER at the
                                               first other token, or MISSING; with the largest probability difference
  rpc_tools.py ops SERVER_LOG                  from a GGML_SCHED_DEBUG=2 log: the ops of the last graph (one decoded
                                               token) per split backend, then each split's inputs and their sizes
  rpc_tools.py bytes SS_TXT                    client socket counters before and after each request: bytes and
                                               segments per decoded token per rpc-server ((n=65) - (n=1)) / 64
  rpc_tools.py cmds DIR                        the client's RPC commands (GGML_RPC_DEBUG=3 lines of DIR/p<P>-n<N>.rpc.log)
                                               per decoded token: count, bytes sent and received, and how many waited
                                               for a reply (one network round trip each), with their time on loopback
"""
import collections, json, math, os, re, sys

REQS = ["R1", "R2a", "R2b", "R3"]


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def probs(j):
    out = []
    for step in j.get("completion_probabilities") or []:
        top = step.get("top_logprobs") or step.get("top_probs") or []
        d = {}
        for t in top:
            p = t["prob"] if "prob" in t else math.exp(t["logprob"])
            d[t.get("id", t.get("token"))] = p
        out.append(d)
    return out


def compare(d, ref, arms):
    for arm in arms.split():
        for r in REQS:
            a, b = load(os.path.join(d, f"{ref}.{r}.json")), load(os.path.join(d, f"{arm}.{r}.json"))
            if not a or not b or "tokens" not in a or "tokens" not in b:
                print(f"{arm} {r} MISSING")
                continue
            ta, tb = a["tokens"], b["tokens"]
            pa, pb = probs(a), probs(b)
            diff = 0.0
            for x, y in zip(pa, pb):
                for k in set(x) & set(y):
                    diff = max(diff, abs(x[k] - y[k]))
            if ta == tb:
                print(f"{arm} {r} SAME n={len(ta)} max_prob_diff={diff:.2e}")
            else:
                i = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
                same = sum(1 for x, y in zip(ta, tb) if x == y)
                print(f"{arm} {r} DIFFER first_at={i} same={same}/{max(len(ta), len(tb))} max_prob_diff={diff:.2e}")


STAMP = re.compile(r"\d+\.\d+\.\d+\.\d+ [DIWE] ")  # llama-server's per-message log prefix
SPLIT = re.compile(r"## SPLIT #(\d+): (.+?) # (\d+) inputs(.*)$")
NODE = re.compile(r"node #\s*\d+ \(\s*([^)]*?)\):")
INPUT = re.compile(r"\[(.*?) \(\s*(\d+[KMGB]?)\)\]")


def ops(path):
    graphs, cur, backend = [], None, None
    with open(path, errors="replace") as f:
        for line in f:
            line = STAMP.sub("", line)
            m = SPLIT.search(line)
            if m:
                if m.group(1) == "0":
                    cur = {"ops": collections.Counter(), "inputs": []}
                    graphs.append(cur)
                backend = re.sub(r"\[.*\]$", "", m.group(2)) if m.group(2).startswith("CPU") else m.group(2)
                for name, size in INPUT.findall(m.group(4)):
                    cur["inputs"].append((m.group(1), backend, name, size))
                continue
            m = NODE.search(line)
            if m and cur is not None and backend is not None:
                cur["ops"][(backend, m.group(1).strip())] += 1
    if not graphs:
        return
    g = graphs[-1]
    print("backend\top\tnodes")
    for (b, o), n in sorted(g["ops"].items()):
        print(f"{b}\t{o}\t{n}")
    print("#split\tbackend\tinput\tsize")
    for s, b, name, size in g["inputs"]:
        print(f"#{s}\t{b}\t{name}\t{size}")
    print(f"#graphs_printed\t{len(graphs)}")


FIELDS = {"sent": r"bytes_(?:sent|acked):(\d+)", "received": r"bytes_received:(\d+)",
          "segs_out": r"segs_out:(\d+)", "segs_in": r"segs_in:(\d+)"}


def counters(rest):
    out = {}
    for k, pat in FIELDS.items():
        vals = [int(v) for v in re.findall(pat, rest)]
        if k == "sent":  # bytes_sent and bytes_acked both match: keep one per socket
            vals = [int(v) for v in re.findall(r"bytes_sent:(\d+)", rest)] or \
                   [int(v) for v in re.findall(r"bytes_acked:(\d+)", rest)]
        out[k] = sum(vals)
    return out


def bytes_(path):
    snap = {}
    with open(path) as f:
        for line in f:
            p = line.split(None, 4)
            if len(p) < 4:
                continue
            when, plen, n, port = p[:4]
            snap[(when, plen, n, port)] = counters(p[4] if len(p) > 4 else "")
    print("server_port\tprompt_tokens\tsent_per_token\treceived_per_token\tsegs_out_per_token\tsegs_in_per_token"
          "\tprompt_sent\tprompt_received")
    keys = sorted({(k[1], k[3]) for k in snap}, key=lambda x: (int(x[0]), x[1]))
    for plen, port in keys:
        try:
            d = {n: {k: snap[("after", plen, n, port)][k] - snap[("before", plen, n, port)][k] for k in FIELDS}
                 for n in ("1", "65")}
        except KeyError:
            continue
        per = {k: (d["65"][k] - d["1"][k]) / 64 for k in FIELDS}
        print(f"{port}\t{plen}\t{per['sent']:.0f}\t{per['received']:.0f}\t{per['segs_out']:.1f}\t{per['segs_in']:.1f}"
              f"\t{d['1']['sent']}\t{d['1']['received']}")


CMD = re.compile(r"\[work\] (\w+) \(in: ([\d.]+) (\w+), out: ([\d.]+) (\w+)\)")
DONE = re.compile(r"\[work\] (\w+) done in ([\d.]+) ms")
UNIT = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}


def cmd_totals(path):
    t = collections.defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0])  # count, in, out, with reply, ms
    with open(path, errors="replace") as f:
        for line in f:
            m = CMD.search(line)
            if m:
                c = t[m.group(1)]
                out = float(m.group(4)) * UNIT[m.group(5)]
                c[0] += 1
                c[1] += float(m.group(2)) * UNIT[m.group(3)]
                c[2] += out
                c[3] += out > 0
                continue
            m = DONE.search(line)
            if m:
                t[m.group(1)][4] += float(m.group(2))
    return t


def cmds(d):
    print("prompt_tokens\tcommand\tper_token\tbytes_out_per_token\tbytes_back_per_token\twith_reply_per_token"
          "\tms_per_token_loopback")
    for plen in ("32", "160"):
        try:
            a = cmd_totals(os.path.join(d, f"p{plen}-n1.rpc.log"))
            b = cmd_totals(os.path.join(d, f"p{plen}-n65.rpc.log"))
        except OSError:
            continue
        tot = [0.0] * 5
        for name in sorted(set(a) | set(b)):
            per = [(b[name][i] - a[name][i]) / 64 for i in range(5)]
            tot = [x + y for x, y in zip(tot, per)]
            print(f"{plen}\t{name}\t{per[0]:.2f}\t{per[1]:.0f}\t{per[2]:.0f}\t{per[3]:.2f}\t{per[4]:.3f}")
        print(f"{plen}\tall\t{tot[0]:.2f}\t{tot[1]:.0f}\t{tot[2]:.0f}\t{tot[3]:.2f}\t{tot[4]:.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "compare":
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "ops":
        ops(sys.argv[2])
    elif cmd == "bytes":
        bytes_(sys.argv[2])
    elif cmd == "cmds":
        cmds(sys.argv[2])
