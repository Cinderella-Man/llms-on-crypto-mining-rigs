#!/usr/bin/env python3
"""C619 (Q1104): llama-server's MTP path on a CPU, the draft context's graph builds with LLAMA_MTP_GRAPH_KEEP off and on.

  server_smoke.py arm NAME OUT PORT SERVER MODEL [VAR=value ...] -- SERVER_ARGS...
      starts SERVER on 127.0.0.1:PORT with the environment variables given (and LLAMA_UBATCH_TRACE=1 and
      LLAMA_SPEC_DUMP=OUT/NAME.dump.jsonl), answers three requests (greedy 96 tokens; temperature 1.0 seed 42, 48
      tokens; greedy 64 tokens on a longer prompt), stops it, and writes OUT/NAME.R{1,2,3}.json, OUT/NAME.server.log
      and OUT/NAME.counts.json: the draft context's micro-batches from the trace ("ubatch-trace: mtp ..."), rebuilt and
      reused, those of up to 8 tokens apart, why each was rebuilt, and the verify steps (the dump's lines)
  server_smoke.py compare OUT GROUP=ARM,ARM,... [...]
      within each group, every arm against the first: each reply's tokens and draft counts, and the dump without its
      "run" field; prints a TSV (group, arm, check, verdict, detail) and a Markdown table of the counts

Standard library only. The prompts are token ids from a Park-Miller generator below the model's vocabulary, as in
2026-11-11_c405_server_gap/run/cpu_check.sh.
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request


def toks(n, s, v):
    out, x = [], s
    for _ in range(n):
        x = (x * 16807) % 2147483647
        out.append(1 + x % (v - 1))
    return out


def http(port, path, body=None, timeout=600):
    url = f"http://127.0.0.1:{port}{path}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


TRACE = re.compile(r"ubatch-trace: mtp n_tokens = (\d+) n_seqs = \d+ n_outputs = (\d+) graph = (rebuilt|reused)(?: why = (\S+))?")


def counts(log_path, dump_path):
    c = {"mtp": 0, "rebuilt": 0, "reused": 0, "small": 0, "small_rebuilt": 0, "small_reused": 0,
         "small_rebuilt_out0": 0, "small_rebuilt_out1": 0, "why": {}, "steps": 0}
    with open(log_path, errors="replace") as f:
        for line in f:
            m = TRACE.search(line)
            if not m:
                continue
            n_tok, n_out, graph, why = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
            c["mtp"] += 1
            c[graph] += 1
            if n_tok <= 8:
                c["small"] += 1
                c["small_" + graph] += 1
                if graph == "rebuilt":
                    c["small_rebuilt_out1" if n_out > 0 else "small_rebuilt_out0"] += 1
            if graph == "rebuilt":
                key = f"{'small' if n_tok <= 8 else 'large'}:{why}"
                c["why"][key] = c["why"].get(key, 0) + 1
    if os.path.exists(dump_path):
        with open(dump_path) as f:
            c["steps"] = sum(1 for line in f if line.strip())
    return c


def arm(argv):
    name, out, port, server, model = argv[:5]
    rest = argv[5:]
    sep = rest.index("--")
    envs = dict(e.split("=", 1) for e in rest[:sep])
    sargs = rest[sep + 1:]
    env = dict(os.environ)
    env.update(envs)
    env["LLAMA_UBATCH_TRACE"] = "1"
    dump = os.path.join(out, f"{name}.dump.jsonl")
    if os.path.exists(dump):
        os.remove(dump)
    env["LLAMA_SPEC_DUMP"] = dump
    log_path = os.path.join(out, f"{name}.server.log")
    cmd = [server, "-m", model, *sargs, "--host", "127.0.0.1", "--port", port]
    with open(log_path, "w") as log:
        log.write("=== " + " ".join(f"{k}={v}" for k, v in envs.items()) + " " + " ".join(cmd) + "\n")
        log.flush()
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    ok = False
    for _ in range(240):
        if p.poll() is not None:
            break
        try:
            if http(port, "/health", timeout=5).get("status") == "ok":
                ok = True
                break
        except Exception:
            pass
        time.sleep(0.5)
    if not ok:
        print(f"{name}: server did not start (rc {p.poll()})")
        p.kill()
        return 1
    n_vocab = http(port, "/v1/models")["data"][0]["meta"]["n_vocab"]
    a, b, c = toks(40, 3, n_vocab), toks(30, 5, n_vocab), toks(20, 9, n_vocab)
    p1 = a + b + a + c  # a block repeated: something a drafter can copy
    p3 = p1 + b
    reqs = [("R1", p1, 96, 0, 1), ("R2", p1, 48, 1.0, 42), ("R3", p3, 64, 0, 1)]
    rc = 0
    for rid, prompt, n, temp, seed in reqs:
        body = {"prompt": prompt, "n_predict": n, "temperature": temp, "top_k": 20, "top_p": 0.95, "min_p": 0,
                "seed": seed, "return_tokens": True, "cache_prompt": False}
        try:
            r = http(port, "/completion", body)
        except Exception as e:
            print(f"{name} {rid}: request failed: {e}")
            rc = 1
            continue
        with open(os.path.join(out, f"{name}.{rid}.json"), "w") as f:
            json.dump(r, f)
        t = r.get("timings", {})
        print(f"{name} {rid} n={len(r.get('tokens', []))} draft_n={t.get('draft_n')} accepted={t.get('draft_n_accepted')}")
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
        print(f"{name}: killed")
    cnt = counts(log_path, dump)
    with open(os.path.join(out, f"{name}.counts.json"), "w") as f:
        json.dump(cnt, f, indent=1, sort_keys=True)
    print(f"{name} counts {json.dumps(cnt, sort_keys=True)}")
    return rc


def load_dump(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                d.pop("run", None)
                rows.append(d)
    return rows


def compare(argv):
    out = argv[0]
    groups = [g.split("=", 1) for g in argv[1:]]
    bad = 0
    table = []
    for gname, arms_s in groups:
        arms = arms_s.split(",")
        ref = arms[0]
        for a in arms:
            for rid in ("R1", "R2", "R3"):
                f = os.path.join(out, f"{a}.{rid}.json")
                g = os.path.join(out, f"{ref}.{rid}.json")
                try:
                    ra, rr = json.load(open(f)), json.load(open(g))
                    same_tok = ra["tokens"] == rr["tokens"]
                    ca = [ra["timings"].get("draft_n"), ra["timings"].get("draft_n_accepted")]
                    cr = [rr["timings"].get("draft_n"), rr["timings"].get("draft_n_accepted")]
                    v = "identical" if same_tok and ca == cr else "DIFFER"
                    det = f"n={len(ra['tokens'])} draft {ca[0]} accepted {ca[1]} (ref {cr[0]} {cr[1]})"
                except Exception as e:
                    v, det = "DIFFER", f"missing or unreadable: {e}"
                bad += v != "identical"
                print(f"{gname}\t{a}\treply-{rid}\t{v}\t{det}")
            try:
                da = load_dump(os.path.join(out, f"{a}.dump.jsonl"))
                dr = load_dump(os.path.join(out, f"{ref}.dump.jsonl"))
                v = "identical" if da == dr and len(da) > 0 else "DIFFER"
                det = f"{len(da)} steps (ref {len(dr)})"
            except Exception as e:
                v, det = "DIFFER", f"missing or unreadable: {e}"
            bad += v != "identical"
            print(f"{gname}\t{a}\tdump\t{v}\t{det}")
            c = json.load(open(os.path.join(out, f"{a}.counts.json")))
            table.append((gname, a, c))
    print()
    print("| group | arm | verify steps | draft micro-batches | rebuilt | of up to 8 tokens: rebuilt / reused | rebuilt, no output / one output | rebuilt per step (up to 8 tokens) | why rebuilt |")
    print("|---|---|---:|---:|---:|---|---|---:|---|")
    for gname, a, c in table:
        per = c["small_rebuilt"] / c["steps"] if c["steps"] else float("nan")
        why = ", ".join(f"{k} {v}" for k, v in sorted(c["why"].items()))
        print(f"| {gname} | {a} | {c['steps']} | {c['mtp']} | {c['rebuilt']} | {c['small_rebuilt']} / {c['small_reused']} | "
              f"{c['small_rebuilt_out0']} / {c['small_rebuilt_out1']} | {per:.3f} | {why} |")
    print()
    print(f"compare: {bad} differences")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("arm", "compare"):
        print(__doc__)
        sys.exit(2)
    sys.exit(arm(sys.argv[2:]) if sys.argv[1] == "arm" else compare(sys.argv[2:]))
