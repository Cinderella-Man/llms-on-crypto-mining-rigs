#!/usr/bin/env python3
"""C803 (Q1407): why the MTP draft's graph needs a new plan, on a CPU, from the allocator's trace (LLAMA_UBATCH_TRACE=1).

  draft_trace.py arm NAME OUT PORT SERVER MODEL [VAR=value ...] -- SERVER_ARGS...
      starts SERVER on 127.0.0.1:PORT with the variables given and LLAMA_UBATCH_TRACE=1, sends FN84's pattern of
      requests (one conversation whose prompt grows by 1,549 (no reply), 896, 3,313, 302, 30 and 92 new tokens with 8
      greedy tokens each, then two fresh 2,049-token prompts with 8 each; token ids from a Park-Miller generator below the
      model's vocabulary), stops it, and writes OUT/NAME.server.log and OUT/NAME.replies.json (each reply's tokens)
  draft_trace.py parse NAME=server.log [...]
      one row per evaluation of the draft context (`ubatch-trace: mtp`) or a re-planned one of the target (`dec`): log,
      request, phase (prompt: before the request's first sampled token; gen: after), context, n_tokens, n_outputs, and
      the allocator's reason for a new plan (`galloc-trigger`: the first tensor larger than its place in the plan, or a
      changed node or leaf count), printed before that evaluation's trace line; on several cards also the scheduler's
      line for each re-plan (`sched-replan`: whether the cards changed, which the allocator's check then skips, and the
      drain's and the plan's microseconds), and the compute buffers allocated or grown before it (`galloc-grow`, MiB
      added; the load's and a per-size scheduler's plan count at the context's next evaluation); on stderr, per log
      and context: the evaluations, the re-plans in each phase, those after a change of cards, the drains' and plans'
      sums, the buffer allocations and growths, and the re-plans by reason
  draft_trace.py replies A.replies.json B.replies.json
      the same tokens in every reply: prints identical or DIFFER

The CPU has one compute buffer per context, so ggml-alloc re-plans by itself where the scheduler on several cards
syncs every card first and counts a reallocation (llama_context::sched_graph_realloc); the trace's trigger is the same
check (ggml_gallocr_needs_realloc). Standard library only.
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from collections import Counter

ANSI = re.compile(r"\x1b\[[0-9;]*m")
NEW = re.compile(r"\| task (\d+) \| new prompt, n_ctx_slot = \d+, n_keep = \d+, task\.n_tokens = (\d+)")
GEN1 = re.compile(r"\| task (\d+) \| .*n_gen = 1, ")
TRIG = re.compile(r"galloc-trigger: (.*)$")
REPLAN = re.compile(r"sched-replan: #\d+ backends_changed = (\d)(.*?) size_changed = (\d) .* drain = (\d+) plan = (\d+) us")
GROW = re.compile(r"galloc-grow: (\S+) buffer ([0-9.]+) -> ([0-9.]+) MiB")
UB = re.compile(r"ubatch-trace: (mtp|dec|enc) n_tokens = (\d+) n_seqs = \d+ n_outputs = (\d+) graph = (\w+)")
# FN84's requests (new tokens, tokens to generate); "fresh" starts an unrelated prompt
TURNS = [(1549, 0), (896, 8), (3313, 8), (302, 8), (30, 8), (92, 8)]
FRESH = [(2049, 8), (2049, 8)]


def toks(n, s, v):
    out, x = [], s
    for _ in range(n):
        x = (x * 16807) % 2147483647
        out.append(1 + x % (v - 1))
    return out


def http(port, path, body=None, timeout=1800):
    url = f"http://127.0.0.1:{port}{path}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def arm(argv):
    name, out, port, server, model = argv[:5]
    rest = argv[5:]
    sep = rest.index("--")
    envs = dict(e.split("=", 1) for e in rest[:sep])
    sargs = rest[sep + 1:]
    env = dict(os.environ)
    env.update(envs)
    env["LLAMA_UBATCH_TRACE"] = "1"
    log_path = os.path.join(out, f"{name}.server.log")
    cmd = [server, "-m", model, *sargs, "--host", "127.0.0.1", "--port", port]
    with open(log_path, "w") as log:
        log.write("=== " + " ".join(f"{k}={v}" for k, v in envs.items()) + " " + " ".join(cmd) + "\n")
        log.flush()
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    ok = False
    for _ in range(600):
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
    replies, rc, prompt, seed = [], 0, [], 3
    reqs = [("turn", n, g) for n, g in TURNS] + [("fresh", n, g) for n, g in FRESH]
    for i, (kind, n, g) in enumerate(reqs):
        seed += 2
        prompt = (prompt if kind == "turn" else []) + toks(n, seed, n_vocab)
        body = {"prompt": prompt, "n_predict": g, "temperature": 0, "top_k": 1, "seed": 1, "return_tokens": True,
                "cache_prompt": True}
        try:
            r = http(port, "/completion", body)
        except Exception as e:
            print(f"{name} request {i}: failed: {e}")
            rc = 1
            break
        got = r.get("tokens", [])
        replies.append(got)
        if kind == "turn":
            prompt = prompt + got
        t = r.get("timings", {})
        print(f"{name} request {i} ({kind}, {n} new tokens): n={len(got)} prompt_n={t.get('prompt_n')} "
              f"draft_n={t.get('draft_n')} accepted={t.get('draft_n_accepted')}")
    with open(os.path.join(out, f"{name}.replies.json"), "w") as f:
        json.dump(replies, f)
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill()
        print(f"{name}: killed")
    return rc


def reason(trig):
    # "node 12 'name' (OP) ne = [...] needs N bytes in CPU, planned M" -> "node 'name' (OP) needs > planned"
    m = re.match(r"(node|src|leaf) (\d+) '([^']*)'(?: of '([^']*)')? \(([^)]*)\) ne = (\[[^\]]*\]) needs (\d+) bytes"
                 r"(?: in \S+, planned (\d+)| ?, planned none)", trig)
    if not m:
        return trig.strip(), trig.strip()
    kind, _, tname, reader, op, ne, need, planned = m.groups()
    short = f"{kind} '{tname}'" + (f" of '{reader}'" if reader else "") + f" ({op})" + \
        (" planned none" if planned is None else " larger than planned")
    full = f"{short} ne = {ne} needs {need} B, planned {'none' if planned is None else planned + ' B'}"
    return short, full


def parse(argv):
    print("log\trequest\tphase\tcontext\tn_tokens\tn_outputs\tgraph\treplan\treason\tcards_changed\tdrain_us\tplan_us\tgrow_mib")
    for spec in argv:
        name, path = spec.split("=", 1)
        req, phase, task, pend, pend_rp, pend_grow = -1, "load", None, [], [], []
        stats = {}
        for raw in open(path, errors="replace"):
            line = ANSI.sub("", raw)
            m = NEW.search(line)
            if m:
                req, phase, task = req + 1, "prompt", int(m.group(1))
                continue
            m = GEN1.search(line)
            if m and task is not None and int(m.group(1)) == task and phase == "prompt":
                phase = "gen"
                continue
            m = TRIG.search(line)
            if m:
                pend.append(m.group(1))
                continue
            m = REPLAN.search(line)
            if m:
                pend_rp.append((int(m.group(1)), m.group(2).strip().removeprefix("(").removesuffix(")"), int(m.group(4)),
                                int(m.group(5))))
                continue
            m = GROW.search(line)
            if m:
                pend_grow.append(float(m.group(3)) - float(m.group(2)))
                continue
            m = UB.search(line)
            if m:
                ctx, n_tok, n_out, graph = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
                ctxn = "draft" if ctx == "mtp" else "target"
                st = stats.setdefault(ctxn, dict(evals=0, replan=Counter(), why=Counter(), cards=0, drain=0, plan=0,
                                                 grows=0, grow_mib=0.0))
                st["evals"] += 1
                replan = len(pend) > 0 or len(pend_rp) > 0
                cards = sum(1 for r in pend_rp if r[0])
                drain = sum(r[2] for r in pend_rp)
                plan = sum(r[3] for r in pend_rp)
                grow = sum(pend_grow)
                if replan:
                    st["replan"][phase] += 1
                    if pend:
                        short, full = reason(pend[0])
                    else:
                        short = full = "cards changed" + (f" ({pend_rp[0][1]})" if pend_rp[0][1] else "")
                    st["why"][short] += 1
                    st["cards"] += cards
                    st["drain"] += drain
                    st["plan"] += plan
                st["grows"] += len(pend_grow)
                st["grow_mib"] += grow
                if ctx == "mtp" or replan:
                    print(f"{name}\t{req}\t{phase}\t{ctxn}\t{n_tok}\t{n_out}\t{graph}\t{int(replan)}\t"
                          f"{full if replan else ''}\t{cards}\t{drain}\t{plan}\t{grow:.2f}")
                pend, pend_rp, pend_grow = [], [], []
        for ctxn, st in sorted(stats.items()):
            sys.stderr.write(f"{name}\t{ctxn}\tevaluations {st['evals']}\treplans prompt {st['replan']['prompt']} "
                             f"gen {st['replan']['gen']} load {st['replan']['load']}\tafter a change of cards "
                             f"{st['cards']}\tdrains {st['drain'] / 1000:.1f} ms\tplans {st['plan'] / 1000:.1f} ms\t"
                             f"buffer growths {st['grows']} ({st['grow_mib']:.2f} MiB)\n")
            for why, n in st["why"].most_common():
                sys.stderr.write(f"{name}\t{ctxn}\t  {n}\t{why}\n")
    return 0


def replies(argv):
    a, b = (json.load(open(p)) for p in argv[:2])
    print("identical" if a == b else "DIFFER", len(a), len(b), sum(len(x) for x in a), sum(len(x) for x in b))
    return 0 if a == b else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"arm": arm, "parse": parse, "replies": replies}.get(mode, lambda a: print(__doc__) or 2)(sys.argv[2:]))
