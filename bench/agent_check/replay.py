"""Re-send recorded agent requests to a server, in order, for speed comparisons on identical prompts.

    python3 replay.py list --log RESULTS/proxy.jsonl
    python3 replay.py pick --log RESULTS/proxy.jsonl --results RESULTS --arm v2-ud --n 8 [--tasks 1-41] [--shortest]
                       [--task-set polyglot|bugfix]
    python3 replay.py run --log RESULTS/proxy.jsonl --traces ARM/TASK/a1/t1[,…] --server http://127.0.0.1:PORT
                      --label v1-ud --out REPLAY.jsonl [--stream recorded|on|off] [--ignore-eos] [--erase-slot]
    python3 replay.py summary REPLAY.jsonl [REPLAY2.jsonl …] [--reference v2-ud]

A trace is one attempt's try as the proxy logged it (tag prefix "<arm>/<task>/a<k>/t<n>"). `run` sends each of its
requests to SERVER's /v1/chat/completions exactly as recorded (messages, tools, sampling), except:
- `max_tokens` is the recorded reply's generated length (llama-server's predicted_n), so every server generates the
  same amount, at most; with --ignore-eos exactly that amount;
- `seed` is fixed per trace and request (CRC32 of the trace plus 7919 × index), the same for every server, so two
  servers sample from the same random numbers;
- `stream` as recorded (pi always streams) unless --stream on/off;
- with --drop-reasoning, the earlier replies' reasoning_content is removed from the history (what keeping the
  reasoning costs in prefill; a speed variant only).
Requests run one at a time, traces in the given order, each trace's requests in their recorded order. With --erase-slot
the server's slot 0 is erased before each trace (needs llama-server's /slots endpoint), so no trace starts from the
previous one's cache; by default the cache carries over as it would for a user. A trace already replayed in full
under the same label in --out is skipped, so an interrupted replay resumes (a partly replayed trace runs again from
its start; `summary` keeps the last row per label, trace and request). A replayed reply never matches the
recorded one, so from the second request on each turn reuses the cache only up to the end of the previous prompt.

`pick` chooses N traces of one arm's scored attempts (optionally within task positions A-B of the run order) that
span its range of live minutes: the attempts at the quantiles (2i + 1) / 2N of live time, so a replay set has short,
middle and long tasks; with --shortest, the N shortest (a pilot's bounded replay time).

`summary` reports per label: requests, ms per prompt token (Σ prompt_ms / Σ prompt_n), ms per generated token
(Σ predicted_ms / Σ predicted_n), the fixed cost per request (client time minus prefill and decode), mean cache_n and
time to first token; then each label's ratio to --reference (default: the first label) with a 95 % bootstrap interval
over traces, and the "same work" minutes: the reference's token counts priced at each label's per-token and per-request
costs. The live rows of the same traces (from the proxy log) appear as label "live:<arm>".
"""
import argparse
import gzip
import json
import random
import re
import sys
import time
import urllib.request
import zlib
from collections import defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import StreamStats, request_facts  # noqa: E402

TRACE_RE = re.compile(r"^(.+/a\d+/t\d+)/(\d+)$")


def load_log(paths):
    """Rows of the proxy logs with a tag, grouped by trace and ordered by turn; plus the bodies by id."""
    rows, bodies = [], {}
    for p in paths:
        p = Path(p)
        rows += [json.loads(l) for l in p.open() if l.strip()]
        bp = Path(str(p) + ".bodies.gz")
        if bp.exists():
            for l in gzip.open(bp, "rt"):
                d = json.loads(l)
                bodies[d["id"]] = d
    traces = defaultdict(list)
    for r in rows:
        m = TRACE_RE.match(r.get("tag") or "")
        if m:
            traces[m.group(1)].append((int(m.group(2)), r))
    return {k: [r for _, r in sorted(v, key=lambda x: (x[0], x[1]["id"]))] for k, v in traces.items()}, bodies


def send(server, body, timeout):
    data = json.dumps(body).encode()
    req = urllib.request.Request(server.rstrip("/") + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    st = StreamStats(t0)
    status, err = None, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            if body.get("stream"):
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    st.feed(chunk)
            else:
                st.finish_json(resp.read())
    except Exception as e:  # noqa: BLE001 - recorded, the replay goes on
        err = f"{type(e).__name__}: {e}"
    row = st.summary()
    row.update(status=status, t_end=round(time.monotonic() - t0, 4))
    if err:
        row["error"] = err
    return row


def cmd_list(a):
    traces, _ = load_log(a.log)
    print("| trace | requests | prompt tokens processed | cached | generated | live minutes |")
    print("|---|---|---|---|---|---|")
    for k, rows in sorted(traces.items()):
        s = lambda f: sum(r.get(f) or 0 for r in rows)
        print(f"| {k} | {len(rows)} | {s('prompt_n')} | {s('cache_n')} | {s('predicted_n')} | "
              f"{s('t_end') / 60:.1f} |")


def cmd_run(a):
    traces, bodies = load_log(a.log)
    names = [t for t in a.traces.split(",") if t] if a.traces else []
    if a.traces_file:
        names += [l.strip() for l in a.traces_file.read_text().splitlines() if l.strip()]
    done = defaultdict(set)
    if a.out.exists():
        for l in a.out.open():
            if l.strip():
                r = json.loads(l)
                if r["label"] == a.label and not r.get("error"):
                    done[r["trace"]].add(r["idx"])
    out = a.out.open("a")
    for name in names:
        if name not in traces:
            sys.exit(f"trace {name} not in the logs (see `replay.py list`)")
        if len(done[name]) >= len(traces[name]):
            print(f"{a.label} {name}: already replayed, skipped", flush=True)
            continue
        if a.erase_slot:
            try:
                urllib.request.urlopen(urllib.request.Request(a.server.rstrip("/") + "/slots/0?action=erase",
                                                              data=b"", method="POST"), timeout=30).read()
            except Exception as e:  # noqa: BLE001
                print(f"slot erase failed: {e}", file=sys.stderr)
        base_seed = zlib.crc32(name.encode())
        for i, rec in enumerate(traces[name]):
            b = bodies.get(rec["id"])
            if not b or not isinstance(b["body"], dict):
                print(f"{name} #{i}: no body recorded, skipped", file=sys.stderr)
                continue
            body = dict(b["body"])
            n_gen = rec.get("predicted_n") or (rec.get("usage") or {}).get("completion_tokens")
            if n_gen:
                body["max_tokens"] = int(n_gen)
            if a.ignore_eos:
                body["ignore_eos"] = True
            body["seed"] = (base_seed + 7919 * i) % (2 ** 31)
            if a.stream != "recorded":
                body["stream"] = a.stream == "on"
            if not body.get("stream"):
                body.pop("stream_options", None)
            if a.drop_reasoning:
                body["messages"] = [{k: v for k, v in m.items() if k != "reasoning_content"}
                                    if m.get("role") == "assistant" else m for m in body["messages"]]
            r = send(a.server, body, a.timeout)
            row = dict(label=a.label, trace=name, idx=i, rec_id=rec["id"], t_wall=time.time(), **request_facts(body),
                       **r, rec={k: rec.get(k) for k in ("prompt_n", "prompt_ms", "cache_n", "predicted_n",
                                                         "predicted_ms", "t_first_token", "t_end")})
            row.pop("timings", None)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{a.label} {name} #{i}: prompt {r.get('prompt_n')} (cache {r.get('cache_n')}) "
                  f"gen {r.get('predicted_n')} in {r['t_end']:.1f} s{' ERROR ' + r['error'] if r.get('error') else ''}",
                  flush=True)


def cmd_pick(a):
    traces, _ = load_log(a.log)
    order = [l.strip() for l in (Path(__file__).resolve().parent / "tasks" / a.task_set / "order.txt").read_text().split()]
    allowed = set(order)
    if a.tasks:
        lo, hi = map(int, a.tasks.split("-"))
        allowed = set(order[lo - 1: hi])
    cands = []
    for p in sorted(Path(a.results, a.arm).glob("*/a*/result.json")):
        r = json.loads(p.read_text())
        if r.get("status") == "scored" and r["task_id"] in allowed and r["tag"] in traces:
            cands.append((sum(q.get("t_end") or 0 for q in traces[r["tag"]]), r["tag"]))
    cands.sort()
    if a.shortest:
        picked = [t for _, t in cands[: a.n]]
    elif len(cands) <= a.n:
        picked = [t for _, t in cands]
    else:
        picked = []
        for i in range(a.n):
            j = min(len(cands) - 1, int((2 * i + 1) / (2 * a.n) * len(cands)))
            while cands[j][1] in picked:
                j = (j + 1) % len(cands)
            picked.append(cands[j][1])
    print(",".join(picked))


def costs(rows):
    pn = sum(r.get("prompt_n") or 0 for r in rows)
    pm = sum(r.get("prompt_ms") or 0 for r in rows)
    gn = sum(r.get("predicted_n") or 0 for r in rows)
    gm = sum(r.get("predicted_ms") or 0 for r in rows)
    fixed = [r["t_end"] - ((r.get("prompt_ms") or 0) + (r.get("predicted_ms") or 0)) / 1000 for r in rows
             if r.get("t_end") is not None]
    return dict(n=len(rows), prompt_n=pn, gen_n=gn, ms_prompt=pm / pn if pn else float("nan"),
                ms_gen=gm / gn if gn else float("nan"), fixed_s=sum(fixed) / len(fixed) if fixed else float("nan"),
                cache_n=sum(r.get("cache_n") or 0 for r in rows) / max(1, len(rows)),
                ttft=median([r["t_first_token"] for r in rows if r.get("t_first_token") is not None] or [float("nan")]),
                wall_min=sum(r.get("t_end") or 0 for r in rows) / 60)


def same_work_min(ref_rows, c):
    """Minutes the reference's requests would take at cost set c."""
    tot = 0.0
    for r in ref_rows:
        tot += c["fixed_s"] + (r.get("prompt_n") or 0) * c["ms_prompt"] / 1000 + \
            (r.get("predicted_n") or 0) * c["ms_gen"] / 1000
    return tot / 60


def load_replay(files):
    """Replay rows without errors, the last one per label, trace and request."""
    last = {}
    for p in files:
        for l in Path(p).open():
            if l.strip():
                r = json.loads(l)
                if not r.get("error"):
                    last[(r["label"], r["trace"], r["idx"])] = r
    return list(last.values())


def cmd_summary(a):
    rows = load_replay(a.files)
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    traces = sorted({r["trace"] for r in rows})
    if a.log:
        live, _ = load_log(a.log)
        for t in traces:
            for r in live.get(t, []):
                by["live:" + t.split("/")[0]].append(dict(r, trace=t))
    labels = list(by)
    ref = a.reference or labels[0]
    print(f"traces: {', '.join(traces)}\n")
    print("| label | requests | ms per prompt token | ms per generated token | fixed s per request | mean cache_n | "
          "TTFT p50 s | wall min | same-work min |")
    print("|---|---|---|---|---|---|---|---|---|")
    cs = {l: costs(by[l]) for l in labels}
    for l in labels:
        c = cs[l]
        print(f"| {l} | {c['n']} | {c['ms_prompt']:.2f} | {c['ms_gen']:.2f} | {c['fixed_s']:.2f} | "
              f"{c['cache_n']:.0f} | {c['ttft']:.2f} | {c['wall_min']:.1f} | {same_work_min(by[ref], c):.1f} |")
    print(f"\nRatios to {ref} (95 % bootstrap over traces, {a.boot} resamples):\n")
    print("| label | ms per prompt token | ms per generated token | same-work minutes |")
    print("|---|---|---|---|")
    rng = random.Random(1)
    for l in labels:
        if l == ref:
            continue
        common = [t for t in traces if any(r["trace"] == t for r in by[l]) and any(r["trace"] == t for r in by[ref])]
        if not common:
            continue
        per = {t: ([r for r in by[l] if r["trace"] == t], [r for r in by[ref] if r["trace"] == t]) for t in common}

        def ratios(ts):
            x = [r for t in ts for r in per[t][0]]
            y = [r for t in ts for r in per[t][1]]
            cx, cy = costs(x), costs(y)
            return (cx["ms_prompt"] / cy["ms_prompt"], cx["ms_gen"] / cy["ms_gen"],
                    same_work_min(y, cx) / same_work_min(y, cy))

        point = ratios(common)
        boots = [ratios([rng.choice(common) for _ in common]) for _ in range(a.boot)]
        cells = []
        for j in range(3):
            v = sorted(b[j] for b in boots)
            cells.append(f"{point[j]:.3f} ({v[int(0.025 * len(v))]:.3f}–{v[int(0.975 * len(v)) - 1]:.3f})")
        print(f"| {l} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--log", nargs="+", required=True)
    p = sub.add_parser("run")
    p.add_argument("--log", nargs="+", required=True)
    p.add_argument("--traces", default="")
    p.add_argument("--traces-file", type=Path)
    p.add_argument("--server", required=True, help="http://host:port of llama-server (not the proxy)")
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--stream", choices=["recorded", "on", "off"], default="recorded")
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--erase-slot", action="store_true")
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--drop-reasoning", action="store_true")
    p = sub.add_parser("pick")
    p.add_argument("--log", nargs="+", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--tasks", help="A-B: positions in the run order")
    p.add_argument("--task-set", default="polyglot", help="the task set whose order.txt --tasks counts in")
    p.add_argument("--shortest", action="store_true")
    p = sub.add_parser("summary")
    p.add_argument("files", nargs="+")
    p.add_argument("--log", nargs="*", help="proxy logs, to add the live rows of the same traces")
    p.add_argument("--reference")
    p.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    {"list": cmd_list, "run": cmd_run, "summary": cmd_summary, "pick": cmd_pick}[a.cmd](a)


if __name__ == "__main__":
    main()
