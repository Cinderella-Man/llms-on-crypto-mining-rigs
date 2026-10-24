"""Replay multi-turn agent sessions against llama-server (Q111): time to first token and prompt tokens read per turn.

    python3 replay_sessions.py pick SESSIONS.jsonl --groups 2 --per-group 4 [--max-steps 14]
    python3 replay_sessions.py run SESSIONS.jsonl --server http://127.0.0.1:PORT --label LABEL --out OUT.jsonl
                               [--ids ID,ID,...] [--history keep|drop] [--max-tokens 2048] [--greedy]
    python3 replay_sessions.py summary OUT.jsonl [OUT2.jsonl ...] [--reference LABEL]

A session (export_bfcl.py) is a list of requests, each the messages added since the previous reply. `run` drives it as
an agent does: every request carries the whole conversation, and the reply the server just gave goes back into the
history (reasoning_content, content, tool calls), so the server sees its own tokens again, as rendered by its chat
template; the recorded tool results answer the live calls in order. When the live reply makes a different number of
calls than the recorded one, the recorded reply goes into the history instead (`mismatch` in the row), so the
conversation stays valid. With --history drop, earlier replies go back without reasoning_content (what most
OpenAI-compatible clients send).

Each request streams, with the recorded sampling settings and a seed fixed per session and request (the same for every
server, so two exact servers give the same replies), and at most --max-tokens (default: 2 x the recorded reply + 256,
capped at 2,048). One row per request: client time to first streamed token and to the end, the server's timings (cache_n,
prompt_n, prompt_ms, predicted_n, predicted_ms; from the agent-cache build also cache_src, cache_ckpt_n,
cache_restore_ms, ckpt_created_n), the reuse class against the previous request (first, exact: everything up to the
previous reply's end; partial: into the previous reply; reply: the previous reply read again; older: less), and a hash
of the reply. Requests run one at a time, sessions in the given order; the server's cache carries over between
sessions, as for one user. Standard library only.

`summary` prints per label: requests, prompt tokens read per generated token, time to first token (median, p90; first
requests and later ones), seconds per session, reuse classes and cache sources, and how many replies equal the
reference label's.
"""
import argparse
import hashlib
import http.client
import json
import sys
import time
import urllib.parse
import zlib
from collections import Counter, defaultdict
from statistics import median

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


def load_sessions(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def stream_chat(server, body, timeout):
    """POSTs a streaming chat completion; returns (reply message, timings, ttft_s, total_s, finish_reason, error)."""
    u = urllib.parse.urlparse(server)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    t0 = time.monotonic()
    ttft = None
    content, reasoning, calls = [], [], {}
    timings, finish, err = {}, None, None
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = conn.getresponse()
        if r.status != 200:
            return None, {}, None, time.monotonic() - t0, None, f"HTTP {r.status}: {r.read()[:300]!r}"
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if "timings" in ev:
                timings = ev["timings"]
            for ch in ev.get("choices", []):
                d = ch.get("delta") or {}
                if ttft is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                    ttft = time.monotonic() - t0
                if d.get("content"):
                    content.append(d["content"])
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                    c["id"] = tc.get("id") or c["id"]
                    f = tc.get("function") or {}
                    c["name"] += f.get("name") or ""
                    c["arguments"] += f.get("arguments") or ""
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    finally:
        conn.close()
    total = time.monotonic() - t0
    msg = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = [{"id": c["id"] or f"call_live_{i}", "type": "function",
                              "function": {"name": c["name"], "arguments": c["arguments"]}}
                             for i, c in sorted(calls.items())]
    return msg, timings, ttft, total, finish, err


def reply_hash(msg):
    if msg is None:
        return None
    key = [msg.get("reasoning_content", ""), msg.get("content", ""),
           [(c["function"]["name"], c["function"]["arguments"]) for c in msg.get("tool_calls", [])]]
    return hashlib.sha1(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:16]


def reuse_class(prev, t):
    """The cache path from the server's counters: prev = (cache_n, prompt_n, predicted_n) of the previous request."""
    if prev is None or not t:
        return "first"
    c = t.get("cache_n", 0)
    pc, pp, pg = prev
    prompt_end = pc + pp
    if c >= prompt_end + pg - 1:
        return "exact"
    if c > prompt_end:
        return "partial"
    if c >= prompt_end - 8:
        return "reply"
    return "older"


def for_history(msg, keep_reasoning):
    out = {"role": "assistant", "content": msg.get("content") or ""}
    if keep_reasoning and msg.get("reasoning_content"):
        out["reasoning_content"] = msg["reasoning_content"]
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


def cmd_run(args):
    sessions = load_sessions(args.sessions)
    if args.ids:
        by_id = {s["id"]: s for s in sessions}
        sessions = [by_id[i] for i in args.ids.split(",")]
    done = set()
    try:
        for line in open(args.out):
            r = json.loads(line)
            if r["label"] == args.label and r.get("last"):
                done.add(r["session"])
    except FileNotFoundError:
        pass
    out = open(args.out, "a")
    keep = args.history == "keep"
    for s in sessions:
        if s["id"] in done:
            print(f"{s['id']}: done already", flush=True)
            continue
        history, prev, live_calls = [], None, []
        t_session = time.monotonic()
        for i, st in enumerate(s["steps"]):
            add = [dict(m) for m in st["add"]]
            k = 0
            for m in add:  # the recorded results answer the calls of the reply that is in the history
                if m["role"] == "tool":
                    m["tool_call_id"] = live_calls[k]["id"] if k < len(live_calls) else m.get("tool_call_id")
                    k += 1
            history += add
            n_max = min(args.max_tokens, 2 * st["n_gen"] + 256)
            seed = zlib.crc32(f"{s['id']}/{i}".encode()) & 0x7fffffff
            body = {"messages": history, "tools": s["tool_lists"][st["tools"]], "parallel_tool_calls": True,
                    "stream": True, "max_tokens": n_max, "seed": seed, "cache_prompt": True,
                    **({"temperature": 0.0} if args.greedy else SAMPLING)}
            msg, t, ttft, total, finish, err = stream_chat(args.server, body, args.timeout)
            rec_calls = st["reply"].get("tool_calls", [])
            mismatch = msg is None or len(msg.get("tool_calls", [])) != len(rec_calls)
            used = st["reply"] if mismatch else msg
            history.append(for_history(used, keep))
            live_calls = used.get("tool_calls", [])
            row = {"label": args.label, "session": s["id"], "step": i, "ttft_s": ttft, "total_s": total,
                   "n_prompt_rec": st["n_prompt"], "n_gen_rec": st["n_gen"], "max_tokens": n_max,
                   "class": reuse_class(prev, t), "mismatch": mismatch, "finish": finish, "error": err,
                   "reply_hash": reply_hash(msg), "last": i == len(s["steps"]) - 1,
                   **{k: t.get(k) for k in ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms",
                                            "cache_src", "cache_ckpt_n", "cache_restore_ms", "ckpt_created_n")}}
            if row["last"]:
                row["session_s"] = time.monotonic() - t_session
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{s['id']} {i:2d}: ttft {ttft or 0:6.2f}s total {total:6.2f}s cache_n {t.get('cache_n')} "
                  f"prompt_n {t.get('prompt_n')} gen {t.get('predicted_n')} {row['class']} {t.get('cache_src', '')}"
                  f"{' MISMATCH' if mismatch else ''}{' ERROR ' + err if err else ''}", flush=True)
            if err:
                sys.exit(f"stopped: {err}")
            prev = (t.get("cache_n", 0), t.get("prompt_n", 0), t.get("predicted_n", 0)) if t else None


def cmd_pick(args):
    """Groups of sessions that share a tool list (one agent's sessions share its system and tool prompt)."""
    sessions = [s for s in load_sessions(args.sessions) if len(s["steps"]) <= args.max_steps]
    groups = defaultdict(list)
    for s in sessions:
        groups[json.dumps(s["tool_lists"][0], sort_keys=True)].append(s)
    ranked = sorted(groups.values(), key=lambda g: -len(g))
    picked = []
    for g in ranked[:args.groups]:
        g = sorted(g, key=lambda s: len(s["steps"]))
        step = max(1, len(g) // args.per_group)
        picked += [s["id"] for s in g[::step][:args.per_group]]
    print(",".join(picked))
    if args.write:
        with open(args.write, "w") as f:
            for sid in picked:
                f.write(json.dumps(next(x for x in sessions if x["id"] == sid), ensure_ascii=False) + "\n")
    for sid in picked:
        s = next(x for x in sessions if x["id"] == sid)
        print(f"  {sid}: {len(s['steps'])} requests, first prompt {s['steps'][0]['n_prompt']} tokens, "
              f"recorded output {sum(st['n_gen'] for st in s['steps'])} tokens", file=sys.stderr)


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def cmd_summary(args):
    rows = []
    for p in args.files:
        rows += [json.loads(line) for line in open(p) if line.strip()]
    last = {}
    for r in rows:
        last[(r["label"], r["session"], r["step"])] = r
    by_label = defaultdict(list)
    for r in last.values():
        by_label[r["label"]].append(r)
    labels = list(dict.fromkeys(r["label"] for r in rows))
    ref = args.reference or labels[0]
    ref_hash = {(r["session"], r["step"]): r["reply_hash"] for r in by_label[ref]}
    print("| Label | Requests | Sessions | Prompt tokens read | per generated token | First token p50 / p90 (s), "
          "first requests | later requests | s per session | Reuse classes | Cache sources | Replies = " + ref + " |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for lab in labels:
        rs = by_label[lab]
        pn = sum(r.get("prompt_n") or 0 for r in rs)
        gn = sum(r.get("predicted_n") or 0 for r in rs)
        first = [r["ttft_s"] for r in rs if r["step"] == 0]
        later = [r["ttft_s"] for r in rs if r["step"] > 0]
        sess = [r["session_s"] for r in rs if r.get("session_s")]
        cls = Counter(r["class"] for r in rs)
        src = Counter(r.get("cache_src") or "-" for r in rs)
        same = sum(1 for r in rs if ref_hash.get((r["session"], r["step"])) == r["reply_hash"])
        print(f"| {lab} | {len(rs)} | {len({r['session'] for r in rs})} | {pn} | {pn / max(gn, 1):.2f} | "
              f"{pct(first, .5):.2f} / {pct(first, .9):.2f} | {pct(later, .5):.2f} / {pct(later, .9):.2f} | "
              f"{(sum(sess) / len(sess)) if sess else float('nan'):.1f} | "
              f"{', '.join(f'{k} {v}' for k, v in cls.most_common())} | "
              f"{', '.join(f'{k} {v}' for k, v in src.most_common())} | {same}/{len(rs)} |")
    mism = Counter(r["label"] for r in last.values() if r["mismatch"])
    if mism:
        print("\nRequests whose live reply made a different number of calls (recorded reply used in the history): "
              + ", ".join(f"{k} {v}" for k, v in mism.items()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("sessions")
    p.add_argument("--groups", type=int, default=2)
    p.add_argument("--per-group", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=14)
    p.add_argument("--write", help="also write the picked sessions, in order, to this file")
    r = sub.add_parser("run")
    r.add_argument("sessions")
    r.add_argument("--server", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--ids")
    r.add_argument("--history", choices=["keep", "drop"], default="keep")
    r.add_argument("--max-tokens", type=int, default=2048)
    r.add_argument("--greedy", action="store_true")
    r.add_argument("--timeout", type=float, default=1800)
    s = sub.add_parser("summary")
    s.add_argument("files", nargs="+")
    s.add_argument("--reference")
    args = ap.parse_args()
    {"pick": cmd_pick, "run": cmd_run, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
