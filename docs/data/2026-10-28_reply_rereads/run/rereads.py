"""Where the agent check's prompt cache stops after a reply (Q260), per turn, from a run's own logs.

    python3 -I rereads.py --run DIR --server http://127.0.0.1:18431 --out turns.tsv [--snippets snippets.jsonl]

DIR holds an agent-check run's proxy.jsonl, proxy.jsonl.bodies.gz and rig/server-rel.log. The server at --server is
the same llama-server build as the run's (chat template and tokenizer only: serve_tiny.sh).

For every request the prompt is rendered with /apply-template and tokenized with /tokenize exactly as the chat
endpoint does (add_special and parse_special on); its length is checked against the run's prompt_n + cache_n.
For every request that follows a reply in the same session, the server log gives the shared prefix of the slot's
cached tokens (the previous prompt plus the generated reply) and the new prompt:
lcp = f_keep * cached tokens = f_sim * new prompt tokens (both printed to three decimals). The divergence offset
inside the reply is lcp - previous prompt length; the template's rendering of the reply (the new prompt's tokens from
the previous prompt's end to the next <|observation|> or <|user|>) shows what sits there: reasoning, content, the
tool call's markup, an argument key or value.
"""
import argparse
import gzip
import json
import re
import sys
import urllib.request
from pathlib import Path

RX = {
    "start": re.compile(r"^=== "),
    "select": re.compile(r"selected slot by LCP similarity, f_sim_best = ([\d.]+) .*f_keep = ([\d.]+)"),
    "launch": re.compile(r"launch_slot_: id +\d+ \| task (\d+) \| processing task"),
    "restore": re.compile(r"task (\d+) \| restored context checkpoint \(pos_min = (\d+), pos_max = (\d+), "
                          r"n_tokens = (\d+), n_past = (\d+), size = ([\d.]+) MiB, ([\d.]+) ms\)"),
    "full": re.compile(r"task (\d+) \| forcing full prompt re-processing"),
    "peval": re.compile(r"task (\d+) \| prompt eval time = +([\d.]+) ms / +(\d+) tokens"),
    "eval": re.compile(r"task (\d+) \| +eval time = +([\d.]+) ms / +(\d+) tokens"),
    "stop": re.compile(r"task (\d+) \| stop processing: n_tokens = (\d+)"),
}
MARKS = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
         "</arg_value>"]


def post(server, path, obj):
    req = urllib.request.Request(server + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def parse_log(path):
    """Tasks in order: f_sim/f_keep of the slot selection before it, restore, prompt/eval counts, cached tokens."""
    tasks, cur, pending, server_n = {}, None, None, 0
    order = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if RX["start"].match(line):
            server_n += 1
            pending = None
            continue
        m = RX["select"].search(line)
        if m:
            pending = (float(m.group(1)), float(m.group(2)))
            continue
        m = RX["launch"].search(line)
        if m:
            tid = int(m.group(1))
            tasks[tid] = dict(task=tid, server=server_n, f_sim=None, f_keep=None, restore=None, full=False)
            if pending:
                tasks[tid]["f_sim"], tasks[tid]["f_keep"] = pending
            pending = None
            order.append(tid)
            continue
        for key in ("restore", "full", "peval", "eval", "stop"):
            m = RX[key].search(line)
            if not m:
                continue
            tid = int(m.group(1))
            t = tasks.get(tid)
            if t is None:
                break
            if key == "restore":
                t["restore"] = dict(pos_min=int(m.group(2)), pos_max=int(m.group(3)), n_past=int(m.group(5)),
                                    mib=float(m.group(6)), ms=float(m.group(7)))
            elif key == "full":
                t["full"] = True
            elif key == "peval":
                t["prompt_ms"], t["prompt_n"] = float(m.group(2)), int(m.group(3))
            elif key == "eval":
                t["eval_ms"], t["eval_n"] = float(m.group(2)), int(m.group(3))
            elif key == "stop":
                t["n_tokens"] = int(m.group(2))
            break
    return [tasks[t] for t in order if "prompt_n" in tasks[t]]


def read_bodies(path):
    out = {}
    with gzip.open(path, "rt") as g:
        for line in g:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["id"]] = r
    return out


def session_of(tag):
    parts = tag.split("/")
    return "/".join(parts[:4]), int(parts[4])


def segment(text, pos):
    """Which part of a rendered reply the character offset pos falls in, and the marker before it."""
    best, where = None, -1
    for mk in MARKS:
        i = text.rfind(mk, 0, pos + 1)
        if i > where:
            best, where = mk, i
    if best is None:
        return "content-before-think" if "<think>" in text else "content", pos
    seg = {"<think>": "reasoning", "</think>": "after-reasoning", "<tool_call>": "tool-name",
           "</tool_call>": "after-call", "<arg_key>": "arg-key", "</arg_key>": "between-key-value",
           "<arg_value>": "arg-value", "</arg_value>": "after-value"}[best]
    if pos < where + len(best):
        seg = "marker:" + best
    return seg, pos - where - len(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:18431")
    ap.add_argument("--out", required=True)
    ap.add_argument("--snippets", help="JSON lines with the rendered text around each divergence (private)")
    a = ap.parse_args()
    run = Path(a.run)
    rows = [json.loads(l) for l in (run / "proxy.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("status") == 200 and r.get("prompt_n") is not None]
    bodies = read_bodies(run / "proxy.jsonl.bodies.gz")
    tasks = parse_log(run / "rig" / "server-rel.log")

    # match log tasks to proxy rows in order, on prompt and generated counts
    j, matched = 0, []
    for r in rows:
        while j < len(tasks) and not (tasks[j]["prompt_n"] == r["prompt_n"] and tasks[j].get("eval_n") == r["predicted_n"]):
            j += 1
        if j == len(tasks):
            sys.exit(f"no log task for {r['id']} {r['tag']}")
        matched.append((r, tasks[j]))
        j += 1

    vocab_ids = {}
    for mk in ("<|observation|>", "<|user|>", "<|assistant|>"):
        tk = post(a.server, "/tokenize", {"content": mk, "add_special": False, "parse_special": True})["tokens"]
        vocab_ids[mk] = tk[0] if len(tk) == 1 else None

    rendered = []
    for r, t in matched:
        body = bodies[r["id"]]["body"]
        prompt = post(a.server, "/apply-template", body)["prompt"]
        ids = post(a.server, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]
        rendered.append(ids)

    out = open(a.out, "w")
    cols = ["tag", "prompt_tokens", "render_ok", "gen", "reply_rendered", "render_minus_gen", "lcp", "lcp_err",
            "hist_lcp_ok", "div_in_reply", "div_from_end", "segment", "seg_offset_chars", "reply_chars",
            "reasoning_chars", "content_chars", "n_calls", "tools", "nonstring_args", "next_cache_n", "restored_pos_max",
            "next_prompt_n", "next_prompt_ms", "reread_tokens", "reread_ms", "next_ttft_s", "finish"]
    out.write("\t".join(cols) + "\n")
    snip = open(a.snippets, "w") if a.snippets else None
    for k in range(len(matched)):
        r, t = matched[k]
        ids = rendered[k]
        total = r["prompt_n"] + (r["cache_n"] or 0)
        ok = len(ids) == total
        if k + 1 == len(matched):
            break
        r2, t2 = matched[k + 1]
        s1, n1 = session_of(r["tag"])
        s2, n2 = session_of(r2["tag"])
        if s1 != s2 or n2 != n1 + 1 or t2["server"] != t["server"] or t2["f_keep"] is None:
            continue
        ids2 = rendered[k + 1]
        P, G = len(ids), r["predicted_n"]
        n_cache = t.get("n_tokens")
        lk = t2["f_keep"] * n_cache if n_cache else None
        ls = t2["f_sim"] * len(ids2)
        err = 0.0005 * max(n_cache or 0, len(ids2))
        lcp = round(ls if lk is None else (lk + ls) / 2)
        hist = 0
        while hist < min(P, len(ids2)) and ids[hist] == ids2[hist]:
            hist += 1
        end = P
        while end < len(ids2) and ids2[end] not in (vocab_ids["<|observation|>"], vocab_ids["<|user|>"]):
            end += 1
        R = end - P
        reply = post(a.server, "/detokenize", {"tokens": ids2[P:end]})["content"]
        d = lcp - P
        if 0 <= d <= R:
            head = post(a.server, "/detokenize", {"tokens": ids2[P:P + d]})["content"] if d else ""
            seg, off = segment(reply, len(head))
        else:
            head, seg, off = "", ("before-reply" if d < 0 else "after-reply"), 0
        rep = bodies[r["id"]].get("reply") or {}
        calls = rep.get("tool_calls") or []
        nonstr = 0
        for c in calls:
            try:
                args = json.loads(c.get("arguments") or "{}")
                nonstr += sum(1 for v in args.values() if not isinstance(v, str))
            except ValueError:
                nonstr += 1
        cache2 = r2["cache_n"] or 0
        reread = max(0, P + R - cache2)
        pms = r2["prompt_ms"] or 0.0
        reread_ms = pms * reread / r2["prompt_n"] if r2["prompt_n"] else 0.0
        rst = t2["restore"]["pos_max"] if t2["restore"] else ""
        vals = [r["tag"], P, int(ok), G, R, R - (G - 1), lcp, round(err, 1), int(hist >= P), d, P + G - 1 - lcp, seg,
                off, len(reply), len(rep.get("reasoning_content") or ""), len(rep.get("content") or ""), len(calls),
                ",".join(c.get("name", "") for c in calls), nonstr, cache2, rst, r2["prompt_n"], round(pms, 1), reread,
                round(reread_ms, 1), r2.get("t_first_token"), r.get("finish_reason")]
        out.write("\t".join(str(v) for v in vals) + "\n")
        if snip:
            c0 = len(head)
            snip.write(json.dumps({"tag": r["tag"], "segment": seg, "div": d, "before": reply[max(0, c0 - 300):c0],
                                   "after": reply[c0:c0 + 300], "reply_head": reply[:200], "reply_tail": reply[-300:],
                                   "reasoning_tail": (rep.get("reasoning_content") or "")[-200:],
                                   "content": (rep.get("content") or "")[:300],
                                   "calls": [dict(name=c.get("name"), arguments=(c.get("arguments") or "")[:400])
                                             for c in calls]}) + "\n")
    out.close()
    print(f"{run}: {len(matched)} requests matched to the log, {sum(len(x) == (m[0]['prompt_n'] + (m[0]['cache_n'] or 0)) for x, m in zip(rendered, matched))} renders equal the server's prompt length")


if __name__ == "__main__":
    main()
