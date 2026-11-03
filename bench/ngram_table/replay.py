#!/usr/bin/env python3
# replay.py: replays recorded coding-agent sessions through Qwen3.8-Flash-Next's n-gram table hash (llama.cpp's
# qwen4exp, `per_layer_token_embd`), so that ple_sim can count the table rows and 4 KiB pages each processed token reads
# and run them through caches. Standard library plus jinja2.
#   replay.py prepare --model SHARD1 --table TABLE_SHARD --out DIR (RUN_DIR... | --sessions SESSIONS.jsonl)
#       renders every request of the agent-check traces in each RUN_DIR (proxy.jsonl + proxy.jsonl.bodies.gz), or of
#       the sessions in a JSONL with messages and tools, with the model's own chat template, twice: the prompt, and
#       the prompt followed by the recorded reply. Writes
#       DIR/texts.bin (for ple_tok), DIR/requests.json, DIR/traces.tsv, DIR/params.txt (the hash constants and where
#       the table and the token embeddings sit in their files) and DIR/render.tsv (render checks)
#   ple_tok SHARD1 DIR/texts.bin DIR/tokens.bin
#       the model's own tokenizer, through libllama (vocab only)
#   replay.py stream --out DIR [--ckpts 4]
#       the tokens llama-server processes, with its prompt cache modelled for a hybrid model: a request that extends
#       the slot's tokens (the previous prompt and reply, tokenized as one text) reuses all of them; one that diverges
#       goes back to the last context checkpoint at or before
#       the divergence (one per earlier prompt end, the last --ckpts kept), or to zero. Every session starts a cold
#       slot. Writes DIR/stream.bin (one record per processed token) and DIR/requests.tsv
#   replay.py check --out DIR [--n 20000]
#       recomputes the rows of the first N records in Python, for ple_sim --check-rows to compare
#   ple_sim DIR OUTDIR [TRAIN_DIR]
#       the caches; see ple_sim.cpp
#   replay.py report --out DIR --tables OUTDIR
#       Markdown tables of the results (OUTDIR/tables.md)
import argparse, gzip, json, os, re, struct, sys
from pathlib import Path

TRACE_RE = re.compile(r"^(.+/a\d+/t\d+)/(\d+)$")
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
# ggml type -> (weights per block, bytes per block), for the types these tensors use
BLOCK = {0: (1, 4), 1: (1, 2), 2: (32, 18), 6: (32, 22), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144),
         13: (256, 176), 14: (256, 210), 18: (256, 98), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136)}
REC = struct.Struct("<6i")  # trace, request, kind (0 prefill, 1 decode), token, previous token, the one before (-1: none)


def gguf(path):
    """Key/value pairs (big arrays skipped), tensor infos {name: (dims, type, offset)} and where the data starts."""
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    rstr = lambda: f.read(rd("<Q")).decode("utf-8")

    def val(t):
        if t in SCALAR:
            return rd(SCALAR[t])
        if t == 8:
            return rstr()
        et, n = rd("<I"), rd("<Q")
        if n > 4096:
            for _ in range(n) if et == 8 else ():
                rstr()
            if et != 8:
                f.seek(n * struct.calcsize(SCALAR[et]), 1)
            return None
        return [val(et) for _ in range(n)]

    assert f.read(4) == b"GGUF", f"{path}: not a GGUF"
    _, nt, nkv = rd("<I"), rd("<Q"), rd("<Q")
    kv = {}
    for _ in range(nkv):
        k = rstr()
        kv[k] = val(rd("<I"))
    tensors = {}
    for _ in range(nt):
        name = rstr()
        dims = [rd("<Q") for _ in range(rd("<I"))]
        tensors[name] = (dims, rd("<I"), rd("<Q"))
    align = kv.get("general.alignment", 32)
    return kv, tensors, (f.tell() + align - 1) // align * align


def row_bytes(dims, t):
    w, b = BLOCK[t]
    assert dims[0] % w == 0
    return dims[0] // w * b


def tensor_layout(path, name):
    """(rows, bytes per row, absolute file offset of row 0) of a tensor, or None if the file lacks it."""
    _, tensors, data0 = gguf(path)
    if name not in tensors:
        return None
    dims, t, off = tensors[name]
    return dims[1], row_bytes(dims, t), data0 + off


def template_env():
    import jinja2

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    # as Hugging Face's and llama.cpp's renderers: plain JSON, no HTML escaping
    env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)
    env.filters["items"] = lambda d: list(d.items())
    env.globals["raise_exception"] = raise_exception
    return env


def tool_args(a):
    """Tool-call arguments as the template expects them (an object); llama-server parses the JSON string too."""
    if isinstance(a, dict):
        return a, True
    try:
        v = json.loads(a) if a else {}
        return (v, True) if isinstance(v, dict) else ({}, False)
    except (ValueError, TypeError):
        return {}, False


def norm_messages(msgs, bad):
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function", tc))
                fn["arguments"], ok = tool_args(fn.get("arguments"))
                bad[0] += not ok
                tcs.append({"type": "function", "function": fn})
            m["tool_calls"] = tcs
        out.append(m)
    return out


def reply_message(reply, bad):
    if isinstance(reply, str):
        import ast
        reply = ast.literal_eval(reply)
    m = {"role": "assistant", "content": reply.get("content") or "", "reasoning_content": reply.get("reasoning_content") or ""}
    if reply.get("tool_calls"):
        m["tool_calls"] = [{"type": "function", "function": {"name": tc["name"], "arguments": tc.get("arguments")}}
                           for tc in reply["tool_calls"]]
    return norm_messages([m], bad)[0]


def load_run(run_dir):
    """The run's traces in the order they ran: [(trace tag, [(turn, row, body record)])]."""
    log = Path(run_dir) / "proxy.jsonl"
    rows = [json.loads(l) for l in log.open() if l.strip()]
    bodies = {}
    for l in gzip.open(str(log) + ".bodies.gz", "rt"):
        d = json.loads(l)
        bodies[d["id"]] = d
    traces = {}
    for r in rows:
        m = TRACE_RE.match(r.get("tag") or "")
        if m:
            traces.setdefault(m.group(1), []).append((int(m.group(2)), r, bodies.get(r["id"])))
    order = sorted(traces, key=lambda k: min(x[1]["t_wall"] for x in traces[k]))
    return [(k, sorted(traces[k], key=lambda x: (x[0], x[1]["id"]))) for k in order]


def agent_check_sessions(run_dirs):
    """(run, tag, task, [(turn, id, body, reply, proxy row)]) per agent-check trace, skipping failed requests."""
    for run_i, run_dir in enumerate(run_dirs):
        for tag, items in load_run(run_dir):
            reqs = []
            for turn, row, rec in items:
                if rec is None or not rec.get("reply") or row.get("status") != 200 or row.get("error"):
                    continue
                reqs.append((turn, row["id"], rec["body"], rec["reply"], row))
            yield run_i, f"{Path(run_dir).name}/{tag}", tag.split("/")[1], reqs


def jsonl_sessions(path):
    """The same from a sessions JSONL (one session per line, with messages and tools): every assistant message is the
    reply to a request holding the messages before it. All sessions form one run, in file order."""
    for line in open(path):
        s = json.loads(line)
        msgs, reqs = s["messages"], []
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            reply = {"content": m.get("content"), "reasoning_content": m.get("reasoning_content"),
                     "tool_calls": [tc.get("function", tc) for tc in m.get("tool_calls") or []]}
            reqs.append((len(reqs) + 1, f"{s['session_id']}/{len(reqs) + 1}", {"messages": msgs[:i], "tools": s["tools"]},
                         reply, {}))
        yield 0, s["session_id"], s["session_id"], reqs


def cmd_prepare(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kv, _, _ = gguf(a.model)
    arch = kv["general.architecture"]
    g = lambda k: kv[f"{arch}.ple.{k}"]
    table = tensor_layout(a.table, "per_layer_token_embd.weight")
    embd = tensor_layout(a.model, "token_embd.weight")
    assert table and embd, "the table or the token embeddings are not in the given files"
    with open(out / "params.txt", "w") as f:
        f.write(f"ngram_size {g('ngram_size')}\nheads_per_ngram {g('heads_per_ngram')}\neos {g('eos_token_id')}\n")
        f.write("multipliers " + " ".join(map(str, g("layer_multipliers"))) + "\n")
        f.write("offsets " + " ".join(map(str, g("head_offsets"))) + "\n")
        f.write("vocab " + " ".join(map(str, g("head_vocab_sizes"))) + "\n")
        f.write(f"table_rows {table[0]}\ntable_row_bytes {table[1]}\ntable_offset {table[2]}\n")
        f.write(f"embd_rows {embd[0]}\nembd_row_bytes {embd[1]}\nembd_offset {embd[2]}\npage 4096\n")
    env = template_env()
    tmpl = env.from_string(kv["tokenizer.chat_template"])
    texts = open(out / "texts.bin", "wb")
    reqs, n_text = [], 0
    tr_rows = ["trace\trun\ttask\ttag\tn_requests"]
    checks = ["tag\tturn\tprompt_chars\tfull_chars\tprefix_ok"]
    tasks = {}
    source = jsonl_sessions(a.sessions) if a.sessions else agent_check_sessions(a.runs)
    for run_i, tag, task, items in source:
        trace_i = len(tr_rows) - 1
        task_i = tasks.setdefault(task, len(tasks))
        for turn, rid, body, reply, row in items:
            bad = [0]
            msgs = norm_messages(body["messages"], bad)
            tools = body.get("tools")
            prompt = tmpl.render(messages=msgs, tools=tools, add_generation_prompt=True)
            full = tmpl.render(messages=msgs + [reply_message(reply, bad)], tools=tools, add_generation_prompt=False)
            ok = full.startswith(prompt)
            checks.append(f"{tag}\t{turn}\t{len(prompt)}\t{len(full)}\t{int(ok)}")
            for t in (prompt, full):
                b = t.encode("utf-8")
                texts.write(struct.pack("<I", len(b)) + b)
            reqs.append({"trace": trace_i, "run": run_i, "turn": turn, "id": rid, "text": n_text, "bad_args": bad[0],
                         "prefix_ok": ok, "flint_prompt_n": row.get("prompt_n"), "flint_cache_n": row.get("cache_n"),
                         "flint_predicted_n": row.get("predicted_n")})
            n_text += 2
        tr_rows.append(f"{trace_i}\t{run_i}\t{task_i}\t{tag}\t{len(items)}")
    texts.close()
    runs = [a.sessions] if a.sessions else [Path(r).name for r in a.runs]
    (out / "requests.json").write_text(json.dumps({"runs": runs, "tasks": list(tasks), "requests": reqs}))
    (out / "traces.tsv").write_text("\n".join(tr_rows) + "\n")
    (out / "render.tsv").write_text("\n".join(checks) + "\n")
    print(f"{len(reqs)} requests in {len(tr_rows) - 1} sessions of {len(runs)} runs, {len(tasks)} tasks; "
          f"prompt not a prefix of prompt+reply: {sum(not r['prefix_ok'] for r in reqs)}; "
          f"tool arguments not an object: {sum(r['bad_args'] for r in reqs)}")


def read_tokens(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        (n,) = struct.unpack_from("<I", data, i)
        out.append(list(struct.unpack_from(f"<{n}i", data, i + 4)))
        i += 4 + 4 * n
    return out


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def cmd_stream(a):
    out = Path(a.out)
    meta = json.loads((out / "requests.json").read_text())
    toks = read_tokens(out / "tokens.bin")
    stops = set(a.stop)
    f = open(out / "stream.bin", "wb")
    rows = ["trace\trun\tturn\tid\tprompt_tokens\treused\tprefill\tdecode\tpath\tflint_prompt_n\tflint_cache_n\tflint_predicted_n"]
    slot, ckpts, cur = [], [], None
    for ri, r in enumerate(meta["requests"]):
        if r["trace"] != cur:  # a new session: cold slot
            slot, ckpts, cur = [], [], r["trace"]
        P, F = toks[r["text"]], toks[r["text"] + 1]
        # the reply's tokens as the prompt+reply render tokenizes them: where the reply's first token merges with the
        # prompt's last one (an empty <think> gives one "\n\n" token), it starts one position early. The reply ends at
        # the sampled stop token, which is never decoded; it and the template's tail after it are dropped
        L = lcp(P, F)
        cut = next((i for i in range(L, len(F)) if F[i] in stops), len(F))
        n = lcp(slot, P)
        if slot and n == len(slot):
            path, reuse = "extend", n
        else:
            fit = [c for c in ckpts if c <= n]
            path, reuse = ("checkpoint", max(fit)) if fit else ("cold" if not slot else "full", 0)
        reuse = min(reuse, len(P) - 1)  # the last prompt token is always evaluated
        for pos in range(reuse, len(P)):
            f.write(REC.pack(r["trace"], ri, 0, P[pos], P[pos - 1] if pos >= 1 else -1, P[pos - 2] if pos >= 2 else -1))
        for pos in range(L, cut):
            f.write(REC.pack(r["trace"], ri, 1, F[pos], F[pos - 1] if pos >= 1 else -1, F[pos - 2] if pos >= 2 else -1))
        rows.append(f"{r['trace']}\t{r['run']}\t{r['turn']}\t{r['id']}\t{len(P)}\t{reuse}\t{len(P) - reuse}\t{cut - L}\t"
                    f"{path}\t{r['flint_prompt_n']}\t{r['flint_cache_n']}\t{r['flint_predicted_n']}")
        slot = F[:cut]
        ckpts = (ckpts + [len(P)])[-a.ckpts:] if a.ckpts > 0 else []
    f.close()
    (out / "requests.tsv").write_text("\n".join(rows) + "\n")
    paths = {}
    for l in rows[1:]:
        p = l.split("\t")[8]
        paths[p] = paths.get(p, 0) + 1
    print(f"{len(rows) - 1} requests; cache paths {paths}; stream {os.path.getsize(out / 'stream.bin') // REC.size} tokens")


def cmd_check(a):
    """Rows of the first N records, by a second implementation of the hash (Python integers, masked to 64 bits)."""
    out = Path(a.out)
    p = {}
    for l in open(out / "params.txt"):
        k, *v = l.split()
        p[k] = [int(x) for x in v]
    mult, offs, voc, eos = p["multipliers"], p["offsets"], p["vocab"], p["eos"][0]
    ng, per = p["ngram_size"][0], p["heads_per_ngram"][0]
    M = (1 << 64) - 1
    data = open(out / "stream.bin", "rb").read(REC.size * a.n)
    with open(out / "check_rows.txt", "w") as f:
        for i in range(len(data) // REC.size):
            _, _, _, t0, t1, t2 = REC.unpack_from(data, i * REC.size)
            ctx, cutoff = [t0], False
            for t in (t1, t2)[: ng - 1]:
                cutoff = cutoff or t < 0 or t == eos
                ctx.append(eos if cutoff else t)
            rows = []
            for n in range(2, ng + 1):
                mixed = (ctx[0] * mult[0]) & M
                for j in range(1, n):
                    mixed ^= (ctx[j] * mult[j]) & M
                for gi in range(per):
                    h = (n - 2) * per + gi
                    rows.append(mixed % voc[h] + offs[h])
            f.write(" ".join(map(str, rows)) + "\n")


def tsv(path):
    lines = open(path).read().splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, l.split("\t"))) for l in lines[1:] if l]


def cmd_report(a):
    """Markdown tables from ple_sim's output (TABLES) and the stream's request list (DIR)."""
    T, W = Path(a.tables), Path(a.out)
    pick = ["256 MiB", "512 MiB", "1 GiB", "2 GiB", "4 GiB", "8 GiB", "16 GiB"]
    out = []
    curve = tsv(T / "lru_curve.tsv")
    for stream, part in (("pages", "all"), ("rows", "all")):
        sizes = ["16 MiB", "32 MiB", "64 MiB", "95 MB", "256 MiB", "1 GiB"] if stream == "rows" else ["64 MiB", "128 MiB"] + pick
        out.append(f"\n{stream}: reads per token (hit rate), LRU\n")
        out.append("| scope | kind | " + " | ".join(sizes) + " |")
        out.append("|---|---|" + "---:|" * len(sizes))
        for scope in ("session", "run", "all"):
            for kind in ("decode", "prefill"):
                r = {x["cache"]: x for x in curve if x["stream"] == stream and x["scope"] == scope and x["kind"] == kind
                     and x["part"] == part}
                if r:
                    out.append(f"| {scope} | {kind} | " + " | ".join(
                        f"{float(r[s]['misses_per_token']):.2f} ({100 * float(r[s]['hit_rate']):.0f} %)" for s in sizes) + " |")
    # per request, scope run, at 2 GiB: the run's first request, other sessions' first requests, later requests
    rr = {int(x["req"]): x for x in tsv(T / "request_reads.tsv")}
    reqs = tsv(W / "requests.tsv")
    seen_run, groups = set(), {"run's first request": [], "a session's first request": [], "later requests": []}
    for i, r in enumerate(reqs):
        if i not in rr:
            continue
        g = ("later requests" if r["path"] != "cold" else
             "a session's first request" if r["run"] in seen_run else "run's first request")
        seen_run.add(r["run"])
        groups[g].append(rr[i])
    col = "2 GiB"
    out.append(f"\nper request, page cache {col}, cold at each run\n")
    out.append("| requests | n | prompt tokens processed (median) | table reads (median) | reads per prompt token | "
               "decoded tokens (median) | reads per decoded token |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    med = lambda v: sorted(v)[len(v) // 2] if v else 0
    for g, v in groups.items():
        if not v:
            continue
        pt = [int(x["prefill_tokens"]) for x in v]
        pr = [int(x[f"prefill_reads_{col}"]) for x in v]
        dt = [int(x["decode_tokens"]) for x in v]
        dr = [int(x[f"decode_reads_{col}"]) for x in v]
        out.append(f"| {g} | {len(v)} | {med(pt)} | {med(pr)} | {sum(pr) / max(1, sum(pt)):.2f} | {med(dt)} | "
                   f"{sum(dr) / max(1, sum(dt)):.2f} |")
    text = "\n".join(out) + "\n"
    (T / "tables.md").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("--model", required=True, help="the GGUF shard with the tokenizer and token_embd (shard 1)")
    s.add_argument("--table", required=True, help="the GGUF shard holding per_layer_token_embd")
    s.add_argument("--out", required=True)
    s.add_argument("--sessions", help="a sessions JSONL (messages and tools per line) instead of agent-check runs")
    s.add_argument("runs", nargs="*", help="agent-check run folders")
    s = sub.add_parser("stream")
    s.add_argument("--out", required=True)
    s.add_argument("--ckpts", type=int, default=4, help="context checkpoints kept per slot (llama-server --ctx-checkpoints)")
    s.add_argument("--stop", type=int, nargs="+", default=[248046, 248044], help="stop tokens (<|im_end|>, <|endoftext|>)")
    s = sub.add_parser("check")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=20000)
    s = sub.add_parser("report")
    s.add_argument("--out", required=True, help="the stream's folder")
    s.add_argument("--tables", required=True, help="ple_sim's output folder")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "stream": cmd_stream, "check": cmd_check, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
