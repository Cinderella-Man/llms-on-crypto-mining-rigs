#!/usr/bin/env python3
"""C1080's tables from run.sh's and steps.sh's output (copied off the build machine). Standard library only:

    python3 -I tables.py OUT_FOLDER DATA_FOLDER

OUT_FOLDER is run.sh's WORK (one folder per build: the responses, server.log or server.log.gz, the *.steps.tsv of
steps.sh, order.txt); DATA_FOLDER is where the tables go:
  arms.tsv        per build, request and arm: tokens, how the reply ended, the draft's tokens and acceptance, the
                  server's own grammar counts, and the first token where the arm's reply differs from the build's chat
                  arm (same: none)
  steps.tsv       per build and request (the chat arm): where the first <tool_call> token is, and at how many steps
                  before and after it the grammar let the whole vocabulary through
  left_out.tsv    per build, request and phase (before / after the first <tool_call>): each set of tokens the grammar
                  left out, at how many steps, and what kind of tokens they are
  server_lines.tsv  per build and chat request: the server's sampling line and, on a lazy grammar, its own trigger lines
  params.tsv      per build: the chat handler, whether the grammar is lazy, its triggers, the generation prompt, the
                  grammar's size and md5, and its trigger rule
  grammar_<build>.gbnf  the tool-call grammar each build built (the same for every request: it depends on the tools)
  tokens.tsv.gz   the chat arm's every token with the grammar's state (steps.sh's columns; the token id and its text
                  only for the model's special tokens, so the reply's text is not in it)
"""
import csv
import glob
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter, OrderedDict

OUT, DATA = sys.argv[1], sys.argv[2]
BUILDS = [b for b in ("cand", "ref", "land3") if os.path.isdir(f"{OUT}/{b}")]
REQS = ["t1_first", "t2_reason", "t5_final"]
ARMS = ["chat", "chat2", "none", "plain"]
SPECIAL_FROM = 248044  # the model's added tokens start here (<|endoftext|>)


def log_lines(b):
    p = f"{OUT}/{b}/server.log"
    f = open(p, "rb") if os.path.exists(p) else gzip.open(p + ".gz", "rb")
    for raw in f:
        yield raw.decode("utf-8", "replace")


def reply(b, r, a):
    j = json.load(open(f"{OUT}/{b}/{r}.{a}.json"))
    v = j.get("__verbose", j)
    end = j["choices"][0]["finish_reason"] if "choices" in j else v.get("stop_type")
    return v["tokens"], j.get("timings", {}), end


def first_diff(x, y):
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


def steps(b, r, a="chat"):
    p = f"{OUT}/{b}/{r}.{a}.steps.tsv"
    raw = open(p, "rb").read().decode("utf-8", "replace").split("\n")
    names = {}
    for line in raw:
        m = re.match(r"# left out at step (\d+) \((\d+)\):(.*)", line)
        if m:
            ents = [e.split("=", 1) for e in m.group(3).split()]
            names[",".join(i for i, _ in ents)] = [(int(i), bytes.fromhex(h) if h != "-" else b"") for i, h in ents]
    rows = list(csv.DictReader((l for l in raw if l and not l.startswith("#")), delimiter="\t",
                               quoting=csv.QUOTE_NONE))
    return rows, names


def kind(tok, bs):
    if tok >= SPECIAL_FROM and bs.startswith(b"<|") and bs in (b"<|endoftext|>", b"<|im_end|>", b"<|fim_pad|>",
                                                               b"<|repo_name|>", b"<|file_sep|>"):
        return "end of turn"
    if not bs:
        return "no text"
    if 0 in bs:
        return "NUL byte"
    try:
        bs.decode("utf-8")
        return "valid text"
    except UnicodeDecodeError as e:
        return "UTF-8 prefix" if e.reason == "unexpected end of data" else "malformed UTF-8 start"


os.makedirs(DATA, exist_ok=True)

# arms.tsv
with open(f"{DATA}/arms.tsv", "w") as f:
    f.write("build\trequest\tarm\ttokens\tended\tdraft_n\tdraft_accepted\tacceptance\tgrammar_n\tgrammar_rejected\t"
            "first_diff_vs_chat\n")
    for b in BUILDS:
        for r in REQS:
            base = reply(b, r, "chat")[0]
            for a in ARMS:
                t, tm, end = reply(b, r, a)
                d = first_diff(base, t)
                acc = tm["draft_n_accepted"] / tm["draft_n"] if tm.get("draft_n") else 0
                f.write(f"{b}\t{r}\t{a}\t{len(t)}\t{end}\t{tm.get('draft_n')}\t{tm.get('draft_n_accepted')}\t"
                        f"{acc:.3f}\t{tm.get('grammar_n', '-')}\t{tm.get('grammar_rejected_n', '-')}\t"
                        f"{'same' if d is None else d}\n")

# steps.tsv, left_out.tsv, tokens.tsv.gz
st = open(f"{DATA}/steps.tsv", "w")
st.write("build\trequest\ttokens\tfirst_call_at\tbefore_call_steps\tbefore_whole_vocab\tbefore_left_out\t"
         "after_call_steps\tafter_whole_vocab\tafter_left_out\treply_token_rejected\tchat2_same_states\n")
lo = open(f"{DATA}/left_out.tsv", "w")
lo.write("build\trequest\tphase\tsteps\tleft_out\tmalformed_utf8_start\tnul_byte\tno_text\tend_of_turn\t"
         "valid_text\n")
tk = gzip.open(f"{DATA}/tokens.tsv.gz", "wt")
tk.write("build\trequest\tstep\tspecial_token\tallowed\tn_allowed\tn_vocab\tawaiting\tphase\tspecial_piece\n")
for b in BUILDS:
    for r in REQS:
        rows, names = steps(b, r)
        rows2, _ = steps(b, r, "chat2")
        same2 = [(x["token"], x["allowed"], x["n_allowed"]) for x in rows] == \
                [(x["token"], x["allowed"], x["n_allowed"]) for x in rows2]
        call = next((i for i, x in enumerate(rows) if x["piece"] == "<tool_call>"), len(rows))
        phases = {"before": rows[:call], "after": rows[call:]}
        whole = {k: sum(x["n_allowed"] == x["n_vocab"] for x in v) for k, v in phases.items()}
        st.write(f"{b}\t{r}\t{len(rows)}\t{call if call < len(rows) else 'none'}\t{len(phases['before'])}\t"
                 f"{whole['before']}\t{len(phases['before']) - whole['before']}\t{len(phases['after'])}\t"
                 f"{whole['after']}\t{len(phases['after']) - whole['after']}\t"
                 f"{sum(x['allowed'] == '0' for x in rows)}\t{'yes' if same2 else 'no'}\n")
        for ph, rs in phases.items():
            sets = Counter()
            for x in rs:
                n_left = int(x["n_vocab"]) - int(x["n_allowed"])
                sets[n_left] += 1
            for n_left, cnt in sorted(sets.items()):
                if n_left == 0:
                    lo.write(f"{b}\t{r}\t{ph}\t{cnt}\t0\t0\t0\t0\t0\t0\n")
                    continue
                # the named list of that size (steps.sh names every list of at most 1024 tokens once)
                ents = next((v for v in names.values() if len(v) == n_left), None)
                if ents is None:
                    lo.write(f"{b}\t{r}\t{ph}\t{cnt}\t{n_left}\t-\t-\t-\t-\t-\n")
                    continue
                k = Counter(kind(t, bs) for t, bs in ents)
                lo.write(f"{b}\t{r}\t{ph}\t{cnt}\t{n_left}\t{k['malformed UTF-8 start']}\t{k['NUL byte']}\t"
                         f"{k['no text']}\t{k['end of turn']}\t{k['valid text'] + k['UTF-8 prefix']}\n")
        for i, x in enumerate(rows):
            tok = int(x["token"])
            piece, tid = (x["piece"], str(tok)) if tok >= SPECIAL_FROM else ("", "-")
            tk.write(f"{b}\t{r}\t{x['step']}\t{tid}\t{x['allowed']}\t{x['n_allowed']}\t{x['n_vocab']}\t"
                     f"{x['awaiting']}\t{'before' if i < call else 'after'}\t{piece}\n")
st.close()
lo.close()
tk.close()

# server_lines.tsv: the sampling line of each chat request and, on a lazy grammar, the server's own trigger lines
sv = open(f"{DATA}/server_lines.tsv", "w")
sv.write("build\trequest\tarm\tsampled\twhile_grammar_applied\trejected\tawaiting_lines\ttriggered_on\n")
samp = re.compile(r"sampling =\s+(\d+) tokens, \d+ drawn on the device, (\d+) while the grammar applied, (\d+) rejected")
for b in BUILDS:
    order = [l.split() for l in open(f"{OUT}/{b}/order.txt") if l.strip()]
    # requests that carry a grammar print the sampling line; group the trigger lines by the next sampling line
    chats = [(r, a) for r, a in order if a in ("chat", "chat2")]
    k, waiting, trig = 0, 0, ""
    for line in log_lines(b):
        waiting += line.count("Grammar still awaiting trigger after token")
        m = re.search(r"Grammar triggered on token (\d+)", line)
        if m:
            trig = m.group(1)
        m = samp.search(line)
        if m and k < len(chats):
            r, a = chats[k]
            sv.write(f"{b}\t{r}\t{a}\t{m.group(1)}\t{m.group(2)}\t{m.group(3)}\t{waiting}\t{trig or '-'}\n")
            k, waiting, trig = k + 1, 0, ""
sv.close()

# params.tsv and the grammars
pa = open(f"{DATA}/params.tsv", "w")
pa.write("build\tchat_handler\tgrammar_lazy\ttriggers\tgeneration_prompt\tgrammar_bytes\tgrammar_md5\t"
         "same_grammar_all_requests\ttrigger_rule\n")
for b in BUILDS:
    handler = next((l.split("Using specialized template: ")[1].strip() for l in log_lines(b)
                    if "Using specialized template: " in l), "autoparser")
    gs = {open(f"{OUT}/{b}/{r}.chat.grammar.txt").read() for r in REQS}
    g = open(f"{OUT}/{b}/t1_first.chat.grammar.txt").read()
    p = json.load(open(f"{OUT}/{b}/t1_first.chat.params.json"))
    trig = ";".join(f"token {t.get('token')} ({t.get('value')})" for t in p.get("grammar_triggers") or []) or "none"
    rule = next((l for l in g.split("\n") if l.startswith("trigger ::=") or l.startswith("root ::=")), "")
    open(f"{DATA}/grammar_{b}.gbnf", "w").write(g)
    pa.write(f"{b}\t{handler}\t{int(bool(p['grammar_lazy']))}\t{trig}\t{json.dumps(p['generation_prompt'])}\t"
             f"{len(g.encode())}\t{hashlib.md5(g.encode()).hexdigest()}\t{'yes' if len(gs) == 1 else 'no'}\t"
             f"{rule}\n")
pa.close()

for n in ("arms.tsv", "steps.tsv", "left_out.tsv", "server_lines.tsv", "params.tsv"):
    print(f"== {n}")
    print(open(f"{DATA}/{n}").read())
