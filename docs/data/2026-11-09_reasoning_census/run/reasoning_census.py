"""Reasoning census of the agent check's replies (Q1152). Standard library only.

    python3 -I reasoning_census.py selftest
    python3 -I reasoning_census.py census --in IN --work WORK --tokenizer BIN --vocab MODEL=FILE.gguf …
                                          --out OUT --private PRIVATE [--loop-by-model by_model.tsv]

How much of what a model generates in the agent check is reasoning, how long the reasoning of one reply gets, and how
many reflection words it holds. It reads the replies the loop census read (docs/data/2026-11-09_loop_census): that
census's arm list, reply reader and token cache are used as they are (loop_census.py is loaded from its folder), so
each part of a reply (the reasoning, the answer, each tool call) is tokenized on its own with its model's vocabulary.

Per reply:
- its tokens by part, and its turn type:
    first       the first request of an attempt (the system prompt and the task, no reply yet);
    after_tool  a request that ends in a tool result, answered with another tool call;
    final       a request that ends in a tool result, answered without a tool call: the final answer;
    unfinished  a reply cut at the agent check's time cap or broken by a server failure, at any turn;
- the reflection words in its reasoning, counted by bench/reasoning_length/reflections.py (registry row Q94): its
  count_block() gets the reasoning's token ids and the vocabulary's token texts and counts the word-initial tokens that
  are a reflection word, and the reasoning tokens in the sentences and paragraphs that such a word opens;
- its decode time (the server's `predicted_ms`; for a reply cut mid-stream, the stream's duration) and its
  reasoning's share of it (its reasoning tokens times its own milliseconds per generated token).

`census` writes tables of counts, task ids and single words to OUT (no reply text) and one row per reply to PRIVATE.
"""
import argparse
import csv
import gzip
import importlib.util
import json
import math
import statistics
import struct
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3] if len(HERE.parents) > 3 else HERE
LOOP_CENSUS = REPO / "docs/data/2026-11-09_loop_census/run/loop_census.py"
REFLECTIONS = REPO / "bench/reasoning_length/reflections.py"
TURN_TYPES = ("first", "after_tool", "final", "unfinished", "other")
TOP = 0.10        # "the longest 10 % of replies"
OPENERS = 40      # sentence openers listed per group and model
LATE = 6          # turns from this one on are counted together


def load(path, name):
    sys.dont_write_bytecode = True  # no __pycache__ beside the two scripts this one borrows
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------------------- input

def request_roles(root, folder):
    """The message roles of every request in one proxy log: id -> roles."""
    out = {}
    with gzip.open(Path(root) / folder / "proxy.jsonl.bodies.gz", "rt") as g:
        for l in g:
            b = json.loads(l)
            out[b["id"]] = tuple(m.get("role") for m in (b.get("body") or {}).get("messages") or [])
    return out


def cut_of(r, finish, finished):
    """Why a reply did not end on its own ("" when it did), as the loop census names it."""
    return ("length" if finish == "length" else "client_gone" if r.get("client_gone") else
            "server_error" if r.get("upstream_error") or r.get("status") != 200 else "" if finish in finished
            else "other")


def turn_type(roles, finish, cut, n_calls):
    if cut:
        return "unfinished"
    if roles and roles[-1] == "user" and "assistant" not in roles:
        return "first"
    if roles and roles[-1] == "tool":
        return "final" if finish == "stop" and not n_calls else "after_tool" if n_calls else "other"
    return "other"


def token_bytes(vocab, dec):
    """Each token's bytes (GPT-2 byte-level BPE), to check that a part's ids give its text back."""
    return [bytes(dec[ch] for ch in t) if all(ch in dec for ch in t) else t.encode("utf-8") for t in vocab]


# --------------------------------------------------------------------------------------------------------- tables

def pct(x, y):
    return f"{100 * x / y:.2f}" if y else ""


def per1000(x, y):
    return f"{1000 * x / y:.2f}" if y else ""


def num(x):
    return "" if x == "" else f"{x:g}"


def pctl(ordered, q):
    """Nearest rank: the smallest value that at least the share q of the values do not exceed."""
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)] if ordered else ""


def top_share(values, frac=TOP):
    """How many values are the largest `frac` of them (rounded up), and their share of the sum."""
    if not values:
        return 0, ""
    k = math.ceil(frac * len(values))
    return k, pct(sum(sorted(values, reverse=True)[:k]), sum(values))


def summary(sel, att, whole=None):
    """One table row: sel are the replies, att the attempts' results whose task time the shares are taken of, whole
    the replies that a row split by turn is a part of (the of_all_* shares)."""
    txt = [r for r in sel if r["n_tok"]]
    whole = txt if whole is None else whole
    reas = [r for r in txt if r["n_reasoning"]]
    fin = [r for r in txt if r["predicted_n"] != ""]
    scored = [r for r in txt if r["scored"]]
    att = [x for x in att if x.get("status") == "scored"]
    wall = sum(x["wall_s"] for x in att)
    tokens, n_reas = sum(r["n_tok"] for r in txt), sum(r["n_reasoning"] for r in txt)
    server = sum(r["predicted_n"] for r in fin)
    c = Counter()
    for r in reas:
        c.update(r["_counts"])
    rs, ts = sorted(r["n_reasoning"] for r in txt), sorted(r["n_tok"] for r in txt)
    k, top_tok = top_share(ts)
    in_top = [r for r in txt if r["top10"]]
    gen_s, reas_s = sum(r["gen_s"] for r in scored), sum(r["reasoning_s"] for r in scored)
    return dict(
        replies=len(txt), replies_with_reasoning=len(reas), tokens=tokens, reasoning_tokens=n_reas,
        content_tokens=sum(r["n_content"] for r in txt), tool_tokens=sum(r["n_tool"] for r in txt),
        reasoning_pct=pct(n_reas, tokens), content_pct=pct(sum(r["n_content"] for r in txt), tokens),
        tool_pct=pct(sum(r["n_tool"] for r in txt), tokens),
        of_all_tokens_pct=pct(tokens, sum(r["n_tok"] for r in whole)),
        of_all_reasoning_pct=pct(n_reas, sum(r["n_reasoning"] for r in whole)),
        server_tokens=server, reasoning_pct_of_server=pct(sum(r["n_reasoning"] for r in fin), server),
        reflection_words=c["reflection_tokens"], reflection_per_1000=per1000(c["reflection_tokens"], n_reas),
        reflection_initial=c["reflection_initial"],
        reflection_initial_per_1000=per1000(c["reflection_initial"], n_reas),
        contrast_words=c["contrast_tokens"], contrast_per_1000=per1000(c["contrast_tokens"], n_reas),
        contrast_initial=c["contrast_initial"], contrast_initial_per_1000=per1000(c["contrast_initial"], n_reas),
        reflection_sentences=c["sentence_count"], reflection_sentence_tokens=c["sentence_tokens"],
        reflection_sentence_pct=pct(c["sentence_tokens"], n_reas),
        reflection_paragraphs=c["paragraph_count"], reflection_paragraph_tokens=c["paragraph_tokens"],
        reflection_paragraph_pct=pct(c["paragraph_tokens"], n_reas),
        reflection_word_sentences=c["sentence_word_count"],
        reflection_word_sentence_tokens=c["sentence_word_tokens"],
        reflection_word_sentence_pct=pct(c["sentence_word_tokens"], n_reas),
        reflection_word_paragraphs=c["paragraph_word_count"],
        reflection_word_paragraph_tokens=c["paragraph_word_tokens"],
        reflection_word_paragraph_pct=pct(c["paragraph_word_tokens"], n_reas),
        reasoning_median=num(statistics.median(rs)) if rs else "", reasoning_p90=pctl(rs, 0.9),
        reasoning_max=rs[-1] if rs else "",
        reasoning_median_nonzero=num(statistics.median(r["n_reasoning"] for r in reas)) if reas else "",
        tokens_median=num(statistics.median(ts)) if ts else "", tokens_p90=pctl(ts, 0.9),
        tokens_max=ts[-1] if ts else "",
        top10_replies=k, top10_token_pct=top_tok, top10_reasoning_pct=top_share(rs)[1] if n_reas else "",
        group_top10_replies=len(in_top), group_top10_tokens=sum(r["n_tok"] for r in in_top),
        group_top10_unfinished=sum(1 for r in in_top if r["cut"]),
        group_top10_in_passed=sum(1 for r in in_top if r["attempt_pass"] == 1),
        attempts=len(att), passed=sum(1 for x in att if x.get("success")),
        tokens_per_attempt=f"{sum(r['n_tok'] for r in scored) / len(att):.0f}" if att else "",
        reasoning_per_attempt=f"{sum(r['n_reasoning'] for r in scored) / len(att):.0f}" if att else "",
        task_s=round(wall, 1), generated_s=round(gen_s, 1), generated_time_pct=pct(gen_s, wall),
        reasoning_s=round(reas_s, 1), reasoning_time_pct=pct(reas_s, wall),
        reasoning_roundtrip=sum(1 for r in reas if r["roundtrip"] == 1))


def write(path, table, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(table)


# --------------------------------------------------------------------------------------------------------- census

def cmd_census(a):
    lc, refl = load(a.loop_census, "loop_census"), load(a.reflections, "reflections")
    arms = lc.load_arms(a.arms)
    vocabs = dict(v.split("=", 1) for v in a.vocab)
    tok = lc.Tokenizer(a.tokenizer, vocabs, a.work)
    out, priv = Path(a.out), Path(a.private)
    out.mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)
    n_logs, n_rows, n_copies, extra = lc.check_coverage(a.inp, arms)
    if extra:
        sys.exit(f"{len(extra)} proxy rows belong to no listed arm, e.g. {extra[:5]}")
    dec = refl.byte_decoder()
    texts, raw = {}, {}
    for model in sorted({x["model"] for x in arms}):
        vocab = refl.gguf_tokens(vocabs[model])
        texts[model], raw[model] = refl.token_texts(vocab), token_bytes(vocab, dec)

    roles, rows, attempts = {}, [], {}
    for arm, r, reply, res, (task, att, try_n, turn) in lc.arm_replies(a.inp, arms):
        if arm["folder"] not in roles:
            roles[arm["folder"]] = request_roles(a.inp, arm["folder"])
        model = arm["model"]
        parts = lc.parts_of(reply)
        ids = [tok.ids(model, t) for _, t in parts]
        n = {k: sum(len(i) for (kind, _), i in zip(parts, ids) if kind == k) for k in ("reasoning", "content", "tool")}
        n_tok = sum(n.values())
        finish = r.get("finish_reason") or ""
        cut = cut_of(r, finish, lc.FINISHED)
        counts, words, initial, tenths, openers, roundtrip = Counter(), Counter(), Counter(), [0] * 10, Counter(), ""
        for (kind, text), i in zip(parts, ids):
            if kind == "reasoning" and i:
                counts, words, initial, tenths, openers = refl.count_block(i, texts[model])
                roundtrip = int(b"".join(raw[model][t] for t in i) == text.encode("utf-8", "replace"))
        if r.get("predicted_n") and r.get("predicted_ms"):
            ms, gen_s = r["predicted_ms"] / r["predicted_n"], r["predicted_ms"] / 1000
        elif n_tok and r.get("t_first_token") is not None and r.get("t_end"):
            gen_s = r["t_end"] - r["t_first_token"]
            ms = 1000 * gen_s / n_tok
        else:
            ms = gen_s = 0.0
        if res is not None:
            attempts[(arm["folder"], arm["arm"], task, att, try_n)] = res
        rows.append(dict(
            group=arm["group"], model=model, folder=arm["folder"], arm=arm["arm"], task=task, attempt=att,
            try_n=try_n, turn=turn, turn_number=str(turn) if turn < LATE else f"{LATE}+", id=r["id"],
            turn_type=turn_type(roles[arm["folder"]].get(r["id"], ()), finish, cut,
                                sum(1 for kind, _ in parts if kind == "tool")),
            finish=finish, cut=cut, predicted_n="" if r.get("predicted_n") is None else r["predicted_n"],
            n_tok=n_tok, n_reasoning=n["reasoning"], n_content=n["content"], n_tool=n["tool"], roundtrip=roundtrip,
            reflection_words=counts["reflection_tokens"], reflection_initial=counts["reflection_initial"],
            contrast_words=counts["contrast_tokens"], contrast_initial=counts["contrast_initial"],
            reflection_sentences=counts["sentence_count"], reflection_sentence_tokens=counts["sentence_tokens"],
            reflection_paragraphs=counts["paragraph_count"], reflection_paragraph_tokens=counts["paragraph_tokens"],
            reflection_word_sentence_tokens=counts["sentence_word_tokens"],
            reflection_word_paragraph_tokens=counts["paragraph_word_tokens"],
            ms_per_token=round(ms, 3), gen_s=round(gen_s, 2), reasoning_s=round(n["reasoning"] * ms / 1000, 2),
            scored=int(res is not None and res.get("status") == "scored"),
            attempt_runaway=(res or {}).get("runaway", ""), attempt_pass=(res or {}).get("success", ""),
            attempt_wall_s=(res or {}).get("wall_s", ""), top10=0,
            _counts=counts, _words=words, _initial=initial, _tenths=tenths, _openers=openers))
    if len(rows) != n_rows:
        sys.exit(f"{n_rows} proxy rows in the listed arms, {len(rows)} read: a tag does not parse")
    other = [r for r in rows if r["turn_type"] == "other"]
    if other:
        sys.exit(f"{len(other)} replies have no turn type, e.g. {[(r['folder'], r['id']) for r in other[:3]]}")
    bad = [r for r in rows if r["roundtrip"] == 0]
    with_reasoning = sum(1 for r in rows if r["n_reasoning"])
    if len(bad) > 0.01 * with_reasoning:
        sys.exit(f"the reasoning's ids do not give its text back in {len(bad)} of {with_reasoning} replies: "
                 f"the wrong vocabulary? e.g. {[(r['model'], r['id']) for r in bad[:3]]}")

    # the longest 10 % of replies of each group and model, by generated tokens
    by_gm = defaultdict(list)
    for r in rows:
        if r["n_tok"]:
            by_gm[(r["group"], r["model"])].append(r)
    for lst in by_gm.values():
        lst.sort(key=lambda r: (-r["n_tok"], r["folder"], r["arm"], r["id"]))
        for r in lst[:math.ceil(TOP * len(lst))]:
            r["top10"] = 1

    arm_order = {(x["folder"], x["arm"]): i for i, x in enumerate(arms)}
    meta = {(x["folder"], x["arm"]): x for x in arms}

    def grouped(keys, split=None):
        """One row per value of `keys`, or per value of `keys` and of the reply's `split` field (the turn type or
        the turn's number); a split row's attempts and task time are those of its `keys`, so its shares add up."""
        sel, att, whole = defaultdict(list), defaultdict(list), defaultdict(list)
        for r in rows:
            sel[tuple(r[k] for k in keys) + ((r[split],) if split else ())].append(r)
            whole[tuple(r[k] for k in keys)].append(r)
        for (folder, arm, task, _, _), res in attempts.items():
            full = dict(meta[(folder, arm)], task=task)
            att[tuple(full[k] for k in keys)].append(res)
        names = keys + ([split] if split else [])
        table = [dict(zip(names, k), **summary(v, att.get(k[:len(keys)], []), whole[k[:len(keys)]]))
                 for k, v in sel.items()]
        return sorted(table, key=lambda x: (x["group"], x["model"], arm_order.get((x.get("folder"), x.get("arm")), 0),
                                            x.get("task", ""), TURN_TYPES.index(x.get("turn_type", "first")),
                                            x.get("turn_number", "")))

    cols = list(summary([], []))
    by_model, by_model_turn = grouped(["group", "model"]), grouped(["group", "model"], "turn_type")
    write(out / "by_model.tsv", by_model, ["group", "model"] + cols)
    write(out / "by_model_turn.tsv", by_model_turn, ["group", "model", "turn_type"] + cols)
    write(out / "by_turn_number.tsv", grouped(["group", "model"], "turn_number"),
          ["group", "model", "turn_number"] + cols)
    write(out / "by_arm.tsv", grouped(["group", "model", "folder", "arm"]), ["group", "model", "folder", "arm"] + cols)
    write(out / "by_arm_turn.tsv", grouped(["group", "model", "folder", "arm"], "turn_type"),
          ["group", "model", "folder", "arm", "turn_type"] + cols)
    write(out / "by_task.tsv", grouped(["group", "model", "task"]), ["group", "model", "task"] + cols)

    # the words behind the rate, where in the reasoning they fall, and what the model's sentences open with
    word_rows, tenth_rows, opener_rows = [], [], []
    for (group, model), lst in sorted(by_gm.items()):
        words, initial, openers, tenths = Counter(), Counter(), Counter(), [0] * 10
        for r in lst:
            words.update(r["_words"])
            initial.update(r["_initial"])
            openers.update(r["_openers"])
            tenths = [x + y for x, y in zip(tenths, r["_tenths"])]
        n_reas = sum(r["n_reasoning"] for r in lst)
        for kind, names in (("reflection", refl.REFLECTION), ("contrast", refl.CONTRAST),
                            ("phrase", ["phrase: " + p for p in refl.PHRASES])):
            for w in names:
                word_rows.append(dict(group=group, model=model, kind=kind, word=w.replace("phrase: ", ""),
                                      count=words[w], sentence_initial="" if kind == "phrase" else initial[w],
                                      per_1000=per1000(words[w], n_reas)))
        tenth_rows.append(dict(group=group, model=model, reflection_words=sum(tenths),
                               **{f"tenth_{i + 1}": t for i, t in enumerate(tenths)}))
        total = sum(openers.values())
        for rank, (w, cnt) in enumerate(openers.most_common(OPENERS), 1):
            opener_rows.append(dict(group=group, model=model, rank=rank, word=w, sentences=cnt,
                                    pct=pct(cnt, total), reflection_word=int(w.lower().rstrip("'") in refl.REFLECTION)))
    write(out / "words.tsv", word_rows, ["group", "model", "kind", "word", "count", "sentence_initial", "per_1000"])
    write(out / "tenths.tsv", tenth_rows, ["group", "model", "reflection_words"] + [f"tenth_{i}" for i in range(1, 11)])
    write(out / "openers.tsv", opener_rows, ["group", "model", "rank", "word", "sentences", "pct", "reflection_word"])

    ident = ["group", "model", "folder", "arm", "task", "attempt", "try_n", "turn"]
    longest = sorted((r for r in rows if r["top10"]),
                     key=lambda r: (r["group"], r["model"], -r["n_tok"], r["folder"], r["arm"], r["id"]))
    write(out / "longest.tsv", longest, ident + ["turn_type", "finish", "cut", "n_tok", "n_reasoning", "n_content",
                                                 "n_tool", "reflection_words", "attempt_runaway", "attempt_pass"])
    write(priv / "replies.tsv", rows, [k for k in rows[0] if not k.startswith("_")])

    if a.loop_by_model:
        with open(a.loop_by_model, newline="") as f:
            theirs = {(x["group"], x["model"]): (int(x["replies"]), int(x["tokens"]))
                      for x in csv.DictReader(f, delimiter="\t")}
        mine = {(x["group"], x["model"]): (x["replies"], x["tokens"]) for x in by_model}
        if mine != theirs:
            sys.exit(f"replies and tokens differ from the loop census: {mine} against {theirs}")

    with open(out / "summary.md", "w") as f:
        f.write(f"{n_logs} proxy logs, {n_rows} replies in {len(arms)} arms ({n_copies} more rows are copies of "
                f"those in a later log), {len(attempts)} attempts with a result. The reasoning's token ids give its "
                f"text back in {sum(1 for r in rows if r['roundtrip'] == 1)} of {with_reasoning} replies with "
                f"reasoning." + (" Replies and tokens per group and model equal the loop census's.\n\n"
                                 if a.loop_by_model else "\n\n"))
        show = ["replies", "replies_with_reasoning", "tokens", "of_all_tokens_pct", "of_all_reasoning_pct",
                "reasoning_pct", "content_pct", "tool_pct",
                "reasoning_pct_of_server", "reflection_words", "reflection_per_1000", "reflection_initial_per_1000",
                "contrast_per_1000", "reflection_sentence_pct", "reflection_word_sentence_pct",
                "reflection_paragraph_pct", "reflection_word_paragraph_pct", "reasoning_median",
                "reasoning_p90", "reasoning_max", "top10_replies", "top10_token_pct", "top10_reasoning_pct",
                "attempts", "tokens_per_attempt", "reasoning_per_attempt", "generated_time_pct", "reasoning_time_pct"]
        for title, table, keys in (("Per group and model", by_model, ["group", "model"]),
                                   ("Per group, model and turn type", by_model_turn, ["group", "model", "turn_type"])):
            f.write(f"## {title}\n\n| " + " | ".join(keys + show) + " |\n|" + "---|" * len(keys + show) + "\n")
            for x in table:
                f.write("| " + " | ".join(str(x[k]) for k in keys + show) + " |\n")
            f.write("\n")
    print((out / "summary.md").read_text())


# ----------------------------------------------------------------------------------------------------- self-test

def selftest(a):
    lc, refl = load(a.loop_census, "loop_census"), load(a.reflections, "reflections")
    assert turn_type(("system", "user"), "tool_calls", "", 2) == "first"
    assert turn_type(("system", "user", "assistant", "tool"), "tool_calls", "", 1) == "after_tool"
    assert turn_type(("system", "user", "assistant", "tool"), "stop", "", 0) == "final"
    assert turn_type(("system", "user", "assistant", "tool"), "", "client_gone", 0) == "unfinished"
    assert turn_type(("system", "user"), "", "server_error", 0) == "unfinished"
    assert turn_type(("system", "user", "assistant", "user"), "stop", "", 0) == "other"
    assert pctl(list(range(1, 11)), 0.9) == 9 and pctl([5], 0.9) == 5 and pctl([], 0.9) == ""
    assert top_share(list(range(1, 11))) == (1, "18.18") and top_share(list(range(1, 12))) == (2, "31.82")
    print("turn types, percentile, top share: ok")

    # a toy vocabulary and one block with three reflection words and two contrast words
    block = ["Wait", ",", " the", " value", " is", " wrong", ".", " But", " maybe", " not", ".", "\n\n", "Hmm", " let",
             " me", " check", ".", " Actually", " fine"]
    vocab = sorted(set(block) | {"The", "Done", "Let"})

    def ids(toks):
        return [vocab.index(t) for t in toks]
    c, words, initial, tenths, _ = refl.count_block(ids(block), vocab)
    assert (c["reasoning_tokens"], c["reflection_tokens"], c["reflection_initial"]) == (19, 3, 3), c
    assert (c["contrast_tokens"], c["contrast_initial"]) == (2, 1), c
    assert (c["sentence_count"], c["sentence_tokens"], c["paragraph_count"], c["paragraph_tokens"]) == (3, 14, 2, 18), c
    assert words["wait"] == words["hmm"] == words["actually"] == 1 and words["phrase: let me check"] == 1, words
    assert tenths == [1, 0, 0, 0, 0, 0, 1, 0, 1, 0], tenths
    assert (c["sentence_word_count"], c["sentence_word_tokens"], c["paragraph_word_tokens"]) == (3, 14, 18), c
    c = refl.count_block(ids(["Let", " me", " check", " the", " value", "."]), vocab)[0]
    assert (c["reflection_tokens"], c["sentence_count"], c["sentence_tokens"]) == (0, 1, 6), c
    assert c["sentence_word_count"] == 0, "a sentence that only a phrase opens"
    print("count_block: ok")

    # the whole census on two built attempts of one arm
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        folder = tmp / "in/2026-01-01_toy/agent-report"
        (tmp / "work").mkdir()
        gguf = tmp / "toy.gguf"

        def s(x):
            b = x.encode()
            return struct.pack("<Q", len(b)) + b
        coded = [t.replace(" ", "Ġ").replace("\n", "Ċ") for t in vocab]
        gguf.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + s("tokenizer.ggml.tokens")
                         + struct.pack("<IIQ", 9, 8, len(coded)) + b"".join(s(t) for t in coded))
        plain = ["The", " value", " is", " fine", ".", " fine"]
        replies = [  # attempt, turn, reasoning, answer, tool call, how it ended
            (1, 1, block, "", True, "tool_calls"), (1, 2, plain, "", True, "tool_calls"),
            (1, 3, ["Hmm", ",", " the", " value", " is", " fine", "."], "Done.", False, "stop"),
            (2, 1, ["The", " value", " is", " wrong", "."], "", True, "tool_calls"),
            (2, 2, [" the", " value"] * 20, "", False, "")]
        cache, log, bodies = {}, [], []
        for att, turn, reasoning, answer, call, finish in replies:
            reply = dict(reasoning_content="".join(reasoning), content=answer,
                         tool_calls=[dict(name="read", arguments='{"path": "a%d%d.py"}' % (att, turn))] if call else [])
            for kind, text in lc.parts_of(reply):
                cache[lc.Tokenizer.key(text)] = ids(reasoning) if kind == "reasoning" else [0] * (5 if call else 3)
            n_tok = len(reasoning) + (5 if call else 3 if answer else 0)
            rid = f"r{att}{turn}"
            row = dict(id=rid, tag=f"x/bugfix-python-toy/a{att}/t1/{turn}", status=200, finish_reason=finish or None)
            if finish:
                row.update(predicted_n=n_tok + 2, predicted_ms=100.0 * (n_tok + 2))
            else:
                row.update(client_gone=True, t_first_token=1.0, t_end=5.0)
            log.append(row)
            msgs = ["system", "user"] + ["assistant", "tool"] * (turn - 1)
            bodies.append(dict(id=rid, body=dict(messages=[dict(role=m) for m in msgs]), reply=reply))
        folder.mkdir(parents=True)
        (folder / "proxy.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
        with gzip.open(folder / "proxy.jsonl.bodies.gz", "wt") as g:
            g.write("".join(json.dumps(b) + "\n" for b in bodies))
        for att, wall, ok, runaway in ((1, 100.0, 1, ""), (2, 200.0, 0, "time")):
            d = folder / f"res/x/bugfix-python-toy/a{att}"
            d.mkdir(parents=True)
            (d / "result.json").write_text(json.dumps(dict(tag=f"x/bugfix-python-toy/a{att}/t1", status="scored",
                                                           wall_s=wall, success=ok, runaway=runaway)))
        with gzip.open(tmp / "work/tokens_toy.jsonl.gz", "wt") as g:
            g.write("".join(json.dumps(dict(k=k, ids=v)) + "\n" for k, v in cache.items()))
        (tmp / "arms.tsv").write_text("group\tmodel\tfolder\tarm\ng\ttoy\t2026-01-01_toy/agent-report\tx\n")
        args = argparse.Namespace(loop_census=a.loop_census, reflections=a.reflections, arms=tmp / "arms.tsv",
                                  vocab=[f"toy={gguf}"], tokenizer="/bin/false", work=tmp / "work", inp=tmp / "in",
                                  out=tmp / "out", private=tmp / "private", loop_by_model=None)
        stdout, sys.stdout = sys.stdout, open(tmp / "census.log", "w")
        try:
            cmd_census(args)
        finally:
            sys.stdout.close()
            sys.stdout = stdout

        def table(name, where="out"):
            with open(tmp / where / name, newline="") as f:
                return list(csv.DictReader(f, delimiter="\t"))
        (m,) = table("by_model.tsv")
        want = dict(replies="5", replies_with_reasoning="5", tokens="95", reasoning_tokens="77", content_tokens="3",
                    tool_tokens="15", reasoning_pct="81.05", server_tokens="63", reasoning_pct_of_server="58.73",
                    reflection_words="4", reflection_per_1000="51.95", contrast_words="2", reflection_sentences="4",
                    reflection_sentence_tokens="21", reflection_sentence_pct="27.27",
                    reflection_word_sentences="4", reflection_word_sentence_pct="27.27",
                    reflection_word_paragraph_tokens="25", reasoning_median="7",
                    reasoning_p90="40", reasoning_max="40", tokens_median="11", top10_replies="1",
                    top10_token_pct="42.11", top10_reasoning_pct="51.95", group_top10_replies="1",
                    group_top10_unfinished="1", attempts="2", passed="1", tokens_per_attempt="48",
                    reasoning_per_attempt="38", task_s="300.0", generated_s="10.3", generated_time_pct="3.43",
                    reasoning_s="7.7", reasoning_time_pct="2.57", reasoning_roundtrip="5")
        assert {k: m[k] for k in want} == want, {k: (m[k], v) for k, v in want.items() if m[k] != v}
        turns = {x["turn_type"]: x for x in table("by_model_turn.tsv")}
        assert list(turns) == ["first", "after_tool", "final", "unfinished"], list(turns)
        assert [turns[t]["reasoning_pct"] for t in turns] == ["70.59", "54.55", "70.00", "100.00"]
        assert [turns[t]["replies"] for t in turns] == ["2", "1", "1", "1"]
        assert [turns[t]["reflection_per_1000"] for t in turns] == ["125.00", "0.00", "142.86", "0.00"]
        assert [turns[t]["reasoning_time_pct"] for t in turns] == ["0.80", "0.20", "0.23", "1.33"]
        assert turns["first"]["attempts"] == "2" and turns["first"]["task_s"] == "300.0"
        assert (turns["first"]["of_all_tokens_pct"], turns["first"]["of_all_reasoning_pct"]) == ("35.79", "31.17")
        assert (m["of_all_tokens_pct"], m["of_all_reasoning_pct"]) == ("100.00", "100.00")
        (lg,) = table("longest.tsv")
        assert (lg["turn_type"], lg["cut"], lg["n_tok"], lg["attempt_runaway"]) == ("unfinished", "client_gone", "40",
                                                                                   "time"), lg
        w = {x["word"]: x for x in table("words.tsv")}
        assert (w["hmm"]["count"], w["hmm"]["sentence_initial"], w["but"]["count"], w["wait,"]["count"]) == \
            ("2", "2", "1", "1"), w
        assert [x["tenth_1"] for x in table("tenths.tsv")] == ["2"]
        assert {x["word"]: x["sentences"] for x in table("openers.tsv")}["The"] == "2"
        assert len(table("replies.tsv", "private")) == 5 and len(table("by_arm_turn.tsv")) == 4
        assert [(x["turn_number"], x["replies"], x["reasoning_tokens"]) for x in table("by_turn_number.tsv")] == \
            [("1", "2", "24"), ("2", "2", "46"), ("3", "1", "7")]
    print("census on a built arm: ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("selftest", selftest), ("census", cmd_census)):
        p = sub.add_parser(name)
        p.add_argument("--loop-census", default=LOOP_CENSUS, help="the loop census's script (reader, token cache)")
        p.add_argument("--reflections", default=REFLECTIONS, help="bench/reasoning_length/reflections.py")
        if name == "census":
            p.add_argument("--in", dest="inp", required=True)
            p.add_argument("--arms", default=None, help="default: the loop census's arms.tsv")
            p.add_argument("--work", required=True, help="the loop census's token cache")
            p.add_argument("--tokenizer", help="llama.cpp's llama-tokenize, for texts the cache lacks")
            p.add_argument("--vocab", action="append", default=[], metavar="MODEL=FILE.gguf")
            p.add_argument("--out", required=True)
            p.add_argument("--private", required=True)
            p.add_argument("--loop-by-model", help="the loop census's by_model.tsv: replies and tokens must match")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    if a.cmd == "census" and a.arms is None:
        a.arms = Path(a.loop_census).parent / "arms.tsv"
    a.fn(a)


if __name__ == "__main__":
    main()
