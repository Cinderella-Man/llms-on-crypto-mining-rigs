"""Step 1 of the copy-span analysis: from agent-check proxy logs to token ids.

For every answered chat request of the given runs it rebuilds the prompt (render_prompt) and the generated text
(build_output), tokenizes both with llama-tokenize on a vocabulary-only GGUF, labels every generated token with its
region (reasoning, answer, call scaffolding, each tool argument) and the markers it follows, and checks the rebuild
against the server's own counts (prompt tokens, generated tokens).

    RUNS="baseline=/path/to/run1 dflash2=/path/to/run2" WORK=/path/to/work TOKENIZE=/path/llama-tokenize \
    VOCAB=/path/vocab.gguf python3 prepare.py

Each run folder holds proxy.jsonl and proxy.jsonl.bodies.gz. Writes, under $WORK/prep/: tokens.jsonl.gz (ids, regions,
markers; no text), texts.jsonl.gz (the rebuilt prompt and output; transcript text, keep private), render_check.csv.
"""
import bisect
import csv
import gzip
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (build_output, load_run, load_vocab, render_prompt, tokenize)  # noqa: E402


def main():
    runs = [r.split("=", 1) for r in os.environ["RUNS"].split()]
    work = os.environ["WORK"]
    tok_bin, vocab = os.environ["TOKENIZE"], os.environ["VOCAB"]
    out_dir = os.path.join(work, "prep")
    os.makedirs(out_dir, exist_ok=True)
    table = load_vocab(vocab)

    reqs = []
    for name, path in runs:
        reqs += load_run(path, name)
        print("%s: %d answered chat requests so far" % (name, len(reqs)), file=sys.stderr)

    prompts, outputs, quot = [], [], []
    meta_out = []
    for r in reqs:
        ptext, quotable = render_prompt(r["body"])
        otext, starts, regions, markers = build_output(r["reply"], r["finish"])
        prompts.append(ptext)
        outputs.append(otext)
        quot.append(quotable)
        meta_out.append((starts, regions, markers))

    uniq = sorted({json.dumps(q, ensure_ascii=False)[1:-1] for qs in quot for q in qs if q})
    texts = prompts + outputs + uniq
    ids = tokenize(texts, tok_bin, vocab, table)
    n = len(reqs)
    p_ids, o_ids = ids[:n], ids[n:2 * n]
    esc_ids = dict(zip(uniq, ids[2 * n:]))

    rows, tok_lines, text_lines = [], [], []
    for r, ptext, otext, pid, oid, qs, (starts, regions, markers) in zip(reqs, prompts, outputs, p_ids, o_ids, quot, meta_out):
        offs, o = [], 0
        for t in oid:
            offs.append(o)
            o += len(table[t])
        reg_idx = [max(0, bisect.bisect_right(starts, off) - 1) for off in offs]
        mk = [[bisect.bisect_left(offs, off), kind] for off, kind in markers]
        aux = []
        for q in qs:
            if q:
                aux += esc_ids[json.dumps(q, ensure_ascii=False)[1:-1]]
        head = {k: r[k] for k in ("run", "arm", "task", "attempt", "try_", "turn", "session", "id", "finish")}
        tok_lines.append(json.dumps(dict(head, ctx=pid, out=oid, reg=reg_idx, rnames=regions, markers=mk, aux=aux)))
        text_lines.append(json.dumps(dict(id=r["id"], prompt=ptext, output=otext)))
        rows.append(dict(head, prompt_tokens_server=r["prompt_tokens"], prompt_tokens_rebuilt=len(pid),
                         generated_server=r["predicted_n"], generated_rebuilt=len(oid), finish=r["finish"]))

    with gzip.open(os.path.join(out_dir, "tokens.jsonl.gz"), "wt", encoding="utf-8") as f:
        f.write("\n".join(tok_lines) + "\n")
    with gzip.open(os.path.join(out_dir, "texts.jsonl.gz"), "wt", encoding="utf-8") as f:
        f.write("\n".join(text_lines) + "\n")
    with open(os.path.join(out_dir, "render_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    exact = sum(1 for x in rows if x["prompt_tokens_rebuilt"] == x["prompt_tokens_server"])
    pdiff = [x["prompt_tokens_rebuilt"] - x["prompt_tokens_server"] for x in rows]
    gdiff = [x["generated_rebuilt"] - x["generated_server"] for x in rows if x["finish"] != "length"]
    print("requests %d; prompt token count equal to the server's in %d (differences: min %d, max %d)" % (
        n, exact, min(pdiff), max(pdiff)))
    print("generated tokens, rebuilt minus server's: mean %.2f, median %d, min %d, max %d; |diff| <= 2 in %d of %d" % (
        statistics.mean(gdiff), statistics.median(gdiff), min(gdiff), max(gdiff),
        sum(1 for d in gdiff if abs(d) <= 2), len(gdiff)))
    print("total generated: rebuilt %d, server %d" % (sum(x["generated_rebuilt"] for x in rows),
                                                      sum(x["generated_server"] for x in rows)))


if __name__ == "__main__":
    main()
