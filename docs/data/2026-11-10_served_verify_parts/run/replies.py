#!/usr/bin/env python3
"""replies.py: the server's own timings per reply of verify_prof.sh's loads (R49), and whether every reply carries
the first one's text (its reasoning and its answer) and draft counts.

  python3 -I replies.py OUT --loads warm,prof [--requests 2] [--tsv replies.tsv] > replies.md

OUT/<load>/reply_greedy-<i>.json are llama-server's replies. Steps by the server's counts, as in
2026-11-10_served_step_split/run/arms.py: generated tokens minus accepted draft tokens (each verify pass gives one
sampled token and the drafts it accepts); ms a step = the reply's generation time over that count. Standard library
only."""
import argparse, hashlib, json, os

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--loads", default="warm,prof"); ap.add_argument("--requests", type=int, default=2)
ap.add_argument("--tsv", default="")
a = ap.parse_args()
cols = ["load", "request", "text_md5", "same_text_as_first", "prompt_n", "predicted_n", "draft_n", "draft_n_accepted",
        "steps_server", "tokens_a_step", "predicted_ms", "ms_a_step_server", "tok_s", "prompt_ms"]
rows, first = [], None
for ld in a.loads.split(","):
    for i in range(1, a.requests + 1):
        p = os.path.join(a.out, ld, f"reply_greedy-{i}.json")
        r = json.load(open(p))
        msg = r["choices"][0]["message"]  # the reasoning and the answer (--reasoning-format deepseek splits them)
        text = (msg.get("reasoning_content") or "") + "\n---\n" + (msg.get("content") or "")
        t = r.get("timings", {})
        pn, dn, da = t.get("predicted_n", 0), t.get("draft_n", 0), t.get("draft_n_accepted", 0)
        key = (hashlib.md5(text.encode()).hexdigest(), pn, dn, da)
        first = first or key
        steps = pn - da
        rows.append(dict(load=ld, request=f"greedy-{i}", text_md5=key[0][:12], same_text_as_first="yes" if key[0] == first[0] else "NO",
                         prompt_n=t.get("prompt_n", 0), predicted_n=pn, draft_n=dn, draft_n_accepted=da, steps_server=steps,
                         tokens_a_step=pn / steps if steps else float("nan"), predicted_ms=t.get("predicted_ms", 0.0),
                         ms_a_step_server=t.get("predicted_ms", 0.0) / steps if steps else float("nan"),
                         tok_s=t.get("predicted_per_second", 0.0), prompt_ms=t.get("prompt_ms", 0.0)))
fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
print("| " + " | ".join(cols) + " |")
print("|---|---" + "|---:" * (len(cols) - 2) + "|")
for r in rows:
    print("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
if a.tsv:
    with open(a.tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(fmt(r[c]) for c in cols) + "\n")
