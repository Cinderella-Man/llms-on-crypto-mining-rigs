"""Does vLLM's prefix caching give the same tokens as no caching for Flint (with the patched attention of
bench/staircase_drafter/vllm/patch_flint_attention.py)? An agent session resends its whole history every turn, so the
long run wants the cache on; this checks it first, on real session contexts.

  collect   for each session in --sessions (agent.py session files), the token context before each of its first
            --turns turns, sent in turn order (with the cache on, each request reuses the previous turn's context as
            its prefix), greedy, --max-tokens new tokens; writes one JSON line per context
  compare   two collect outputs (one from a server without the cache, one with): exact matches, the first diverging
            position of the others

usage: python vllm_cache_check.py collect --server URL --sessions DIR [DIR ...] --out OUT.jsonl [--turns 4]
                                          [--max-tokens 48] [--limit 12]
       python vllm_cache_check.py compare NOCACHE.jsonl CACHE.jsonl
Standard library only.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agx  # noqa: E402
from agent import post  # noqa: E402


def cmd_collect(a):
    files = []
    for d in a.sessions:
        files += sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json") and not f.endswith(".error.json"))
    with open(a.out, "w") as out:
        for path in files[:a.limit]:
            s = json.load(open(path))
            seq, _ = agx.session_sequence(s["prompt_tokens"], s["turns"])
            for t in s["turns"][:a.turns]:
                ctx = seq[:t["context_len"]]
                r = post(a.server.rstrip("/") + "/v1/completions",
                         {"model": "flint", "prompt": ctx, "max_tokens": a.max_tokens, "temperature": 0, "seed": 0,
                          "return_token_ids": True, "skip_special_tokens": False})
                ids = r["choices"][0].get("token_ids") or []
                out.write(json.dumps({"session": s["session_id"], "turn": t["turn"], "context": len(ctx),
                                      "tokens": ids}) + "\n")
                out.flush()
                print(f"{s['session_id']} turn {t['turn']}: {len(ctx)} context tokens, {len(ids)} out", flush=True)


def cmd_compare(a):
    rows = [{(r["session"], r["turn"]): r for r in agx.read_jsonl(p)} for p in (a.nocache, a.cache)]
    keys = sorted(set(rows[0]) & set(rows[1]))
    same, diverge = 0, []
    for k in keys:
        x, y = rows[0][k]["tokens"], rows[1][k]["tokens"]
        if x == y:
            same += 1
        else:
            first = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            diverge.append((k, rows[0][k]["context"], first))
    print(f"{same} of {len(keys)} contexts give the same tokens with and without the cache")
    for (sid, turn), ctx, first in diverge:
        print(f"  {sid} turn {turn} ({ctx} context tokens): first difference at token {first}")
    sys.exit(0 if same == len(keys) else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--server", required=True)
    c.add_argument("--sessions", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--turns", type=int, default=4)
    c.add_argument("--max-tokens", type=int, default=48)
    c.add_argument("--limit", type=int, default=12, help="sessions")
    m = sub.add_parser("compare")
    m.add_argument("nocache")
    m.add_argument("cache")
    a = ap.parse_args()
    {"collect": cmd_collect, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
