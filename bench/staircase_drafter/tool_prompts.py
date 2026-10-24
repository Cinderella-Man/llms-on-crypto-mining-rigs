"""More tool-call prompts for the staircase guesser's training answers: first user turns with their tool lists from two
public tool-calling sets, rendered as OpenAI-style messages plus `tools`, so llama-server's template writes them in
Flint's own tool format.

- Team-ACE/ToolACE (Apache-2.0): a JSON array of functions in the system prompt (BFCL-style schemas, "dict" types).
- glaiveai/glaive-function-calling-v2 (Apache-2.0): function JSON objects in the system prompt, a "USER: ... ASSISTANT:"
  transcript; rows without functions are skipped.

usage: python tool_prompts.py OUT.jsonl CACHE_DIR --extend PROMPTS.jsonl [--glaive N] [--seed N]
  --extend: prompts already used (one or more files concatenated); new prompts are numbered after their last id and
            exact duplicates are dropped
needs: pip install huggingface_hub pyarrow
"""
import argparse
import json
import random

from prompts import TOOL_SYSTEM, shard


def fix_schema(x):
    """BFCL-style "dict" types to JSON-schema "object", recursively; drop null "required" fields."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k == "required" and v is None:
                continue
            if k == "type" and v == "dict":
                v = "object"
            out[k] = fix_schema(v)
        return out
    if isinstance(x, list):
        return [fix_schema(v) for v in x]
    return x


def toolace_rows(cache, rng):
    import pyarrow.parquet as pq
    rows = []
    for r in pq.read_table(shard(cache, "Team-ACE/ToolACE", "default/train/")).to_pylist():
        sys_text = r["system"] or ""
        i = sys_text.find("[{")
        if i < 0:
            continue
        try:
            funcs, _ = json.JSONDecoder().raw_decode(sys_text[i:])
        except ValueError:
            continue
        user = next((m["value"] for m in r["conversations"] if m.get("from") == "user"), None)
        if not user or not funcs:
            continue
        tools = [{"type": "function", "function": fix_schema(f)} for f in funcs if isinstance(f, dict) and "name" in f]
        rows.append({"category": "tool", "source": "Team-ACE/ToolACE", "tools": tools,
                     "messages": [{"role": "system", "content": TOOL_SYSTEM}, {"role": "user", "content": user}]})
    rng.shuffle(rows)
    return rows


def glaive_rows(cache, rng, n):
    import pyarrow.parquet as pq
    table = pq.read_table(shard(cache, "glaiveai/glaive-function-calling-v2", "default/train/")).to_pylist()
    rng.shuffle(table)
    rows = []
    dec = json.JSONDecoder()
    for r in table:
        if len(rows) >= n:
            break
        sys_text, chat = r["system"] or "", r["chat"] or ""
        i = sys_text.find("{")
        funcs = []
        while 0 <= i < len(sys_text):
            try:
                f, j = dec.raw_decode(sys_text[i:])
            except ValueError:
                break
            funcs.append(f)
            k = sys_text.find("{", i + j)
            i = k
        if not funcs or not chat.startswith("USER: "):
            continue
        user = chat[len("USER: "):].split("\n\n\nASSISTANT:")[0].strip()
        if not user:
            continue
        tools = [{"type": "function", "function": fix_schema(f)} for f in funcs if isinstance(f, dict) and "name" in f]
        rows.append({"category": "tool", "source": "glaiveai/glaive-function-calling-v2", "tools": tools,
                     "messages": [{"role": "system", "content": TOOL_SYSTEM}, {"role": "user", "content": user}]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("cache")
    ap.add_argument("--extend", required=True)
    ap.add_argument("--glaive", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = toolace_rows(args.cache, rng) + glaive_rows(args.cache, rng, args.glaive)
    rng.shuffle(rows)
    old = [json.loads(l) for l in open(args.extend)]
    offset = 1 + max(int(r["id"].split("_")[-1]) for r in old)
    seen = {json.dumps([r["messages"], r.get("tools")], sort_keys=True) for r in old}
    n = 0
    with open(args.out, "w") as f:
        for r in rows:
            key = json.dumps([r["messages"], r.get("tools")], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            r["id"] = f"tool_{offset + n}"
            f.write(json.dumps(r) + "\n")
            n += 1
    print(f"{n} tool prompts -> {args.out} (ids from {offset})")


if __name__ == "__main__":
    main()
