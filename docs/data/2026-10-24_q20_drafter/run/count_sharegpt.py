"""Q20 (2026-10-25): how much of the generated drafter text answers ShareGPT prompts (Aeala/ShareGPT_Vicuna_unfiltered),
per writer and prompt file; the published data leaves those out.

usage: python3 -I count_sharegpt.py WORK_DIR
  WORK_DIR holds the drafter's prompt files and answer files (gen-<label>.jsonl).
"""
import collections
import json
import sys

W = sys.argv[1]
src, per_file = {}, collections.Counter()
for f in ("prompts.jsonl", "prompts-ext.jsonl", "prompts-ext2.jsonl", "prompts-tools.jsonl"):
    for line in open(f"{W}/{f}"):
        r = json.loads(line)
        src[r["id"]] = r.get("source", "")
        if "sharegpt" in src[r["id"]].lower():
            per_file[f] += 1
print("ShareGPT prompts per file:", dict(per_file), "total", sum(per_file.values()))
print("sources matched:", sorted({s for s in src.values() if "sharegpt" in s.lower()}))
groups = {"Q8_0 llama.cpp": "A B B2 B3 C C2 C2s", "G3 llama.cpp": "E1 E2 F1 F2", "UD-Q4_K_XL llama.cpp": "FV3",
          "NVFP4 vLLM": "V1"}
tot = collections.Counter()
for g, names in groups.items():
    c = collections.Counter()
    for nm in names.split():
        for line in open(f"{W}/gen-{nm}.jsonl"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            k = len(r["gen_tokens"])
            c["n"] += 1
            c["t"] += k
            if "sharegpt" in src.get(r["id"], r.get("source", "")).lower():
                c["sn"] += 1
                c["st"] += k
    tot.update(c)
    print(f"{g}: {c['sn']} of {c['n']} answers from ShareGPT prompts, {c['st'] / 1e6:.2f} of {c['t'] / 1e6:.2f} M "
          f"answer tokens ({c['st'] / max(c['t'], 1):.1%})")
print(f"total: {tot['sn']} of {tot['n']} answers, {tot['st'] / 1e6:.2f} of {tot['t'] / 1e6:.2f} M answer tokens "
      f"({tot['st'] / tot['t']:.1%})")
