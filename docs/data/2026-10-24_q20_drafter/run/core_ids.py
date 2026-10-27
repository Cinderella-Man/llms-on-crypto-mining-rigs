"""Q20/M3 (2026-10-25): the clean-by-construction core: answers to my own prompts on the Python standard library,
llama.cpp proper and the allowed site-packages (edit, agent and implement-this-function "code" categories of
strict-ids.txt), without maths and without the generic tool-call sets until their sources are cleared. Writes
WORK_DIR/strict-ids-core.txt and prints answers, answer tokens and how many are dumped, per writer.

usage: python3 -I core_ids.py WORK_DIR
  WORK_DIR holds the drafter's prompt files, answer files (gen-<label>.jsonl), the dump maps (map-d*.jsonl,
  dumpD6*.log) and strict-ids.txt (from count_strict.py and strict_sitepkgs.py).
"""
import collections
import glob
import json
import sys

W = sys.argv[1]
cat = {}
for f in glob.glob(f"{W}/prompts*.jsonl"):
    if "allused" in f or "all12" in f or ".v1." in f:
        continue
    for line in open(f):
        r = json.loads(line)
        cat[r["id"]] = r["category"]
strict = [x.strip() for x in open(f"{W}/strict-ids.txt") if x.strip()]
core = [i for i in strict if cat.get(i) in ("edit", "agent", "code")]
with open(f"{W}/strict-ids-core.txt", "w") as f:
    f.write("\n".join(core) + "\n")
dumped = set()
for m in ("map-d1.jsonl", "map-d2.jsonl", "map-d3.jsonl", "map-d4.done.jsonl", "map-d5.done.jsonl"):
    dumped |= {json.loads(line)["id"] for line in open(f"{W}/{m}")}
for x in "abc":
    sid2id = {json.loads(line)["seq_id"]: json.loads(line)["id"] for line in open(f"{W}/map-d6{x}.jsonl")}
    for line in open(f"{W}/dumpD6{x}.log", errors="replace"):
        if line.startswith("sequence "):
            s = int(line.split()[1].rstrip(":"))
            if s in sid2id:
                dumped.add(sid2id[s])
coreset = set(core)
groups = {"Q8_0": "A B B2 B3 C C2 C2s", "G3": "E1 E2 F1 F2", "UD-Q4_K_XL": "FV3", "NVFP4 vLLM": "V1"}
tot = collections.Counter()
for w, names in groups.items():
    c = collections.Counter()
    for nm in names.split():
        for line in open(f"{W}/gen-{nm}.jsonl"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r["id"] in coreset:
                k = len(r["gen_tokens"])
                c["n"] += 1
                c["t"] += k
                c[r["category"] + "_t"] += k
                if r["id"] in dumped:
                    c["dn"] += 1
                    c["dt"] += k
    tot.update(c)
    print(f"{w}: {c['n']} answers / {c['t'] / 1e6:.2f} M (edit {c['edit_t'] / 1e6:.2f}, agent {c['agent_t'] / 1e6:.2f}); "
          f"dumped {c['dn']} / {c['dt'] / 1e6:.2f} M")
print(f"CORE: {tot['n']} answers / {tot['t'] / 1e6:.2f} M answer tokens (edit {tot['edit_t'] / 1e6:.2f} M, agent "
      f"{tot['agent_t'] / 1e6:.2f} M); dumped {tot['dn']} / {tot['dt'] / 1e6:.2f} M; undumped {tot['n'] - tot['dn']}")
