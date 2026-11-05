#!/usr/bin/env python3
"""rig_lines_table.py: C200r's server lines as one table. Reads the line-<arm>.out files fn_line.sh wrote (its JSON
line per arm), writes one TSV row per arm and request, and prints the comparison against the first arm (decode and
prompt tok/s per request, the change in per cent) and the identity hashes of the two temperature-0 prompts.
  python3 rig_lines_table.py OUT.tsv f32=<dir>/line-f32.out q8=<dir>/line-q8.out ...
The first arm is the reference; arms that copy the same bytes (f32 and staged, q8 and q8staged) must give the same
hashes."""
import json
import re
import sys

out_tsv, arms = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
COLS = ["arm", "env", "started", "request", "prompt_tokens", "cached_tokens", "prompt_tok_s", "first_token_s",
        "decode_tokens", "decode_tok_s", "mem_available_mib", "ident_md5"]
rows, by_arm, envs = [], {}, {}
for arm, path in arms:
    text = open(path).read()
    head = re.search(r"^== \S+: .*?\[env ([^\]]*)\] mode \S+ (\S+)", text, re.M)
    env = " ".join(head.group(1).split()) if head else ""
    started = head.group(2) if head else ""
    js = [l[5:] for l in text.splitlines() if l.startswith("JSON ")]
    if not js:
        sys.exit(f"{path}: no JSON line (the line did not finish)")
    reqs = json.loads(js[-1])
    by_arm[arm], envs[arm] = {r["name"]: r for r in reqs}, env
    for r in reqs:
        rows.append([arm, env, started, r["name"], r["prompt_n"], r["cache_n"], r["prompt_tps"], r["ttft_s"],
                     r["predicted_n"], r["decode_tps"], r["mem_available_mib"], r.get("md5", "")])
with open(out_tsv, "w") as f:
    f.write("\t".join(COLS) + "\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")

ref = arms[0][0]
names = [n for n in by_arm[ref] if not n.startswith("ident")]
print("| Request | Prompt tokens | " + " | ".join(f"decode {a}" for a, _ in arms) + " | "
      + " | ".join(f"prompt {a}" for a, _ in arms) + " |")
print("|---|---:|" + "---:|" * (2 * len(arms)))


def cell(arm, name, key):
    v = by_arm[arm].get(name, {}).get(key)
    if v is None:
        return "–"
    if arm == ref:
        return f"{v:.1f}"
    return f"{v:.1f} ({100 * (v / by_arm[ref][name][key] - 1):+.0f} %)"


for n in names:
    print(f"| {n} | {by_arm[ref][n]['prompt_n']:,} | " + " | ".join(cell(a, n, "decode_tps") for a, _ in arms) + " | "
          + " | ".join(cell(a, n, "prompt_tps") for a, _ in arms) + " |")
print()
for a, _ in arms:
    h = {n: r.get("md5", "") for n, r in by_arm[a].items() if n.startswith("ident")}
    print(f"ident {a} [{envs[a]}]: " + ", ".join(f"{n} {v}" for n, v in h.items()))
# prompt token counts must match, or the arms did not answer the same requests
for a, _ in arms[1:]:
    for n in by_arm[ref]:
        if by_arm[a].get(n, {}).get("prompt_n") != by_arm[ref][n]["prompt_n"]:
            print(f"WARNING {a} {n}: prompt tokens {by_arm[a].get(n, {}).get('prompt_n')} vs {by_arm[ref][n]['prompt_n']}")
