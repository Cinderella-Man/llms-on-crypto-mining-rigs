"""Q20/M3 (2026-10-25): which drafter answers are licence-clean, from the licence rulings and the data audit:
- clean by construction (the core): answers to my own edit, agent and "implement this function" prompts built from
  the Python standard library, llama.cpp's own code (not vendor/ or build folders) and the site-packages files the
  ruling allows (MIT, BSD, Apache-2.0, PSF; strict_sitepkgs.py's per-package decisions);
- approved: OpenR1-Math-220k default (crawled competition problems; maths kept at 10-15 %) and
  OpenAssistant/oasst2 root prompter messages (chat; synthetic, deleted and unreviewed ones left out);
- excluded: ShareGPT, UltraChat, both Magicoder sets, NuminaMath-CoT (~45 % GPT-4-written rows) and ToolACE,
  glaive-function-calling-v2, Hermes function calling (LLM-written), llama.cpp vendor/ and build folders, copyleft and
  other site-packages, and every prompt that shares a 12-word span with a Q51 test prompt (q51-overlap-ids.txt from
  q51_overlap.py, so that Q51's scores stay honest).
Writes WORK_DIR/strict-ids-core.txt (the core), strict-ids.txt (core + OpenR1 + oasst2), and prints
answers and answer tokens per group and writer, with how many are dumped.

usage: python3 -I count_strict.py WORK_DIR
  WORK_DIR holds the drafter's prompt files (prompts*.jsonl), answer files (gen-<label>.jsonl), the dump maps
  (map-d*.jsonl, dumpD6*.log) and q51-overlap-ids.txt.
"""
import collections
import glob
import json
import sys

W = sys.argv[1]
STDLIB = set(sys.stdlib_module_names) | {"idlelib", "lib2to3", "turtledemo", "ensurepip", "test"}
EXCLUDED_HF = {"Aeala/ShareGPT_Vicuna_unfiltered", "HuggingFaceH4/ultrachat_200k",
               "ise-uiuc/Magicoder-Evol-Instruct-110K", "ise-uiuc/Magicoder-OSS-Instruct-75K", "AI-MO/NuminaMath-CoT",
               "Team-ACE/ToolACE", "glaiveai/glaive-function-calling-v2", "NousResearch/hermes-function-calling-v1"}
OASST2 = "OpenAssistant/oasst2"
OPENR1 = "open-r1/OpenR1-Math-220k"
SITE_OK = {"pyarrow", "huggingface_hub", "pip", "fsspec", "httpcore2", "anyio", "httpx2", "yaml", "_yaml",
           "packaging", "h11", "idna", "click", "filelock", "truststore"}
SITE_OUT = ("pip/_vendor/certifi", "pip/_vendor/resolvelib", "pyarrow/include/arrow/vendored/portable-snippets",
            "pyarrow/include/arrow/vendored/strptime.h")


def group(cat, src):
    # data-set sources first: the old "code" category came from Magicoder, a data set, not from source files
    if src == OPENR1:
        return "openr1"
    if src == OASST2:
        return "oasst2"
    if src in EXCLUDED_HF or cat not in ("edit", "agent", "code"):
        return "excluded:" + src
    top = src.split("/")[0]
    if top == "site-packages":
        rest = src.split("/", 1)[1]
        ok = rest.split("/")[0] in SITE_OK and not rest.startswith(SITE_OUT)
        return "core:site-packages" if ok else "excluded:site-packages"
    if top.startswith("config-3.") or top.removesuffix(".py") in STDLIB:
        return "core:stdlib"
    if top == "vendor" or top.startswith("build"):
        return "excluded:llama.cpp-" + ("vendor" if top == "vendor" else "build")
    return "core:llama.cpp"


overlap = {x.strip() for x in open(f"{W}/q51-overlap-ids.txt") if x.strip()}
prompts = {}
for f in glob.glob(f"{W}/prompts*.jsonl"):
    if any(x in f for x in ("allused", "all12", "all123", ".v1.")):
        continue
    for line in open(f):
        r = json.loads(line)
        g = group(r["category"], r.get("source", ""))
        if r["id"] in overlap:
            g = "excluded:q51-overlap"
        prompts[r["id"]] = (r["category"], g)
dumped = set()
for m in ("map-d1.jsonl", "map-d2.jsonl", "map-d3.jsonl", "map-d4.done.jsonl", "map-d5.done.jsonl"):
    dumped |= {json.loads(line)["id"] for line in open(f"{W}/{m}")}
for x in "abc":
    try:
        sid2id = {json.loads(line)["seq_id"]: json.loads(line)["id"] for line in open(f"{W}/map-d6{x}.jsonl")}
        for line in open(f"{W}/dumpD6{x}.log", errors="replace"):
            if line.startswith("sequence ") and int(line.split()[1].rstrip(":")) in sid2id:
                dumped.add(sid2id[int(line.split()[1].rstrip(":"))])
    except FileNotFoundError:
        pass
writers = {"Q8_0": "A B B2 B3 C C2 C2s", "G3": "E1 E2 F1 F2", "UD-Q4_K_XL": "FV3", "NVFP4 vLLM": "V1 V2"}
stat = collections.Counter()
lists = collections.defaultdict(list)
for w, names in writers.items():
    for nm in names.split():
        try:
            fh = open(f"{W}/gen-{nm}.jsonl")
        except FileNotFoundError:
            continue
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            cat, g = prompts.get(r["id"], (r.get("category", "?"), "unknown"))
            k = len(r["gen_tokens"])
            kind = g.split(":")[0]
            stat[(g, "n")] += 1
            stat[(g, "t")] += k
            stat[(kind, w, "n")] += 1
            stat[(kind, w, "t")] += k
            if r["id"] in dumped:
                stat[(kind, "dn")] += 1
                stat[(kind, "dt")] += k
            stat[(kind, cat, "t")] += k
            lists[kind].append(r["id"])
for name, ids in (("strict-ids-core.txt", lists["core"]),
                  ("strict-ids.txt", lists["core"] + lists["openr1"] + lists["oasst2"])):
    with open(f"{W}/{name}", "w") as f:
        f.write("\n".join(ids) + "\n")
print("answers / M answer tokens by source group:")
for g in sorted({k[0] for k in stat if len(k) == 2 and k[1] in ("n", "t")}, key=lambda g: -stat[(g, "t")]):
    print(f"  {g}: {stat[(g, 'n')]} / {stat[(g, 't')] / 1e6:.2f}")
for kind in ("core", "openr1", "oasst2"):
    n = sum(stat[(kind, w, "n")] for w in writers)
    t = sum(stat[(kind, w, "t")] for w in writers)
    per_w = ", ".join(f"{w} {stat[(kind, w, 't')] / 1e6:.2f}" for w in writers)
    per_c = ", ".join(f"{c} {stat[(kind, c, 't')] / 1e6:.2f}" for c in ("edit", "agent", "code", "reason", "chat")
                      if stat[(kind, c, "t")])
    print(f"{kind}: {n} answers / {t / 1e6:.2f} M ({per_w}; {per_c}); dumped {stat[(kind, 'dn')]} / "
          f"{stat[(kind, 'dt')] / 1e6:.2f} M")
