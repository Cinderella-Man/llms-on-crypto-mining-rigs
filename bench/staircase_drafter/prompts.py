"""Build the training prompt set for the staircase guesser (Q20): first user turns from licence-clean public data sets
(SETS: OpenR1-Math's crawled competition problems), plus whole-file edits and coding-agent turns built from local source
files,
as OpenAI-style messages (and tools) that llama-server's /apply-template renders.

Data sets (the owner's rulings and the data audit, 2026-10-25): only sources that can be published. EXCLUDED_SETS were
the first runs' mix (the EAGLE-3 plan's) and are not used any more: ShareGPT (scraped chats, unclear terms),
UltraChat and both Magicoder sets (written by ChatGPT/GPT-3.5), NuminaMath-CoT unfiltered (~45 % of rows written by
GPT-4: synthetic_math, synthetic_amc, orca_math) and Hermes function calling (LLM-written, generator not named).
NuminaMath-CoT may only come back with NUMINA_SOURCES (its crawled sources) and the owner's sign-off; maths as a whole
waits for that sign-off too. For the edit and agent prompts, source
trees must be ones whose licence allows it (the Python standard library, llama.cpp's own code, permissive packages);
clean_prompts.py builds code-first prompts on exactly those.

The edit task texts below are also Q51's (bench/expert_overlap/prompts.py), so prompts built here can share 12-word
spans with Q51's test prompts: drop those before training (the Q51 rule), as clean_prompts.py does by itself. The
data-set rows are other people's prompts, and the source files differ from Q51's.

usage: python prompts.py OUT.jsonl CACHE_DIR SRC_DIR [SRC_DIR ...] [--scale X] [--seed N] [--extend PROMPTS.jsonl]
  CACHE_DIR: where the data-set shards are downloaded (Hugging Face's parquet conversions, one shard per set)
  SRC_DIR:   source trees the edit and agent prompts take files from (e.g. a Python standard library, a llama.cpp tree)
  --extend: write only prompts not already in PROMPTS.jsonl (same messages and tools), numbered after its last id,
             so their answers and dumps join the existing ones (more data for the same pipeline)
needs: pip install huggingface_hub pyarrow
"""
import argparse
import json
import os
import random

SETS = [
    # (category, repo, config/split prefix in refs/convert/parquet, rows at scale 1)
    ("reason", "open-r1/OpenR1-Math-220k", "default/train/", 300),   # crawled competition problems
]
# NuminaMath-CoT, if the owner signs it off, only with rows whose `source` is a crawled one (synthetic_math,
# synthetic_amc and orca_math are GPT-4's); dataset_rows applies this filter to it
NUMINA_SOURCES = {"olympiads", "aops_forum", "amc_aime", "cn_k12", "math", "gsm8k"}
EXCLUDED_SETS = [   # the first runs' mix; not publishable (see the module docstring), kept for the record
    ("chat", "Aeala/ShareGPT_Vicuna_unfiltered", "default/train/", 1100),
    ("chat", "HuggingFaceH4/ultrachat_200k", "default/train_sft/", 900),
    ("code", "ise-uiuc/Magicoder-Evol-Instruct-110K", "default/train/", 900),
    ("code", "ise-uiuc/Magicoder-OSS-Instruct-75K", "default/train/", 500),
    ("tool", "NousResearch/hermes-function-calling-v1", "func_calling_singleturn/train/", 500),
    ("tool", "NousResearch/hermes-function-calling-v1", "glaive_func_calling/train/", 400),
    ("reason", "AI-MO/NuminaMath-CoT", "default/train/", 350),   # unfiltered
]
N_EDIT, N_AGENT = 1100, 1100

TOOL_SYSTEM = "You are a helpful assistant with access to tools. Call a tool when it helps answer the user."

EDITS_ANY = [
    "Add a short comment above each block that explains what it does. Output the complete updated file.",
    "Rename unclear variables and functions so their names say what they hold or do, and change nothing else. "
    "Output the complete updated file.",
    "Refactor the longest function into smaller functions without changing behaviour. Output the complete updated "
    "file.",
    "Add input validation with clear error messages wherever the code trusts its input. Output the complete updated "
    "file.",
    "Remove dead code and unused imports or includes, and change nothing else. Output the complete updated file.",
    "Make the error messages more specific and consistent. Output the complete updated file.",
    "Fix any bugs you find and mark each fix with a short comment. Output the complete updated file.",
    "Reformat the file to a consistent style (indentation, spacing, line length 100) without changing behaviour. "
    "Output the complete updated file.",
    "Add a header comment that explains the purpose of the file and how to use it. Output the complete updated file.",
    "Replace magic numbers and repeated literals with named constants. Output the complete updated file.",
]
EDITS_PY = [
    "Add type hints to every function and a docstring to each one. Output the complete updated file.",
    "Replace os.path with pathlib throughout and keep the behaviour the same. Output the complete updated file.",
    "Replace print calls with the logging module. Output the complete updated file.",
    "Convert string formatting to f-strings throughout. Output the complete updated file.",
    "Add a main() function with argparse if the file can be run as a script; otherwise add doctest examples to the "
    "public functions. Output the complete updated file.",
    "Use dataclasses where a class only holds data. Output the complete updated file.",
]

AGENT_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file from the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write a whole text file in the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command in the repository and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "search_code", "description": "Search the repository for a regular expression.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
]
AGENT_SYSTEMS = [
    "You are a coding agent working in a software repository. Use the tools to inspect and change files.",
    "You are an autonomous software engineer. Work through the task with the tools; read before you write.",
    "You are a careful coding assistant with repository access. Make the smallest correct change.",
]
AGENT_FIRST = [
    "The tests for {f} fail since the last commit. Find out why and fix it.",
    "Add a command-line flag to {f} that limits how many items it processes.",
    "Find every TODO in the repository and list them with file and line.",
    "Run the test suite and fix the first failing test.",
    "{f} is slow on large inputs. Profile it and make the hot path faster.",
    "Add a unit test for the main function in {f}.",
    "The build prints a warning about an unused variable in {f}. Fix it.",
    "Rename the most used helper function in {f} to a clearer name everywhere it is called.",
    "Check whether {f} handles empty input, and make it handle it gracefully.",
    "Search for uses of deprecated APIs in the repository and replace them in {f}.",
    "Document the public functions in {f}.",
    "Add a changelog entry describing the last change to {f}.",
    "There is a crash report pointing at {f}. Reproduce it with a small command and fix it.",
    "Make {f} read its settings from environment variables with sensible defaults.",
]
AGENT_SECOND = [
    "Review {f} for bugs and fix the most serious one with write_file.",
    "Add error handling to {f} and write the updated file.",
    "Add logging to the important steps in {f} and write the updated file.",
    "Simplify the most complicated function in {f} and write the updated file.",
    "Write a test file for {f} that covers its main behaviour.",
    "Explain what {f} does, then fix anything that looks wrong with write_file.",
    "Add comments to the non-obvious parts of {f} and write the updated file.",
    "Make {f} safer against bad input and write the updated file.",
]
SRC_EXT = {".py": "python", ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp", ".sh": "bash", ".js": "javascript",
           ".ts": "typescript", ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby", ".cu": "cuda",
           ".swift": "swift", ".kt": "kotlin", ".m": "objc"}


def shard(cache, repo, prefix):
    from huggingface_hub import hf_hub_download, list_repo_files
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset", revision="refs/convert/parquet")
                   if f.startswith(prefix) and f.endswith(".parquet"))
    return hf_hub_download(repo, files[0], repo_type="dataset", revision="refs/convert/parquet", cache_dir=cache)


def first_user(row):
    """(system, user) from the common chat layouts of the data sets."""
    for key in ("conversations", "messages", "conversation"):
        conv = row.get(key)
        if conv is None:
            continue
        system = None
        for m in conv:
            role = m.get("from", m.get("role"))
            text = m.get("value", m.get("content"))
            if role == "system":
                system = text
            elif role in ("human", "user"):
                return system, text
        return None, None
    for key in ("instruction", "problem", "prompt", "question"):
        if row.get(key):
            return None, row[key]
    return None, None


def dataset_rows(cat, repo, prefix, n, cache, rng):
    import pyarrow.parquet as pq
    table = pq.read_table(shard(cache, repo, prefix)).to_pylist()
    rng.shuffle(table)
    out = []
    for row in table:
        if len(out) >= n:
            break
        if repo == "AI-MO/NuminaMath-CoT" and row.get("source") not in NUMINA_SOURCES:
            continue
        system, user = first_user(row)
        if not user or len(user) > 12000 or len(user) < 8:
            continue
        r = {"category": cat, "source": repo, "messages": [{"role": "user", "content": user}]}
        if cat == "tool":
            try:
                tools = json.loads(row.get("tools") or "[]")
            except ValueError:
                continue
            if not tools:
                continue
            r["tools"] = [t if "function" in t else {"type": "function", "function": t} for t in tools]
            r["messages"].insert(0, {"role": "system", "content": TOOL_SYSTEM})
        out.append(r)
    return out


def source_files(dirs, lo=25, hi=180, max_chars=7000):
    files = []
    for d in dirs:
        for root, subdirs, names in os.walk(d):
            subdirs[:] = [s for s in subdirs if not s.startswith(".") and s not in ("test", "tests", "__pycache__",
                                                                                     "build", "node_modules")]
            for name in names:
                ext = os.path.splitext(name)[1]
                if ext not in SRC_EXT:
                    continue
                path = os.path.join(root, name)
                try:
                    text = open(path, encoding="utf-8").read()
                except (UnicodeDecodeError, OSError):
                    continue
                if lo <= text.count("\n") <= hi and len(text) <= max_chars and "\x00" not in text:
                    files.append((os.path.relpath(path, d), ext, text))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("cache")
    ap.add_argument("srcs", nargs="+")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--extend")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = []
    for cat, repo, prefix, n in SETS:
        got = dataset_rows(cat, repo, prefix, round(n * args.scale), args.cache, rng)
        print(f"{repo} {prefix}: {len(got)}")
        rows += got
    files = source_files(args.srcs)
    rng.shuffle(files)
    print(f"{len(files)} source files")
    for i in range(round(N_EDIT * args.scale)):
        name, ext, text = files[i % len(files)]
        tasks = EDITS_ANY + (EDITS_PY if ext == ".py" else [])
        task = rng.choice(tasks)
        rows.append({"category": "edit", "source": name, "messages": [
            {"role": "user", "content": f"Here is {os.path.basename(name)}:\n\n```{SRC_EXT[ext]}\n{text}```\n\n{task}"}]})
    for i in range(round(N_AGENT * args.scale)):
        name, ext, text = files[(i * 7 + 3) % len(files)]
        system = rng.choice(AGENT_SYSTEMS)
        if i % 2 == 0:
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": rng.choice(AGENT_FIRST).format(f=name)}]
        else:
            call = {"id": "call_0", "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": name})}}
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": rng.choice(AGENT_SECOND).format(f=name)},
                    {"role": "assistant", "content": "", "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": "call_0", "content": text}]
        rows.append({"category": "agent", "source": name, "tools": AGENT_TOOLS, "messages": msgs})
    rng.shuffle(rows)
    offset, seen = 0, set()
    if args.extend:
        old = [json.loads(l) for l in open(args.extend)]
        offset = 1 + max(int(r["id"].split("_")[-1]) for r in old)
        seen = {json.dumps([r["messages"], r.get("tools")], sort_keys=True) for r in old}
    kept = []
    for r in rows:
        key = json.dumps([r["messages"], r.get("tools")], sort_keys=True)
        if key not in seen:
            seen.add(key)
            kept.append(r)
    rows = kept
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            r["id"] = f"{r['category']}_{offset + i}"
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
