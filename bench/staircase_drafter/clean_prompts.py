"""Licence-clean prompts for the staircase guesser (Q20/M3), code first: every prompt comes from a source the owner
cleared (2026-10-25) and nothing overlaps Q51's test prompts.

- code: "implement this function": a documented Python function's signature and docstring (with its module's
  imports, and its class line for a method), from the standard library and the allowed packages, asked for in full;
- edit: whole-file edits (prompts.py's task texts) on files of the allowed trees;
- agent: coding-agent turns (prompts.py's tool schema and task texts) on the same files;
- reason: OpenR1-Math-220k default problems (crawled competition problems; the owner, 2026-10-25 14:40: maths kept at
  10-15 % of the clean set), with --n-math;
- chat: OpenAssistant/oasst2 root prompter messages (not synthetic, not deleted, passed review; all languages), with
  --n-chat.
Allowed trees: the Python standard library (--stdlib, without site-packages and test folders), llama.cpp without
vendor/ and build*/ folders (--llama), and the permissively licensed packages in site-packages (--site, ALLOWED_SITE
minus EXCLUDED_SITE). Files that Q51's prompts are built from are left out (--q51: {"source": name} lines), and so is
every prompt that shares a 12-word span with a Q51 prompt (--q51-prompts, Q51's prompts file).

usage: python clean_prompts.py OUT.jsonl CACHE --stdlib DIR --llama DIR --site DIR [--extend USED.jsonl]
                               [--n-code 12000] [--n-edit 6000] [--n-agent 20000] [--n-math 0] [--n-chat 0] [--seed 30]
                               [--q51 Q51_SOURCES.jsonl] [--q51-prompts Q51_PROMPTS.jsonl] [--id-offset 200000]
  --extend: prompts already used: no duplicates of them, ids numbered after them (else ids from --id-offset)
needs: pip install huggingface_hub pyarrow
"""
import argparse
import ast
import json
import os
import random

import prompts as P

ALLOWED_SITE = ["pyarrow", "huggingface_hub", "pip", "fsspec", "httpcore2", "anyio", "httpx2", "yaml", "_yaml",
                "packaging", "h11", "idna", "click", "filelock", "truststore"]   # MIT, BSD, Apache-2.0, PSF
EXCLUDED_SITE = ["pip/_vendor/certifi", "pip/_vendor/resolvelib", "pyarrow/include/arrow/vendored/portable-snippets",
                 "pyarrow/include/arrow/vendored/strptime.h"]                    # MPL-2.0, ISC, CC0, unclear
# vendored code is used only where it was judged one by one (anything else in these folders is skipped)
VENDORED_OK = {"pip/_vendor": {"rich", "pygments", "urllib3", "requests", "cachecontrol", "packaging", "truststore",
                               "msgpack", "idna", "platformdirs", "distlib", "distro", "tomli", "pyproject_hooks",
                               "__init__.py"},
               "pyarrow/include/arrow/vendored": {"double-conversion", "datetime", "datetime.h"}}
EDITS_EXTRA = [   # task texts beyond prompts.py's, so that the same files give new prompts
    "Split the file into clearly separated sections with a comment heading each one. Output the complete updated file.",
    "Make every function return early on invalid input instead of nesting conditions. Output the complete updated file.",
    "Add a small self-test at the bottom of the file that exercises its main functions. Output the complete updated "
    "file.",
    "Rewrite the comments so they explain why the code does what it does, not what it does. Output the complete "
    "updated file.",
    "Find the function with the most branches and simplify its control flow. Output the complete updated file.",
    "Make all public names follow one naming convention and update every use. Output the complete updated file.",
    "Add timing or counting instrumentation around the expensive operations, off by default. Output the complete "
    "updated file.",
    "Make the code safe to call from several threads, with the smallest change. Output the complete updated file.",
]
CODE_TASKS = [
    "Implement this Python function. Keep its signature and docstring, and return the complete function.",
    "Write the body of `{name}` so that it does what its docstring says. Output the whole function.",
    "Implement `{name}` below. Then explain in two sentences how your implementation handles edge cases.",
    "Here is a function stub from a Python module. Implement it, keeping the code style of the standard library.",
    "Implement `{name}`, and add three doctest examples to its docstring that show how it behaves.",
]


def site_skip(site, pkg):
    """Paths under site/pkg (relative to it) never used: EXCLUDED_SITE and every vendored entry not in VENDORED_OK."""
    skip = [os.path.relpath(x, pkg) for x in EXCLUDED_SITE if x.startswith(pkg + "/")]
    for folder, ok in VENDORED_OK.items():
        if folder.startswith(pkg + "/") and os.path.isdir(os.path.join(site, folder)):
            skip += [os.path.relpath(os.path.join(folder, x), pkg) for x in os.listdir(os.path.join(site, folder))
                     if x not in ok]
    return skip


def walk(root, rel_prefix, skip_dir, skip_path=()):
    """(source name, ext, text) of the usable files under root (prompts.source_files' size limits)."""
    out = []
    for dirpath, subdirs, names in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        subdirs[:] = [s for s in subdirs if not s.startswith(".") and not skip_dir(s)
                      and not any(os.path.normpath(os.path.join(rel_dir, s)).startswith(x) for x in skip_path)]
        for name in names:
            ext = os.path.splitext(name)[1]
            rel = os.path.normpath(os.path.join(rel_dir, name))
            if ext not in P.SRC_EXT or any(rel.startswith(x) for x in skip_path):
                continue
            try:
                text = open(os.path.join(dirpath, name), encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            if 25 <= text.count("\n") <= 180 and len(text) <= 7000 and "\x00" not in text:
                out.append((os.path.join(rel_prefix, rel) if rel_prefix else rel, ext, text))
    return out


def stub_prompts(root, rng, banned, rel_prefix="", skip_path=()):
    """(source, prompt text) for documented Python functions with a 3-100-line body under root."""
    out = []
    for dirpath, subdirs, names in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        subdirs[:] = [s for s in subdirs if s not in ("site-packages", "test", "tests", "idle_test", "__pycache__")
                      and not s.startswith((".", "config-"))
                      and not any(os.path.normpath(os.path.join(rel_dir, s)).startswith(x) for x in skip_path)]
        for name in names:
            if not name.endswith(".py"):
                continue
            rel = os.path.normpath(os.path.join(rel_dir, name))
            if os.path.basename(rel) in banned or any(rel.startswith(x) for x in skip_path):
                continue
            rel = os.path.join(rel_prefix, rel) if rel_prefix else rel
            try:
                src = open(os.path.join(dirpath, name), encoding="utf-8").read()
                tree = ast.parse(src)
            except (UnicodeDecodeError, OSError, SyntaxError, ValueError):
                continue
            lines = src.splitlines()
            imports = [ast.get_source_segment(src, n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
            imports = "\n".join(x for x in imports if x)[:800]
            targets = [(None, n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for c in (n for n in tree.body if isinstance(n, ast.ClassDef)):
                targets += [(c.name, n) for n in c.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for cls, fn in targets:
                doc = ast.get_docstring(fn)
                if not doc or len(doc) < 30 or len(fn.body) < 2:
                    continue
                body_lines = fn.end_lineno - fn.body[1].lineno + 1
                if not 3 <= body_lines <= 100:
                    continue
                sig = "\n".join(lines[fn.lineno - 1:fn.body[0].lineno - 1]).rstrip()
                indent = " " * (fn.col_offset + 4)
                quoted = "\n".join(indent + x if x else "" for x in ('"""' + doc + '"""').splitlines())
                stub = (f"class {cls}:\n    ...\n\n" if cls else "") + f"{sig}\n{quoted}\n"
                task = rng.choice(CODE_TASKS).format(name=fn.name)
                head = f"{imports}\n\n" if imports else ""
                out.append((rel, f"{task}\n\n```python\n{head}{stub}```"))
    return out


def oasst2_rows(cache, rng, n):
    """Chat prompts from OpenAssistant/oasst2 (Apache-2.0): root prompter messages (parent_id null) that are not
    synthetic, not deleted and passed review; all languages, the English share printed."""
    import pyarrow.parquet as pq
    table = pq.read_table(P.shard(cache, "OpenAssistant/oasst2", "default/train/")).to_pylist()
    ok = [r for r in table if r.get("parent_id") is None and r.get("role") == "prompter" and r.get("synthetic") is False
          and r.get("deleted") is False and r.get("review_result") is True and 8 <= len(r.get("text") or "") <= 12000]
    rng.shuffle(ok)
    take = ok[:n]
    en = sum(r.get("lang") == "en" for r in take)
    print(f"oasst2: {len(ok)} eligible root prompts ({sum(r.get('lang') == 'en' for r in ok) / max(len(ok), 1):.1%} "
          f"English), {len(take)} taken ({en / max(len(take), 1):.1%} English)")
    return [{"category": "chat", "source": "OpenAssistant/oasst2", "lang": r.get("lang"),
             "messages": [{"role": "user", "content": r["text"]}]} for r in take]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("cache")
    ap.add_argument("--stdlib", required=True)
    ap.add_argument("--llama", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--extend", help="earlier prompts: no duplicates of them, ids numbered after them")
    ap.add_argument("--id-offset", type=int, default=200000, help="first id without --extend")
    ap.add_argument("--q51")
    ap.add_argument("--q51-prompts")
    ap.add_argument("--n-code", type=int, default=12000)
    ap.add_argument("--n-edit", type=int, default=6000)
    ap.add_argument("--n-agent", type=int, default=20000)
    ap.add_argument("--n-math", type=int, default=0)
    ap.add_argument("--n-chat", type=int, default=0)
    ap.add_argument("--seed", type=int, default=30)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    banned = set()
    if args.q51:
        banned = {os.path.basename(json.loads(x).get("source", "")) for x in open(args.q51)} - {""}
    files = walk(args.stdlib, "", lambda s: s in ("site-packages", "test", "tests", "__pycache__")
                 or s.startswith("config-"))
    files += walk(args.llama, "", lambda s: s.startswith(("build", "vendor")) or s in ("test", "tests", "node_modules"))
    for pkg in ALLOWED_SITE:
        root = os.path.join(args.site, pkg)
        if os.path.isdir(root):
            skip = site_skip(args.site, pkg)
            files += walk(root, os.path.join("site-packages", pkg), lambda s: s in ("tests", "test", "__pycache__"), skip)
    files = [f for f in files if os.path.basename(f[0]) not in banned]
    rng.shuffle(files)
    stubs = stub_prompts(args.stdlib, rng, banned)
    for pkg in ALLOWED_SITE:
        root = os.path.join(args.site, pkg)
        if os.path.isdir(root):
            skip = site_skip(args.site, pkg)
            stubs += stub_prompts(root, rng, banned, os.path.join("site-packages", pkg), skip)
    rng.shuffle(stubs)
    print(f"{len(files)} source files, {len(stubs)} documented functions")
    spans = set()   # the Q51 rule: no 12-word span shared with a Q51 prompt (lower-cased, split on whitespace)
    if args.q51_prompts:
        for line in open(args.q51_prompts):
            w = " ".join((m.get("content") or "") for m in json.loads(line)["messages"]).lower().split()
            spans |= {" ".join(w[i:i + 12]) for i in range(len(w) - 11)}

    def clean_texts(texts):   # task texts that share 12 words with Q51 by themselves are not used at all
        out = []
        for t in texts:
            w = t.lower().split()
            if not any(" ".join(w[i:i + 12]) in spans for i in range(len(w) - 11)):
                out.append(t)
        return out
    edits_any, edits_py = clean_texts(P.EDITS_ANY + EDITS_EXTRA), clean_texts(P.EDITS_PY)
    print(f"edit task texts usable: {len(edits_any)} of {len(P.EDITS_ANY) + len(EDITS_EXTRA)} (any), "
          f"{len(edits_py)} of {len(P.EDITS_PY)} (Python)")
    rows = [{"category": "code", "source": src, "messages": [{"role": "user", "content": text}]}
            for src, text in stubs[:args.n_code]]
    for i in range(args.n_edit):
        name, ext, text = files[i % len(files)]
        task = rng.choice(edits_any + (edits_py if ext == ".py" else []))
        rows.append({"category": "edit", "source": name, "messages": [
            {"role": "user", "content": f"Here is {os.path.basename(name)}:\n\n```{P.SRC_EXT[ext]}\n{text}```\n\n{task}"}]})
    for i in range(args.n_agent):
        name, ext, text = files[(i * 7 + 3) % len(files)]
        system = rng.choice(P.AGENT_SYSTEMS)
        if i % 2 == 0:
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": rng.choice(P.AGENT_FIRST).format(f=name)}]
        else:
            call = {"id": "call_0", "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": name})}}
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": rng.choice(P.AGENT_SECOND).format(f=name)},
                    {"role": "assistant", "content": "", "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": "call_0", "content": text}]
        rows.append({"category": "agent", "source": name, "tools": P.AGENT_TOOLS, "messages": msgs})
    if args.n_math:
        rows += P.dataset_rows("reason", "open-r1/OpenR1-Math-220k", "default/train/", args.n_math + 2000, args.cache,
                               rng)
    if args.n_chat:
        rows += oasst2_rows(args.cache, rng, args.n_chat)
    offset, seen = args.id_offset, set()
    if args.extend:
        old = [json.loads(x) for x in open(args.extend)]
        offset = 1 + max(int(r["id"].split("_")[-1]) for r in old)
        seen = {json.dumps([r["messages"], r.get("tools")], sort_keys=True) for r in old}
    kept, n_math, n_q51 = [], 0, 0
    rng.shuffle(rows)
    for r in rows:
        key = json.dumps([r["messages"], r.get("tools")], sort_keys=True)
        if key in seen or (r["category"] == "reason" and n_math >= args.n_math):
            continue
        w = " ".join((m.get("content") or "") for m in r["messages"]).lower().split()
        if spans and any(" ".join(w[i:i + 12]) in spans for i in range(len(w) - 11)):
            n_q51 += 1
            continue
        seen.add(key)
        n_math += r["category"] == "reason"
        kept.append(r)
    with open(args.out, "w") as f:
        for i, r in enumerate(kept):
            r["id"] = f"{r['category']}_{offset + i}"
            f.write(json.dumps(r) + "\n")
    counts = {c: sum(r["category"] == c for r in kept) for c in ("code", "edit", "agent", "reason", "chat")}
    print(f"{len(kept)} prompts -> {args.out}: {counts}; {n_q51} left out for sharing 12 words with Q51")


if __name__ == "__main__":
    main()
