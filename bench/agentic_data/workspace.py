"""Agent workspaces built only from files a licence scan cleared (license_scan.py), each with a manifest that ties
every file to its origin, licence and hash.

Two kinds:
  stdlib   one workspace per pure-Python CPython module: stdmods/<module>.py copied verbatim from the pinned image's
           Lib, an empty stdmods/__init__.py and tests/__init__.py, and CPython's LICENSE. Only modules whose scan
           verdict is plain "ok" (no notice of their own) and that have no C accelerator replacing their Python
           code are taken. Tests come later, from the "write tests" sessions.
  tree     a copy of a source tree restricted to the scan's "ok" files (e.g. a package with its own test suite).
  elixir   one Mix project per Elixir standard-library module (a file of lib/elixir/lib, or another app's lib/):
           lib/<path>.ex is the scanned file with every module it defines moved into the Stdmods namespace (so it
           does not replace the loaded standard library, which Mix itself uses) and a first comment line saying so
           (texts.json's change note, written by Flint or the owner); the rename covers code (by Elixir's parser,
           sandbox/ex_tool.exs) and the doctests in its documentation, and nothing else. The project files come from
           Elixir's own `mix new stdmods --module Stdmods` in the sandbox: mix.exs, .formatter.exs and
           test/test_helper.exs unchanged, and its test/stdmods_test.exs with the module's name in the doctest line
           and the sample test left out. Plus Elixir's NOTICE and the Apache-2.0 text. Kept only if it compiles and
           its doctests pass in the Elixir sandbox.

Manifest (<out>/<id>.manifest.json): workspace id, codebase id, the approved source id (sources.json), licence (SPDX),
origin (image digest or tree path and commit), the change summary (the PSF licence asks for one; Apache-2.0 for
notices on changed files), the test command, and every file with its role, origin path, sha256 and scan verdict.

usage: python workspace.py stdlib --modules fractions,shlex --scan SCAN.tsv --image IMAGE --codebase ID --out DIR
       python workspace.py tree TREE --scan SCAN.tsv --codebase ID --source-id SID --licence SPDX --origin TEXT
                                 --test-command 'python3 -m pytest -q' --out DIR [--include GLOB ...]
       python workspace.py elixir TREE --modules option_parser,version|all --scan SCAN.tsv --notice NOTICE
                                 --licence-file APACHE-2.0.txt --out DIR [--codebase ID] [--image IMAGE]
"""
import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agx  # noqa: E402

ACCELERATED = {"heapq", "bisect", "json", "datetime", "decimal", "collections", "functools", "operator", "io",
               "pickle", "csv", "statistics", "zoneinfo", "struct", "array", "random", "base64", "quopri",
               "binascii", "codecs", "copyreg", "locale", "threading", "warnings"}   # C code replaces their Python
SKIP = {"this", "antigravity", "keyword", "token", "opcode", "sre_compile", "sre_constants", "sre_parse", "site",
        "pydoc", "typing", "stat", "struct", "socket", "ssl", "selectors", "signal", "subprocess", "multiprocessing",
        "os", "posixpath", "ntpath", "genericpath", "pty", "tty", "getpass", "webbrowser", "turtle", "imp", "nturl2path",
        "_pydecimal", "symtable", "tracemalloc", "cProfile", "pdb", "bdb", "code", "codeop", "runpy", "pkgutil",
        "zipimport", "importlib", "inspect", "dis", "ast"}   # interpreter internals, platform or process code


def scan_rows(path):
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        return {r[0]: dict(zip(head, r)) for r in (line.rstrip("\n").split("\t") for line in f)}


def write_manifest(out, ws_id, man):
    files = []
    root = os.path.join(out, ws_id)
    for rel, sha in sorted(agx.tree_files(root).items()):
        meta = man["origin_files"].get(rel, {"role": "scaffold", "origin": None, "verdict": "ours (empty)"})
        files.append({"path": rel, "sha256": sha, "bytes": os.path.getsize(os.path.join(root, rel)), **meta})
    man["files"] = files
    man.pop("origin_files")
    man["sha256"] = agx.sha256_text(json.dumps(files, sort_keys=True))
    with open(os.path.join(out, ws_id + ".manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    return man


def cmd_stdlib(a):
    scan = scan_rows(a.scan)
    os.makedirs(a.out, exist_ok=True)
    ctr = subprocess.run(["docker", "create", "--label", agx.LABEL, a.image], capture_output=True, text=True,
                         check=True).stdout.strip()
    lib = "/usr/local/lib/python3.12"
    try:
        lic = os.path.join(a.out, "_LICENSE.txt")
        subprocess.run(["docker", "cp", f"{ctr}:{lib}/LICENSE.txt", lic], check=True)
        if a.modules == "all":   # every single-file public module with a plain ok verdict, 2-120 KB
            mods = sorted(p[:-3] for p, r in scan.items() if "/" not in p and p.endswith(".py") and r["verdict"] == "ok"
                          and not p.startswith("_") and p[:-3] not in ACCELERATED | SKIP
                          and 2000 <= int(r["bytes"]) <= 120000)
        else:
            mods = [x.strip() for x in a.modules.split(",") if x.strip()]
        for m in mods:
            row = scan.get(f"{m}.py")
            if row is None or row["verdict"] != "ok":
                print(f"skip {m}: scan verdict {row and row['verdict']}", file=sys.stderr)
                continue
            if m in ACCELERATED:
                print(f"skip {m}: a C accelerator replaces its Python code", file=sys.stderr)
                continue
            ws = os.path.join(a.out, m)
            shutil.rmtree(ws, ignore_errors=True)
            os.makedirs(os.path.join(ws, "stdmods"))
            os.makedirs(os.path.join(ws, "tests"))
            subprocess.run(["docker", "cp", f"{ctr}:{lib}/{m}.py", os.path.join(ws, "stdmods", f"{m}.py")], check=True)
            if agx.sha256_file(os.path.join(ws, "stdmods", f"{m}.py")) != row["sha256"]:
                raise SystemExit(f"{m}.py differs from the scanned file")
            open(os.path.join(ws, "stdmods", "__init__.py"), "w").close()
            open(os.path.join(ws, "tests", "__init__.py"), "w").close()
            shutil.copy(lic, os.path.join(ws, "LICENSE"))
            man = {"workspace_id": f"{a.codebase}:{m}", "codebase_id": a.codebase, "source_id": "python-stdlib",
                   "licence": "Python-2.0", "origin": {"image": a.image, "lib": lib},
                   "changes": f"stdmods/{m}.py is Lib/{m}.py of CPython 3.12.15, unchanged, placed in the package "
                              f"folder stdmods/; stdmods/__init__.py and tests/__init__.py are empty files",
                   "test_command": ["python3", "-m", "unittest"], "module": m, "module_file": f"stdmods/{m}.py",
                   "origin_files": {f"stdmods/{m}.py": {"role": "code", "origin": f"{lib}/{m}.py",
                                                        "verdict": row["verdict"]},
                                    "LICENSE": {"role": "licence", "origin": f"{lib}/LICENSE.txt",
                                                "verdict": "licence text"}}}
            write_manifest(a.out, m, man)
            print(f"{m}: {row['bytes']} bytes")
    finally:
        subprocess.run(["docker", "rm", ctr], capture_output=True)


def cmd_tree(a):
    scan = scan_rows(a.scan)
    ws_id = a.codebase.replace("/", "_")
    ws = os.path.join(a.out, ws_id)
    shutil.rmtree(ws, ignore_errors=True)
    origin_files = {}
    for rel, row in scan.items():
        if not row["verdict"].startswith("ok"):
            continue
        if a.include and not any(fnmatch.fnmatchcase(rel, g) for g in a.include):
            continue
        src = os.path.join(a.tree, rel)
        if agx.sha256_file(src) != row["sha256"]:
            raise SystemExit(f"{rel} changed since the scan")
        dst = os.path.join(ws, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        origin_files[rel] = {"role": "licence" if row["verdict"] == "ok:licence-file" else "code", "origin": src,
                             "verdict": row["verdict"]}
    man = {"workspace_id": a.codebase, "codebase_id": a.codebase, "source_id": a.source_id, "licence": a.licence,
           "origin": {"tree": os.path.abspath(a.tree), "what": a.origin},
           "changes": "files whose licence scan verdict is not ok were left out; the rest are unchanged",
           "test_command": a.test_command.split(), "origin_files": origin_files}
    write_manifest(a.out, ws_id, man)
    print(f"{ws_id}: {len(origin_files)} files")


EX_SKIP = {"kernel", "kernel/special_forms", "kernel/utils", "kernel/typespec", "macro", "macro/env", "code",
           "code/formatter", "code/normalizer", "code/fragment", "code/identifier", "code/typespec", "module",
           "module/types", "protocol", "module/parallel_checker", "module/behaviour", "module/locals_tracker",
           "behaviour", "record/extractor", "exception", "inspect/algebra", "io", "file", "system", "port", "node",
           "process", "application", "agent/server", "task/supervised", "supervisor/default", "gen_server",
           "supervisor", "dynamic_supervisor", "registry", "partition_supervisor", "config/provider"}   # compiler,
# runtime and process internals: renamed copies of these clash with what Mix and ExUnit run on
EX_PROMPT = re.compile(r"^\s*(?:iex|\.\.\.)(?:\(\d+\))?>")


def ex_rename(src, info):
    """The source with the helper's alias positions prefixed by the Stdmods namespace, and the same module names
    renamed in doctest lines (prompts and their expected results); prose and string literals are left as they are.
    Returns (source, code references renamed, doctest lines renamed)."""
    edits = [(r["line"], r["column"], r["text"], f"{agx.EX_NAMESPACE}.{r['text']}") for r in info["refs"]]
    out, n = agx.apply_at(src, edits)
    names = sorted(info["modules"], key=len, reverse=True)
    pat = re.compile(r"(?<![\w.:])(" + "|".join(re.escape(x) for x in names) + r")(?!\w)")
    lines, in_test, d = out.split("\n"), False, 0
    for i, line in enumerate(lines):
        s = line.strip()
        if EX_PROMPT.match(line):
            in_test = True
        elif not s or s.startswith('"""') or s.startswith("'''"):
            in_test = False
        if in_test:
            new = pat.sub(rf"{agx.EX_NAMESPACE}.\1", line)
            d += new != line
            lines[i] = new
    return "\n".join(lines), n, d


def ex_mixnew(out, image):
    """mix new's project files, generated once in the sandbox into out/_mixnew."""
    d = os.path.join(out, "_mixnew")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    subprocess.run(["docker", "run", "--rm", "--label", agx.LABEL, "--network", "none", "--cpu-shares", "256",
                    "--read-only", "--tmpfs", "/tmp:rw,exec", "-v", f"{os.path.abspath(d)}:/work", image, "sh", "-c",
                    "cd /tmp && mix new stdmods --module Stdmods > /dev/null && cp -r stdmods/. /work/"], check=True)
    return {"dir": os.path.abspath(d), "command": "mix new stdmods --module Stdmods", "image": image,
            "files": {p: s for p, s in agx.tree_files(d).items() if p in agx.EX_MIXNEW + [agx.EX_DOCTEST]}}


def cmd_elixir(a):
    scan = scan_rows(a.scan)
    os.makedirs(a.out, exist_ok=True)
    texts = json.load(open(a.texts))
    notes = sorted((t for t in texts["templates"].get("change_note", []) if t.get("lang") == "elixir"),
                   key=lambda t: t["id"])
    if not notes:
        raise SystemExit(f"{a.texts} has no Elixir change note (agent.py texts with the spec's langs)")
    note = notes[0]
    mixnew = ex_mixnew(a.out, a.image)
    doctest_template = open(os.path.join(mixnew["dir"], agx.EX_DOCTEST), encoding="utf-8").read()
    lib = "lib/elixir/lib/"
    if a.modules == "all":   # every ok file of Elixir's own library, 2-120 KB, outside the internals
        rels = sorted(p for p, r in scan.items() if p.startswith(lib) and p.endswith(".ex") and r["verdict"] == "ok"
                      and p[len(lib):-3] not in EX_SKIP and 2000 <= int(r["bytes"]) <= 120000)
    else:   # option_parser, calendar/iso (Elixir's library) or a path below the tree (lib/eex/lib/eex/compiler.ex)
        rels = [m if m.startswith("lib/") else f"{lib}{m}.ex" for m in (x.strip() for x in a.modules.split(",")) if m]
    notice_sha, lic_sha = agx.sha256_file(a.notice), agx.sha256_file(a.licence_file)
    kept = 0
    for rel in rels:
        row = scan.get(rel)
        if row is None or row["verdict"] != "ok":
            print(f"skip {rel}: scan verdict {row and row['verdict']}", file=sys.stderr)
            continue
        sub = rel.split("/lib/", 2)[-1] if rel.startswith(lib) else rel.split("/lib/", 1)[1]
        m = sub[:-3].replace("/", "_")
        name = f"ex-{m}"   # the workspace's folder: Python's standard library has modules of the same names (string)
        ws = os.path.join(a.out, name)
        shutil.rmtree(ws, ignore_errors=True)
        mod_rel = f"lib/{sub}"
        os.makedirs(os.path.dirname(os.path.join(ws, mod_rel)), exist_ok=True)
        src_path = os.path.join(a.tree, rel)
        if agx.sha256_file(src_path) != row["sha256"]:
            raise SystemExit(f"{rel} differs from the scanned file")
        orig = open(src_path, encoding="utf-8").read()
        with open(os.path.join(ws, mod_rel), "w", encoding="utf-8") as f:
            f.write(orig)
        sb = agx.Sandbox(f"agx-ws-{name[:40]}", ws, a.image, "agx-ws",
                         config={"test_command": ["mix", "test"], "timeout": 300, "max_output": 200000}).start()
        try:
            info = sb.ex_tool("modules", mod_rel)
            if not info["modules"]:
                print(f"skip {rel}: defines no module at its top level", file=sys.stderr)
                shutil.rmtree(ws)
                continue
            new, n_ref, n_doc = ex_rename(orig, info)
            mods = [f"{agx.EX_NAMESPACE}.{x}" for x in info["modules"]]
            head = agx.ex_change_note(note["text"], f"Elixir 1.18.4 {rel.removeprefix('lib/')}", mods)
            with open(os.path.join(ws, mod_rel), "w", encoding="utf-8") as f:
                f.write(head + new)
            for p in agx.EX_MIXNEW:
                os.makedirs(os.path.dirname(os.path.join(ws, p)) or ws, exist_ok=True)
                shutil.copy(os.path.join(mixnew["dir"], p), os.path.join(ws, p))
            doctests = sum(1 for line in new.split("\n") if re.match(r"^\s*iex(?:\(\d+\))?>", line))
            dt_path = None
            if doctests:
                dt_path = agx.EX_DOCTEST
                with open(os.path.join(ws, dt_path), "w", encoding="utf-8") as f:
                    f.write(agx.ex_doctest_from(doctest_template, mods))
            shutil.copy(a.notice, os.path.join(ws, "NOTICE"))
            shutil.copy(a.licence_file, os.path.join(ws, "LICENSE"))
            before = agx.tree_files(ws)
            r = sb.tool("run_tests", {})
            run, fails = agx.ex_counts(r["output"])
            if agx.tree_files(ws) != before:
                raise SystemExit(f"{m}: mix test wrote into the workspace (MIX_BUILD_PATH not set in the image?)")
        finally:
            sb.stop()
        if r["exit_code"] != 0 or fails or run < a.min_doctests:
            why = "does not compile after the rename" if agx.EX_BROKEN.search(r["output"]) else \
                "mix test gave no summary (a crash or a timeout)" if fails is None else \
                f"{fails} of {run} doctests fail after the rename" if fails else f"{run} doctests"
            print(f"skip {rel}: {why}", file=sys.stderr)
            shutil.rmtree(ws)
            continue
        origin_files = {
            mod_rel: {"role": "code", "origin": src_path, "origin_sha256": row["sha256"], "verdict": row["verdict"],
                      "change": f"modules renamed into {agx.EX_NAMESPACE} ({n_ref} code references, {n_doc} doctest "
                                f"lines) and the change note as the first line"},
            "NOTICE": {"role": "notice", "origin": "Elixir's NOTICE at the v1.18.4 tag", "verdict": "notice text"},
            "LICENSE": {"role": "licence", "origin": "the Apache License 2.0 text", "verdict": "licence text"}}
        for p in agx.EX_MIXNEW:
            origin_files[p] = {"role": "scaffold", "origin": "mix new stdmods --module Stdmods (Elixir 1.18.4)",
                               "verdict": "generated by Elixir's Mix"}
        if dt_path:
            origin_files[dt_path] = {"role": "scaffold", "origin": "mix new's test file: doctest line naming the "
                                                                   "workspace's modules, sample test left out",
                                     "verdict": "generated by Elixir's Mix"}
        man = {"workspace_id": f"{a.codebase}:{m}", "codebase_id": a.codebase, "source_id": a.source_id,
               "licence": "Apache-2.0", "lang": "elixir", "image": a.image,
               "origin": {"tree": os.path.abspath(a.tree), "what": a.origin},
               "changes": f"{mod_rel} is Elixir 1.18.4's {rel.removeprefix('lib/')} with its modules renamed into "
                          f"the {agx.EX_NAMESPACE} namespace and a first comment line saying so; the project files "
                          f"are Elixir's mix new output{', its test file reduced to the doctest line' if dt_path else ''}",
               "test_command": ["mix", "test"], "name": name, "module": m, "module_file": mod_rel, "modules": mods,
               "doctest_file": dt_path, "doctests": run, "notice_sha256": notice_sha, "licence_sha256": lic_sha,
               "mixnew": mixnew, "change_note": {"id": note["id"], "texts_sha256": agx.sha256_file(a.texts),
                                                 "source_file": f"Elixir 1.18.4 {rel.removeprefix('lib/')}"},
               "origin_files": origin_files}
        write_manifest(a.out, name, man)
        kept += 1
        print(f"{name}: {row['bytes']} bytes, {n_ref} code references and {n_doc} doctest lines renamed, {run} doctests pass")
    print(f"{kept} of {len(rels)} Elixir workspaces kept")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stdlib")
    s.add_argument("--modules", required=True)
    s.add_argument("--scan", required=True)
    s.add_argument("--image", required=True)
    s.add_argument("--codebase", default="cpython-3.12.15-lib")
    s.add_argument("--out", required=True)
    t = sub.add_parser("tree")
    t.add_argument("tree")
    t.add_argument("--scan", required=True)
    t.add_argument("--codebase", required=True)
    t.add_argument("--source-id", required=True)
    t.add_argument("--licence", required=True)
    t.add_argument("--origin", required=True)
    t.add_argument("--test-command", required=True)
    t.add_argument("--include", nargs="*")
    t.add_argument("--out", required=True)
    e = sub.add_parser("elixir")
    e.add_argument("tree", help="Elixir's source tree (the image's /usr/local/src/elixir)")
    e.add_argument("--modules", required=True)
    e.add_argument("--scan", required=True)
    e.add_argument("--notice", required=True, help="Elixir's NOTICE file of the same version")
    e.add_argument("--licence-file", required=True, help="the Apache License 2.0 text")
    e.add_argument("--texts", required=True, help="texts.json with the Elixir change note")
    e.add_argument("--codebase", default="elixir-1.18.4-lib")
    e.add_argument("--source-id", default="elixir-stdlib")
    e.add_argument("--origin", default="/usr/local/src/elixir of elixir:1.18.4-otp-27@sha256:a7b2c4772e05616cf0a3323758ba5abc3288af452a33d1263705840a52154e85")
    e.add_argument("--image", default=agx.EX_IMAGE)
    e.add_argument("--min-doctests", type=int, default=1, help="keep modules with at least this many doctests "
                   "(compiler internals have none)")
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    {"stdlib": cmd_stdlib, "tree": cmd_tree, "elixir": cmd_elixir}[a.cmd](a)


if __name__ == "__main__":
    main()
