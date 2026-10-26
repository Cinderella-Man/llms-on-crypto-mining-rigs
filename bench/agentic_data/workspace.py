"""Agent workspaces built only from files a licence scan cleared (license_scan.py), each with a manifest that ties
every file to its origin, licence and hash.

Two kinds:
  stdlib   one workspace per pure-Python CPython module: stdmods/<module>.py copied verbatim from the pinned image's
           Lib, an empty stdmods/__init__.py and tests/__init__.py, and CPython's LICENSE. Only modules whose scan
           verdict is plain "ok" (no notice of their own) and that have no C accelerator replacing their Python
           code are taken. Tests come later, from the "write tests" sessions.
  tree     a copy of a source tree restricted to the scan's "ok" files (e.g. a package with its own test suite).

Manifest (<out>/<id>.manifest.json): workspace id, codebase id, the approved source id (sources.json), licence (SPDX),
origin (image digest or tree path and commit), the change summary (the PSF licence asks for one; Apache-2.0 for
notices on changed files), the test command, and every file with its role, origin path, sha256 and scan verdict.

usage: python workspace.py stdlib --modules fractions,shlex --scan SCAN.tsv --image IMAGE --codebase ID --out DIR
       python workspace.py tree TREE --scan SCAN.tsv --codebase ID --source-id SID --licence SPDX --origin TEXT
                                 --test-command 'python3 -m pytest -q' --out DIR [--include GLOB ...]
"""
import argparse
import fnmatch
import json
import os
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
    a = ap.parse_args()
    {"stdlib": cmd_stdlib, "tree": cmd_tree}[a.cmd](a)


if __name__ == "__main__":
    main()
