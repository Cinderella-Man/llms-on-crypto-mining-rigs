"""Agentic tasks that are clean by construction: every statement is a texts.json template (written by Flint or the
owner) filled with names from the workspace, and every workspace change is ours, recorded as a diff.

  write-tests   stage 1, per workspace: write a unittest file for the module (no tests exist yet). Verified after the
                session: the module unchanged, the test file present, all its tests pass.
  from-sessions stage 2, per stage-1 session whose test file passed: a derived workspace with that file (Flint's
                output, so publishable), and changes checked against it in the sandbox:
                  fix_bug    one small mutation of the module (comparison, arithmetic or boolean operator, a small
                             integer constant, True/False, a dropped `not`) that makes 1 to 30 % of the tests fail;
                  implement  one documented function's body replaced by `raise NotImplementedError` (docstring kept)
                             that makes some tests fail.
                Verified after the session: the test file unchanged, all tests pass.

Each task row carries its workspace (a ready folder with the change applied, and a manifest naming every file's
origin), the template id, the placeholders, the statement, the change and its validation.

usage: python tasks.py write-tests --workspaces WS_DIR --texts texts.json --out tasks-s1.jsonl [--seed 1] [--attempts 1]
       python tasks.py from-sessions --sessions SESS_DIR --workspaces WS_DIR --texts texts.json --out DIR
                       [--per-module 3] [--implement 1] [--max-candidates 60] [--seed 2] [--image IMAGE]
"""
import argparse
import ast
import difflib
import json
import os
import random
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agx  # noqa: E402

COMPARE = {ast.Lt: ("<", "<="), ast.LtE: ("<=", "<"), ast.Gt: (">", ">="), ast.GtE: (">=", ">"),
           ast.Eq: ("==", "!="), ast.NotEq: ("!=", "=="), ast.In: ("in", "not in"), ast.NotIn: ("not in", "in"),
           ast.Is: ("is", "is not"), ast.IsNot: ("is not", "is")}
BINOP = {ast.Add: ("+", "-"), ast.Sub: ("-", "+"), ast.Mult: ("*", "//"), ast.FloorDiv: ("//", "*")}
BOOLOP = {ast.And: ("and", "or"), ast.Or: ("or", "and")}


def fill(template, values):
    out = template["text"]
    for k, v in values.items():
        out = out.replace("{" + k + "}", v)
    return out


def pick_template(texts, kind, rng):
    t = rng.choice(texts["templates"][kind])
    return t, {"id": t["id"], "kind": kind, "texts_sha256": TEXTS_SHA}


def mutations(src):
    """[(function name, line, kind, new source)] for single-token mutations inside functions."""
    data = src.encode()
    starts = [0]
    for line in data.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))

    def off(lineno, col):
        return starts[lineno - 1] + col
    tree = ast.parse(src)
    found = {}

    def scope(fn, cls=None):
        name = f"{cls}.{fn.name}" if cls else fn.name
        doc = fn.body[0] if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(
            getattr(fn.body[0], "value", None), ast.Constant) else None
        for node in ast.walk(fn):
            if node is doc or (doc is not None and node is doc.value):
                continue
            edit = None
            if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in COMPARE:
                a, b = off(node.left.end_lineno, node.left.end_col_offset), off(node.comparators[0].lineno,
                                                                               node.comparators[0].col_offset)
                old, new = COMPARE[type(node.ops[0])]
                if data[a:b].decode().strip() == old:
                    edit = (a, b, data[a:b].decode().replace(old, new, 1), "comparison")
            elif isinstance(node, ast.BinOp) and type(node.op) in BINOP:
                a, b = off(node.left.end_lineno, node.left.end_col_offset), off(node.right.lineno, node.right.col_offset)
                old, new = BINOP[type(node.op)]
                if data[a:b].decode().strip() == old:
                    edit = (a, b, data[a:b].decode().replace(old, new, 1), "arithmetic")
            elif isinstance(node, ast.BoolOp) and type(node.op) in BOOLOP:
                a, b = off(node.values[0].end_lineno, node.values[0].end_col_offset), off(node.values[1].lineno,
                                                                                         node.values[1].col_offset)
                old, new = BOOLOP[type(node.op)]
                if data[a:b].decode().strip() == old:
                    edit = (a, b, data[a:b].decode().replace(old, new, 1), "boolean")
            elif isinstance(node, ast.Constant) and type(node.value) is bool:
                a, b = off(node.lineno, node.col_offset), off(node.end_lineno, node.end_col_offset)
                edit = (a, b, str(not node.value), "constant")
            elif isinstance(node, ast.Constant) and type(node.value) is int and node.value in (0, 1, 2):
                a, b = off(node.lineno, node.col_offset), off(node.end_lineno, node.end_col_offset)
                if data[a:b].decode() == str(node.value):
                    edit = (a, b, {0: "1", 1: "2", 2: "1"}[node.value], "constant")
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                a, b = off(node.lineno, node.col_offset), off(node.operand.lineno, node.operand.col_offset)
                if data[a:b].decode().strip() == "not":
                    edit = (a, b, "", "dropped not")
            if edit:
                found[(edit[0], edit[1])] = (name, node.lineno, edit)
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope(n)
        elif isinstance(n, ast.ClassDef):
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    scope(m, n.name)
    out = []
    for (a, b), (name, line, (_, _, new, kind)) in sorted(found.items()):
        s = (data[:a] + new.encode() + data[b:]).decode()
        try:
            ast.parse(s)
        except SyntaxError:
            continue
        out.append((name, line, kind, s))
    return out


def stubs(src):
    """[(function name, new source)] with a documented function's body after the docstring replaced by
    `raise NotImplementedError` (bodies of 3 or more statements)."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out = []

    def one(fn, cls=None):
        if len(fn.body) < 4 or not ast.get_docstring(fn):
            return
        first, last = fn.body[1].lineno, fn.end_lineno
        indent = re.match(r"\s*", lines[first - 1]).group(0)
        new = lines[:first - 1] + [indent + "raise NotImplementedError\n"] + lines[last:]
        s = "".join(new)
        try:
            ast.parse(s)
        except SyntaxError:
            return
        out.append((f"{cls}.{fn.name}" if cls else fn.name, s))
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_"):
            one(n)
        elif isinstance(n, ast.ClassDef) and not n.name.startswith("_"):
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith("_"):
                    one(m, n.name)
    return out


def cmd_write_tests(a):
    texts = json.load(open(a.texts))
    rng = random.Random(a.seed)
    rows = []
    for f in sorted(os.listdir(a.workspaces)):
        if not f.endswith(".manifest.json"):
            continue
        man = json.load(open(os.path.join(a.workspaces, f)))
        m = man["module"]
        for k in range(a.attempts):   # several attempts per module: other templates and seeds, other test files
            tmpl, tref = pick_template(texts, "write_tests", rng)
            ph = {"module_file": man["module_file"], "test_file": f"tests/test_{m}.py",
                  "test_command": " ".join(man["test_command"])}
            rows.append({
                "task_id": f"s1-{m}-wt" + (f"-a{k}" if a.attempts > 1 else ""), "type": "write_tests",
                "codebase_id": man["codebase_id"], "source_id": man["source_id"], "licence": man["licence"],
                "workspace_id": man["workspace_id"], "workspace_dir": os.path.abspath(os.path.join(a.workspaces, m)),
                "workspace_manifest": {"path": os.path.abspath(os.path.join(a.workspaces, f)), "sha256": man["sha256"]},
                "parent_session": None, "template": tref, "placeholders": ph, "statement": fill(tmpl, ph),
                "test_command": man["test_command"], "change": None,
                "verify": {"unchanged": [man["module_file"]], "exists": [ph["test_file"]], "target": "", "min_tests": 5}})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} write-tests tasks -> {a.out}")


FAIL_LINE = re.compile(r"^(?:FAIL|ERROR): (\w+) \(([\w.]+)\)", re.M)


def failing(output):
    return sorted({f"{cls}.{name}" if not cls.endswith(name) else cls for name, cls in FAIL_LINE.findall(output)})


def cmd_from_sessions(a):
    texts = json.load(open(a.texts))
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for f in sorted(os.listdir(a.sessions)):
        if not f.endswith(".json") or f.endswith(".error.json"):
            continue
        s = json.load(open(os.path.join(a.sessions, f)))
        t = s["task"]
        if t["type"] != "write_tests" or not s["verify"]["passed"]:
            continue
        man = json.load(open(t["workspace_manifest"]["path"]))
        m, mod_rel, test_rel = man["module"], man["module_file"], t["placeholders"]["test_file"]
        test_src = s["final_files"][test_rel]["content"]
        tag = s["session_id"].removeprefix(f"s1-{m}-wt").lstrip("-") or "a0"   # the stage-1 attempt
        base = os.path.join(a.out, f"base-{m}-{tag}")
        shutil.rmtree(base, ignore_errors=True)
        shutil.copytree(t["workspace_dir"], base)
        with open(os.path.join(base, test_rel), "w") as fh:
            fh.write(test_src)
        orig = open(os.path.join(base, mod_rel)).read()
        sb = agx.Sandbox(f"agx-tasks-{m}-{tag}", base, a.image, "agx-tasks",
                         config={"test_command": t["test_command"], "timeout": 120, "max_output": 200000}).start()
        try:
            r0 = sb.tool("run_tests", {})
            n_tests = int((re.search(r"Ran (\d+) tests?", r0["output"]) or [0, 0])[1])
            if r0["exit_code"] != 0 or n_tests < 5:
                print(f"{m}: the session's tests do not pass here ({r0['exit_code']}, {n_tests} tests)")
                continue
            cands = [("fix_bug", name, line, kind, src) for name, line, kind, src in mutations(orig)]
            rng.shuffle(cands)
            impl = [("implement", name, None, "stub", src) for name, src in stubs(orig)]
            rng.shuffle(impl)
            kept = {"fix_bug": [], "implement": []}
            want = {"fix_bug": a.per_module, "implement": a.implement}
            tried = 0
            for typ, name, line, kind, src in cands[:a.max_candidates] + impl[:a.max_candidates // 3]:
                if len(kept[typ]) >= want[typ] or any(k[1] == name for k in kept[typ]):
                    continue
                tried += 1
                with open(os.path.join(base, mod_rel), "w") as fh:
                    fh.write(src)
                r = sb.tool("run_tests", {})
                fails = failing(r["output"])
                cap = n_tests if typ == "implement" else max(1, int(0.3 * n_tests))   # a stub may fail them all
                if r["exit_code"] != 0 and 1 <= len(fails) <= cap and \
                        "SyntaxError" not in r["output"]:
                    kept[typ].append((typ, name, line, kind, src, fails, r["exit_code"]))
            with open(os.path.join(base, mod_rel), "w") as fh:
                fh.write(orig)
        finally:
            sb.stop()
        for typ in ("fix_bug", "implement"):
            for i, (_, name, line, kind, src, fails, code) in enumerate(kept[typ]):
                tid = f"{a.prefix}-{m}-{tag}-{typ}-{i}"
                ws = os.path.join(a.out, tid, "work")
                shutil.rmtree(os.path.dirname(ws), ignore_errors=True)
                shutil.copytree(base, ws)
                with open(os.path.join(ws, mod_rel), "w") as fh:
                    fh.write(src)
                diff = "".join(difflib.unified_diff(orig.splitlines(True), src.splitlines(True), "a/" + mod_rel,
                                                    "b/" + mod_rel, n=1))
                files = []
                for rel, sha in agx.tree_files(ws).items():
                    origin = {"role": "code", "origin": f"{man['workspace_id']} (CPython 3.12.15 Lib)"} \
                        if rel == mod_rel else {"role": "tests", "origin": f"session {s['session_id']} (Flint)",
                                                "writer": s["writer"]["id"]} if rel == test_rel else \
                        next(({"role": x["role"], "origin": x["origin"]} for x in man["files"] if x["path"] == rel),
                             {"role": "unknown", "origin": None})
                    files.append({"path": rel, "sha256": sha, **origin})
                tman = {"workspace_id": tid, "base": man["workspace_id"], "base_manifest_sha256": man["sha256"],
                        "codebase_id": man["codebase_id"], "source_id": man["source_id"], "licence": man["licence"],
                        "changes": man["changes"] + f"; {test_rel} written by Flint in session {s['session_id']}; "
                                   f"{mod_rel} changed by this pipeline for the task (diff in the task row)",
                        "files": files}
                tman["sha256"] = agx.sha256_text(json.dumps(files, sort_keys=True))
                mp = os.path.join(a.out, tid, "manifest.json")
                with open(mp, "w") as fh:
                    json.dump(tman, fh, indent=1)
                tmpl, tref = pick_template(texts, typ, rng)
                if typ == "fix_bug":
                    ph = {"test_file": test_rel, "failing_tests": ", ".join(fails[:5]), "module_file": mod_rel}
                else:
                    ph = {"function": name, "module_file": mod_rel, "test_file": test_rel}
                rows.append({"task_id": tid, "type": typ, "codebase_id": man["codebase_id"],
                             "source_id": man["source_id"], "licence": man["licence"], "workspace_id": tid,
                             "workspace_dir": os.path.abspath(ws),
                             "workspace_manifest": {"path": os.path.abspath(mp), "sha256": tman["sha256"]},
                             "parent_session": s["session_id"], "template": tref, "placeholders": ph,
                             "statement": fill(tmpl, ph), "test_command": t["test_command"],
                             "change": {"file": mod_rel, "function": name, "line": line, "kind": kind, "diff": diff},
                             "verify": {"unchanged": [test_rel], "exists": [], "target": "", "min_tests": n_tests},
                             "validation": {"tests": n_tests, "exit_before": 0, "exit_after": code,
                                            "failing": fails}})
        print(f"{m}: {n_tests} tests, {tried} changes tried, kept {len(kept['fix_bug'])} fix_bug and "
              f"{len(kept['implement'])} implement")
    with open(os.path.join(a.out, "tasks.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} tasks -> {os.path.join(a.out, 'tasks.jsonl')}")


TEXTS_SHA = None


def main():
    global TEXTS_SHA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-tests")
    w.add_argument("--workspaces", required=True)
    w.add_argument("--texts", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--seed", type=int, default=1)
    w.add_argument("--attempts", type=int, default=1, help="write-tests tasks per module (other templates and seeds)")
    s = sub.add_parser("from-sessions")
    s.add_argument("--sessions", required=True)
    s.add_argument("--workspaces", required=True)
    s.add_argument("--texts", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--per-module", type=int, default=3)
    s.add_argument("--implement", type=int, default=1)
    s.add_argument("--max-candidates", type=int, default=60)
    s.add_argument("--seed", type=int, default=2)
    s.add_argument("--image", default=agx.SANDBOX_IMAGE)
    s.add_argument("--prefix", default="s2", help="task id prefix (another round on the same sessions needs its own)")
    a = ap.parse_args()
    TEXTS_SHA = agx.sha256_file(a.texts)
    {"write-tests": cmd_write_tests, "from-sessions": cmd_from_sessions}[a.cmd](a)


if __name__ == "__main__":
    main()
