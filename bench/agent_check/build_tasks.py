"""Build the polyglot task set from Exercism's Python and JavaScript tracks.

    python3 build_tasks.py EXERCISM_PYTHON EXERCISM_JAVASCRIPT [--out tasks/polyglot] [--list polyglot_exercises.txt]

EXERCISM_PYTHON and EXERCISM_JAVASCRIPT are checkouts of github.com/exercism/python and github.com/exercism/javascript
(MIT) at the commits in tasks/polyglot/commits.json. The exercise list is Aider's polyglot selection for these two
tracks (34 Python, 49 JavaScript), minus excluded_exercises.txt: JavaScript's ledger is a refactoring exercise whose
untouched stub already passes every test, so the empty-change check cannot fail (82 tasks remain). Needs node and the shared node_modules of env/ (AC_NODE_MODULES) for the
JavaScript test splitter.

Per exercise, tasks/polyglot/<track>-<slug>/ gets:
- workspace/: what the agent sees: the stub, the visible tests (about a third of the cases: positions 0, 3, 6, ... of
  each test class or describe block, so every section shows its API and error format), helper and data files, and for
  JavaScript a package.json, babel.config.js and jest.config.js that run jest from a shared node_modules;
- hidden/: the full canonical test file (JavaScript: every skipped test switched on), which scores the attempt;
- reference/: Exercism's example solution under the stub's name, for validation only;
- prompt.md: the instructions (introduction, instructions, track appendix) and the task statement;
- task.json: metadata (source commit, files, test counts, test command).
"""
import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

JS_PACKAGE = {
    "name": "exercise",
    "private": True,
    "description": "Exercism exercise (MIT); tests run with jest from a shared node_modules.",
    "scripts": {"test": "jest"},
}
JS_BABEL = "module.exports = {\n  presets: [['@exercism/babel-preset-javascript', { corejs: '3.40' }]],\n  plugins: [],\n};\n"

PROMPT = """# {title}

{instructions}

---

Implement this exercise in `{stub}`, the stub in the current directory. Keep the names of the existing functions and
classes, since the tests import them. Use only the {lang} standard library; no packages can be installed.

`{test_file}` holds some of the tests; run them with `{test_cmd}`. A hidden test suite with more cases will be run on
your solution when you finish.
"""


def git_head(repo):
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def instructions(ex):
    parts = []
    for name in ("introduction.md", "instructions.md", "instructions.append.md"):
        p = ex / ".docs" / name
        if p.exists():
            parts.append(p.read_text().strip())
    return "\n\n".join(parts)


def title_of(slug):
    return " ".join(w.capitalize() for w in slug.split("-"))


def cut_python(src):
    """Keep the test methods at positions 0, 3, 6, ... of each class; return (visible source, total, visible)."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    drop = set()
    total = visible = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        tests = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name.startswith("test")]
        for j, fn in enumerate(tests):
            total += 1
            if j % 3 == 0:
                visible += 1
                continue
            # the method (with its decorators) and the blank lines in front of it; the blank lines after it stay
            # and separate the previous kept line from whatever follows
            first = min([fn.lineno] + [d.lineno for d in fn.decorator_list]) - 1
            while first > 0 and not lines[first - 1].strip():
                first -= 1
            drop.update(range(first, fn.end_lineno))
    text = "".join(l for i, l in enumerate(lines) if i not in drop)
    ast.parse(text)  # position 0 is always kept, so no class body ends up empty
    return text, total, visible


def build_python(ex, slug, dest):
    cfg = json.loads((ex / ".meta" / "config.json").read_text())["files"]
    stub, test_file = cfg["solution"][0], cfg["test"][0]
    ws, hidden, ref = dest / "workspace", dest / "hidden", dest / "reference"
    for d in (ws, hidden, ref):
        d.mkdir(parents=True)
    src = (ex / test_file).read_text()
    visible_src, total, visible = cut_python(src)
    (ws / test_file).write_text(visible_src)
    (hidden / test_file).write_text(src)
    shutil.copy(ex / stub, ws / stub)
    for extra in cfg["test"][1:]:  # helper modules the tests import (paasio's test_utils.py)
        shutil.copy(ex / extra, ws / extra)
    shutil.copy(ex / cfg["example"][0], ref / stub)
    return dict(stub=stub, test_file=test_file, helpers=cfg["test"][1:], total_tests=total, visible_tests=visible,
                test_cmd="pytest", lang="Python")


def build_javascript(ex, slug, dest):
    cfg = json.loads((ex / ".meta" / "config.json").read_text())["files"]
    stub, test_file = cfg["solution"][0], cfg["test"][0]
    ws, hidden, ref = dest / "workspace", dest / "hidden", dest / "reference"
    for d in (ws, hidden, ref):
        d.mkdir(parents=True)
    r = subprocess.run(["node", str(HERE / "cut_tests.mjs"), str(ex / test_file), str(ws / test_file),
                        str(hidden / test_file)], capture_output=True, text=True, check=True)
    counts = json.loads(r.stdout)
    shutil.copy(ex / stub, ws / stub)
    shutil.copy(ex / cfg["example"][0], ref / stub)
    skip = {stub, test_file, "package.json", "babel.config.js", "eslint.config.mjs", "LICENSE", ".npmrc",
            ".gitignore"}
    for p in sorted(ex.rglob("*")):
        rel = p.relative_to(ex)
        if p.is_file() and rel.parts[0] not in (".meta", ".docs") and str(rel) not in skip:
            (ws / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, ws / rel)  # jest.config.js, data files (grep)
    (ws / "package.json").write_text(json.dumps(JS_PACKAGE, indent=2) + "\n")
    (ws / "babel.config.js").write_text(JS_BABEL)
    return dict(stub=stub, test_file=test_file, helpers=[], total_tests=counts["total"],
                visible_tests=counts["visible"], test_cmd="npm test", lang="JavaScript")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("python_repo", type=Path)
    ap.add_argument("javascript_repo", type=Path)
    ap.add_argument("--out", type=Path, default=HERE / "tasks" / "polyglot")
    ap.add_argument("--list", type=Path, default=HERE / "polyglot_exercises.txt")
    ap.add_argument("--exclude", type=Path, default=HERE / "excluded_exercises.txt")
    a = ap.parse_args()
    if "AC_NODE_MODULES" not in os.environ:
        sys.exit("set AC_NODE_MODULES to the shared node_modules (see env/)")
    repos = {"python": a.python_repo, "javascript": a.javascript_repo}
    commits = {t: git_head(r) for t, r in repos.items()}
    for p in a.out.glob("*-*"):
        if p.is_dir():
            shutil.rmtree(p)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    excluded = set(a.exclude.read_text().split("\n"))
    for line in a.list.read_text().split("\n"):
        if not line.strip() or line in excluded:
            continue
        track, slug = line.split()
        ex = repos[track] / "exercises" / "practice" / slug
        dest = a.out / f"{track}-{slug}"
        meta = (build_python if track == "python" else build_javascript)(ex, slug, dest)
        meta.update(id=dest.name, track=track, slug=slug, family="polyglot",
                    source=dict(repo=f"https://github.com/exercism/{track}", commit=commits[track],
                                path=f"exercises/practice/{slug}"),
                    reference_lines=sum(1 for l in (dest / "reference" / meta["stub"]).read_text().splitlines()
                                        if l.strip()))
        prompt = PROMPT.format(title=title_of(slug), instructions=instructions(ex), stub=meta["stub"],
                               test_file=meta["test_file"], test_cmd=meta["test_cmd"], lang=meta.pop("lang"))
        (dest / "prompt.md").write_text(prompt)
        (dest / "task.json").write_text(json.dumps(meta, indent=2) + "\n")
        rows.append(meta)
        print(f"{dest.name:40s} tests {meta['visible_tests']:3d} / {meta['total_tests']:3d} visible")
    for t, r in repos.items():
        shutil.copy(r / "LICENSE", a.out / f"LICENSE.exercism-{t}")
    (a.out / "commits.json").write_text(json.dumps(commits, indent=2) + "\n")
    print(f"{len(rows)} tasks; commits {commits}")


if __name__ == "__main__":
    main()
