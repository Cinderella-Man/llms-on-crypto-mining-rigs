"""Build the bugfix task set: small bug fixes in working code, from the polyglot tasks.

    python3 build_bugfix.py [--bugs bugfix_bugs.json] [--out tasks/bugfix]

Each entry of bugfix_bugs.json names a polyglot task and one small change to its reference solution (Exercism's
example solution) that breaks it: a lost pair of parentheses, an off-by-one, a wrong constant, a missing check. The
agent gets the broken file and the task's full test suite, and has to find and fix the bug. A typical fix takes 5 to 10
tool calls: run the tests, read the code, edit, run the tests again.

Per entry, tasks/bugfix/bugfix-<polyglot id>/ gets:
- workspace/: the polyglot workspace with the stub replaced by the broken reference solution and the visible test file
  replaced by the full one (hidden/), so the failing tests show the bug;
- hidden/ and reference/: as in the polyglot task (the attempt is scored by the full test file, unchanged);
- prompt.md: the polyglot task's instructions (introduction, instructions, track appendix) and the bug-fix statement;
- task.json: the polyglot metadata, family "bugfix", the source task and the injected change.

order.txt keeps the entries' order (the tracks alternate). validate.py --task-set bugfix checks that the reference
passes, the broken workspace fails, and the visible test file is the full one.
"""
import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLYGLOT = HERE / "tasks" / "polyglot"

STATEMENT = """
---

`{stub}` in the current directory implements this exercise, but it has a bug: some of the tests in `{test_file}` fail.
Run them with `{test_cmd}`, find the bug and fix it in `{stub}`. Keep the fix small, keep the names of the existing
functions and classes, and don't change the tests. Use only the {lang} standard library; no packages can be installed.
`{test_file}` will be run on your fix when you finish.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bugs", type=Path, default=HERE / "bugfix_bugs.json")
    ap.add_argument("--out", type=Path, default=HERE / "tasks" / "bugfix")
    a = ap.parse_args()
    bugs = json.loads(a.bugs.read_text())
    for p in a.out.glob("bugfix-*"):
        if p.is_dir():
            shutil.rmtree(p)
    a.out.mkdir(parents=True, exist_ok=True)
    order = []
    for b in bugs:
        src = POLYGLOT / b["task"]
        meta = json.loads((src / "task.json").read_text())
        stub, tf = meta["stub"], meta["test_file"]
        ref = (src / "reference" / stub).read_text()
        if ref.count(b["old"]) != 1:
            raise SystemExit(f"{b['task']}: the text to change occurs {ref.count(b['old'])} times in the reference")
        dest = a.out / f"bugfix-{b['task']}"
        shutil.copytree(src / "workspace", dest / "workspace")
        shutil.copytree(src / "hidden", dest / "hidden")
        shutil.copytree(src / "reference", dest / "reference")
        (dest / "workspace" / stub).write_text(ref.replace(b["old"], b["new"]))
        shutil.copy(src / "hidden" / tf, dest / "workspace" / tf)
        prompt = (src / "prompt.md").read_text()
        head = prompt[: prompt.rindex("\n---\n")]
        lang = "Python" if meta["track"] == "python" else "JavaScript"
        (dest / "prompt.md").write_text(head + STATEMENT.format(stub=stub, test_file=tf, test_cmd=meta["test_cmd"],
                                                                lang=lang))
        meta.update(id=dest.name, family="bugfix", source_task=b["task"], visible_tests=meta["total_tests"],
                    bug=dict(what=b["what"], old=b["old"], new=b["new"]))
        (dest / "task.json").write_text(json.dumps(meta, indent=2) + "\n")
        order.append(dest.name)
        print(f"{dest.name:48s} {b['what']}")
    for p in POLYGLOT.glob("LICENSE.exercism-*"):
        shutil.copy(p, a.out / p.name)
    shutil.copy(POLYGLOT / "commits.json", a.out / "commits.json")
    (a.out / "order.txt").write_text("\n".join(order) + "\n")
    print(f"{len(order)} tasks in {a.out}")


if __name__ == "__main__":
    main()
