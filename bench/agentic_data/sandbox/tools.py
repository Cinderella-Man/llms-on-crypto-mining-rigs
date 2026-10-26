"""The agent's tools, run inside the session's sandbox container (one call per `docker exec`).

Reads one JSON request on stdin, {"tool": NAME, "args": {...}, "config": {...}}, runs it with /work as the project
root and writes one JSON reply on stdout: {"ok", "output", "exit_code", "files_read", "files_written",
"truncated"}. Only paths under /work (and /tmp for scratch) are reachable; the container has no network, a read-only
root and a private /tmp. Standard library only, so it runs in the bare python image.

Tools (the schema the model sees is built from texts.json; the names and parameters are fixed here):
  read_file(path, start_line=None, end_line=None)   list_dir(path=".")
  write_file(path, content)                         search(pattern, path=".")
  edit_file(path, old_string, new_string)           run_tests(target="")
  run_command(command)
The pi profile (the pi agent's tool names and parameters): read(path, offset, limit), write(path, content),
  edit(path, edits=[{oldText, newText}, ...]), bash(command, timeout).
Config: test_command (argv list), timeout (s), max_output (characters), max_lines (read_file), messages (the tool's
own lines by id, with {placeholders}: a session's requests carry texts.json's, written by Flint; DEFAULT_MESSAGES
serve the pipeline's own checks, whose output never enters a dataset).
"""
import json
import os
import re
import subprocess
import sys

ROOT = "/work"
SCRATCH = "/tmp"
DEFAULT_MESSAGES = {
    "path_empty": "path is empty", "path_outside": "{path} is outside the project folder {root}",
    "path_missing": "{path} does not exist", "is_folder": "{path} is a folder", "not_folder": "{path} is not a folder",
    "content_type": "content must be a string",
    "edit_args": "old_string and new_string must be strings, old_string not empty",
    "edit_count": "old_string occurs {n} times in {path}; it must occur exactly once",
    "edits_shape": "edits must be a non-empty list of oldText/newText objects",
    "edit_i_args": "edit {i}: oldText and newText must be strings, oldText not empty",
    "edit_i_count": "edit {i}: oldText occurs {n} times in {path}; it must occur exactly once",
    "bad_regex": "bad regular expression: {detail}", "command_empty": "command is empty",
    "blocked": "only paths inside {root} (and {scratch}) can be used", "unknown_tool": "unknown tool {name}",
    "no_param": "{tool} has no parameter {param}", "not_integer": "{param} must be an integer",
    "not_edits": "{param} must be a JSON array of objects with oldText and newText",
    "missing_param": "missing parameter {params}", "no_result": "the tool produced no result",
    "timeout": "the tool did not finish within {seconds} s", "wrote": "wrote {n} lines to {path}",
    "edited": "edited {path}", "applied": "applied {n} edit(s) to {path}", "no_matches": "no matches",
    "no_output": "(no output)", "exit_code": "[exit code {code}]", "stopped": "[stopped after {seconds} s]",
    "lines_shown": "[showing lines {first}-{last} of {total}]", "clipped": "[... {n} characters omitted ...]",
    "more_entries": "[more entries omitted]", "more_matches": "[more matches omitted]",
}
MESSAGES = dict(DEFAULT_MESSAGES)


def say(key, **values):
    text = MESSAGES[key]
    for k, v in values.items():
        text = text.replace("{" + k + "}", str(v))
    return text


def as_error(text):
    """A failed call's output: the line itself when it already says it is an error, else "error: " in front."""
    return text if text.lower().startswith("error") else "error: " + text


def fail(key, **values):
    return ValueError(as_error(say(key, **values)))


def resolve(path, must_exist=True):
    if not isinstance(path, str) or not path.strip():
        raise fail("path_empty")
    p = os.path.realpath(os.path.join(ROOT, path))
    if not (p == ROOT or p.startswith(ROOT + "/") or p.startswith(SCRATCH + "/")):
        raise fail("path_outside", path=path, root=ROOT)
    if must_exist and not os.path.exists(p):
        raise fail("path_missing", path=path)
    return p


def rel(p):
    return os.path.relpath(p, ROOT) if p.startswith(ROOT) else p


def clip(text, limit):
    if len(text) <= limit:
        return text, False
    head = int(limit * 0.6)
    tail = limit - head
    return f"{text[:head]}\n{say('clipped', n=len(text) - limit)}\n{text[-tail:]}", True


def read_file(a, c):
    p = resolve(a.get("path"))
    if os.path.isdir(p):
        raise fail("is_folder", path=a.get("path"))
    with open(p, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines(keepends=True)
    s, e = a.get("start_line"), a.get("end_line")
    s = max(1, int(s)) if s not in (None, "") else 1
    e = min(len(lines), int(e)) if e not in (None, "") else len(lines)
    chunk = lines[s - 1:e]
    note = ""
    if len(chunk) > c["max_lines"]:
        chunk = chunk[:c["max_lines"]]
        note = "\n" + say("lines_shown", first=s, last=s + c["max_lines"] - 1, total=len(lines))
    return "".join(chunk) + note, 0, [rel(p)], []


def write_file(a, c):
    p = resolve(a.get("path"), must_exist=False)
    content = a.get("content")
    if not isinstance(content, str):
        raise fail("content_type")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    n = content.count(chr(10)) + (1 if content and not content.endswith(chr(10)) else 0)
    return say("wrote", n=n, path=rel(p)), 0, [], [rel(p)]


def edit_file(a, c):
    p = resolve(a.get("path"))
    old, new = a.get("old_string"), a.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str) or not old:
        raise fail("edit_args")
    with open(p, encoding="utf-8") as f:
        text = f.read()
    n = text.count(old)
    if n != 1:
        raise fail("edit_count", n=n, path=rel(p))
    with open(p, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new, 1))
    return say("edited", path=rel(p)), 0, [rel(p)], [rel(p)]


def list_dir(a, c):
    p = resolve(a.get("path") or ".")
    if not os.path.isdir(p):
        raise fail("not_folder", path=a.get("path"))
    out = []
    for n in sorted(os.listdir(p)):
        if n in ("__pycache__", ".pytest_cache"):
            continue
        out.append(n + ("/" if os.path.isdir(os.path.join(p, n)) else ""))
    return "\n".join(out[:500]) + ("\n" + say("more_entries") if len(out) > 500 else ""), 0, [], []


def search(a, c):
    try:
        rx = re.compile(a.get("pattern") or "")
    except re.error as e:
        raise fail("bad_regex", detail=e)
    base = resolve(a.get("path") or ".")
    files = [base] if os.path.isfile(base) else []
    if not files:
        for d, subdirs, names in os.walk(base):
            subdirs[:] = sorted(s for s in subdirs if s not in ("__pycache__", ".git"))
            files += [os.path.join(d, n) for n in sorted(names)]
    hits, read = [], []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    if rx.search(line):
                        hits.append(f"{rel(fp)}:{i}: {line.rstrip()[:300]}")
                        if rel(fp) not in read:
                            read.append(rel(fp))
        except (UnicodeDecodeError, OSError):
            continue
        if len(hits) > 200:
            break
    return ("\n".join(hits[:200]) or say("no_matches")) + ("\n" + say("more_matches") if len(hits) > 200 else ""), \
        0, read, []


# absolute paths other than /work, /tmp and /dev/null, and ".." at the start of a path: the outputs that enter the
# data must come from the project's own files (the gate scans them again). URLs are not paths (a URI module's tests
# are full of them); the container has no network anyway.
BLOCKED = re.compile(r"(?<![\w.~-])/(?!work\b|tmp\b|dev/null\b)[A-Za-z_]|(?:^|[\s;|&=])\.\.(?:/|\s|$)")
URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s'\"]*")


def run(argv, c, shell=False):
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": SCRATCH, "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8",
           "PYTHONPATH": ROOT}
    # the image's toolchain settings (the Elixir sandbox: Mix's home and build folder in /tmp, crash dumps)
    env.update({k: v for k, v in os.environ.items() if k.startswith(("MIX_", "HEX_", "ERL_", "ELIXIR_"))})
    try:
        r = subprocess.run(argv, cwd=ROOT, env=env, shell=shell, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=c["timeout"])
        out, code = r.stdout.decode("utf-8", errors="replace"), r.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", errors="replace") + "\n" + say("stopped", seconds=c["timeout"])
        code = 124
    return (out if out.strip() else say("no_output")) + "\n" + say("exit_code", code=code), code


def run_tests(a, c):
    target = (a.get("target") or "").strip()
    argv = list(c["test_command"]) + (target.split() if target else [])
    out, code = run(argv, c)
    return out, code, [], []


def run_command(a, c):
    cmd = a.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        raise fail("command_empty")
    if BLOCKED.search(URL.sub("", cmd)):
        raise fail("blocked", root=ROOT, scratch=SCRATCH)
    out, code = run(["/bin/sh", "-c", cmd], c)
    return out, code, [], []


# the pi profile: the names and parameters of the pi agent's four default tools, on the same implementations
def read(a, c):
    off, lim = a.get("offset"), a.get("limit")
    start = int(off) if off not in (None, "") else None
    end = (start or 1) + int(lim) - 1 if lim not in (None, "") else None
    return read_file({"path": a.get("path"), "start_line": start, "end_line": end}, c)


def edit(a, c):
    edits = a.get("edits")
    if isinstance(edits, str):
        try:
            edits = json.loads(edits)
        except ValueError:
            raise fail("edits_shape")
    if not isinstance(edits, list) or not edits or not all(isinstance(e, dict) for e in edits):
        raise fail("edits_shape")
    p = resolve(a.get("path"))
    with open(p, encoding="utf-8") as f:
        text = f.read()
    for i, e in enumerate(edits, 1):
        old, new = e.get("oldText"), e.get("newText")
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            raise fail("edit_i_args", i=i)
        n = text.count(old)
        if n != 1:
            raise fail("edit_i_count", i=i, n=n, path=rel(p))
        text = text.replace(old, new, 1)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return say("applied", n=len(edits), path=rel(p)), 0, [rel(p)], [rel(p)]


def bash(a, c):
    t = a.get("timeout")
    c = dict(c, timeout=min(int(t), c["timeout"])) if t not in (None, "") and str(t).isdigit() and int(t) > 0 else c
    return run_command({"command": a.get("command")}, c)


TOOLS = {"read_file": read_file, "write_file": write_file, "edit_file": edit_file, "list_dir": list_dir,
         "search": search, "run_tests": run_tests, "run_command": run_command,
         "read": read, "write": write_file, "edit": edit, "bash": bash}


def main():
    req = json.loads(sys.stdin.read())
    c = {"test_command": ["python3", "-m", "unittest"], "timeout": 120, "max_output": 20000, "max_lines": 2000}
    c.update(req.get("config") or {})
    MESSAGES.update(c.get("messages") or {})
    fn = TOOLS.get(req.get("tool"))
    try:
        if fn is None:
            raise fail("unknown_tool", name=req.get("tool"))
        out, code, fr, fw = fn(req.get("args") or {}, c)
        ok = True
    except (ValueError, OSError, TypeError) as e:
        out, code, fr, fw, ok = as_error(str(e)), 1, [], [], False
    out, cut = clip(out, c["max_output"])
    json.dump({"ok": ok, "output": out, "exit_code": code, "files_read": fr, "files_written": fw, "truncated": cut},
              sys.stdout)


if __name__ == "__main__":
    main()
