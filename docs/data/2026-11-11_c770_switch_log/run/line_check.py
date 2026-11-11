#!/usr/bin/env python3
"""C770 (Q1355): check llama-server's switch line in each arm's two logs.

Each argument is <arm>:<env.tsv>:<base server.log>:<branch server.log>. env.tsv holds the arm's variables, one
NAME=value per line with "read" or "unread": what the branch's line should list them as. Pass, for each arm:
- the base's log has no switch line;
- the branch's log has exactly one, at the INFO level, and it is "switches: none set (...)" with no variables, or
  "switches: <n> set: <read>; not read by this build: <unread> (...)" with n the number of variables, each list the
  arm's variables of that kind in name order, as NAME=value (a value that is empty or holds a space in double quotes),
  and a name with KEY, TOKEN, SECRET or PASSWORD as NAME=(hidden);
- a hidden variable's value appears nowhere in the branch's log.
Also counted, not part of the verdict: the library's own line of LLAMA_SAMPLING_TOP_K_ROWS ("one top-k over the
output rows"), which the server drops at its default verbosity.
Usage: line_check.py [--tsv lines.tsv] <arm>:<env.tsv>:<base log>:<branch log> ...
"""
import argparse
import re
import shlex
import sys

TAG = "common_params_print_info: switches: "
LIB = "one top-k over the output rows"
SECRET = ("KEY", "TOKEN", "SECRET", "PASSWORD")
TAIL = re.compile(r" \(this build reads (\d+) LLAMA_ and GGML_ names\)$")


def read_env(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            kv, kind = line.rsplit("\t", 1)
            name, _, value = kv.partition("=")
            out.append((name, value, kind))
    return out


def shown(name, value):
    if any(s in name for s in SECRET):
        return f"{name}=(hidden)"
    return f"{name}={value}"


def switch_lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    lines = [l for l in text.splitlines() if TAG in l]
    return text, lines, text.count(LIB)


def check_arm(arm, env_path, base_log, tip_log, rows):
    env = read_env(env_path)
    problems = []
    _, base_lines, base_lib = switch_lines(base_log)
    text, tip_lines, tip_lib = switch_lines(tip_log)
    if base_lines:
        problems.append(f"the base's log has {len(base_lines)} switch lines")
    if len(tip_lines) != 1:
        problems.append(f"the branch's log has {len(tip_lines)} switch lines, not 1")
    line = tip_lines[0] if tip_lines else ""
    level = line.split()[1] if len(line.split()) > 1 else ""
    if tip_lines and level != "I":
        problems.append(f"the line's level is {level!r}, not I")
    body = line.split(TAG, 1)[1] if TAG in line else ""
    m = TAIL.search(body)
    n_names = int(m.group(1)) if m else 0
    if not m:
        problems.append("the line does not end with the count of names the build reads")
    body = body[: m.start()] if m else body
    if not env:
        if body != "none set":
            problems.append(f"no variable set, but the line says {body!r}")
    else:
        want_read = sorted((n, v) for n, v, k in env if k == "read")
        want_unread = sorted((n, v) for n, v, k in env if k == "unread")
        head = f"{len(env)} set:"
        if not body.startswith(head):
            problems.append(f"the line does not start with {head!r}: {body!r}")
        rest = body[len(head):]
        read_part, _, unread_part = rest.partition(" not read by this build: ")
        read_part = read_part.rstrip(";")
        got_read = shlex.split(read_part)
        got_unread = shlex.split(unread_part)
        exp_read = [shown(n, v) for n, v in want_read]
        exp_unread = [shown(n, v) for n, v in want_unread]
        if got_read != exp_read:
            problems.append(f"read: {got_read} against {exp_read}")
        if got_unread != exp_unread:
            problems.append(f"not read: {got_unread} against {exp_unread}")
    for n, v, _ in env:
        if any(s in n for s in SECRET) and v and v in text:
            problems.append(f"{n}'s value is in the branch's log")
    rows.append([arm, "base", str(len(base_lines)), str(base_lib), ""])
    rows.append([arm, "tip", str(len(tip_lines)), str(tip_lib), line.split("I cmn", 1)[-1].strip() if line else ""])
    verdict = "OK" if not problems else "FAIL: " + "; ".join(problems)
    print(f"{arm}: line {verdict} ({len(env)} variables, {n_names} names read by the build; the library's own "
          f"LLAMA_SAMPLING_TOP_K_ROWS line {base_lib} times in the base's log, {tip_lib} in the branch's)")
    return not problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv")
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args()
    rows = []
    ok = True
    for spec in a.arms:
        arm, env_path, base_log, tip_log = spec.split(":", 3)
        ok = check_arm(arm, env_path, base_log, tip_log, rows) and ok
    if a.tsv:
        with open(a.tsv, "w", encoding="utf-8") as f:
            f.write("arm\tbuild\tswitch_lines\tlibrary_line\tline\n")
            for r in rows:
                f.write("\t".join(r) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
