"""An independent check that agent sessions hold no text of the project's own tests: the agent check's tasks
(bench/agent_check/tasks: statements, workspace files, hidden tests, reference solutions, licence files) and Q51's
prompts.

gate.py's sessions stage already excludes any 12-word span shared with them. This check looks closer, in lowercased
whitespace-split words as the gate does: every N-word span (default 8 and 12) of each session's generated text
(reasoning, answer and tool calls, as the model wrote them), of its tool results and of its task statement against the
spans of the test texts, and the agent check's exercise names (affine-cipher, affine_cipher, ...; names of two words
or more) anywhere in those texts. It lists every hit with its session, turn and the test file it comes from.

usage: python heldout_check.py SESSIONS [...] --agent-tasks DIR [--q51 Q51.jsonl] [--only SESSIONS_SEQ.jsonl]
                               [--n 8,12] [--out HITS.json]
  SESSIONS: agent.py run's sessions/ folders or session files; --only keeps the sessions of an export
  (sessions_seq.jsonl, e.g. the sessions a store was built from)
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agx  # noqa: E402


def test_texts(tasks_dir, q51):
    """(label, text) of every test file and Q51 message."""
    out = []
    for d, _, names in os.walk(tasks_dir):
        for n in sorted(names):
            p = os.path.join(d, n)
            try:
                out.append((os.path.relpath(p, tasks_dir), open(p, encoding="utf-8").read()))
            except (UnicodeDecodeError, OSError):
                pass
    if q51:
        for r in agx.read_jsonl(q51):
            for m in r["messages"]:
                out.append((f"q51:{r['id']}", m.get("content") or ""))
    return out


def exercise_names(tasks_dir):
    names = set()
    for d in os.listdir(os.path.join(tasks_dir, "bugfix")):
        m = re.match(r"bugfix-(?:python|javascript)-(.+)$", d)
        if m and "-" in m[1]:
            names |= {m[1], m[1].replace("-", "_"), m[1].replace("-", "")}
    return sorted(names)


def session_texts(s):
    """(where, text) of what a session holds: the statement, each turn's generated text and tool results."""
    yield "statement", s["task"]["statement"]
    for t in s["turns"]:
        yield f"turn {t['turn']} generated", t.get("text") or ""
        for c in t.get("calls") or []:
            yield f"turn {t['turn']} result of {c.get('name')}", str(c.get("result") or "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--agent-tasks", required=True)
    ap.add_argument("--q51")
    ap.add_argument("--only")
    ap.add_argument("--n", default="8,12")
    ap.add_argument("--out")
    a = ap.parse_args()
    ns = [int(x) for x in a.n.split(",")]
    spans = {n: {} for n in ns}   # span -> the test file it comes from
    for label, text in test_texts(a.agent_tasks, a.q51):
        w = agx.words(text)
        for n in ns:
            for i in range(len(w) - n + 1):
                spans[n].setdefault(" ".join(w[i:i + n]), label)
    names = exercise_names(a.agent_tasks)
    keep = None
    if a.only:
        keep = {json.loads(line)["session_id"] for line in open(a.only)}
    files = []
    for p in a.sessions:
        files += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.endswith(".json")
                  and not f.endswith(".error.json")] if os.path.isdir(p) else [p]
    hits, n_sessions, n_words = [], 0, 0
    for path in files:
        s = json.load(open(path))
        if keep is not None and s["session_id"] not in keep:
            continue
        n_sessions += 1
        for where, text in session_texts(s):
            w = agx.words(text)
            n_words += len(w)
            low = text.lower()
            for name in names:
                if re.search(r"(?<![a-z0-9_])" + re.escape(name) + r"(?![a-z0-9_])", low):
                    hits.append({"session": s["session_id"], "where": where, "kind": "name", "match": name})
            for n in ns:
                for i in range(len(w) - n + 1):
                    src = spans[n].get(" ".join(w[i:i + n]))
                    if src:
                        hits.append({"session": s["session_id"], "where": where, "kind": f"{n} words",
                                     "match": " ".join(w[i:i + n]), "test_file": src})
    if keep is not None and n_sessions != len(keep):
        print(f"warning: {len(keep) - n_sessions} sessions of --only not found among the files", file=sys.stderr)
    by = {}
    for h in hits:
        k = (h["kind"], h.get("test_file", ""), h["where"].split(" ", 2)[-1] if h["where"].startswith("turn") else
             h["where"])
        by[k] = by.get(k, 0) + 1
    print(f"{n_sessions} sessions, {n_words} words; spans of {', '.join(f'{n}: {len(spans[n])}' for n in ns)}; "
          f"exercise names {len(names)}; hits {len(hits)}")
    for (kind, src, where), c in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"  {c:6d}  {kind:9s}  {where:30s}  {src}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"sessions": n_sessions, "words": n_words, "names": names, "hits": hits}, f, indent=1)


if __name__ == "__main__":
    main()
