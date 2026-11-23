#!/usr/bin/env python3
"""Mirror the experiment registry's open entries as GitHub issues, so two people never work on the same idea.

The registry (`quants/experiments.md`) stays the record: numbers, results and data links live there. Each open entry
gets one issue whose assignee says who is working on it:

- a planned entry becomes an open issue;
- a running entry becomes an open issue assigned to the account that runs this script;
- a parked entry becomes a closed issue ("not planned"), so a search still finds the idea and why it was parked;
- a finished entry gets no issue, and an issue whose entry has since finished is closed with a link to the entry.

Anyone claims an idea by commenting on its issue; the run lists unassigned issues with comments from others, so the
owner assigns them (on a personal account's repo every collaborator gets write access, so contributors are not added).

Each issue carries the marker `<!-- registry: Qnnn -->`, which is how later runs find it. Issues without a marker are
ideas from others: the run lists the open ones the registry does not mention yet, so each gets its entry.

Dry run by default: it reads the repo's issues (nothing else, and not even that with --offline), prints what it would
do, and with --preview writes a page to check by eye. Only --post changes GitHub.

    python3 -I scripts/registry_issues.py --preview preview.md          # dry run
    python3 -I scripts/registry_issues.py --post --limit 5              # do the first 5 steps
    python3 -I scripts/registry_issues.py --post                        # do everything (paced)

GitHub allows about 500 issue writes an hour, so writes are spaced (--sleep, 7.5 s by default): a first run of
about 900 issues takes about 2.5 hours. A run that stops can be started again: it finds what it already made.

Patterns that must never reach GitHub beyond the built-in ones (machine names and the like) come from
--private-patterns FILE (one regular expression per line, case-insensitive) or the PRIVATE_PATTERNS environment
variable naming such a file. --post refuses to run without them, and an issue whose text matches any pattern is held
back and listed instead.

Needs Python 3.8+ and, unless --offline, the GitHub CLI (`gh`) logged in.
"""

import argparse
import json
import os
import posixpath
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

REPO = "Cinderella-Man/llms-on-crypto-mining-rigs"
REGISTRY = "quants/experiments.md"
MARKER = re.compile(r"<!-- registry: (Q\d+) -->")

LABELS = {
    "registry": ("ededed", "Mirrors an entry of quants/experiments.md, the record"),
    "status: planned": ("0e8a16", "Planned in the registry: free to take, say so in a comment first"),
    "status: running": ("fbca04", "Someone is working on it: see the assignee"),
    "status: parked": ("cfd3d7", "Parked in the registry: the evidence argues against it; the reason is in the issue"),
    "model: flint": ("1d76db", "Flint"),
    "model: flash-next": ("1d76db", "Swift Qwen3.8-Flash-Next"),
    "model: ornith": ("1d76db", "Ornith-1.5-122B"),
    "model: qwen3.6": ("1d76db", "Qwen3.6-35B-A3B"),
    "needs: pascal card": ("d93f0b", "Runs or is timed on a Pascal card (P106, GTX 1060/1070/1080)"),
    "needs: cuda gpu": ("f9d0c4", "Needs a CUDA GPU of any kind (quality checks, generation, correctness)"),
    "engine code": ("5319e7", "Code in the llama.cpp fork (Miner Llama)"),
}

SECTION_MODEL = {"Flint": "model: flint", "Qwen3.8-Flash-Next": "model: flash-next", "Qwen3.6-35B-A3B": "model: qwen3.6"}
TEXT_MODEL = [
    (re.compile(r"\bFlint\b"), "model: flint"),
    (re.compile(r"Flash-Next|\bSwift\b"), "model: flash-next"),
    (re.compile(r"\bOrnith\b"), "model: ornith"),
    (re.compile(r"Qwen3\.6|35B-A3B"), "model: qwen3.6"),
]
NEEDS_PASCAL = re.compile(r"\brigs?\b|\bP106\b|GTX 10[5-8]0", re.I)
NEEDS_GPU = re.compile(r"quant machine|LLMRig|build machine|\b3090s?\b|\bone card\b|\bGPUs?\b", re.I)
CPU_ONLY = re.compile(r"\bCPU\b", re.I)
ENGINE = re.compile(r"\bfork\b|\bkernel|`pascal-p106", re.I)

# Built-in guards; machine names and the like come from --private-patterns, never from this public file.
GUARDS = [
    re.compile(r"/home/|/srv/|/mnt/", re.I),
    re.compile(r"\bGPU-[0-9a-f]{8}-", re.I),
    re.compile(r"\b[0-9a-f]{2}(:[0-9a-f]{2}){5}\b", re.I),
    re.compile(r"\b(?!127\.0\.0\.1\b)(\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
]


@dataclass
class Entry:
    id: str
    line: int
    section: str
    question: str
    status: str
    result: str
    how: str
    state: str = ""        # planned, running, parked, finished, dropped
    sure: bool = True      # False: no status word found, kept open by default
    title: str = ""
    labels: list = field(default_factory=list)

    @property
    def num(self):
        return int(self.id[1:])


# ---------------------------------------------------------------- the registry

def split_cells(line):
    """A table row's cells, split on pipes that are not escaped (GitHub's rule: a pipe splits even inside code)."""
    s = line.rstrip("\n").strip()
    cells = re.split(r"(?<!\\)\|", s)
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1].strip() == "":
        cells = cells[:-1]
    return [c.strip().replace("\\|", "|") for c in cells]


def plain(text):
    return re.sub(r"[*_`]", "", text).strip().lower()


OPEN_WORDS = re.compile(r"\b(planned|next|first|reopened|re-scoped|narrowed|later|last|queued|open|again|"
                        r"still worth|to do|ready|waits|waiting)\b")
DONE_WORDS = re.compile(r"\b(done|closed|merged|superseded|dropped|retired|abandoned|answered|measured|built|"
                        r"written|negligible|decided|passed|finished|fixed|landed|ported|confirmed|covered by|"
                        r"in v\d[\d.]*|ships|shipped|regression)\b")
OPEN_STEP = re.compile(r"\b(next|planned)\b")
DROP_WORDS = re.compile(r"\b(closed|superseded|dropped|retired|abandoned)\b")


def verdict(text):
    """open, finished, dropped or None, from whichever kind of status word comes first in text."""
    o, d = OPEN_WORDS.search(text), DONE_WORDS.search(text)
    if o and (not d or o.start() < d.start()):
        return "planned"
    if d:
        return "dropped" if DROP_WORDS.match(text, d.start()) and "merged" not in text[:d.start()] else "finished"
    return None


def open_step(s):
    """A finished part followed by a step still to do that has no entry of its own ("code done (C22), rig run next"),
    not the journal's old "planned" ("planned 2026-11-12; done 2026-11-12", "Before: planned") nor a step another
    entry carries ("on a P106: Q1315, planned")."""
    d = DONE_WORDS.search(s)
    if not d:
        return False
    rest = re.split(r"\bbefore:", s[d.end():])[0]
    for m in OPEN_STEP.finditer(rest):
        before = rest[:m.start()]
        if re.search(r"(?<![\w.])\d[\d,]*\s*$|\bnot\s*$", before):  # a count ("702 planned") or "not planned"
            continue
        if not re.search(r"\b[qcrs]\d+\b[^.;]{0,25}$", before):
            return True
    return False


def classify(e):
    """A status cell is a journal whose newest verdict is usually the first bold phrase ("**closed 2026-11-16:**
    ... Before: planned"), but not always ("planned 2026-11-12; **done 2026-11-13**"). So: the first bold phrase
    with a status word decides; then the cell's opening clause; then "Before: <word>"; else the entry is unclear and
    kept open (listed in the preview), since a missing issue costs more than a needless one."""
    s, q = plain(e.status), plain(e.question)
    if re.match(r"parked\b", s) or q.startswith("parked") or re.match(r"(planned|open)\b[^.;]*\bparked\b", s):
        return "parked", True
    if re.match(r"(running|in progress|started)\b", s):
        return "running", True
    v = None
    for bold in re.findall(r"\*\*(.+?)\*\*", e.status):
        if re.match(r"parked\b", plain(bold)):
            return "parked", True
        v = verdict(plain(bold))
        if v:
            break
    if not v:
        v = verdict(re.split(r"[:;.]\s", s, maxsplit=1)[0])
    if v == "finished" and open_step(s):
        return "planned", False
    if v:
        return v, True
    m = re.search(r"\bbefore: (\w+)", s)
    if m and m.group(1) in ("planned", "parked"):
        return m.group(1), True
    return "planned", False


def read_registry(path):
    entries, section, bad = [], "", []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.startswith("### "):
                section = line[4:].strip()
            m = re.match(r"^\|\s*(Q\d+)\s*\|", line)
            if not m:
                continue
            c = split_cells(line)
            if len(c) < 5:
                bad.append((m.group(1), n, len(c)))
                c += [""] * (5 - len(c))
            if len(c) > 5:  # an unescaped pipe in the text: keep the first and last cells' meaning
                c = [c[0], " | ".join(c[1:len(c) - 3]), c[-3], c[-2], c[-1]]
                bad.append((m.group(1), n, "extra pipes"))
            e = Entry(c[0], n, section, c[1], c[2], c[3], c[4])
            e.state, e.sure = classify(e)
            e.title = make_title(e)
            e.labels = make_labels(e)
            entries.append(e)
    return entries, bad


# ---------------------------------------------------------------- titles, labels, bodies

def outside_code(text, fn):
    """Apply fn to the parts of text that are not code spans."""
    parts = re.split(r"(`+[^`]*?`+)", text)
    return "".join(p if p.startswith("`") else fn(p) for p in parts)


def no_mentions(text):
    """Stop GitHub from linking `#123` to this repo's issues (they are mostly upstream pull requests) or pinging @names."""
    def fix(p):
        p = re.sub(r"(?<![\w/&`\[])#(\d+)\b", r"`#\1`", p)
        p = re.sub(r"(?<![\w.`/])@([A-Za-z][\w-]*)", r"`@\1`", p)
        return p
    return outside_code(text, fix)


def make_title(e):
    t = e.question.strip()
    while True:  # leading "(C885, a contribution, 2026-11-19)" and the like
        m = re.match(r"^\((?:[^()]|\([^()]*\))*\)\s*", t)
        if not m:
            break
        t = t[m.end():]
    t = re.sub(r"^(\*\*)?(Parked|Dropped|New)[:.]\s*(\*\*)?", "", t, flags=re.I)
    m = re.match(r"^\*\*(.+?)\*\*", t)
    if m:
        t = m.group(1)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    out, depth, tick = "", 0, False
    for i, ch in enumerate(t):  # the first sentence, not cut inside brackets or code
        if ch == "`":
            tick = not tick
        elif not tick and ch == "(":
            depth += 1
        elif not tick and ch == ")":
            depth = max(0, depth - 1)
        if (not tick and depth == 0 and ch in ".:;" and (i + 1 == len(t) or t[i + 1] == " ")
                and not re.search(r"(e\.g|i\.e|\best|\bvs|\bcf|\bno)$", out, flags=re.I)):
            break
        out += ch
    t = out.replace("**", "").strip().rstrip(",")
    if len(t) > 110:
        t = t[:110].rsplit(" ", 1)[0].rstrip(",;:(") + "…"
    t = t[:1].upper() + t[1:]
    return no_mentions(f"{t} ({e.id})")


def make_labels(e):
    labels = ["registry"]
    if e.state in ("planned", "running", "parked"):
        labels.append(f"status: {e.state}")
    # The idea's own words name its model; the section only when they name none (the Flash-Next section also holds
    # engine-wide entries, so it is no evidence on its own).
    models = [lab for rx, lab in TEXT_MODEL if rx.search(e.question)]
    if not models:
        models = [lab for key, lab in SECTION_MODEL.items() if e.section.startswith(key) and key != "Qwen3.8-Flash-Next"]
    labels += models
    where = f"{e.how} {e.status}"
    if NEEDS_PASCAL.search(where):
        labels.append("needs: pascal card")
    gpu = [m for m in NEEDS_GPU.finditer(e.how)]
    if gpu and not CPU_ONLY.search(e.how[gpu[0].start():gpu[0].start() + 40]):
        labels.append("needs: cuda gpu")
    if ENGINE.search(e.how):
        labels.append("engine code")
    return labels


def absolute_links(text, repo, branch):
    def fix(m):
        target = m.group(1)
        if target.startswith("#"):
            url = f"https://github.com/{repo}/blob/{branch}/{REGISTRY}{target}"
        else:
            path, _, anchor = target.partition("#")
            path = posixpath.normpath(posixpath.join(posixpath.dirname(REGISTRY), path))
            url = f"https://github.com/{repo}/blob/{branch}/{path}" + (f"#{anchor}" if anchor else "")
        return f"]({url})"
    return re.sub(r"\]\((?!https?://|mailto:)([^)\s]+)\)", fix, text)


def md(text, repo, branch):
    return no_mentions(absolute_links(text.replace("<br>", "\n\n"), repo, branch))


def make_body(e, repo, ref, branch):
    link = f"https://github.com/{repo}/blob/{ref}/{REGISTRY}#L{e.line}"
    out = [
        f"<!-- registry: {e.id} -->",
        f"Entry **{e.id}** of the experiment registry, [`{REGISTRY}` line {e.line}]({link}) (as of `{ref[:8]}`). "
        f"The registry entry is the record and may be newer than this copy: search the file on `{branch}` for "
        f"`| {e.id} |`.",
        "",
        "**The idea**",
        "",
        md(e.question, repo, branch),
        "",
        "**Status in the registry**",
        "",
        md(e.status, repo, branch) or "—",
    ]
    if e.how and e.how not in ("—", "-"):
        out += ["", "**Where it runs, how**", "", md(e.how, repo, branch)]
    out += [
        "",
        "---",
        "Want to take this on? Say so in a comment first, so you can be assigned and nobody else starts it. "
        "A pull request that does it links this issue. The issue is closed when its registry entry is.",
    ]
    return "\n".join(out)


# ---------------------------------------------------------------- GitHub

WRITE_GAP = 7.5


def gh(args, stdin=None, write=False):
    if write:
        time.sleep(WRITE_GAP)
    for attempt in range(6):
        p = subprocess.run(["gh"] + args, input=stdin, capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout
        if "rate limit" in (p.stderr or "").lower() and attempt < 5:
            wait = 60 * (attempt + 1)
            print(f"  rate limited, waiting {wait} s", file=sys.stderr)
            time.sleep(wait)
            continue
        raise SystemExit(f"gh {' '.join(args[:3])} … failed: {p.stderr.strip()}")


def read_issues(repo):
    out = gh(["issue", "list", "-R", repo, "--state", "all", "--limit", "5000",
              "--json", "number,title,state,stateReason,labels,assignees,body,comments"])
    return json.loads(out)


# ---------------------------------------------------------------- the plan

@dataclass
class Step:
    kind: str          # create, close, relabel
    entry: Entry
    issue: dict = None
    detail: str = ""


def guard_hits(text, extra):
    return [rx.pattern for rx in GUARDS + extra if rx.search(text)]


def plan(entries, issues, repo, ref, branch, extra):
    by_id = {}
    for i in issues:
        m = MARKER.search(i.get("body") or "")
        if m:
            by_id.setdefault(m.group(1), i)
    steps, held = [], []
    for e in sorted(entries, key=lambda x: x.num):
        issue = by_id.get(e.id)
        if issue is None:
            if e.state in ("planned", "running", "parked"):
                body = make_body(e, repo, ref, branch)
                hits = guard_hits(e.title + "\n" + body, extra)
                if hits:
                    held.append((e, hits))
                else:
                    steps.append(Step("create", e, detail=body))
            continue
        is_open = issue["state"] == "OPEN"
        if e.state in ("finished", "dropped") and is_open:
            steps.append(Step("close", e, issue, "completed" if e.state == "finished" else "not planned"))
        elif e.state == "parked" and is_open:
            steps.append(Step("close", e, issue, "not planned"))
        elif e.state in ("planned", "running") and not is_open:
            steps.append(Step("reopen", e, issue))
        have = {l["name"] for l in issue.get("labels", [])}
        want = {l for l in e.labels if l.startswith("status: ")}
        drop = {l for l in have if l.startswith("status: ")} - want
        add = want - have
        if add or drop:
            steps.append(Step("relabel", e, issue, json.dumps({"add": sorted(add), "remove": sorted(drop)})))
    return steps, held, by_id


def unregistered(issues, registry_text):
    """Open issues with no marker that the registry does not link: ideas from others that need an entry."""
    out = []
    for i in issues:
        if i["state"] != "OPEN" or MARKER.search(i.get("body") or ""):
            continue
        n = i["number"]
        if not re.search(rf"/issues/{n}\b|\bissue #{n}\b", registry_text):
            out.append(i)
    return out


def claims(by_id, entries, me):
    """Registry issues assigned to someone other than me whose entry does not say running yet."""
    state = {e.id: e.state for e in entries}
    out = []
    for qid, i in by_id.items():
        who = [a["login"] for a in i.get("assignees", []) if a["login"] != me]
        if who and i["state"] == "OPEN" and state.get(qid) != "running":
            out.append((qid, i, who))
    return out


def asks(by_id, me):
    """Open registry issues nobody is assigned to where someone else has commented: most are claims ("I'm taking
    this"). Only collaborators can assign themselves, so the owner assigns them; GitHub allows assigning anyone who
    has commented on the issue."""
    out = []
    for qid, i in sorted(by_id.items(), key=lambda kv: int(kv[0][1:])):
        if i["state"] != "OPEN" or i.get("assignees"):
            continue
        others = [c for c in i.get("comments", []) if (c.get("author") or {}).get("login") not in (me, None)]
        if others:
            last = others[-1]
            out.append((qid, i, sorted({c["author"]["login"] for c in others}), last["body"].strip()[:140]))
    return out


# ---------------------------------------------------------------- doing it

def ensure_labels(repo):
    for name, (colour, desc) in LABELS.items():
        gh(["label", "create", name, "-R", repo, "--color", colour, "--description", desc, "--force"], write=True)


def run_step(s, repo, ref):
    e = s.entry
    link = f"https://github.com/{repo}/blob/{ref}/{REGISTRY}#L{e.line}"
    if s.kind == "create":
        args = ["issue", "create", "-R", repo, "--title", e.title, "--body-file", "-"]
        for lab in e.labels:
            args += ["--label", lab]
        if e.state == "running":
            args += ["--assignee", "@me"]
        url = gh(args, stdin=s.detail, write=True).strip()
        if e.state == "parked":  # the reason is in the body's status, so no comment (one write fewer)
            gh(["issue", "close", url.rsplit("/", 1)[-1], "-R", repo, "--reason", "not planned"], write=True)
        return url
    n = str(s.issue["number"])
    if s.kind == "close":
        gh(["issue", "close", n, "-R", repo, "--reason", s.detail,
            "--comment", f"Closed: the registry entry {e.id} is {e.state} ([the entry]({link}))."], write=True)
    elif s.kind == "reopen":
        gh(["issue", "reopen", n, "-R", repo, "--comment", f"Reopened: the registry entry {e.id} is {e.state} again."],
           write=True)
    elif s.kind == "relabel":
        d = json.loads(s.detail)
        args = ["issue", "edit", n, "-R", repo]
        for lab in d["add"]:
            args += ["--add-label", lab]
        for lab in d["remove"]:
            args += ["--remove-label", lab]
        gh(args, write=True)
    return f"#{n}"


# ---------------------------------------------------------------- the preview

def esc(t):
    return t.replace("|", "\\|")


def write_preview(path, a, entries, bad, steps, held, issues, by_id, unreg, claimed, asked, examples):
    st = Counter(e.state for e in entries)
    kinds = Counter(s.kind + (f" ({s.entry.state})" if s.kind == "create" else "") for s in steps)
    lab = Counter(l for s in steps if s.kind == "create" for l in s.entry.labels)
    created = [s for s in steps if s.kind == "create"]
    w = []
    w += ["# Registry to issues: dry run", ""]
    w += [f"Registry `{REGISTRY}` at `{a.ref[:8]}` ({a.branch}): {len(entries)} entries, "
          + ", ".join(f"{v} {k}" for k, v in st.most_common()) + ".",
          f"Repo `{a.repo}`: {len(issues)} issues now, {len(by_id)} of them mirror an entry.", ""]
    w += ["## What `--post` would do", "", "| step | count |", "|---|---:|"]
    w += [f"| {k} | {v} |" for k, v in sorted(kinds.items())]
    w += [f"| held back by the privacy guard | {len(held)} |", ""]
    w += ["Open issues for planned and running entries; running ones assigned to the account that posts; parked ones "
          "created and closed at once as \"not planned\". Finished and dropped entries get no issue.", ""]
    w += ["## Labels it would create", "", "| label | colour | meaning | issues |", "|---|---|---|---:|"]
    w += [f"| `{n}` | `#{c}` | {d} | {lab.get(n, 0)} |" for n, (c, d) in LABELS.items()]
    w += ["", "Model, \"needs\" and \"engine code\" labels are guessed from the entry's words and its \"How\" cell; "
          "some will be wrong and can be fixed by hand.", ""]
    w += ["## To check by hand", ""]
    run = [e for e in entries if e.state == "running"]
    w += [f"### Entries marked running ({len(run)}): each issue would be assigned to you", ""]
    w += [f"- {esc(e.title)}: status \"{esc(e.status[:140])}\"" for e in sorted(run, key=lambda x: x.num)]
    unsure = [e for e in entries if not e.sure]
    w += ["", f"### Entries with no status word, kept open as planned ({len(unsure)})", "",
          "Their status cell opens with a note (a measurement, upstream news) and names no verdict.", ""]
    w += [f"- {esc(e.title)}: \"{esc(e.status[:160])}\"" for e in sorted(unsure, key=lambda x: x.num)]
    odd = [e for e in entries if e.state in ("finished", "dropped")
           and re.search(r"\b(planned|running|next)\b", plain(e.status))]
    w += ["", f"### Entries read as finished whose status also says planned, next or running ({len(odd)})", "",
          "These get no issue. Most name the entry that carries the open half; any that does not may need one.", ""]
    w += [f"- {esc(e.title)}: \"{esc(e.status[:160])}\"" for e in sorted(odd, key=lambda x: x.num)]
    w += ["", f"### Held back by the privacy guard ({len(held)})", ""]
    w += [f"- {esc(e.title)}: matches {', '.join(f'`{h}`' for h in hits)}" for e, hits in held] or ["None."]
    if bad:
        w += ["", f"### Rows with odd pipes ({len(bad)}): parsed, check their text", ""]
        w += [f"- {q}, line {n}: {why}" for q, n, why in bad]
    w += ["", f"### Open issues the registry does not mention ({len(unreg)}): each needs an entry", ""]
    w += [f"- #{i['number']}: {esc(i['title'])}" for i in unreg] or ["None."]
    w += ["", f"### Issues someone else has taken whose entry does not say running ({len(claimed)})", ""]
    w += [f"- {q} (#{i['number']}): {', '.join(who)}" for q, i, who in claimed] or ["None."]
    w += ["", f"### Unassigned issues with a comment from someone else ({len(asked)}): assign them or answer", "",
          f"Assign with `gh issue edit N -R {a.repo} --add-assignee LOGIN`.", ""]
    w += [f"- {q} (#{i['number']}), {', '.join(who)}: \"{esc(last)}\"" for q, i, who, last in asked] or ["None."]
    w += ["", "## Example issues in full", ""]
    for s in examples:
        w += [f"<details><summary><b>{esc(s.entry.title)}</b> · {', '.join(s.entry.labels)}</summary>", "",
              s.detail, "", "</details>", ""]
    w += ["## Every issue it would create, in this order", "", "| | entry | title | labels |", "|---:|---|---|---|"]
    w += [f"| {k} | {s.entry.id} | {esc(s.entry.title)} | {', '.join(l for l in s.entry.labels if l != 'registry')} |"
          for k, s in enumerate(created, 1)]
    others = [s for s in steps if s.kind != "create"]
    if others:
        w += ["", "## Other steps", ""]
        w += [f"- {s.kind} #{s.issue['number']} ({s.entry.id}, {s.entry.state}) {s.detail}" for s in others]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(w) + "\n")


# ---------------------------------------------------------------- main

def main():
    global WRITE_GAP
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--registry", default=REGISTRY, help="path to experiments.md (default: %(default)s)")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--branch", default="v5", help="branch the issue's links point to (default: %(default)s)")
    ap.add_argument("--ref", help="commit the line links pin (default: git rev-parse HEAD beside the registry)")
    ap.add_argument("--private-patterns", default=os.environ.get("PRIVATE_PATTERNS"))
    ap.add_argument("--preview", help="write a page with the plan, the labels and example issues")
    ap.add_argument("--plan-json", help="write every step as JSON")
    ap.add_argument("--offline", action="store_true", help="do not read the repo's issues (dry run only)")
    ap.add_argument("--post", action="store_true", help="do it: create labels and issues, close and relabel")
    ap.add_argument("--limit", type=int, default=0, help="with --post, do only the first N steps")
    ap.add_argument("--sleep", type=float, default=WRITE_GAP,
                    help="seconds before each GitHub write (default: %(default)s)")
    a = ap.parse_args()
    WRITE_GAP = a.sleep

    if not a.ref:
        a.ref = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(a.registry)), "rev-parse", "HEAD"],
                               capture_output=True, text=True, check=True).stdout.strip()
    extra = []
    if a.private_patterns:
        with open(a.private_patterns, encoding="utf-8") as f:
            extra = [re.compile(l.strip(), re.I) for l in f if l.strip() and not l.startswith("#")]
    if a.post and (a.offline or not extra):
        raise SystemExit("--post needs the repo's issues (no --offline) and --private-patterns")

    entries, bad = read_registry(a.registry)
    issues = [] if a.offline else read_issues(a.repo)
    steps, held, by_id = plan(entries, issues, a.repo, a.ref, a.branch, extra)
    with open(a.registry, encoding="utf-8") as f:
        unreg = unregistered(issues, f.read())
    me = "" if a.offline else gh(["api", "user", "--jq", ".login"]).strip()
    claimed = claims(by_id, entries, me)
    asked = asks(by_id, me)

    kinds = Counter(s.kind for s in steps)
    print(f"{len(entries)} entries; steps: {dict(kinds)}; held back: {len(held)}; "
          f"unregistered open issues: {len(unreg)}; claims to record: {len(claimed)}; "
          f"comments to answer or assign: {len(asked)}")
    if a.preview:
        pick, seen = [], set()
        for s in steps:  # one example of each state, plus the longest
            if s.kind == "create" and s.entry.state not in seen:
                pick.append(s)
                seen.add(s.entry.state)
        longest = max((s for s in steps if s.kind == "create"), key=lambda s: len(s.detail), default=None)
        if longest and longest not in pick:
            pick.append(longest)
        contrib = next((s for s in steps if s.kind == "create" and "contribution" in s.entry.question), None)
        if contrib and contrib not in pick:
            pick.append(contrib)
        write_preview(a.preview, a, entries, bad, steps, held, issues, by_id, unreg, claimed, asked, pick)
        print(f"preview: {a.preview}")
    if a.plan_json:
        with open(a.plan_json, "w", encoding="utf-8") as f:
            json.dump([{"step": s.kind, "entry": s.entry.id, "state": s.entry.state, "title": s.entry.title,
                        "labels": s.entry.labels, "issue": (s.issue or {}).get("number"), "detail": s.detail}
                       for s in steps], f, ensure_ascii=False, indent=1)
    if not a.post:
        print("dry run: nothing changed on GitHub")
        return
    ensure_labels(a.repo)
    todo = steps[:a.limit] if a.limit else steps
    for k, s in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {s.kind} {s.entry.id}: {run_step(s, a.repo, a.ref)}", flush=True)


if __name__ == "__main__":
    main()
