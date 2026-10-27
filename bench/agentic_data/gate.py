"""The gate of the agentic data pipeline: every stage is checked before the next one reads it, and every gate record
names its parents' records by sha256, so each published row chains back to an approved code base, an approved text
and an approved writer.

  workspaces  workspace manifests (workspace.py): the code base is approved in codebases.json and its source in
              sources.json; every code or licence file is the scanned file (same sha256, verdict ok); scaffold files
              are empty; the folder on disk equals the manifest; no file shares 12 words with a held-out test.
  tasks       task rows (tasks.py): the workspace manifest is gated (stage 1: a parent workspaces record; stage 2:
              its base is, its test file is a passed session's file, and the session is in a parent sessions
              record); the folder equals the manifest; the template is in texts.json and every text there is
              Flint's (an approved writer) or the owner's; the statement is the template filled with the
              placeholders, whose values are names found in the workspace; no 12-word overlap with held-out tests.
  sessions    session files (agent.py run): the task is in a parent tasks record, byte for byte; system prompt, tool
              schema and first messages are texts.json's; the writer and its model file are approved; the token
              chain is append-only and matches the turn records; every tool call ran in the sandbox on paths inside
              the workspace; no tool output or generated text carries a licence or copyright line that is not in
              the workspace's own files, or a path outside the workspace and CPython's library; no 12-word overlap
              with held-out tests. A session that breaks the pipeline's rules fails the gate; one whose content trips
              the licence, path or held-out checks is marked "excluded: …" and goes no further.
  answers     the exports for the store and the dump (export.py: turns.jsonl, sessions_seq.jsonl): every row is a
              passed session's tokens, exactly. The record has check_sources.py's layout, so its dump and derived
              stages chain from it (--parent).
  dataset     the publishable folder (export.py hf): every row is a passed session, its licence fields match the
              code base, LICENSE, NOTICE, README.md and LICENSES/<id>.txt for every licence are present.

usage: python gate.py STAGE FILES... [--parent G.gate.json ...] --sources SOURCES.json --codebases codebases.json
                     [--texts texts.json ...] [--scan SCAN.tsv ...] [--q51 Q51.jsonl] [--agent-tasks DIR] [--out G.gate.json]
Writes the gate record (default: the first file + .gate.json) and its manifest (.tsv), exits 1 on any failure.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agx  # noqa: E402
import license_scan  # noqa: E402

LICENCE_LINE = re.compile("|".join(rx for _, rx in license_scan.FAMILIES) +
                          r"|SPDX-License-Identifier|^\W*(?:Copyright\b|\(c\)\s|©)", re.I | re.M)
ALLOWED_PATHS = ("/work", "/tmp", "/usr/local/lib/python3.12/", "/usr/local/bin/python", "/opt/agx/",
                 "/bin/sh:", "/usr/local/lib/elixir/", "/usr/local/lib/erlang/",
                 *(f"/usr/local/bin/{b}" for b in ("elixir", "elixirc", "iex", "mix", "erl", "erlc", "escript",
                                                   "epmd")))   # CPython, our tool runner, the shell's own error
# prefix ("/bin/sh: 1: x: not found"), and the Elixir sandbox's runtimes and their launchers (Apache-2.0)
# a folder under /srv, /mnt or /home named in a published row: a real name follows, so the model's own placeholders
# ("~/.mix would be at /root/.mix or /home/...") pass
MACHINE_PATH = re.compile(r"/(?:srv|mnt|home)/[A-Za-z0-9_][\w.-]*")
ABS_PATH = re.compile(r"(?<![\w.~:/-])(/(?:usr|etc|opt|home|root|var|srv|mnt|lib|bin|sbin|proc|sys|dev|run)\b[^\s'\"]*)")


def load_parent(path, src, errors):
    raw = open(path, "rb").read()
    rec = json.loads(raw)
    if rec.get("verdict") != "PASS":
        errors.append(f"parent {path}: verdict {rec.get('verdict')}")
    for f in rec["files"]:
        if not os.path.exists(f["path"]):
            errors.append(f"parent {path}: {f['path']} is missing")
        elif agx.file_id(f["path"]) != f:
            errors.append(f"parent {path}: {f['path']} changed after it was gated")
    for sid in rec.get("source_ids", []):
        if sid not in src["by_id"]:
            errors.append(f"parent {path}: source {sid} is no longer approved")
    for w in rec.get("writers", []):
        if w not in src["writer_ids"]:
            errors.append(f"parent {path}: writer {w} is no longer approved")
    rows = {}
    if rec.get("manifest"):
        lines = open(rec["manifest"]["path"]).read().splitlines()
        head = lines[0].split("\t")
        rows = {r[0]: dict(zip(head, r)) for r in (x.split("\t") for x in lines[1:])}
    return rec, rows, {"path": os.path.abspath(path), "sha256": agx.sha256_bytes(raw), "stage": rec["stage"]}


def load_sources(path):
    raw = open(path, "rb").read()
    s = json.loads(raw)
    s["sha256"], s["path"] = agx.sha256_bytes(raw), os.path.abspath(path)
    s["by_id"] = {x["id"]: x for x in s["sources"]}
    s["writer_ids"] = {w["id"] for w in s["writers"]}
    return s


def writer_ok(src, writer):
    """The writer id is approved and the recorded model file is one of its files."""
    w = next((x for x in src["writers"] if x["id"] == writer.get("id")), None)
    if w is None:
        return f"writer {writer.get('id')!r} not approved"
    if w.get("files") and not any(f["name"] == writer.get("file") and f["bytes"] == writer.get("bytes")
                                  for f in w["files"]):
        return f"model file {writer.get('file')} ({writer.get('bytes')} bytes) is not one of {w['id']}'s files"
    return None


def texts_ok(texts, src):
    errs = [writer_ok(src, texts["writer"])]
    for k in ("tools", "params", "messages"):
        errs += [f"texts: {k} {i} has no text" for i, t in texts.get(k, {}).items() if not t.get("text")]
    if texts.get("spec", {}).get("version") != 2:
        errs.append("texts: not written from texts_spec.json version 2 (facts only, copy check, Flint's judge)")
    missing = sorted(set(agx.DEFAULT_MESSAGES) - set(texts.get("messages", {})))
    if missing:
        errs.append(f"texts: no tool messages for {missing}")
    return [e for e in errs if e]


def ws_lines(root):
    """Normalized lines of every text file in a workspace folder (licence and copyright lines may be quoted from
    these)."""
    out = set()
    for rel in agx.tree_files(root):
        try:
            for line in open(os.path.join(root, rel), encoding="utf-8"):
                out.add(" ".join(line.split()))
        except UnicodeDecodeError:
            pass
    return out


def foreign_licence_lines(text, allowed_lines):
    bad = []
    for line in text.splitlines():
        if LICENCE_LINE.search(line) and " ".join(line.split()) not in allowed_lines:
            fams = license_scan.families_in(line)
            if fams or re.search(r"SPDX-License-Identifier|Copyright\b|\(c\)\s|©", line, re.I):
                bad.append(line.strip()[:120])
    return bad


def foreign_paths(text):
    return sorted({p for p in ABS_PATH.findall(text) if not p.startswith(ALLOWED_PATHS)})[:5]


def ex_changed_ok(root, f, man, texts_by_sha):
    """An Elixir module file is its scanned original with the Stdmods prefix added and texts.json's change note as its
    first line, and nothing else."""
    cur = open(os.path.join(root, f["path"]), encoding="utf-8").read()
    if agx.sha256_file(f["origin"]) != f["origin_sha256"]:
        return [f"{f['path']}: its original {f['origin']} changed"]
    orig = open(f["origin"], encoding="utf-8").read()
    first, _, rest = cur.partition("\n")
    bad = []
    if f"{agx.EX_NAMESPACE}." in orig or rest.replace(f"{agx.EX_NAMESPACE}.", "") != orig:
        bad.append(f"{f['path']}: differs from its original by more than the {agx.EX_NAMESPACE} prefix")
    cn = man.get("change_note") or {}
    T = texts_by_sha.get(cn.get("texts_sha256"))
    tp = None if T is None else next((x for x in T["templates"].get("change_note", []) if x["id"] == cn.get("id")),
                                     None)
    if tp is None:
        bad.append(f"{f['path']}: its change note is not in the given texts files")
    elif first + "\n" != agx.ex_change_note(tp["text"], cn["source_file"], man["modules"]):
        bad.append(f"{f['path']}: the first line is not the change note")
    return bad


def ex_found(name, text):
    """Whether an Elixir test or function name shows in the workspace's text: a test's name (ExUnit puts a describe
    block's name in front of it: both parts must show), a function's name (Stdmods.URI.parse/1, doctest
    Stdmods.URI.parse/1 (3)), a module's last part (doctest module Stdmods.URI (1))."""
    if name.startswith("test "):
        n = name[len("test "):]
        return n in text or any(n[:i].strip() in text and n[i:].strip() in text
                                for i in range(1, len(n)) if n[i] == " ")
    parts = name.split(" ")
    if name.startswith("doctest "):
        leaf = parts[2].split(".")[-1] if parts[1] == "module" else parts[1].rsplit(".", 1)[-1].split("/")[0]
    else:
        leaf = name.rsplit(".", 1)[-1].split("/")[0]
    return leaf in text


def stage_workspaces(a, src, cb, errors, rows, rec, parents):
    texts_by_sha = load_texts(a, src, errors, rec) if a.texts else {}   # Elixir change notes
    scans = {}
    for s in a.scan or []:
        with open(s) as f:
            head = f.readline().rstrip("\n").split("\t")
            scans.update({r[0]: dict(zip(head, r)) for r in (x.rstrip("\n").split("\t") for x in f)})
    spans = agx.heldout_spans(a.q51, a.agent_tasks)
    ids = set()
    for path in a.files:
        man = json.load(open(path))
        bad = []
        c = cb.get(man["codebase_id"])
        if c is None or not str(c.get("status", "")).startswith("approved"):
            bad.append(f"code base {man['codebase_id']} not approved ({c and c.get('status')})")
        if man["source_id"] not in src["by_id"]:
            bad.append(f"source {man['source_id']} not in sources.json")
        root = os.path.join(os.path.dirname(path), os.path.basename(path).removesuffix(".manifest.json"))
        disk = agx.tree_files(root)
        if {f["path"]: f["sha256"] for f in man["files"]} != disk:
            bad.append("the folder differs from the manifest")
        ex = man.get("lang") == "elixir"
        for f in man["files"]:
            if ex and f["role"] in ("licence", "notice"):   # pinned texts: the Apache-2.0 text, Elixir's NOTICE
                want = c and c.get(f"{f['role']}_file", {}).get("sha256")
                if not want or f["sha256"] != want:
                    bad.append(f"{f['path']}: not the {f['role']} file codebases.json pins")
            elif f["role"] in ("code", "licence"):
                rel = os.path.relpath(f["origin"], man["origin"].get("lib") or man["origin"].get("tree"))
                row = scans.get(rel)
                if row is None or row["sha256"] != f.get("origin_sha256", f["sha256"]):
                    bad.append(f"{f['path']}: not the scanned file {rel}")
                elif f["role"] == "code" and row["verdict"] != "ok" and not row["verdict"].startswith("ok:notice"):
                    bad.append(f"{f['path']}: scan verdict {row['verdict']}")
                if f["role"] == "code" and f.get("origin_sha256"):
                    bad += ex_changed_ok(root, f, man, texts_by_sha)
                if f["role"] == "code":
                    hit = agx.shared12(open(os.path.join(root, f["path"]), encoding="utf-8").read(), spans)
                    if hit:
                        bad.append(f"{f['path']} shares 12 words with a held-out test: {hit!r}")
            elif f["role"] == "scaffold" and ex:   # mix new's files, as Elixir's generator wrote them
                want = agx.ex_scaffold(f["path"], man)
                tmpl = man["mixnew"]["files"]
                if any(agx.sha256_file(os.path.join(man["mixnew"]["dir"], p)) != s for p, s in tmpl.items()):
                    bad.append("mix new's output changed since the workspaces were built")
                elif want is None or agx.sha256_text(want) != f["sha256"]:
                    bad.append(f"{f['path']}: not mix new's file")
            elif f["role"] == "scaffold" and f["bytes"] != 0:
                bad.append(f"{f['path']}: scaffold file not empty")
        ids.add(man["source_id"])
        rows.append([man["workspace_id"], man["sha256"], man["codebase_id"], man["licence"], str(len(man["files"])),
                     "; ".join(bad) or "ok"])
        errors += [f"{man['workspace_id']}: {b}" for b in bad]
    rec["source_ids"], rec["writers"] = sorted(ids), []
    return ["workspace_id", "manifest_sha256", "codebase_id", "licence", "files", "check"]


def load_texts(a, src, errors, rec):
    """{sha256: texts} of every --texts file (a run may extend its texts between stages)."""
    out = {}
    for path in a.texts:
        t = json.load(open(path))
        errors += texts_ok(t, src)
        out[agx.sha256_file(path)] = t
    rec["texts_json"] = [{"path": os.path.abspath(p), "sha256": agx.sha256_file(p)} for p in a.texts]
    return out


def stage_tasks(a, src, cb, errors, rows, rec, parents):
    by_sha = load_texts(a, src, errors, rec)
    writers_t = {t["writer"]["id"] for t in by_sha.values()}
    spans = agx.heldout_spans(a.q51, a.agent_tasks)
    ws_ok = {r["manifest_sha256"] for p in parents if p[0]["stage"] == "workspaces" for r in p[1].values()
             if r["check"] == "ok"}
    sess_ok = {k: v for p in parents if p[0]["stage"] == "sessions" for k, v in p[1].items() if v["check"] == "ok"}
    ids, writers = set(), set()
    for path in a.files:
        for t in agx.read_jsonl(path):
            bad = []
            man = json.load(open(t["workspace_manifest"]["path"]))
            if man["sha256"] != t["workspace_manifest"]["sha256"]:
                bad.append("manifest changed")
            if t["parent_session"] is None:
                if man["sha256"] not in ws_ok:
                    bad.append("workspace manifest not in a passed workspaces record")
            else:
                if man.get("base_manifest_sha256") not in ws_ok:
                    bad.append("base workspace not in a passed workspaces record")
                srow = sess_ok.get(t["parent_session"])
                if srow is None:
                    bad.append(f"parent session {t['parent_session']} not passed")
                else:   # the file the stage-1 session wrote (the statement may name the doctests' file instead)
                    test_rel = next((f["path"] for f in man["files"] if f["role"] == "tests"),
                                    t["placeholders"]["test_file"])
                    sha = next((f["sha256"] for f in man["files"] if f["path"] == test_rel), None)
                    s = json.load(open(srow["file"]))
                    if s["final_files"].get(test_rel, {}).get("sha256") != sha:
                        bad.append(f"{test_rel} is not the session's file")
                    writers.add(s["writer"]["id"])
            disk = agx.tree_files(t["workspace_dir"])
            if {f["path"]: f["sha256"] for f in man["files"]} != disk:
                bad.append("the folder differs from the manifest")
            T = by_sha.get(t["template"]["texts_sha256"])
            tp = None if T is None else next((x for k in T["templates"] for x in T["templates"][k]
                                              if x["id"] == t["template"]["id"]), None)
            if tp is None:
                bad.append("template not in the given texts files")
            else:
                text = tp["text"]
                for k, v in t["placeholders"].items():
                    text = text.replace("{" + k + "}", v)
                if text != t["statement"]:
                    bad.append("statement is not the filled template")
            for k, v in t["placeholders"].items():
                names = v.split(", ") if k == "failing_tests" else [v]
                for n in names:
                    if k.endswith("_file") and n not in disk and n not in t["verify"].get("exists", []):
                        bad.append(f"{k} {n} not in the workspace")
                    elif k in ("failing_tests", "function") and t.get("lang") == "elixir":
                        text = "\n".join(open(os.path.join(t["workspace_dir"], p), encoding="utf-8",
                                              errors="replace").read() for p in disk if p.endswith((".ex", ".exs")))
                        if not ex_found(n, text):
                            bad.append(f"{k} {n} not found in the workspace")
                    elif k in ("failing_tests", "function"):
                        leaf = n.split(".")[-1]
                        if not any(leaf in open(os.path.join(t["workspace_dir"], p), encoding="utf-8",
                                                errors="replace").read() for p in disk if p.endswith(".py")):
                            bad.append(f"{k} {n} not found in the workspace")
            hit = agx.shared12(t["statement"], spans)
            if hit:
                bad.append(f"statement shares 12 words with a held-out test: {hit!r}")
            ids.add(t["source_id"])
            rows.append([t["task_id"], agx.sha256_text(json.dumps(t, sort_keys=True)), t["type"], t["codebase_id"],
                         t["template"]["id"], t["parent_session"] or "", "; ".join(bad) or "ok"])
            errors += [f"{t['task_id']}: {b}" for b in bad]
    writers |= writers_t
    rec["source_ids"], rec["writers"] = sorted(ids), sorted(writers)
    return ["task_id", "task_sha256", "type", "codebase_id", "template_id", "parent_session", "check"]


def stage_sessions(a, src, cb, errors, rows, rec, parents):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import agent
    by_sha = load_texts(a, src, errors, rec)
    task_rows = {k: v for p in parents if p[0]["stage"] == "tasks" for k, v in p[1].items() if v["check"] == "ok"}
    spans = agx.heldout_spans(a.q51, a.agent_tasks)
    ids, writers, tot = set(), set(), {"sessions": 0, "turns": 0, "generated": 0, "passed": 0}
    files = []
    for p in a.files:
        files += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.endswith(".json")
                  and not f.endswith(".error.json")] if os.path.isdir(p) else [p]
    for path in files:
        s = json.load(open(path))
        t, bad, flags = s["task"], [], []   # bad: the pipeline broke its rules (gate fails); flags: content excluded
        tr = task_rows.get(t["task_id"])
        if tr is None or tr["task_sha256"] != agx.sha256_text(json.dumps(t, sort_keys=True)):
            bad.append("task not in a passed tasks record (or changed)")
        T = by_sha.get(s["texts"]["sha256"])
        if T is None:
            bad.append("its texts.json is not one of the given texts files")
        else:
            if s["tools"] != agent.tool_schema(T, s.get("tools_profile", "agx"), agx.text_variant(t)):
                bad.append("tool schema is not texts.json's")
            if s.get("tool_messages_sha256") != agx.sha256_text(json.dumps(agx.flat_messages(T), sort_keys=True)):
                bad.append("the tool runner's messages are not texts.json's")
            system = next((x["text"] for x in T["system"] if x["id"] == s["texts"]["system_id"]), None)
            if s["messages_start"] != [{"role": "system", "content": system},
                                       {"role": "user", "content": t["statement"]}]:
                bad.append("first messages are not the system text and the statement")
        w = writer_ok(src, s["writer"])
        if w:
            bad.append(w)
        if s.get("template_matches_server") is False:
            bad.append("the chat template rendered differently from the server's")
        # token chain
        n = len(s["prompt_tokens"])
        for turn in s["turns"]:
            if turn["context_len"] != n or not turn["gen_tokens"]:
                bad.append(f"turn {turn['turn']}: token chain broken")
                break
            if "chunk_tokens" in turn:
                n += len(turn["gen_tokens"]) - (1 if turn["stop_id"] is not None else 0) + len(turn["chunk_tokens"])
            else:
                n += len(turn["gen_tokens"])
        if n != s["sequence_len"]:
            bad.append("sequence length does not match the turns")
        # workspace lines that outputs may quote
        root = t["workspace_dir"]
        allowed = ws_lines(root) if os.path.isdir(root) else set()
        for turn in s["turns"]:
            gen_text = turn["text"]
            for c in turn["calls"]:
                r = c["result"]
                for fp in r.get("files_read", []) + r.get("files_written", []):
                    if (fp.startswith("/") and not fp.startswith("/tmp/")) or fp.startswith(".."):   # /tmp: scratch
                        flags.append(f"turn {turn['turn']}: {c['name']} touched {fp}")
                if r.get("infra"):
                    flags.append(f"turn {turn['turn']}: sandbox problem {r['infra'][:80]}")
                written = " ".join(str(v) for v in c["arguments"].values())
                for line in foreign_licence_lines(r["output"], allowed | ws_lines_written(s)):
                    flags.append(f"turn {turn['turn']}: tool output has a licence line not in the workspace: {line!r}")
                fpaths = foreign_paths(r["output"])
                if fpaths:
                    flags.append(f"turn {turn['turn']}: tool output names paths outside the workspace: {fpaths}")
                hit = agx.shared12(r["output"], spans) or agx.shared12(written, spans)
                if hit:
                    flags.append(f"turn {turn['turn']}: shares 12 words with a held-out test: {hit!r}")
            for line in foreign_licence_lines(gen_text, allowed):
                flags.append(f"turn {turn['turn']}: generated licence line not in the workspace: {line!r}")
            hit = agx.shared12(gen_text, spans)
            if hit:
                flags.append(f"turn {turn['turn']}: generated text shares 12 words with a held-out test: {hit!r}")
        gen = sum(len(x["gen_tokens"]) for x in s["turns"])
        tot["sessions"] += 1
        tot["turns"] += len(s["turns"])
        tot["generated"] += gen
        tot["passed"] += bool(s["verify"]["passed"])
        ids.add(t["source_id"])
        writers.add(s["writer"]["id"])
        rows.append([s["session_id"], os.path.abspath(path), agx.sha256_file(path), t["task_id"], t["codebase_id"],
                     t["licence"], s["writer"]["id"], str(len(s["turns"])), str(gen), s["end"],
                     str(s["verify"]["passed"]), "; ".join(dict.fromkeys(bad)) or
                     ("excluded: " + "; ".join(dict.fromkeys(flags)) if flags else "ok")])
        tot["excluded"] = tot.get("excluded", 0) + bool(flags and not bad)
        errors += [f"{s['session_id']}: {b}" for b in dict.fromkeys(bad)]
    rec["source_ids"], rec["writers"], rec["summary"] = sorted(ids), sorted(writers), tot
    return ["session_id", "file", "file_sha256", "task_id", "codebase_id", "licence", "writer", "turns",
            "generated_tokens", "end", "verified", "check"]


def ws_lines_written(s):
    """Lines the session itself wrote into files: a later read of them may quote them back."""
    out = set()
    for turn in s["turns"]:
        for c in turn["calls"]:
            args = c.get("parsed") or c["arguments"]
            texts = [args.get(k) for k in ("content", "new_string")]
            texts += [e.get("newText") for e in args.get("edits") or [] if isinstance(e, dict)]
            for v in texts:
                if isinstance(v, str):
                    out |= {" ".join(x.split()) for x in v.splitlines()}
    return out


def stage_answers(a, src, cb, errors, rows, rec, parents):
    sess = {k: v for p in parents if p[0]["stage"] == "sessions" for k, v in p[1].items() if v["check"] == "ok"}
    cache, ids, writers = {}, set(), set()
    for path in a.files:
        for r in agx.read_jsonl(path):
            bad = []
            row = sess.get(r["session_id"])
            if row is None:
                bad.append("session not passed")
            else:
                if row["file"] not in cache:
                    cache[row["file"]] = json.load(open(row["file"]))
                s = cache[row["file"]]
                if agx.sha256_file(row["file"]) != row["file_sha256"]:
                    bad.append("session file changed")
                seq, ranges = agx.session_sequence(s["prompt_tokens"], s["turns"])
                if "messages" in r:   # turn_prompts.jsonl (gate it in its own record: check_sources reads tokens)
                    import export
                    if r["messages"] != export.messages_of(s, r["turn"]):
                        bad.append("messages differ from the session's conversation before this turn")
                    r = dict(r, prompt_tokens=[], gen_tokens=[], writer=s["writer"]["id"])
                elif "turn" in r:
                    t = s["turns"][r["turn"]]
                    if r["gen_tokens"] != t["gen_tokens"] or r["prompt_tokens"] != seq[:t["context_len"]]:
                        bad.append("tokens differ from the session's turn")
                else:
                    seq = r["prompt_tokens"] + r["gen_tokens"]
                    if r["prompt_tokens"] != s["prompt_tokens"] or len(seq) != s["sequence_len"] \
                            or r["gen_ranges"] != s["gen_ranges"]:
                        bad.append("sequence differs from the session's")
                ids.add(s["task"]["source_id"])
                writers.add(s["writer"]["id"])
                if r.get("writer") != s["writer"]["id"]:
                    bad.append("writer differs")
            rows.append([r["id"], r["writer"] if "writer" in r else "", r.get("category", ""), r.get("source_id", ""),
                         str(len(r["prompt_tokens"])), str(len(r["gen_tokens"])), os.path.basename(path),
                         "; ".join(bad) or "ok"])
            errors += [f"{r['id']}: {b}" for b in bad]
    rec["source_ids"], rec["writers"] = sorted(ids), sorted(writers)
    return ["answer_id", "writer", "category", "source_id", "prompt_tokens", "answer_tokens", "file", "check"]


def stage_dataset(a, src, cb, errors, rows, rec, parents):
    sess = {k: v for p in parents if p[0]["stage"] == "sessions" for k, v in p[1].items() if v["check"] == "ok"}
    spans = agx.heldout_spans(a.q51, a.agent_tasks)
    folder = a.files[0]
    for need in ("LICENSE", "NOTICE", "README.md"):
        if not os.path.exists(os.path.join(folder, need)):
            errors.append(f"{need} missing")
    card = open(os.path.join(folder, "README.md")).read() if os.path.exists(os.path.join(folder, "README.md")) else ""
    ids, writers, lic = set(), set(), set()
    for r in agx.read_jsonl(os.path.join(folder, "data", "tokens.jsonl")):
        if r["session_id"] not in sess:
            errors.append(f"tokens.jsonl: {r['session_id']} is not a passed session")
    for path in [os.path.join(folder, "data", "sessions.jsonl")]:
        for r in agx.read_jsonl(path):
            bad = []
            row = sess.get(r["session_id"])
            if row is None:
                bad.append("session not passed")
            elif r["licence"] != row["licence"] or r["codebase_id"] != row["codebase_id"]:
                bad.append("licence or code base differs from the session's")
            lic.add(r["licence"])
            text = json.dumps(r["messages"], ensure_ascii=False)
            hit = agx.shared12(" ".join(m.get("content") or "" for m in r["messages"]), spans)
            if hit:
                bad.append(f"shares 12 words with a held-out test: {hit!r}")
            m = MACHINE_PATH.search(text)
            if m:
                bad.append(f"machine path in the row: {m[0]!r}")
            ids.add(r["source_id"])
            writers.add(r["writer"])
            rows.append([r["session_id"], r["codebase_id"], r["licence"], os.path.basename(path), "; ".join(bad) or "ok"])
            errors += [f"{r['session_id']}: {b}" for b in bad]
    for spdx in lic:
        if not os.path.exists(os.path.join(folder, "LICENSES", f"{spdx}.txt")):
            errors.append(f"LICENSES/{spdx}.txt missing")
    if not card.startswith("---"):
        errors.append("README.md has no YAML header")
    rec["source_ids"], rec["writers"] = sorted(ids), sorted(writers)
    return ["session_id", "codebase_id", "licence", "file", "check"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["workspaces", "tasks", "sessions", "answers", "dataset"])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--parent", nargs="*", default=[])
    ap.add_argument("--sources", required=True)
    ap.add_argument("--codebases", required=True)
    ap.add_argument("--texts", nargs="*", default=[], help="texts.json files (a run may extend its texts)")
    ap.add_argument("--scan", nargs="*")
    ap.add_argument("--q51")
    ap.add_argument("--agent-tasks")
    ap.add_argument("--out")
    a = ap.parse_args()
    src = load_sources(a.sources)
    cb_raw = open(a.codebases, "rb").read()
    cb = {c["id"]: c for c in json.loads(cb_raw)["codebases"]}
    out = a.out or a.files[0].rstrip("/") + ".gate.json"
    errors, rows = [], []
    rec = {"gate": "agentic_data/gate.py", "stage": a.stage, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "sources_json": {"path": src["path"], "sha256": src["sha256"]},
           "codebases_json": {"path": os.path.abspath(a.codebases), "sha256": agx.sha256_bytes(cb_raw)},
           "heldout": {"q51": a.q51 and agx.file_id(a.q51), "agent_tasks": a.agent_tasks and agx.file_id(a.agent_tasks)}}
    if a.stage not in ("workspaces",) and not a.parent:
        raise SystemExit("--parent: the gate records of this stage's inputs are required")
    if a.stage in ("tasks", "sessions") and not a.texts:
        raise SystemExit("--texts is required")
    if not (a.q51 and a.agent_tasks) and a.stage in ("workspaces", "tasks", "sessions", "dataset"):
        raise SystemExit("--q51 and --agent-tasks (the held-out tests) are required")
    parents = []
    for p in a.parent:
        prec, prows, ref = load_parent(p, src, errors)
        parents.append((prec, prows))
        rec.setdefault("parents", []).append(ref)
    stage = {"workspaces": stage_workspaces, "tasks": stage_tasks, "sessions": stage_sessions,
             "answers": stage_answers, "dataset": stage_dataset}[a.stage]
    head = stage(a, src, cb, errors, rows, rec, parents)
    rec["files"] = [agx.file_id(f) for f in a.files]
    manifest = out.removesuffix(".json") + ".tsv"
    with open(manifest, "w") as f:
        f.write("\t".join(head) + "\n")
        for r in rows:
            f.write("\t".join(str(x).replace("\t", " ").replace("\n", " ") for x in r) + "\n")
    rec["manifest"] = agx.file_id(manifest)
    rec["rows"], rec["failures"], rec["first_failures"] = len(rows), len(errors), errors[:50]
    rec["verdict"] = "FAIL" if errors else "PASS"
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    for r in [r for r in rows if str(r[-1]).startswith("excluded")][:10]:
        print("EXCLUDED:", r[0], r[-1][:300])
    for e in errors[:25]:
        print("FAIL:", e)
    print(f"{rec['verdict']}: {a.stage}, {len(rows)} rows, {len(errors)} failures; gate record {out}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
