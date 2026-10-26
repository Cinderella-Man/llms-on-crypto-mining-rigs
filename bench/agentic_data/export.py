"""Exports of gated sessions for the three consumers. Reads only the sessions a passed sessions gate record lists as
ok.

  train/turns.jsonl         one row per assistant turn in generate.py's layout (id, category, source, seed,
                            prompt_tokens = the exact context, gen_tokens, stop_type) plus session_id, turn, writer,
                            source_id: for the answer store (ngram_replay/datastore_corpus.py), which takes the
                            generated tokens only, and for check_sources.py's answer-level checks
  train/turn_prompts.jsonl  {id, messages}: each turn's conversation so far, for datastore_corpus.py's Q51 check
  train/sessions_seq.jsonl  one row per session: prompt_tokens = the first prompt, gen_tokens = the rest of the
                            sequence (generated turns and tool results), gen_ranges = the generated [start, end)
                            spans of the whole sequence: for feats.py seqs and llama-staircase-dump, which replay a
                            session once; training reads only the generated spans
  hf/                       the publishable dataset: data/sessions.jsonl (messages, tools, task, code base, licence,
                            writer, verification), data/tokens.jsonl (exact token ids and generated spans), LICENSE
                            (Apache-2.0, for this dataset's own parts), LICENSES/<SPDX id>.txt (every code base's
                            licence text), NOTICE (attributions and change summaries) and README.md (the card)

usage: python export.py SESSIONS.gate.json --codebases codebases.json --apache LICENSE_APACHE_2.0.txt --out DIR
                        [--name NAME] [--turn-id-offset 300000] [--session-id-offset 400000]
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agx  # noqa: E402

STOP = {150001: "eos", 150003: "eos", 150005: "eos", 150018: "eos"}


def messages_of(s, upto=None):
    """OpenAI-style messages of a session (reasoning kept as reasoning_content, as Flint's makers ask), optionally
    only the conversation before turn `upto`."""
    msgs = [dict(m) for m in s["messages_start"]]
    for t in s["turns"]:
        if upto is not None and t["turn"] >= upto:
            break
        m = {"role": "assistant", "reasoning_content": t["reasoning"], "content": t["content"]}
        if t["calls"]:
            m["tool_calls"] = [{"id": c["id"], "type": "function", "function": {
                "name": c["name"], "arguments": json.dumps(c.get("parsed") or c["arguments"], ensure_ascii=False)}}
                               for c in t["calls"]]
        msgs.append(m)
        for c in t["calls"]:
            msgs.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": c["result"]["output"]})
    return msgs


def notice(codebases, used, name, notice_files=None):
    lines = [f"{name}", "",
             "This dataset's own parts (the pipeline's choices and arrangement, the task records and metadata) are "
             "licensed under the Apache License 2.0 (LICENSE). The text the model generated was written by Flint, "
             "Recursive's open-weights model (https://huggingface.co/RecursiveSI/flint, Apache-2.0); the harness "
             "texts (system prompts, tool descriptions, task statement templates) were written by Flint as well.", "",
             "Code from the following projects appears in the sessions: in workspace files the agent read, in test "
             "and command output, in tracebacks, and in the model's edits of those files. That code stays under its "
             "own licence, whose full text is in LICENSES/; its copyright notices are kept in the files. The "
             "projects named here do not endorse this dataset.", ""]
    for cid in sorted(used):
        c = codebases[cid]
        lines += [f"* {c['name']} ({c['origin']})", f"  Licence: {c['licence']} (LICENSES/{c['licence']}.txt)",
                  f"  {c['attribution']}", f"  Changes: {c['changes']}", ""]
        if (notice_files or {}).get(cid):   # the project's own NOTICE file (Apache-2.0 section 4(d)), verbatim
            lines += [f"  {c['name']}'s NOTICE file:", ""]
            lines += ["    " + x if x else "" for x in open(notice_files[cid], encoding="utf-8").read().splitlines()]
            lines += [""]
    lines += ["Per session, the row's task.change field holds the diff this pipeline applied to the workspace before "
              "the session (an injected bug or a removed function body), and the messages show every change the "
              "model made."]
    return "\n".join(lines) + "\n"


CARD = """---
license:
{licences}
pretty_name: {name}
language:
- en
tags:
- agentic
- code
- tool-use
- synthetic
size_categories:
- {size}
---

# {name}

Coding-agent sessions written by [Flint](https://huggingface.co/RecursiveSI/flint) (Apache-2.0) on permissively
licensed code, for training speculative-decoding guessers and as an open record of how the model works through
small software tasks.

- **Sessions:** {n_sessions} ({n_passed} pass their task's own check), {n_turns} assistant turns, {n_gen} generated
  tokens.
- **Code bases:** {codebases}.
- **Task types:** {types}.
- **Writer:** {writer}; sampling {sampling}.

## Provenance

Every session traces to an approved task statement (a template written by Flint, filled with names from the workspace),
an approved code base pinned to an exact version with its licence file and notices kept, the harness's own system
prompt and tool definitions (written by Flint), and the model. Tool results come from running the tools on the
workspace in a sandbox without network access. Each row names its code base, licence, task, template and writer;
the pipeline's gate records chain every row back to these. Code in the sessions stays under its own licence
(LICENSES/, NOTICE).

## Fields

`data/sessions.jsonl`: `session_id`, `task` (type, statement, the pipeline's change as a diff, the check), `codebase_id`,
`source_id`, `licence` (SPDX), `workspace_files` (path, sha256, origin), `writer`, `engine`, `sampling`, `tools`,
`messages` (system, user, assistant with `reasoning_content` and `tool_calls`, tool), `end_reason`, `verified`.
`data/tokens.jsonl`: `session_id`, `token_ids` (the whole session in Flint's tokenizer), `gen_ranges` (the generated
spans).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gate")
    ap.add_argument("--codebases", required=True)
    ap.add_argument("--apache", required=True, help="the Apache License 2.0 text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="flint-agentic-sessions")
    ap.add_argument("--turn-id-offset", type=int, default=300000)
    ap.add_argument("--session-id-offset", type=int, default=400000)
    a = ap.parse_args()
    rec = json.load(open(a.gate))
    if rec["stage"] != "sessions" or rec["verdict"] != "PASS":
        raise SystemExit("the gate record must be a passed sessions record")
    lines = open(rec["manifest"]["path"]).read().splitlines()
    head = lines[0].split("\t")
    ok = [dict(zip(head, x.split("\t"))) for x in lines[1:]]
    ok = [r for r in ok if r["check"] == "ok"]
    cbj = json.load(open(a.codebases))
    codebases = {c["id"]: c for c in cbj["codebases"]}
    writers = cbj.get("writers_published_as", {})
    licence_text = {}   # code base -> its licence file in a session's workspace (the scanned original)
    notice_text = {}    # code base -> its NOTICE file in a session's workspace (Elixir's)
    os.makedirs(os.path.join(a.out, "train"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "hf", "data"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "hf", "LICENSES"), exist_ok=True)
    tid, sidn = a.turn_id_offset, a.session_id_offset
    used, types, stats = set(), {}, {"sessions": 0, "passed": 0, "turns": 0, "gen": 0}
    writer = None
    with open(os.path.join(a.out, "train", "turns.jsonl"), "w") as ft, \
            open(os.path.join(a.out, "train", "turn_prompts.jsonl"), "w") as fp, \
            open(os.path.join(a.out, "train", "sessions_seq.jsonl"), "w") as fs, \
            open(os.path.join(a.out, "hf", "data", "sessions.jsonl"), "w") as fh, \
            open(os.path.join(a.out, "hf", "data", "tokens.jsonl"), "w") as fk:
        for r in ok:
            s = json.load(open(r["file"]))
            if agx.sha256_file(r["file"]) != r["file_sha256"]:
                raise SystemExit(f"{r['file']} changed after the gate")
            t = s["task"]
            seq, ranges = agx.session_sequence(s["prompt_tokens"], s["turns"])
            meta = {"category": "agentic", "source": t["codebase_id"], "source_id": t["source_id"],
                    "writer": s["writer"]["id"], "session_id": s["session_id"]}
            for turn in s["turns"]:
                ft.write(json.dumps({"id": f"agentic_{tid}", **meta, "turn": turn["turn"], "seed": turn["seed"],
                                     "prompt_tokens": seq[:turn["context_len"]], "gen_tokens": turn["gen_tokens"],
                                     "stop_type": turn["stop"]}) + "\n")
                fp.write(json.dumps({"id": f"agentic_{tid}", "session_id": s["session_id"], "turn": turn["turn"],
                                     "messages": messages_of(s, turn["turn"])},
                                    ensure_ascii=False) + "\n")
                tid += 1
            n0 = len(s["prompt_tokens"])
            fs.write(json.dumps({"id": f"agentic_{sidn}", **meta, "prompt_tokens": seq[:n0], "gen_tokens": seq[n0:],
                                 "gen_ranges": ranges, "stop_type": s["turns"][-1]["stop"] if s["turns"] else None})
                     + "\n")
            sidn += 1
            man = json.load(open(t["workspace_manifest"]["path"]))
            task = {k: t[k] for k in ("task_id", "type", "statement", "placeholders", "change", "verify")}
            task["template_id"] = t["template"]["id"]
            fh.write(json.dumps({
                "session_id": s["session_id"], "task": task, "codebase_id": t["codebase_id"],
                "source_id": t["source_id"], "licence": t["licence"],
                "workspace_files": [{k: f.get(k) for k in ("path", "sha256", "role", "origin")
                                     if not (k == "origin" and str(f.get(k, "")).startswith("/srv"))}
                                    for f in man["files"]],
                "writer": s["writer"]["id"], "model": writers.get(s["writer"]["id"]),
                "engine": {k: v for k, v in s["server"].items() if k in ("engine", "build_info", "version")},
                "sampling": s["sampling"], "tools": s["tools"], "messages": messages_of(s),
                "end_reason": s["end"], "verified": bool(s["verify"]["passed"]),
                "turns": len(s["turns"]), "generated_tokens": sum(len(x["gen_tokens"]) for x in s["turns"])},
                ensure_ascii=False) + "\n")
            fk.write(json.dumps({"session_id": s["session_id"], "token_ids": seq, "gen_ranges": ranges}) + "\n")
            used.add(t["codebase_id"])
            lf = next((f["path"] for f in man["files"] if f.get("role") == "licence"), None)
            if lf and t["codebase_id"] not in licence_text:
                licence_text[t["codebase_id"]] = os.path.join(t["workspace_dir"], lf)
            nf = next((f["path"] for f in man["files"] if f.get("role") == "notice"), None)
            if nf and t["codebase_id"] not in notice_text:
                notice_text[t["codebase_id"]] = os.path.join(t["workspace_dir"], nf)
            types[t["type"]] = types.get(t["type"], 0) + 1
            stats["sessions"] += 1
            stats["passed"] += bool(s["verify"]["passed"])
            stats["turns"] += len(s["turns"])
            stats["gen"] += sum(len(x["gen_tokens"]) for x in s["turns"])
            writer = s["writer"]["id"]
    hf = os.path.join(a.out, "hf")
    shutil.copy(a.apache, os.path.join(hf, "LICENSE"))
    lic = {"Apache-2.0"}
    for cid in used:
        c = codebases[cid]
        if cid not in licence_text:
            raise SystemExit(f"no licence file in {cid}'s workspaces")
        shutil.copy(licence_text[cid], os.path.join(hf, "LICENSES", f"{c['licence']}.txt"))
        lic.add(c["licence"])
    shutil.copy(a.apache, os.path.join(hf, "LICENSES", "Apache-2.0.txt"))
    with open(os.path.join(hf, "NOTICE"), "w") as f:
        f.write(notice(codebases, used, a.name, notice_text))
    hf_ids = sorted({"apache-2.0"} | {codebases[c]["hf_licence"] for c in used})
    n = stats["sessions"]
    card = CARD.format(licences="\n".join(f"- {x}" for x in hf_ids), name=a.name,
                       size="n<1K" if n < 1000 else "1K<n<10K" if n < 10000 else "10K<n<100K",
                       n_sessions=n, n_passed=stats["passed"], n_turns=stats["turns"], n_gen=stats["gen"],
                       codebases=", ".join(f"{codebases[c]['name']} ({codebases[c]['licence']})" for c in sorted(used)),
                       types=", ".join(f"{k} {v}" for k, v in sorted(types.items())), writer=writer,
                       sampling="temperature 1.0, top-k 20, top-p 0.95, min-p 0")
    if any(codebases[c]["hf_licence"] == "other" for c in used):
        card = card.replace("pretty_name:", "license_name: python-2.0\nlicense_link: LICENSES/Python-2.0.txt\n"
                                            "pretty_name:", 1)
    with open(os.path.join(hf, "README.md"), "w") as f:
        f.write(card)
    print(json.dumps({"sessions": n, **stats, "turn_rows": tid - a.turn_id_offset, "licences": sorted(lic)}))


if __name__ == "__main__":
    main()
