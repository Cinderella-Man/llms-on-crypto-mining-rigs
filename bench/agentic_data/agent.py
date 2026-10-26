"""The session generator: our own agent loop, Flint as the model, each session in its own sandbox container.

Nothing in a session's context comes from anywhere but these: the system prompt and tool descriptions of texts.json
(written by Flint or the owner, never by another vendor's model), the task statement (a texts.json template filled
with names from the workspace), Flint's own tokens, and tool results produced by the sandbox from the task's
workspace (sandbox/tools.py). The context is kept as exact token ids: the first prompt is Flint's chat template
rendered here and tokenized by the server; every later turn appends the generated ids and the tokenized tool results,
the way the template would render them. So the whole session is one append-only token sequence, which the drafter's
dump can replay once, and every generated id is the one the server sampled.

usage:
  python agent.py texts --spec texts_spec.json --out texts.json --server URL --api llama|vllm --template T.jinja
                        --writer ID --model-file GGUF [--seed 7100] [--conc 8] [--heldout-q51 Q51.jsonl]
                        [--heldout-tasks DIR] [--extend OLD_TEXTS.json]
  python agent.py run --tasks tasks.jsonl --texts texts.json --out DIR --work DIR --server URL --api llama|vllm
                      --template T.jinja --writer ID --model-file GGUF [--conc 8] [--max-turns 24] [--max-gen 8192]
                      [--max-ctx 32768] [--time-cap-min 30] [--seed 60] [--run-tag agx] [--stop-file PATH]
                      [--tools agx|pi|mixed] [--repeat 1] [--claims DIR]
Output of run: DIR/sessions/<task_id>.json per session (see SESSION FORMAT in README.md), DIR/run.log. Resumable:
finished sessions are skipped; a session that failed for infrastructure reasons is written as .error.json and run
again next time. Create the stop file to start no new sessions.
"""
import argparse
import concurrent.futures
import json
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agx  # noqa: E402

HARNESS_VERSION = "agentic_data/agent.py 2026-10-26"
PROFILES = {   # tool names and parameter types are fixed here; every description comes from texts.json
    "agx": {     # our own tool set
        "list_dir": ({"path": "string"}, []),
        "read_file": ({"path": "string", "start_line": "integer", "end_line": "integer"}, ["path"]),
        "search": ({"pattern": "string", "path": "string"}, ["pattern"]),
        "edit_file": ({"path": "string", "old_string": "string", "new_string": "string"},
                      ["path", "old_string", "new_string"]),
        "write_file": ({"path": "string", "content": "string"}, ["path", "content"]),
        "run_tests": ({"target": "string"}, []),
        "run_command": ({"command": "string"}, ["command"]),
    },
    "pi": {      # the names and parameters of the pi agent's four default tools, the harness the rig's agentic speed
                 # is measured with: the store and the drafter learn the tool calls the deployed agent makes
        "read": ({"path": "string", "offset": "integer", "limit": "integer"}, ["path"]),
        "bash": ({"command": "string", "timeout": "integer"}, ["command"]),
        "edit": ({"path": "string", "edits": "edits"}, ["path", "edits"]),
        "write": ({"path": "string", "content": "string"}, ["path", "content"]),
    },
}
TOOL_PARAMS = PROFILES["agx"]


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


class Server:
    def __init__(self, url, api, model="flint"):
        self.url, self.api, self.model = url.rstrip("/"), api, model

    def info(self):
        try:
            if self.api == "llama":
                p = get(self.url + "/props")
                return {"engine": "llama.cpp llama-server", "build_info": p.get("build_info"),
                        "model_file": os.path.basename(p.get("model_path") or ""), "n_ctx": p.get("default_generation_settings", {}).get("n_ctx")}
            return {"engine": "vllm", "version": get(self.url + "/version").get("version")}
        except (urllib.error.URLError, OSError, ValueError) as e:
            return {"engine": self.api, "error": str(e)}

    def tokenize(self, text):
        if self.api == "llama":
            return post(self.url + "/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]
        return post(self.url + "/tokenize", {"model": self.model, "prompt": text, "add_special_tokens": False})["tokens"]

    def apply_template(self, messages, tools):
        if self.api != "llama":
            return None
        body = {"messages": messages}
        if tools:
            body["tools"] = tools
        return post(self.url + "/apply-template", body)["prompt"]

    def generate(self, ids, n, seed, slot=None):
        t0 = time.time()
        if self.api == "llama":
            body = {"prompt": ids, "n_predict": n, "seed": seed, "return_tokens": True, "cache_prompt": True,
                    "stream": False, **agx.SAMPLING}
            if slot is not None:
                body["id_slot"] = slot
            r = post(self.url + "/completion", body)
            gen, stop = list(r["tokens"]), r.get("stop_type")
            stop = "eos" if stop == "eos" else ("limit" if stop == "limit" else stop)
            return {"gen": gen, "text": r.get("content", ""), "stop": stop, "ms": int((time.time() - t0) * 1000),
                    "timings": {k: r.get("timings", {}).get(k) for k in ("prompt_n", "prompt_ms", "predicted_n",
                                                                        "predicted_ms", "cache_n")}}
        r = post(self.url + "/v1/completions", {"model": self.model, "prompt": ids, "max_tokens": n, "seed": seed,
                                                "stop_token_ids": agx.END_IDS, "return_token_ids": True,
                                                "skip_special_tokens": False, **agx.SAMPLING})
        ch = r["choices"][0]
        gen = list(ch["token_ids"])
        if ch["finish_reason"] == "length":
            stop = "limit"
        elif ch["finish_reason"] == "stop":
            stop = "eos"
            if not gen or gen[-1] not in agx.END_IDS:
                gen.append(int(ch["stop_reason"]))
        else:
            stop = ch["finish_reason"]
        text = ch.get("text", "")
        for s in ("<|end_of_text|>", "<|user|>", "<|observation|>", "<|developer|>"):
            text = text.removesuffix(s)
        return {"gen": gen, "text": text, "stop": stop, "ms": int((time.time() - t0) * 1000), "timings": {}}


def writer_record(a):
    rec = {"id": a.writer, "file": os.path.basename(a.model_file), "bytes": os.path.getsize(a.model_file)}
    for side in (a.model_file + ".verified.json",):
        try:
            v = json.load(open(side))
            rec["sha256"] = v.get("sha256")
        except (OSError, ValueError):
            pass
    return rec


def tool_schema(texts, profile="agx", lp=None):
    """The tools as the model sees them: fixed names and parameters, descriptions from texts.json (keys of the pi
    profile carry the prefix "pi:"; a test runner's own run_tests texts the prefix agx.text_variant gives, e.g. "ex:"
    for mix test, "pytest:")."""
    pre = "" if profile == "agx" else profile + ":"

    def text(kind, key):
        own = f"{lp}:{key}" if lp else None
        return texts[kind][own if own in texts[kind] else key]["text"]
    tools = []
    for name, (params, required) in PROFILES[profile].items():
        props = {}
        for p, t in params.items():
            d = text("params", f"{pre}{name}.{p}")
            if t == "edits":
                item = {k: {"type": "string", "description": text("params", f"{pre}{name}.{p}.{k}")}
                        for k in ("oldText", "newText")}
                props[p] = {"type": "array", "description": d,
                            "items": {"type": "object", "properties": item, "required": ["oldText", "newText"]}}
            else:
                props[p] = {"type": t, "description": d}
        tools.append({"type": "function", "function": {"name": name, "description": text("tools", pre + name),
                                                       "parameters": {"type": "object", "properties": props,
                                                                      "required": required}}})
    return tools


def coerce(call, profile="agx", messages=None):
    """The call's arguments with integer and array parameters converted; (args, error text or None). The error lines
    are texts.json's messages (the tool runner's defaults outside sessions)."""
    def err(key, **values):
        text = (messages or agx.DEFAULT_MESSAGES)[key]
        for k, v in values.items():
            text = text.replace("{" + k + "}", str(v))
        return None, "error: " + text
    spec = PROFILES[profile].get(call["name"])
    if spec is None:
        return err("unknown_tool", name=call["name"])
    params, required = spec
    args = {}
    for k, v in call["arguments"].items():
        if k not in params:
            return err("no_param", tool=call["name"], param=k)
        if params[k] == "integer":
            try:
                v = int(str(v).strip())
            except ValueError:
                return err("not_integer", param=k)
        elif params[k] == "edits":
            try:
                v = json.loads(v) if isinstance(v, str) else v
            except ValueError:
                return err("not_edits", param=k)
        args[k] = v
    missing = [k for k in required if k not in args]
    if missing:
        return err("missing_param", params=", ".join(missing))
    return args, None


# ---------------------------------------------------------------- texts: Flint writes the harness's own texts

def spec_items(spec):
    """(kind, id, instruction, facts, lang, profile, about) for every text texts_spec.json (version 2) asks for.
    `about` is a line the writer reads before the facts and the judge does not (which tool a parameter belongs to)."""
    V, tones, lengths = spec["variants"], spec["tones"], spec["lengths"]
    items = []
    for prof, facts in spec["system"].items():
        for v in range(V["system"] if prof == "agx" else V.get(f"system_{prof}", 0)):
            items.append(("system", f"system-{v}" if prof == "agx" else f"system-{prof}-{v}", "system", facts, None,
                          prof))
    for prof, tools in spec["tools"].items():
        pre = "" if prof == "agx" else prof + ":"
        items += [("tool", pre + name, "tool", {"tool": name, **f}, None, None) for name, f in tools.items()]
        for key, f in spec["params"][prof].items():   # "edit.edits.oldText": tool edit, parameter edits.oldText
            tool, param = key.split(".", 1)
            items.append(("param", pre + key, "param", f, None, None, f"The parameter {param} of the tool {tool}."))
    for r in spec["runners"].values():   # run_tests per test runner: unittest, pytest:, ex: (mix test)
        pre = r["prefix"] + ":" if r["prefix"] else ""
        items.append(("tool", pre + "run_tests", "tool", {"tool": "run_tests", **r["tool"]}, None, None))
        items.append(("param", pre + "run_tests.target", "param", r["target"], None, None,
                      "The parameter target of the tool run_tests."))
    for lang, tasks in spec["tasks"].items():
        pre = tasks.get("prefix")
        for kind, f in tasks.items():
            if kind == "prefix":
                continue
            for v in range(V.get(kind, 0)):
                facts = f if kind == "change_note" else dict(f, tone=tones[v % len(tones)],
                                                            length=lengths[v % len(lengths)])
                items.append((kind, f"{pre}-{kind}-{v}" if pre else f"{kind}-{v}",
                              "note" if kind == "change_note" else "template", facts,
                              None if lang == "python" else lang, None))
    items += [("message", mid, "message", f, None, None) for mid, f in spec["messages"].items()]
    return [x if len(x) == 7 else x + (None,) for x in items]


def authored_grams(spec, extra=()):
    """The 5-word runs of what Flint is shown that this pipeline's authors wrote: the spec's instructions and facts
    (and extra request lines). Copying means reusing words of the request; a run shared by chance with texts Flint
    never sees (texts_spec_v1.json, the tool runner's default messages) is not copying, and texts_report.py shows
    those overlaps for review. Strings shorter than five words carry no run."""
    out = [s for s in extra if len(s.split()) >= 5]

    def walk(x):
        if isinstance(x, str):
            if len(x.split()) >= 5:
                out.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(spec)
    grams = set()
    for s in out:
        grams |= agx.ngrams(s)
    return grams, agx.sha256_text(json.dumps(sorted(out)))


LIMITS = {"system": (40, 260), "tool": (3, 40), "param": (2, 20), "template": (5, 160), "note": (5, 40),
          "message": (1, 25)}   # words


def cmd_texts(a):
    """Flint writes every text of texts_spec.json (version 2) from its facts: tries with other seeds, a judge pass by
    Flint against the facts, and for a text that only failed the copy check, Flint rewording its own text."""
    spec_raw = open(a.spec, "rb").read()
    spec = json.loads(spec_raw)
    if spec.get("version") != 2:
        raise SystemExit(f"{a.spec}: texts_spec.json version 2 expected")
    I = spec["instructions"]
    server = Server(a.server, a.api)
    render = agx.load_template(a.template)
    spans = agx.heldout_spans(a.heldout_q51, a.heldout_tasks)
    items = spec_items(spec)
    grams, grams_sha = authored_grams(spec, [x[6] for x in items if x[6]])
    if a.only:   # a few texts by id (a trial, or a rerun into a separate file)
        items = [x for x in items if x[1] in set(a.only.split(","))]
    old = json.load(open(a.extend)) if a.extend else None
    if old:   # keep every text already written; write only the missing ones (and the ones that failed)
        def written(d):
            return {k for k, v in d.items() if v.get("text")}
        have = {s["id"] for s in old["system"]} | written(old["tools"]) | written(old["params"]) | \
            {t["id"] for k in old["templates"] for t in old["templates"][k]} | written(old.get("messages", {}))
        items = [x for x in items if x[1] not in have]

    def check(instr, text, facts, about=None):
        if not text or any(s in text for s in ("<tool_call>", "<think>", "</think>", "<|")):
            return "empty or markup"
        lo, hi = LIMITS[instr]
        n = len(text.split())
        if not lo <= n <= hi:
            return f"{n} words"
        if instr in ("tool", "param", "note", "message") and "\n" in text:
            return "more than one line"
        want = sorted(set(re.findall(r"\{[a-z_]+\}", json.dumps(facts))))
        for p in want:
            if text.count(p) != 1:
                return f"placeholder {p} {text.count(p)} times"
        if set(re.findall(r"\{[^{}]*\}", text)) - set(want):
            return "braces that are not its placeholders"
        idents = set()   # the facts' identifiers (tool and parameter names, examples), all a text may write with "_"

        def values(x):
            if isinstance(x, dict):
                for v in x.values():
                    values(v)
            elif isinstance(x, list):
                for v in x:
                    values(v)
            elif isinstance(x, str):
                idents.add(x)
                idents.update(re.findall(r"\w+", x))
        values(facts)
        values(about or "")   # the tool and parameter names of a parameter's request line
        stray = [w for w in re.findall(r"(?<![{\w])[A-Za-z]+(?:_[A-Za-z0-9]+)+(?![\w}])", text) if w not in idents]
        if stray:
            return f"names that are not its facts' identifiers: {stray[:3]}"
        wrap = facts.get("wrapped in")
        if wrap:
            left, right = wrap.split()
            if not (text.startswith(left) and text.endswith(right)):
                return f"not wrapped in {wrap}"
        if instr == "message" and re.search(r"bracket|parenthes|wrapped|\bstyle\b", text, re.I):
            return "describes its own format"
        for label in (facts.get("tone"), facts.get("length")):
            if instr == "template" and label and label.lower() in text.lower():
                return f"names its tone or length: {label!r}"
        hit = agx.shared_ngram(text, grams)
        if hit:
            return f"copies: {hit!r}"
        if agx.shared12(text, spans):
            return "shares 12 words with a held-out test"
        return None

    def ask(content, seed):
        ids = server.tokenize(render([{"role": "user", "content": content}], None))
        g = server.generate(ids, a.max_gen, seed)
        _, text, _, _ = agx.parse_reply(g["text"])
        return g, text.strip().strip('"').strip("`").strip()

    def judge(facts, text, seed):
        g, reply = ask(f"{I['judge']}\n\nFacts:\n{json.dumps(facts, indent=1, ensure_ascii=False)}\n\nText:\n{text}",
                       seed)
        first = (reply.splitlines() or [""])[0].strip().upper()
        verdict = "ok" if first.startswith("NO") else "problem" if first.startswith("YES") else "unclear"
        return verdict, reply[:300]

    def one(i, item):
        kind, key, instr, facts, lang, prof, about = item
        request = f"{I[instr]}\n\n{about + chr(10) if about else ''}{json.dumps(facts, indent=1, ensure_ascii=False)}"
        tries, copied = [], None
        steps = [("write", request, t) for t in range(a.tries)]
        k = 0
        while k < len(steps):
            how, content, t = steps[k]
            k += 1
            seed = a.seed + i * 100 + k
            g, text = ask(content, seed)
            bad = check(instr, text, facts, about) if g["stop"] == "eos" else f"stop {g['stop']}"
            verdict = reason = None
            if not bad:
                verdict, reason = judge(facts, text, seed + 50)
                bad = None if verdict == "ok" else f"judge: {verdict}: {reason[:150]}"
            tries.append({"how": how, "seed": seed, "gen_tokens": len(g["gen"]), "problem": bad})
            if not bad:
                return {"id": key, "kind": kind, "text": text, "seed": seed, "how": how,
                        **({"profile": prof} if prof else {}), **({"lang": lang} if lang else {}),
                        "facts_sha256": agx.sha256_text(json.dumps(facts, sort_keys=True)),
                        "request_sha256": agx.sha256_text(content), "judge": reason,
                        "gen_tokens_sha256": agx.sha256_text(json.dumps(g["gen"])), "tries": tries}
            if bad.startswith("copies") and copied is None and how == "write":
                copied = text   # the first text that failed only by copying: Flint rewords it, a.tries times
            if k == len(steps) and copied is not None and steps[-1][0] == "write":
                steps += [("rephrase", f"{I['rephrase']}\n\n{copied}", t) for t in range(a.tries)]
        return {"id": key, "kind": kind, "text": None, "tries": tries, **({"lang": lang} if lang else {})}

    done = []

    def logged(i, item):
        r = one(i, item)
        done.append(r)
        with open(a.out + ".partial.jsonl", "a") as f:   # every finished text at once, so a stopped run loses none
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        why = f": {r['text'][:200]!r}" if r["text"] else \
            " (" + " | ".join(t["problem"][:160] for t in r["tries"][-2:]) + ")"
        print(f"{time.strftime('%H:%M:%S')} {len(done)}/{len(items)} {r['id']}: "
              f"{'ok' if r['text'] else 'FAILED'} after {len(r['tries'])} tries{why}", file=sys.stderr, flush=True)
        return r
    with concurrent.futures.ThreadPoolExecutor(a.conc) as ex:
        res = list(ex.map(lambda x: logged(*x), enumerate(items)))
    kinds = sorted({x[0] for x in spec_items(spec)} - {"system", "tool", "param", "message"})
    out = {"about": "The harness's texts, written by Flint from the facts in texts_spec.json (agent.py texts): "
                    "system prompts, tool and parameter descriptions, statement templates, change notes and the tool "
                    "runner's messages. Only these texts, filled with names from the workspace, enter a session.",
           "writer": writer_record(a), "server": server.info(), "sampling": agx.SAMPLING,
           "spec": {"file": os.path.basename(a.spec), "sha256": agx.sha256_bytes(spec_raw), "version": 2},
           "checks": {"copy_run_words": 5, "authored_texts_sha256": grams_sha, "judge": "Flint, against the facts"},
           "system": [r for r in res if r["kind"] == "system" and r["text"]],
           "tools": {r["id"]: r for r in res if r["kind"] == "tool"},
           "params": {r["id"]: r for r in res if r["kind"] == "param"},
           "templates": {k: [r for r in res if r["kind"] == k and r["text"]] for k in kinds},
           "messages": {r["id"]: r for r in res if r["kind"] == "message"},
           "failed": [r for r in res if not r["text"]]}
    if old:
        out["system"] = old["system"] + out["system"]
        out["tools"] = {**old["tools"], **out["tools"]}
        out["params"] = {**old["params"], **out["params"]}
        out["messages"] = {**old.get("messages", {}), **out["messages"]}
        out["templates"] = {k: old["templates"].get(k, []) + out["templates"].get(k, [])
                            for k in sorted(set(old["templates"]) | set(out["templates"]))}
        out["extended_from"] = {"file": os.path.basename(a.extend), "sha256": agx.sha256_file(a.extend),
                                "writer": old["writer"], "server": old["server"]}
    agx.write_json(a.out, out)
    print(f"{sum(1 for r in res if r['text'])} of {len(res)} texts written "
          f"({sum(1 for r in res if r.get('how') == 'rephrase')} reworded); failed: {[r['id'] for r in out['failed']]}")
    if any(not r["text"] for r in res if r["kind"] in ("tool", "param", "message")):
        sys.exit(1)


# ---------------------------------------------------------------- run: sessions

def verify(task, sb, work, start):
    """The task's own check, after the session: files that must stay unchanged, then the check command."""
    v = task["verify"]
    now = agx.tree_files(work)
    changed = [p for p in v.get("unchanged", []) if now.get(p) != start.get(p)]
    res = {"unchanged_ok": not changed, "changed": changed}
    r = sb.tool("run_tests", {"target": v.get("target", "")})
    res["exit_code"] = r["exit_code"]
    if task.get("lang") == "elixir":
        res["tests_run"] = agx.ex_counts(r["output"])[0]
    else:
        m = re.search(r"Ran (\d+) tests?", r["output"])
        res["tests_run"] = int(m.group(1)) if m else 0
    res["output_tail"] = r["output"][-1500:]
    docs_ok = True
    if v.get("docs_sha256"):   # Elixir: the module's documentation, and so its doctests, as it was
        try:
            res["docs_sha256"] = sb.ex_tool("docs", task["change"]["file"])["sha256"]
        except (RuntimeError, ValueError) as e:   # the module no longer parses
            res["docs_sha256"] = None
            res["docs_error"] = str(e)[:300]
        docs_ok = res["docs_sha256"] == v["docs_sha256"]
        res["docs_ok"] = docs_ok
    res["passed"] = (not changed and docs_ok and r["exit_code"] == 0 and res["tests_run"] >= v.get("min_tests", 1)
                     and all(now.get(p) for p in v.get("exists", [])))
    if v.get("exists") and not all(now.get(p) for p in v["exists"]):
        res["missing"] = [p for p in v["exists"] if not now.get(p)]
    return res


def final_files(work, start, limit=200000):
    """Files the session created or changed, with their content (text, up to limit characters each)."""
    out = {}
    for p, sha in agx.tree_files(work).items():
        if start.get(p) != sha:
            try:
                out[p] = {"sha256": sha, "content": open(os.path.join(work, p), encoding="utf-8").read()[:limit]}
            except UnicodeDecodeError:
                out[p] = {"sha256": sha, "content": None}
    out["__deleted__"] = sorted(set(start) - set(agx.tree_files(work)))
    return out


def run_session(task, a, server, render, texts, slot, writer, info):
    sid = task.get("_session_id", task["task_id"])
    task = {k: v for k, v in task.items() if k != "_session_id"}
    work = os.path.join(a.work, sid)
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(task["workspace_dir"], work)
    start = agx.tree_files(work)
    rng = random.Random(f"{a.seed}:{sid}")
    profile = rng.choice(["agx", "pi"]) if a.tools == "mixed" else a.tools
    tools = tool_schema(texts, profile, agx.text_variant(task))
    tool_lines = agx.flat_messages(texts)   # the tool runner's own lines, written by Flint
    system = rng.choice([s for s in texts["system"] if s.get("profile", "agx") == profile])
    messages = [{"role": "system", "content": system["text"]}, {"role": "user", "content": task["statement"]}]
    prompt_text = render(messages, tools)
    prompt_ids = server.tokenize(prompt_text)
    srv_text = server.apply_template(messages, tools)
    sb = agx.Sandbox(f"agx-{a.run_tag}-{slot}-{agx.sha256_text(sid)[:10]}", work, task.get("image") or a.image, a.run_tag,
                     config={"test_command": task["test_command"], "timeout": 120, "max_output": 20000,
                             "max_lines": 2000, "messages": tool_lines}).start()
    context, turns, end, t0 = list(prompt_ids), [], None, time.time()
    seed0 = a.seed * 1000 + int(agx.sha256_text(sid)[:6], 16) % 100000 * 100
    last_calls = []
    try:
        for turn in range(a.max_turns):
            room = a.max_ctx - len(context)
            if room < 512:
                end = "context"
                break
            g = server.generate(context, min(a.max_gen, room), seed0 + turn, slot)
            gen = g["gen"]
            reasoning, content, calls, problems = agx.parse_reply(g["text"])
            rec = {"turn": turn, "context_len": len(context), "seed": seed0 + turn, "gen_tokens": gen,
                   "stop": g["stop"], "stop_id": gen[-1] if gen and gen[-1] in agx.END_IDS else None,
                   "text": g["text"], "reasoning": reasoning, "content": content, "problems": problems, "calls": [],
                   "gen_ms": g["ms"], "timings": g["timings"]}
            turns.append(rec)
            if g["stop"] != "eos" or not calls:
                end = "answer" if g["stop"] == "eos" else "limit"
                break
            tool_msgs = []
            for i, c in enumerate(calls[:a.max_calls]):
                cid = f"call_{turn}_{i}"
                args, err = coerce(c, profile, tool_lines)
                res = sb.tool(c["name"], args) if err is None else {
                    "ok": False, "output": err, "exit_code": 1, "files_read": [], "files_written": [],
                    "truncated": False, "ms": 0}
                rec["calls"].append({"id": cid, "name": c["name"], "arguments": c["arguments"], "parsed": args,
                                     "result": res})
                tool_msgs.append({"role": "tool", "tool_call_id": cid, "content": res["output"]})
            if len(calls) > a.max_calls:
                rec["problems"].append(f"{len(calls)} calls, {a.max_calls} run")
            chunk = server.tokenize(render(tool_msgs, None))
            rec["chunk_tokens"] = chunk
            rec["stop_matches_chunk"] = bool(gen) and gen[-1] == chunk[0]
            context = context + (gen[:-1] if rec["stop_id"] is not None else gen) + chunk
            sig = json.dumps([(c["name"], c["arguments"]) for c in calls], sort_keys=True)
            last_calls = (last_calls + [sig])[-4:]
            if len(last_calls) == 4 and len(set(last_calls)) == 1:
                end = "loop"
                break
            if time.time() - t0 > a.time_cap_min * 60:
                end = "time"
                break
        else:
            end = "turns"
        check = verify(task, sb, work, start)
        files = final_files(work, start)
    finally:
        sb.stop()
    seq, ranges = agx.session_sequence(prompt_ids, turns)
    sess = {"session_id": sid, "harness": HARNESS_VERSION, "task": task, "writer": writer, "server": info,
            "sampling": agx.SAMPLING, "texts": {"path": os.path.abspath(a.texts), "sha256": agx.sha256_file(a.texts),
                                               "system_id": system["id"]},
            "tools_profile": profile, "lang": task.get("lang", "python"), "tools": tools,
            "tool_messages_sha256": agx.sha256_text(json.dumps(tool_lines, sort_keys=True)),
            "tools_sha256": agx.sha256_text(json.dumps(tools, sort_keys=True)),
            "messages_start": messages, "prompt_tokens": prompt_ids,
            "template_matches_server": None if srv_text is None else srv_text == prompt_text,
            "turns": turns, "end": end, "verify": check, "final_files": files,
            "sequence_len": len(seq), "gen_ranges": ranges, "seconds": round(time.time() - t0, 1),
            "limits": {"max_turns": a.max_turns, "max_gen": a.max_gen, "max_ctx": a.max_ctx,
                       "time_cap_min": a.time_cap_min, "max_calls": a.max_calls}}
    shutil.rmtree(work, ignore_errors=True)
    return sess


def cmd_run(a):
    os.makedirs(os.path.join(a.out, "sessions"), exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    if a.claims:
        os.makedirs(a.claims, exist_ok=True)
    texts = json.load(open(a.texts))
    tasks = agx.read_jsonl(a.tasks)
    for p in ([a.tools] if a.tools != "mixed" else list(PROFILES)):
        for v in {agx.text_variant(t) for t in tasks}:
            tool_schema(texts, p, v)   # every description the run needs is in texts.json
    agx.flat_messages(texts)           # and every line of the tool runner
    render = agx.load_template(a.template)
    server = Server(a.server, a.api)
    info = server.info()
    writer = writer_record(a)
    done = {f.removesuffix(".json") for f in os.listdir(os.path.join(a.out, "sessions")) if f.endswith(".json")
            and not f.endswith(".error.json")}
    # --repeat K runs each task K times (other seeds, other system prompts): sessions <task_id>-r<k>
    runs = [dict(t, _session_id=t["task_id"] if a.repeat == 1 else f"{t['task_id']}-r{k}")
            for k in range(a.repeat) for t in tasks]
    todo = [t for t in runs if t["_session_id"] not in done]
    slots = queue.Queue()
    for i in range(a.conc):
        slots.put(i)
    lock = threading.Lock()
    stat = {"done": 0, "gen": 0, "err": 0, "t0": time.time()}
    log = open(os.path.join(a.out, "run.log"), "a")

    def say(msg):
        with lock:
            log.write(time.strftime("%H:%M:%S ") + msg + "\n")
            log.flush()

    say(f"start: {len(todo)} of {len(tasks)} tasks, {a.conc} at a time, server {info}")

    def job(task):
        if a.stop_file and os.path.exists(a.stop_file):
            return
        if a.claims:   # several writers share one task list: each task is claimed by an exclusive create
            try:
                os.close(os.open(os.path.join(a.claims, task["_session_id"]), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                return
        slot = slots.get()
        path = os.path.join(a.out, "sessions", task["_session_id"])
        try:
            s = run_session(task, a, server, render, texts, slot, writer, info)
            agx.write_json(path + ".json", s)
            if os.path.exists(path + ".error.json"):
                os.remove(path + ".error.json")
            n = sum(len(t["gen_tokens"]) for t in s["turns"])
            with lock:
                stat["done"] += 1
                stat["gen"] += n
            say(f"{task['_session_id']}: {len(s['turns'])} turns, {n} generated, end {s['end']}, verified "
                f"{s['verify']['passed']}; total {stat['done']} sessions, {stat['gen']} generated tokens, "
                f"{(time.time() - stat['t0']) / 60:.1f} min")
        except Exception as e:  # noqa: BLE001 - infrastructure failure: logged, retried on the next run
            with lock:
                stat["err"] += 1
            agx.write_json(path + ".error.json", {"session_id": task["_session_id"], "error": repr(e)})
            if a.claims:
                os.remove(os.path.join(a.claims, task["_session_id"]))   # back to the pool
            say(f"{task['_session_id']}: ERROR {e!r}")
        finally:
            slots.put(slot)

    try:
        with concurrent.futures.ThreadPoolExecutor(a.conc) as ex:
            list(ex.map(job, todo))
    finally:
        n = agx.cleanup(a.run_tag)
        say(f"end: {stat['done']} sessions, {stat['gen']} generated tokens, {stat['err']} errors, "
            f"{n} containers removed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("texts", "run"):
        p = sub.add_parser(name)
        p.add_argument("--server", required=True)
        p.add_argument("--api", choices=["llama", "vllm"], default="llama")
        p.add_argument("--template", required=True, help="Flint's chat_template.jinja (the Hugging Face folder)")
        p.add_argument("--writer", required=True, help="the writer's id in sources.json")
        p.add_argument("--model-file", required=True, help="the model file the server loaded (name and size recorded)")
        p.add_argument("--conc", type=int, default=8)
        p.add_argument("--out", required=True)
        if name == "texts":
            p.add_argument("--spec", required=True)
            p.add_argument("--seed", type=int, default=7100)
            p.add_argument("--tries", type=int, default=4)
            p.add_argument("--max-gen", type=int, default=6144)
            p.add_argument("--heldout-q51")
            p.add_argument("--heldout-tasks")
            p.add_argument("--extend", help="an earlier texts.json: keep its texts, write only the missing ones")
            p.add_argument("--only", help="comma-separated text ids: write only these")
        else:
            p.add_argument("--tasks", required=True)
            p.add_argument("--texts", required=True)
            p.add_argument("--work", required=True)
            p.add_argument("--image", default=agx.SANDBOX_IMAGE)
            p.add_argument("--max-turns", type=int, default=24)
            p.add_argument("--max-calls", type=int, default=8, help="tool calls run per turn")
            p.add_argument("--max-gen", type=int, default=8192)
            p.add_argument("--max-ctx", type=int, default=32768)
            p.add_argument("--time-cap-min", type=float, default=30)
            p.add_argument("--seed", type=int, default=60)
            p.add_argument("--run-tag", default="agx")
            p.add_argument("--tools", choices=["agx", "pi", "mixed"], default="agx",
                           help="tool set: ours, the pi agent's names and parameters, or one of them per session")
            p.add_argument("--stop-file")
            p.add_argument("--claims", help="folder of task claims shared by several writers")
            p.add_argument("--repeat", type=int, default=1, help="sessions per task (other seeds and system prompts)")
    a = ap.parse_args()
    {"texts": cmd_texts, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
