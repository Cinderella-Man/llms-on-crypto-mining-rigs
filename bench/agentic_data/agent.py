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
PLACEHOLDERS = {"write_tests": ["{module_file}", "{test_file}", "{test_command}"],
                "fix_bug": ["{test_file}", "{failing_tests}", "{module_file}"],
                "implement": ["{function}", "{module_file}", "{test_file}"],
                "change_note": ["{source_file}", "{modules}"]}   # the first line of a changed file (workspace.py)


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


def tool_schema(texts, profile="agx", lang=None):
    """The tools as the model sees them: fixed names and parameters, descriptions from texts.json (keys of the pi
    profile carry the prefix "pi:"; a language's own descriptions, e.g. run_tests with mix test, the prefix "ex:")."""
    pre = "" if profile == "agx" else profile + ":"
    lp = agx.LANG_PREFIX.get(lang)

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


def coerce(call, profile="agx"):
    """The call's arguments with integer and array parameters converted; (args, error text or None)."""
    spec = PROFILES[profile].get(call["name"])
    if spec is None:
        return None, f"error: unknown tool {call['name']!r}"
    params, required = spec
    args = {}
    for k, v in call["arguments"].items():
        if k not in params:
            return None, f"error: {call['name']} has no parameter {k!r}"
        if params[k] == "integer":
            try:
                v = int(str(v).strip())
            except ValueError:
                return None, f"error: {k} must be an integer"
        elif params[k] == "edits":
            try:
                v = json.loads(v) if isinstance(v, str) else v
            except ValueError:
                return None, f"error: {k} must be a JSON array of objects with oldText and newText"
        args[k] = v
    missing = [k for k in required if k not in args]
    if missing:
        return None, f"error: missing parameter {', '.join(missing)}"
    return args, None


# ---------------------------------------------------------------- texts: Flint writes the harness's own texts

def cmd_texts(a):
    spec_raw = open(a.spec, "rb").read()
    spec = json.loads(spec_raw)
    server = Server(a.server, a.api)
    render = agx.load_template(a.template)
    spans = agx.heldout_spans(a.heldout_q51, a.heldout_tasks)
    def param_request(k, w):   # "pi:edit.edits.oldText" -> tool edit, parameter oldText
        tool, param = k.split(":")[-1].split(".")[0], k.split(".")[-1]
        return spec["param_request"].replace("{tool}", tool).replace("{param}", param).replace("{what}", w)
    items = [("system", f"system-{v}", spec["system"], "") for v in range(spec["variants"]["system"])]
    items += [("system", f"system-pi-{v}", spec["system_pi"], "") for v in range(spec["variants"].get("system_pi", 0))]
    for tools_key, params_key, pre in (("tool", "param", ""), ("tool_pi", "param_pi", "pi:")):
        items += [("tool", pre + k, spec["tool_request"].replace("{tool}", k).replace("{what}", w), w)
                  for k, w in spec.get(tools_key, {}).items()]
        items += [("param", pre + k, param_request(k, w), w) for k, w in spec.get(params_key, {}).items()]
    for kind in PLACEHOLDERS:
        for v in range(spec["variants"].get(kind, 0)):
            req = spec[kind].replace("{tone}", spec["tones"][v % len(spec["tones"])]) \
                .replace("{length}", spec["lengths"][v % len(spec["lengths"])])
            items.append((kind, f"{kind}-{v}", req, ""))
    for lang, ls in spec.get("langs", {}).items():   # a language's own tool descriptions and templates
        lp = agx.LANG_PREFIX[lang]
        items += [("tool", f"{lp}:{k}", spec["tool_request"].replace("{tool}", k).replace("{what}", w), w, lang)
                  for k, w in ls.get("tool", {}).items()]
        items += [("param", f"{lp}:{k}", param_request(k, w), w, lang) for k, w in ls.get("param", {}).items()]
        for kind in PLACEHOLDERS:
            for v in range(ls.get("variants", {}).get(kind, 0)):
                req = ls[kind].replace("{tone}", spec["tones"][v % len(spec["tones"])]) \
                    .replace("{length}", spec["lengths"][v % len(spec["lengths"])])
                items.append((kind, f"{lp}-{kind}-{v}", req, "", lang))
    items = [x if len(x) == 5 else x + (None,) for x in items]
    old = json.load(open(a.extend)) if a.extend else None
    if old:   # keep every text already written; write only the missing ones
        have = {s["id"] for s in old["system"]} | set(old["tools"]) | set(old["params"]) | \
            {t["id"] for k in old["templates"] for t in old["templates"][k]}
        items = [x for x in items if x[1] not in have]

    def valid(kind, text, what=""):
        if not text or any(s in text for s in ("<tool_call>", "<think>", "</think>", "<|")):
            return "empty or markup"
        n = len(text.split())
        if kind == "system" and not 40 <= n <= 260:
            return f"{n} words"
        if kind == "tool" and n > 40 or kind == "param" and n > 20 or kind == "change_note" and n > 40:
            return f"{n} words"
        if kind == "change_note" and "\n" in text.strip():
            return "more than one line"
        if what:   # a description must be about its tool or parameter: half of the request's content words
            want = {w[:5] for w in re.findall(r"[a-z]{4,}", what.lower())} - {"optio", "defau", "relat", "proje"}
            have = {w[:5] for w in re.findall(r"[a-z]{4,}", text.lower())}
            if want and len(want & have) < max(1, len(want) // 2):
                return "not about the requested tool or parameter"
        if kind in PLACEHOLDERS:
            for p in PLACEHOLDERS[kind]:
                if text.count(p) != 1:
                    return f"placeholder {p} {text.count(p)} times"
            if set(re.findall(r"\{[^{}]*\}", text)) - set(PLACEHOLDERS[kind]):
                return "other braces"
        elif "{" in text or "}" in text:
            return "braces"
        if agx.shared12(text, spans):
            return "shares 12 words with a held-out test"
        return None

    def one(i, item):
        kind, key, req, what, lang = item
        ids = server.tokenize(render([{"role": "user", "content": req}], None))
        tries = []
        for t in range(a.tries):
            seed = a.seed + i * 10 + t
            g = server.generate(ids, a.max_gen, seed)
            _, content, _, _ = agx.parse_reply(g["text"])
            text = content.strip().strip('"').strip()
            bad = valid(kind, text, what) if g["stop"] == "eos" else f"stop {g['stop']}"
            tries.append({"seed": seed, "gen_tokens": len(g["gen"]), "problem": bad})
            if not bad:
                return {"id": key, "kind": kind, "text": text, "seed": seed,
                        **({"profile": "pi" if key.startswith("system-pi") else "agx"} if kind == "system" else {}),
                        **({"lang": lang} if lang else {}),
                        "request_sha256": agx.sha256_text(req),
                        "gen_tokens_sha256": agx.sha256_text(json.dumps(g["gen"])), "tries": tries}
        return {"id": key, "kind": kind, "text": None, "tries": tries, **({"lang": lang} if lang else {})}

    with concurrent.futures.ThreadPoolExecutor(a.conc) as ex:
        res = list(ex.map(lambda x: one(*x), enumerate(items)))
    out = {"about": "The harness's natural-language texts, written by Flint from texts_spec.json's requests "
                    "(agent.py texts). Only these texts, filled with names from the workspace, enter a session.",
           "writer": writer_record(a), "server": server.info(), "sampling": agx.SAMPLING,
           "spec": {"file": os.path.basename(a.spec), "sha256": agx.sha256_bytes(spec_raw)},
           "system": [r for r in res if r["kind"] == "system" and r["text"]],
           "tools": {r["id"]: r for r in res if r["kind"] == "tool"},
           "params": {r["id"]: r for r in res if r["kind"] == "param"},
           "templates": {k: [r for r in res if r["kind"] == k and r["text"]] for k in PLACEHOLDERS},
           "failed": [r for r in res if not r["text"]]}
    if old:
        out["system"] = old["system"] + out["system"]
        out["tools"] = {**old["tools"], **out["tools"]}
        out["params"] = {**old["params"], **out["params"]}
        out["templates"] = {k: old["templates"].get(k, []) + out["templates"][k] for k in PLACEHOLDERS}
        out["extended_from"] = {"file": os.path.basename(a.extend), "sha256": agx.sha256_file(a.extend),
                                "writer": old["writer"], "server": old["server"]}
    agx.write_json(a.out, out)
    print(f"{sum(1 for r in res if r['text'])} of {len(res)} texts written; failed: "
          f"{[r['id'] for r in out['failed']]}")
    if any(not r["text"] for r in res if r["kind"] in ("tool", "param")):
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
    tools = tool_schema(texts, profile, task.get("lang"))
    system = rng.choice([s for s in texts["system"] if s.get("profile", "agx") == profile])
    messages = [{"role": "system", "content": system["text"]}, {"role": "user", "content": task["statement"]}]
    prompt_text = render(messages, tools)
    prompt_ids = server.tokenize(prompt_text)
    srv_text = server.apply_template(messages, tools)
    sb = agx.Sandbox(f"agx-{a.run_tag}-{slot}-{agx.sha256_text(sid)[:10]}", work, task.get("image") or a.image, a.run_tag,
                     config={"test_command": task["test_command"], "timeout": 120, "max_output": 20000,
                             "max_lines": 2000}).start()
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
                args, err = coerce(c, profile)
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
        for lang in {t.get("lang") for t in tasks}:
            tool_schema(texts, p, lang)   # every description the run needs is in texts.json
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
