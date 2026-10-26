"""Shared helpers of the agentic data pipeline (bench/agentic_data): hashes, JSON lines, the 12-word overlap rule,
licence markers in text, Flint's chat template, and the session sandbox. Standard library only, plus jinja2 for the
chat template (agent.py)."""
import hashlib
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import license_scan  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX_IMAGE = "agx-sandbox:py312"
EX_IMAGE = "agx-sandbox:ex118"   # sandbox/Dockerfile.elixir


def _tool_runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location("agx_tool_runner", os.path.join(HERE, "sandbox", "tools.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DEFAULT_MESSAGES = _tool_runner().DEFAULT_MESSAGES   # the pipeline's own wording, outside sessions only


def text_variant(task):
    """The texts.json prefix of the run_tests texts a task's test runner needs: "ex" (mix test), "pytest", or None
    (python3 -m unittest)."""
    if task.get("lang") == "elixir":
        return "ex"
    return "pytest" if "pytest" in " ".join(task.get("test_command") or []) else None


def flat_messages(texts):
    """{id: line} of the tool runner's messages in texts.json (written by Flint); every id the tool runner uses."""
    m = {k: v["text"] for k, v in (texts.get("messages") or {}).items() if v.get("text")}
    missing = sorted(set(DEFAULT_MESSAGES) - set(m))
    if missing:
        raise SystemExit(f"texts.json has no tool messages for {missing} (agent.py texts with texts_spec.json v2)")
    return m


def ngrams(text, n=5):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def shared_ngram(text, grams, n=5):
    """The first run of n words that text shares with grams (a set of ngrams()), or None."""
    w = re.findall(r"[a-z0-9]+", text.lower())
    for i in range(len(w) - n + 1):
        g = " ".join(w[i:i + n])
        if g in grams:
            return g
    return None
LABEL = "pascal-quants=1"
END_IDS = [150001, 150003, 150005, 150018]   # <|end_of_text|>, <|user|>, <|observation|>, <|developer|>
OBSERVATION_ID = 150005
SAMPLING = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0}   # the makers' sampling


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_text(s):
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def file_id(path):
    """Size and sha256 of a file, or of a folder (its files' relative paths and hashes)."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        h, n = hashlib.sha256(), 0
        for d, subdirs, names in sorted(os.walk(path)):
            subdirs.sort()
            for name in sorted(names):
                fp = os.path.join(d, name)
                h.update(f"{os.path.relpath(fp, path)}\0{sha256_file(fp)}\0".encode())
                n += os.path.getsize(fp)
        return {"path": path, "bytes": n, "dir_sha256": h.hexdigest()}
    return {"path": path, "bytes": os.path.getsize(path), "sha256": sha256_file(path)}


def tree_files(root):
    """{relative path: sha256} of every file under root (no __pycache__)."""
    out = {}
    for d, subdirs, names in os.walk(root):
        subdirs[:] = sorted(s for s in subdirs if s != "__pycache__")
        for n in sorted(names):
            fp = os.path.join(d, n)
            out[os.path.relpath(fp, root)] = sha256_file(fp)
    return out


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def words(text):
    return text.lower().split()


def spans12(texts, n=12):
    spans = set()
    for t in texts:
        w = words(t)
        spans |= {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    return spans


def shared12(text, spans, n=12):
    """The first n-word span of text that is in spans, or None."""
    w = words(text)
    for i in range(len(w) - n + 1):
        s = " ".join(w[i:i + n])
        if s in spans:
            return s
    return None


def heldout_spans(q51_prompts=None, agent_check_tasks=None):
    """The 12-word spans of everything the project's own tests use: Q51's prompts (all message contents) and the agent
    check's task statements and workspace files (bench/agent_check/tasks). Nothing generated may share one."""
    texts = []
    if q51_prompts:
        for r in read_jsonl(q51_prompts):
            texts += [m.get("content") or "" for m in r["messages"]]
    if agent_check_tasks:
        for d, _, names in os.walk(agent_check_tasks):
            for n in names:
                try:
                    texts.append(open(os.path.join(d, n), encoding="utf-8").read())
                except (UnicodeDecodeError, OSError):
                    pass
    return spans12(texts)


def apply_at(src, edits):
    """src with each (line, column, old, new) edit applied: lines and columns count from 1, columns in characters.
    An edit whose old text is not at its place is skipped; a dropped `not` takes the spaces after it along.
    Returns (new source, number applied)."""
    lines = src.split("\n")
    n = 0
    by_line = {}
    for e in edits:
        by_line.setdefault(e[0], []).append(e)
    for ln, es in by_line.items():
        s = lines[ln - 1]
        for _, col, old, new in sorted(es, key=lambda e: -e[1]):   # right to left: earlier columns stay valid
            i = col - 1
            if s[i:i + len(old)] != old:
                continue
            j = i + len(old)
            if old == "not" and new == "":
                j += len(s[j:]) - len(s[j:].lstrip(" "))
            s = s[:i] + new + s[j:]
            n += 1
        lines[ln - 1] = s
    return "\n".join(lines), n


# ---- Elixir workspaces (workspace.py elixir): the Mix project around a module comes from Elixir's own generator,
# `mix new stdmods --module Stdmods` in the sandbox; this pipeline writes no file text of its own

EX_NAMESPACE = "Stdmods"
EX_MIXNEW = ["mix.exs", ".formatter.exs", "test/test_helper.exs"]   # copied unchanged
EX_DOCTEST = "test/stdmods_test.exs"   # mix new's test file, its sample test left out and its doctest line naming
#                                        the workspace's modules


def ex_doctest_from(template, modules):
    """mix new's test file with the workspace's modules in its doctest line and the generated sample test left out:
    its first two lines (defmodule, use ExUnit.Case), one doctest line per module, its last line (end)."""
    lines = template.rstrip("\n").split("\n")
    if len(lines) < 4 or not lines[2].strip().startswith("doctest ") or lines[-1] != "end":
        raise ValueError("not the test file mix new writes")
    indent = lines[2][:len(lines[2]) - len(lines[2].lstrip())]
    return "\n".join(lines[:2] + [f"{indent}doctest {m}" for m in modules] + [lines[-1]]) + "\n"


def ex_scaffold(path, man):
    """The content this pipeline put at path in the Elixir workspace of manifest man (from its mix new output), or
    None."""
    d = man.get("mixnew", {}).get("dir")
    if not d:
        return None
    if path in EX_MIXNEW:
        return open(os.path.join(d, path), encoding="utf-8").read()
    if path == man.get("doctest_file"):
        return ex_doctest_from(open(os.path.join(d, EX_DOCTEST), encoding="utf-8").read(), man["modules"])
    return None


def ex_change_note(template, source_file, modules):
    """The first line of a renamed module file: an Elixir comment holding the texts.json change note (written by Flint
    or the owner) with its placeholders filled. Apache-2.0 section 4(b) asks changed files to say they were changed."""
    text = template.replace("{source_file}", source_file).replace("{modules}", ", ".join(modules))
    return "# " + " ".join(text.split()) + "\n"


EX_SUMMARY = re.compile(r"^(?:(\d+) doctests?, )?(?:(\d+) tests?, )?(\d+) failures?(?:, (\d+) invalid)?", re.M)
EX_FAIL = re.compile(r"^\s+\d+\) ((?:test|doctest) .+) \(([\w.]+)\)\s*$", re.M)
EX_BROKEN = re.compile(r"== Compilation error|\*\* \((?:CompileError|SyntaxError|TokenMissingError|"
                       r"MismatchedDelimiterError)")


def ex_counts(output):
    """(tests run, failures) from ExUnit's summary line ("35 doctests, 6 tests, 1 failure"); (0, None) without one.
    Invalid tests (a failed setup) count as failures."""
    m = None
    for m in EX_SUMMARY.finditer(output):
        pass
    if m is None:
        return 0, None
    d, t, f, inv = (int(x or 0) for x in m.groups())
    return d + t, f + inv


def ex_failing(output):
    """The failed tests' names as ExUnit prints them ("test parses a switch", "doctest Stdmods.URI.parse/1 (3)")."""
    return sorted({name for name, _ in EX_FAIL.findall(output)})


RESTRICTED = license_scan.RESTRICTED


def licence_markers(text):
    """Licence families named in a text (license_scan's signatures). Restricted ones fail a gate; permissive ones
    must also occur in the session's own project files."""
    return license_scan.families_in(text)


# ---- Flint's chat template (the Hugging Face chat_template.jinja, rendered as transformers renders it) ----

def load_template(path):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    env.filters["tojson"] = tojson
    tmpl = env.from_string(open(path).read())

    def render(messages, tools=None, add_generation_prompt=True):
        msgs = []
        for m in messages:   # the template iterates arguments.items(): string arguments become objects
            if m.get("tool_calls"):
                m = dict(m, tool_calls=[dict(tc, function=dict(tc["function"], arguments=json.loads(tc["function"]["arguments"])))
                                        if isinstance(tc["function"].get("arguments"), str) else tc
                                        for tc in m["tool_calls"]])
            msgs.append(m)
        return tmpl.render(messages=msgs, tools=tools, add_generation_prompt=add_generation_prompt,
                           bos_token="<|begin_of_text|>", eos_token="<|end_of_text|>")
    return render


TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


def parse_reply(text):
    """(reasoning, content, tool calls, problems) from one generated turn. Calls inside the reasoning don't count."""
    reasoning, rest, problems = "", text, []
    if "</think>" in text:
        pre, rest = text.split("</think>", 1)
        reasoning = pre[len("<think>"):] if pre.startswith("<think>") else pre
        if "<tool_call>" in reasoning:
            problems.append("tool call inside the reasoning")
    elif text.lstrip().startswith("<think>"):
        return text.lstrip()[len("<think>"):], "", [], ["reasoning not closed"]
    calls = []
    for m in TOOL_CALL.finditer(rest):
        body = m.group(1)
        name = body.split("<arg_key>", 1)[0].strip()
        args = {k.strip(): v for k, v in ARG.findall(body)}
        calls.append({"name": name, "arguments": args})
    content = TOOL_CALL.sub("", rest)
    if "<tool_call>" in content:
        problems.append("unterminated tool call")
        content = content.split("<tool_call>", 1)[0]
    return reasoning, content.strip(), calls, problems


def session_sequence(prompt_ids, turns):
    """The whole session as one token sequence, and the [start, end) ranges of generated tokens in it (what the
    drafter trains on). A turn that called tools ends with its stop token, which is also the first token of the tool
    results (<|observation|>); when the model stopped with another end token, the template's token takes its place
    and that position is not counted as generated."""
    seq, ranges = list(prompt_ids), []
    for t in turns:
        s = len(seq)
        if "chunk_tokens" in t:
            body = t["gen_tokens"][:-1] if t["stop_id"] is not None else t["gen_tokens"]
            seq += body + t["chunk_tokens"]
            ranges.append([s, s + len(body) + (1 if t.get("stop_matches_chunk") else 0)])
        else:
            seq += t["gen_tokens"]
            ranges.append([s, s + len(t["gen_tokens"])])
    return seq, ranges


# ---- the sandbox: one container per session (sandbox/Dockerfile) ----

class Sandbox:
    def __init__(self, name, workdir, image=SANDBOX_IMAGE, run_tag="agx", config=None, memory="2g"):
        self.name, self.workdir, self.image, self.run_tag = name, os.path.abspath(workdir), image, run_tag
        self.config = config or {}
        self.memory = memory

    def start(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        argv = ["docker", "run", "-d", "--name", self.name, "--label", LABEL, "--label", f"agx-run={self.run_tag}",
                "--cpu-shares", "256", "--network", "none", "--memory", self.memory, "--pids-limit", "512",
                "--read-only", "--tmpfs", "/tmp:rw,exec,size=512m", "-v", f"{self.workdir}:/work", "-w", "/work",
                self.image, "sleep", "infinity"]
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"sandbox {self.name} did not start: {r.stderr.strip()}")
        return self

    def tool(self, name, args, timeout=None):
        req = json.dumps({"tool": name, "args": args, "config": self.config})
        limit = (self.config.get("timeout") or 120) + 30 if timeout is None else timeout
        t0 = time.time()
        try:
            r = subprocess.run(["docker", "exec", "-i", self.name, "python3", "-I", "/opt/agx/tools.py"], input=req,
                               capture_output=True, text=True, timeout=limit)
            out = json.loads(r.stdout) if r.stdout.strip() else {
                "ok": False, "output": "error: " + self.line("no_result"), "exit_code": r.returncode,
                "files_read": [], "files_written": [], "truncated": False, "infra": r.stderr[-500:]}
        except subprocess.TimeoutExpired:
            out = {"ok": False, "output": "error: " + self.line("timeout", seconds=limit), "exit_code": 124,
                   "files_read": [], "files_written": [], "truncated": False, "infra": "timeout"}
        out["ms"] = int((time.time() - t0) * 1000)
        return out

    def line(self, key, **values):
        """One of the tool runner's messages: the session's (texts.json) or the default."""
        text = (self.config.get("messages") or DEFAULT_MESSAGES)[key]
        for k, v in values.items():
            text = text.replace("{" + k + "}", str(v))
        return text

    def ex_tool(self, cmd, path, timeout=120):
        """The Elixir helper (sandbox/ex_tool.exs) on a file of the workspace: its JSON reply. Not a model tool."""
        r = subprocess.run(["docker", "exec", self.name, "elixir", "/opt/agx/ex_tool.exs", cmd, path],
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"ex_tool {cmd} {path}: {r.stderr.strip()[-500:]}")
        return json.loads(r.stdout.strip().splitlines()[-1])

    def stop(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


def cleanup(run_tag):
    ids = subprocess.run(["docker", "ps", "-aq", "--filter", f"label=agx-run={run_tag}"], capture_output=True,
                         text=True).stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
    return len(ids)
