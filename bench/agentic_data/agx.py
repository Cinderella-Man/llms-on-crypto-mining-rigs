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
                "ok": False, "output": "error: the tool produced no result", "exit_code": r.returncode,
                "files_read": [], "files_written": [], "truncated": False, "infra": r.stderr[-500:]}
        except subprocess.TimeoutExpired:
            out = {"ok": False, "output": f"error: the tool did not finish within {limit} s", "exit_code": 124,
                   "files_read": [], "files_written": [], "truncated": False, "infra": "timeout"}
        out["ms"] = int((time.time() - t0) * 1000)
        return out

    def stop(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


def cleanup(run_tag):
    ids = subprocess.run(["docker", "ps", "-aq", "--filter", f"label=agx-run={run_tag}"], capture_output=True,
                         text=True).stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
    return len(ids)
