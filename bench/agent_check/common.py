"""Shared pieces of the agent check: tasks, the sandbox around pi and the tests, the hidden-test check, and the parser
for llama-server's streamed responses (used by proxy.py and replay.py).

Inside every sandbox the same paths hold, so pi's system prompt (which names the working directory) is identical for
every task, and the same commands run on the laptop (bwrap) and on the quant machine (Docker):

    /work           the attempt's workspace (writable)
    /session        pi's session folder (writable)
    /home/agent     HOME (private, empty)
    /opt/pi-agent   the attempt's copy of the check's pi agent directory (models.json, settings.json, check.ts;
                    writable, since pi locks its settings and opens auth.json there)
    /opt/pi         the pi release (node_modules/@earendil-works/pi-coding-agent/...)
    /opt/node       node (bin/node, bin/npm)
    /opt/ac-node    the shared node_modules for jest (env/package.json)
    /opt/ac-py      the shared Python environment with pytest (env/requirements.txt; a relocatable venv)
"""
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = HERE / "tasks" / "polyglot"
PI_CLI_REL = "node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
SANDBOX_PATH = "/opt/ac-py/bin:/opt/node/bin:/usr/local/bin:/usr/bin:/bin"
IGNORE = shutil.ignore_patterns("node_modules", "__pycache__", ".pytest_cache", ".ac-*")


# ---------------------------------------------------------------- tasks

def load_tasks(ids=None):
    """All tasks in run order (tasks/polyglot/order.txt), or the given ids in the given order."""
    order = [l.strip() for l in (TASKS / "order.txt").read_text().splitlines() if l.strip()]
    ids = ids or order
    out = []
    for i in ids:
        meta = json.loads((TASKS / i / "task.json").read_text())
        meta["dir"] = TASKS / i
        out.append(meta)
    return out


def make_workspace(task, dest, node_modules, solution_from=None, hidden=False):
    """Copy a task's workspace to dest. solution_from: a directory whose files replace the workspace's (reference/);
    hidden: put the full hidden tests in place of the visible ones; node_modules: the path the JavaScript workspace's
    node_modules symlink points to, as seen inside the sandbox."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(task["dir"] / "workspace", dest)
    if solution_from:
        for p in Path(solution_from).iterdir():
            shutil.copy(p, dest / p.name)
    if hidden:
        put_hidden_tests(task, dest)
    if task["track"] == "javascript":
        os.symlink(node_modules, dest / "node_modules")
    return dest


def put_hidden_tests(task, dest):
    for p in (task["dir"] / "hidden").iterdir():
        shutil.copy(p, dest / p.name)


def diff_workspace(task, work):
    """Unified diff of the agent's changes against the task's workspace."""
    r = subprocess.run(["diff", "-ruN", "--exclude=node_modules", "--exclude=__pycache__", "--exclude=.pytest_cache",
                        "--exclude=.ac-*", str(task["dir"] / "workspace"), str(work)],
                       capture_output=True, text=True, errors="replace")
    return r.stdout


def tests_untouched(task, work):
    tf = task["test_file"]
    a, b = task["dir"] / "workspace" / tf, Path(work) / tf
    return b.exists() and a.read_bytes() == b.read_bytes()


# ---------------------------------------------------------------- sandbox

class Sandbox:
    """Wraps a command so it runs with the fixed in-sandbox paths of the module docstring.

    kind "bwrap" (the laptop: pi and its tools see only these paths), "docker" (the quant machine: one container per
    command, image built from docker/Dockerfile), or "none" (no isolation, host paths; for the mock self-test only).
    """

    def __init__(self, kind, node_dir=None, pi_root=None, ac_node=None, ac_py=None, image=None, label=None,
                 prefix="ac"):
        self.kind = kind
        self.node_dir, self.pi_root, self.ac_node, self.ac_py = node_dir, pi_root, ac_node, ac_py
        self.image, self.label, self.prefix = image, label, prefix
        if kind in ("bwrap", "none"):
            for name in ("node_dir", "pi_root", "ac_node", "ac_py"):
                if not getattr(self, name) or not Path(getattr(self, name)).exists():
                    raise SystemExit(f"sandbox {kind}: --{name.replace('_', '-')} missing or not found")

    @classmethod
    def from_args(cls, a):
        return cls(a.sandbox, a.node_dir, a.pi_root, a.ac_node, a.ac_py, a.image, a.docker_label, a.docker_prefix)

    @staticmethod
    def add_args(ap):
        ap.add_argument("--sandbox", choices=["bwrap", "docker", "none"], default="bwrap")
        ap.add_argument("--node-dir", help="node install root (bin/node); bwrap/none")
        ap.add_argument("--pi-root", help="pi release folder holding node_modules/@earendil-works; bwrap/none")
        ap.add_argument("--ac-node", help="shared node folder holding node_modules (env/package.json); bwrap/none")
        ap.add_argument("--ac-py", help="shared Python venv with pytest (env/requirements.txt); bwrap/none")
        ap.add_argument("--image", default="agent-check:0.87.1", help="docker image (docker/Dockerfile)")
        ap.add_argument("--docker-label", default="pascal-quants=1", help="label on every container")
        ap.add_argument("--docker-prefix", default="ac", help="name prefix of the containers, for cleanup")

    def p(self, name):
        """In-sandbox path of a fixed location."""
        if self.kind != "none":
            return {"pi": "/opt/pi", "node": "/opt/node", "ac_node": "/opt/ac-node", "ac_py": "/opt/ac-py"}[name]
        return str(Path({"pi": self.pi_root, "node": self.node_dir, "ac_node": self.ac_node,
                         "ac_py": self.ac_py}[name]).resolve())

    def node_modules(self):
        return f"{self.p('ac_node')}/node_modules"

    def path_env(self):
        if self.kind != "none":
            return SANDBOX_PATH
        return f"{self.p('ac_py')}/bin:{self.p('node')}/bin:/usr/local/bin:/usr/bin:/bin"

    def pi_cmd(self):
        return [f"{self.p('node')}/bin/node", f"{self.p('pi')}/{PI_CLI_REL}"]

    def wrap(self, cmd, work, session=None, agent_dir=None, env=None, net=True, name=None, home=None):
        """Return (argv, cwd, popen_env). work/session/agent_dir are host paths. home: host folder used as HOME by
        the "none" sandbox."""
        env = dict(env or {})
        work = Path(work).resolve()
        if self.kind == "none":
            full = {"PATH": self.path_env(), "HOME": str(home or work), "LANG": "C.UTF-8", "TERM": "dumb"}
            if agent_dir:
                full["PI_CODING_AGENT_DIR"] = str(Path(agent_dir).resolve())
            full.update(env)
            return cmd, str(work), full
        base_env = {"PATH": SANDBOX_PATH, "HOME": "/home/agent", "LANG": "C.UTF-8", "TERM": "dumb"}
        if agent_dir:
            base_env["PI_CODING_AGENT_DIR"] = "/opt/pi-agent"
        base_env.update(env)
        if self.kind == "bwrap":
            argv = ["bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-ipc",
                    "--unshare-uts", "--clearenv",
                    "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc"]
            for d in ("bin", "lib", "lib64", "sbin"):
                if Path("/" + d).is_symlink():
                    argv += ["--symlink", os.readlink("/" + d), "/" + d]
                elif Path("/" + d).exists():
                    argv += ["--ro-bind", "/" + d, "/" + d]
            argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/home/agent",
                     "--ro-bind", str(Path(self.node_dir).resolve()), "/opt/node",
                     "--ro-bind", str(Path(self.pi_root).resolve()), "/opt/pi",
                     "--ro-bind", str(Path(self.ac_node).resolve()), "/opt/ac-node",
                     "--ro-bind", str(Path(self.ac_py).resolve()), "/opt/ac-py",
                     "--bind", str(work), "/work"]
            if session:
                argv += ["--bind", str(Path(session).resolve()), "/session"]
            if agent_dir:
                argv += ["--bind", str(Path(agent_dir).resolve()), "/opt/pi-agent"]
            if not net:
                argv += ["--unshare-net"]
            argv += ["--chdir", "/work"]
            for k, v in base_env.items():
                argv += ["--setenv", k, v]
            return argv + cmd, None, None
        # docker
        argv = ["docker", "run", "--rm", "-i", "--init", "--label", self.label,
                "--network", "host" if net else "none", "--tmpfs", "/home/agent", "--tmpfs", "/tmp",
                "-v", f"{work}:/work", "-w", "/work"]
        if name:
            argv += ["--name", name]
        if session:
            argv += ["-v", f"{Path(session).resolve()}:/session"]
        if agent_dir:
            argv += ["-v", f"{Path(agent_dir).resolve()}:/opt/pi-agent"]
        for k, v in base_env.items():
            argv += ["-e", f"{k}={v}"]
        return argv + [self.image] + cmd, None, None

    def kill(self, proc, name=None):
        """Stop a wrapped process and everything it started."""
        if self.kind == "docker" and name:
            subprocess.run(["docker", "kill", name], capture_output=True)
        try:
            os.killpg(proc.pid, 15)
        except ProcessLookupError:
            return
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass


# ---------------------------------------------------------------- the hidden-test check

def check_cmd(task, sb):
    if task["track"] == "python":
        return [f"{sb.p('ac_py')}/bin/python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                task["test_file"]]
    return [f"{sb.p('node')}/bin/node", "node_modules/jest/bin/jest.js", "--ci", "--watchman=false",
            "--json", "--outputFile=.ac-jest.json", task["test_file"]]


def run_check(task, work, sb, timeout=180, name=None):
    """Run the task's test file in work (no network). Returns dict(passed: bool, n_pass, n_fail, n_total, seconds,
    timeout, output)."""
    argv, cwd, env = sb.wrap(check_cmd(task, sb), work, net=False, name=name, home=work)
    t0 = time.monotonic()
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        sb.kill(proc, name)
        out, _ = proc.communicate()
    out = out.decode("utf-8", "replace")
    res = dict(passed=proc.returncode == 0 and not timed_out, seconds=round(time.monotonic() - t0, 2),
               timeout=timed_out, n_pass=None, n_fail=None, n_total=None)
    if task["track"] == "python":
        counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|errors?)", out.splitlines()[-1]
                                                  if out.strip() else "")}
        res["n_pass"] = counts.get("passed", 0)
        res["n_fail"] = counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0)
    else:
        jf = Path(work) / ".ac-jest.json"
        if jf.exists():
            j = json.loads(jf.read_text())
            res["n_pass"], res["n_fail"] = j["numPassedTests"], j["numFailedTests"] + j["numRuntimeErrorTestSuites"]
    if res["n_pass"] is not None:
        res["n_total"] = res["n_pass"] + res["n_fail"]
    res["output"] = out[-20000:]
    return res


# ---------------------------------------------------------------- llama-server's stream

class StreamStats:
    """Collects what one chat-completions response says about itself, from its SSE stream or its JSON body:
    first chunk and first token arrival, finish reason, tool calls, the final timings and usage, and the assembled
    reply (reasoning, text, tool calls) for diagnosing cache reuse."""

    def __init__(self, t0):
        self.t0 = t0
        self.buf = b""
        self.t_first_chunk = self.t_first_token = None
        self.finish_reason = None
        self.timings = self.usage = None
        self.reasoning, self.content = [], []
        self.tool_calls = {}
        self.error = None
        self.n_chunks = 0

    def feed(self, data):
        self.buf += data
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                self.chunk(json.loads(payload))
            except ValueError:
                continue

    def chunk(self, c):
        now = time.monotonic()
        self.n_chunks += 1
        if self.t_first_chunk is None:
            self.t_first_chunk = now
        if "error" in c:
            self.error = c["error"]
        for ch in c.get("choices") or []:
            d = ch.get("delta") or ch.get("message") or {}
            got = False
            if d.get("reasoning_content"):
                self.reasoning.append(d["reasoning_content"]); got = True
            if d.get("content"):
                self.content.append(d["content"]); got = True
            for tc in d.get("tool_calls") or []:
                got = True
                slot = self.tool_calls.setdefault(tc.get("index", len(self.tool_calls)), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
            if got and self.t_first_token is None:
                self.t_first_token = now
            if ch.get("finish_reason"):
                self.finish_reason = ch["finish_reason"]
        if c.get("timings"):
            self.timings = c["timings"]
        if c.get("usage"):
            self.usage = c["usage"]

    def finish_json(self, body):
        try:
            self.chunk(json.loads(body))
        except ValueError:
            pass

    def summary(self):
        rel = lambda t: None if t is None else round(t - self.t0, 4)
        t = self.timings or {}
        return dict(t_first_chunk=rel(self.t_first_chunk), t_first_token=rel(self.t_first_token),
                    finish_reason=self.finish_reason, n_tool_calls=len(self.tool_calls),
                    prompt_n=t.get("prompt_n"), prompt_ms=t.get("prompt_ms"), cache_n=t.get("cache_n"),
                    predicted_n=t.get("predicted_n"), predicted_ms=t.get("predicted_ms"),
                    timings=self.timings, usage=self.usage, error=self.error)

    def reply(self):
        return dict(reasoning_content="".join(self.reasoning), content="".join(self.content),
                    tool_calls=[self.tool_calls[k] for k in sorted(self.tool_calls)])


def request_facts(body):
    """What the proxy and the replay log about a request body (a parsed JSON object)."""
    tools = body.get("tools") or []
    return dict(stream=bool(body.get("stream")), has_tools=bool(tools), n_tools=len(tools),
                n_messages=len(body.get("messages") or []), max_tokens=body.get("max_tokens"),
                grammar=bool(body.get("grammar") or body.get("json_schema") or body.get("response_format")),
                sampling={k: body[k] for k in ("temperature", "top_p", "top_k", "min_p", "seed") if k in body})
