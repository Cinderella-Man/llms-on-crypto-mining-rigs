"""Run pi headless on the polyglot tasks against one server (one arm) and score every attempt.

    python3 run.py --arm v2-ud --base-url http://127.0.0.1:PORT/v1 --out RESULTS [--tasks 6] [--attempts 1]
                   [--parallel 1] [--turn-cap 40] [--time-cap-min 30] --sandbox bwrap --node-dir … --pi-root …
                   --ac-node … --ac-py …

Per attempt (task × attempt number), in RESULTS/<arm>/<task>/a<k>/:
- a fresh copy of the task's workspace, mounted at /work in the sandbox (the same path for every task, so pi's system
  prompt is byte-identical and its prefix reusable by the server);
- pi in JSON mode with the check's own agent folder (RESULTS/pi-agent: models.json pointing at --base-url,
  settings.json with retries off and a 30-minute HTTP idle timeout, the check.ts extension that fixes the sampling,
  caps max_tokens and tags requests for the proxy), copied into the attempt's folder because pi writes there; every
  event line is written to events.jsonl with its arrival time;
- caps: --turn-cap turns and --time-cap-min minutes, after which pi is stopped and the attempt is flagged a runaway;
- the hidden tests run on the final workspace in the sandbox without network; success = the full test file passes and
  the visible test file is unchanged (pass and tests_untouched are also reported apart);
- result.json (the per-attempt row), diff.patch, check.txt, stderr.txt, session/, metrics before and after.

Infrastructure failures (the server or tunnel unreachable, a broken stream, an HTTP 5xx, pi failing before any reply)
are never scored: the try's folder is renamed a<k>-t<n>-infra and the attempt runs again (up to --max-infra tries)
once the server answers /health again. A finished attempt has a result.json and is skipped on the next run, so the
runner resumes where it stopped. Tasks run in tasks/<set>/order.txt order (--task-set, default polyglot); --tasks N
takes the first N, A-B positions A to B, or a comma list of ids. All tasks of attempt 1 run before attempt 2.
--start-deadline-min bounds an arm's time: no attempt starts after that many minutes, and the ones left are reported
as skipped (a later run with the same command picks them up).
"""
import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import HERE, Sandbox, diff_workspace, load_tasks, make_workspace, put_hidden_tests, run_check, \
    tests_untouched

MAKERS_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0}
UNKNOWN_TOOL = re.compile(r"^Tool \S+ not found")
INVALID_ARGS = re.compile(r"^Validation failed for tool")
MARKUP = re.compile(r"<tool_call>|<arg_key>|<arg_value>|</tool_call>")
REJECTED = re.compile(r"\b(400|413|422)\b|exceed|context size|too long", re.I)
LOCK = threading.Lock()


def log(msg):
    with LOCK:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def parse_range(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def select_tasks(spec, task_set="polyglot"):
    tasks = load_tasks(task_set=task_set)
    if spec in (None, "all"):
        return tasks
    if spec.isdigit():
        return tasks[: int(spec)]
    if re.fullmatch(r"\d+-\d+", spec):
        a, b = map(int, spec.split("-"))
        return tasks[a - 1: b]
    return load_tasks(spec.split(","))


def write_agent_dir(dest, base_url, context, max_tokens):
    dest.mkdir(parents=True, exist_ok=True)
    models = json.loads((HERE / "pi" / "models.json").read_text())
    prov = models["providers"]["agent-check"]
    prov["baseUrl"] = base_url
    prov["models"][0]["contextWindow"] = context
    prov["models"][0]["maxTokens"] = max_tokens
    (dest / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    shutil.copy(HERE / "pi" / "settings.json", dest / "settings.json")
    shutil.copy(HERE / "pi" / "check.ts", dest / "check.ts")
    return dest


def server_root(base_url):
    return re.sub(r"/v1/?$", "", base_url.rstrip("/"))


def http_get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001 - any failure means "not available"
        return None, str(e)


def wait_healthy(base_url, minutes):
    deadline = time.monotonic() + 60 * minutes
    warned = False
    while True:
        st, _ = http_get(server_root(base_url) + "/health")
        if st == 200:
            return True
        if time.monotonic() > deadline:
            return False
        if not warned:
            log(f"server not healthy at {server_root(base_url)}/health; waiting up to {minutes} min")
            warned = True
        time.sleep(10)


class Attempt:
    """Counters gathered from pi's event stream while it runs."""

    def __init__(self):
        self.turns = 0
        self.tool_calls = Counter()
        self.tool_errors = 0
        self.unknown_tool = 0
        self.invalid_args = 0
        self.markup_in_text = 0
        self.call_in_think = 0
        self.stop_reasons = Counter()
        self.error_messages = []
        self.usage = Counter()
        self.compactions = 0
        self.retries = 0
        self.settled = False
        self.n_events = 0
        self.t_first_event = None
        self.assistant_messages = 0

    def event(self, e, t):
        self.n_events += 1
        if self.t_first_event is None:
            self.t_first_event = t
        typ = e.get("type")
        if typ == "turn_start":
            self.turns += 1
        elif typ == "tool_execution_start":
            self.tool_calls[e.get("toolName", "?")] += 1
        elif typ == "tool_execution_end" and e.get("isError"):
            self.tool_errors += 1
            text = "".join(c.get("text", "") for c in (e.get("result") or {}).get("content", []) if isinstance(c, dict))
            self.unknown_tool += bool(UNKNOWN_TOOL.match(text.strip()))
            self.invalid_args += bool(INVALID_ARGS.match(text.strip()))
        elif typ == "message_end" and (e.get("message") or {}).get("role") == "assistant":
            m = e["message"]
            self.assistant_messages += 1
            self.stop_reasons[m.get("stopReason", "?")] += 1
            if m.get("errorMessage"):
                self.error_messages.append(m["errorMessage"])
            for k, v in (m.get("usage") or {}).items():
                if isinstance(v, (int, float)) and k in ("input", "output", "cacheRead", "cacheWrite"):
                    self.usage[k] += v
            for c in m.get("content") or []:
                if c.get("type") == "text" and MARKUP.search(c.get("text", "")):
                    self.markup_in_text += 1
                if c.get("type") == "thinking" and "<tool_call>" in c.get("thinking", ""):
                    self.call_in_think += 1
        elif typ == "compaction_start":
            self.compactions += 1
        elif typ == "auto_retry_start":
            self.retries += 1
        elif typ == "agent_settled":
            self.settled = True


def run_pi(a, sb, task, adir, tag, work, agent_dir, name):
    """Run one pi process; return (Attempt, wall seconds, exit code, runaway reason)."""
    session = adir / "session"
    session.mkdir(parents=True, exist_ok=True)
    agent_dir = Path(shutil.copytree(agent_dir, adir / "pi-agent"))  # pi writes a lock and auth.json there
    in_sb = sb.kind != "none"
    cmd = sb.pi_cmd() + ["--mode", "json", "--provider", "agent-check", "--model", "flint",
                         "--no-context-files", "--no-extensions",
                         "-e", "/opt/pi-agent/check.ts" if in_sb else str(agent_dir / "check.ts"),
                         "--no-skills", "--no-prompt-templates", "--no-themes", "--no-approve", "--offline",
                         "--session-dir", "/session" if in_sb else str(session)]
    if a.append_system_prompt:
        cmd += ["--append-system-prompt", a.append_system_prompt]
    cmd += ["--", (task["dir"] / "prompt.md").read_text()]
    env = {"AC_TAG": tag, "AC_SAMPLING": json.dumps(a.sampling), "AC_MAX_TOKENS": str(a.max_tokens),
           "PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0"}
    home = adir.parent / f".home-{adir.name}"
    if sb.kind == "none":
        home.mkdir(exist_ok=True)
    argv, cwd, penv = sb.wrap(cmd, work, session=session, agent_dir=agent_dir, env=env, name=name, home=home)
    st = Attempt()
    t0 = time.monotonic()
    with (adir / "stderr.txt").open("wb") as err, (adir / "events.jsonl").open("w") as ev:
        proc = subprocess.Popen(argv, cwd=cwd, env=penv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=err, start_new_session=True)

        def reader():
            buf = b""
            for data in iter(lambda: proc.stdout.read1(65536), b""):
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    t = round(time.monotonic() - t0, 3)
                    try:
                        e = json.loads(line.rstrip(b"\r"))
                    except ValueError:
                        e = {"type": "_unparsed", "raw": line.decode("utf-8", "replace")[:2000]}
                    st.event(e, t)
                    ev.write(json.dumps({"t": t, "e": e}) + "\n")

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        runaway = ""
        while proc.poll() is None:
            if st.turns > a.turn_cap:
                runaway = "turns"
            elif time.monotonic() - t0 > 60 * a.time_cap_min:
                runaway = "time"
            if runaway:
                sb.kill(proc, name)
                break
            time.sleep(0.25)
        proc.wait()
        th.join(10)
    if home.exists():
        shutil.rmtree(home, ignore_errors=True)
    return st, round(time.monotonic() - t0, 2), proc.returncode, runaway


def classify(st, code, runaway):
    """'' for a scorable attempt, else ('infra' | 'rejected', message)."""
    if runaway:
        return "", ""
    msgs = " | ".join(st.error_messages)
    if st.stop_reasons.get("error"):
        return ("rejected" if REJECTED.search(msgs) and "502" not in msgs else "infra"), msgs[:2000]
    if st.assistant_messages == 0:
        return "infra", f"pi exited {code} without a reply"
    if not st.settled and code != 0:
        return "infra", f"pi exited {code} before settling"
    return "", ""


def attempt(a, sb, task, k, slots, agent_dir):
    tdir = a.out / a.arm / task["id"]
    adir = tdir / f"a{k}"
    if (adir / "result.json").exists():
        return json.loads((adir / "result.json").read_text())
    tries = len(list(tdir.glob(f"a{k}-t*-infra")))
    slot = slots.get()
    if a.deadline is not None and time.monotonic() > a.deadline:
        slots.put(slot)
        log(f"{a.arm}/{task['id']}/a{k} skipped: past the deadline")
        return dict(arm=a.arm, task_id=task["id"], attempt=k, status="skipped")
    try:
        while True:
            n = tries + 1
            if adir.exists():
                shutil.rmtree(adir)
            adir.mkdir(parents=True)
            if not wait_healthy(a.base_url, a.health_wait_min):
                raise SystemExit(f"server down for {a.health_wait_min} min; stopping")
            sroot = server_root(a.base_url)
            st_, metrics = http_get(sroot + "/metrics")
            if st_ == 200:
                (adir / "metrics_before.txt").write_text(metrics)
            work = make_workspace(task, a.work_root / f"s{slot}" / "work", sb.node_modules())
            tag = f"{a.arm}/{task['id']}/a{k}/t{n}"
            name = f"{sb.prefix}-{os.getpid()}-s{slot}"
            started = time.strftime("%Y-%m-%dT%H:%M:%S")
            log(f"{tag} start")
            st, wall, code, runaway = run_pi(a, sb, task, adir, tag, work, agent_dir, name)
            st_, metrics = http_get(sroot + "/metrics")
            if st_ == 200:
                (adir / "metrics_after.txt").write_text(metrics)
            kind, msg = classify(st, code, runaway)
            if kind == "infra":
                tries = n
                adir.rename(tdir / f"a{k}-t{n}-infra")
                (tdir / f"a{k}-t{n}-infra" / "infra.txt").write_text(msg + "\n")
                log(f"{tag} infrastructure failure ({msg[:160]}); {'retrying' if n < a.max_infra else 'giving up'}")
                if n < a.max_infra:
                    continue
                row = dict(arm=a.arm, task_id=task["id"], attempt=k, try_n=n, status="infra_exhausted",
                           error_message=msg)
                adir.mkdir(parents=True, exist_ok=True)
                (adir / "result.json").write_text(json.dumps(row, indent=1) + "\n")
                return row
            untouched = tests_untouched(task, work)
            (adir / "diff.patch").write_text(diff_workspace(task, work))
            score = a.work_root / f"s{slot}" / "score"
            if score.exists():
                shutil.rmtree(score)
            shutil.copytree(work, score, symlinks=True)
            put_hidden_tests(task, score)
            chk = run_check(task, score, sb, timeout=a.check_timeout, name=name + "-check")
            (adir / "check.txt").write_text(chk.pop("output"))
            for d in (work, score):
                shutil.rmtree(d, ignore_errors=True)
            malformed = st.unknown_tool + st.invalid_args + st.markup_in_text
            row = dict(
                arm=a.arm, task_id=task["id"], track=task["track"], attempt=k, try_n=n, tag=tag, status="scored",
                success=int(chk["passed"] and untouched), hidden_pass=int(chk["passed"]),
                tests_untouched=int(untouched), n_pass=chk["n_pass"], n_fail=chk["n_fail"], n_total=chk["n_total"],
                check_timeout=int(chk["timeout"]), check_s=chk["seconds"],
                turns=st.turns, tool_calls=sum(st.tool_calls.values()), tool_calls_by_name=dict(st.tool_calls),
                tool_errors=st.tool_errors, malformed_calls=malformed, unknown_tool=st.unknown_tool,
                invalid_args=st.invalid_args,
                markup_in_text=st.markup_in_text, call_in_think=st.call_in_think,
                length_stops=st.stop_reasons.get("length", 0), stop_reasons=dict(st.stop_reasons),
                runaway=runaway, compactions=st.compactions, retries=st.retries,
                error_type=kind, error_message=msg,
                wall_s=wall, t_first_event=st.t_first_event, pi_exit=code,
                usage_input=st.usage["input"], usage_output=st.usage["output"],
                usage_cache_read=st.usage["cacheRead"], started=started,
                finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
            (adir / "result.json").write_text(json.dumps(row, indent=1) + "\n")
            log(f"{tag} {'PASS' if row['success'] else 'fail'} turns {st.turns} calls {row['tool_calls']} "
                f"malformed {malformed} {wall:.0f} s{' runaway ' + runaway if runaway else ''}")
            return row
    finally:
        slots.put(slot)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help="name of this server configuration, e.g. v2-ud")
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible base URL pi talks to (the proxy), …/v1")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--task-set", default="polyglot", help="folder under tasks/ (polyglot, bugfix)")
    ap.add_argument("--tasks", default="all", help="all | N (first N) | A-B (positions) | id,id,…")
    ap.add_argument("--start-deadline-min", type=float, default=None,
                    help="start no attempt after this many minutes from the runner's start")
    ap.add_argument("--attempts", default="1", help="e.g. 1, 1-2, 3-5")
    ap.add_argument("--parallel", type=int, default=1, help="attempts in flight (the server's slots)")
    ap.add_argument("--turn-cap", type=int, default=40)
    ap.add_argument("--time-cap-min", type=float, default=30)
    ap.add_argument("--check-timeout", type=float, default=180)
    ap.add_argument("--context", type=int, default=262144, help="pi's contextWindow; the server's -c")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--sampling", type=json.loads, default=MAKERS_SAMPLING)
    ap.add_argument("--append-system-prompt", default=None)
    ap.add_argument("--max-infra", type=int, default=3, help="tries per attempt before giving up on it")
    ap.add_argument("--health-wait-min", type=float, default=30)
    ap.add_argument("--work-root", type=Path, default=Path(tempfile.gettempdir()) / "agent-check-work")
    Sandbox.add_args(ap)
    a = ap.parse_args()
    sb = Sandbox.from_args(a)
    a.work_root = a.work_root / a.arm
    a.out.mkdir(parents=True, exist_ok=True)
    agent_dir = write_agent_dir(a.out / "pi-agent" / a.arm, a.base_url, a.context, a.max_tokens)
    a.deadline = None if a.start_deadline_min is None else time.monotonic() + 60 * a.start_deadline_min
    tasks = select_tasks(a.tasks, a.task_set)
    attempts = parse_range(a.attempts)
    jobs = [(t, k) for k in attempts for t in tasks]
    (a.out / a.arm).mkdir(parents=True, exist_ok=True)
    with (a.out / a.arm / "runs.jsonl").open("a") as f:
        f.write(json.dumps(dict(started=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv[1:],
                                tasks=[t["id"] for t in tasks], attempts=attempts)) + "\n")
    slots = queue.Queue()
    for i in range(a.parallel):
        slots.put(i)
    log(f"arm {a.arm}: {len(tasks)} tasks × {len(attempts)} attempts, {a.parallel} in flight, sandbox {a.sandbox}")
    with ThreadPoolExecutor(a.parallel) as ex:
        rows = list(ex.map(lambda j: attempt(a, sb, j[0], j[1], slots, agent_dir), jobs))
    scored = [r for r in rows if r.get("status") == "scored"]
    ok = sum(r["success"] for r in scored)
    log(f"arm {a.arm}: {len(scored)} scored, {ok} passed, "
        f"{sum(1 for r in rows if r.get('status') == 'infra_exhausted')} gave up after infrastructure failures, "
        f"{sum(1 for r in rows if r.get('status') == 'skipped')} skipped past the deadline")


if __name__ == "__main__":
    main()
