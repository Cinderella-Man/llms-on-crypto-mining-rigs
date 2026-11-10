#!/usr/bin/env python3
"""A stand-in for llama-server, for testing the runner, proxy and replay without a model.

    python3 mock_server.py --port 18080 [--fail-every 4] [--drop-every 5] [--malformed-every 6]

It ignores llama-server's own flags (-m, -ngl, -c, …), so rig_server.sh can start it in place of a llama-server
binary (BIN=…/mock_server.py, its own options among the server flags) for a dry run of the rig scripts.

It speaks llama-server's OpenAI-compatible chat endpoint (streamed and not), /health, /metrics, /v1/models and
/slots, and ends each response with llama-server's `timings` and `usage`. Its "agent" follows a script keyed on the
number of assistant turns already in the request, for the task named in the X-Agent-Check header:
reasoning and `ls`, read the visible tests, write Exercism's reference solution into the stub, run the tests, finish.
Requests without the header (replays) get the same script.
- --fail-every K: on tasks whose CRC32 is 0 mod K it writes the stub back unchanged, so the hidden tests fail;
- --drop-every K: on the first try of tasks whose CRC32 is 1 mod K it drops the connection in the middle of the
  second turn's stream, an infrastructure failure the runner must re-run;
- --malformed-every K: on tasks whose CRC32 is 2 mod K its first reply carries tool-call markup as text;
- --loop-tasks ID[:N],…: on the listed tasks, in tries t1 to tN of an attempt (every try without :N), the agent
  never writes the solution: from its second reply on it reasons --loop-chars characters (four a token) and reads the
  test file again, until the runner stops it (run.py's token mark, or its turn or time cap).
The timings are synthetic: prompt_n and cache_n from a 4-characters-per-token estimate and the previous request's
prefix, at --prefill-ms and --decode-ms per token. A client that closes the connection while a chat reply is being
"decoded" cancels it, streamed or not, as llama-server does. For race_proxy.py's self-test:
- --name NAME: each chat reply's reasoning starts with "[NAME seed S] " (S the request's seed);
- --seed-length K: the reasoning is written 1 + (seed mod K) times, so the reply's length follows the seed;
- a streamed request with `timings_per_token` gets the timings so far in every chunk with a delta;
- --error-status CODE: every chat request gets that HTTP status and an error body;
- --log FILE: one JSON line per chat request: the tag, seed, whether it streamed, the pieces sent, start and end
  (wall clock) and the outcome (done, cancelled, dropped, error).

It also speaks the native endpoints bench/deep_lane uses: /tokenize (four UTF-8 bytes a token, the id their value),
/apply-template (a ChatML-like rendering of the messages and tools) and /completion (a prompt as text or token ids,
n_predict, cache_prompt, return_tokens). /completion's reply is a function of the prompt alone at temperature 0 and of
the prompt and the seed otherwise, and its `timings` carry what llama-server's do: cache_n, cache_src, prompt_n,
prompt_ms, prompt_per_second, predicted_n, predicted_ms, predicted_per_second and, when the server line has -md, the
draft counters draft_n and draft_n_accepted. Its model of the server:
- the cache is rolled back as a recurrent model's is: to the newest of the last four checkpoints at or below the shared
  prefix, one checkpoint four tokens before the end of each prompt (cache_src checkpoint), or not at all (none); a
  prompt that continues the whole cache reuses it (live); a request generates at least one token;
- a step costs --decode-ms plus --depth-ms per 1,000 tokens of prompt; with -md every step drafts --spec-draft-n-max
  tokens (default 3), each accepted with probability --draft-accept while the ones before it were, and costs
  --draft-step-cost times a plain step.
At its start it prints the GGML_, LLAMA_ and CUDA_ variables of its environment.
"""
import argparse
import json
import os
import random
import select
import socket
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent / "tasks"
NATIVE = ("/tokenize", "/apply-template", "/completion")
LETTERS = b"abcdefghijklmnopqrstuvwxyz "


def task_dir(task_id):
    """The task's folder in any task set, or None (the same lookup as common.task_dir)."""
    for d in sorted(TASK_ROOT.glob(f"*/{task_id}")) if task_id else []:
        if (d / "task.json").exists():
            return d
    return None


class State:
    lock = threading.Lock()
    last_prompt = ""
    tries = {}  # task -> set of try ids seen
    kv = []  # /completion: the token ids in the cache (the last prompt and its reply)
    ckpts = []  # /completion: the checkpoints' positions, oldest first


def crc(s):
    return zlib.crc32(s.encode())


def tok_ids(text):
    """The mock's tokenizer: four UTF-8 bytes a token, the id their little-endian value."""
    b = text.encode()
    return [int.from_bytes(b[i:i + 4], "little") for i in range(0, len(b), 4)]


def tok_text(ids):
    return b"".join(i.to_bytes(4, "little").rstrip(b"\0") for i in ids).decode("utf-8", "replace")


def render(body):
    """A ChatML-like prompt of a chat request's tools and messages, ending where the assistant's turn starts."""
    out = []
    if body.get("tools"):
        out.append("<|im_start|>system\n# Tools\n" + json.dumps(body["tools"]) + "<|im_end|>\n")
    for m in body.get("messages", []):
        c = m.get("content") or ""
        if not isinstance(c, str):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        think = f"<think>\n{m['reasoning_content']}\n</think>\n" if m.get("reasoning_content") else ""
        calls = "".join("\n<tool_call>" + json.dumps(t.get("function", t)) + "</tool_call>"
                        for t in m.get("tool_calls") or [])
        out.append(f"<|im_start|>{m.get('role', 'user')}\n{think}{c}{calls}<|im_end|>\n")
    return "".join(out) + "<|im_start|>assistant\n"


def complete(body, a):
    """/completion: (reply, seconds to sleep) for a request body."""
    prompt = body.get("prompt", "")
    ids = tok_ids(prompt) if isinstance(prompt, str) else [int(t) for t in prompt]
    n = max(1, int(body.get("n_predict", 16)))
    with State.lock:
        common = 0
        for x, y in zip(State.kv, ids):
            if x != y:
                break
            common += 1
        if not body.get("cache_prompt", True):
            cache_n, src = 0, "none"
        elif common and common == len(State.kv):
            cache_n, src = min(common, len(ids) - 1), "live"
        else:
            cache_n = max((c for c in State.ckpts if c <= common), default=0)
            src = "checkpoint" if cache_n else "none"
        State.ckpts = [c for c in State.ckpts if c <= cache_n and c != len(ids) - 4]
        if len(ids) > 4:
            State.ckpts = (State.ckpts + [len(ids) - 4])[-4:]
        greedy = body.get("temperature", 1.0) == 0 or body.get("top_k") == 1
        seed = zlib.crc32(json.dumps(ids).encode()) ^ (0 if greedy else int(body.get("seed", 0)) + 1)
        rng = random.Random(seed)
        out = [int.from_bytes(bytes(rng.choice(LETTERS) for _ in range(4)), "little") for _ in range(n)]
        State.kv = ids + out
    steps, drafted, accepted = n, 0, 0
    if a.draft:
        rng, steps, made = random.Random(seed ^ 0x5BD1E995), 0, 0
        while made < n:
            k = 0
            while k < a.draft_n_max and rng.random() < a.draft_accept:
                k += 1
            steps, drafted, accepted, made = steps + 1, drafted + a.draft_n_max, accepted + k, made + 1 + k
    prompt_n = max(1, len(ids) - cache_n)
    prompt_ms = max(prompt_n * a.prefill_ms, 1e-6)
    predicted_ms = steps * (a.decode_ms + a.depth_ms * len(ids) / 1000) * (a.draft_step_cost if a.draft else 1)
    predicted_ms = max(predicted_ms, 1e-6)
    timings = {"cache_n": cache_n, "cache_src": src, "prompt_n": prompt_n, "prompt_ms": prompt_ms,
               "prompt_per_token_ms": prompt_ms / prompt_n, "prompt_per_second": prompt_n * 1000 / prompt_ms,
               "predicted_n": n, "predicted_ms": predicted_ms, "predicted_per_token_ms": predicted_ms / n,
               "predicted_per_second": n * 1000 / predicted_ms}
    if a.draft:
        timings.update(draft_n=drafted, draft_n_accepted=accepted)
    reply = {"content": tok_text(out), "stop": True, "stop_type": "limit", "tokens_predicted": n,
             "tokens_evaluated": len(ids), "timings": timings}
    if body.get("return_tokens"):
        reply["tokens"] = out
    return reply, (prompt_ms + predicted_ms) / 1000


def reply_for(task_id, n_assistant, args, try_id):
    """(reasoning, content, tool_calls, drop) for this step of the script."""
    tdir = task_dir(task_id)
    task = json.loads((tdir / "task.json").read_text()) if tdir else None
    stub = task["stub"] if task else "solution.py"
    test_file = task["test_file"] if task else "solution_test.py"
    test_cmd = task["test_cmd"] if task else "pytest"
    k = crc(task_id or "")
    loop_tries = args.loop.get(task_id or "", 0)
    if n_assistant >= 1 and try_id[1:].isdigit() and int(try_id[1:]) <= loop_tries:
        text = f"Reply {n_assistant}: the tests in {test_file} may hide another case; let me read them once more. "
        return ((text * (args.loop_chars // len(text) + 1))[:args.loop_chars], "",
                [("read", {"path": test_file})], False)
    if n_assistant == 0:
        content = ""
        if args.malformed_every and k % args.malformed_every == 2:
            content = "<tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"
        return ("The task is in the prompt. I will list the files first.", content,
                [("bash", {"command": "ls -la"})], False)
    if n_assistant == 1:
        drop = bool(args.drop_every and k % args.drop_every == 1 and try_id == "t1")
        return (f"Now read the visible tests in {test_file}.", "", [("read", {"path": test_file})], drop)
    if n_assistant == 2:
        if task and not (args.fail_every and k % args.fail_every == 0):
            body = (tdir / "reference" / stub).read_text()
        elif task:
            body = (tdir / "workspace" / stub).read_text()
        else:
            body = "def solve():\n    return 42\n"
        return ("I know how to solve it; writing the whole file.", "", [("write", {"path": stub, "content": body})],
                False)
    if n_assistant == 3:
        return ("Run the visible tests.", "", [("bash", {"command": test_cmd})], False)
    return ("The tests ran; I am done.", f"I implemented `{stub}`.", [], False)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send_json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def pause(self, seconds):
        """Sleep, or return False as soon as the client has closed the connection."""
        end = time.monotonic() + seconds
        while True:
            left = end - time.monotonic()
            if select.select([self.connection], [], [], max(0.0, min(left, 0.05)))[0]:
                try:
                    if self.connection.recv(1, socket.MSG_PEEK) == b"":
                        return False
                except OSError:
                    return False
                time.sleep(max(0.0, min(left, 0.05)))
            if left <= 0:
                return True

    def log_chat(self, row):
        if self.server.args.log:
            with State.lock, open(self.server.args.log, "a") as f:
                f.write(json.dumps(dict(row, t_end=time.time())) + "\n")

    def do_GET(self):
        if self.path.startswith("/health"):
            return self.send_json(200, {"status": "ok"})
        if self.path.startswith("/v1/models"):
            return self.send_json(200, {"object": "list", "data": [{"id": "flint", "object": "model"}]})
        if self.path.startswith("/metrics"):
            b = b"llamacpp:prompt_tokens_total 0\nllamacpp:tokens_predicted_total 0\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            return self.wfile.write(b)
        return self.send_json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        if self.path.startswith("/slots"):
            return self.send_json(200, {"id_slot": 0, "n_erased": 0})
        if self.path.startswith(NATIVE):
            body = json.loads(raw)
            if self.path.startswith("/tokenize"):
                return self.send_json(200, {"tokens": tok_ids(body.get("content", ""))})
            if self.path.startswith("/apply-template"):
                return self.send_json(200, {"prompt": render(body)})
            reply, wait = complete(body, self.server.args)
            time.sleep(wait)
            return self.send_json(200, reply)
        if not self.path.startswith("/v1/chat/completions"):
            return self.send_json(404, {"error": "not found"})
        body = json.loads(raw)
        a = self.server.args
        seed = body.get("seed")
        log = dict(tag=self.headers.get("X-Agent-Check"), seed=seed, stream=bool(body.get("stream")), pieces=0,
                   t_start=time.time(), outcome="done")
        if a.error_status:
            self.log_chat(dict(log, outcome="error"))
            return self.send_json(a.error_status, {"error": {"code": a.error_status, "type": "server_error",
                                                             "message": "mock: --error-status"}})
        tag = (self.headers.get("X-Agent-Check") or "").split("/")
        task_id = tag[1] if len(tag) >= 2 else None
        try_id = tag[3] if len(tag) >= 4 else ""
        msgs = body.get("messages", [])
        n_assistant = sum(1 for m in msgs if m.get("role") == "assistant")
        reasoning, content, calls, drop = reply_for(task_id, n_assistant, a, try_id)
        if a.seed_length and isinstance(seed, int):
            reasoning = " ".join([reasoning] * (1 + seed % a.seed_length))
        if a.name:
            reasoning = f"[{a.name} seed {seed}] {reasoning}"
        prompt = json.dumps(msgs) + json.dumps(body.get("tools", []))
        with State.lock:
            common = 0
            for x, y in zip(State.last_prompt, prompt):
                if x != y:
                    break
                common += 1
            State.last_prompt = prompt
        cache_n, total = common // 4, len(prompt) // 4
        prompt_n = max(1, total - cache_n)
        gen_text = reasoning + content + json.dumps([c[1] for c in calls])
        predicted_n = max(1, len(gen_text) // 4)
        if body.get("max_tokens"):
            predicted_n = min(predicted_n, int(body["max_tokens"]))
        prompt_ms = prompt_n * a.prefill_ms
        if not self.pause(prompt_ms / 1000):
            return self.log_chat(dict(log, outcome="cancelled"))
        cid = f"chatcmpl-mock{random.randrange(1 << 30)}"
        tool_calls = [{"index": i, "id": f"call_{random.randrange(1 << 30):x}", "type": "function",
                       "function": {"name": name, "arguments": json.dumps(args)}} for i, (name, args) in
                      enumerate(calls)]
        finish = "tool_calls" if calls else "stop"
        timings = {"cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": prompt_ms,
                   "predicted_n": predicted_n, "predicted_ms": predicted_n * a.decode_ms}
        usage = {"prompt_tokens": cache_n + prompt_n, "completion_tokens": predicted_n,
                 "total_tokens": cache_n + prompt_n + predicted_n,
                 "prompt_tokens_details": {"cached_tokens": cache_n}}
        if not body.get("stream"):
            if not self.pause(predicted_n * a.decode_ms / 1000):
                return self.log_chat(dict(log, outcome="cancelled"))
            msg = {"role": "assistant", "content": content or None, "reasoning_content": reasoning}
            if tool_calls:
                msg["tool_calls"] = [{k: v for k, v in tc.items() if k != "index"} for tc in tool_calls]
            self.send_json(200, {"id": cid, "object": "chat.completion", "model": "flint",
                                 "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                 "usage": usage, "timings": timings})
            return self.log_chat(dict(log, pieces=1))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj):
            data = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n"
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()
            log["pieces"] += 1

        def chunk(delta, finish_reason=None, **extra):
            return dict(id=cid, object="chat.completion.chunk", model="flint",
                        choices=[{"index": 0, "delta": delta, "finish_reason": finish_reason}], **extra)

        class Gone(Exception):
            pass

        def piece(delta):  # one "token"; with timings_per_token its chunk carries the timings so far, as llama-server's
            n_out[0] += 1
            k = min(n_out[0], predicted_n)
            so_far = {"timings": dict(timings, predicted_n=k, predicted_ms=k * a.decode_ms)} if tpt else {}
            send(chunk(delta, **so_far))
            if not self.pause(per_piece):
                raise Gone

        per_piece = a.decode_ms / 1000
        tpt, n_out = bool(body.get("timings_per_token")), [0]
        try:
            send(chunk({"role": "assistant", "content": None}))
            for i in range(0, len(reasoning), 12):
                piece({"reasoning_content": reasoning[i:i + 12]})
                if drop and i > 24:
                    self.close_connection = True
                    self.wfile.flush()
                    self.connection.shutdown(2)
                    return self.log_chat(dict(log, outcome="dropped"))
            for i in range(0, len(content), 24):
                piece({"content": content[i:i + 24]})
            for tc in tool_calls:
                args = tc["function"]["arguments"]
                send(chunk({"tool_calls": [{"index": tc["index"], "id": tc["id"], "type": "function",
                                            "function": {"name": tc["function"]["name"], "arguments": ""}}]}))
                for i in range(0, len(args), 200):
                    send(chunk({"tool_calls": [{"index": tc["index"], "function": {"arguments": args[i:i + 200]}}]}))
            send(chunk({}, finish))
            if (body.get("stream_options") or {}).get("include_usage"):
                send(dict(id=cid, object="chat.completion.chunk", model="flint", choices=[], usage=usage,
                          timings=timings))
            else:
                send(chunk({}, None, timings=timings))
            send(b"[DONE]")
            self.wfile.write(b"0\r\n\r\n")
        except (Gone, BrokenPipeError, ConnectionResetError):
            self.close_connection = True
            return self.log_chat(dict(log, outcome="cancelled"))
        self.log_chat(log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--fail-every", type=int, default=0)
    ap.add_argument("--drop-every", type=int, default=0)
    ap.add_argument("--malformed-every", type=int, default=0)
    ap.add_argument("--loop-tasks", default="")
    ap.add_argument("--loop-chars", type=int, default=2000)
    ap.add_argument("--prefill-ms", type=float, default=0.05)
    ap.add_argument("--decode-ms", type=float, default=1.0)
    ap.add_argument("--depth-ms", type=float, default=0.0)
    ap.add_argument("--draft-accept", type=float, default=0.7)
    ap.add_argument("--draft-step-cost", type=float, default=1.3)
    ap.add_argument("--name", default="")
    ap.add_argument("--seed-length", type=int, default=0)
    ap.add_argument("--error-status", type=int, default=0)
    ap.add_argument("--log", default="")
    a, ignored = ap.parse_known_args()
    a.loop = {t: int(n or 1 << 30) for t, _, n in (s.partition(":") for s in a.loop_tasks.split(",") if s)}
    # llama-server's own draft flags, read from the ignored ones: registered with argparse, -md would also match -m
    a.draft = "-md" in ignored or "--model-draft" in ignored
    a.draft_n_max = 3
    if "--spec-draft-n-max" in ignored[:-1]:
        a.draft_n_max = int(ignored[ignored.index("--spec-draft-n-max") + 1])
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.args = a
    srv.daemon_threads = True
    print(f"mock llama-server on {a.host}:{a.port}" + (f" (ignored: {' '.join(ignored)})" if ignored else ""),
          flush=True)
    print("environment: " + " ".join(f"{k}={v}" for k, v in sorted(os.environ.items())
                                     if k.startswith(("GGML_", "LLAMA_", "CUDA_"))), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
