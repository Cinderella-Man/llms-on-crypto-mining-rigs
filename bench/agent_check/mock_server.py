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
- --malformed-every K: on tasks whose CRC32 is 2 mod K its first reply carries tool-call markup as text.
The timings are synthetic: prompt_n and cache_n from a 4-characters-per-token estimate and the previous request's
prefix, at --prefill-ms and --decode-ms per token.
"""
import argparse
import json
import random
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent / "tasks"


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


def crc(s):
    return zlib.crc32(s.encode())


def reply_for(task_id, n_assistant, args, try_id):
    """(reasoning, content, tool_calls, drop) for this step of the script."""
    tdir = task_dir(task_id)
    task = json.loads((tdir / "task.json").read_text()) if tdir else None
    stub = task["stub"] if task else "solution.py"
    test_file = task["test_file"] if task else "solution_test.py"
    test_cmd = task["test_cmd"] if task else "pytest"
    k = crc(task_id or "")
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
        if not self.path.startswith("/v1/chat/completions"):
            return self.send_json(404, {"error": "not found"})
        body = json.loads(raw)
        a = self.server.args
        tag = (self.headers.get("X-Agent-Check") or "").split("/")
        task_id = tag[1] if len(tag) >= 2 else None
        try_id = tag[3] if len(tag) >= 4 else ""
        msgs = body.get("messages", [])
        n_assistant = sum(1 for m in msgs if m.get("role") == "assistant")
        reasoning, content, calls, drop = reply_for(task_id, n_assistant, a, try_id)
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
        time.sleep(prompt_ms / 1000)
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
            time.sleep(predicted_n * a.decode_ms / 1000)
            msg = {"role": "assistant", "content": content or None, "reasoning_content": reasoning}
            if tool_calls:
                msg["tool_calls"] = [{k: v for k, v in tc.items() if k != "index"} for tc in tool_calls]
            return self.send_json(200, {"id": cid, "object": "chat.completion", "model": "flint",
                                        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                        "usage": usage, "timings": timings})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj):
            data = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n"
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()

        def chunk(delta, finish_reason=None, **extra):
            return dict(id=cid, object="chat.completion.chunk", model="flint",
                        choices=[{"index": 0, "delta": delta, "finish_reason": finish_reason}], **extra)

        per_piece = a.decode_ms / 1000
        send(chunk({"role": "assistant", "content": None}))
        for i in range(0, len(reasoning), 12):
            send(chunk({"reasoning_content": reasoning[i:i + 12]}))
            time.sleep(per_piece)
            if drop and i > 24:
                self.close_connection = True
                self.wfile.flush()
                self.connection.shutdown(2)
                return
        for i in range(0, len(content), 24):
            send(chunk({"content": content[i:i + 24]}))
            time.sleep(per_piece)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--fail-every", type=int, default=0)
    ap.add_argument("--drop-every", type=int, default=0)
    ap.add_argument("--malformed-every", type=int, default=0)
    ap.add_argument("--prefill-ms", type=float, default=0.05)
    ap.add_argument("--decode-ms", type=float, default=1.0)
    a, ignored = ap.parse_known_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.args = a
    srv.daemon_threads = True
    print(f"mock llama-server on {a.host}:{a.port}" + (f" (ignored: {' '.join(ignored)})" if ignored else ""),
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
