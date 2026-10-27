"""Mock llama-server for harness smoke tests: /props, /tokenize, /completion. Logs request bodies.

usage: python mock_server.py PORT REQUESTS.jsonl

Answers every request at once, with no model: a LiveCodeBench prompt gets a small Python program, a BFCL prompt
with tools gets either a call to the first offered function or a refusal (seeded per request), and a turn after a
tool result gets a plain-text answer. Point run.py, multi_turn.py run or ../livecodebench_flint/run.py at
http://127.0.0.1:PORT to check the client, the manifest and the scoring path end to end; REQUESTS.jsonl records each
request's sampling settings (everything but the prompt).
"""
import json
import random
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = open(sys.argv[2], "a")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"default_generation_settings": {"n_ctx": 40000}, "total_slots": 4, "model_path": "mock.gguf",
                    "build_info": "mock", "chat_template": "x"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            return self._send({"tokens": list(range(len(body["content"]) // 4))})
        LOG.write(json.dumps({k: v for k, v in body.items() if k != "prompt"}) + "\n")
        LOG.flush()
        p = body["prompt"]
        rnd = random.Random(body["seed"] * 7919 + len(p))
        if "```python" in p:  # LiveCodeBench
            text = "<think>plan</think>```python\nprint(int(input())*2)\n```"
        elif "<tools>" in p:
            tail = p.rsplit("<|assistant|>", 2)
            names = re.findall(r'"name": "([^"]+)"', p)
            if "<|observation|>" in tail[-2] if len(tail) > 2 else False:
                text = "<think>done</think>All done."
            elif rnd.random() < 0.5 and names:
                text = f"<think>call</think><tool_call>{names[0]}</tool_call>"
            else:
                text = "<think>no</think>I cannot help."
        else:
            text = "<think>x</think>hello"
        self._send({"content": text, "tokens_predicted": len(text) // 4, "tokens_evaluated": len(p) // 4,
                    "stop_type": "eos", "truncated": False, "timings": {"predicted_per_second": 1}})


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
