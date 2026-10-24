"""Pass-through logging proxy between pi and llama-server (standard library only).

    python3 -I proxy.py --listen 127.0.0.1:PORT --upstream 127.0.0.1:PORT --log DIR/proxy.jsonl

Every request is forwarded unchanged (bytes of the body as received; hop-by-hop headers dropped; the response body
re-framed with chunked encoding when it streams). pi drops llama-server's `timings`, so for every chat request the
proxy writes one JSON line to --log with:
- id, the X-Agent-Check tag ("<arm>/<task>/<attempt>/<try>/<turn>"), path, request size, whether it streams,
  carries tools (backend sampling is off for those) or a grammar, max_tokens and sampling fields;
- wall times relative to the request's arrival: response headers, first chunk, first token (reasoning, text or tool
  call), end; the HTTP status, finish reason and number of tool calls;
- llama-server's timings (prompt_n, prompt_ms, cache_n, predicted_n, predicted_ms) and usage;
- errors: upstream unreachable or broken, client gone mid-stream.
The request body and the assembled reply go to DIR/proxy.jsonl.bodies.gz (one gzip member and JSON line per request,
keyed by id), which replay.py re-sends. Other paths (/health, /metrics, /slots, /v1/models) pass through unlogged.
"""
import argparse
import gzip
import http.client
import itertools
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import StreamStats, request_facts  # noqa: E402

HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer", "upgrade",
       "content-length", "host", "accept-encoding"}
LOGGED = ("/v1/chat/completions", "/chat/completions", "/completion", "/v1/completions")


class Log:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.bodies = Path(str(path) + ".bodies.gz")
        self.lock = threading.Lock()
        self.ids = itertools.count(1)
        self.prefix = time.strftime("%Y%m%d-%H%M%S")

    def next_id(self):
        return f"{self.prefix}-{next(self.ids):06d}"

    def write(self, row, body, reply):
        with self.lock:
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            with gzip.open(self.bodies, "ab") as g:
                g.write((json.dumps({"id": row["id"], "body": body, "reply": reply}) + "\n").encode())


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 3600

    def log_message(self, *a):
        pass

    def read_body(self):
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            parts = []
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip(), 16)
                if size == 0:
                    while self.rfile.readline().strip():
                        pass
                    break
                parts.append(self.rfile.read(size))
                self.rfile.readline()
            return b"".join(parts)
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        self.forward(b"")

    def do_POST(self):
        self.forward(self.read_body())

    def do_DELETE(self):
        self.forward(self.read_body())

    def forward(self, raw):
        t0 = time.monotonic()
        srv = self.server
        logged = self.command == "POST" and self.path.split("?")[0] in LOGGED
        row = dict(id=srv.log.next_id() if logged else None, tag=self.headers.get("X-Agent-Check"),
                   path=self.path, t_wall=time.time(), req_bytes=len(raw))
        body = None
        if logged:
            try:
                body = json.loads(raw)
                row.update(request_facts(body))
            except ValueError:
                row["bad_json"] = True
        stats = StreamStats(t0)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        headers["Content-Length"] = str(len(raw))
        headers["Connection"] = "close"
        conn = http.client.HTTPConnection(*srv.upstream, timeout=srv.upstream_timeout)
        status = None
        try:
            try:
                conn.request(self.command, self.path, body=raw if raw or self.command == "POST" else None,
                             headers=headers)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as e:
                row["upstream_error"] = f"{type(e).__name__}: {e}"
                msg = json.dumps({"error": {"code": 502, "message": f"proxy: upstream failed: {e}",
                                            "type": "proxy_error"}}).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                status = 502
                return
            status = resp.status
            row["t_headers"] = round(time.monotonic() - t0, 4)
            streaming = "text/event-stream" in (resp.getheader("Content-Type") or "")
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            chunks = []
            if streaming:
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    while True:
                        try:
                            data = resp.read1(65536)
                        except (OSError, http.client.HTTPException) as e:
                            row["upstream_error"] = f"broken stream: {type(e).__name__}: {e}"
                            break
                        if not data:
                            break
                        stats.feed(data)
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                        self.wfile.flush()
                    if "upstream_error" in row:
                        self.close_connection = True  # no terminating chunk: the client sees a broken stream
                    else:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, socket.timeout) as e:
                    row["client_gone"] = type(e).__name__
                    self.close_connection = True
            else:
                try:
                    data = resp.read()
                except (OSError, http.client.HTTPException) as e:
                    data = b""
                    row["upstream_error"] = f"broken body: {type(e).__name__}: {e}"
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError) as e:
                    row["client_gone"] = type(e).__name__
                    self.close_connection = True
                chunks.append(data)
                if logged:
                    stats.finish_json(data)
        finally:
            conn.close()
            if logged:
                row["status"] = status
                row["t_end"] = round(time.monotonic() - t0, 4)
                row.update(stats.summary())
                srv.log.write(row, body if body is not None else raw.decode("utf-8", "replace"), stats.reply())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", required=True, help="host:port")
    ap.add_argument("--upstream", required=True, help="host:port of llama-server (or the tunnel to it)")
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--upstream-timeout", type=float, default=3600,
                    help="seconds of silence from upstream before a request fails (a long prefill sends nothing)")
    a = ap.parse_args()
    host, port = a.listen.rsplit(":", 1)
    uh, up = a.upstream.rsplit(":", 1)
    srv = ThreadingHTTPServer((host, int(port)), Handler)
    srv.daemon_threads = True
    srv.upstream = (uh, int(up))
    srv.upstream_timeout = a.upstream_timeout
    srv.log = Log(a.log)
    print(f"proxy {a.listen} -> {a.upstream}, log {a.log}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
