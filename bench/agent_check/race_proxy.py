"""First-finish race proxy between pi and N llama-servers (standard library only; registry Q1195).

    python3 -I race_proxy.py serve --listen 127.0.0.1:PORT --upstream 127.0.0.1:P1 --upstream 127.0.0.1:P2 \
        --log DIR/proxy.jsonl [--win-at reply|reasoning] [--count [--count-wait-s S]] [--keepalive-s 15]
    python3 -I race_proxy.py summary DIR/proxy.samples.jsonl [--one ONE/proxy.jsonl] [--marks 500,1000] > race.md

Every chat request goes to every upstream at once, sample i with its own seed: the request's `seed`, or the CRC32 of
its X-Agent-Check tag when it has none, plus 104729 × i (so sample 0 keeps a seed the request set). The first sample to
finish wins (--win-at reply: its whole reply arrived and ended with a finish reason); with --win-at reasoning a
streamed sample wins as soon as its reasoning ends (its first text or tool call: Q93's rule), and the rest of it is
forwarded live. The other samples are cancelled the moment the winner is chosen: their connections are closed, which
llama-server reads as a cancelled task. A sample that fails (server unreachable, an HTTP error, a broken stream) never
wins and does not stop the others; only when every sample failed does the client get the error (the first HTTP error
an upstream sent, else 502; on a stream whose headers went out, the stream is cut, as proxy.py does).

A streamed request gets its response headers as soon as one upstream answers, then an SSE comment line every
--keepalive-s seconds while the race runs (SSE clients skip comment lines), then the winner's stream as llama-server
sent it. The client's first token therefore arrives when the winner is chosen. Streamed samples ask llama-server for
`timings_per_token`, so a cancelled sample's last chunk says how many tokens it generated (--no-timings-per-token
turns that off; the count is then its chunks with a delta). A client that goes away cancels every sample.

The next request of a conversation carries the winner's reply, so every losing server has to catch up: its cache holds
the previous prompt and its own partial reply, the new prompt diverges where the reply starts, and a hybrid model like
Flash-Next cannot roll its recurrent state back, so the server restores its newest checkpoint at or before that point
(4 or `-ub` + 4 tokens before the end of the previous prompt) and reads those tokens, the winner's reply and the tool
result as prompt. The winner reads only the tool result. Each sample row records what its server did on the previous
request of the same trace (`prev`), so `summary` gives the losers' extra read in tokens and milliseconds.

Count mode (--count; registry Q1499): nobody is cancelled. Every sample runs to its end, and only then does the
client get the winner's reply (the rule's: the first whole reply, or with --win-at reasoning the first whole reply to
end its reasoning), so every server is idle when the next request comes and all of them start it together. The
request is logged as a pair: each sample's tokens and seconds, and which finished first. One run then gives, on the
same requests, what one server alone takes, what the race takes under either rule, and what a second sample started
late would take (`summary`'s count tables), where a race run gives one arm and a task's time differs between runs by
minutes. A count run is slower than one server (each request lasts as long as its slowest sample), so its own minutes
per task are not a result. --count-wait-s S stops the samples still running S seconds after the first whole reply
(default 0: no limit; a reply can run to the request's max_tokens): such a sample is logged as cancelled with its
tokens so far, and its seconds and tokens are lower bounds.

Logs:
- --log DIR/proxy.jsonl: one row per chat request in proxy.py's layout (report.py and replay.py read it), with the
  winner's timings and reply; t_first_chunk and t_first_token are when the client got them; `race` holds the number
  of samples, the winner, its upstream and seed, the rule and when the winner was chosen; in count mode also `count`,
  `first` (the sample whose whole reply came first), `first_reasoning` (the whole reply whose reasoning ended first),
  `t_all` (when every sample had ended) and `pair`: per sample its outcome, tokens, seconds and end of reasoning.
  Bodies and the winner's reply go to DIR/proxy.jsonl.bodies.gz as with proxy.py.
- --samples (default DIR/proxy.samples.jsonl): one row per sample: request id, tag, trace, turn, sample index,
  upstream, seed, outcome (won, finished: complete after the winner; cancelled; error), status, times (headers, first
  token, end of reasoning, end) from the request's arrival, finish reason, generated tokens (`predicted_n`, for a
  cancelled sample its last per-token count), chunks with a delta, characters of reasoning, text and tool arguments,
  tool calls, prompt_n, cache_n, prompt_ms, predicted_ms, the draft counters, `prev` and the error; `t_marks`, when a
  streamed sample passed each of --marks generated tokens (default 250, 500, 1000, 2000 and 4000); in count mode
  `count` and `first`.
GET /health answers 200 only when every upstream does; every other path goes to the first upstream unchanged.
"""
import argparse
import gzip
import http.client
import itertools
import json
import re
import select
import socket
import statistics
import sys
import threading
import time
import zlib
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import StreamStats, request_facts  # noqa: E402
from proxy import HOP, LOGGED  # noqa: E402

SEED_STEP = 104729  # replay.py's step between the seeds of repeats
POLL_S = 0.1  # how often a race looks for a client that went away and for a keep-alive due
MARKS = (250, 500, 1000, 2000, 4000)  # generated tokens at which a streamed sample's time is kept (Q1199's marks)


class Stats(StreamStats):
    """StreamStats plus what the race needs: chunks with a delta, when the reasoning ended, and when the sample
    passed each of `marks` generated tokens (the server's count in a chunk with a delta, else the chunks with one)."""

    def __init__(self, t0, marks=()):
        super().__init__(t0)
        self.n_deltas = 0
        self.t_answer = None
        self.marks, self.t_marks = sorted(marks), {}

    def chunk(self, c):
        had = self.n_deltas
        for ch in c.get("choices") or []:
            d = ch.get("delta") or ch.get("message") or {}
            if d.get("reasoning_content") or d.get("content") or d.get("tool_calls"):
                self.n_deltas += 1
            if self.t_answer is None and (d.get("content") or d.get("tool_calls") or ch.get("finish_reason")):
                self.t_answer = time.monotonic()
        super().chunk(c)
        if self.n_deltas > had and len(self.t_marks) < len(self.marks):
            n = (c.get("timings") or {}).get("predicted_n")
            n = n if isinstance(n, int) else self.n_deltas
            while len(self.t_marks) < len(self.marks) and n >= self.marks[len(self.t_marks)]:
                self.t_marks[self.marks[len(self.t_marks)]] = time.monotonic()


class Sample:
    """One upstream's sample of a request, read in its own thread; the race reads its fields under race.cv."""

    def __init__(self, race, i, upstream, seed, raw):
        self.race, self.i, self.upstream, self.seed, self.raw = race, i, upstream, seed, raw
        self.stats = Stats(race.t0, race.marks)
        self.chunks = []
        self.status = self.reason = None
        self.headers = []
        self.streaming = False
        self.t_headers = self.t_end = None
        self.state = "running"  # running, done, failed, cancelled
        self.error = None
        self.conn = None
        self.cancelled = False
        self.thread = threading.Thread(target=self.run, daemon=True)

    def cancel(self):
        self.cancelled = True
        sock = self.conn.sock if self.conn else None
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def end(self, state, error=None):
        with self.race.cv:
            if self.cancelled and state != "done":
                state, error = "cancelled", None
            self.state, self.error, self.t_end = state, error, time.monotonic()
            self.race.event(self)

    def run(self):
        r = self.race
        headers = dict(r.headers, **{"Content-Length": str(len(self.raw)), "Connection": "close"})
        self.conn = http.client.HTTPConnection(*self.upstream, timeout=r.timeout)
        try:
            self.conn.request("POST", r.path, body=self.raw, headers=headers)
            if self.cancelled:
                return self.end("cancelled")
            resp = self.conn.getresponse()
            with r.cv:
                self.status, self.reason = resp.status, resp.reason
                self.headers = [(k, v) for k, v in resp.getheaders() if k.lower() not in HOP]
                self.streaming = "text/event-stream" in (resp.getheader("Content-Type") or "")
                self.t_headers = time.monotonic()
                r.event(self)
            if resp.status != 200:
                self.chunks.append(resp.read())
                return self.end("failed", f"HTTP {resp.status}")
            if not self.streaming:
                self.stats.marks = []  # a whole body says nothing about when its tokens came
                data = resp.read()
                self.chunks.append(data)
                self.stats.finish_json(data)
                ok = self.stats.finish_reason is not None and not self.stats.error and not r.stream
                return self.end("done" if ok else "failed", None if ok else "no complete reply in the body")
            while True:
                data = resp.read1(65536)
                if not data:
                    break
                with r.cv:
                    self.stats.feed(data)
                    self.chunks.append(data)
                    r.event(self)
                if self.stats.done:
                    break
            ok = self.stats.finish_reason is not None and not self.stats.error and r.stream
            self.end("done" if ok else "failed", None if ok else "the stream ended before its finish")
        except (OSError, http.client.HTTPException, ValueError) as e:
            self.end("failed", f"{type(e).__name__}: {e}")
        finally:
            self.conn.close()

    def row(self):
        rel = lambda t: None if t is None else round(t - self.race.t0, 4)
        st, t = self.stats, self.stats.timings or {}
        return dict(upstream=f"{self.upstream[0]}:{self.upstream[1]}", seed=self.seed, status=self.status,
                    t_headers=rel(self.t_headers), t_first_token=rel(st.t_first_token), t_answer=rel(st.t_answer),
                    t_end=rel(self.t_end), finish_reason=st.finish_reason, predicted_n=t.get("predicted_n"),
                    n_deltas=st.n_deltas, reasoning_chars=sum(map(len, st.reasoning)),
                    content_chars=sum(map(len, st.content)),
                    tool_args_chars=sum(len(c["arguments"]) for c in st.tool_calls.values()),
                    n_tool_calls=len(st.tool_calls), prompt_n=t.get("prompt_n"), cache_n=t.get("cache_n"),
                    prompt_ms=t.get("prompt_ms"), predicted_ms=t.get("predicted_ms"), draft_n=t.get("draft_n"),
                    draft_n_accepted=t.get("draft_n_accepted"), t_marks={str(m): rel(x) for m, x in st.t_marks.items()},
                    error=self.error or st.error)


class Race:
    """The samples of one request and the rule that picks the winner."""

    def __init__(self, t0, path, headers, stream, win_at, timeout, count=False, marks=()):
        self.t0, self.path, self.headers, self.stream, self.timeout = t0, path, headers, stream, timeout
        self.win_at = win_at if stream else "reply"
        self.count, self.marks = count, marks
        self.cv = threading.Condition()
        self.samples = []
        self.finished = []  # complete samples in the order they ended
        self.answered = []  # streamed samples in the order their reasoning ended
        self.changed = False

    def event(self, s):  # under cv
        if s.state == "done" and s not in self.finished:
            self.finished.append(s)
        if s.stats.t_answer is not None and s.streaming and s.status == 200 and s not in self.answered:
            self.answered.append(s)
        self.changed = True
        self.cv.notify_all()

    def pick(self):  # under cv
        if self.win_at == "reasoning":
            return next((s for s in self.answered if s.state != "failed"), None)
        return self.finished[0] if self.finished else None

    def first_reasoning(self):  # under cv: the whole reply whose reasoning ended first
        return next((s for s in self.answered if s.state == "done"), None)

    def pick_whole(self):  # under cv; count mode, the samples ended: the rule's winner among the whole replies
        w = self.first_reasoning() if self.win_at == "reasoning" else None
        return w or (self.finished[0] if self.finished else None)

    def running(self):
        return [s for s in self.samples if s.state == "running"]


def trace_of(tag):
    """(trace, turn) of an X-Agent-Check tag "<arm>/<task>/<attempt>/<try>/<turn>", or (tag, None)."""
    m = re.match(r"^(.+)/(\d+)$", tag or "")
    return (m.group(1), int(m.group(2))) if m else (tag, None)


class Log:
    def __init__(self, path, samples):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.bodies = Path(str(path) + ".bodies.gz")
        self.samples = Path(samples) if samples else self.path.with_name(self.path.stem + ".samples.jsonl")
        self.lock = threading.Lock()
        self.ids = itertools.count(1)
        self.prefix = time.strftime("%Y%m%d-%H%M%S")

    def next_id(self):
        return f"{self.prefix}-{next(self.ids):06d}"

    def write(self, row, samples, body, reply):
        with self.lock:
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            with self.samples.open("a") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
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

    def send_bytes(self, status, headers, data, reason=None):
        self.send_response(status, reason)
        for k, v in headers:
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_error_json(self, status, message):
        msg = json.dumps({"error": {"code": status, "message": message, "type": "proxy_error"}}).encode()
        self.send_bytes(status, [("Content-Type", "application/json")], msg)

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            return self.health()
        self.passthrough(b"")

    def do_POST(self):
        raw = self.read_body()
        if self.path.split("?")[0] in LOGGED[:2]:
            return self.race(raw)
        self.passthrough(raw)

    def do_DELETE(self):
        self.passthrough(self.read_body())

    def health(self):
        out, ok = {}, True
        for h, p in self.server.upstreams:
            conn = http.client.HTTPConnection(h, p, timeout=10)
            try:
                conn.request("GET", "/health")
                r = conn.getresponse()
                r.read()
                out[f"{h}:{p}"] = r.status
                ok = ok and r.status == 200
            except (OSError, http.client.HTTPException) as e:
                out[f"{h}:{p}"] = f"{type(e).__name__}: {e}"
                ok = False
            finally:
                conn.close()
        body = json.dumps({"status": "ok" if ok else "unavailable", "upstreams": out}).encode()
        self.send_bytes(200 if ok else 503, [("Content-Type", "application/json")], body)

    def passthrough(self, raw):
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        headers["Content-Length"] = str(len(raw))
        headers["Connection"] = "close"
        conn = http.client.HTTPConnection(*self.server.upstreams[0], timeout=self.server.upstream_timeout)
        try:
            conn.request(self.command, self.path, body=raw if raw or self.command == "POST" else None,
                         headers=headers)
            r = conn.getresponse()
            self.send_bytes(r.status, r.getheaders(), r.read(), r.reason)
        except (OSError, http.client.HTTPException) as e:
            self.send_error_json(502, f"race proxy: upstream failed: {e}")
        finally:
            conn.close()

    def client_gone(self):
        try:
            if select.select([self.connection], [], [], 0)[0]:
                return self.connection.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True
        return False

    def write_chunk(self, data):
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def race(self, raw):
        t0 = time.monotonic()
        srv = self.server
        tag = self.headers.get("X-Agent-Check")
        row = dict(id=srv.log.next_id(), tag=tag, path=self.path, t_wall=time.time(), req_bytes=len(raw))
        try:
            body = json.loads(raw)
            row.update(request_facts(body))
        except ValueError:
            row["bad_json"] = True
            self.send_error_json(400, "race proxy: the request body is not JSON")
            return
        stream = bool(body.get("stream"))
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        r = Race(t0, self.path, headers, stream, srv.win_at, srv.upstream_timeout, srv.count, srv.marks)
        seed = body.get("seed")
        base = seed if isinstance(seed, int) and seed >= 0 else zlib.crc32((tag or row["id"]).encode())
        for i, up in enumerate(srv.upstreams):
            b = dict(body, seed=(base + SEED_STEP * i) % (2 ** 31))
            if stream and srv.timings_per_token:
                b["timings_per_token"] = True
            r.samples.append(Sample(r, i, up, b["seed"], json.dumps(b).encode()))
        trace, turn = trace_of(tag)
        with srv.lock:
            prev = srv.prev.get(trace)
        for s in r.samples:
            s.thread.start()
        sent_headers, winner, client_status, t_win, t_flush, t_all = False, None, None, None, None, None
        last_write = time.monotonic()
        try:
            while True:  # the race: wait for a winner or for every sample to fail; in count mode, for every sample
                with r.cv:
                    if not r.changed:
                        r.cv.wait(POLL_S)
                    r.changed = False
                    left = r.running()
                    if not r.count:
                        winner = r.pick()
                    elif not left or (srv.count_wait_s and r.finished
                                      and time.monotonic() - r.finished[0].t_end >= srv.count_wait_s):
                        winner = r.pick_whole()
                    first_ok = next((s for s in r.samples if s.status == 200 and s.streaming), None)
                if winner or not left:
                    break
                if self.client_gone():
                    row["client_gone"] = "before the winner"
                    break
                if stream and first_ok and not sent_headers:
                    self.send_response(200, first_ok.reason)
                    for k, v in first_ok.headers:
                        self.send_header(k, v)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    sent_headers, last_write = True, time.monotonic()
                    row["t_headers"] = round(last_write - t0, 4)
                elif sent_headers and srv.keepalive_s and time.monotonic() - last_write >= srv.keepalive_s:
                    self.write_chunk(b": race\n\n")
                    last_write = time.monotonic()
            for s in r.samples:
                if s is not winner:
                    s.cancel()
            if winner is None:
                if "client_gone" in row:
                    self.close_connection = True
                elif sent_headers:
                    row["upstream_error"] = "every sample failed"
                    self.close_connection = True  # no terminating chunk: the client sees a broken stream
                else:
                    bad = next((s for s in r.samples if s.status not in (None, 200)), None)
                    if bad:
                        client_status = bad.status
                        self.send_bytes(bad.status, bad.headers, b"".join(bad.chunks), bad.reason)
                    else:
                        client_status = 502
                        errors = "; ".join(f"{s.upstream[0]}:{s.upstream[1]} {s.error}" for s in r.samples)
                        self.send_error_json(502, f"race proxy: every upstream failed: {errors}")
                    row["upstream_error"] = "every sample failed"
                return
            t_win = time.monotonic()
            if r.count:  # the rule was met when the winner's reply (or its reasoning) ended, the client served now
                t_all = t_win
                t_win = (winner.stats.t_answer if r.win_at == "reasoning" else winner.t_end) or t_win
            with srv.lock:
                srv.prev[trace] = winner.i
            client_status = winner.status
            if not stream:
                self.send_bytes(winner.status, winner.headers, b"".join(winner.chunks), winner.reason)
                t_flush = time.monotonic()
                return
            if not sent_headers:
                self.send_response(200, winner.reason)
                for k, v in winner.headers:
                    self.send_header(k, v)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                row["t_headers"] = round(time.monotonic() - t0, 4)
            n = 0
            while True:  # the winner's stream: what came so far, then the rest live
                with r.cv:
                    while n == len(winner.chunks) and winner.state == "running":
                        r.cv.wait(POLL_S)
                    data, n, state = b"".join(winner.chunks[n:]), len(winner.chunks), winner.state
                if data:
                    self.write_chunk(data)
                    t_flush = t_flush or time.monotonic()
                if state != "running" and n == len(winner.chunks):
                    break
            if state == "done":
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                row["upstream_error"] = f"the winner broke after it was chosen: {winner.error}"
                self.close_connection = True
        except (BrokenPipeError, ConnectionResetError, socket.timeout) as e:
            row["client_gone"] = type(e).__name__
            self.close_connection = True
            for s in r.samples:
                s.cancel()
        finally:
            row["status"] = client_status
            row["t_end"] = round(time.monotonic() - t0, 4)
            threading.Thread(target=self.finish_log, daemon=True,
                             args=(row, r, winner, body, t0, t_win, t_flush, t_all, trace, turn, prev)).start()

    def finish_log(self, row, r, winner, body, t0, t_win, t_flush, t_all, trace, turn, prev):
        """Logs the request and its samples once the losers' threads have ended (off the client's path)."""
        for s in r.samples:
            s.thread.join(30)
        rel = lambda t: None if t is None else round(t - t0, 4)
        samples = []
        first = r.finished[0] if r.finished else None
        for s in r.samples:
            outcome = {"done": "finished", "failed": "error", "cancelled": "cancelled",
                       "running": "cancelled"}[s.state]
            if s is winner:
                outcome = "won"
            samples.append(dict(id=row["id"], tag=row["tag"], trace=trace, turn=turn, i=s.i, n=len(r.samples),
                                win_at=r.win_at, outcome=outcome, winner=s is winner,
                                prev=None if prev is None else ("won" if prev == s.i else "lost"), **s.row()))
            if r.count:
                samples[-1].update(count=True, first=s is first)
        if winner:
            row.update(winner.stats.summary())
            for k in ("t_first_chunk", "t_first_token"):  # when the client got them
                if row.get(k) is not None:
                    row[k] = round(max(row[k], (t_flush or t_win) - t0), 4)
            row["race"] = dict(n=len(r.samples), winner=winner.i,
                               upstream=f"{winner.upstream[0]}:{winner.upstream[1]}", seed=winner.seed,
                               win_at=r.win_at, t_win=rel(t_win), outcomes=[x["outcome"] for x in samples])
            reply = winner.stats.reply()
        else:
            row["race"] = dict(n=len(r.samples), winner=None, win_at=r.win_at,
                               outcomes=[x["outcome"] for x in samples])
            reply = None
        if r.count:
            ans = r.first_reasoning()
            row["race"].update(count=True, first=first.i if first else None, first_reasoning=ans.i if ans else None,
                               t_all=rel(t_all),
                               pair=[dict(i=x["i"], outcome=x["outcome"], tokens=tokens(x), seconds=x["t_end"],
                                          t_answer=x["t_answer"]) for x in samples])
        self.server.log.write(row, samples, body, reply)


# ---------------------------------------------------------------- summary

def tokens(s):
    """A sample's generated tokens: the server's count, else its chunks with a delta (marked by the caller)."""
    return s["predicted_n"] if s.get("predicted_n") is not None else s.get("n_deltas") or 0


def task_of(tag):
    """(arm, task) of a trace or tag."""
    parts = (tag or "").split("/")
    return (parts[0], parts[1]) if len(parts) >= 2 else (tag, None)


def fmt(x, nd=1):
    return "" if x is None else f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}"


def summary(a):
    reqs = defaultdict(list)  # a request's samples; two proxies started in the same second number alike
    for k, p in enumerate(a.samples):
        for l in open(p):
            if l.strip():
                s = json.loads(l)
                reqs[k, s["id"]].append(s)
    by_arm = defaultdict(list)
    for rid, ss in reqs.items():
        by_arm[task_of(ss[0]["tag"])[0]].append(sorted(ss, key=lambda s: s["i"]))
    out = ["# First-finish race: samples", "",
           f"From {', '.join(str(p) for p in a.samples)}. Tokens are each sample's generated tokens (the server's "
           "count; for a cancelled sample its count when it was cancelled).", "",
           "## Per arm", "",
           "| Arm | Requests | Samples a request | Won by sample (0, 1, …) | Errors | Cancelled | Finished late "
           "| Winner's tokens | Every sample's tokens | Ratio | Counted from chunks |",
           "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, rs in sorted(by_arm.items()):
        n = max(len(ss) for ss in rs)
        wins = [sum(1 for ss in rs for s in ss if s["winner"] and s["i"] == i) for i in range(n)]
        cnt = lambda o: sum(1 for ss in rs for s in ss if s["outcome"] == o)
        won = sum(tokens(s) for ss in rs for s in ss if s["winner"])
        every = sum(tokens(s) for ss in rs for s in ss)
        chunks = sum(1 for ss in rs for s in ss if s.get("predicted_n") is None and s["outcome"] != "error")
        out.append(f"| {arm} | {len(rs)} | {n} | {', '.join(map(str, wins))} | {cnt('error')} | {cnt('cancelled')} "
                   f"| {cnt('finished')} | {won:,} | {every:,} | {every / won if won else 0:.3f} | {chunks} |")
    out += ["", "## The losers' catch-up", "",
            "Per request that follows a raced request of the same trace: each server that lost the previous request "
            "against the one that won it, on the same request.", "",
            "| Arm | Pairs | Extra prompt tokens, mean | median | Extra prompt ms, mean | median "
            "| Previous winner wins again |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, rs in sorted(by_arm.items()):
        dn, dms, again, nreq = [], [], 0, 0
        for ss in rs:
            won_prev = [s for s in ss if s["prev"] == "won"]
            if not won_prev:
                continue
            w = won_prev[0]
            nreq += 1
            again += 1 if w["winner"] else 0
            for s in ss:
                if s["prev"] == "lost" and s.get("prompt_n") is not None and w.get("prompt_n") is not None:
                    dn.append(s["prompt_n"] - w["prompt_n"])
                    if s.get("prompt_ms") is not None and w.get("prompt_ms") is not None:
                        dms.append(s["prompt_ms"] - w["prompt_ms"])
        med = lambda v: statistics.median(v) if v else None
        mean = lambda v: statistics.fmean(v) if v else None
        out.append(f"| {arm} | {len(dn)} | {fmt(mean(dn))} | {fmt(med(dn))} | {fmt(mean(dms))} | {fmt(med(dms))} "
                   f"| {again} of {nreq} |")
    per_run = defaultdict(lambda: defaultdict(int))  # one server's runs: arm -> task -> generated tokens
    for p in a.one or []:
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                if r.get("status") == 200 and r.get("predicted_n") is not None and "race" not in r:
                    arm, t = task_of(r.get("tag"))
                    per_run[arm][t] += r["predicted_n"]
    one = {t: statistics.fmean(run[t] for run in per_run.values() if t in run)
           for t in {t for run in per_run.values() for t in run}}
    out += ["", "## Per task", "",
            "Every request of the task's attempts (with re-run tries) is counted. One server's tokens are the mean over "
            f"the runs given with --one ({', '.join(sorted(per_run)) or 'none'})." if a.one else
            "Every request of the task's attempts (with re-run tries) is counted.", "",
            "| Arm | Task | Requests | Winner's tokens | Every sample's tokens"
            + (" | One server's tokens | Share of one server |" if a.one else " |"),
            "|---|---|---:|---:|---:" + ("|---:|---:|" if a.one else "|")]
    for arm, rs in sorted(by_arm.items()):
        per = defaultdict(lambda: [0, 0, 0])
        for ss in rs:
            t = task_of(ss[0]["tag"])[1]
            per[t][0] += 1
            per[t][1] += sum(tokens(s) for s in ss if s["winner"])
            per[t][2] += sum(tokens(s) for s in ss)
        tot = [0, 0]
        for t, (nr, w, e) in sorted(per.items(), key=lambda kv: str(kv[0])):
            line = f"| {arm} | {t} | {nr} | {w:,} | {e:,}"
            if a.one:
                o = one.get(t)
                line += f" | {fmt(o)} | {w / o:.3f} |" if o else " | | |"
                if o:
                    tot[0] += w
                    tot[1] += o
            out.append(line + ("" if a.one else " |"))
        if a.one and tot[1]:
            out.append(f"| {arm} | tasks of both | | {tot[0]:,} | | {fmt(tot[1])} | {tot[0] / tot[1]:.3f} |")
    print("\n".join(out + count_tables(by_arm, a.marks)))


def at_tokens(s, m):
    """Seconds from the request's arrival to sample s's m-th generated token: its t_marks, and a straight line
    between its known points (the first token, the marks, the end) for any other m."""
    pts = [(0, s.get("t_first_token"))] + sorted((int(k), v) for k, v in (s.get("t_marks") or {}).items())
    pts = [(n, t) for n, t in pts + [(tokens(s), s["t_end"])] if t is not None]
    for (n0, t0), (n1, t1) in zip(pts, pts[1:]):
        if n0 <= m <= n1:
            return t1 if n1 == n0 else t0 + (t1 - t0) * (m - n0) / (n1 - n0)
    return s["t_end"]


def extra_read(s, ss):
    """Seconds of sample s's prompt read beyond the previous winner's on the same request: the catch-up of a server
    that lost the request before (0 for the others, and where a prompt time is missing)."""
    w = next((x for x in ss if x["prev"] == "won"), None)
    if s["prev"] != "lost" or w is None or s.get("prompt_ms") is None or w.get("prompt_ms") is None:
        return 0.0
    return max(0.0, (s["prompt_ms"] - w["prompt_ms"]) / 1000)


def count_tables(by_arm, marks):
    """Count mode's tables: one server alone, each rule and a late second sample, all on the same requests."""
    arms = {arm: [ss for ss in rs if ss[0].get("count")] for arm, rs in by_arm.items()}
    if not any(arms.values()):
        return []
    rules, late, counted = [], [], []
    for arm, rs in sorted(arms.items()):
        n = max((len(ss) for ss in rs), default=0)
        use = [ss for ss in rs if len(ss) == n >= 2 and any(s["winner"] for s in ss)
               and all(s["outcome"] != "error" and s.get("t_end") is not None for s in ss)]
        if not use:
            continue
        early = sum(1 for ss in use if any(s["outcome"] == "cancelled" for s in ss))
        counted.append(f"{arm}: {len(use)} of {len(rs)} requests, {early} of them with a sample stopped early")
        own = [[s["t_end"] - extra_read(s, ss) for s in ss] for ss in use]  # per request, each sample less its catch-up
        alone_s = [sum(o[i] for o in own) for i in range(n)]
        alone_tok = [sum(tokens(ss[i]) for ss in use) for i in range(n)]
        base_s, base_tok = statistics.fmean(alone_s), statistics.fmean(alone_tok)

        def line(what, kept, sec, before, tok):
            rules.append(f"| {arm} | {what} | {len(use)} | {kept} | {sec:,.1f} | {sec / base_s:.3f} | {before:,.1f} "
                         f"| {before / base_s:.3f} | {tok:,} | {tok / base_tok:.3f} |")

        def rule(what, key):  # key(sample): the time that decides the rule, None for a sample that cannot be kept
            kept, sec, before, tok = [0] * n, 0.0, 0.0, 0
            for ss, o in zip(use, own):
                can = [s for s in ss if key(s) is not None]
                if not can:
                    return
                w = min(can, key=key)
                kept[w["i"]] += 1
                sec += w["t_end"]
                before += o[min(can, key=lambda s: key(s) - extra_read(s, ss))["i"]]
                tok += tokens(w)
            line(what, ", ".join(map(str, kept)), sec, before, tok)

        for i in range(n):
            line(f"server {i} alone", "", sum(ss[i]["t_end"] for ss in use), alone_s[i], alone_tok[i])
        rule("the first whole reply", lambda s: s["t_end"] if s["outcome"] != "cancelled" else None)
        rule("the first to end its reasoning", lambda s: s.get("t_answer"))
        for p in range(n):
            for m in marks:
                started, kept2, sec = 0, 0, 0.0
                for ss, o in zip(use, own):
                    t = o[p]
                    if tokens(ss[p]) > m:
                        started += 1
                        t2 = at_tokens(ss[p], m) - extra_read(ss[p], ss) + min(
                            (o[i] for i in range(n) if i != p and ss[i]["outcome"] != "cancelled"), default=t)
                        if t2 < t:
                            kept2, t = kept2 + 1, t2
                    sec += t
                late.append(f"| {arm} | {p} | {m:,} | {len(use)} | {started} | {kept2} | {sec:,.1f} "
                            f"| {sec / alone_s[p]:.3f} |")
    if not counted:
        return ["", "## Count mode", "", "No request with a kept reply and no failed sample."]
    return ["", "## Count mode: one server, each rule and a late second sample on the same requests", "",
            "In count mode no sample is cancelled, and the agent gets the kept reply once every sample has ended. "
            "Seconds run from the request's arrival at the proxy to the end of a sample's reply. A server that lost "
            "the request before reads the kept reply first (the catch-up); \"before the catch-up\" takes that read "
            "out of its seconds (est.: its prompt time less the previous winner's on the same request). One server "
            "is the mean of the servers alone before the catch-up, which a server alone never reads. A rule's "
            "seconds hold the catch-up that the run's own rule caused. A sample stopped early (--count-wait-s) "
            "counts with its seconds and tokens so far, which are lower bounds, and is never the first whole reply.",
            "", "Counted (a kept reply and no failed sample): " + "; ".join(counted) + ".", "",
            "| Arm | What | Requests | Kept from server (0, 1, …) | Seconds | Share of one server "
            "| Seconds before the catch-up (est.) | Share of one server | Tokens kept | Share of one server |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"] + rules + [
            "", "### A late second sample", "",
            "The first server starts at once, the others only when its reply passes the mark, and the reply that "
            "ends first is kept. Seconds are before the catch-up (est.) and before the longer read of a server that "
            "sat out earlier requests; a reply's time at a mark is its `t_marks`, or a straight line between the "
            "points the row has.", "",
            "| Arm | First server | Mark (tokens) | Requests | Second sample started | Second sample kept "
            "| Seconds (est.) | Share of the first server alone |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"] + late


def marks_arg(text):
    return tuple(sorted({int(x) for x in text.split(",") if x.strip()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the proxy")
    s.add_argument("--listen", required=True, help="host:port")
    s.add_argument("--upstream", action="append", required=True,
                   help="host:port of a llama-server (or the tunnel to it); once per server, in sample order")
    s.add_argument("--log", required=True, type=Path)
    s.add_argument("--samples", type=Path, help="the per-sample log (default: next to --log, .samples.jsonl)")
    s.add_argument("--win-at", choices=["reply", "reasoning"], default="reply",
                   help="reply: the first whole reply wins; reasoning: the first streamed sample to end its reasoning")
    s.add_argument("--count", action="store_true",
                   help="count mode: no sample is cancelled; the client gets the winner's reply once every sample has "
                        "ended, and the request is logged as a pair")
    s.add_argument("--count-wait-s", type=float, default=0,
                   help="count mode: stop the samples still running this long after the first whole reply (0: never)")
    s.add_argument("--marks", type=marks_arg, default=MARKS,
                   help="generated tokens at which each streamed sample's time is kept, comma-separated")
    s.add_argument("--keepalive-s", type=float, default=15,
                   help="an SSE comment line this often while a streamed race runs (0: none)")
    s.add_argument("--no-timings-per-token", dest="timings_per_token", action="store_false",
                   help="don't ask the servers for timings in every streamed chunk")
    s.add_argument("--upstream-timeout", type=float, default=3600,
                   help="seconds of silence from an upstream before its sample fails (a long prefill sends nothing)")
    m = sub.add_parser("summary", help="tables from the per-sample logs")
    m.add_argument("samples", nargs="+", type=Path)
    m.add_argument("--one", nargs="+", type=Path,
                   help="proxy.jsonl of runs on one server, for the share per task (raced rows in it are skipped)")
    m.add_argument("--marks", type=marks_arg, default=MARKS[1:],
                   help="count mode's late second sample: the first sample's generated tokens at which it starts")
    a = ap.parse_args()
    if a.cmd == "summary":
        return summary(a)
    host, port = a.listen.rsplit(":", 1)
    srv = ThreadingHTTPServer((host, int(port)), Handler)
    srv.daemon_threads = True
    srv.upstreams = [(h, int(p)) for h, p in (u.rsplit(":", 1) for u in a.upstream)]
    srv.upstream_timeout = a.upstream_timeout
    srv.win_at, srv.keepalive_s, srv.timings_per_token = a.win_at, a.keepalive_s, a.timings_per_token
    srv.count, srv.count_wait_s, srv.marks = a.count, a.count_wait_s, a.marks
    srv.log = Log(a.log, a.samples)
    srv.lock = threading.Lock()
    srv.prev = {}  # trace -> the sample index that won its last request
    mode = "count mode: every sample runs to its end, " if a.count else ""
    print(f"race proxy {a.listen} -> {', '.join(a.upstream)} ({mode}win at {a.win_at}), log {a.log}, samples "
          f"{srv.log.samples}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
