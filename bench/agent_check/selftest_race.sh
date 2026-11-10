#!/usr/bin/env bash
# selftest_race.sh OUT: race_proxy.py against mock_server.py servers of different speeds (no model, no sandbox, light;
# standard library only). Uses ports AC_TEST_PORT to AC_TEST_PORT+18 on 127.0.0.1 (default 18190).
# Checks, and exits non-zero if any fails:
# - two and three servers: the fastest one's reply reaches the client, streamed and not, and the others are cancelled
#   (the mocks log the cancel); each sample carries its seed (the tag's CRC32 + 104729 × i);
# - the first complete reply wins, not the fastest server: a slower server with a shorter sample beats a faster one;
# - errors do not lose the request: an HTTP 500, an unreachable server and a stream dropped half-way each lose, and the
#   request is answered by the others; when every server fails the client gets the first HTTP error;
# - --win-at reasoning picks the first sample to end its reasoning, cancels the rest then and forwards the winner live;
# - keep-alive comment lines while a streamed race runs, which the stream parser skips;
# - the next request of a trace records which server won and which lost the one before (`prev`);
# - a client that goes away cancels every sample; /health is 200 only when every upstream is up;
# - two requests at once; the request log (proxy.py's layout, with the winner's timings), the bodies and `summary`.
set -euo pipefail
OUT=${1:?usage: selftest_race.sh OUT}
HERE=$(cd "$(dirname "$0")" && pwd)
P=${AC_TEST_PORT:-18190}
rm -rf "$OUT"; mkdir -p "$OUT/mocks"
PIDS=()
trap 'kill "${PIDS[@]}" 2> /dev/null' EXIT
mock() {  # mock OFFSET NAME ARGS…
  local off=$1 name=$2
  shift 2
  python3 "$HERE/mock_server.py" --port $((P + off)) --name "$name" --log "$OUT/mocks/$name.jsonl" "$@" \
    > "$OUT/mocks/$name.out" 2>&1 &
  PIDS+=($!)
}
race() {  # race OFFSET NAME OPTIONS… -- UPSTREAM-OFFSETS…
  local off=$1 name=$2 opts=() ups=()
  shift 2
  while [ "$1" != -- ]; do opts+=("$1"); shift; done
  shift
  for u in "$@"; do ups+=(--upstream 127.0.0.1:$((P + u))); done
  python3 -I "$HERE/race_proxy.py" serve --listen 127.0.0.1:$((P + off)) "${ups[@]}" --log "$OUT/$name/proxy.jsonl" \
    "${opts[@]}" > "$OUT/$name.out" 2>&1 &
  PIDS+=($!)
}
mock 0 fast --decode-ms 5
mock 1 mid --decode-ms 20
mock 2 slow --decode-ms 60
mock 3 err --error-status 500
mock 4 drop --decode-ms 1 --drop-every 2
mock 5 lenA --decode-ms 3 --seed-length 40
mock 6 lenB --decode-ms 12 --seed-length 40
# 7: nothing listens there
race 10 r2 -- 0 2
race 11 r3 -- 2 1 0
race 12 r3err -- 3 7 1
race 13 rdrop -- 4 2
race 14 rall -- 3 7
race 15 rlen -- 5 6
race 16 rreason --win-at reasoning -- 2 0
race 17 rkeep --keepalive-s 0.05 -- 2 1
python3 -I "$HERE/proxy.py" --listen 127.0.0.1:$((P + 18)) --upstream 127.0.0.1:$P --log "$OUT/one/proxy.jsonl" \
  > "$OUT/one.out" 2>&1 &
PIDS+=($!)
sleep 1.5
python3 -I - "$HERE" "$OUT" "$P" <<'EOF'
import gzip, http.client, json, socket, subprocess, sys, threading, time, zlib
from pathlib import Path
here, out, P = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, str(here))
from common import StreamStats
fails = []
def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        fails.append(what)

def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).exists() else []

def ask(off, tag, stream=True, seed=None, n_assistant=0, close_after=None):
    """Send one chat request; returns (status, raw bytes, parsed StreamStats)."""
    msgs = [{"role": "user", "content": "fix the bug"}]
    for k in range(n_assistant):
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": f"c{k}", "type": "function",
                  "function": {"name": "bash", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": f"c{k}",
                  "content": "ok"}]
    body = {"model": "flint", "messages": msgs, "stream": stream, "max_tokens": 4096,
            "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}]}
    if stream:
        body["stream_options"] = {"include_usage": True}
    if seed is not None:
        body["seed"] = seed
    c = http.client.HTTPConnection("127.0.0.1", P + off, timeout=60)
    c.request("POST", "/v1/chat/completions", body=json.dumps(body),
              headers={"Content-Type": "application/json", "X-Agent-Check": tag})
    if close_after is not None:
        time.sleep(close_after)
        c.sock.shutdown(socket.SHUT_RDWR)
        c.close()
        return None, b"", None
    r = c.getresponse()
    raw = r.read()
    c.close()
    st = StreamStats(time.monotonic())
    if stream and r.status == 200:
        st.feed(raw)
    else:
        st.finish_json(raw)
    return r.status, raw, st

def samples(name, tag, n):
    """The race's sample rows of a tag, waiting for the log (it is written after the losers have stopped)."""
    for _ in range(100):
        rows = [r for r in jl(out / name / "proxy.samples.jsonl") if r["tag"] == tag]
        if len(rows) >= n:
            return sorted(rows, key=lambda r: r["i"])
        time.sleep(0.1)
    return rows

def request_row(name, tag):
    rows = [r for r in jl(out / name / "proxy.jsonl") if r["tag"] == tag]
    return rows[-1] if rows else {}

def mock_rows(name, seed, want=1):
    for _ in range(50):
        rows = [r for r in jl(out / "mocks" / f"{name}.jsonl") if r["seed"] == seed]
        if len(rows) >= want:
            return rows
        time.sleep(0.1)
    return rows

def seeds(tag, n):
    base = zlib.crc32(tag.encode())
    return [(base + 104729 * i) % 2 ** 31 for i in range(n)]

def outcomes(rows):
    return [r["outcome"] for r in rows]

# two servers, streamed and not
for stream in (True, False):
    tag = f"race/t-two/a1/t1/{'s' if stream else 'n'}1"
    st, raw, s = ask(10, tag, stream)
    rows = samples("r2", tag, 2)
    sd = seeds(tag, 2)
    check(st == 200 and s.reply()["reasoning_content"].startswith(f"[fast seed {sd[0]}]")
          and s.finish_reason == "tool_calls" and len(s.tool_calls) == 1,
          f"2 servers, {'streamed' if stream else 'not streamed'}: the client gets the fast server's whole reply")
    check(outcomes(rows) == ["won", "cancelled"] and [r["seed"] for r in rows] == sd,
          f"  the slow one is cancelled; seeds are the tag's CRC32 + 104729 × i ({outcomes(rows)})")
    check([m["outcome"] for m in mock_rows("slow", sd[1])] == ["cancelled"]
          and [m["outcome"] for m in mock_rows("fast", sd[0])] == ["done"],
          "  the slow mock saw the cancel, the fast one finished")
    if stream:
        check(b"data: [DONE]" in raw and s.timings and s.usage, "  the stream ends with [DONE], timings and usage")
        check(rows[1]["n_deltas"] >= 1 and rows[1]["predicted_n"] == rows[1]["n_deltas"]
              and rows[1]["prompt_n"] is not None,
              f"  the cancelled sample's tokens so far come from timings_per_token ({rows[1]['predicted_n']})")
        rq = request_row("r2", tag)
        check(rq.get("race", {}).get("winner") == 0 and rq.get("predicted_n") == rows[0]["predicted_n"]
              and rq["t_first_token"] >= rq["race"]["t_win"] and rq.get("stream") is True,
              "  the request row has the winner's timings and the client's first-token time")
        bodies = {b["id"]: b for b in map(json.loads, gzip.open(out / "r2/proxy.jsonl.bodies.gz", "rt"))}
        b = bodies.get(rq.get("id"), {})
        check("seed" not in b.get("body", {}) and b.get("reply", {}).get("reasoning_content", "").startswith("[fast"),
              "  the bodies log keeps the client's request and the winner's reply")

# three servers, the fastest last
tag = "race/t-three/a1/t1/1"
st, raw, s = ask(11, tag)
rows = samples("r3", tag, 3)
sd = seeds(tag, 3)
check(st == 200 and s.reply()["reasoning_content"].startswith(f"[fast seed {sd[2]}]")
      and outcomes(rows) == ["cancelled", "cancelled", "won"],
      f"3 servers: the fastest (third) wins, the other two are cancelled ({outcomes(rows)})")
check(mock_rows("slow", sd[0])[0]["outcome"] == "cancelled" and mock_rows("mid", sd[1])[0]["outcome"] == "cancelled",
      "  both slower mocks saw the cancel")
tag = "race/t-three/a1/t1/n1"
st, raw, s = ask(11, tag, stream=False)
rows = samples("r3", tag, 3)
check(st == 200 and json.loads(raw)["choices"][0]["message"]["reasoning_content"].startswith("[fast")
      and outcomes(rows) == ["cancelled", "cancelled", "won"], "3 servers, not streamed: the fastest wins")

# errors on some servers
for stream in (True, False):
    tag = f"race/t-err/a1/t1/{'s' if stream else 'n'}1"
    st, raw, s = ask(12, tag, stream)
    rows = samples("r3err", tag, 3)
    check(st == 200 and s.reply()["reasoning_content"].startswith("[mid") and outcomes(rows) == ["error", "error", "won"]
          and rows[0]["status"] == 500 and "HTTP 500" in rows[0]["error"] and rows[1]["status"] is None
          and "Connection" in rows[1]["error"],
          f"an HTTP 500 and an unreachable server lose, the third answers ({'streamed' if stream else 'not streamed'})")
drop_task = next(t for t in (f"t-drop{k}" for k in range(100)) if zlib.crc32(t.encode()) % 2 == 1)
tag = f"race/{drop_task}/a1/t1/2"
st, raw, s = ask(13, tag, n_assistant=1)
rows = samples("rdrop", tag, 2)
check(st == 200 and s.reply()["reasoning_content"].startswith("[slow") and s.finish_reason
      and outcomes(rows) == ["error", "won"] and mock_rows("drop", rows[0]["seed"])[0]["outcome"] == "dropped",
      f"a faster server whose stream breaks half-way loses, the slow one answers ({outcomes(rows)})")
for stream in (True, False):
    tag = f"race/t-all/a1/t1/{'s' if stream else 'n'}1"
    st, raw, s = ask(14, tag, stream)
    rows = samples("rall", tag, 2)
    rq = request_row("rall", tag)
    check(st == 500 and json.loads(raw)["error"]["message"] == "mock: --error-status"
          and outcomes(rows) == ["error", "error"] and rq.get("upstream_error") == "every sample failed",
          f"every server failing: the client gets the first HTTP error ({'streamed' if stream else 'not streamed'})")

# the first whole reply wins, not the fastest server
tag = "race/t-len/a1/t1/1"
st, raw, s = ask(15, tag, seed=31)  # lenA: 1 + 31 % 40 = 32 times the reasoning; lenB: 1 + (31 + 104729) % 40 = 1
rows = samples("rlen", tag, 2)
check(st == 200 and s.reply()["reasoning_content"].startswith("[lenB seed 104760]") and outcomes(rows) == ["cancelled", "won"]
      and rows[0]["seed"] == 31, "a slower server with a shorter sample beats a faster one with a longer sample")
tag = "race/t-len/a1/t1/2"
st, raw, s = ask(15, tag, seed=0)  # lenA once, lenB ten times
rows = samples("rlen", tag, 2)
check(st == 200 and s.reply()["reasoning_content"].startswith("[lenA seed 0]") and outcomes(rows) == ["won", "cancelled"],
      "  and with seeds the other way round the faster server wins")

# --win-at reasoning
tag = "race/t-reason/a1/t1/5"
st, raw, s = ask(16, tag, n_assistant=4)
rows = samples("rreason", tag, 2)
rq = request_row("rreason", tag)
check(st == 200 and s.reply()["reasoning_content"].startswith("[fast") and s.reply()["content"] == "I implemented `solution.py`."
      and s.finish_reason == "stop" and outcomes(rows) == ["cancelled", "won"],
      "--win-at reasoning: the first to end its reasoning wins and its whole reply follows")
check(rq.get("race", {}).get("t_win") is not None and rows[1]["t_answer"] <= rq["race"]["t_win"] < rows[1]["t_end"]
      and rows[0]["t_answer"] is None and rq["race"]["win_at"] == "reasoning",
      "  chosen when its reasoning ended, before its reply did; the loser was cancelled still reasoning")

# keep-alive comments; a client that goes away
tag = "race/t-keep/a1/t1/1"
st, raw, s = ask(17, tag)
check(st == 200 and raw.index(b": race") < raw.index(b"data:") and s.reply()["reasoning_content"].startswith("[mid")
      and s.finish_reason, "keep-alive comment lines before the winner's stream; the parser skips them")
tag = "race/t-gone/a1/t1/1"
ask(17, tag, stream=False, close_after=0.1)  # the faster mock would answer after ~0.5 s
rows = samples("rkeep", tag, 2)
rq = request_row("rkeep", tag)
check(outcomes(rows) == ["cancelled", "cancelled"] and rq.get("client_gone")
      and [m["outcome"] for m in mock_rows("slow", rows[0]["seed"]) + mock_rows("mid", rows[1]["seed"])]
      == ["cancelled", "cancelled"], "a client that goes away cancels every sample")

# the next request of a trace: who won and who lost the one before
for turn in (1, 2):
    st, raw, s = ask(10, f"race/t-prev/a1/t1/{turn}", n_assistant=turn - 1)
    ask(18, f"one/t-prev/a1/t1/{turn}", n_assistant=turn - 1)
rows = samples("r2", "race/t-prev/a1/t1/2", 2)
check([r["prev"] for r in rows] == ["won", "lost"] and [r["turn"] for r in rows] == [2, 2]
      and rows[0]["trace"] == "race/t-prev/a1/t1", "the next request records the previous winner and loser (prev)")

# two at once
res = []
th = [threading.Thread(target=lambda k=k: res.append(ask(11, f"race/t-par{k}/a1/t1/1"))) for k in range(2)]
[t.start() for t in th]
[t.join() for t in th]
check(len(res) == 2 and all(r[0] == 200 and r[2].reply()["reasoning_content"].startswith("[fast") for r in res),
      "two requests at once through one race proxy")

# health
def health(off):
    c = http.client.HTTPConnection("127.0.0.1", P + off, timeout=20)
    c.request("GET", "/health")
    r = c.getresponse()
    return r.status, json.loads(r.read())
check(health(11)[0] == 200 and health(12)[0] == 503, "/health is 200 with every upstream up, 503 with one down")

# summary
time.sleep(0.5)
sm = subprocess.run([sys.executable, "-I", str(here / "race_proxy.py"), "summary", str(out / "r2/proxy.samples.jsonl"),
                     "--one", str(out / "one/proxy.jsonl")], capture_output=True, text=True)
(out / "summary.md").write_text(sm.stdout + sm.stderr)
lines = sm.stdout.splitlines()
catch = [l for l in lines if l.startswith("| race |") and "of" in l.split("|")[-2]]
share = [l for l in lines if l.startswith("| race | t-prev |")]
check(sm.returncode == 0 and "## Per arm" in sm.stdout and any(l.startswith("| race | 4 | 2 | 4, 0 |") for l in lines)
      and catch and catch[0].split("|")[2].strip() == "1" and share
      and 0.8 < float(share[0].split("|")[-2]) < 1.25,  # the mock's reply names its seed, so lengths differ a little
      "summary: wins per server, the losers' catch-up pairs, the share against one server")
sys.exit(1 if fails else 0)
EOF
echo "race selftest passed; logs in $OUT (summary.md)"
