#!/usr/bin/env bash
# selftest_race.sh OUT: race_proxy.py against mock_server.py servers of different speeds (no model, no sandbox, light;
# standard library only). Uses ports AC_TEST_PORT to AC_TEST_PORT+21 on 127.0.0.1 (default 18190).
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
# - count mode (--count): nobody is cancelled, the client gets the first whole reply only once every sample has ended,
#   streamed and not, and the request is logged as a pair (each sample's tokens and seconds, which came first), with
#   the time each sample passed --marks tokens; with --win-at reasoning the client follows the first whole reply to
#   end its reasoning; --count-wait-s stops a sample that runs on after the first whole reply; `summary` prints the
#   count tables (one server alone, each rule, a late second sample);
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
race 19 rcount --count --marks 10,50 -- 5 6
race 20 rcountr --count --win-at reasoning -- 2 0
race 21 rwait --count --count-wait-s 0.3 -- 5 6
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

# count mode: nobody is cancelled, the reply comes once every sample has ended, each request is a pair
# seed 71: lenA writes its reasoning 32 times and lenB (seed 104800) once; seed 40: lenA once, lenB (104769) ten times
near = lambda a, b: abs(a - b) <= 0.002
for turn, seed, stream, first in ((1, 71, True, 1), (2, 40, True, 0), (3, 71, False, 1)):
    tag = f"count/t-count/a1/t1/{turn}"
    how = "streamed" if stream else "not streamed"
    st, raw, s = ask(19, tag, stream, seed=seed, n_assistant=turn - 1)
    rows = samples("rcount", tag, 2)
    rq = request_row("rcount", tag)
    rc = rq.get("race", {})
    name, sd = ("lenA", "lenB")[first], (seed, seed + 104729)
    check(st == 200 and s.reply()["reasoning_content"].startswith(f"[{name} seed {sd[first]}]")
          and outcomes(rows) == (["finished", "won"] if first else ["won", "finished"])
          and all(r.get("count") for r in rows) and [r["first"] for r in rows] == [first == 0, first == 1],
          f"count mode, {how}: the client gets the first whole reply and the other sample runs to its end "
          f"({outcomes(rows)})")
    check([m["outcome"] for m in mock_rows("lenA", sd[0], 2 if seed == 71 and turn == 3 else 1)
           + mock_rows("lenB", sd[1], 2 if seed == 71 and turn == 3 else 1)].count("done") >= 2
          and "cancelled" not in [m["outcome"] for m in mock_rows("lenA", sd[0]) + mock_rows("lenB", sd[1])],
          "  both mocks finished, neither saw a cancel")
    late = max(r["t_end"] for r in rows)
    check(rc.get("count") is True and rc.get("first") == first and rc.get("winner") == first
          and near(rc["t_win"], rows[first]["t_end"]) and rc["t_all"] >= late - 0.002
          and rq["t_first_token"] >= late - 0.002 and rows[1 - first]["t_end"] > rows[first]["t_end"],
          "  the reply reached the client only once the slower sample had ended")
    check([(x["i"], x["outcome"], x["tokens"], x["seconds"]) for x in rc.get("pair", [])]
          == [(r["i"], r["outcome"], r["predicted_n"], r["t_end"]) for r in rows]
          and all(r["predicted_n"] and r["predicted_n"] >= r["n_deltas"] for r in rows),
          f"  the request row holds the pair: each sample's tokens and seconds "
          f"({[(x['tokens'], x['seconds']) for x in rc.get('pair', [])]})")
    if turn == 1:
        long_, short = rows[0], rows[1]  # lenA wrote 32 times the reasoning, lenB once
        tm = long_["t_marks"]
        check(set(tm) == {"10", "50"} and long_["t_first_token"] <= tm["10"] < tm["50"] < long_["t_end"]
              and "50" not in short["t_marks"] and rc.get("first_reasoning") == 1,
              f"  each sample's time at the marks of 10 and 50 tokens ({tm}); the short one never reached 50")
    if turn == 2:
        check([r["prev"] for r in rows] == ["lost", "won"],
              "  the next request records who won and who lost the one before, as in a race")
    if not stream:
        check(all(r["t_marks"] == {} for r in rows), "  a reply that is not streamed has no marks")
tag = "countr/t-reason/a1/t1/5"
st, raw, s = ask(20, tag, n_assistant=4)
rows = samples("rcountr", tag, 2)
rq = request_row("rcountr", tag)
rc = rq.get("race", {})
check(st == 200 and s.reply()["reasoning_content"].startswith("[fast") and s.finish_reason == "stop"
      and outcomes(rows) == ["finished", "won"] and rc.get("win_at") == "reasoning" and rc.get("first_reasoning") == 1
      and near(rc["t_win"], rows[1]["t_answer"]) and rq["t_first_token"] >= rows[0]["t_end"] - 0.002
      and [m["outcome"] for m in mock_rows("slow", rows[0]["seed"])] == ["done"],
      "count mode with --win-at reasoning: the first whole reply to end its reasoning, after the slow one ended")
tag = "wait/t-wait/a1/t1/1"
st, raw, s = ask(21, tag, seed=60)  # lenA writes its reasoning 21 times, lenB (seed 104789) 30 times, four times slower
rows = samples("rwait", tag, 2)
rc = request_row("rwait", tag).get("race", {})
check(st == 200 and s.reply()["reasoning_content"].startswith("[lenA seed 60]") and outcomes(rows) == ["won", "cancelled"]
      and rows[1]["t_end"] >= rows[0]["t_end"] + 0.3 - 0.002 and rows[1]["predicted_n"] >= 1
      and rows[1]["finish_reason"] is None and rc.get("first") == 0 and rc["pair"][1]["outcome"] == "cancelled"
      and [m["outcome"] for m in mock_rows("lenB", 104789)] == ["cancelled"],
      f"--count-wait-s 0.3: a sample still running 0.3 s after the first whole reply is stopped "
      f"({rows[0]['t_end']} s, {rows[1]['t_end']} s)")

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
check("## Count mode" not in sm.stdout, "summary: no count tables for a race without --count")
sm = subprocess.run([sys.executable, "-I", str(here / "race_proxy.py"), "summary", str(out / "rcount/proxy.samples.jsonl"),
                     str(out / "rwait/proxy.samples.jsonl"), "--marks", "10,50"], capture_output=True, text=True)
(out / "summary_count.md").write_text(sm.stdout + sm.stderr)
cells = lambda l: [c.strip() for c in l.strip().strip("|").split("|")]
tab = {(c[0], c[1]): c for c in map(cells, sm.stdout.splitlines()) if len(c) == 10 and c[0] in ("count", "wait")}
num = lambda x: float(x.replace(",", ""))
a0, a1, rep_, rea = (tab.get(("count", k)) for k in ("server 0 alone", "server 1 alone", "the first whole reply",
                                                     "the first to end its reasoning"))
rows3 = [samples("rcount", f"count/t-count/a1/t1/{t}", 2) for t in (1, 2, 3)]
want = sum(min(r["t_end"] for r in rr) for rr in rows3)
check(sm.returncode == 0 and all((a0, a1, rep_, rea)) and a0[2] == "3" and rep_[3] == "1, 2"
      and abs(num(rep_[4]) - want) <= 0.06 and num(rep_[4]) < min(num(a0[4]), num(a1[4]))
      and num(rep_[5]) < 1 and num(rep_[7]) < 1 and num(rep_[9]) < 1
      and int(rep_[8].replace(",", "")) == sum(min(rr, key=lambda r: r["t_end"])["predicted_n"] for rr in rows3),
      f"summary, count tables: the first whole reply's seconds and tokens are each request's sooner sample's "
      f"({rep_[4] if rep_ else None} s against {a0[4] if a0 else None} and {a1[4] if a1 else None} alone)")
late = [c for c in map(cells, sm.stdout.splitlines()) if len(c) == 8 and c[0] == "count" and c[1] == "0"]
check(len(late) == 2 and [c[2] for c in late] == ["10", "50"] and all(int(c[4]) >= 1 for c in late)
      and all(num(c[7]) <= 1 for c in late) and num(late[0][6]) <= num(late[1][6]) + 0.05
      and tab.get(("wait", "the first whole reply"), [""] * 4)[3] == "1, 0"
      and "wait: 1 of 1 requests, 1 of them with a sample stopped early" in sm.stdout,
      "summary, count tables: a late second sample at each mark; the stopped sample is counted as stopped early")
sys.exit(1 if fails else 0)
EOF
echo "race selftest passed; logs in $OUT (summary.md, summary_count.md)"
