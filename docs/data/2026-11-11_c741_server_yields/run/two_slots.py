#!/usr/bin/env python3
"""C741 (Q1310): llama-server with two slots: request A, then request B while A is being answered, and reads of the
server state (/slots and /metrics) all along.

  python3 -I two_slots.py --port P --a A.json --b B.json --out DIR [--after 16] [--poll-ms 2]

A.json and B.json are /completion request bodies. A is posted at once. A poller GETs /slots and /metrics in turn, one
read every poll-ms; the first /slots answer that shows a slot processing with n_decoded >= after posts B (once). The
poller stops when A and B are answered. If A ends before that, B is posted then and client.json says so.
Times are time.monotonic() in ms: the steady clock of llama-server's step trace (LLAMA_STEP_TRACE), whose header gives
its 0 ms on that clock.
Out: DIR/A.json and DIR/B.json (the replies), DIR/reads.tsv (kind, sent, received, ms, HTTP status, slots processing
in a /slots answer), DIR/client.json (when A and B were posted and answered, whether B came during A, the reads' count).
Standard library only.
"""
import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request


def now():
    return time.monotonic() * 1000.0


def http(url, body=None, timeout=1800):
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"} if body else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # noqa: BLE001 - recorded, the check reads it
        return -1, json.dumps({"error": str(e)}).encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--after", type=int, default=16)
    ap.add_argument("--poll-ms", type=float, default=2.0)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    with open(args.a, "rb") as f:
        body_a = f.read()
    with open(args.b, "rb") as f:
        body_b = f.read()
    os.makedirs(args.out, exist_ok=True)

    replies = {}
    times = {}

    def run(name, body):
        times[name + "_post_ms"] = now()
        replies[name] = http(base + "/completion", body)
        times[name + "_done_ms"] = now()

    ta = threading.Thread(target=run, args=("a", body_a))
    ta.start()
    tb = None
    b_during_a = False
    reads = []
    i = 0
    while ta.is_alive() or (tb is not None and tb.is_alive()) or tb is None:
        if tb is None and not ta.is_alive():
            # A ended before B could be posted during it
            tb = threading.Thread(target=run, args=("b", body_b))
            tb.start()
            continue
        kind = "slots" if i % 2 == 0 else "metrics"
        i += 1
        t_s = now()
        st, data = http(base + "/" + kind, timeout=600)
        t_r = now()
        n_proc = -1
        if kind == "slots" and st == 200:
            try:
                slots = json.loads(data)
                n_proc = sum(1 for s in slots if s.get("is_processing"))
                if tb is None and any(s.get("is_processing") and (s.get("next_token") or [{}])[0].get("n_decoded", 0) >= args.after
                                      for s in slots):
                    b_during_a = True
                    tb = threading.Thread(target=run, args=("b", body_b))
                    tb.start()
            except ValueError:
                pass
        reads.append((kind, t_s, t_r, t_r - t_s, st, n_proc))
        time.sleep(args.poll_ms / 1000.0)
    ta.join()
    tb.join()

    for name in ("a", "b"):
        st, data = replies[name]
        with open(os.path.join(args.out, name.upper() + ".json"), "wb") as f:
            f.write(data)
        times[name + "_status"] = st
    with open(os.path.join(args.out, "reads.tsv"), "w") as f:
        f.write("kind\tsent_ms\treceived_ms\tms\tstatus\tprocessing\n")
        for r in reads:
            f.write(f"{r[0]}\t{r[1]:.3f}\t{r[2]:.3f}\t{r[3]:.3f}\t{r[4]}\t{r[5]}\n")
    client = dict(times)
    client.update({"b_during_a": b_during_a, "after": args.after, "poll_ms": args.poll_ms, "reads": len(reads),
                   "reads_failed": sum(1 for r in reads if r[4] != 200)})
    with open(os.path.join(args.out, "client.json"), "w") as f:
        json.dump(client, f, indent=1)
    print(json.dumps(client))
    return 0 if replies["a"][0] == 200 and replies["b"][0] == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
