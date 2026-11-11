#!/usr/bin/env python3
"""C693 (Q1226): llama-server's draft-mtp on the CPU backend with each value of the two switches.

Every arm starts llama-server with the tiny qwen4exp model that has one MTP block (test-llama-archs's random weights,
saved with -o <dir> --mtp), drafts three tokens a step with the model's own MTP layer, and answers the same three
requests: greedy 96 tokens, temperature 1.0 with seed 42 for 48 tokens, greedy 64 tokens on a longer prompt. The prompts
are read in calls of --batch tokens, so a prompt takes several calls and the draft catches up after each.

An arm is <name>=<llama-server binary>[,<VAR>=<value>...]. The arm named base is the reference: every other arm must
give its tokens and its drafted and accepted counts for every request. The server log also shows where the target kept
the h rows the draft reads: a line "h_nextn buffer size" when they are in the device buffer (on the CPU only with
LLAMA_MTP_H_DEV=2 or with LLAMA_MTP_CATCHUP_OVERLAP's two halves), and "LLAMA_MTP_H_DEV=0, extracting" when the
switch keeps them on the host. The servers log at verbosity 4, where llama-server shows libllama's info lines.

Usage: python3 -I server_smoke.py --model <gguf> --out <dir> [--port 18693] [--threads 4] [--batch 64] [--ubatch 32]
           --arm base=<bin> --arm <name>=<bin>[,VAR=value...] ...
Output: <out>/<arm>.server.log, <out>/<arm>.<request>.json, <out>/replies.tsv (one row per arm and request),
        <out>/arms.tsv (one row per arm: identical replies, the log's markers, what the arm's switches expect, verdict).
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request

MARKERS = {
    "buffer": "h_nextn buffer size",
    "host0": "LLAMA_MTP_H_DEV=0, extracting",
    "no_events": "the device has no events",
    "overlap_on": "LLAMA_MTP_CATCHUP_OVERLAP: a prompt call's catch-up runs",
    "overlap_off": "LLAMA_MTP_CATCHUP_OVERLAP: needs the target's h rows",
}


def toks(n, s, v):
    """n deterministic token ids in [1, v - 1] (a Park-Miller generator, as C405's smoke check)."""
    x, out = s, []
    for _ in range(n):
        x = (x * 16807) % 2147483647
        out.append(1 + x % (v - 1))
    return out


def http(port, path, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def expected(arm_env, base):
    """What the arm's switches should do on a CPU build: the h rows in the device buffer, the host0 line."""
    if base:
        return {"buffer": False, "host0": False}
    h = arm_env.get("LLAMA_MTP_H_DEV", "1")
    ovl = arm_env.get("LLAMA_MTP_CATCHUP_OVERLAP", "0") not in ("", "0")
    return {"buffer": h == "2" or (ovl and h != "0"), "host0": h == "0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=18693)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ubatch", type=int, default=32)
    ap.add_argument("--arm", action="append", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    arms = []
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        parts = rest.split(",")
        env = dict(p.split("=", 1) for p in parts[1:] if p)
        arms.append((name, parts[0], env))
    assert arms[0][0] == "base", "the first arm must be base"

    replies = {}
    rows = []
    for name, binary, env in arms:
        log_path = os.path.join(a.out, f"{name}.server.log")
        cmd = [binary, "-m", a.model, "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "-ngl", "0",
               "-t", str(a.threads), "-c", "4096", "-b", str(a.batch), "-ub", str(a.ubatch), "-np", "1",
               "--no-webui", "--host", "127.0.0.1", "--port", str(a.port),
               "-lv", "4"]  # libllama's info lines (the buffer's size, the switch's line) are at verbosity 4 here
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env})
        try:
            for _ in range(180):
                if proc.poll() is not None:
                    raise RuntimeError(f"{name}: server exited with {proc.returncode}")
                try:
                    if http(a.port, "/health", timeout=5).get("status") == "ok":
                        break
                except Exception:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError(f"{name}: server did not start")
            nv = http(a.port, "/v1/models")["data"][0]["meta"]["n_vocab"]
            ta, tb, tc = toks(40, 3, nv), toks(30, 5, nv), toks(20, 9, nv)
            p1 = ta + tb + ta + tc  # a block repeated: something a drafter can copy
            p3 = p1 + tb
            for req, prompt, n, temp, seed in (("R1", p1, 96, 0, 1), ("R2", p1, 48, 1.0, 42), ("R3", p3, 64, 0, 1)):
                body = {"prompt": prompt, "n_predict": n, "temperature": temp, "top_k": 20, "top_p": 0.95, "min_p": 0,
                        "seed": seed, "return_tokens": True, "cache_prompt": False}
                r = http(a.port, "/completion", body)
                with open(os.path.join(a.out, f"{name}.{req}.json"), "w") as f:
                    json.dump(r, f)
                t = r.get("timings", {})
                replies[(name, req)] = (r.get("tokens"), t.get("draft_n"), t.get("draft_n_accepted"))
                print(f"{name} {req} n={len(r.get('tokens') or [])} drafted={t.get('draft_n')} accepted={t.get('draft_n_accepted')}",
                      flush=True)
        except Exception as e:
            print(f"{name}: ERROR {e}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        with open(log_path, errors="replace") as f:
            text = f.read()
        found = {k: m in text for k, m in MARKERS.items()}
        exp = expected(env, name == "base")
        n_same = 0
        for req in ("R1", "R2", "R3"):
            mine, ref = replies.get((name, req)), replies.get(("base", req))
            same = mine is not None and ref is not None and mine[0] is not None and mine == ref
            n_same += same
            rows.append((name, req, "" if mine is None else len(mine[0] or []), "" if mine is None else mine[1],
                         "" if mine is None else mine[2], "identical" if same else "DIFFERENT"))
        where_ok = found["buffer"] == exp["buffer"] and found["host0"] == exp["host0"]
        verdict = "OK" if n_same == 3 and where_ok else "FAIL"
        arms_row = (name, ",".join(f"{k}={v}" for k, v in env.items()) or "-", f"{n_same}/3",
                    "yes" if found["buffer"] else "no", "yes" if exp["buffer"] else "no",
                    "yes" if found["host0"] else "no", "yes" if exp["host0"] else "no",
                    "yes" if found["no_events"] else "no",
                    "on" if found["overlap_on"] else ("off" if found["overlap_off"] else "-"), verdict)
        with open(os.path.join(a.out, "arms.tsv"), "a") as f:
            if f.tell() == 0:
                f.write("arm\tswitches\tidentical_to_base\tdevice_buffer\tdevice_buffer_expected\thost0_line\t"
                        "host0_expected\tno_events_line\toverlap\tverdict\n")
            f.write("\t".join(str(x) for x in arms_row) + "\n")
        print("ARM\t" + "\t".join(str(x) for x in arms_row), flush=True)

    with open(os.path.join(a.out, "replies.tsv"), "w") as f:
        f.write("arm\trequest\tn_tokens\tdraft_n\tdraft_n_accepted\tagainst_base\n")
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")


if __name__ == "__main__":
    main()
