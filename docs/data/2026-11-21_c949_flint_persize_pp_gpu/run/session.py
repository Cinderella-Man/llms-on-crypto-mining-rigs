#!/usr/bin/env python3
"""C949 (Q1687): one coding agent's tool-call session against llama-server's /completion endpoint.

Eight requests, each greedy (temperature 0) but R3 (temperature 0.7, top-k 8, seed 7), with the generated token ids
returned. Every tool call is constrained by a JSON schema (a grammar from the first token, so the server samples one
token a step); R1 and R8 are plain text. R4 to R8 are later turns of one conversation that grows by the model's own
reply and a tool result each turn, with the prompt cache on; the results are sized so that one call is under 64 new
tokens, two are a few hundred and one is over 600 (Flint's preset splits prompts of up to 600 tokens into micro-batches
of 64 and longer ones into micro-batches of 128).

  R1  plain text, a Python snippet continued, 320 tokens (ignore_eos), no prompt cache
  R2  the task, the first tool call (schema), up to 96 tokens, no prompt cache
  R3  R2's prompt at temperature 0.7, seed 7 (schema), no prompt cache
  R4  + R2's reply + a directory listing, the next tool call (schema), prompt cache on
  R5  + R4's reply + a ~45-line module, the next tool call (schema)
  R6  + R5's reply + a ~100-line test file, the next tool call (schema)
  R7  + R6's reply + a one-line result, the next tool call (schema)
  R8  + R7's reply + a short result and a question, 128 tokens of plain text
  R9, R10, R11  fresh tool-call prompts (schema, no prompt cache) of exactly 264, 645 and 648 tokens, padded with the
      server's /tokenize, so that a prompt call ends in a micro-batch of 4, 1 and 4 tokens behind earlier ones

Usage: python3 -I session.py <server URL> <output folder> <round tag>
Writes <out>/<round>-<request>.json (the server's reply) and appends one line per request to <out>/session.tsv:
round, request, kind, prompt characters, generated tokens, prompt tokens read, cached tokens, the md5 of the token ids
(first 12 hex digits) and the reply's first 80 characters (JSON-escaped). Nothing timed is written.
"""
import hashlib
import json
import os
import sys
import urllib.request

URL, OUT, ROUND = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]

# the tool call's schema, as in C903's processor check (docs/data/2026-11-21_c903_persize_pipeline/run/c903_server.sh)
SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"enum": ["read_file", "write_file", "list_dir"]},
        "arguments": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "maxLength": 16},
                "line": {"type": "integer", "minimum": 0, "maximum": 999},
            },
            "required": ["path", "line"],
        },
    },
    "required": ["name", "arguments"],
}

P1 = (
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n    if n < 2:\n        return n\n'
    "    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b\n\nprint(fibonacci(10))\n# next:"
)

TASK = (
    "You are a coding agent working in a Python repository. Each step, call one tool as JSON and wait for its result.\n"
    "Task: the test tests/test_stats.py::test_mean_empty fails with ZeroDivisionError. Find the bug and fix it.\n"
    "Call the first tool as JSON:"
)

LISTING = (
    "src/\n  __init__.py\n  io_utils.py\n  stats.py\ntests/\n  test_io_utils.py\n  test_stats.py\n"
    "README.md\npyproject.toml\n"
)

STATS = '''"""Small statistics helpers used by the report scripts."""
import math


def mean(values):
    """Arithmetic mean of a list of numbers."""
    total = 0.0
    for v in values:
        total += v
    return total / len(values)


def median(values):
    """Middle value of a list of numbers (the mean of the two middle ones for an even count)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def variance(values):
    """Population variance."""
    if not values:
        return 0.0
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def stdev(values):
    """Population standard deviation."""
    return math.sqrt(variance(values))


def clamp(value, low, high):
    """Limit value to the range [low, high]."""
    if low > high:
        raise ValueError("low must not exceed high")
    return max(low, min(high, value))
'''


def tests_file():
    lines = ["import math", "", "import pytest", "", "from src.stats import clamp, mean, median, stdev, variance", ""]
    for i in range(1, 21):
        data = ", ".join(str((i * k) % 17) for k in range(1, 6))
        lines += [
            "",
            f"def test_mean_{i}():",
            f"    data = [{data}]",
            "    assert math.isclose(mean(data), sum(data) / len(data))",
        ]
    lines += ["", "", "def test_mean_empty():", "    assert mean([]) == 0.0", ""]
    lines += ["", "def test_clamp_order():", "    with pytest.raises(ValueError):", "        clamp(1, 5, 2)", ""]
    return "\n".join(lines) + "\n"


TESTS = tests_file()


def complete(name, kind, prompt, n, temp=0.0, seed=1, schema=True, cache=False, ignore_eos=False):
    body = {
        "prompt": prompt, "n_predict": n, "temperature": temp, "top_k": 8, "top_p": 1.0, "min_p": 0.0,
        "seed": seed, "return_tokens": True, "cache_prompt": cache,
    }
    if schema:
        body["json_schema"] = SCHEMA
    if ignore_eos:
        body["ignore_eos"] = True
    req = urllib.request.Request(URL + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            reply = json.load(r)
    except Exception as e:  # recorded, so a failed request shows in the table
        reply = {"error": str(e)}
    with open(os.path.join(OUT, f"{ROUND}-{name}.json"), "w") as f:
        json.dump(reply, f)
    toks = reply.get("tokens") or []
    md5 = hashlib.md5(json.dumps(toks, separators=(",", ":")).encode()).hexdigest()[:12]
    t = reply.get("timings") or {}
    row = [ROUND, name, kind, str(len(prompt)), str(len(toks)), str(t.get("prompt_n", "")), str(t.get("cache_n", "")),
           md5, json.dumps(str(reply.get("content", reply.get("error", "")))[:80])]
    with open(os.path.join(OUT, "session.tsv"), "a") as f:
        f.write("\t".join(row) + "\n")
    print(" ".join(row[:8]), flush=True)
    return str(reply.get("content", ""))


os.makedirs(OUT, exist_ok=True)
complete("R1", "plain", P1, 320, schema=False, ignore_eos=True)
r2 = complete("R2", "tool", TASK, 96)
complete("R3", "tool t0.7", TASK, 96, temp=0.7, seed=7)
h = TASK + " " + r2 + "\nResult:\n" + LISTING + "Call the next tool as JSON:"
r4 = complete("R4", "tool turn 2", h, 96, cache=True)
h += " " + r4 + "\nResult:\n" + STATS + "Call the next tool as JSON:"
r5 = complete("R5", "tool turn 3", h, 96, cache=True)
h += " " + r5 + "\nResult:\n" + TESTS + "Call the next tool as JSON:"
r6 = complete("R6", "tool turn 4", h, 96, cache=True)
h += " " + r6 + "\nResult: ok.\nCall the next tool as JSON:"
r7 = complete("R7", "tool turn 5", h, 96, cache=True)
h += " " + r7 + "\nResult: written.\nIn two sentences, explain the fix:\n"
complete("R8", "plain turn 6", h, 128, schema=False, cache=True)


def tokens(text):
    req = urllib.request.Request(URL + "/tokenize", data=json.dumps({"content": text, "add_special": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return len(json.load(r)["tokens"])


def padded(target):
    """A tool-call prompt of exactly `target` tokens (the server's own tokenizer, BOS included): the task, then as many
    words of the module and the test file as fit, then the call; up to 15 words of filler close the gap."""
    head, tail = TASK + "\nResult:\n", "\nCall the next tool as JSON:"
    words = (STATS + TESTS + STATS + TESTS).split(" ")
    lo, hi = 0, len(words)
    while lo < hi:  # the fewest words whose prompt reaches the target
        mid = (lo + hi) // 2
        if tokens(head + " ".join(words[:mid]) + tail) >= target:
            hi = mid
        else:
            lo = mid + 1
    for w in (lo, lo - 1, lo - 2):
        for j in range(16):
            p = head + " ".join(words[:w]) + " a" * j + tail
            if tokens(p) == target:
                return p
    return None


# Fresh prompts (no prompt cache) sized so that a prompt call ends in a micro-batch of 1 to 8 tokens behind earlier
# micro-batches of the same call, still on the cards: the server sends a prompt's last 4 tokens as a call of their own,
# and calls of at most 512 tokens (-b 512). R9: 264 tokens, a call of 260 in micro-batches of 64 (prompts of up to 600
# tokens; Flint's taper); R10 and R11: 645 and 648 tokens, calls of 512 and then 129 and 132 in micro-batches of 128
# (128 + 1 and 128 + 4).
for name, target in (("R9", 264), ("R10", 645), ("R11", 648)):
    p = padded(target)
    if p is None:
        print(f"{ROUND} {name}: no prompt of exactly {target} tokens found", flush=True)
        continue
    complete(name, f"tool {target} tokens", p, 96)
