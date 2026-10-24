"""Exactness of the agent cache paths (Q111): every request to the server under test must give the same top
log-probabilities, bit for bit, as a cold server that processes the whole prompt.

    python check_exact.py --test http://127.0.0.1:PORT --cold http://127.0.0.1:PORT2 --sessions SESSIONS.jsonl
                          [--scenarios prefix,turns,trim,gen] [--restart-cmd CMD] [--gen-at TEXT]

The cold server gets every request with cache_prompt off (a fresh slot each time: the same micro-batches as a server
that has just started). Scenarios (each prints the server's cache path and PASS/FAIL per request):
- prefix: a session's system and tool prompt with one user message, then another system prompt (the slot is replaced),
  then the first system prompt with a second user message (restored from --prefix-cache), then, after --restart-cmd
  restarted the test server, a third user message (restored from the file);
- turns: a multi-turn chat that sends each reply back as the next request's history (reuse of the generated tokens);
- trim: raw token prompts on /completion that diverge inside the live window, before the end-of-prompt checkpoint
  (restored and cut back), and nowhere usable;
- gen: a long reply whose re-sent copy diverges after a generation checkpoint (--checkpoint-gen with --gen-at).
Greedy decoding, 8 tokens per request unless stated, n_probs 8. Exit status 1 if any request differs.
"""
import argparse
import json
import random
import subprocess
import sys
import time
import urllib.request

FAILS = []


def post(url, path, body, timeout=1800):
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(url, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    sys.exit(f"{url} not ready")


def probs_of(res):
    """The top log-probabilities of every generated token, from /completion or /v1/chat/completions."""
    if "completion_probabilities" in res:
        return [(p.get("id"), [(t.get("id"), t.get("logprob")) for t in p.get("top_logprobs", [])])
                for p in res["completion_probabilities"]]
    lp = res["choices"][0].get("logprobs") or {}
    return [(p.get("token"), [(t.get("token"), t.get("logprob")) for t in p.get("top_logprobs", [])])
            for p in lp.get("content", [])]


def timings(res):
    return res.get("timings", {})


def first_token_gap(a, b):
    """Largest |log-probability difference| among the first token's top candidates present in both, and whether the
    top candidate is the same: the distribution after the prompt, before any sampled token can differ."""
    if not a or not b:
        return float("nan"), False
    da, db = dict(a[0][1]), dict(b[0][1])
    common = set(da) & set(db)
    gap = max((abs(da[k] - db[k]) for k in common), default=float("nan"))
    return gap, bool(a[0][1]) and bool(b[0][1]) and a[0][1][0][0] == b[0][1][0][0]


def compare(name, test_res, cold_res):
    a, b = probs_of(test_res), probs_of(cold_res)
    t = timings(test_res)
    ok = a == b and len(a) > 0
    first_diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    gap, same_top = first_token_gap(a, b)
    extra = "" if ok else f" first_diff={first_diff} len_cold={len(b)} first_token_gap={gap:.2e} same_top1={same_top}"
    print(f"{'PASS' if ok else 'FAIL'} {name}: cache_src={t.get('cache_src')} cache_n={t.get('cache_n')} "
          f"prompt_n={t.get('prompt_n')} ckpt_n={t.get('cache_ckpt_n')} restore_ms={t.get('cache_restore_ms')} "
          f"tokens={len(a)}{extra}", flush=True)
    if not ok:
        FAILS.append(name)
    return t


def chat(url, messages, tools, cache, n=8, extra=None):
    body = {"messages": messages, "max_tokens": n, "temperature": 0, "logprobs": True, "top_logprobs": 8,
            "cache_prompt": cache, "stream": False}
    if tools:
        body["tools"] = tools
        body["parallel_tool_calls"] = True
    body.update(extra or {})
    return post(url, "/v1/chat/completions", body)


def complete(url, tokens, cache, n=8, extra=None):
    body = {"prompt": tokens, "n_predict": n, "temperature": 0, "n_probs": 8, "cache_prompt": cache,
            "stream": False, "return_tokens": True}
    body.update(extra or {})
    return post(url, "/completion", body)


def slot(n):
    return {"id_slot": n} if n is not None else {}


def both_chat(args, name, messages, tools, n=8):
    t = chat(args.test, messages, tools, True, n, slot(args.test_slot))
    c = chat(args.cold, messages, tools, False, n, slot(args.cold_slot))
    return compare(name, t, c), t


def both_complete(args, name, tokens, n=8, extra=None):
    t = complete(args.test, tokens, True, n, {**(extra or {}), **slot(args.test_slot)})
    c = complete(args.cold, tokens, False, n, {**(extra or {}), **slot(args.cold_slot)})
    return compare(name, t, c), t


def load_session(path, i):
    with open(path) as f:
        for k, line in enumerate(f):
            if k == i:
                return json.loads(line)
    sys.exit(f"no session {i} in {path}")


def scen_prefix(args):
    s1 = load_session(args.sessions, args.session)
    s2 = load_session(args.sessions, args.session + 1)
    tools1, tools2 = s1["tool_lists"][0], s2["tool_lists"][0]
    sys1 = {"role": "system", "content": "You are a careful assistant that uses tools. Session A."}
    sys2 = {"role": "system", "content": "You are a different assistant. Session B."}
    rnd = random.Random(1)
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()

    def user(k):
        return {"role": "user", "content": f"Task {k}: " + " ".join(rnd.choice(words) for _ in range(60))}

    both_chat(args, "prefix/first (saves the prefix)", [sys1, user(1)], tools1)
    both_chat(args, "prefix/other system prompt", [sys2, user(2)], tools2)
    t, _ = both_chat(args, "prefix/first prompt again (restore)", [sys1, user(3)], tools1)
    if args.expect_restore and t.get("cache_src") != "prefix":
        FAILS.append("prefix/restore not used")
        print("FAIL prefix: the saved prefix was not restored", flush=True)
    if args.restart_cmd:
        subprocess.run(args.restart_cmd, shell=True, check=True)
        wait_ready(args.test)
        t, _ = both_chat(args, "prefix/after restart (restore from file)", [sys1, user(4)], tools1)
        if args.expect_restore and t.get("cache_src") != "prefix":
            FAILS.append("prefix/restore after restart not used")
            print("FAIL prefix: the saved prefix was not restored after the restart", flush=True)


def scen_prefix_again(args):
    """After a restart of the test server: the first system prompt with a new user message, from the saved file."""
    s1 = load_session(args.sessions, args.session)
    sys1 = {"role": "system", "content": "You are a careful assistant that uses tools. Session A."}
    t, _ = both_chat(args, "prefix/after restart (restore from file)",
                     [sys1, {"role": "user", "content": "Task 9: summarize what the tools can do, briefly."}],
                     s1["tool_lists"][0])
    if args.expect_restore and t.get("cache_src") != "prefix":
        FAILS.append("prefix/restore after restart not used")
        print("FAIL prefix: the saved prefix was not restored after the restart", flush=True)


def scen_pick(args):
    """Prints the text of a token that the gen scenario's greedy reply contains around position 300 (for --gen-at)."""
    rnd = random.Random(3)
    p0 = [rnd.randrange(1000, 60000) for _ in range(2000)]
    gen = complete(args.cold, p0, False, 400, {"ignore_eos": True, **slot(args.cold_slot)}).get("tokens") or []
    for tok in gen[300:] + gen[100:300]:
        text = post(args.cold, "/detokenize", {"tokens": [tok]}).get("content", "")
        back = post(args.cold, "/tokenize", {"content": text, "add_special": False, "parse_special": True}).get("tokens", [])
        if text.strip() and "," not in text and back == [tok]:
            print(f"GEN_AT={json.dumps(text)} id={tok} first_at={gen.index(tok)} count={gen.count(tok)}", flush=True)
            return
    sys.exit("pick: no usable token")


def scen_bs(args):
    """Requests without token probabilities (backend sampling stays on with -bs): prints each reply's text."""
    s = load_session(args.sessions, args.session)
    for i in range(3):
        res = post(args.test, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": f"Request {i}: list three tools you have."}],
            "tools": s["tool_lists"][0], "max_tokens": 24, "temperature": 0, "cache_prompt": True, **slot(args.test_slot)})
        print(f"BS {i} {json.dumps(res['choices'][0]['message'].get('content'))} "
              f"cache_n={timings(res).get('cache_n')}", flush=True)


def scen_turns(args):
    s = load_session(args.sessions, args.session)
    tools = s["tool_lists"][0]
    messages = [{"role": "system", "content": "You are a careful assistant that uses tools."}]
    for i, st in enumerate(s["steps"][:args.turns]):
        # the random model makes no tool calls, so the session's user messages carry the conversation
        users = [m for m in st["add"] if m["role"] == "user"]
        messages += users or [{"role": "user", "content": f"Step {i}: go on with the task."}]
        _, t = both_chat(args, f"turns/{i}", messages, tools, n=args.turn_tokens)
        msg = t["choices"][0]["message"]
        reply = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("reasoning_content"):
            reply["reasoning_content"] = msg["reasoning_content"]
        messages.append(reply)


def scen_trim(args):
    rnd = random.Random(2)
    p0 = [rnd.randrange(1000, 60000) for _ in range(2000)]
    # the slot: p0 and 600 generated tokens; the end-of-prompt checkpoint sits 4 tokens before p0's end
    _, t = both_complete(args, "trim/p0 + 600 generated", p0, n=600, extra={"ignore_eos": True})
    gen = t.get("tokens") or []
    if len(gen) < 600:
        sys.exit("trim: the server did not return the generated tokens (return_tokens)")
    x = [rnd.randrange(1000, 60000) for _ in range(30)]
    both_complete(args, "trim/diverge 50 before the end (live window)", p0 + gen[:550] + x)
    y = [rnd.randrange(1000, 60000) for _ in range(40)]
    t, _ = both_complete(args, "trim/diverge at 1500 (checkpoint past it, cut back)", p0[:1500] + y)
    if args.expect_trim and t.get("cache_src") != "checkpoint":
        FAILS.append("trim/checkpoint not used")
        print("FAIL trim: no checkpoint restore for a divergence inside its window", flush=True)
    z = [rnd.randrange(1000, 60000) for _ in range(40)]
    both_complete(args, "trim/diverge at 300 (no usable cells)", p0[:300] + z)
    both_complete(args, "trim/exact repeat", p0[:300] + z)


def scen_gen(args):
    rnd = random.Random(3)
    p0 = [rnd.randrange(1000, 60000) for _ in range(2000)]
    _, t = both_complete(args, "gen/p0 + 1500 generated", p0, n=1500, extra={"ignore_eos": True})
    gen = t.get("tokens") or []
    z = [rnd.randrange(1000, 60000) for _ in range(40)]
    d = args.gen_diverge
    t, _ = both_complete(args, f"gen/diverge {d} tokens into the reply", p0 + gen[:d] + z)
    print(f"     generation checkpoints created in the long reply: see the server log; restored n_tokens = "
          f"{t.get('cache_ckpt_n')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--cold", required=True)
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--session", type=int, default=0)
    ap.add_argument("--scenarios", default="prefix,turns,trim,gen")
    ap.add_argument("--restart-cmd")
    ap.add_argument("--expect-restore", action="store_true")
    ap.add_argument("--expect-trim", action="store_true")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--turn-tokens", type=int, default=48)
    ap.add_argument("--gen-diverge", type=int, default=400)
    ap.add_argument("--test-slot", type=int, help="id_slot of the test requests (one server for both: -np 2)")
    ap.add_argument("--cold-slot", type=int, help="id_slot of the cold requests")
    args = ap.parse_args()
    wait_ready(args.test)
    wait_ready(args.cold)
    for s in args.scenarios.split(","):
        {"prefix": scen_prefix, "prefix_again": scen_prefix_again, "turns": scen_turns, "trim": scen_trim,
         "gen": scen_gen, "pick": scen_pick, "bs": scen_bs}[s](args)
    print(f"{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
