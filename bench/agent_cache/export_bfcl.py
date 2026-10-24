"""Multi-turn agent sessions for the cache replay (Q111), from BFCL v4 multi-turn runs of Flint.

    python export_bfcl.py RAW.jsonl SESSIONS.jsonl [--template TEMPLATE.jinja] [--min-steps 4]

RAW.jsonl is a `multi_turn.py run` output (bench/bfcl_flint/): per attempt, Flint's raw reply of every step. Each
attempt is replayed through the same conversation loop (BFCL's Python backends regenerate the tool results), and every
request is written as an OpenAI chat-completions client would send it:
- `tools`: the tool list of each step (an index into the session's distinct lists; miss_func turns add functions);
- `add`: the messages appended before the step's request, after the previous step's reply: tool results, a new user
  message;
- `reply`: the recorded reply as the client sends it back (reasoning_content, content, tool_calls with JSON string
  arguments and ids), used by the replay when its live reply has a different number of calls;
- `n_prompt`, `n_gen`: the recorded prompt and reply lengths in tokens.
Needs bfcl_eval (the quant machine's BFCL venv); the replay itself needs only Python's standard library.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bfcl_flint"))
import bfcl_common as bc  # noqa: E402
import multi_turn as mt  # noqa: E402


def oai_reply(msg, step_idx):
    """bfcl_common.assistant_message as a chat-completions client sends it: arguments as JSON text, call ids."""
    out = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("reasoning_content"):
        out["reasoning_content"] = msg["reasoning_content"]
    if msg.get("tool_calls"):
        out["tool_calls"] = [{"id": f"call_{step_idx}_{i}", "type": "function",
                              "function": {"name": tc["function"]["name"],
                                           "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False)}}
                             for i, tc in enumerate(msg["tool_calls"])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--template", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bfcl_flint",
                                                       "template.jinja"))
    ap.add_argument("--min-steps", type=int, default=4)
    args = ap.parse_args()

    tmpl = bc.make_template(open(args.template).read())
    entries = {e["id"]: e for e in mt.load_entries(exclude=mt.read_ids(mt.EXCLUDED))}
    out = open(args.out, "w")
    n_sessions = n_requests = 0
    for line in open(args.raw):
        rec = json.loads(line)
        steps = [s for t in rec["turns"] for s in t]
        if rec["id"] not in entries or len(steps) < args.min_steps or any(s.get("error") for s in steps):
            continue
        stored = iter(steps)
        snaps = []  # (messages so far, tools) at each request

        def generate(prompt, messages, tools):
            snaps.append((json.loads(json.dumps(messages)), tools))
            return dict(next(stored))

        res = mt.converse(entries[rec["id"]], tmpl, generate, f"export_a{rec['attempt']}")
        if next(stored, None) is not None or [len(t) for t in res["turns"]] != [len(t) for t in rec["turns"]]:
            print(f"skip {rec['id']} attempt {rec['attempt']}: the replay does not follow the stored steps")
            continue

        tool_lists, out_steps, n_seen = [], [], 0
        for i, ((messages, tools), st) in enumerate(zip(snaps, steps)):
            if tools not in tool_lists:
                tool_lists.append(tools)
            # the messages before this request: everything after the previous request's messages and its reply
            add = messages[n_seen + (1 if i > 0 else 0):]
            parsed = bc.split_output(st["content"], tools)
            reply = oai_reply(bc.assistant_message(parsed), i)
            prev = out_steps[-1]["reply"].get("tool_calls", []) if out_steps else []
            k = 0
            for m in add:
                if m["role"] == "tool":
                    m["tool_call_id"] = prev[k]["id"] if k < len(prev) else f"call_{i - 1}_{k}"
                    k += 1
            out_steps.append({"tools": tool_lists.index(tools), "add": add, "reply": reply,
                              "n_prompt": st["n_prompt"], "n_gen": st["tokens_predicted"]})
            n_seen = len(messages)
        out.write(json.dumps({"id": f"{rec['id']}/a{rec['attempt']}", "classes": entries[rec["id"]]["involved_classes"],
                              "tool_lists": tool_lists, "steps": out_steps}, ensure_ascii=False) + "\n")
        n_sessions += 1
        n_requests += len(out_steps)
    print(f"{n_sessions} sessions, {n_requests} requests -> {args.out}")


if __name__ == "__main__":
    main()
