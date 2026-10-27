"""Token sequences of an agent check's requests, for cost_sim.py: each request's prompt as Flint's chat template renders
it (messages and tools, then the assistant prefix), and its reply (reasoning, content, tool calls) as the assistant
turn the template renders after that prompt. Input: the agent check's proxy bodies log (bench/agent_check, one
request body and its reply per line, gzip). One output row per request: {"id", "tag", "prompt_tokens", "gen_tokens",
"predicted_n"} (predicted_n: the tokens the server reported generating, from the proxy log next to the bodies, when
it has the request).

usage: python agent_tokens.py BODIES.jsonl.gz [BODIES.jsonl.gz ...] --model HF_DIR --out SEQ.jsonl
Needs transformers (Flint's tokenizer and chat template from HF_DIR).
"""
import argparse
import gzip
import json
import os

from transformers import AutoTokenizer


def as_dict(a):
    if isinstance(a, dict):
        return a
    try:
        v = json.loads(a)
    except (TypeError, ValueError):
        return {"value": a}
    return v if isinstance(v, dict) else {"value": v}


def tool_calls(tcs):
    out = []
    for tc in tcs or []:
        fn = tc.get("function", tc)
        out.append({"type": "function", "function": {"name": fn["name"], "arguments": as_dict(fn.get("arguments", {}))}})
    return out


def text(c):
    if isinstance(c, list):   # OpenAI content parts
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c


def messages(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        m["content"] = text(m.get("content"))
        if m.get("tool_calls"):
            m["tool_calls"] = tool_calls(m["tool_calls"])
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bodies", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    n_rows, n_gen, n_pred, n_bad = 0, 0, 0, 0
    with open(args.out, "w") as fo:
        for path in args.bodies:
            meta = {}
            log = path[:-len(".bodies.gz")] if path.endswith(".bodies.gz") else None
            if log and os.path.exists(log):
                for line in open(log):
                    r = json.loads(line)
                    meta[r["id"]] = r
            for line in gzip.open(path, "rt"):
                r = json.loads(line)
                body, reply = r.get("body") or {}, r.get("reply") or (r.get("body") or {}).get("reply")
                if not body.get("messages") or reply is None:
                    continue
                msgs, tools = messages(body["messages"]), body.get("tools")
                prompt = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)
                asst = {"role": "assistant", "reasoning_content": reply.get("reasoning_content") or "",
                        "content": reply.get("content") or "", "tool_calls": tool_calls(reply.get("tool_calls"))}
                full = tok.apply_chat_template(msgs + [asst], tools=tools, tokenize=False)
                if not full.startswith(prompt):
                    n_bad += 1
                    continue
                p_ids = tok.encode(prompt, add_special_tokens=False)
                g_ids = tok.encode(full[len(prompt):], add_special_tokens=False)
                m = meta.get(r["id"], {})
                row = {"id": r["id"], "tag": m.get("tag", ""), "prompt_tokens": p_ids, "gen_tokens": g_ids,
                       "predicted_n": m.get("predicted_n")}
                fo.write(json.dumps(row) + "\n")
                n_rows += 1
                n_gen += len(g_ids)
                n_pred += m.get("predicted_n") or 0
    print(f"{n_rows} requests, {n_gen} reply tokens (server: {n_pred} generated), {n_bad} skipped")


main()
