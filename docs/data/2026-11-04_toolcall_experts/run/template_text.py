"""FN4's calibration sessions as the model reads them. calib_text.py (../../2026-10-28_ornith35_kld/run/) wrote FN4's
text in a plain format, tool calls as `name({json})` lines; at work the model reads and writes them in its chat
template: `<tool_call>`, `<function=…>`, `<parameter=…>` tags and `<tool_response>` blocks. This writes the same sessions
(the ones calib_text.py picks: agentic run 1's sessions outside the 40 held out, in file order, until its plain text
reaches MAX_CHARS) through Qwen3.8-Flash-Next's chat template (bench/copy_spans/common.py's render_prompt, which
matches the Jinja template byte for byte), without the system block (the same in every session; calib_text.py leaves
the system prompt out too). Like calib_text.py, a session gives at most PER_SESSION characters from its start; a cut
inside a message closes it with `<|im_end|>`, so a cut tool call or result ends there. With RESPONSE_MAX > 0 each tool
result keeps only its first RESPONSE_MAX characters (a whole file read takes most of a session's characters
otherwise, so the cap puts more of each session's turns and tool calls into the text). Tokenize it with
--parse-special, as the server does.

usage: python3 -I template_text.py SESSIONS.jsonl TEST_ROWS.jsonl OUT.txt [MAX_CHARS 1500000] [PER_SESSION 12000]
                                   [RESPONSE_MAX 0]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "bench", "copy_spans")))
from common import render_prompt  # noqa: E402

END = "<|im_end|>\n"


def plain_len(s, per):
    """The length calib_text.py gives this session (its selection rule)."""
    parts = []
    for m in s["messages"]:
        if m["role"] == "system":
            continue
        body = [m.get("reasoning_content") or "", m.get("content") or ""]
        for c in m.get("tool_calls") or []:
            body.append(f'{c["function"]["name"]}({c["function"]["arguments"]})')
        parts.append(f"### {m['role']}\n" + "\n".join(b for b in body if b))
    return len("\n".join(parts)[:per] + "\n\n")


def main():
    sessions, rows, out = sys.argv[1:4]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 1_500_000
    per = int(sys.argv[5]) if len(sys.argv) > 5 else 12_000
    resp_max = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    held_out = {json.loads(line)["session_id"] for line in open(rows)}
    n = chars = plain = whole = calls = 0
    with open(out, "w") as f:
        for line in open(sessions):
            s = json.loads(line)
            if s["session_id"] in held_out:
                continue
            msgs = s["messages"]
            if resp_max:
                msgs = [dict(m, content=m["content"][:resp_max]) if m["role"] == "tool" and isinstance(m.get("content"), str)
                        else m for m in msgs]
            text, _quotable = render_prompt({"messages": msgs, "tools": s.get("tools")}, generation_prompt=False)
            text = text[text.index(END) + len(END):]  # the system block, with the tool list
            if len(text) > per:
                text = text[:per]
                if not text.endswith(END):
                    text += END
            else:
                whole += 1
            f.write(text)
            calls += text.count("<tool_call>")
            n += 1
            chars += len(text)
            plain += plain_len(s, per)
            if plain >= limit:
                break
    print(f"{n} sessions (none of the {len(held_out)} held out; {whole} whole), {calls} tool calls, "
          f"{chars / 1e6:.2f} M characters -> {out}")


if __name__ == "__main__":
    main()
