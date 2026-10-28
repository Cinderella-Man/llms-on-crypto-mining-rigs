"""Calibration text for an importance matrix: agentic run 1's sessions (Flint-written, gated, publishable) that are not
among the 40 held-out sessions the agent-text KL divergence is scored on (../../2026-10-27_ornith_kld/run/agent_text.py
writes those), in the same plain format: each message becomes "### <role>" and its reasoning, text and tool calls,
sessions separated by a blank line. The system prompt, the same in every session, is left out, and each session gives
at most PER_SESSION characters from its start, so the text spans many sessions; sessions in file order until
MAX_CHARS characters.

usage: python3 -I calib_text.py SESSIONS.jsonl TEST_ROWS.jsonl OUT.txt [MAX_CHARS 1500000] [PER_SESSION 12000]
"""
import json
import sys


def main():
    sessions, rows, out = sys.argv[1:4]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 1_500_000
    per = int(sys.argv[5]) if len(sys.argv) > 5 else 12_000
    held_out = {json.loads(line)["session_id"] for line in open(rows)}
    n = chars = 0
    with open(out, "w") as f:
        for line in open(sessions):
            s = json.loads(line)
            if s["session_id"] in held_out:
                continue
            parts = []
            for m in s["messages"]:
                if m["role"] == "system":
                    continue
                body = [m.get("reasoning_content") or "", m.get("content") or ""]
                for c in m.get("tool_calls") or []:
                    body.append(f'{c["function"]["name"]}({c["function"]["arguments"]})')
                parts.append(f"### {m['role']}\n" + "\n".join(b for b in body if b))
            text = "\n".join(parts)[:per] + "\n\n"
            f.write(text)
            n += 1
            chars += len(text)
            if chars >= limit:
                break
    print(f"{n} sessions (none of the {len(held_out)} held out), {chars / 1e6:.2f} M characters -> {out}")


if __name__ == "__main__":
    main()
