"""Plain text of held-out agent sessions, for an agent-text KL-divergence base (llama-perplexity tokenizes it with the
model's own tokenizer). The sessions are the 40 scored held-out sessions of agentic run 1 (Flint-written, gated,
publishable), from the run's Hugging Face export; each message becomes "### <role>" and its reasoning, text and tool
calls, sessions separated by a blank line.

usage: python3 -I agent_text.py SESSIONS.jsonl TEST_ROWS.jsonl OUT.txt
"""
import json
import sys


def main():
    sessions, rows, out = sys.argv[1:]
    keep = {json.loads(line)["session_id"] for line in open(rows)}
    n = chars = 0
    with open(out, "w") as f:
        for line in open(sessions):
            s = json.loads(line)
            if s["session_id"] not in keep:
                continue
            parts = []
            for m in s["messages"]:
                body = [m.get("reasoning_content") or "", m.get("content") or ""]
                for c in m.get("tool_calls") or []:
                    body.append(f'{c["function"]["name"]}({c["function"]["arguments"]})')
                parts.append(f"### {m['role']}\n" + "\n".join(b for b in body if b))
            text = "\n".join(parts) + "\n\n"
            f.write(text)
            n += 1
            chars += len(text)
    print(f"{n} sessions, {chars / 1e6:.2f} M characters -> {out}")


if __name__ == "__main__":
    main()
