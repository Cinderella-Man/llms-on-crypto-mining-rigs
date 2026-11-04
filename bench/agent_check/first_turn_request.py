#!/usr/bin/env python3
"""The agent check's first request to llama-server, as a chat completion body (Q607).

    python3 first_turn_request.py --tasks tasks/bugfix [--task NAME] [--tokens 2780] [--chars-per-token 3.6]
                                  [--template-tokens 250] [--max-tokens 256] > request.json

The agent's first turn is the request that carries a whole task for the first time: pi's system prompt, its four tools
and the task's `prompt.md`, about 2,780 tokens for the first bugfix task, against the 79 tokens of the draft smoke
tests' coding request. A server that fits the short request can still run out of host memory on this one (Q561: the
prompt crossed `-b 2048`, and the drafter's layer-input buffers grow with `-b`).

The body has the shape of that turn: a short system message, the four tools of the agent check's pi (read, bash, edit,
write, each with a JSON schema; a request with tools takes the server's tool-call path, with backend sampling off), and
the task as the user message. It is built from the task's own files, at run time:
- the task is the first line of `<tasks>/order.txt` (or --task); its `prompt.md` goes in whole;
- the task's workspace files (the stub, then the test file: what the agent reads in its next calls) follow, cut at a line
  boundary, until the request reaches --tokens, estimated as characters / --chars-per-token plus --template-tokens for
  what the chat template adds around the messages and the tools. The defaults (3.6 characters per token for this mix
  of English text, code and JSON; 250 tokens for Swift Flash-Next's template, which opens with a reasoning-effort
  paragraph and renders the tools in its own format) come from tokenizing the rendered request with that model's
  vocabulary: 2,946 tokens for a body this tool built at the then defaults. The server's own count is in the reply
  (`usage.prompt_tokens`; the smoke scripts print it): tune the two when it is off, or for another model's template.
No task text is stored with the tool: rebuild the task folder with `python3 build_bugfix.py --out <folder>/tasks/bugfix`
(first_turn_request.sh does) and point --tasks at it. The summary goes to stderr, the body to stdout.
"""
import argparse
import json
import sys
from pathlib import Path

SYSTEM = (
    "You are a coding agent working in the directory /work. You can read files, run shell commands, edit files and "
    "write files with the tools you are given. Do the task the user describes, keep your changes small, run the tests "
    "before you finish, and say briefly what you changed."
)


def tool(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


TOOLS = [
    tool("read", "Read a text file from the working directory, optionally from a line offset and up to a number of lines.",
         {"path": {"type": "string", "description": "Path of the file, relative to /work"},
          "offset": {"type": "integer", "description": "First line to read, counting from 1"},
          "limit": {"type": "integer", "description": "Maximum number of lines to read"}}, ["path"]),
    tool("bash", "Run a shell command in the working directory and return its output and exit status.",
         {"command": {"type": "string", "description": "The command to run"},
          "timeout": {"type": "integer", "description": "Seconds to wait before the command is stopped"}}, ["command"]),
    tool("edit", "Replace one exact piece of text in a file with another. The old text must occur exactly once.",
         {"path": {"type": "string", "description": "Path of the file, relative to /work"},
          "oldText": {"type": "string", "description": "The exact text to replace"},
          "newText": {"type": "string", "description": "The text that takes its place"}}, ["path", "oldText", "newText"]),
    tool("write", "Write a file in the working directory, replacing it if it exists.",
         {"path": {"type": "string", "description": "Path of the file, relative to /work"},
          "content": {"type": "string", "description": "The whole new content of the file"}}, ["path", "content"]),
]


def cut_lines(text, chars):
    """The longest run of whole lines of text that fits in chars characters."""
    if len(text) <= chars:
        return text
    return text[:chars].rsplit("\n", 1)[0] + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, required=True, help="a task set folder holding order.txt and the task folders")
    ap.add_argument("--task", help="a task folder name (default: the first line of order.txt)")
    ap.add_argument("--tokens", type=int, default=2780, help="the size to aim for, in tokens of the server's count")
    ap.add_argument("--chars-per-token", type=float, default=3.6)
    ap.add_argument("--template-tokens", type=int, default=250, help="tokens the chat template adds around the content")
    ap.add_argument("--max-tokens", type=int, default=256, help="the reply's cap (the agent's first reply is ~80 tokens)")
    a = ap.parse_args()

    name = a.task or a.tasks.joinpath("order.txt").read_text().split()[0]
    task = a.tasks / name
    prompt = (task / "prompt.md").read_text()
    meta = json.loads((task / "task.json").read_text()) if (task / "task.json").exists() else {}
    files = [meta.get("stub"), meta.get("test_file")]
    files = [f for f in files if f and (task / "workspace" / f).exists()] or sorted(
        p.name for p in (task / "workspace").iterdir() if p.is_file())

    fixed = len(SYSTEM) + len(json.dumps(TOOLS))
    budget = int((a.tokens - a.template_tokens) * a.chars_per_token) - fixed - len(prompt)
    user, shown = prompt, []
    for f in files:
        text = (task / "workspace" / f).read_text()
        head = f"\n\nThe file `{f}` in the working directory:\n\n```\n"
        room = budget - len(head) - len("```\n")
        if room < 200:
            break
        body = cut_lines(text, room)
        user += head + body + ("" if body.endswith("\n") else "\n") + "```\n"
        budget -= len(head) + len(body) + 4
        shown.append(f"{f} ({len(body)} of {len(text)} characters)")
        if len(body) < len(text):
            break

    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], "tools": TOOLS,
            "max_tokens": a.max_tokens, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0, "stream": False}
    out = json.dumps(body)
    total = len(SYSTEM) + len(json.dumps(TOOLS)) + len(user)
    print(f"first_turn_request: task {name}; prompt.md {len(prompt)} characters; files {', '.join(shown) or 'none'}; "
          f"body {len(out)} bytes, {total} characters of content, est. {total / a.chars_per_token + a.template_tokens:.0f} tokens "
          f"(target {a.tokens}: {a.chars_per_token} characters per token and {a.template_tokens} for the template)", file=sys.stderr)
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
