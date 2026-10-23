"""Build the prompt set for the expert-overlap measurement (Q51): chat, code writing, whole-file code edits and
agentic tool calls, as OpenAI-style messages (and tools) that llama-server's /apply-template renders.

usage: python prompts.py OUT.jsonl FILE [FILE ...]   (the FILEs are the sources the edit prompts rewrite)
"""
import json
import os
import sys

CHAT = [
    "Explain how a Bloom filter works, when it is the right tool, and when it is not.",
    "Write a polite email declining a meeting next Tuesday and proposing two other times.",
    "Compare TCP and QUIC for a mobile app that streams short video clips.",
    "Summarize the main causes of the 1929 stock market crash in five bullet points.",
    "Plan a three-day trip to Krakow for a family with two children aged 6 and 10.",
    "What are the trade-offs of running SQLite in production for a small web service?",
    "Explain the difference between the mean and the median, with an example where they differ a lot.",
    "I keep postponing work I find boring. Give me a practical plan to fix that over the next two weeks.",
    "How should I prepare for a system design interview in four weeks?",
    "Explain what the Polish proverb 'Nie chwal dnia przed zachodem slonca' means and give an English equivalent.",
]

CODE = [
    "Write a Python class implementing an LRU cache with a per-entry time-to-live, with unit tests.",
    "Write a Rust command-line tool that counts words, lines and bytes in the files given as arguments.",
    "Implement debounce and throttle in JavaScript, with a short explanation of the difference and Jest tests.",
    "Write a SQL query that returns the ten customers with the highest total order value in 2025, given tables "
    "customers(id, name), orders(id, customer_id, created_at) and order_items(order_id, price, quantity).",
    "Write an Elixir GenServer that rate-limits calls per user with a token bucket, with ExUnit tests.",
    "Write a bash script that rotates log files in a directory: gzip files older than one day, delete files older "
    "than 30 days, and print a summary.",
    "Write a Go HTTP server with two JSON endpoints, structured logging and graceful shutdown on SIGTERM.",
    "Write a C function that parses one line of CSV with quoted fields and escaped quotes into an array of strings.",
    "Write a Python asyncio web crawler that fetches pages from one domain with at most 5 concurrent requests.",
    "Write a type-safe event emitter in TypeScript where each event name has its own payload type.",
]

EDITS = [
    "Add type hints to every function and add a docstring to each one. Output the complete updated file.",
    "Replace os.path with pathlib throughout and keep the behaviour the same. Output the complete updated file.",
    "Add a --verbose command-line option that prints what the script is doing. Output the complete updated file.",
    "Add error handling with clear messages for missing or malformed input. Output the complete updated file.",
    "Rename the variables so the names say what they hold, and change nothing else. Output the complete updated file.",
    "Add logging with the logging module instead of print calls. Output the complete updated file.",
    "Refactor the longest function into smaller functions without changing behaviour. Output the complete updated "
    "file.",
    "Add a short comment above each block that explains what it does. Output the complete updated file.",
]

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file from the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write a whole text file in the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command in the repository and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "search_code", "description": "Search the repository for a regular expression.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
]

SYSTEM = "You are a coding agent working in a Python repository. Use the tools to inspect and change files."

AGENT_TASKS = [
    "The test tests/test_parse.py::test_ids fails with KeyError: 'id'. Find the cause and fix it.",
    "Add a --dry-run flag to scripts/sync.py that prints what would be copied without copying.",
    "Find every place that opens files without a context manager and fix them.",
    "The CI log says 'flake8: E501 line too long' in src/report.py. Fix the long lines.",
    "Rename the function load_cfg to load_config everywhere in the repository.",
    "Write a unit test for utils.slugify covering empty strings, unicode and repeated dashes.",
]


def agent_with_history(task, source_name, source):
    """A second agent turn: the model already read a file and now continues."""
    call = {"id": "call_0", "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"path": source_name})}}
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_0", "content": source},
    ]


def main():
    out_path, files = sys.argv[1], sys.argv[2:]
    sources = [(os.path.basename(f), open(f).read()) for f in files]
    rows = []
    for i, p in enumerate(CHAT):
        rows.append({"id": f"chat_{i}", "category": "chat", "messages": [{"role": "user", "content": p}]})
    for i, p in enumerate(CODE):
        rows.append({"id": f"code_{i}", "category": "code", "messages": [{"role": "user", "content": p}]})
    for i, (name, src) in enumerate(sources):
        task = EDITS[i % len(EDITS)]
        rows.append({"id": f"edit_{i}", "category": "edit",
                     "messages": [{"role": "user", "content": f"Here is {name}:\n\n```\n{src}```\n\n{task}"}]})
    for i, t in enumerate(AGENT_TASKS):
        rows.append({"id": f"agent_{i}", "category": "agent", "tools": TOOLS,
                     "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t}]})
    for i, (name, src) in enumerate(sources[:4]):
        rows.append({"id": f"agent2_{i}", "category": "agent", "tools": TOOLS,
                     "messages": agent_with_history(
                         f"Review {name} for bugs and fix the most serious one with write_file.", name, src)})
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} prompts", file=sys.stderr)


if __name__ == "__main__":
    main()
