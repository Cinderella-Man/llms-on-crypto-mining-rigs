#!/usr/bin/env python3
"""How many tokens a chat completion body is, as the model's own chat template and vocabulary count it (no server).

    VOCAB_MODEL=<first shard of a GGUF> LLAMA_TOKENIZE=<llama-tokenize> python3 -I count_tokens.py request.json

Reads `tokenizer.chat_template` from the GGUF header, renders the body's messages and tools with it (jinja2), and counts
the rendered text and the body's parts with llama.cpp's llama-tokenize (CPU, vocabulary only: no weights are read).
Used to set first_turn_request.py's defaults: the table says what the content weighs and what the template adds. The
server's own count (minja renders the same template) is `usage.prompt_tokens` in the reply and can differ by a few tokens.
"""
import json
import os
import struct
import subprocess
import sys

SIZES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def rstr(f):
    return f.read(rd(f, "Q")).decode("utf-8", "replace")


def skip(f, t):
    if t == 8:
        f.seek(rd(f, "Q"), 1)
    elif t == 9:
        at, n = rd(f, "I"), rd(f, "Q")
        for _ in range(n):
            skip(f, at)
    else:
        f.seek(struct.calcsize("<" + SIZES[t]), 1)


def chat_template(path):
    with open(path, "rb") as f:
        f.read(4)
        rd(f, "I")
        rd(f, "Q")
        for _ in range(rd(f, "Q")):
            key, t = rstr(f), rd(f, "I")
            if key == "tokenizer.chat_template":
                return rstr(f)
            skip(f, t)
    raise SystemExit("no tokenizer.chat_template in " + path)


def main():
    model, tok = os.environ["VOCAB_MODEL"], os.environ["LLAMA_TOKENIZE"]
    req = json.load(open(sys.argv[1]))
    import jinja2

    def raise_exception(m):
        raise Exception(m)

    env = jinja2.Environment()
    env.globals.update(raise_exception=raise_exception, strftime_now=lambda fmt: "2026-01-01")
    env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False)
    rendered = env.from_string(chat_template(model)).render(
        messages=req["messages"], tools=req.get("tools"), add_generation_prompt=True, enable_thinking=True,
        bos_token="", eos_token="")

    def count(text, special):
        cmd = [tok, "-m", model, "--stdin", "--ids", "--no-bos"] + ([] if special else ["--no-parse-special"])
        out = subprocess.run(cmd, input=text.encode(), capture_output=True, check=True).stdout.decode()
        return len(json.loads(out.strip().splitlines()[-1]))

    total = count(rendered, True)
    parts = {"system message": req["messages"][0]["content"],
             "tools (one JSON line each)": "\n".join(json.dumps(t) for t in req.get("tools", [])),
             "user message": req["messages"][1]["content"]}
    print("part\tcharacters\ttokens\tcharacters_per_token")
    s = 0
    for name, text in parts.items():
        n = count(text, False)
        s += n
        print(f"{name}\t{len(text)}\t{n}\t{len(text) / n:.2f}")
    print(f"sum of the parts\t{sum(len(t) for t in parts.values())}\t{s}\t")
    print(f"rendered prompt\t{len(rendered)}\t{total}\t")
    print(f"added by the template (rendered - sum)\t\t{total - s}\t")


if __name__ == "__main__":
    main()
