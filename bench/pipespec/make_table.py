#!/usr/bin/env python3
"""Writes a continuation table for llama-pipespec's `cont` guesser (--pipe-table FILE): the generated tokens of a set of
transcripts in bench/ngram_replay's format (one JSON object per line with "id" and "gen_tokens", e.g. Q51's gen.jsonl),
one output per line as space-separated token ids.

Leave out transcripts of the prompts you are about to run, or the table guesses from the answer itself.
Usage: make_table.py <gen.jsonl> <table.txt> [--exclude REGEX]   (REGEX matched against the transcript id)
"""
import json
import re
import sys


def main(args):
    if len(args) < 2:
        sys.exit(__doc__)
    src, dst = args[0], args[1]
    exclude = None
    if len(args) >= 4 and args[2] == "--exclude":
        exclude = re.compile(args[3])
    n_rows = n_tok = 0
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            row = json.loads(line)
            if exclude and exclude.search(str(row.get("id", ""))):
                continue
            out.write(" ".join(str(t) for t in row["gen_tokens"]) + "\n")
            n_rows += 1
            n_tok += len(row["gen_tokens"])
    print(f"{n_rows} outputs, {n_tok} tokens -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1:])
