"""Check that the Hugging Face chat template and tokenizer give the prompt ids llama-server gave (generate.py rows).

generate.py --api vllm builds prompts with Flint's Hugging Face directory instead of llama-server's /apply-template
and /tokenize. This compares the two on rows already generated, tool prompts included (their tool lists go through
`tools`), and prints the first difference of every mismatch.

usage: python check_prompts.py FLINT_HF_DIR N GEN.jsonl:PROMPTS.jsonl [GEN.jsonl:PROMPTS.jsonl ...]
Checks the first N rows of each generation file against the prompts of the same id. Needs transformers.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate import prompt_ids_hf  # noqa: E402


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    n = int(sys.argv[2])
    bad_total = 0
    for pair in sys.argv[3:]:
        gen_path, prompts_path = pair.split(":")
        prompts = {}
        for line in open(prompts_path):
            r = json.loads(line)
            prompts[r["id"]] = r
        checked = bad = tools = 0
        for line in open(gen_path):
            if checked >= n:
                break
            g = json.loads(line)
            row = prompts.get(g["id"])
            if row is None:
                continue
            ids = prompt_ids_hf(tok, row)
            checked += 1
            tools += "tools" in row
            if ids != g["prompt_tokens"]:
                bad += 1
                i = next((k for k, (a, b) in enumerate(zip(ids, g["prompt_tokens"])) if a != b),
                         min(len(ids), len(g["prompt_tokens"])))
                print(f"  {g['id']}: lengths {len(ids)} / {len(g['prompt_tokens'])}, first difference at {i}: "
                      f"hf {tok.decode(ids[max(0, i - 5):i + 5])!r} llama {tok.decode(g['prompt_tokens'][max(0, i - 5):i + 5])!r}")
        print(f"{gen_path}: {checked} rows ({tools} with tools), {bad} mismatches")
        bad_total += bad
    sys.exit(1 if bad_total else 0)


if __name__ == "__main__":
    main()
