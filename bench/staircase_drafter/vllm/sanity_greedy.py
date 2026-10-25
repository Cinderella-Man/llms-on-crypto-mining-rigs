"""Greedy answers from vLLM and from a llama-server copy of Flint on the same prompt ids, side by side.

A sanity check before generate.py --api vllm writes training text: the two engines (and quantizations) should agree on
the first tokens of most greedy answers and both should read as Flint. Prints, per prompt, the length of the common
prefix of the two greedy answers and the start of each answer.

usage: python sanity_greedy.py VLLM_URL LLAMA_URL PROMPTS.jsonl FLINT_HF_DIR [--n 6] [--tokens 160]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate import END_IDS, post, prompt_ids_hf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vllm")
    ap.add_argument("llama")
    ap.add_argument("prompts")
    ap.add_argument("tokenizer")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=160)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = [json.loads(line) for line, _ in zip(open(args.prompts), range(args.n * 20))]
    rows = [r for r in rows if len(prompt_ids_hf(tok, r)) <= 1024][:args.n]
    agree = []
    for r in rows:
        ids = prompt_ids_hf(tok, r)
        v = post(f"{args.vllm}/v1/completions", {"model": "flint", "prompt": ids, "max_tokens": args.tokens,
                                                 "temperature": 0.0, "stop_token_ids": END_IDS,
                                                 "return_token_ids": True})["choices"][0]["token_ids"]
        lc = post(f"{args.llama}/completion", {"prompt": ids, "n_predict": args.tokens, "temperature": 0.0,
                                               "top_k": 1, "return_tokens": True, "cache_prompt": False})["tokens"]
        n = next((i for i, (a, b) in enumerate(zip(v, lc)) if a != b), min(len(v), len(lc)))
        agree.append(n)
        print(f"== {r['id']} ({r['category']}, {len(ids)} prompt tokens): common prefix {n} of {len(v)} / {len(lc)}")
        print(f"   vllm : {tok.decode(v)[:300]!r}")
        print(f"   llama: {tok.decode(lc)[:300]!r}")
    print(f"common prefix per prompt: {agree}")


if __name__ == "__main__":
    main()
