#!/usr/bin/env python3
# Builds the long-context KLD evaluation text from rendered agent conversations (bench/bfcl_flint/multi_turn.py render):
# conversations cut to exactly CTX tokens and joined, so that llama-perplexity -c CTX scores one conversation per chunk.
#
#   build_eval.py RENDERED.jsonl MODEL.gguf LLAMA_TOKENIZE OUT_DIR [--ctx 8192] [--per-category 3] [--attempt 1]
#
# MODEL.gguf gives the vocabulary (llama-tokenize loads it vocab-only); LLAMA_TOKENIZE is a llama-tokenize binary of the
# same llama.cpp tree as the llama-perplexity that makes the reference. Selection: for every file in RENDERED (the
# model whose run produced the conversation) and every BFCL category, the first --per-category conversations of that
# attempt in task-ID order, each task at most once over all files.
# Checks, all fatal: the rendered prompt has exactly the token count the server reported when the model saw it; the
# cut conversation re-tokenizes to exactly its first CTX tokens; the joined file tokenizes to exactly those chunks.
# Output: OUT_DIR/eval.txt (for llama-perplexity -f ... --parse-special --no-escape), OUT_DIR/chunks.jsonl (one record
# per chunk: file, id, category, tokens), OUT_DIR/build.log.
# Needs numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
import argparse, json, os, subprocess, sys, tempfile
from collections import defaultdict


def log(msg, f=[None]):
    print(msg, flush=True)
    if f[0]:
        f[0].write(msg + '\n')
        f[0].flush()


def tokenize(binary, model, text):
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', suffix='.txt', delete=False) as t:
        t.write(text)
    try:
        out = subprocess.run([binary, '-m', model, '-f', t.name, '--ids', '--no-escape', '--no-bos', '--log-disable'],
                             capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(t.name)
    line = [l for l in out.splitlines() if l.startswith('[')][-1]
    return json.loads(line)


def bytes_decoder():
    """GPT-2 byte-level BPE: the inverse of bytes_to_unicode"""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rendered'); ap.add_argument('model'); ap.add_argument('tokenize'); ap.add_argument('out')
    ap.add_argument('--ctx', type=int, default=8192); ap.add_argument('--per-category', type=int, default=3)
    ap.add_argument('--attempt', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log.__defaults__[0][0] = open(os.path.join(a.out, 'build.log'), 'w')

    import gguf
    r = gguf.GGUFReader(a.model)
    model_type = r.get_field('tokenizer.ggml.model').contents()
    tok_strs = r.get_field('tokenizer.ggml.tokens').contents()
    tok_types = r.get_field('tokenizer.ggml.token_type').contents()
    if model_type != 'gpt2':
        sys.exit(f'tokenizer {model_type}: only byte-level BPE (gpt2) is supported')
    dec = bytes_decoder()

    def detok(ids):
        out = bytearray()
        for i in ids:
            s = tok_strs[i]
            if tok_types[i] in (3, 4):  # control and user-defined tokens are literal text
                out += s.encode('utf-8')
            else:
                out += bytes(dec[c] for c in s)
        return bytes(out)

    recs = [json.loads(l) for l in open(a.rendered)]
    by = defaultdict(list)
    for x in recs:
        if x['attempt'] == a.attempt:
            by[(x['file'], x['id'].rsplit('_', 1)[0])].append(x)
    files = sorted({f for f, _ in by})
    cats = sorted({c for _, c in by})
    log(f'{len(recs)} rendered conversations; files {files}; categories {cats}')

    used, chunks, skipped = set(), [], 0
    for cat in cats:
        for f in files:
            n = 0
            for x in sorted(by[(f, cat)], key=lambda x: int(x['id'].rsplit('_', 1)[1])):
                if n == a.per_category:
                    break
                if x['id'] in used:
                    continue
                n_prompt = len(tokenize(a.tokenize, a.model, x['prompt']))
                if n_prompt != x['n_prompt']:
                    log(f'SKIP {f} {x["id"]}: rendered prompt has {n_prompt} tokens, the server counted {x["n_prompt"]}')
                    skipped += 1
                    continue
                ids = tokenize(a.tokenize, a.model, x['prompt'] + x['output'])
                if len(ids) < a.ctx:
                    log(f'SKIP {f} {x["id"]}: {len(ids)} tokens < {a.ctx}')
                    skipped += 1
                    continue
                cut = ids[:a.ctx]
                try:
                    text = detok(cut).decode('utf-8')
                except UnicodeDecodeError:
                    log(f'SKIP {f} {x["id"]}: the cut splits a UTF-8 character')
                    skipped += 1
                    continue
                if tokenize(a.tokenize, a.model, text) != cut:
                    log(f'SKIP {f} {x["id"]}: the cut conversation does not re-tokenize to its first {a.ctx} tokens')
                    skipped += 1
                    continue
                used.add(x['id'])
                n += 1
                chunks.append({'file': f, 'id': x['id'], 'category': cat, 'n_prompt': x['n_prompt'],
                               'n_tokens_full': len(ids), 'text': text, 'tokens': cut})
                log(f'chunk {len(chunks) - 1}: {f} {x["id"]} ({len(ids)} tokens, prompt {n_prompt} = server)')
            if n < a.per_category:
                log(f'WARNING {f} {cat}: only {n} conversations')

    joined = ''.join(c['text'] for c in chunks)
    all_ids = tokenize(a.tokenize, a.model, joined)
    want = [t for c in chunks for t in c['tokens']]
    if all_ids != want:
        sys.exit(f'the joined file tokenizes to {len(all_ids)} tokens, not the {len(want)} of its chunks')
    with open(os.path.join(a.out, 'eval.txt'), 'w', encoding='utf-8') as f:
        f.write(joined)
    with open(os.path.join(a.out, 'chunks.jsonl'), 'w') as f:
        for c in chunks:
            f.write(json.dumps({k: v for k, v in c.items() if k != 'text'}) + '\n')
    log(f'{len(chunks)} chunks of {a.ctx} tokens ({len(all_ids)} tokens) in {a.out}/eval.txt; {skipped} skipped; '
         f'the joined file re-tokenizes to exactly these chunks')


if __name__ == '__main__':
    main()
