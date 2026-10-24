#!/usr/bin/env python3
# A subset of a build_eval.py evaluation text: the chosen chunks, joined again, for a reference that is too slow to run
# on all of them (e.g. BF16 with experts in system RAM).
#
#   subset_eval.py EVAL_DIR MODEL.gguf LLAMA_TOKENIZE OUT_DIR --chunks 0,1,3,...
#
# OUT_DIR/eval.txt and OUT_DIR/chunks.jsonl as build_eval.py writes them, each record with its index in EVAL_DIR as
# "orig_chunk" (analyze.py --only-chunks uses the same list to compare dumps made on the full text). The joined subset
# must re-tokenize to exactly the chosen chunks' tokens.
# Needs numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_eval import bytes_decoder, tokenize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('eval_dir'); ap.add_argument('model'); ap.add_argument('tokenize'); ap.add_argument('out')
    ap.add_argument('--chunks', required=True)
    a = ap.parse_args()
    import gguf
    r = gguf.GGUFReader(a.model)
    tok_strs = r.get_field('tokenizer.ggml.tokens').contents()
    tok_types = r.get_field('tokenizer.ggml.token_type').contents()
    dec = bytes_decoder()

    def detok(ids):
        return b''.join(tok_strs[i].encode('utf-8') if tok_types[i] in (3, 4) else bytes(dec[c] for c in tok_strs[i])
                        for i in ids).decode('utf-8')

    meta = [json.loads(l) for l in open(os.path.join(a.eval_dir, 'chunks.jsonl'))]
    pick = [int(x) for x in a.chunks.split(',')]
    sub = [dict(meta[i], orig_chunk=i) for i in pick]
    text = ''.join(detok(c['tokens']) for c in sub)
    ids = tokenize(a.tokenize, a.model, text)
    want = [t for c in sub for t in c['tokens']]
    if ids != want:
        sys.exit(f'the subset tokenizes to {len(ids)} tokens, not the {len(want)} of its chunks')
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'eval.txt'), 'w', encoding='utf-8') as f:
        f.write(text)
    with open(os.path.join(a.out, 'chunks.jsonl'), 'w') as f:
        for c in sub:
            f.write(json.dumps(c) + '\n')
    print(f'{len(sub)} chunks ({len(ids)} tokens) in {a.out}: ' +
          ', '.join(f'{c["orig_chunk"]}:{c["file"]}:{c["id"]}' for c in sub))


if __name__ == '__main__':
    main()
