"""Extract the tensors the staircase guesser borrows from Flint: the token embedding table (frozen input embeddings),
the output head and the output norm (the guesser's head starts from the output-head rows of its vocabulary).

usage: python flint_weights.py FLINT.gguf OUT_DIR    (needs numpy and gguf-py, e.g. PYTHONPATH=<llama.cpp>/gguf-py)
writes OUT_DIR/token_embd.npy, output.npy (float16, [n_vocab, n_embd]) and output_norm.npy (float32, [n_embd])
"""
import os
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


def main():
    path, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    reader = GGUFReader(path)
    want = {"token_embd.weight": "token_embd", "output.weight": "output", "output_norm.weight": "output_norm"}
    for t in reader.tensors:
        if t.name not in want:
            continue
        x = dequantize(t.data, t.tensor_type)
        shape = [int(s) for s in reversed(t.shape)]
        x = np.asarray(x, dtype=np.float32).reshape(shape)
        dtype = np.float32 if t.name == "output_norm.weight" else np.float16
        np.save(os.path.join(out, want[t.name] + ".npy"), x.astype(dtype))
        print(t.name, t.tensor_type.name, shape, float(np.abs(x).mean()))


if __name__ == "__main__":
    main()
