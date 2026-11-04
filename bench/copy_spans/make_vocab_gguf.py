"""Write a tokenizer-only GGUF from a model's GGUF, so llama-tokenize can load the vocabulary of an architecture the
build does not know (Qwen3.8-Flash-Next's `qwen4exp`) without reading any weights.

    python3 make_vocab_gguf.py MODEL.gguf VOCAB.gguf

The file keeps every tokenizer.ggml.* entry (token list, types, merges, pre-tokenizer name, special ids) and
declares a one-layer `qwen3` model with no tensors; llama-tokenize loads it with vocab_only. The pre-tokenizer is
the one named in the model file (`qwen35`), so a build that tokenizes with that name gives the model's tokens.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_gguf_kv  # noqa: E402


def wstr(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_str(k, v):
    return wstr(k) + struct.pack("<I", 8) + wstr(v)


def kv_u32(k, v):
    return wstr(k) + struct.pack("<II", 4, v)


def kv_f32(k, v):
    return wstr(k) + struct.pack("<If", 6, v)


def kv_arr_str(k, vals):
    return wstr(k) + struct.pack("<IIQ", 9, 8, len(vals)) + b"".join(wstr(x) for x in vals)


def kv_arr_i32(k, vals):
    return wstr(k) + struct.pack("<IIQ", 9, 5, len(vals)) + struct.pack("<%di" % len(vals), *vals)


def main(src, dst):
    kv = read_gguf_kv(src)
    tok = {k: v for k, v in kv.items() if k.startswith("tokenizer.ggml.")}
    entries = [
        kv_str("general.architecture", "qwen3"),
        kv_str("general.name", "vocab only"),
        kv_u32("qwen3.block_count", 1),
        kv_u32("qwen3.context_length", 4096),
        kv_u32("qwen3.embedding_length", 8),
        kv_u32("qwen3.feed_forward_length", 8),
        kv_u32("qwen3.attention.head_count", 1),
        kv_u32("qwen3.attention.head_count_kv", 1),
        kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6),
    ]
    for k, v in tok.items():
        if k in ("tokenizer.ggml.model", "tokenizer.ggml.pre"):
            entries.append(kv_str(k, v))
        elif k == "tokenizer.ggml.tokens" or k == "tokenizer.ggml.merges":
            entries.append(kv_arr_str(k, v))
        elif k == "tokenizer.ggml.token_type":
            entries.append(kv_arr_i32(k, v))
        elif isinstance(v, int) and not isinstance(v, bool):
            entries.append(kv_u32(k, v))
        else:
            raise SystemExit("unhandled tokenizer entry " + k)
    blob = b"GGUF" + struct.pack("<IQQ", 3, 0, len(entries)) + b"".join(entries)
    blob += b"\0" * (-len(blob) % 32)
    with open(dst, "wb") as f:
        f.write(blob)
    print("wrote %s: %d tokenizer entries, %d tokens, %d bytes" % (
        dst, len(tok), len(tok["tokenizer.ggml.tokens"]), len(blob)))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
