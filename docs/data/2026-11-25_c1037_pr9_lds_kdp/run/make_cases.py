#!/usr/bin/env python3
"""Ornith-1.5-122B's attention call as test-backend-ops --test-file lines (R232 (a), Q1706).

Each line is one FLASH_ATTN_EXT node shaped as llama.cpp builds it for Ornith's 12 gated-attention layers:
head 256, 32 query heads over 2 KV heads (GQA 16), the query a permuted view of [256, 32, tokens], K and V permuted
views of a 49,152-cell cache (heads side by side in each cell), an F16 mask [n_kv, tokens], scale 1/16, no ALiBi,
no softcap, no sinks, F32 accumulation (op_params[3] = GGML_PREC_F32 = 10), n_kv_max 0.

Line format (tests/test-backend-ops.cpp, make_test_cases_from_file):
  op type ne0 ne1 ne2 ne3 n_params p... n_src {type ne0..ne3 nb0..nb3}... name

Usage: python3 -I make_cases.py > cases.txt
       python3 -I make_cases.py --depths 256,8192 --rows 1 --types q8_0 > one.txt
"""
import argparse
import struct

GGML_OP_FLASH_ATTN_EXT = 71   # enum ggml_op in ggml/include/ggml.h of pascal-p106-v5 af00bb13b
TYPES = {"f32": (0, 4, 1), "f16": (1, 2, 1), "q8_0": (8, 34, 32)}   # id, bytes per block, values per block
GGML_PREC_F32 = 10

HEAD = 256        # head size (K and V)
N_HEAD = 32       # query heads
N_HEAD_KV = 2     # KV heads
KV_SIZE = 49152   # cache cells: the v5 preset's -c 49152 with -np 1


def row_size(t, n):
    _, bs, bl = TYPES[t]
    return n // bl * bs


def f32_bits(x):
    return struct.unpack("<i", struct.pack("<f", x))[0]


def case(kv_type, n_kv, n_tok):
    tid, bs, _ = TYPES[kv_type]
    params = [f32_bits(1.0 / HEAD ** 0.5), 0, 0, GGML_PREC_F32, 0]
    # query: [256, 32, n_tok] F32 contiguous, permuted (0, 2, 1, 3) -> ne [256, n_tok, 32, 1]
    q = (0, (HEAD, n_tok, N_HEAD, 1), (4, 4 * HEAD * N_HEAD, 4 * HEAD, 4 * HEAD * N_HEAD * n_tok))
    # K and V: views of the cache [256*2, KV_SIZE] -> [256, 2, n_kv] permuted -> ne [256, n_kv, 2, 1]
    cell = row_size(kv_type, HEAD * N_HEAD_KV)
    kv = (tid, (HEAD, n_kv, N_HEAD_KV, 1), (bs, cell, row_size(kv_type, HEAD), cell * KV_SIZE))
    # mask: [n_kv, n_tok, 1, 1] F16 contiguous
    m = (1, (n_kv, n_tok, 1, 1), (2, 2 * n_kv, 2 * n_kv * n_tok, 2 * n_kv * n_tok))
    srcs = [q, kv, kv, m]
    out = [GGML_OP_FLASH_ATTN_EXT, 0, HEAD, N_HEAD, n_tok, 1, len(params), *params, len(srcs)]
    for t, ne, nb in srcs:
        out += [t, *ne, *nb]
    name = f"ornith_{kv_type}_kv{n_kv}_nt{n_tok}"
    return " ".join(str(v) for v in out) + " " + name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="q8_0,f16")
    ap.add_argument("--depths", default="256,4096,8192,16384,32768,49152",
                    help="n_kv values; llama.cpp pads n_kv to 256, so depth 0 is 256")
    ap.add_argument("--rows", default="1,2,3,4,8", help="query tokens: decode, the pipe and drafts")
    ap.add_argument("--prompt-rows", default="512", help="the preset's micro-batch (-ub 512)")
    ap.add_argument("--prompt-depths", default="256,4096")
    a = ap.parse_args()
    for t in a.types.split(","):
        for d in (int(x) for x in a.depths.split(",")):
            for r in (int(x) for x in a.rows.split(",") if x):
                print(case(t, d, r))
        for d in (int(x) for x in a.prompt_depths.split(",") if x):
            for r in (int(x) for x in a.prompt_rows.split(",") if x):
                print(case(t, d, r))


if __name__ == "__main__":
    main()
