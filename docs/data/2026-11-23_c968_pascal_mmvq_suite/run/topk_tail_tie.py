#!/usr/bin/env python3
"""C968 (Q1698, Q1725): the kept TOPK_MOE_TAIL case's router, read from its record.

Usage: python3 -I topk_tail_tie.py <case>.in <case>.ref
The case: n_expert=256, n_tokens=9, n_expert_used=8, sigmoid gating, norm, scale 2.5, p=0.8, n_skip=8. Its record's
inputs (.in) hold the logits (the second tensor, 256 x 9 float32, after a 1,024-float sentinel); its reference (.ref)
holds the processor's cut weights, cut ids and expert product, in that order (graph order). Per token this prints the
processor's 8 chosen experts (the cut slot last) and the sigmoid probabilities in float32 of ranks 7 to 10, with the gap
between rank 8 (the last chosen, cut by the tail) and rank 9 (the first left out) in float32 ulps: a gap of 0 or a few
ulps lets another sigmoid (the card's expf) choose the other expert for the cut slot.
"""
import math
import struct
import sys

IN, REF = sys.argv[1], sys.argv[2]
NE, NT, K = 256, 9, 8
b = open(IN, "rb").read()
logits = struct.unpack_from("<%df" % (NE * NT), b, 1024 * 4)
r = open(REF, "rb").read()
ref = struct.unpack("<%df" % (len(r) // 4), r)
w_cut, ids_cut = ref[:K * NT], ref[K * NT:2 * K * NT]


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def ulps(a, b):
    ia = struct.unpack("<i", struct.pack("<f", a))[0]
    ib = struct.unpack("<i", struct.pack("<f", b))[0]
    return abs(ia - ib)


print("token\tprocessor's experts (cut slot last)\tranks 7-10 (expert:probability)\trank 8 - rank 9 in ulps")
for t in range(NT):
    row = logits[t * NE:(t + 1) * NE]
    p = [f32(1.0 / (1.0 + math.exp(-f32(x)))) for x in row]
    order = sorted(range(NE), key=lambda e: -p[e])
    chosen = [int(x) for x in ids_cut[t * K:(t + 1) * K]]
    ranks = " ".join(f"{e}:{p[e]:.9f}" for e in order[6:10])
    print(f"{t}\t{' '.join(map(str, chosen))}\t{ranks}\t{ulps(p[order[7]], p[order[8]])}")
