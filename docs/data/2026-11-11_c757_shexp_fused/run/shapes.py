#!/usr/bin/env python3
"""shapes.py: C757 (Q1333). The launch shape mmvq-pascal-glu.cu's host code picks for a gate and an up product of K
weights per row (its glu_lpr_log2, glu_slot, glu_shape_of and glu_smem, written out again): per type pair and token
count, the lanes per row and rows per warp pass of each type, the rows per row block, the bytes of the staged
activations and of a warp pair's staging buffers and results, and the warp pairs per block (the most, up to 4, that fit
48 KB), with the block's shared memory.

  python3 -I shapes.py [--k 2560] [--rows 640] [--nsm 5] > shapes.tsv

Standard library only."""
import argparse

UB = {"q2_0": 36, "q3_K": 110, "q4_K": 144, "q5_K": 176, "iq4_xs": 136}
UW = {"q2_0": 128, "q3_K": 256, "q4_K": 256, "q5_K": 256, "iq4_xs": 256}
AL = {"q2_0": 4, "q3_K": 2, "q4_K": 16, "q5_K": 16, "iq4_xs": 8}
# the file's (Swift GSQ-RCO Q2_0) shared-expert gate+up pairs that are not fused with the routed experts, by layers
PAIRS = [("q2_0", "q3_K", 8), ("q3_K", "q3_K", 6), ("q3_K", "q2_0", 3), ("q2_0", "q4_K", 3), ("q2_0", "iq4_xs", 3),
         ("iq4_xs", "q3_K", 3), ("q4_K", "q4_K", 2), ("q4_K", "q3_K", 2), ("q4_K", "q2_0", 2), ("q3_K", "iq4_xs", 2),
         ("iq4_xs", "q2_0", 2), ("q5_K", "q3_K", 1), ("q3_K", "q5_K", 1), ("iq4_xs", "q4_K", 1)]


def lpr_log2(nunits):
    l = 0
    while l < 5 and nunits % (2 << l) == 0:
        l += 1
    return l


def slot(t, l):
    ub, al, lpr = UB[t], AL[t], 1 << l
    seg = ub << l
    nu = (15 + seg + 8 + 15) // 16
    w = 16 if al >= 16 else 8 if al == 8 else 4
    nl = 128 // w
    best, best_cost = nu * 16, None
    for pad in range(8):
        s = (nu + pad) * 16
        cost = 0
        for l0 in range(0, 32, nl):
            worst = 0
            for bank in range(nl):
                addrs = {((lane >> l) * s + (lane & (lpr - 1)) * ub) // w * w for lane in range(l0, l0 + nl)
                         if (((lane >> l) * s + (lane & (lpr - 1)) * ub) // w) % nl == bank}
                worst = max(worst, len(addrs))
            cost += worst
        if best_cost is None or cost < best_cost:
            best, best_cost = s, cost
    return best


def shape(tg, tu, k, nt):
    need_s = any(t in ("q4_K", "q5_K") for t in (tg, tu))
    need_q3 = "q3_K" in (tg, tu)
    ls, slots, ws = [], [], []
    rbr_log2 = 0
    for t in (tg, tu):
        l = lpr_log2(k // UW[t])
        s = slot(t, l)
        ls.append(l); slots.append(s); ws.append((32 >> l) * s)
        rbr_log2 = max(rbr_log2, 5 - l)
    ngroups = k // 128
    qg, dg = 32 * nt + 4, 4 * nt + 1
    act = ngroups * qg * 4 + (ngroups * dg * 8 if need_q3 else 0) + ngroups * dg * 4 * (1 + need_s + need_q3)
    act = (act + 15) // 16 * 16
    pstage = ws[0] + ws[1]
    res = 4 * nt * (1 << rbr_log2) * 4
    np_ = 4
    while np_ > 1 and act + np_ * (pstage + res) > 48 * 1024:
        np_ -= 1
    return ls, slots, rbr_log2, act, pstage, res, np_, act + np_ * (pstage + res)


ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=2560); ap.add_argument("--rows", type=int, default=640)
ap.add_argument("--nsm", type=int, default=5)
a = ap.parse_args()
print("gate\tup\tlayers\ttokens\tgate_lanes_per_row\tup_lanes_per_row\tgate_slot_B\tup_slot_B\trows_per_row_block\t"
      "row_blocks\tactivations_B\tpair_staging_B\tpair_results_B\tpairs_per_block\tblock_smem_B\tblocks_by_smem_per_SM")
for tg, tu, n in PAIRS:
    both = all(t in ("q2_0", "q3_K") for t in (tg, tu))
    for nt in range(2, 9 if both else 5):
        ls, slots, rl, act, pstage, res, np_, smem = shape(tg, tu, a.k, nt)
        print(f"{tg}\t{tu}\t{n}\t{nt}\t{1 << ls[0]}\t{1 << ls[1]}\t{slots[0]}\t{slots[1]}\t{1 << rl}\t"
              f"{(a.rows + (1 << rl) - 1) >> rl}\t{act}\t{pstage}\t{res}\t{np_}\t{smem}\t{96 * 1024 // smem}")
