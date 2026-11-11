#!/usr/bin/env bash
# shapes.sh: C726's launch shapes, from the host code of mmvq-pascal-q3kmt.cu (q3m_act_bytes, q3m_conflicts, q3m_pad,
# q3m_nw and the launch's blocks per SM) replayed in awk, for Q3_K rows of K weights (row_bytes = 110 K / 256) at 2 to 8
# tokens and R = 1 and 2, with the switches at their defaults (a product where fewer than 2 warps a block fit stays on
# the k-column kernel). No GPU.
# Usage: bash shapes.sh [K ...] > shapes.tsv   (default: 2560 6144, Qwen3.8-Flash-Next's)
set -euo pipefail
KS=${*:-2560 6144}
awk -v ks="$KS" '
function lpr_log2(nsb,   l) { l = 0; while (l < 5 && nsb % (2 * 2^l) == 0) l++; return l }
function act_bytes(nt, nsb,   q, d, h) {
    q = nsb * (16 * nt + 1) * 16; d = int((nsb * (8 * nt + 1) * 4 + 7) / 8) * 8; h = nsb * (8 * nt + 1) * 8
    return int((q + d + h + 15) / 16) * 16 }
# the most distinct words of one bank that a 32-bit read of the same word of every lane s super-block asks for
function conflicts(ll, r, rb, slot,   L, worst, i, lane, s, h, w, bank, n, k, dup, word, seen) {
    L = 2^ll; worst = 0
    for (i = 0; i < r; i++) {
        for (lane = 0; lane < 32; lane++) {
            s = int(lane / L) * r + i; h = (s * rb) % 16
            word[lane] = int((s * slot + (h + 110 * (lane % L)) - ((h + 110 * (lane % L)) % 4)) / 4)
        }
        for (bank = 0; bank < 32; bank++) {
            n = 0
            for (lane = 0; lane < 32; lane++) {
                if (word[lane] % 32 != bank) continue
                dup = 0; for (k = 0; k < n; k++) if (seen[k] == word[lane]) dup = 1
                if (!dup) seen[n++] = word[lane]
            }
            if (n > worst) worst = n
        }
    }
    return worst }
function pad(ll, r, rb,   p, best, bn, n) {
    best = 0; bn = 1e9
    for (p = 0; p < 8; p++) { n = conflicts(ll, r, rb, 16 * 2^(ll + 3) + 16 * p); if (n < bn) { best = p; bn = n } }
    PADC = bn; return best }
BEGIN {
    printf "K\trow_bytes\tL\tNT\tR\tpad_units\tconflict_degree\tconflict_degree_pad0\tslot_bytes\tactivations_bytes\twarp_staging_bytes\twarps\tsmem_bytes\tblocks_per_SM\twarps_per_SM\n"
    n = split(ks, K, " ")
    for (ki = 1; ki <= n; ki++) {
        nsb = K[ki] / 256; rb = nsb * 110; ll = lpr_log2(nsb)
        for (nt = 2; nt <= 8; nt++) for (r = 1; r <= 2; r++) {
            if (nsb % 2 != 0) { printf "%d\t%d\t%d\t%d\t%d\tdeclined (odd super-blocks)\n", K[ki], rb, 2^ll, nt, r; continue }
            p = pad(ll, r, rb); c = PADC; c0 = conflicts(ll, r, rb, 16 * 2^(ll + 3))
            slot = 16 * 2^(ll + 3) + 16 * p; act = act_bytes(nt, nsb); ws = (32 / 2^ll) * r * slot
            nw = int(6 / r); while (nw > 0 && act + nw * ws > 49152) nw--
            if (nw < 2) { printf "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\tdeclined (%s)\n", K[ki], rb, 2^ll, nt, r, p, c, c0, slot, act, ws, nw, nw ? "one warp a block" : "48 KB"; continue }
            smem = act + nw * ws; bps = int(98304 / smem); t = int(2048 / (nw * 32)); if (t < bps) bps = t; if (bps < 1) bps = 1
            printf "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", K[ki], rb, 2^ll, nt, r, p, c, c0, slot, act, ws, nw, smem, bps, bps * nw
        }
    }
}'
