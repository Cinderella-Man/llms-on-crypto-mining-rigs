// Lossless codes for BF16 weight matrices (Q621), host side: the encoders, a bit-exact CPU decoder of each code that
// runs the same byte-permute steps as the CUDA kernels (prmt is emulated with the PTX semantics), and the statistics
// the entropy tool reports. Used by src/lossless_entropy.cpp (CPU only) and src/lossless_main.cu (the P106 bench).
//
// A BF16 weight is 16 bits: sign (bit 15), exponent (bits 14..7), mantissa (bits 6..0). Both codes keep its low byte
// (exponent bit 0 and the 7 mantissa bits) raw and code its high byte, sign | exponent >> 1, which carries nearly all of
// the redundancy: the exponents of a weight matrix sit in a narrow band. "pair" below is exponent >> 1, the high byte's
// low 7 bits.
//
// x12 (12 bits a weight): each weight's high byte is a 4-bit index into a 16-entry table per row: the 8 pairs
//   P, P-1, ..., P-7 of a window chosen per row, times the two signs. Nibble = bank << 3 | sign << 2 | j, with
//   d = P - pair, bank = d >= 4, j = d & 3. The kernel decodes 4 high bytes with two byte permutes and an add:
//   prmt(t0, t1, nib & 0x7777) gives the bank-1 value P-4-j with its sign (t0 = bytes P-4..P-7, t1 = t0 | 0x80 each),
//   and prmt(0x04040404, 0x04040404, nib) gives 4 where the nibble's bit 3 is clear (bit 3 set makes prmt replicate the
//   selected byte's top bit, which is 0), lifting bank 0 to P-j. Weights whose pair falls outside the window are
//   escapes: their true value is in a per-row list (column, BF16) and the kernel puts it in place before the products,
//   so the products run in the plain kernel's order on the same values and the output equals the BF16 kernel's bit for
//   bit.
// x11 (11 bits a weight): a 3-bit index into an 8-entry table per row (pairs P..P-3 times the two signs; one prmt
//   decodes 4 high bytes), packed 32 weights to three 32-bit words: word k (k = 0, 1, 2) holds in bits 0..2 of nibble m
//   the index of weight 8k + m and in bit 3 of nibble m bit k of the index of weight 24 + m. Escapes as in x12.
//
// Layout of both: a row of K weights is K / 8 (x12) or K / 32 (x11) chunks; the low bytes are a plane of K bytes a row,
// the indices a plane of K / 2 bytes a row, chunk after chunk (x12), or three planes of K / 8 bytes a row, word k of
// every chunk in plane k (x11), so a lane's loads of one chunk are coalesced across a warp.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace lossless {

// PTX prmt.b32 in its default mode: bytes 0..3 of a, then 4..7 of b; each selector nibble picks a byte with its 3 low
// bits, and its bit 3 replicates that byte's top bit over the result byte.
inline uint32_t prmt_host(uint32_t a, uint32_t b, uint32_t c) {
    const uint64_t src = (uint64_t) b << 32 | a;
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t s = (c >> (4 * i)) & 0xF;
        uint32_t v = (uint32_t) (src >> (8 * (s & 7))) & 0xFF;
        if (s & 8) {
            v = (v & 0x80) ? 0xFF : 0x00;
        }
        r |= v << (8 * i);
    }
    return r;
}

inline int pair_of(uint16_t u) { return (u >> 8) & 0x7F; }
inline int sign_of(uint16_t u) { return u >> 15; }

// the window [top - (width - 1), top] that holds the most of hist (128 pairs); top >= width - 1
inline int best_window(const uint64_t * hist, int width, uint64_t * covered) {
    uint64_t best = 0, run = 0;
    int top = width - 1;
    for (int p = 0; p < 128; ++p) {
        run += hist[p];
        if (p >= width) {
            run -= hist[p - width];
        }
        if (p >= width - 1 && run > best) {
            best = run;
            top = p;
        }
    }
    if (covered) {
        *covered = best;
    }
    return top;
}

struct encoded {
    int code = 0;              // 11 or 12
    int K = 0, N = 0;
    std::vector<uint8_t>  lo;  // N * K
    std::vector<uint32_t> idx; // x12: N * K / 8 words; x11: N * 3 * K / 32 words
    std::vector<uint32_t> rowinfo; // N + 1: the row's first escape (bits 0..23) | its window's top pair P << 24
    std::vector<uint32_t> esc;     // column | BF16 << 16, by row then column
    int top(int r) const { return (int) (rowinfo[r] >> 24); }
    uint32_t esc_begin(int r) const { return rowinfo[r] & 0xFFFFFFu; }
    // what the kernel reads per call: the planes, the row words, the escapes
    size_t bytes() const { return lo.size() + idx.size() * 4 + rowinfo.size() * 4 + esc.size() * 4; }
    double bits_per_weight() const { return 8.0 * bytes() / ((double) K * N); }
};

// x12 and x11 encoders. w: N rows of K BF16 values.
inline encoded encode(const uint16_t * w, int K, int N, int code) {
    encoded e;
    e.code = code;
    e.K = K;
    e.N = N;
    const int width = code == 12 ? 8 : 4;
    e.lo.resize((size_t) N * K);
    e.idx.assign(code == 12 ? (size_t) N * K / 8 : (size_t) N * 3 * K / 32, 0u);
    e.rowinfo.resize(N + 1);
    std::vector<uint8_t> sel(K);
    for (int r = 0; r < N; ++r) {
        const uint16_t * row = w + (size_t) r * K;
        uint64_t hist[128] = { 0 };
        for (int k = 0; k < K; ++k) {
            hist[pair_of(row[k])]++;
        }
        const int P = best_window(hist, width, nullptr);
        e.rowinfo[r] = (uint32_t) e.esc.size() | (uint32_t) P << 24;
        for (int k = 0; k < K; ++k) {
            const uint16_t u = row[k];
            e.lo[(size_t) r * K + k] = (uint8_t) (u & 0xFF);
            const int d = P - pair_of(u);
            if (d < 0 || d >= width) {
                e.esc.push_back((uint32_t) k | (uint32_t) u << 16);
                sel[k] = 0;
                continue;
            }
            sel[k] = code == 12 ? (uint8_t) ((d >= 4) << 3 | sign_of(u) << 2 | (d & 3)) : (uint8_t) (sign_of(u) << 2 | d);
        }
        if (code == 12) {
            for (int c = 0; c < K / 8; ++c) {
                uint32_t n = 0;
                for (int j = 0; j < 8; ++j) {
                    n |= (uint32_t) sel[8 * c + j] << (4 * j);
                }
                e.idx[(size_t) r * K / 8 + c] = n;
            }
        } else {
            for (int c = 0; c < K / 32; ++c) {
                uint32_t wd[3] = { 0, 0, 0 };
                for (int k = 0; k < 3; ++k) {
                    for (int m = 0; m < 8; ++m) {
                        wd[k] |= (uint32_t) sel[32 * c + 8 * k + m] << (4 * m);
                        wd[k] |= (uint32_t) ((sel[32 * c + 24 + m] >> k) & 1) << (4 * m + 3);
                    }
                }
                for (int k = 0; k < 3; ++k) {
                    e.idx[((size_t) r * 3 + k) * (K / 32) + c] = wd[k];
                }
            }
        }
    }
    e.rowinfo[N] = (uint32_t) e.esc.size();
    return e;
}

// The tables of a row: x12 t0 = bytes P-4..P-7, x11 t0 = bytes P..P-3; t1 = t0 with the sign bits.
inline void tables(int code, int P, uint32_t & t0, uint32_t & t1) {
    const uint32_t b = (uint32_t) (code == 12 ? P - 4 : P);
    t0 = b * 0x01010101u - 0x03020100u;
    t1 = t0 | 0x80808080u;
}

// x11: the selectors of group g (weights 8g .. 8g+7 of a 32-weight chunk) as 8 nibbles with bit 3 clear
inline uint32_t x11_sel(const uint32_t * wd, int g) {
    if (g < 3) {
        return wd[g] & 0x77777777u;
    }
    return ((wd[0] >> 3) & 0x11111111u) | ((wd[1] >> 2) & 0x22222222u) | ((wd[2] >> 1) & 0x44444444u);
}

// Decodes row r back to BF16 with the kernels' steps (escapes put back in place), for the CPU check.
inline void decode_row(const encoded & e, int r, uint16_t * out) {
    const int K = e.K;
    uint32_t t0, t1;
    tables(e.code, e.top(r), t0, t1);
    const uint8_t * lo = e.lo.data() + (size_t) r * K;
    auto emit4 = [&](int k0, uint32_t H) {
        uint32_t L;
        memcpy(&L, lo + k0, 4);
        const uint32_t u01 = prmt_host(L, H, 0x5140), u23 = prmt_host(L, H, 0x7362);
        out[k0 + 0] = (uint16_t) (u01 & 0xFFFF);
        out[k0 + 1] = (uint16_t) (u01 >> 16);
        out[k0 + 2] = (uint16_t) (u23 & 0xFFFF);
        out[k0 + 3] = (uint16_t) (u23 >> 16);
    };
    if (e.code == 12) {
        for (int c = 0; c < K / 8; ++c) {
            const uint32_t n = e.idx[(size_t) r * K / 8 + c];
            const uint32_t m = n & 0x77777777u;
            emit4(8 * c + 0, prmt_host(t0, t1, m)       + prmt_host(0x04040404u, 0x04040404u, n));
            emit4(8 * c + 4, prmt_host(t0, t1, m >> 16) + prmt_host(0x04040404u, 0x04040404u, n >> 16));
        }
    } else {
        for (int c = 0; c < K / 32; ++c) {
            const uint32_t wd[3] = { e.idx[((size_t) r * 3 + 0) * (K / 32) + c], e.idx[((size_t) r * 3 + 1) * (K / 32) + c],
                                     e.idx[((size_t) r * 3 + 2) * (K / 32) + c] };
            for (int g = 0; g < 4; ++g) {
                const uint32_t s = x11_sel(wd, g);
                emit4(32 * c + 8 * g + 0, prmt_host(t0, t1, s));
                emit4(32 * c + 8 * g + 4, prmt_host(t0, t1, s >> 16));
            }
        }
    }
    for (uint32_t i = e.esc_begin(r); i < e.esc_begin(r + 1); ++i) {
        out[e.esc[i] & 0xFFFF] = (uint16_t) (e.esc[i] >> 16);
    }
}

// ---- statistics for the entropy tool -----------------------------------------------------------------------------

inline double entropy(const uint64_t * h, int n, uint64_t total) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        if (h[i]) {
            const double p = (double) h[i] / (double) total;
            s -= p * std::log2(p);
        }
    }
    return s;
}

// total bits of an optimal (unlimited-length) Huffman code for the counts
inline double huffman_bits(const uint64_t * h, int n) {
    std::vector<uint64_t> q;
    for (int i = 0; i < n; ++i) {
        if (h[i]) {
            q.push_back(h[i]);
        }
    }
    if (q.size() < 2) {
        return q.empty() ? 0.0 : (double) q[0];
    }
    // the sum of the merged weights is the total code length
    std::make_heap(q.begin(), q.end(), std::greater<uint64_t>());
    double bits = 0.0;
    while (q.size() > 1) {
        std::pop_heap(q.begin(), q.end(), std::greater<uint64_t>());
        const uint64_t a = q.back();
        q.pop_back();
        std::pop_heap(q.begin(), q.end(), std::greater<uint64_t>());
        const uint64_t b = q.back();
        q.pop_back();
        bits += (double) (a + b);
        q.push_back(a + b);
        std::push_heap(q.begin(), q.end(), std::greater<uint64_t>());
    }
    return bits;
}

} // namespace lossless
