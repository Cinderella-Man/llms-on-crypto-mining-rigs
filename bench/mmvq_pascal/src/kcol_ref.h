// Host reference for the k-column products (src/kcol_mm.cuh): type names and block sizes, ggml's reference
// dequantizers for the ten types (ggml/src/ggml-quants.c, MIT, the ggml authors), and the activations a kernel used,
// read back from its records. Used by src/kcol_main.cu and verify/kcol_unit_check.cu.
// With KCOL_BENCH_IQ (build_kcol.sh IQ_TREE=...) also the grid types and IQ1_M, from the grids of build/iq_grids.h.
#pragma once

#include "kcol_mm.cuh"
#ifdef KCOL_BENCH_IQ
#include "iq_grids.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace kcol {

inline double half_le(const uint8_t * p) {
    const unsigned short h = (unsigned short) (p[0] | (p[1] << 8));
    const int e = (h >> 10) & 0x1f, m = h & 0x3ff;
    const double v = e == 0 ? m / 16777216.0 : e == 31 ? (m ? NAN : INFINITY) : ldexp(1024.0 + m, e - 25);
    return h & 0x8000 ? -v : v;
}
inline int u16_le(const uint8_t * p) { return p[0] | (p[1] << 8); }

inline const char * kname(const int t) {
    switch (t) {
        case KQ8_0: return "q8_0"; case KQ2_0: return "q2_0"; case KQ3_K: return "q3_K"; case KQ4_K: return "q4_K";
        case KQ5_K: return "q5_K"; case KQ6_K: return "q6_K"; case KIQ4_XS: return "iq4_xs"; case KQ4_0: return "q4_0";
        case KQ5_0: return "q5_0"; case KIQ4_NL: return "iq4_nl"; case KIQ2_XXS: return "iq2_xxs"; case KIQ2_XS: return "iq2_xs";
        case KIQ2_S: return "iq2_s"; case KIQ3_XXS: return "iq3_xxs"; case KIQ3_S: return "iq3_s"; case KIQ1_M: return "iq1_m";
        default: return "?";
    }
}
inline int kbs(const int t) { // bytes and weights per block
    switch (t) {
        case KQ8_0: return 34; case KQ2_0: return 18; case KQ3_K: return 110; case KQ4_K: return 144;
        case KQ5_K: return 176; case KQ6_K: return 210; case KQ4_0: return 18; case KQ5_0: return 22;
        case KIQ4_NL: return 18; case KIQ2_XXS: return 66; case KIQ2_XS: return 74; case KIQ2_S: return 82;
        case KIQ3_XXS: return 98; case KIQ3_S: return 110; case KIQ1_M: return 56; default: return 136;
    }
}
inline int kqk(const int t) {
    return t == KQ8_0 || t == KQ4_0 || t == KQ5_0 || t == KIQ4_NL ? 32 : t == KQ2_0 ? 64 : 256;
}

// The grid types and IQ1_M (Q713), from ggml's reference dequantizers (dequantize_row_iq2_xxs .. _iq1_m of ggml-quants.c).
inline int gtype_of_rt(const int t) {
    return t == KIQ2_XXS || t == KIQ2_XS || t == KIQ2_S || t == KIQ3_XXS || t == KIQ3_S || t == KIQ1_M ? t : -1;
}
inline void dequant_iq(const int t, const uint8_t * x, double * y) {
#ifdef KCOL_BENCH_IQ
    using namespace kcol_grids;
    auto sgn = [](const int s, const int j) { return (s >> j) & 1 ? -1.0 : 1.0; };
    const uint8_t * qs = x + 2;
    if (t == KIQ2_XXS) {
        const double d = half_le(x);
        for (int ib = 0; ib < 8; ++ib) {
            uint32_t a0, a1;
            memcpy(&a0, qs + 8 * ib, 4);
            memcpy(&a1, qs + 8 * ib + 4, 4);
            const double db = d * (0.5 + (a1 >> 28)) * 0.25;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * g = (const uint8_t *) (iq2xxs_grid + ((a0 >> (8 * l)) & 0xff));
                const int s = ksigns_iq2xs[(a1 >> (7 * l)) & 127];
                for (int j = 0; j < 8; ++j) y[32 * ib + 8 * l + j] = db * g[j] * sgn(s, j);
            }
        }
    } else if (t == KIQ2_XS) {
        const double d = half_le(x);
        for (int ib = 0; ib < 8; ++ib) {
            const int sc = x[66 + ib];
            for (int l = 0; l < 4; ++l) {
                const int e = u16_le(qs + 2 * (4 * ib + l));
                const double db = d * (0.5 + ((l < 2 ? sc : sc >> 4) & 0xf)) * 0.25;
                const uint8_t * g = (const uint8_t *) (iq2xs_grid + (e & 511));
                const int s = ksigns_iq2xs[e >> 9];
                for (int j = 0; j < 8; ++j) y[32 * ib + 8 * l + j] = db * g[j] * sgn(s, j);
            }
        }
    } else if (t == KIQ2_S) {
        const double d = half_le(x);
        for (int ib = 0; ib < 8; ++ib) {
            const int sc = x[74 + ib], qh = x[66 + ib];
            for (int l = 0; l < 4; ++l) {
                const double db = d * (0.5 + ((l < 2 ? sc : sc >> 4) & 0xf)) * 0.25;
                const uint8_t * g = (const uint8_t *) (iq2s_grid + (qs[4 * ib + l] | ((qh << (8 - 2 * l)) & 0x300)));
                const int s = qs[32 + 4 * ib + l];
                for (int j = 0; j < 8; ++j) y[32 * ib + 8 * l + j] = db * g[j] * sgn(s, j);
            }
        }
    } else if (t == KIQ3_XXS) {
        const double d = half_le(x);
        for (int ib = 0; ib < 8; ++ib) {
            uint32_t aux;
            memcpy(&aux, x + 66 + 4 * ib, 4);
            const double db = d * (0.5 + (aux >> 28)) * 0.5;
            for (int l = 0; l < 4; ++l) {
                const int s = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                const uint8_t * g1 = (const uint8_t *) (iq3xxs_grid + qs[8 * ib + 2 * l]);
                const uint8_t * g2 = (const uint8_t *) (iq3xxs_grid + qs[8 * ib + 2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    y[32 * ib + 8 * l + j]     = db * g1[j] * sgn(s, j);
                    y[32 * ib + 8 * l + j + 4] = db * g2[j] * sgn(s, j + 4);
                }
            }
        }
    } else if (t == KIQ3_S) {
        const double d = half_le(x);
        for (int ib = 0; ib < 8; ++ib) {
            const int qh = x[66 + ib], sc = (x[106 + ib / 2] >> (4 * (ib % 2))) & 0xf;
            const double db = d * (1 + 2 * sc);
            for (int l = 0; l < 4; ++l) {
                const int s = x[74 + 4 * ib + l];
                const uint8_t * g1 = (const uint8_t *) (iq3s_grid + (qs[8 * ib + 2 * l] | ((qh << (8 - 2 * l)) & 256)));
                const uint8_t * g2 = (const uint8_t *) (iq3s_grid + (qs[8 * ib + 2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[32 * ib + 8 * l + j]     = db * g1[j] * sgn(s, j);
                    y[32 * ib + 8 * l + j + 4] = db * g2[j] * sgn(s, j + 4);
                }
            }
        }
    } else { // IQ1_M
        int sc[4];
        for (int k = 0; k < 4; ++k) sc[k] = u16_le(x + 48 + 2 * k);
        uint8_t h[2] = { (uint8_t) (((sc[0] >> 12) | ((sc[1] >> 8) & 0xf0)) & 0xff), (uint8_t) ((((sc[2] >> 4) & 0xf00) | (sc[3] & 0xf000)) >> 8) };
        const double d = half_le(h);
        for (int ib = 0; ib < 8; ++ib) {
            const int s = sc[ib / 2] >> (6 * (ib % 2));
            for (int l = 0; l < 4; ++l) {
                const int qhl = (x[32 + 2 * ib + l / 2] >> (4 * (l % 2))) & 0xf;
                const double dl = d * (2 * ((l < 2 ? s : s >> 3) & 7) + 1);
                const int8_t * g = (const int8_t *) (iq1s_grid + (x[4 * ib + l] | ((qhl & 7) << 8)));
                const double delta = qhl & 8 ? -0.125 : 0.125;
                for (int j = 0; j < 8; ++j) y[32 * ib + 8 * l + j] = dl * (g[j] + delta);
            }
        }
    }
#else
    (void) t; (void) x; (void) y;
#endif
}

// One row of weights as doubles, from ggml's reference dequantizers.
inline void dequant_row(const int t, const uint8_t * row, const int K, double * out) {
    static const int8_t kv[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
    const int bs = kbs(t), qk = kqk(t);
    for (int ib = 0; ib < K / qk; ++ib) {
        const uint8_t * x = row + (size_t) ib * bs;
        double * y = out + (size_t) ib * qk;
        if (t == KQ8_0) {
            const double d = half_le(x);
            for (int j = 0; j < 32; ++j) y[j] = d * (int8_t) x[2 + j];
        } else if (t == KQ2_0) {
            const double d = half_le(x);
            for (int j = 0; j < 64; ++j) y[j] = d * (((x[2 + j / 4] >> (2 * (j % 4))) & 3) - 1);
        } else if (t == KQ4_0) {
            const double d = half_le(x);
            for (int j = 0; j < 16; ++j) {
                y[j]      = d * ((x[2 + j] & 0xF) - 8);
                y[j + 16] = d * ((x[2 + j] >> 4) - 8);
            }
        } else if (t == KQ5_0) {
            const double d = half_le(x);
            uint32_t qh;
            memcpy(&qh, x + 2, 4);
            for (int j = 0; j < 16; ++j) {
                const int xh0 = ((qh >> (j + 0)) << 4) & 0x10;
                const int xh1 = ((qh >> (j + 12))) & 0x10;
                y[j]      = d * (((x[6 + j] & 0xF) | xh0) - 16);
                y[j + 16] = d * (((x[6 + j] >> 4) | xh1) - 16);
            }
        } else if (t == KIQ4_NL) {
            const double d = half_le(x);
            for (int j = 0; j < 16; ++j) {
                y[j]      = d * kv[x[2 + j] & 0xF];
                y[j + 16] = d * kv[x[2 + j] >> 4];
            }
        } else if (t == KQ3_K) {
            const uint8_t * hm = x;
            const uint8_t * q  = x + 32;
            const double d_all = half_le(x + 108);
            uint32_t aux[4];
            memcpy(aux, x + 96, 12);
            const uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4);
            aux[3] = ((aux[1] >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4);
            aux[0] = (aux[0] & 0x0f0f0f0f) | (((tmp >> 0) & 0x03030303) << 4);
            aux[1] = (aux[1] & 0x0f0f0f0f) | (((tmp >> 2) & 0x03030303) << 4);
            const int8_t * sc = (const int8_t *) aux;
            int is = 0;
            uint8_t m = 1;
            for (int n = 0; n < 256; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    for (int h = 0; h < 2; ++h) {
                        const double dl = d_all * (sc[is++] - 32);
                        for (int l = 0; l < 16; ++l) {
                            *y++ = dl * ((int) ((q[l + 16 * h] >> shift) & 3) - ((hm[l + 16 * h] & m) ? 0 : 4));
                        }
                    }
                    shift += 2;
                    m <<= 1;
                }
                q += 32;
            }
        } else if (t == KQ4_K || t == KQ5_K) {
            const double d = half_le(x), dmin = half_le(x + 2);
            const uint8_t * scales = x + 4;
            const uint8_t * qh = x + 16;
            const uint8_t * ql = x + (t == KQ5_K ? 48 : 16);
            for (int s = 0; s < 8; ++s) {
                int sc, mn;
                if (s < 4) { sc = scales[s] & 63; mn = scales[s + 4] & 63; }
                else { sc = (scales[s + 4] & 0xF) | ((scales[s - 4] >> 6) << 4); mn = (scales[s + 4] >> 4) | ((scales[s] >> 6) << 4); }
                for (int l = 0; l < 32; ++l) {
                    int v = (s & 1) ? ql[32 * (s / 2) + l] >> 4 : ql[32 * (s / 2) + l] & 0xF;
                    if (t == KQ5_K) v += (qh[l] >> s) & 1 ? 16 : 0;
                    y[32 * s + l] = d * sc * v - dmin * mn;
                }
            }
        } else if (t == KQ6_K) {
            const uint8_t * ql = x;
            const uint8_t * qh = x + 128;
            const int8_t  * sc = (const int8_t *) (x + 192);
            const double d = half_le(x + 208);
            for (int n = 0; n < 2; ++n) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int q1 = ((ql[64 * n + l] & 0xF)      | (((qh[32 * n + l] >> 0) & 3) << 4)) - 32;
                    const int q2 = ((ql[64 * n + l + 32] & 0xF) | (((qh[32 * n + l] >> 2) & 3) << 4)) - 32;
                    const int q3 = ((ql[64 * n + l] >> 4)       | (((qh[32 * n + l] >> 4) & 3) << 4)) - 32;
                    const int q4 = ((ql[64 * n + l + 32] >> 4)  | (((qh[32 * n + l] >> 6) & 3) << 4)) - 32;
                    y[128 * n + l +  0] = d * sc[8 * n + is + 0] * q1;
                    y[128 * n + l + 32] = d * sc[8 * n + is + 2] * q2;
                    y[128 * n + l + 64] = d * sc[8 * n + is + 4] * q3;
                    y[128 * n + l + 96] = d * sc[8 * n + is + 6] * q4;
                }
            }
        } else if (gtype_of_rt(t) >= 0) {
            dequant_iq(t, x, y);
        } else { // IQ4_XS
            const double d = half_le(x);
            const int sh = u16_le(x + 2);
            for (int b = 0; b < 8; ++b) {
                const int ls = ((x[4 + b / 2] >> (4 * (b % 2))) & 0xf) | (((sh >> (2 * b)) & 3) << 4);
                const double dl = d * (ls - 32);
                for (int j = 0; j < 16; ++j) {
                    y[32 * b + j]      = dl * kv[x[8 + 16 * b + j] & 0xf];
                    y[32 * b + j + 16] = dl * kv[x[8 + 16 * b + j] >> 4];
                }
            }
        }
    }
}

// The activations the kernel uses, read back from its records: value i of column j is q * d8.
template <int T>
inline void act_from_records(const std::vector<uint32_t> & rec, const int K, const int ncols, std::vector<double> & a) {
    constexpr int UW = traits<T>::UW, NAB = UW / 32, RS = rec_words<T>();
    const int nunits = K / UW;
    a.assign((size_t) K * ncols, 0.0);
    for (int j = 0; j < ncols; ++j) {
        for (int blk = 0; blk < K / 32; ++blk) {
            const int u = blk / NAB, b = blk % NAB;
            const uint32_t * r = rec.data() + ((size_t) j * nunits + u) * RS;
            float d;
            memcpy(&d, r + UW / 4 + 2 * b, 4);
            const double f = T == KQ3_K ? (double) (1 << (2 * std::min(b & 3, 2))) : 1.0;
            for (int i = 0; i < 32; ++i) {
                const int8_t q = ((const int8_t *) r)[32 * b + act_pos<traits<T>::PERM>(i)];
                a[(size_t) j * K + blk * 32 + i] = q * (double) d * f;
            }
        }
    }
}

inline void act_values(const int t, const std::vector<uint32_t> & rec, const int K, const int ncols, std::vector<double> & a) {
    switch (t) {
        case KQ8_0:   act_from_records<KQ8_0>(rec, K, ncols, a);   break;
        case KQ2_0:   act_from_records<KQ2_0>(rec, K, ncols, a);   break;
        case KQ3_K:   act_from_records<KQ3_K>(rec, K, ncols, a);   break;
        case KQ4_K:   act_from_records<KQ4_K>(rec, K, ncols, a);   break;
        case KQ5_K:   act_from_records<KQ5_K>(rec, K, ncols, a);   break;
        case KQ6_K:   act_from_records<KQ6_K>(rec, K, ncols, a);   break;
        case KQ4_0:   act_from_records<KQ4_0>(rec, K, ncols, a);   break;
        case KQ5_0:   act_from_records<KQ5_0>(rec, K, ncols, a);   break;
        case KIQ4_NL: act_from_records<KIQ4_NL>(rec, K, ncols, a); break;
        case KIQ2_XXS: act_from_records<KIQ2_XXS>(rec, K, ncols, a); break;
        case KIQ2_XS:  act_from_records<KIQ2_XS>(rec, K, ncols, a);  break;
        case KIQ2_S:   act_from_records<KIQ2_S>(rec, K, ncols, a);   break;
        case KIQ3_XXS: act_from_records<KIQ3_XXS>(rec, K, ncols, a); break;
        case KIQ3_S:   act_from_records<KIQ3_S>(rec, K, ncols, a);   break;
        case KIQ1_M:   act_from_records<KIQ1_M>(rec, K, ncols, a);   break;
        default:      act_from_records<KIQ4_XS>(rec, K, ncols, a); break;
    }
}

} // namespace kcol
