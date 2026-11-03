// Q2_0 expert products of a k-token pass (--q2_0-mtok) at the shapes of Qwen3.8-Flash-Next's routed experts: 48 layers,
// 10 of 512 experts per token, 640 wide, hidden 2560; gate/up (2560 -> 640) fused with SwiGLU, down (640 -> 2560). A
// speculative verify or a pipelined guess runs k = 2 to 8 tokens through one pass, and the k tokens share some of their
// experts. The question is how the cost of that pass grows with k.
//
// "pertok": Miner Llama's one-token Pascal Q2_0 kernel (q2_0_mv.cuh, the fork's mul_mat_vec_q_pascal) launched once
//   per token: every (token, slot) entry reads its expert's rows, whichever other token picked the same expert. That is
//   the cost without sharing, at the best one-token speed.
// "mtok": the fork's multi-token kernel (mul_mat_vec_q2_0_pascal_mtok of ggml/src/ggml-cuda/mmvq-pascal-mtok.cu,
//   branch pascal-p106-v3-b12076-mtok, copied below without the bias, the other GLU operations and the shared-expert
//   launch): one launch per step; block y is entry y of the step's (token, slot) entries, and among the entries that
//   picked one expert every gcap-th leads a group, whose block reads the expert's rows once and applies each decoded
//   weight to all members; the other blocks exit. The members' activations are staged in shared memory once per block,
//   regrouped as the one-token kernel regroups them. Same lane groups, staging and arithmetic as the one-token kernel,
//   so every output is bit-identical to pertok's.
// "mtok_cN": the same with groups of at most N entries (N = 1: no sharing, the kernel's own cost per entry);
//   "mtok_rbN": N row blocks per block (default 4 where an expert has 8 or more); "mtok_gdir": the gate/up weights
//   loaded directly instead of staged through shared memory; "mtok_mb4": compiled for 4 resident blocks per SM (at most
//   128 registers) instead of 3 (170).
//
// Routing (route profiles): "random" draws each token's 10 experts independently and uniformly; "local" models
// consecutive tokens of one text (an assumption, no Flash-Next routing trace exists yet): expert popularity skewed so
// that a quarter of the experts take 44 % of the picks (what Flint's routing showed in Q51), and each expert of the
// token before kept with probability 0.16 (Flint's share of experts between neighbouring tokens); "copy" the same with
// 0.6 (highly repetitive text); "trace" reads one layer's routing from a file (--route-trace: one token per line, its 10
// expert ids). Steps are consecutive windows of k tokens of one routed sequence.
#pragma once

#include "common.cuh"
#include "data.cuh"
#include "q2_0_mv.cuh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace mb {
namespace q2mt {

using q2mv::h2f_lo;
using q2mv::silu;
using q2mv::u16_at;
using q2mv::word_at;

#define Q2MT_NW  4  // warps per block
#define Q2MT_C   2  // Q2_0 blocks per lane and iteration
#define Q2MT_WB  36 // weight bytes per lane and iteration
#define Q2MT_CNA 4  // Q8_1 activation blocks per lane chunk
#define Q2MT_US  36 // staged words per lane chunk: 32 of int8 values, 4 of padding
#define Q2MT_MAX_COLS 8

enum { Q2MT_SEG = 0, Q2MT_TILE = 1, Q2MT_DIRECT = 2 };

struct mtok_args {
    const uint8_t * x;            // up (or the only matrix): all experts
    const uint8_t * gate;         // gate (fused SwiGLU) or nullptr
    const char    * y;            // Q8_1 activations
    const int32_t * ids;          // [token][slot]
    float         * dst;          // [token][slot][row]
    int             nrows;
    int             row_bytes;
    int             lpr_log2;
    int             niter;
    int             nrowblocks;
    int             rb;
    int             gcap;
    int             ncols;        // tokens
    int             nslots;
    int             nq;           // Q8_1 blocks per activation row
    int             act_u;
    int             act_tok;
    int             tile_bytes;
    int             y_slot_bytes; // bytes between the slots' activations (0: one vector per token)
    int             y_col_bytes;  // bytes between the tokens' activations
    int64_t         expert_bytes;
};

static __device__ __forceinline__ bool group(const mtok_args & p, int & nt, int & yoff, int & doff, int & expert) {
    const int lane = threadIdx.x;
    const int ne   = p.ncols * p.nslots;
    const int y    = blockIdx.y;
    const int ey   = p.ids[y];
    int      rank = 0;
    uint32_t mk[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const int  i   = lane + 32*h;
        const bool hit = i < ne && p.ids[i] == ey;
        rank += __popc(__ballot_sync(0xFFFFFFFF, hit && i < y));
        mk[h] = __ballot_sync(0xFFFFFFFF, hit && i >= y);
    }
    if (rank % p.gcap != 0) {
        return false;
    }
    int      total = 0;
    int      h     = -1;
    int      r     = 0;
    uint32_t m     = 0;
#pragma unroll
    for (int hh = 0; hh < 4; ++hh) {
        const int c = __popc(mk[hh]);
        if (h < 0 && lane < total + c) {
            h = hh;
            r = lane - total;
            m = mk[hh];
        }
        total += c;
    }
    nt = min(p.gcap, total);
    if (lane < nt) {
        for (int q = 0; q < r; ++q) {
            m &= m - 1;
        }
        const int i = 32*h + __ffs(m) - 1;
        const int t = i / p.nslots;
        const int k = i - t*p.nslots;
        yoff = k*p.y_slot_bytes + t*p.y_col_bytes;
        doff = i*p.nrows;
    }
    expert = ey;
    return true;
}

static __device__ __forceinline__ void decode(const uint32_t q, int * out) {
    constexpr uint32_t t = 0x020100FF;
    out[0] = (int) __byte_perm(t, t, q);
    out[1] = (int) __byte_perm(t, t, q >> 2);
    out[2] = (int) __byte_perm(t, t, q >> 16);
    out[3] = (int) __byte_perm(t, t, q >> 18);
}

static __device__ __forceinline__ const char * stage_tile(const char * x, const mtok_args & p, const int rblk, char * tiles) {
    constexpr int UN = 4;
    const int wrows  = 32 >> p.lpr_log2;
    const int wrow0  = (rblk * Q2MT_NW + (int) threadIdx.y) * wrows;
    const int wbytes = wrows * p.row_bytes;
    uint4       * dsm = (uint4 *) (tiles + threadIdx.y * wbytes);
    const uint4 * src = (const uint4 *) (x + (int64_t) wrow0 * p.row_bytes);
    const int     nu  = (max(0, min(wrows, p.nrows - wrow0)) * p.row_bytes) >> 4;
    __syncwarp();
    for (int u0 = threadIdx.x; u0 < nu; u0 += UN * 32) {
        uint4 v[UN];
#pragma unroll
        for (int j = 0; j < UN; ++j) {
            if (u0 + j * 32 < nu) {
                v[j] = __ldg(src + u0 + j * 32);
            }
        }
#pragma unroll
        for (int j = 0; j < UN; ++j) {
            if (u0 + j * 32 < nu) {
                dsm[u0 + j * 32] = v[j];
            }
        }
    }
    __syncwarp();
    return (const char *) dsm;
}

template <bool G, int ST, int NT>
static __device__ __forceinline__ void rows(const mtok_args & p, const char * xu, const char * xg, char * smw, char * tiles,
                                            const char * act, const int doff) {
    constexpr int M  = G ? 2 : 1;
    constexpr int WB = Q2MT_WB;
    constexpr int NU = (WB + 15) / 16;
    const int lane     = threadIdx.x;
    const int lpr_log2 = p.lpr_log2;
    const int lpr      = 1 << lpr_log2;
    const int sl       = lane & (lpr - 1);
    const int grp      = lane >> lpr_log2;
    const int wstep    = WB << lpr_log2;
    [[maybe_unused]] const int seg_units = wstep >> 4;
    (void) xg; (void) smw; (void) tiles;

    int dc[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        dc[j] = __shfl_sync(0xFFFFFFFF, doff, j);
    }
    for (int rbi = 0; rbi < p.rb; ++rbi) {
        const int rblk = (int) blockIdx.x * p.rb + rbi;
        if (rblk >= p.nrowblocks) {
            break;
        }
        const int row0 = ((rblk * Q2MT_NW + (int) threadIdx.y) << (5 - lpr_log2)) + grp;
        const char * rw[M];
        rw[0] = xu + (int64_t) min(row0, p.nrows - 1) * p.row_bytes + (ST == Q2MT_DIRECT ? sl * WB : 0);
        if constexpr (G) {
            rw[1] = xg + (int64_t) min(row0, p.nrows - 1) * p.row_bytes + (ST == Q2MT_DIRECT ? sl * WB : 0);
        }
        [[maybe_unused]] const char * smt = nullptr;
        if constexpr (ST == Q2MT_TILE) {
            smt = stage_tile(xu, p, rblk, tiles) + grp * p.row_bytes + sl * WB;
        }
        float acc[NT][M];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int m = 0; m < M; ++m) {
                acc[j][m] = 0.0f;
            }
        }
        for (int it = 0; it < p.niter; ++it) {
            uint32_t w[M][WB / 4];
            if constexpr (ST == Q2MT_SEG) {
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    uint4       * dsm = (uint4 *) (smw + (m * WB << lpr_log2));
                    const uint4 * src = (const uint4 *) rw[m];
#pragma unroll
                    for (int j = 0; j < NU; ++j) {
                        const int u = sl + (j << lpr_log2);
                        if (j < NU - 1 || u < seg_units) {
                            dsm[u] = __ldg(src + u);
                        }
                    }
                }
                __syncwarp();
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const uint32_t * s = (const uint32_t *) (smw + (m * WB << lpr_log2) + sl * WB);
#pragma unroll
                    for (int k = 0; k < WB / 4; ++k) {
                        w[m][k] = s[k];
                    }
                }
                __syncwarp();
            } else if constexpr (ST == Q2MT_TILE) {
                const uint32_t * s = (const uint32_t *) smt;
#pragma unroll
                for (int k = 0; k < WB / 4; ++k) {
                    w[0][k] = s[k];
                }
                smt += wstep;
            } else {
#pragma unroll
                for (int m = 0; m < M; ++m) {
#pragma unroll
                    for (int k = 0; k < WB / 4; ++k) {
                        w[m][k] = __ldg((const unsigned int *) rw[m] + k);
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
                rw[m] += wstep;
            }
            const int    c  = (it << lpr_log2) + sl;
            const char * au = act + c * (Q2MT_US * 4);
            const char * ad = act + p.act_u + c * (Q2MT_CNA * 4);
#pragma unroll
            for (int b = 0; b < Q2MT_C; ++b) {
                const int o = b * Q2MV_BS;
                float d[M];
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    d[m] = h2f_lo(u16_at(w[m], o));
                }
                int i0[NT][M];
#pragma unroll
                for (int sb = 0; sb < 2; ++sb) {
                    int q[M][8];
#pragma unroll
                    for (int m = 0; m < M; ++m) {
                        decode(word_at(w[m], o + 2 + 8*sb), q[m]);
                        decode(word_at(w[m], o + 6 + 8*sb), q[m] + 4);
                    }
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        const uint4 * pu = (const uint4 *) (au + j * p.act_tok + (2*b + sb) * 32);
                        const uint4   u0 = pu[0];
                        const uint4   u1 = pu[1];
                        float2 a2 = make_float2(0.0f, 0.0f);
                        if (sb == 1) {
                            a2 = *(const float2 *) (ad + j * p.act_tok + 8*b);
                        }
#pragma unroll
                        for (int m = 0; m < M; ++m) {
                            int isum = __dp4a(q[m][0], (int) u0.x, 0);
                            isum = __dp4a(q[m][1], (int) u0.y, isum);
                            isum = __dp4a(q[m][2], (int) u0.z, isum);
                            isum = __dp4a(q[m][3], (int) u0.w, isum);
                            isum = __dp4a(q[m][4], (int) u1.x, isum);
                            isum = __dp4a(q[m][5], (int) u1.y, isum);
                            isum = __dp4a(q[m][6], (int) u1.z, isum);
                            isum = __dp4a(q[m][7], (int) u1.w, isum);
                            if (sb == 0) {
                                i0[j][m] = isum;
                            } else {
                                acc[j][m] = fmaf(d[m], fmaf((float) isum, a2.y, (float) i0[j][m] * a2.x), acc[j][m]);
                            }
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                if (off < lpr) {
#pragma unroll
                    for (int m = 0; m < M; ++m) {
                        acc[j][m] += __shfl_xor_sync(0xFFFFFFFF, acc[j][m], off, 32);
                    }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            if (sl != 0 || row0 >= p.nrows) {
                break;
            }
            float result = acc[j][0];
            if constexpr (G) {
                result *= silu(acc[j][M - 1]);
            }
            p.dst[dc[j] + row0] = result;
        }
    }
}

template <bool G, int ST, int NT, int NTM>
static __device__ __forceinline__ void rows_nt(const int nt, const mtok_args & p, const char * xu, const char * xg,
                                               char * smw, char * tiles, const char * act, const int doff) {
    if constexpr (NT == NTM) {
        rows<G, ST, NT>(p, xu, xg, smw, tiles, act, doff);
    } else {
        if (nt == NT) {
            rows<G, ST, NT>(p, xu, xg, smw, tiles, act, doff);
        } else {
            rows_nt<G, ST, NT + 1, NTM>(nt, p, xu, xg, smw, tiles, act, doff);
        }
    }
}

// MINB: resident blocks per SM the registers must allow (the fork's MMVQP_MTOK_MIN_BLOCKS, 3 by default)
template <bool G, int ST, int NTM, int MINB>
__launch_bounds__(Q2MT_NW*32, MINB)
static __global__ void mmvq_mtok(const mtok_args p) {
    constexpr int M = G ? 2 : 1;
    __shared__ uint4 sm[ST == Q2MT_SEG ? Q2MT_NW * 32 * M * Q2MT_WB / 16 : 1];
    extern __shared__ uint4 q2mt_dyn[];
    const int lane = threadIdx.x;
    int nt = 0, yoff = 0, doff = 0, e = 0;
    if (!group(p, nt, yoff, doff, e)) {
        return;
    }
    char * tiles = (char *) q2mt_dyn;
    char * act   = tiles + p.tile_bytes;
    const int tid = threadIdx.y * 32 + lane;
    for (int j = 0; j < nt; ++j) {
        const char * yj = p.y + __shfl_sync(0xFFFFFFFF, yoff, j);
        char       * aj = act + j * p.act_tok;
        for (int q = tid; q < p.nq; q += Q2MT_NW * 32) {
            const unsigned int * ys = (const unsigned int *) (yj + q * (int) sizeof(block_q8_1));
            uint32_t v[9];
#pragma unroll
            for (int k = 0; k < 9; ++k) {
                v[k] = __ldg(ys + k);
            }
            uint4 * du = (uint4 *) (aj + ((q / Q2MT_CNA) * Q2MT_US + (q % Q2MT_CNA) * 8) * 4);
            du[0] = make_uint4(__byte_perm(v[1], v[2], 0x6420), __byte_perm(v[1], v[2], 0x7531),
                               __byte_perm(v[3], v[4], 0x6420), __byte_perm(v[3], v[4], 0x7531));
            du[1] = make_uint4(__byte_perm(v[5], v[6], 0x6420), __byte_perm(v[5], v[6], 0x7531),
                               __byte_perm(v[7], v[8], 0x6420), __byte_perm(v[7], v[8], 0x7531));
            ((float *) (aj + p.act_u))[q] = h2f_lo(v[0]);
        }
    }
    __syncthreads();
    const char * xu  = (const char *) p.x + (int64_t) e * p.expert_bytes;
    const char * xg  = G ? (const char *) p.gate + (int64_t) e * p.expert_bytes : nullptr;
    char       * smw = (char *) sm + (ST == Q2MT_SEG ? threadIdx.y * (32 * M * Q2MT_WB) + ((lane >> p.lpr_log2) * M * Q2MT_WB << p.lpr_log2) : 0);
    rows_nt<G, ST, 1, NTM>(nt, p, xu, xg, smw, tiles, act, doff);
}

template <int MINB>
static const void * mtok_func_mb(const bool G, const int st, const int ntm) {
#define Q2MT_F(G_, ST_) (ntm <= 2 ? (const void *) mmvq_mtok<G_, ST_, 2, MINB> : ntm <= 4 ? (const void *) mmvq_mtok<G_, ST_, 4, MINB> : \
                         (const void *) mmvq_mtok<G_, ST_, 8, MINB>)
    if (G) {
        return st == Q2MT_SEG ? Q2MT_F(true, Q2MT_SEG) : st == Q2MT_DIRECT ? Q2MT_F(true, Q2MT_DIRECT) : nullptr;
    }
    return st == Q2MT_SEG ? Q2MT_F(false, Q2MT_SEG) : st == Q2MT_TILE ? Q2MT_F(false, Q2MT_TILE) : nullptr;
#undef Q2MT_F
}

static const void * mtok_func(const bool G, const int st, const int ntm, const int minb) {
    return minb == 4 ? mtok_func_mb<4>(G, st, ntm) : mtok_func_mb<3>(G, st, ntm);
}

// The launch ggml_cuda_mmvq_pascal_mtok makes for contiguous, 16-byte aligned matrices of all experts: staging, row
// blocks per block, group size and shared memory. gcap 0: the token count; rb 0: the default; gdir: the gate's direct loads.
struct mtok_launch {
    mtok_args p;
    const void * func = nullptr;
    dim3 grid;
    int  smem = 0;
    int  st = Q2MT_SEG;
};

static bool mtok_setup(const int K, const int N, const bool G, const int ncols, const int nslots, const int gcap_opt,
                       const int rb_opt, const bool gdir, const int minb, mtok_launch & L) {
    const int blocks = K / Q2MV_QK;
    if (K % Q2MV_QK != 0 || blocks % Q2MT_C != 0 || ncols * nslots > 128 || ncols > Q2MT_MAX_COLS) {
        return false;
    }
    const int chunks = blocks / Q2MT_C;
    int lpr_log2 = 0;
    while (lpr_log2 < 5 && chunks % (2 << lpr_log2) == 0) {
        lpr_log2++;
    }
    const int  row_bytes = blocks * Q2MV_BS;
    const bool seg_ok    = (Q2MT_WB << lpr_log2) % 16 == 0 && row_bytes % 16 == 0;
    const int  wbytes    = (32 >> lpr_log2) * row_bytes;
    const bool tile_ok   = !G && wbytes % 16 == 0 && ((int64_t) N * row_bytes) % 16 == 0 && Q2MT_NW * wbytes <= 48 * 1024;
    int st;
    if (G) {
        st = seg_ok && !gdir ? Q2MT_SEG : Q2MT_DIRECT;
    } else if (seg_ok) {
        st = Q2MT_SEG;
    } else if (tile_ok) {
        st = Q2MT_TILE;
    } else {
        return false;
    }
    const int nq         = blocks * (Q2MV_QK / 32);
    const int act_u      = (nq / Q2MT_CNA) * Q2MT_US * 4;
    const int act_tok    = act_u + nq * 4;
    const int gcap       = gcap_opt > 0 ? gcap_opt : ncols;
    const int tile_bytes = st == Q2MT_TILE ? Q2MT_NW * wbytes : 0;
    const int smem       = tile_bytes + gcap * act_tok;
    const int smem_seg   = st == Q2MT_SEG ? Q2MT_NW * 32 * (G ? 2 : 1) * Q2MT_WB : 0;
    if (smem + smem_seg > 48 * 1024) {
        return false;
    }
    const int rows_per_block = Q2MT_NW * (32 >> lpr_log2);
    const int nrowblocks     = (N + rows_per_block - 1) / rows_per_block;
    const int rb             = std::min(nrowblocks, rb_opt > 0 ? rb_opt : (nrowblocks >= 8 ? 4 : 1));
    L.p = {};
    L.p.nrows        = N;
    L.p.row_bytes    = row_bytes;
    L.p.lpr_log2     = lpr_log2;
    L.p.niter        = chunks >> lpr_log2;
    L.p.nrowblocks   = nrowblocks;
    L.p.rb           = rb;
    L.p.gcap         = gcap;
    L.p.ncols        = ncols;
    L.p.nslots       = nslots;
    L.p.nq           = nq;
    L.p.act_u        = act_u;
    L.p.act_tok      = act_tok;
    L.p.tile_bytes   = tile_bytes;
    L.p.expert_bytes = (int64_t) N * row_bytes;
    L.func = mtok_func(G, st, gcap, minb);
    L.grid = dim3((unsigned) ((nrowblocks + rb - 1) / rb), (unsigned) (ncols * nslots), 1);
    L.smem = smem;
    L.st   = st;
    return L.func != nullptr;
}

// ---- routing -------------------------------------------------------------------------------------------------------

struct route_profile {
    std::string name;
    double keep    = 0.0;  // probability that a token keeps each expert of the token before
    double top_q   = 0.25; // popularity: the top_q of the experts take top_mass of the picks
    double top_mass = 0.25; // 0.25: uniform
};

// popularity weights over a random order of the experts, w_r ~ 1 / (r + 1)^s with s set so that the top_q of the
// experts carry top_mass of the weight
static std::vector<double> popularity(const int nexp, const route_profile & rp, uint64_t seed) {
    std::vector<double> w(nexp, 1.0);
    if (rp.top_mass > rp.top_q + 1e-9) {
        double lo = 0.0, hi = 4.0;
        for (int it = 0; it < 60; ++it) {
            const double s = 0.5 * (lo + hi);
            double tot = 0.0, top = 0.0;
            for (int r = 0; r < nexp; ++r) {
                const double x = pow(r + 1.0, -s);
                tot += x;
                if (r < (int) (rp.top_q * nexp)) top += x;
            }
            (top / tot < rp.top_mass ? lo : hi) = s;
        }
        for (int r = 0; r < nexp; ++r) w[r] = pow(r + 1.0, -0.5 * (lo + hi));
    }
    std::vector<int> order(nexp);
    for (int e = 0; e < nexp; ++e) order[e] = e;
    for (int e = nexp - 1; e > 0; --e) std::swap(order[e], order[hash_u32(seed * 31 + e) % (uint32_t) (e + 1)]);
    std::vector<double> cdf(nexp);
    double acc = 0.0;
    for (int r = 0; r < nexp; ++r) {
        acc += w[r];
        cdf[order[r]] = w[r];
    }
    acc = 0.0;
    for (int e = 0; e < nexp; ++e) {
        acc += cdf[e];
        cdf[e] = acc;
    }
    for (auto & c : cdf) c /= acc;
    return cdf;
}

// ntok consecutive tokens' expert ids (ntok * nused, token-major), from the profile or from a trace
static std::vector<int> route_tokens(const route_profile & rp, const std::vector<std::vector<int>> & trace, const int nexp,
                                     const int nused, const int ntok, const uint64_t seed) {
    std::vector<int> ids;
    ids.reserve((size_t) ntok * nused);
    if (rp.name == "trace") {
        for (int t = 0; t < ntok; ++t) {
            const auto & row = trace[(size_t) t % trace.size()];
            for (int j = 0; j < nused; ++j) ids.push_back(row[j] % nexp);
        }
        return ids;
    }
    const std::vector<double> cdf = popularity(nexp, rp, seed);
    uint64_t h = seed * 0x9E3779B97F4A7C15ull;
    auto rnd = [&]() { return (hash_u32(h++) + 0.5) / 4294967296.0; };
    std::vector<int> prev;
    for (int t = 0; t < ntok; ++t) {
        std::vector<int> cur;
        for (const int e : prev) {
            if (rnd() < rp.keep) cur.push_back(e);
        }
        while ((int) cur.size() < nused) {
            const int e = (int) (std::lower_bound(cdf.begin(), cdf.end(), rnd()) - cdf.begin());
            if (e < nexp && std::find(cur.begin(), cur.end(), e) == cur.end()) cur.push_back(e);
        }
        for (int j = nused - 1; j > 0; --j) std::swap(cur[j], cur[hash_u32(h++) % (uint32_t) (j + 1)]);
        ids.insert(ids.end(), cur.begin(), cur.end());
        prev = cur;
    }
    return ids;
}

// a routing trace: one token per line, its expert ids separated by blanks (more than nused: the first nused)
static bool read_trace(const std::string & path, const int nused, std::vector<std::vector<int>> & trace) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::vector<int> row;
        int e;
        while ((int) row.size() < nused && is >> e) row.push_back(e);
        if ((int) row.size() == nused) trace.push_back(row);
    }
    return !trace.empty();
}

static int distinct(const int * ids, const int n) {
    std::vector<int> v(ids, ids + n);
    std::sort(v.begin(), v.end());
    return (int) (std::unique(v.begin(), v.end()) - v.begin());
}

} // namespace q2mt
} // namespace mb
