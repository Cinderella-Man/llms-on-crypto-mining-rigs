// Q2_0 matrix-vector products of one token (--q2_0) at the shapes of Qwen3.8-Flash-Next's experts: 48 layers, each with
// 10 of 512 routed experts and one shared expert, all 640 wide, hidden 2560. Gate and up (2560 -> 640) run as one kernel
// fused with SwiGLU, as llama.cpp runs them; down is 640 -> 2560.
//
// Q2_0 (ggml-common.h): 18 bytes per 64 weights, an FP16 scale d, then 16 bytes of 2-bit codes, weight j in bits
// 2 (j % 4), 2 (j % 4) + 1 of byte j / 4. Value d * (code - 1), codes 0..3 -> -1, 0, 1, 2.
//
// "llama": llama.cpp's mul_mat_vec_q<GGML_TYPE_Q2_0, 1, has_fusion, small_k = true> as a compute 6.1 card runs it
//   (ggml/src/ggml-cuda/mmvq.cu and vec_dot_q2_0_q8_1 of vecdotq.cuh, as in Miner Llama's pascal-p106-v3 branch at
//   37e717ff6; MIT, the ggml authors), without bias or the shared-expert launch. A Q2_0 row of 2560 or 640 weights has
//   fewer blocks than one pass of a 4-warp block covers (64 blocks of 64 weights), so llama.cpp takes its small-K shape:
//   4 warps and 4 rows per block, each thread one 32-weight half block per pass; 80 (K = 2560) or 20 (K = 640) of the
//   128 threads hold work, and the partial sums meet in shared memory.
// "pascal": Miner Llama's Pascal Q2_0 kernel (the Q2_0 path of mul_mat_vec_q_pascal for one token, copied below from
//   the fork's ggml/src/ggml-cuda/mmvq-pascal.cu and mmvq-pascal-impl.cuh, branch pascal-p106-v3-q2_0mv) at the launch
//   shape the fork picks (q2p_pick, copied below): lane groups of 2^l lanes sized to the row, R rows per lane group
//   sharing every activation load, two blocks (36 bytes) per lane and iteration, codes looked up with __byte_perm in a
//   4-byte table, activations split once per load into even and odd bytes. Gate/up (K = 2560): 4 lanes per row, R = 2,
//   each lane group staging its row segments through shared memory with 128-bit loads. Down (K = 640, 180-byte rows):
//   1 lane per row, R = 1, each warp staging all its 32 rows at once (a contiguous 5,760-byte tile).
// "rR_seg", "rR_tile", "rR_dir": the same kernel at R rows per lane group (1 or 2) with segment staging, tile staging
//   or direct 32-bit loads, where the row allows it (GGML_CUDA_MMVQ_PASCAL_Q2_0 selects the same shapes in the fork).
#pragma once

#include "common.cuh"
#include "data.cuh"

namespace mb {
namespace q2mv {

#define Q2MV_QK 64
#define Q2MV_BS 18

// n Q2_0 blocks: scales uniform in [0.5, 1.5) / 64, codes uniform random
static __global__ void gen_q2_0(uint8_t * __restrict__ dst, const size_t nblocks, const uint64_t seed) {
    for (size_t b = (size_t) blockIdx.x * blockDim.x + threadIdx.x; b < nblocks; b += (size_t) gridDim.x * blockDim.x) {
        uint8_t * p = dst + b * Q2MV_BS;
        const uint64_t base = seed * 0x2545F4914F6CDD1Dull + b * 8;
        *(half *) p = __float2half((0.5f + u01(hash_u32(base))) * (1.0f / 64.0f)); // 18-byte blocks: 2-byte aligned
        for (int j = 0; j < 16; j += 4) {
            const uint32_t h = hash_u32(base + 1 + j);
            for (int k = 0; k < 4; ++k) {
                p[2 + j + k] = (uint8_t) (h >> (8 * k));
            }
        }
    }
}

// ---- llama.cpp ---------------------------------------------------------------------------------------------------

struct llama_args {
    const uint8_t    * x;         // up (or the only matrix): all experts
    const uint8_t    * gate;      // gate (fused SwiGLU) or nullptr
    const block_q8_1 * y;
    const int32_t    * ids;       // expert of each slot (blockIdx.y), or nullptr
    float            * dst;       // [slot][row]
    int                ncols;
    int                nrows;
    int                y_slot_blocks; // Q8_1 blocks between the slots' activations (0: one vector for all)
    int64_t            expert_blocks; // weight blocks per expert
};

static __device__ __forceinline__ float vec_dot_q2_0_q8_1(const void * vbq, const block_q8_1 * bq8_1, const int kbx,
                                                          const int iqs) {
    const uint8_t * b2 = (const uint8_t *) vbq + (size_t) kbx * Q2MV_BS;
    const float     d2 = __half2float(*(const half *) b2);
    const int16_t * qs = (const int16_t *) (b2 + 2) + iqs * 4;
    const block_q8_1 * bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = ((const int *) bq8_1_chunk->qs)[j*2 + 0];
        const int v = ((const int *) bq8_1_chunk->qs)[j*2 + 1];
        // unpack even and odd crumbs into byte values
        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        // unshuffle values
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = __dp4a(u, qx, sumi);
        sumi = __dp4a(v, qy, sumi);
    }
    const float d8 = __low2float(bq8_1_chunk->ds);
    return d2 * d8 * sumi;
}

static __device__ __forceinline__ float silu(const float x) {
    return x / (1.0f + expf(-x));
}

// mul_mat_vec_q<GGML_TYPE_Q2_0, 1, G, small_k = true>: qk 64, qi 2, vdr 1, 4 warps, 4 rows per block
template <bool G>
__launch_bounds__(128, 1)
static __global__ void mmvq_llama(const llama_args a) {
    constexpr int qk = Q2MV_QK, qi = 2, vdr = 1, nwarps = 4, rpb = 4, warp_size = 32;
    const int tid  = warp_size*threadIdx.y + threadIdx.x;
    const int row0 = rpb*blockIdx.x;
    const int blocks_per_row_x = a.ncols / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;
    const int slot = blockIdx.y;
    const int e    = a.ids ? a.ids[slot] : 0;
    const block_q8_1 * y = a.y + (size_t) slot * a.y_slot_blocks;
    const int64_t kbx_offset = e*a.expert_blocks + (int64_t) row0*blocks_per_row_x;

    float tmp[rpb] = { 0.0f };
    float tmp_gate[rpb] = { 0.0f };
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
        for (int i = 0; i < rpb; ++i) {
            tmp[i] += vec_dot_q2_0_q8_1(a.x + kbx_offset*Q2MV_BS, &y[kby], i*blocks_per_row_x + kbx, kqs);
            if constexpr (G) {
                tmp_gate[i] += vec_dot_q2_0_q8_1(a.gate + kbx_offset*Q2MV_BS, &y[kby], i*blocks_per_row_x + kbx, kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1][rpb][warp_size];
    __shared__ float tmp_shared_gate[G ? nwarps-1 : 1][rpb][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < rpb; ++i) {
            tmp_shared[threadIdx.y-1][i][threadIdx.x] = tmp[i];
            if constexpr (G) {
                tmp_shared_gate[threadIdx.y-1][i][threadIdx.x] = tmp_gate[i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int i = 0; i < rpb; ++i) {
#pragma unroll
        for (int l = 0; l < nwarps-1; ++l) {
            tmp[i] += tmp_shared[l][i][threadIdx.x];
            if constexpr (G) {
                tmp_gate[i] += tmp_shared_gate[l][i][threadIdx.x];
            }
        }
#pragma unroll
        for (int off = warp_size/2; off > 0; off >>= 1) {
            tmp[i] += __shfl_xor_sync(0xffffffff, tmp[i], off, warp_size);
            if constexpr (G) {
                tmp_gate[i] += __shfl_xor_sync(0xffffffff, tmp_gate[i], off, warp_size);
            }
        }
        if (threadIdx.x == i && row0 + i < a.nrows) {
            float result = tmp[i];
            if constexpr (G) {
                result *= silu(tmp_gate[i]);
            }
            a.dst[(size_t) slot*a.nrows + row0 + i] = result;
        }
    }
}

static const void * llama_func(const bool G) {
    return G ? (const void *) mmvq_llama<true> : (const void *) mmvq_llama<false>;
}

// ---- Miner Llama's Pascal kernel -----------------------------------------------------------------------------------

#define Q2P_NW 4          // warps per block
#define Q2P_C  2          // blocks per lane and iteration
#define Q2P_WB 36         // weight bytes per lane and iteration
#define Q2P_NA 4          // Q8_1 activation blocks per lane and iteration
#define Q2P_YB 144        // activation bytes per lane and iteration
#define Q2P_TILE_MAX (48 * 1024)

enum { Q2P_DIRECT = 0, Q2P_SEG = 1, Q2P_TILE = 2 };

struct pascal_args {
    const uint8_t    * x;
    const uint8_t    * gate;
    const block_q8_1 * y;
    const int32_t    * ids;
    float            * dst;
    int                nrows;
    int                row_bytes;
    int                lpr_log2;      // lanes per row = 1 << lpr_log2
    int                niter;         // iterations per lane
    int                y_slot_blocks;
    int64_t            expert_bytes;
};

static __device__ __forceinline__ float h2f_lo(const uint32_t w) {
    return __half2float(__ushort_as_half((unsigned short) (w & 0xffff)));
}

static __device__ __forceinline__ uint32_t word_at(const uint32_t * w, const int off) {
    return (off & 3) == 0 ? w[off >> 2] : __byte_perm(w[off >> 2], w[(off >> 2) + 1], 0x5432);
}

static __device__ __forceinline__ uint32_t u16_at(const uint32_t * w, const int off) {
    return (off & 3) == 0 ? (w[off >> 2] & 0xffff) : (w[off >> 2] >> 16);
}

// mmvqp_act<4> for Q2_0: the int8 activations of each Q8_1 block as the even and odd bytes of each pair of words
struct act4 {
    int   u[Q2P_NA][8];
    float d[Q2P_NA];
};

static __device__ __forceinline__ void load_act(act4 & a, const char * p) {
    uint32_t w[Q2P_YB / 4];
#pragma unroll
    for (int i = 0; i < Q2P_YB / 16; ++i) {
        const uint4 v = __ldg((const uint4 *) p + i);
        w[4*i + 0] = v.x; w[4*i + 1] = v.y; w[4*i + 2] = v.z; w[4*i + 3] = v.w;
    }
#pragma unroll
    for (int b = 0; b < Q2P_NA; ++b) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            a.u[b][2*k + 0] = (int) __byte_perm(w[9*b + 1 + 2*k], w[9*b + 2 + 2*k], 0x6420);
            a.u[b][2*k + 1] = (int) __byte_perm(w[9*b + 1 + 2*k], w[9*b + 2 + 2*k], 0x7531);
        }
        a.d[b] = h2f_lo(w[9*b]);
    }
}

// mmvqp_q2_0_isum
static __device__ __forceinline__ int q2_0_isum(const uint32_t q, const int * u, int isum) {
    constexpr uint32_t t = 0x020100FF; // codes 0, 1, 2, 3 -> -1, 0, 1, 2
    isum = __dp4a((int) __byte_perm(t, t, q),       u[0], isum);
    isum = __dp4a((int) __byte_perm(t, t, q >> 2),  u[1], isum);
    isum = __dp4a((int) __byte_perm(t, t, q >> 16), u[2], isum);
    isum = __dp4a((int) __byte_perm(t, t, q >> 18), u[3], isum);
    return isum;
}

// mmvqp_dot<GGML_TYPE_Q2_0, 2>
static __device__ __forceinline__ float dot2(const uint32_t * w, const act4 & a, float acc) {
#pragma unroll
    for (int b = 0; b < Q2P_C; ++b) {
        const int o = b * Q2MV_BS;
        const float d  = h2f_lo(u16_at(w, o));
        const int   i0 = q2_0_isum(word_at(w, o + 6),  a.u[2*b + 0] + 4, q2_0_isum(word_at(w, o + 2),  a.u[2*b + 0], 0));
        const int   i1 = q2_0_isum(word_at(w, o + 14), a.u[2*b + 1] + 4, q2_0_isum(word_at(w, o + 10), a.u[2*b + 1], 0));
        acc = fmaf(d, fmaf((float) i1, a.d[2*b + 1], (float) i0 * a.d[2*b + 0]), acc);
    }
    return acc;
}

// mmvqp_stage_tile
template <int R>
static __device__ __forceinline__ const char * stage_tile(const char * x, const pascal_args & p) {
    extern __shared__ uint4 q2p_tile_sm[];
    constexpr int UN = 4;
    const int wrows  = (32 >> p.lpr_log2) * R;
    const int wrow0  = (int) (blockIdx.x * Q2P_NW + threadIdx.y) * wrows;
    const int wbytes = wrows * p.row_bytes;
    uint4       * dsm = (uint4 *) ((char *) q2p_tile_sm + threadIdx.y * wbytes);
    const uint4 * src = (const uint4 *) (x + (int64_t) wrow0 * p.row_bytes);
    const int     nu  = (max(0, min(wrows, p.nrows - wrow0)) * p.row_bytes) >> 4;
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

// mmvqp_rows_1t<GGML_TYPE_Q2_0, R, 2, staged, fk, 16, tile> for MUL_MAT_ID (ids) and MUL_MAT, gate or plain
template <int R, int MODE, bool G>
__launch_bounds__(Q2P_NW*32, 1)
static __global__ void mmvq_pascal(const pascal_args p) {
    constexpr int M  = G ? 2 : 1;
    [[maybe_unused]] constexpr int NU = (Q2P_WB + 15) / 16; // segment staging: 16-byte units per lane and row
    static_assert(MODE != Q2P_TILE || !G, "tile staging: one matrix");
    __shared__ uint4 sm[MODE == Q2P_SEG ? Q2P_NW : 1][MODE == Q2P_SEG ? 32 * R * M * Q2P_WB / 16 : 1];

    const int lane     = threadIdx.x;
    const int lpr_log2 = p.lpr_log2;
    const int lpr      = 1 << lpr_log2;
    const int sl       = lane & (lpr - 1);
    const int grp      = lane >> lpr_log2;
    const int row0     = (((blockIdx.x * Q2P_NW + threadIdx.y) << (5 - lpr_log2)) + grp) * R;
    const int slot     = blockIdx.y;
    const int e        = p.ids ? p.ids[slot] : 0;

    const char * xm[M];
    xm[0] = (const char *) p.x + e * p.expert_bytes;
    if constexpr (G) {
        xm[1] = (const char *) p.gate + e * p.expert_bytes;
    }
    const char * rw[M][R];
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            rw[m][i] = xm[m] + (int64_t) min(row0 + i, p.nrows - 1) * p.row_bytes + (MODE == Q2P_DIRECT ? sl * Q2P_WB : 0);
        }
    }
    const char * yp = (const char *) (p.y + (size_t) slot * p.y_slot_blocks) + sl * Q2P_YB;
    const int wstep = Q2P_WB << lpr_log2;
    const int ystep = Q2P_YB << lpr_log2;
    [[maybe_unused]] char * smw = (char *) sm[MODE == Q2P_SEG ? threadIdx.y : 0] +
        (MODE == Q2P_SEG ? grp * R * M * Q2P_WB << lpr_log2 : 0);
    [[maybe_unused]] const int seg_units = (Q2P_WB << lpr_log2) >> 4;
    [[maybe_unused]] const char * smt = nullptr;
    if constexpr (MODE == Q2P_TILE) {
        smt = stage_tile<R>(xm[0], p) + grp * R * p.row_bytes + sl * Q2P_WB;
    }

    float acc[M][R];
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            acc[m][i] = 0.0f;
        }
    }

    for (int it = 0; it < p.niter; ++it) {
        if constexpr (MODE == Q2P_SEG) {
#pragma unroll
            for (int m = 0; m < M; ++m) {
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    uint4       * dsm = (uint4 *) (smw + ((m*R + i) * Q2P_WB << lpr_log2));
                    const uint4 * src = (const uint4 *) rw[m][i];
#pragma unroll
                    for (int j = 0; j < NU; ++j) {
                        const int u = sl + (j << lpr_log2);
                        if (j < NU - 1 || u < seg_units) {
                            dsm[u] = __ldg(src + u);
                        }
                    }
                }
            }
            __syncwarp();
        }
        act4 a;
        load_act(a, yp);
#pragma unroll
        for (int m = 0; m < M; ++m) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                uint32_t w[Q2P_WB / 4];
                if constexpr (MODE == Q2P_TILE) {
                    const uint32_t * s = (const uint32_t *) (smt + i * p.row_bytes);
#pragma unroll
                    for (int k = 0; k < Q2P_WB / 4; ++k) {
                        w[k] = s[k];
                    }
                } else if constexpr (MODE == Q2P_SEG) {
                    const uint32_t * s = (const uint32_t *) (smw + ((m*R + i) * Q2P_WB << lpr_log2) + sl * Q2P_WB);
#pragma unroll
                    for (int k = 0; k < Q2P_WB / 4; ++k) {
                        w[k] = s[k];
                    }
                } else {
#pragma unroll
                    for (int k = 0; k < Q2P_WB / 4; ++k) {
                        w[k] = __ldg((const unsigned int *) rw[m][i] + k);
                    }
                }
                acc[m][i] = dot2(w, a, acc[m][i]);
                rw[m][i] += wstep;
            }
        }
        if constexpr (MODE == Q2P_SEG) {
            __syncwarp();
        }
        if constexpr (MODE == Q2P_TILE) {
            smt += wstep;
        }
        yp += ystep;
    }

#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            if (off < lpr) {
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    acc[m][i] += __shfl_xor_sync(0xFFFFFFFF, acc[m][i], off, 32);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const int row = row0 + i;
        if ((i & (lpr - 1)) != sl || row >= p.nrows) {
            continue;
        }
        float result = acc[0][i];
        if constexpr (G) {
            result *= silu(acc[M - 1][i]);
        }
        p.dst[(size_t) slot * p.nrows + row] = result;
    }
}

static const void * pascal_func(const int R, const int mode, const bool G) {
    if (G) {
        if (mode == Q2P_TILE) return nullptr;
        if (R == 1) return mode == Q2P_SEG ? (const void *) mmvq_pascal<1, Q2P_SEG, true> : (const void *) mmvq_pascal<1, Q2P_DIRECT, true>;
        if (R == 2) return mode == Q2P_SEG ? (const void *) mmvq_pascal<2, Q2P_SEG, true> : (const void *) mmvq_pascal<2, Q2P_DIRECT, true>;
        return nullptr;
    }
    switch (mode) {
        case Q2P_TILE:   return R == 1 ? (const void *) mmvq_pascal<1, Q2P_TILE, false>   : R == 2 ? (const void *) mmvq_pascal<2, Q2P_TILE, false>   : nullptr;
        case Q2P_SEG:    return R == 1 ? (const void *) mmvq_pascal<1, Q2P_SEG, false>    : R == 2 ? (const void *) mmvq_pascal<2, Q2P_SEG, false>    : nullptr;
        default:         return R == 1 ? (const void *) mmvq_pascal<1, Q2P_DIRECT, false> : R == 2 ? (const void *) mmvq_pascal<2, Q2P_DIRECT, false> : nullptr;
    }
}

// The launch geometry ggml_cuda_mmvq_pascal computes for Q2_0 (contiguous rows, 16-byte aligned matrices): lanes per row,
// iterations, whether segment and tile staging are possible, and the tile's dynamic shared memory.
struct q2p_geom {
    int  lpr_log2 = 0;
    int  niter    = 0;
    bool seg_ok   = false;
    int  tile_smem = 0; // 0: tile staging not possible at this R
};

static bool q2p_geometry(const int K, const int N, const int R, q2p_geom & g) {
    const int blocks = K / Q2MV_QK;
    if (K % Q2MV_QK != 0 || blocks % Q2P_C != 0) {
        return false;
    }
    const int chunks = blocks / Q2P_C;
    g.lpr_log2 = 0;
    while (g.lpr_log2 < 5 && chunks % (2 << g.lpr_log2) == 0) {
        g.lpr_log2++;
    }
    g.niter = chunks >> g.lpr_log2;
    const int64_t row_bytes = (int64_t) blocks * Q2MV_BS;
    g.seg_ok = (Q2P_WB << g.lpr_log2) % 16 == 0 && row_bytes % 16 == 0;
    const int64_t wbytes = (int64_t) (32 >> g.lpr_log2) * R * row_bytes;
    g.tile_smem = wbytes % 16 == 0 && ((int64_t) N * row_bytes) % 16 == 0 && Q2P_NW * wbytes <= Q2P_TILE_MAX ?
        (int) (Q2P_NW * wbytes) : 0;
    return true;
}

static unsigned q2p_blocks(const int N, const int lpr_log2, const int R);

// mmvqp_choose(GGML_TYPE_Q2_0, gate) and the host's choices in ggml_cuda_mmvq_pascal: gate R = 2, or R = 1 where R = 2
// leaves fewer than 4 blocks per SM over the grid's nmats matrices (experts), staged by segment where it can, else
// direct; plain R = 1, tile-staged where segments can't be staged, else by segment, else direct
static void q2p_pick(const int K, const int N, const bool G, const int nmats, const int nsm, int & R, int & mode) {
    R = G ? 2 : 1;
    q2p_geom g;
    q2p_geometry(K, N, R, g);
    if (G && (int) q2p_blocks(N, g.lpr_log2, 2) * nmats < 4 * nsm) {
        R = 1;
        q2p_geometry(K, N, R, g);
    }
    if (!G && !g.seg_ok && g.tile_smem > 0) {
        mode = Q2P_TILE;
    } else {
        mode = g.seg_ok ? Q2P_SEG : Q2P_DIRECT;
    }
}

static unsigned q2p_blocks(const int N, const int lpr_log2, const int R) {
    const int rows_per_block = Q2P_NW * (32 >> lpr_log2) * R;
    return (unsigned) ((N + rows_per_block - 1) / rows_per_block);
}

} // namespace q2mv
} // namespace mb
