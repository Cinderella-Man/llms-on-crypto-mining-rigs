// persistent_pipe: what would a token-persistent pipeline across a layer split's cards save over the host-driven loop?
//
// N cards (2-16) each hold a stand-in for one card's share of a token: L matrix-vector products over a dummy FP16
// weight matrix (n x n, 210 MB at n = 10,240), each y = x / 2 + tanh(W x) / 2 on a residual of n values per token
// (n = 10,240 is Flash-Next's four hyper-connection streams of 2,560: 40,960 bytes in F32, 10,880 in Q8_0). Card 0
// starts every pass from an embedding of the token ids (a hash of id, pass and position), the last card picks the next
// ids by argmax, and every pass depends on the one before it, as decoding does. The same work runs two ways:
//   - driven by the host, as llama.cpp's layer split runs it: a kernel launch per product per card, the residual
//     handed on through the host (two copies with a host synchronize at every hop; one peer copy that the driver stages
//     through the host, ordered by events; or the llama.cpp fork's manual staged hop in chunks), the token ids read back
//     by the host after the last card and copied to card 0 with the next pass;
//   - token-persistent: one kernel per card, launched once for all passes, that polls its input mailbox in mapped
//     pinned host memory (a sequence flag per chunk), runs its products behind grid barriers, writes the residual into
//     the next card's mailbox (after its last product, or block by block while that product runs), and on the last card
//     mails the token ids back to card 0. The host only launches the kernels and reads the results.
// Every variant runs the same device functions on the same blocks of work, so every variant's token ids and final
// residual must equal the first variant's bit for bit; the run fails (exit code 2) when they do not, or when a mailbox
// times out. See ../README.md.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#define CUDA_CHECK(x)                                                                                        \
    do {                                                                                                     \
        cudaError_t err_ = (x);                                                                              \
        if (err_ != cudaSuccess) {                                                                           \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(err_));      \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)

namespace {

constexpr int   THREADS       = 256;                      // threads per block, every kernel but the argmax
constexpr int   WARPS         = THREADS / 32;
constexpr int   ROWS_PER_WARP = 4;
constexpr int   VB_ROWS       = WARPS * ROWS_PER_WARP;    // 32 rows a virtual block: one Q8_0 block per column
constexpr int   Q8_BYTES      = 34;                       // ggml's block_q8_0: an FP16 scale, 32 int8 values
constexpr int   MAX_CARDS     = 16;
constexpr int   MAX_CHUNKS    = 16;
constexpr int   NMAX          = 16384;
constexpr int   ROW_SHIFT     = 1031;  // product s of a card reads row (r + s * ROW_SHIFT) mod n of its weights
constexpr float W_SCALE       = 4.5f;  // weights uniform in [-W_SCALE, W_SCALE) / sqrt(n): tanh's argument has std ~1.5

enum wire_type : int { WIRE_F32 = 0, WIRE_Q8 = 1 };

// A card's input mailbox in mapped pinned host memory; the data follows the header. The sender raises `hint` to the
// pass number when it starts its last product (the receiver polls slowly until then, then fast), and flag[c] when
// chunk c of that pass is in the data. Sequence numbers only grow, so nothing is ever reset during a run.
struct mail_hdr {
    unsigned hint;
    unsigned pad0[15];
    unsigned flag[MAX_CHUNKS];
    unsigned pad1[64 - 16 - MAX_CHUNKS];
};
constexpr size_t MAIL_HDR = sizeof(mail_hdr);
static_assert(MAIL_HDR == 256, "mailbox header");

// ---- values ---------------------------------------------------------------------------------------------------------

__host__ __device__ inline uint32_t mix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

// uniform in [-1, 1), exact in float
__host__ __device__ inline float hash_unit(uint32_t a, uint32_t b, uint32_t c) {
    const uint32_t h = mix32(mix32(mix32(a ^ 0x9e3779b9u) + b) ^ (c * 0x85ebca6bu));
    return (float) (h >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

__device__ __forceinline__ float embed(int tok, int pass, int j) {
    return hash_unit(0x51ed0000u + (uint32_t) tok, (uint32_t) pass, (uint32_t) j);
}

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void spin(long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {
    }
}

// ---- the work, shared by every variant ------------------------------------------------------------------------------

// One virtual block of a product: rows [32 vb, 32 vb + 32) of y = x / 2 + tanh(W' x) / 2 for each of the K columns of
// x (K x n, column-major), W' the card's weights with the rows shifted by s * ROW_SHIFT. Each warp computes 4 rows,
// each lane a fixed share of every row, reduced in a fixed order; the 32 x K results land in sh (32 a column).
template <int K>
__device__ __forceinline__ void product_vb(const __half * __restrict__ W, const float * x, const int n, const int s,
                                           const int vb, float * sh) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r0   = vb * VB_ROWS + warp * ROWS_PER_WARP;
    const uint4 * wr[ROWS_PER_WARP];
#pragma unroll
    for (int rr = 0; rr < ROWS_PER_WARP; ++rr) {
        const int row = (int) (((long long) r0 + rr + (long long) s * ROW_SHIFT) % n);
        wr[rr] = (const uint4 *) (W + (size_t) row * n);
    }
    float acc[ROWS_PER_WARP][K];
#pragma unroll
    for (int rr = 0; rr < ROWS_PER_WARP; ++rr) {
#pragma unroll
        for (int c = 0; c < K; ++c) {
            acc[rr][c] = 0.0f;
        }
    }
    const int n8 = n / 8;
#pragma unroll 2
    for (int i = lane; i < n8; i += 32) {
        float w[ROWS_PER_WARP][8];
#pragma unroll
        for (int rr = 0; rr < ROWS_PER_WARP; ++rr) {
            const uint4 v = __ldg(wr[rr] + i);
            const __half2 * h = (const __half2 *) &v;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float2 f = __half22float2(h[q]);
                w[rr][2 * q]     = f.x;
                w[rr][2 * q + 1] = f.y;
            }
        }
#pragma unroll
        for (int c = 0; c < K; ++c) {
            const float4 * xc = (const float4 *) (x + (size_t) c * n) + 2 * i;
            const float4 a = __ldcg(xc);
            const float4 b = __ldcg(xc + 1);
#pragma unroll
            for (int rr = 0; rr < ROWS_PER_WARP; ++rr) {
                float t = acc[rr][c];
                t = fmaf(w[rr][0], a.x, t);
                t = fmaf(w[rr][1], a.y, t);
                t = fmaf(w[rr][2], a.z, t);
                t = fmaf(w[rr][3], a.w, t);
                t = fmaf(w[rr][4], b.x, t);
                t = fmaf(w[rr][5], b.y, t);
                t = fmaf(w[rr][6], b.z, t);
                t = fmaf(w[rr][7], b.w, t);
                acc[rr][c] = t;
            }
        }
    }
#pragma unroll
    for (int rr = 0; rr < ROWS_PER_WARP; ++rr) {
#pragma unroll
        for (int c = 0; c < K; ++c) {
            float v = acc[rr][c];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, off);
            }
            if (lane == 0) {
                const float xr = __ldcg(x + (size_t) c * n + r0 + rr);
                sh[c * VB_ROWS + warp * ROWS_PER_WARP + rr] = fmaf(0.5f, xr, 0.5f * tanhf(v));
            }
        }
    }
    __syncthreads();
}

template <int K> __device__ __forceinline__ void store_vb(const float * sh, float * y, const int n, const int vb) {
    if (threadIdx.x < K * VB_ROWS) {
        const int c = threadIdx.x / VB_ROWS, i = threadIdx.x % VB_ROWS;
        y[(size_t) c * n + vb * VB_ROWS + i] = sh[c * VB_ROWS + i];
    }
}

template <int K> __device__ __forceinline__ void load_vb(const float * y, float * sh, const int n, const int vb) {
    if (threadIdx.x < K * VB_ROWS) {
        const int c = threadIdx.x / VB_ROWS, i = threadIdx.x % VB_ROWS;
        sh[c * VB_ROWS + i] = __ldcg(y + (size_t) c * n + vb * VB_ROWS + i);
    }
}

// The 32 x K values of virtual block vb, from sh, onto the wire (device or mapped host memory): F32 as they are
// (column-major, as y), or Q8_0 (column c's row of n / 32 blocks, ggml's block_q8_0 and its rounding), each block's
// 34 bytes written by one store of 17 lanes.
template <int K>
__device__ __forceinline__ void emit_vb(const float * sh, unsigned char * wire, const int type, const int n, const int vb) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (warp >= K) {
        return;
    }
    const float v = sh[warp * VB_ROWS + lane];
    if (type == WIRE_F32) {
        ((float *) wire)[(size_t) warp * n + vb * VB_ROWS + lane] = v;
        return;
    }
    float amax = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    const float d  = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const int   q  = (int) roundf(v * id) & 0xff;
    const int   lo = __shfl_sync(0xffffffffu, q, (2 * lane - 2) & 31);
    const int   hi = __shfl_sync(0xffffffffu, q, (2 * lane - 1) & 31);
    if (lane < Q8_BYTES / 2) {
        const unsigned short u = lane == 0 ? __half_as_ushort(__float2half(d)) : (unsigned short) (lo | (hi << 8));
        ((unsigned short *) (wire + ((size_t) warp * (n / VB_ROWS) + vb) * Q8_BYTES))[lane] = u;
    }
}

// Warp-wide: the 32 values of column c, virtual block vb, from the wire into x. CV: read with ld.global.cv (mapped
// host memory that another card writes while this kernel runs).
template <bool CV>
__device__ __forceinline__ void take_vb(const unsigned char * wire, const int type, float * x, const int n, const int c,
                                        const int vb, const int lane) {
    const size_t j = (size_t) c * n + vb * VB_ROWS + lane;
    if (type == WIRE_F32) {
        const float * p = (const float *) wire + j;
        x[j] = CV ? __ldcv(p) : *p;
        return;
    }
    const unsigned char * blk = wire + ((size_t) c * (n / VB_ROWS) + vb) * Q8_BYTES;
    const unsigned short dbits = CV ? __ldcv((const unsigned short *) blk) : *(const unsigned short *) blk;
    const signed char    q     = CV ? __ldcv((const signed char *) blk + 2 + lane) : ((const signed char *) blk)[2 + lane];
    x[j] = __half2float(__ushort_as_half(dbits)) * (float) q;
}

// the larger value wins, the lower index on a tie: exact, so any reduction order gives the same answer
__device__ __forceinline__ void better(float & bv, int & bi, const float v, const int i) {
    if (v > bv || (v == bv && i < bi)) {
        bv = v;
        bi = i;
    }
}

// block-wide argmax; the result in every thread of warp 0
__device__ __forceinline__ void block_argmax(float & bv, int & bi) {
    __shared__ float sv[32];
    __shared__ int   si[32];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        better(bv, bi, __shfl_xor_sync(0xffffffffu, bv, off), __shfl_xor_sync(0xffffffffu, bi, off));
    }
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (warp == 0) {
        for (int w = 1; w < nw; ++w) {
            better(bv, bi, sv[w], si[w]);
        }
    }
    __syncthreads();
}

// ---- the host-driven kernels ----------------------------------------------------------------------------------------

__global__ void k_init_weights(__half * W, const int n, const uint32_t seed) {
    const size_t total = (size_t) n * n;
    const float  scale = W_SCALE / sqrtf((float) n);
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < total; i += (size_t) gridDim.x * blockDim.x) {
        W[i] = __float2half(hash_unit(seed, (uint32_t) (i / n), (uint32_t) (i % n)) * scale);
    }
}

__global__ void k_embed(float * x, const int * toks, const int K, const int n, const int pass) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < K * n; idx += gridDim.x * blockDim.x) {
        x[idx] = embed(toks[idx / n], pass, idx % n);
    }
}

template <int K>
__global__ void __launch_bounds__(THREADS) k_product(const __half * W, const float * x, float * y, const int n, const int s) {
    __shared__ float sh[K * VB_ROWS];
    product_vb<K>(W, x, n, s, blockIdx.x, sh);
    store_vb<K>(sh, y, n, blockIdx.x);
}

template <int K>
__global__ void __launch_bounds__(THREADS) k_cast(const float * y, unsigned char * wire, const int type, const int n) {
    __shared__ float sh[K * VB_ROWS];
    load_vb<K>(y, sh, n, blockIdx.x);
    __syncthreads();
    emit_vb<K>(sh, wire, type, n, blockIdx.x);
}

__global__ void k_uncast(const unsigned char * wire, const int type, float * x, const int K, const int n) {
    const int V = n / VB_ROWS, lane = threadIdx.x % 32;
    const int nw = gridDim.x * blockDim.x / 32;
    for (int q = (blockIdx.x * blockDim.x + threadIdx.x) / 32; q < K * V; q += nw) {
        take_vb<false>(wire, type, x, n, q / V, q % V, lane);
    }
}

__global__ void k_argmax(const float * y, const int n, int * toks) {
    float bv = -INFINITY;
    int   bi = INT_MAX;
    const float * col = y + (size_t) blockIdx.x * n;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        better(bv, bi, col[j], j);
    }
    block_argmax(bv, bi);
    if (threadIdx.x == 0) {
        toks[blockIdx.x] = bi;
    }
}

__global__ void k_arch(int * out) {
#ifdef __CUDA_ARCH__
    *out = __CUDA_ARCH__;
#endif
}

// ---- the token-persistent kernel ------------------------------------------------------------------------------------

struct pipe_args {
    int card, ncards, n, layers, passes, wire, chunks, direct, start_tok;
    const __half * W;
    float * buf0;
    float * buf1;
    unsigned char * in_mail;       // this card's mailbox (header, then data), as this card addresses it
    unsigned char * out_mail;      // the next card's; the last card's is card 0's
    int * results;                 // last card: every pass's token ids, in mapped host memory
    unsigned * host_abort;         // mapped host memory: a card's timeout or the host's watchdog stops every card
    unsigned long long * bar;      // device: the grid barrier's counter
    unsigned * state;              // device: [0] abort; [2 + c] chunk c readable (pass); [32 + c] chunk c's blocks out
    float * part_v;                // device: per block and column, the argmax's partial value
    int * part_i;                  //   and index
    int * toks;                    // device: card 0's token ids for the pass
    unsigned long long * stamps;   // device: per pass, the input in, the last product done, the output posted (ns)
    long long fast_cycles;         // clock cycles between two polls of a flag in host memory after the hint
    long long slow_cycles;         //   and before it
    unsigned long long timeout_ns; // a wait longer than this stops the run
};

__device__ __forceinline__ void raise_abort(const pipe_args & a) {
    *(volatile unsigned *) a.state      = 1;
    *(volatile unsigned *) a.host_abort = 1;
    __threadfence_system();
}

// Waits until *f >= target, `spacing` cycles between reads. Returns false on an abort (the host's flag is read only by
// the waits on host memory, `host` true) or after timeout_ns, which raises the abort for every card.
__device__ bool wait_seq(const volatile unsigned * f, const unsigned target, const long long spacing, const bool host,
                         const pipe_args & a) {
    unsigned long long t0 = 0;
    for (unsigned polls = 1; *f < target; ++polls) {
        spin(spacing);
        if ((polls & 255) == 0) {
            if (*(volatile unsigned *) a.state || (host && *(volatile unsigned *) a.host_abort)) {
                *(volatile unsigned *) a.state = 1;
                return false;
            }
            const unsigned long long t = gtime();
            if (t0 == 0) {
                t0 = t;
            } else if (t - t0 > a.timeout_ns) {
                raise_abort(a);
                return false;
            }
        }
    }
    return true;
}

// A grid barrier from one monotonic counter (every block resident: a cooperative launch); false after an abort.
__device__ bool grid_barrier(const pipe_args & a) {
    __shared__ int ok;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned long long ticket = atomicAdd(a.bar, 1ull);
        const unsigned long long target = (ticket / gridDim.x + 1) * gridDim.x;
        int good = 1;
        for (unsigned spins = 1; *(volatile unsigned long long *) a.bar < target; ++spins) {
            if (spins > 1024) {
                spin(32);  // a long wait: the input of the next pass
            }
            if ((spins & 255) == 0 && *(volatile unsigned *) a.state) {
                good = 0;
                break;
            }
        }
        __threadfence();
        ok = good;
    }
    __syncthreads();
    return ok;
}

__device__ __forceinline__ int chunk_begin(const int c, const int V, const int C) { return c * V / C; }

// The block has written `count` virtual blocks of chunk c into the mailbox; the block that completes the chunk for this
// pass raises its flag.
__device__ void post_chunk(const pipe_args & a, volatile mail_hdr * out, const int c, const int count, const unsigned seq,
                           const int V) {
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned per  = chunk_begin(c + 1, V, a.chunks) - chunk_begin(c, V, a.chunks);
        const unsigned done = atomicAdd(a.state + 32 + c, (unsigned) count) + count;
        if (done == seq * per) {
            __threadfence_system();
            out->flag[c] = seq;
        }
    }
}

template <int K> __global__ void __launch_bounds__(THREADS) k_pipe(const pipe_args a) {
    __shared__ float sh[K * VB_ROWS];
    __shared__ int   ok_s;
    const int  n = a.n, V = n / VB_ROWS, C = a.chunks, G = gridDim.x, b = blockIdx.x, t = threadIdx.x;
    const int  warp = t / 32, lane = t % 32;
    const bool first = a.card == 0, last = a.card == a.ncards - 1;
    volatile mail_hdr * in  = (volatile mail_hdr *) a.in_mail;
    volatile mail_hdr * out = (volatile mail_hdr *) a.out_mail;
    const unsigned char * in_data  = a.in_mail + MAIL_HDR;
    unsigned char *       out_data = a.out_mail + MAIL_HDR;
    volatile unsigned *   st       = a.state;

    for (int p = 0; p < a.passes; ++p) {
        const unsigned seq  = p + 1;              // what this card posts for pass p
        const unsigned want = first ? p : p + 1;  // card 0 waits for pass p - 1's token ids, the others for pass p

        // ---- the input
        if (first) {
            if (b == 0 && t == 0) {
                if (p == 0) {
                    for (int c = 0; c < K; ++c) {
                        a.toks[c] = (a.start_tok + c * 7919) % n;
                    }
                } else if (wait_seq(&in->hint, want, a.slow_cycles, true, a) &&
                           wait_seq(&in->flag[0], want, a.fast_cycles, true, a)) {
                    __threadfence_system();
                    for (int c = 0; c < K; ++c) {
                        a.toks[c] = __ldcv((const int *) in_data + c);
                    }
                }
                __threadfence();
            }
            if (!grid_barrier(a)) {
                return;
            }
            for (int idx = b * THREADS + t; idx < K * n; idx += G * THREADS) {
                a.buf0[idx] = embed(__ldcg(a.toks + idx / n), p, idx % n);
            }
        } else {
            // block 0 polls the host's memory, one flag at a time; the readers of the chunks wait on the device
            if (b == 0 && t == 0) {
                bool ok = wait_seq(&in->hint, want, a.slow_cycles, true, a);
                for (int c = 0; ok && c < C; ++c) {
                    ok = wait_seq(&in->flag[c], want, a.fast_cycles, true, a);
                    if (ok) {
                        __threadfence_system();
                        st[2 + c] = want;
                    }
                }
                __threadfence();
            }
            for (int c = 0; c < C; ++c) {
                if ((1 + c) % G != b) {
                    continue;
                }
                if (t == 0) {
                    ok_s = wait_seq(st + 2 + c, want, 64, false, a);
                    __threadfence_system();
                }
                __syncthreads();
                if (!ok_s) {
                    return;
                }
                const int vb0 = chunk_begin(c, V, C), nvb = chunk_begin(c + 1, V, C) - vb0;
                for (int q = warp; q < K * nvb; q += WARPS) {
                    take_vb<true>(in_data, a.wire, a.buf0, n, q / nvb, vb0 + q % nvb, lane);
                }
            }
        }
        if (!grid_barrier(a)) {
            return;
        }
        if (b == 0 && t == 0) {
            a.stamps[3 * p] = gtime();
        }

        // ---- the products
        for (int s = 0; s < a.layers; ++s) {
            const float * x = (s & 1) ? a.buf1 : a.buf0;
            float *       y = (s & 1) ? a.buf0 : a.buf1;
            const bool final_product = s == a.layers - 1;
            if (final_product && b == 0 && t == 0) {
                out->hint = seq;
            }
            const bool to_wire = final_product && a.direct && !last;
            int cur = -1, pending = 0;
            for (int vb = b; vb < V; vb += G) {
                product_vb<K>(a.W, x, n, s, vb, sh);
                if (to_wire) {
                    int c = vb * C / V;
                    while (vb >= chunk_begin(c + 1, V, C)) {
                        ++c;
                    }
                    while (vb < chunk_begin(c, V, C)) {
                        --c;
                    }
                    if (c != cur) {
                        if (pending) {
                            post_chunk(a, out, cur, pending, seq, V);
                        }
                        cur     = c;
                        pending = 0;
                    }
                    emit_vb<K>(sh, out_data, a.wire, n, vb);
                    ++pending;
                } else {
                    store_vb<K>(sh, y, n, vb);
                }
                __syncthreads();
            }
            if (to_wire && pending) {
                post_chunk(a, out, cur, pending, seq, V);
            }
            if (!grid_barrier(a)) {
                return;
            }
        }
        if (b == 0 && t == 0) {
            a.stamps[3 * p + 1] = gtime();
        }
        const float * res = (a.layers & 1) ? a.buf1 : a.buf0;

        // ---- the output
        if (!last) {
            // one block writes the chunks in order, so the next card reads chunk c while chunk c + 1 crosses this
            // card's link
            if (!a.direct && b == 1 % G) {
                for (int c = 0; c < C; ++c) {
                    for (int vb = chunk_begin(c, V, C); vb < chunk_begin(c + 1, V, C); ++vb) {
                        load_vb<K>(res, sh, n, vb);
                        __syncthreads();
                        emit_vb<K>(sh, out_data, a.wire, n, vb);
                        __syncthreads();
                    }
                    __threadfence_system();
                    __syncthreads();
                    if (t == 0) {
                        out->flag[c] = seq;
                    }
                }
            }
        } else {
            for (int c = 0; c < K; ++c) {
                float bv = -INFINITY;
                int   bi = INT_MAX;
                for (int j = b * THREADS + t; j < n; j += G * THREADS) {
                    better(bv, bi, __ldcg(res + (size_t) c * n + j), j);
                }
                block_argmax(bv, bi);
                if (t == 0) {
                    a.part_v[b * K + c] = bv;
                    a.part_i[b * K + c] = bi;
                }
            }
            if (!grid_barrier(a)) {
                return;
            }
            if (b == 0) {
                if (t < K) {
                    float bv = -INFINITY;
                    int   bi = INT_MAX;
                    for (int g = 0; g < G; ++g) {
                        better(bv, bi, __ldcg(a.part_v + g * K + t), __ldcg(a.part_i + g * K + t));
                    }
                    a.results[(size_t) p * K + t] = bi;
                    ((int *) out_data)[t]         = bi;
                    __threadfence_system();
                }
                __syncthreads();
                if (t == 0) {
                    out->flag[0]        = seq;
                    a.stamps[3 * p + 2] = gtime();
                }
            }
        }
        if (!grid_barrier(a)) {
            return;
        }
        if (!last && b == 0 && t == 0) {
            a.stamps[3 * p + 2] = gtime();
        }
    }
}

// ---- host side ------------------------------------------------------------------------------------------------------

struct options {
    std::string devices;       // CUDA devices in pipeline order; empty: 0 .. cards - 1
    int         cards  = 0;    // 0: every visible device
    int         n      = 10240;
    int         k      = 1;
    int         layers = 4;
    double      card_ms = 0;   // > 0: as many products per card as come closest to this
    int         passes = 1000;
    int         warmup = 20;
    std::string wire   = "f32";
    std::string variants = "sync,peer,staged,persist,persist-direct";
    int         staged_chunks = 2;
    int         mail_chunks   = 4;
    int         grid          = 0;   // 0: every block the cards hold at once
    double      poll_us       = 2.0;
    double      slow_poll_us  = 50.0;
    double      timeout_s     = 10.0;
    std::string sched         = "auto";
    unsigned    seed          = 1;
    std::string csv;
    bool        smoke = false;
};

struct card_t {
    int dev = 0;
    cudaDeviceProp prop{};
    int clock_khz = 0;
    cudaStream_t s = nullptr;
    __half * W = nullptr;
    float * buf[2] = { nullptr, nullptr };
    unsigned char * wire_in = nullptr;
    unsigned char * wire_out = nullptr;
    int * toks = nullptr;
    int * toks_out = nullptr;
    // the persistent kernel
    unsigned long long * bar = nullptr;
    unsigned long long * stamps = nullptr;
    unsigned * state = nullptr;
    float * part_v = nullptr;
    int * part_i = nullptr;
    int * ptoks = nullptr;
    unsigned char * mail = nullptr;      // this card's input mailbox, host pointer
    unsigned char * in_dev = nullptr;    // the mailboxes as this card addresses them
    unsigned char * out_dev = nullptr;
    int * results_dev = nullptr;
    unsigned * abort_dev = nullptr;
    int grid = 0;
    // the hop to the next card
    void * stage_host = nullptr;         // sync: the pinned buffer between the two copies
    void * ring = nullptr;               // staged: the pinned slots
    cudaEvent_t ev_done = nullptr;       // peer: the copy is done (this card)
    cudaEvent_t filled[MAX_CHUNKS] = {}; // staged: slot filled (this card), slot drained (the next card)
    cudaEvent_t drained[MAX_CHUNKS] = {};
    double cal_ms = 0;                   // the card's products alone, launched back to back
};

struct run_result {
    std::string name;
    bool ok = false;                     // finished without a timeout
    std::vector<int> toks;               // every pass's ids
    std::vector<float> y;                // the last card's final residual
    std::vector<double> pass_us;         // per timed pass
    std::vector<double> compute_us;      // per timed pass: the sum over the cards of their products' time
    std::vector<std::vector<double>> card_us;      // persistent: per card, per timed pass
    std::vector<std::vector<double>> boundary_us;  // persistent: per boundary (the last one: back to card 0)
    double wall_s = 0;
};

options opt;
std::vector<card_t> cards;
int K = 1, N_ = 0, V_ = 0, L_ = 0, WIRE = WIRE_F32;
size_t wire_bytes = 0;
int * results_host = nullptr;
unsigned * abort_host = nullptr;
int * htoks = nullptr;
FILE * g_csv = nullptr;

using clk = std::chrono::steady_clock;
double us_between(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n == 0 ? 0.0 : (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
}

double pctl(std::vector<double> v, double p) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t) (p * (v.size() - 1) + 0.5))];
}

double mean(const std::vector<double> & v) {
    double s = 0;
    for (double x : v) {
        s += x;
    }
    return v.empty() ? 0.0 : s / v.size();
}

std::vector<int> parse_ints(const std::string & s) {
    std::vector<int> out;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t end = s.find(',', pos);
        if (end == std::string::npos) {
            end = s.size();
        }
        if (end > pos) {
            out.push_back(atoi(s.substr(pos, end - pos).c_str()));
        }
        pos = end + 1;
    }
    return out;
}

std::vector<std::string> parse_list(const std::string & s) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t end = s.find(',', pos);
        if (end == std::string::npos) {
            end = s.size();
        }
        if (end > pos) {
            out.push_back(s.substr(pos, end - pos));
        }
        pos = end + 1;
    }
    return out;
}

void csv_row(const std::string & variant, const std::string & key, double value, const char * unit) {
    if (g_csv) {
        fprintf(g_csv, "%s,%d,%d,%s,%d,%s,%.4f,%s\n", variant.c_str(), N_, K, opt.wire.c_str(), L_, key.c_str(), value,
                unit);
        fflush(g_csv);
    }
}

template <typename F> void with_k(int k, F && f) {
    switch (k) {
        case 1: f(std::integral_constant<int, 1>{}); break;
        case 2: f(std::integral_constant<int, 2>{}); break;
        case 4: f(std::integral_constant<int, 4>{}); break;
        case 8: f(std::integral_constant<int, 8>{}); break;
        default: fprintf(stderr, "--k must be 1, 2, 4 or 8\n"); exit(1);
    }
}

int start_tok(int c) { return (int) ((opt.seed * 2654435761u) % (unsigned) opt.n + (unsigned) c * 7919u) % opt.n; }

void launch_product(card_t & c, int s) {
    with_k(K, [&](auto kc) {
        constexpr int KK = decltype(kc)::value;
        k_product<KK><<<V_, THREADS, 0, c.s>>>(c.W, c.buf[s & 1], c.buf[(s + 1) & 1], opt.n, s);
    });
}

void launch_cast(card_t & c, const float * y, unsigned char * wire) {
    with_k(K, [&](auto kc) {
        constexpr int KK = decltype(kc)::value;
        k_cast<KK><<<V_, THREADS, 0, c.s>>>(y, wire, WIRE, opt.n);
    });
}

// ms of the card's L products launched back to back, median of `reps`
double calibrate(card_t & c, int layers, int reps) {
    CUDA_CHECK(cudaSetDevice(c.dev));
    CUDA_CHECK(cudaMemsetAsync(c.toks, 0, K * sizeof(int), c.s));
    k_embed<<<64, THREADS, 0, c.s>>>(c.buf[0], c.toks, K, opt.n, 0);
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    std::vector<double> t;
    for (int r = 0; r < reps + 2; ++r) {
        CUDA_CHECK(cudaEventRecord(e0, c.s));
        for (int s = 0; s < layers; ++s) {
            launch_product(c, s);
        }
        CUDA_CHECK(cudaEventRecord(e1, c.s));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (r >= 2) {
            t.push_back(ms);
        }
    }
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return median(t);
}

// the residual (or its Q8_0 cast) from card i to card i + 1, the way variant v hands it on
void hop(const std::string & v, int i, const void * src, void * dst, size_t bytes) {
    card_t & a = cards[i];
    card_t & b = cards[i + 1];
    if (v == "sync") {
        CUDA_CHECK(cudaSetDevice(a.dev));
        CUDA_CHECK(cudaMemcpyAsync(a.stage_host, src, bytes, cudaMemcpyDeviceToHost, a.s));
        CUDA_CHECK(cudaStreamSynchronize(a.s));
        CUDA_CHECK(cudaSetDevice(b.dev));
        CUDA_CHECK(cudaMemcpyAsync(dst, a.stage_host, bytes, cudaMemcpyHostToDevice, b.s));
    } else if (v == "peer") {
        CUDA_CHECK(cudaSetDevice(a.dev));
        CUDA_CHECK(cudaMemcpyPeerAsync(dst, b.dev, src, a.dev, bytes, a.s));
        CUDA_CHECK(cudaEventRecord(a.ev_done, a.s));
        CUDA_CHECK(cudaSetDevice(b.dev));
        CUDA_CHECK(cudaStreamWaitEvent(b.s, a.ev_done, 0));
    } else {
        // the fork's GGML_CUDA_HOP_STAGED: chunk k through pinned slot k, its D2H on card a's stream, an event, its H2D
        // on card b's stream behind it; a slot's next D2H waits for its last H2D
        const size_t chunk = ((bytes + opt.staged_chunks - 1) / opt.staged_chunks + 255) / 256 * 256;
        for (size_t k = 0, off = 0; off < bytes; ++k, off += chunk) {
            const size_t len = std::min(chunk, bytes - off);
            CUDA_CHECK(cudaSetDevice(a.dev));
            CUDA_CHECK(cudaStreamWaitEvent(a.s, a.drained[k], 0));
            CUDA_CHECK(cudaMemcpyAsync((char *) a.ring + off, (const char *) src + off, len, cudaMemcpyDeviceToHost, a.s));
            CUDA_CHECK(cudaEventRecord(a.filled[k], a.s));
            CUDA_CHECK(cudaSetDevice(b.dev));
            CUDA_CHECK(cudaStreamWaitEvent(b.s, a.filled[k], 0));
            CUDA_CHECK(cudaMemcpyAsync((char *) dst + off, (const char *) a.ring + off, len, cudaMemcpyHostToDevice, b.s));
            CUDA_CHECK(cudaEventRecord(a.drained[k], b.s));
        }
    }
}

void read_final(run_result & r) {
    card_t & c = cards.back();
    r.y.resize((size_t) K * opt.n);
    CUDA_CHECK(cudaSetDevice(c.dev));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(r.y.data(), c.buf[L_ & 1], r.y.size() * sizeof(float), cudaMemcpyDeviceToHost));
}

run_result run_host(const std::string & v) {
    const int P = opt.warmup + opt.passes;
    run_result r;
    r.name = v;
    r.toks.resize((size_t) P * K);
    double compute = 0;
    for (const card_t & c : cards) {
        compute += c.cal_ms * 1000.0;
    }
    for (int c = 0; c < K; ++c) {
        htoks[c] = start_tok(c);
    }
    const int embed_blocks = std::min(256, (K * opt.n + THREADS - 1) / THREADS);
    const clk::time_point t0 = clk::now();
    clk::time_point prev = t0;
    for (int p = 0; p < P; ++p) {
        card_t & c0 = cards[0];
        CUDA_CHECK(cudaSetDevice(c0.dev));
        CUDA_CHECK(cudaMemcpyAsync(c0.toks, htoks, K * sizeof(int), cudaMemcpyHostToDevice, c0.s));
        k_embed<<<embed_blocks, THREADS, 0, c0.s>>>(c0.buf[0], c0.toks, K, opt.n, p);
        for (int i = 0; i < N_; ++i) {
            card_t & c = cards[i];
            CUDA_CHECK(cudaSetDevice(c.dev));
            if (i > 0 && WIRE == WIRE_Q8) {
                k_uncast<<<V_ * K / WARPS + 1, THREADS, 0, c.s>>>(c.wire_in, WIRE, c.buf[0], K, opt.n);
            }
            for (int s = 0; s < L_; ++s) {
                launch_product(c, s);
            }
            float * res = c.buf[L_ & 1];
            if (i + 1 < N_) {
                const void * src = res;
                void *       dst = cards[i + 1].buf[0];
                if (WIRE == WIRE_Q8) {
                    launch_cast(c, res, c.wire_out);
                    src = c.wire_out;
                    dst = cards[i + 1].wire_in;
                }
                hop(v, i, src, dst, wire_bytes);
            } else {
                k_argmax<<<K, 1024, 0, c.s>>>(res, opt.n, c.toks_out);
                CUDA_CHECK(cudaMemcpyAsync(htoks, c.toks_out, K * sizeof(int), cudaMemcpyDeviceToHost, c.s));
                CUDA_CHECK(cudaStreamSynchronize(c.s));
            }
        }
        const clk::time_point now = clk::now();
        if (p >= std::max(1, opt.warmup)) {
            r.pass_us.push_back(us_between(prev, now));
            r.compute_us.push_back(compute);
        }
        prev = now;
        for (int c = 0; c < K; ++c) {
            r.toks[(size_t) p * K + c] = htoks[c];
        }
    }
    CUDA_CHECK(cudaGetLastError());
    r.wall_s = us_between(t0, clk::now()) / 1e6;
    r.ok     = true;
    read_final(r);
    return r;
}

run_result run_persistent(const std::string & v, bool direct) {
    const int P = opt.warmup + opt.passes;
    run_result r;
    r.name = v;
    r.toks.resize((size_t) P * K);
    for (card_t & c : cards) {
        memset(c.mail, 0, MAIL_HDR);
        CUDA_CHECK(cudaSetDevice(c.dev));
        CUDA_CHECK(cudaMemset(c.bar, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(c.state, 0, 64 * sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(c.stamps, 0, (size_t) P * 3 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    memset(results_host, 0xff, (size_t) P * K * sizeof(int));
    *(volatile unsigned *) abort_host = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);

    double expect_s = 0;
    for (const card_t & c : cards) {
        expect_s += c.cal_ms / 1000.0 + 0.002;
    }
    const double limit_s = expect_s * P * 3 + opt.timeout_s;

    const clk::time_point t0 = clk::now();
    for (int i = 0; i < N_; ++i) {
        card_t & c = cards[i];
        pipe_args a{};
        a.card        = i;
        a.ncards      = N_;
        a.n           = opt.n;
        a.layers      = L_;
        a.passes      = P;
        a.wire        = WIRE;
        a.chunks      = opt.mail_chunks;
        a.direct      = direct ? 1 : 0;
        a.start_tok   = start_tok(0);
        a.W           = c.W;
        a.buf0        = c.buf[0];
        a.buf1        = c.buf[1];
        a.in_mail     = c.in_dev;
        a.out_mail    = c.out_dev;
        a.results     = c.results_dev;
        a.host_abort  = c.abort_dev;
        a.bar         = c.bar;
        a.state       = c.state;
        a.part_v      = c.part_v;
        a.part_i      = c.part_i;
        a.toks        = c.ptoks;
        a.stamps      = c.stamps;
        a.fast_cycles = (long long) (opt.poll_us * c.clock_khz / 1000.0);
        a.slow_cycles = (long long) (opt.slow_poll_us * c.clock_khz / 1000.0);
        a.timeout_ns  = (unsigned long long) (opt.timeout_s * 1e9);
        void * params[] = { &a };
        CUDA_CHECK(cudaSetDevice(c.dev));
        with_k(K, [&](auto kc) {
            constexpr int KK = decltype(kc)::value;
            CUDA_CHECK(cudaLaunchCooperativeKernel((const void *) k_pipe<KK>, dim3(c.grid), dim3(THREADS), params, 0, c.s));
        });
    }
    // the host's part: wait, and stop every card if the run takes far longer than it should
    bool watchdog = false;
    for (;;) {
        bool done = true;
        for (card_t & c : cards) {
            CUDA_CHECK(cudaSetDevice(c.dev));
            const cudaError_t q = cudaStreamQuery(c.s);
            if (q == cudaErrorNotReady) {
                done = false;
            } else if (q != cudaSuccess) {
                CUDA_CHECK(q);
            }
        }
        if (done) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (!watchdog && us_between(t0, clk::now()) / 1e6 > limit_s) {
            *(volatile unsigned *) abort_host = 1;
            watchdog = true;
        }
    }
    r.wall_s = us_between(t0, clk::now()) / 1e6;
    for (card_t & c : cards) {
        CUDA_CHECK(cudaSetDevice(c.dev));
        CUDA_CHECK(cudaGetLastError());
    }
    r.ok = *(volatile unsigned *) abort_host == 0;
    if (!r.ok) {
        printf("  %s: %s\n", v.c_str(), watchdog ? "stopped by the host's watchdog" : "a mailbox wait timed out");
    }
    for (size_t i = 0; i < r.toks.size(); ++i) {
        r.toks[i] = ((volatile int *) results_host)[i];
    }
    std::vector<std::vector<unsigned long long>> st(N_, std::vector<unsigned long long>((size_t) P * 3));
    for (int i = 0; i < N_; ++i) {
        CUDA_CHECK(cudaSetDevice(cards[i].dev));
        CUDA_CHECK(cudaMemcpy(st[i].data(), cards[i].stamps, st[i].size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
    }
    r.card_us.assign(N_, {});
    r.boundary_us.assign(N_, {});
    auto d_us = [](unsigned long long a, unsigned long long b) { return ((double) b - (double) a) / 1000.0; };
    for (int p = std::max(1, opt.warmup); p < P && r.ok; ++p) {
        r.pass_us.push_back(d_us(st[N_ - 1][3 * (p - 1) + 2], st[N_ - 1][3 * p + 2]));
        double compute = 0;
        for (int i = 0; i < N_; ++i) {
            const double cu = d_us(st[i][3 * p], st[i][3 * p + 1]);
            r.card_us[i].push_back(cu);
            compute += cu;
            // from card i's last product done to card i + 1's input in (the last: card 0's next pass), on two cards'
            // timers
            const double bu = i + 1 < N_ ? d_us(st[i][3 * p + 1], st[i + 1][3 * p])
                                         : (p + 1 < P ? d_us(st[i][3 * p + 1], st[0][3 * (p + 1)]) : NAN);
            if (!std::isnan(bu)) {
                r.boundary_us[i].push_back(bu);
            }
        }
        r.compute_us.push_back(compute);
    }
    read_final(r);
    return r;
}

void usage() {
    printf("usage: persistent_pipe [options]\n"
           "  --cards N            the first N visible devices (default: all, 2-16)\n"
           "  --devices LIST       CUDA devices in pipeline order, comma-separated (instead of --cards)\n"
           "  --n N                residual values per token, a multiple of 256 up to %d (default 10240)\n"
           "  --k K                tokens per pass: 1, 2, 4 or 8 (default 1)\n"
           "  --layers L           products per card per pass (default 4; ~1.4 ms each on a P106 at n = 10240)\n"
           "  --card-ms T          instead of --layers: as many products as come closest to T ms on the first card\n"
           "  --passes P           timed passes (default 1000)\n"
           "  --warmup W           untimed passes before them (default 20)\n"
           "  --wire f32|q8        the residual between cards in F32 or Q8_0 (default f32)\n"
           "  --variants LIST      sync,peer,staged,persist,persist-direct (default: all, in this order)\n"
           "  --staged-chunks C    chunks of the staged hop, 1-16 (default 2)\n"
           "  --mail-chunks C      chunks of a mailbox, each with its flag, 1-16 (default 4)\n"
           "  --grid G             blocks of each persistent kernel (default: every block the card holds at once)\n"
           "  --poll-us X          time between two polls of a flag in host memory (default 2)\n"
           "  --slow-poll-us X     the same before the sender's hint (default 50)\n"
           "  --timeout-s S        a mailbox wait longer than this stops the run (default 10)\n"
           "  --sched auto|spin|yield|blocking   the host thread's wait on the devices (default auto)\n"
           "  --seed S             the first token ids (default 1)\n"
           "  --csv FILE           variant,cards,k,wire,layers,key,value,unit\n"
           "  --smoke              20 passes after 2: every path once\n",
           NMAX);
}

}  // namespace

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                usage();
                exit(1);
            }
            return argv[++i];
        };
        if (a == "--cards") opt.cards = atoi(next());
        else if (a == "--devices") opt.devices = next();
        else if (a == "--n") opt.n = atoi(next());
        else if (a == "--k") opt.k = atoi(next());
        else if (a == "--layers") opt.layers = atoi(next());
        else if (a == "--card-ms") opt.card_ms = atof(next());
        else if (a == "--passes") opt.passes = atoi(next());
        else if (a == "--warmup") opt.warmup = atoi(next());
        else if (a == "--wire") opt.wire = next();
        else if (a == "--variants") opt.variants = next();
        else if (a == "--staged-chunks") opt.staged_chunks = atoi(next());
        else if (a == "--mail-chunks") opt.mail_chunks = atoi(next());
        else if (a == "--grid") opt.grid = atoi(next());
        else if (a == "--poll-us") opt.poll_us = atof(next());
        else if (a == "--slow-poll-us") opt.slow_poll_us = atof(next());
        else if (a == "--timeout-s") opt.timeout_s = atof(next());
        else if (a == "--sched") opt.sched = next();
        else if (a == "--seed") opt.seed = (unsigned) atoi(next());
        else if (a == "--csv") opt.csv = next();
        else if (a == "--smoke") opt.smoke = true;
        else { usage(); return a == "-h" || a == "--help" ? 0 : 1; }
    }
    if (opt.smoke) {
        opt.passes = 20;
        opt.warmup = 2;
    }
    K = opt.k;
    with_k(K, [](auto) {});
    if (opt.n < 256 || opt.n % 256 || opt.n > NMAX || opt.layers < 1 || opt.passes < 2 || opt.warmup < 0 ||
        opt.staged_chunks < 1 || opt.staged_chunks > MAX_CHUNKS || opt.mail_chunks < 1 || opt.mail_chunks > MAX_CHUNKS) {
        usage();
        return 1;
    }
    if (opt.wire == "f32") WIRE = WIRE_F32;
    else if (opt.wire == "q8") WIRE = WIRE_Q8;
    else { usage(); return 1; }
    const std::vector<std::string> variants = parse_list(opt.variants);
    for (const std::string & v : variants) {
        if (v != "sync" && v != "peer" && v != "staged" && v != "persist" && v != "persist-direct") {
            fprintf(stderr, "unknown variant %s\n", v.c_str());
            return 1;
        }
    }
    unsigned sched = cudaDeviceScheduleAuto;
    if (opt.sched == "spin") sched = cudaDeviceScheduleSpin;
    else if (opt.sched == "yield") sched = cudaDeviceScheduleYield;
    else if (opt.sched == "blocking") sched = cudaDeviceScheduleBlockingSync;
    else if (opt.sched != "auto") { usage(); return 1; }

    int n_dev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_dev));
    std::vector<int> devs = parse_ints(opt.devices);
    if (devs.empty()) {
        const int nc = opt.cards > 0 ? opt.cards : n_dev;
        for (int i = 0; i < nc; ++i) {
            devs.push_back(i);
        }
    }
    if (devs.size() < 2 || devs.size() > MAX_CARDS) {
        fprintf(stderr, "needs 2-%d cards (%zu given, %d visible)\n", MAX_CARDS, devs.size(), n_dev);
        return 1;
    }
    for (int d : devs) {
        if (d < 0 || d >= n_dev) {
            fprintf(stderr, "device %d is not visible (%d devices)\n", d, n_dev);
            return 1;
        }
    }
    if (opt.csv.size()) {
        g_csv = fopen(opt.csv.c_str(), "w");
        if (!g_csv) {
            perror(opt.csv.c_str());
            return 1;
        }
        fprintf(g_csv, "variant,cards,k,wire,layers,key,value,unit\n");
    }
    N_ = (int) devs.size();
    V_ = opt.n / VB_ROWS;
    wire_bytes = WIRE == WIRE_F32 ? (size_t) K * opt.n * sizeof(float) : (size_t) K * V_ * Q8_BYTES;
    const size_t max_wire = std::max((size_t) K * opt.n * sizeof(float), (size_t) K * sizeof(int));

    // the cards
    cards.resize(N_);
    for (int i = 0; i < N_; ++i) {
        card_t & c = cards[i];
        c.dev = devs[i];
        CUDA_CHECK(cudaInitDevice(c.dev, sched, 0));
        CUDA_CHECK(cudaSetDevice(c.dev));
        CUDA_CHECK(cudaGetDeviceProperties(&c.prop, c.dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&c.clock_khz, cudaDevAttrClockRate, c.dev));
        int coop = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, c.dev));
        CUDA_CHECK(cudaStreamCreateWithFlags(&c.s, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&c.W, (size_t) opt.n * opt.n * sizeof(__half)));
        k_init_weights<<<1024, THREADS, 0, c.s>>>(c.W, opt.n, 0x1000u + (uint32_t) i);
        for (float *& bp : c.buf) {
            CUDA_CHECK(cudaMalloc(&bp, (size_t) K * opt.n * sizeof(float)));
        }
        CUDA_CHECK(cudaMalloc(&c.wire_in, max_wire));
        CUDA_CHECK(cudaMalloc(&c.wire_out, max_wire));
        CUDA_CHECK(cudaMalloc(&c.toks, K * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&c.toks_out, K * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&c.bar, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&c.state, 64 * sizeof(unsigned)));
        CUDA_CHECK(cudaMalloc(&c.stamps, (size_t) (opt.warmup + opt.passes) * 3 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&c.ptoks, K * sizeof(int)));
        int per_sm = 0;
        with_k(K, [&](auto kc) {
            constexpr int KK = decltype(kc)::value;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_pipe<KK>, THREADS, 0));
        });
        const int resident = per_sm * c.prop.multiProcessorCount;
        c.grid = opt.grid > 0 ? std::min(opt.grid, resident) : resident;
        CUDA_CHECK(cudaMalloc(&c.part_v, (size_t) c.grid * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.part_i, (size_t) c.grid * K * sizeof(int)));
        CUDA_CHECK(cudaEventCreateWithFlags(&c.ev_done, cudaEventDisableTiming));
        for (cudaEvent_t & e : c.filled) {
            CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        }
        int code_arch = 0;
        int * d_arch;
        CUDA_CHECK(cudaMalloc(&d_arch, sizeof(int)));
        k_arch<<<1, 1>>>(d_arch);
        CUDA_CHECK(cudaMemcpy(&code_arch, d_arch, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_arch));
        printf("card %d: device %d, %s, PCI %02x:%02x.0, compute %d.%d, %d SMs, %d MHz, code for compute %d.%d; "
               "persistent grid %d (%d a SM resident)%s\n",
               i, c.dev, c.prop.name, c.prop.pciBusID, c.prop.pciDeviceID, c.prop.major, c.prop.minor,
               c.prop.multiProcessorCount, c.clock_khz / 1000, code_arch / 100, (code_arch % 100) / 10, c.grid, per_sm,
               coop ? "" : "; NO cooperative launch");
        if (!coop) {
            return 1;
        }
    }
    // the drained events of boundary i are recorded on card i + 1
    for (int i = 0; i + 1 < N_; ++i) {
        CUDA_CHECK(cudaSetDevice(cards[i + 1].dev));
        for (cudaEvent_t & e : cards[i].drained) {
            CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        }
    }
    // pinned host memory, once every context exists
    for (int i = 0; i < N_; ++i) {
        card_t & c = cards[i];
        CUDA_CHECK(cudaHostAlloc((void **) &c.mail, MAIL_HDR + max_wire, cudaHostAllocPortable | cudaHostAllocMapped));
        memset(c.mail, 0, MAIL_HDR + max_wire);
        CUDA_CHECK(cudaHostAlloc(&c.stage_host, max_wire, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(&c.ring, max_wire, cudaHostAllocPortable));
    }
    const int P = opt.warmup + opt.passes;
    CUDA_CHECK(cudaHostAlloc((void **) &results_host, (size_t) P * K * sizeof(int), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &abort_host, 64, cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &htoks, 64 * sizeof(int), cudaHostAllocPortable));
    for (int i = 0; i < N_; ++i) {
        card_t & c = cards[i];
        CUDA_CHECK(cudaSetDevice(c.dev));
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &c.in_dev, c.mail, 0));
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &c.out_dev, cards[(i + 1) % N_].mail, 0));
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &c.results_dev, results_host, 0));
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &c.abort_dev, abort_host, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    for (int i = 0; i + 1 < N_; ++i) {
        int peer = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&peer, cards[i + 1].dev, cards[i].dev));
        if (peer) {
            printf("cards %d -> %d: peer access possible (the peer copy may not go through the host)\n", i, i + 1);
        }
    }

    // the products per card, and each card's time for them alone
    L_ = opt.layers;
    if (opt.card_ms > 0) {
        const double one = calibrate(cards[0], 1, 5);
        L_ = std::max(1, (int) lround(opt.card_ms / one));
        printf("one product on card 0: %.3f ms, so %d products a card for %.2f ms\n", one, L_, opt.card_ms);
    }
    double compute_ms = 0;
    for (int i = 0; i < N_; ++i) {
        cards[i].cal_ms = calibrate(cards[i], L_, 10);
        compute_ms += cards[i].cal_ms;
    }
    const double w_mb = (double) opt.n * opt.n * sizeof(__half) / 1e6;
    printf("\n%d cards, n = %d, %d token%s a pass, %d products a card (%.1f MB of weights each, read %d times a pass), "
           "the residual in %s: %zu bytes a hop; %d + %d passes\n",
           N_, opt.n, K, K > 1 ? "s" : "", L_, w_mb, L_, WIRE == WIRE_F32 ? "F32" : "Q8_0", wire_bytes, opt.warmup,
           opt.passes);
    printf("the products alone, launched back to back (ms a pass):");
    for (int i = 0; i < N_; ++i) {
        printf(" %.3f", cards[i].cal_ms);
        csv_row("calibration", "card" + std::to_string(i) + "_ms", cards[i].cal_ms, "ms");
    }
    printf("; sum %.3f ms, %.1f GB/s on card 0\n", compute_ms, w_mb * L_ / cards[0].cal_ms / 1000.0);
    csv_row("calibration", "sum_ms", compute_ms, "ms");
    csv_row("calibration", "hop_bytes", (double) wire_bytes, "B");
    printf("persistent mailboxes: %d chunks; polls every %.2f us after the hint, %.1f us before; host wait: %s\n\n",
           opt.mail_chunks, opt.poll_us, opt.slow_poll_us, opt.sched.c_str());

    // the variants
    std::vector<run_result> runs;
    for (const std::string & v : variants) {
        printf("running %s ...\n", v.c_str());
        fflush(stdout);
        if (v == "persist" || v == "persist-direct") {
            runs.push_back(run_persistent(v, v == "persist-direct"));
        } else {
            runs.push_back(run_host(v));
        }
    }

    // the checks: every variant's ids and final residual against the first variant's
    int failures = 0;
    const run_result * ref = nullptr;
    for (const run_result & r : runs) {
        if (r.ok) {
            ref = &r;
            break;
        }
    }
    printf("\n%-15s %26s %9s %9s %11s %12s %11s  %s\n", "variant", "ms a pass: median [p10,p90]", "mean", "tok/s",
           "compute ms", "overhead ms", "us a hop", "check");
    for (const run_result & r : runs) {
        if (!r.ok) {
            printf("%-15s FAILED (timeout)\n", r.name.c_str());
            ++failures;
            continue;
        }
        uint64_t h = 1469598103934665603ull;
        for (int x : r.toks) {
            h = (h ^ (uint32_t) x) * 1099511628211ull;
        }
        for (float f : r.y) {
            uint32_t u;
            memcpy(&u, &f, 4);
            h = (h ^ u) * 1099511628211ull;
        }
        std::string check;
        if (&r == ref) {
            check    = "reference";
        } else {
            size_t first_bad = r.toks.size();
            for (size_t i = 0; i < r.toks.size(); ++i) {
                if (r.toks[i] != ref->toks[i]) {
                    first_bad = i;
                    break;
                }
            }
            const bool same_y = r.y.size() == ref->y.size() && memcmp(r.y.data(), ref->y.data(), r.y.size() * 4) == 0;
            if (first_bad == r.toks.size() && same_y) {
                check = "identical";
            } else {
                ++failures;
                check = first_bad < r.toks.size() ? "DIFFERS from pass " + std::to_string(first_bad / K)
                                                  : "DIFFERS (final residual)";
            }
        }
        const double m = mean(r.pass_us) / 1000.0, c = mean(r.compute_us) / 1000.0;
        printf("%-15s %8.3f [%7.3f,%7.3f] %9.3f %9.2f %11.3f %12.3f %11.1f  %s, checksum %016llx\n", r.name.c_str(),
               median(r.pass_us) / 1000.0, pctl(r.pass_us, 0.1) / 1000.0, pctl(r.pass_us, 0.9) / 1000.0, m,
               K * 1000.0 / m, c, m - c, (m - c) * 1000.0 / N_, check.c_str(), (unsigned long long) h);
        csv_row(r.name, "pass_ms_median", median(r.pass_us) / 1000.0, "ms");
        csv_row(r.name, "pass_ms_p10", pctl(r.pass_us, 0.1) / 1000.0, "ms");
        csv_row(r.name, "pass_ms_p90", pctl(r.pass_us, 0.9) / 1000.0, "ms");
        csv_row(r.name, "pass_ms_mean", m, "ms");
        csv_row(r.name, "tok_s", K * 1000.0 / m, "tok/s");
        csv_row(r.name, "compute_ms", c, "ms");
        csv_row(r.name, "overhead_ms", m - c, "ms");
        csv_row(r.name, "overhead_us_per_hop", (m - c) * 1000.0 / N_, "us");
        csv_row(r.name, "wall_s", r.wall_s, "s");
        csv_row(r.name, "identical", check == "reference" || check == "identical" ? 1 : 0, "");
    }

    // the persistent runs per card and boundary
    for (const run_result & r : runs) {
        if (!r.ok || r.card_us.empty()) {
            continue;
        }
        printf("\n%s per card, median us a pass: products", r.name.c_str());
        for (int i = 0; i < N_; ++i) {
            printf(" %.0f", median(r.card_us[i]));
            csv_row(r.name, "card" + std::to_string(i) + "_products_us", median(r.card_us[i]), "us");
        }
        printf("\n%s per boundary (card i's last product done -> card i+1's input in; the last: back to card 0; on two "
               "cards' timers), median us:",
               r.name.c_str());
        for (int i = 0; i < N_; ++i) {
            printf(" %.1f", median(r.boundary_us[i]));
            csv_row(r.name, "boundary" + std::to_string(i) + "_us", median(r.boundary_us[i]), "us");
        }
        printf("\n");
    }

    // what the persistent pipeline saves against each host-driven variant
    printf("\n");
    for (const run_result & h : runs) {
        if (!h.ok || !h.card_us.empty()) {
            continue;
        }
        for (const run_result & p : runs) {
            if (!p.ok || p.card_us.empty()) {
                continue;
            }
            const double mh = mean(h.pass_us) / 1000.0, mp = mean(p.pass_us) / 1000.0;
            printf("%s against %s: %.3f ms a pass less (%.1f %%), %.1f us a hop\n", p.name.c_str(), h.name.c_str(),
                   mh - mp, 100.0 * (mh - mp) / mh, (mh - mp) * 1000.0 / N_);
            csv_row(p.name + "_vs_" + h.name, "saved_ms_per_pass", mh - mp, "ms");
            csv_row(p.name + "_vs_" + h.name, "saved_pct", 100.0 * (mh - mp) / mh, "%");
        }
    }
    printf("\nchecks: %s\n", failures ? "FAILED" : "every variant gives the same token ids and final residual");
    if (g_csv) {
        fclose(g_csv);
    }
    return failures ? 2 : 0;
}
