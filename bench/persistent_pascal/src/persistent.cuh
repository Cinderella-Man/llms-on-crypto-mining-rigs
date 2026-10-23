// One Flint MoE layer at one token in a single kernel launch.
//
// Work list. The layer is cut into warp-sized work items, numbered in an order that respects every dependency:
//   router rows (2 per item) | shared gate/up (4 rows) | routed gate/up (4 rows of one expert, experts in turn)
//   | shared down (8 rows) | routed down (8 rows of one expert, grouped by rows: g0 e0..e5, g1 e0..e5, ...)
// Each warp takes the next item from a global counter (it claims its next item before working on the current one, so
// the atomic's latency hides behind the work) until the list is exhausted.
//
// Dependencies are counters in global memory instead of kernel boundaries or grid barriers:
//   - the warp that finishes the last router item runs the top-6 selection and publishes it;
//   - the warp that finishes the last shared gate/up item quantizes the shared expert's 1,024 activations and publishes
//     them; the same per routed expert for its 512 activations ("team" counters: one per expert);
//   - routed gate/up items wait for the top-6, shared down items for the shared activations, routed down items for
//     their own expert's activations only;
//   - every 8-row output group counts its 7 contributions (6 routed experts and the shared expert); the warp that adds
//     the 7th computes the weighted sum, the shared expert and the residual for those 8 rows. Nobody waits for it.
// Every block first computes the RMS norm and the Q8_1 quantization of the layer input into its own shared memory, so
// neither needs a dependency.
//
// Why it cannot deadlock. A work item only ever waits for results of items with smaller numbers. A warp takes items in
// increasing order, and holds at most two (the current one and the one it claimed next). So the smallest unfinished item
// always belongs to a warp that is working on it, and all its inputs are complete: it finishes, and by induction all do.
// This holds for any grid size, whether or not all blocks are resident at once (blocks that start late find the list
// empty), so the kernel needs neither a cooperative launch nor an occupancy guarantee. A wait that exceeds spin_limit
// (a bug, or a fault injected for testing) sets an error flag and every warp leaves; the host then runs the
// multi-launch path instead.
//
// Arithmetic. Every product, reduction, quantization and sum is the same device function the multi-launch path uses,
// fed with the same values, so the output is bit-identical to it. The norm runs 256 threads standing in for the norm
// kernel's 512 (two each, with the same warp and block reductions); a router row is one warp standing in for the
// 256-thread mmvf block (eight sums per lane, the same butterfly reductions); a product pass may cover more rows per
// lane group than the fork's launch shape (RG), which shares activation loads but leaves each row's sum unchanged.
//
// The counters reset themselves: the last warp to leave zeroes them for the next launch.
#pragma once

#include "fork_kernels.cuh"

namespace pk {

// Flint (49 layers, 47 of them MoE): hidden size, routed experts, experts per token, intermediate sizes.
constexpr int H  = 2048;
constexpr int NE = 256;
constexpr int KU = 6;
constexpr int F  = 512;
constexpr int FS = 1024;

constexpr int PK_THREADS = 256;  // threads per block of the persistent kernel
constexpr int PK_NW      = PK_THREADS / 32;

// Work items
constexpr int RT_ROWS = 2;                // router rows per item
constexpr int SG_ROWS = 4;                // shared gate/up rows per item
constexpr int EG_ROWS = 4;                // routed gate/up rows per item
constexpr int CR      = 8;                // rows per output group (shared and routed down items)
constexpr int N_RT    = NE / RT_ROWS;     // 128
constexpr int N_SG    = FS / SG_ROWS;     // 256
constexpr int N_EG1   = F / EG_ROWS;      // 128 per expert
constexpr int N_EG    = KU * N_EG1;       // 768
constexpr int N_GRP   = H / CR;           // 256 output groups
constexpr int N_SD    = N_GRP;            // 256
constexpr int N_ED    = KU * N_GRP;       // 1,536
constexpr int I_RT    = 0;
constexpr int I_SG    = I_RT + N_RT;
constexpr int I_EG    = I_SG + N_SG;
constexpr int I_SD    = I_EG + N_EG;
constexpr int I_ED    = I_SD + N_SD;
constexpr int N_ITEMS = I_ED + N_ED;      // 2,944

// Counters and flags (32-bit words)
constexpr int S_NEXT    = 0;              // next work item
constexpr int S_RT_DONE = 1;              // router items done
constexpr int S_TOPK    = 2;              // top-6 published
constexpr int S_SG_DONE = 3;              // shared gate/up items done
constexpr int S_SACT    = 4;              // shared activations quantized
constexpr int S_EG_DONE = 5;              // + e: routed gate/up items of expert slot e done
constexpr int S_EACT    = S_EG_DONE + KU; // + e: activations of expert slot e quantized
constexpr int S_EXIT    = S_EACT + KU;    // warps that have left
constexpr int S_ERR     = S_EXIT + 1;     // a wait timed out (reset by the host only)
constexpr int S_COMB    = 32;             // + g: contributions to output group g
constexpr int N_SYNC    = S_COMB + N_GRP;

// Trace (TRACE = true): per item class c (0 router, 1 shared gate/up, 2 routed gate/up, 3 shared down, 4 routed down)
// first start, last end, wait spins and items; then single events. Times are %globaltimer nanoseconds.
enum { TR_CLASS = 0, TR_START = 20, TR_PROLOGUE = 21, TR_TOPK = 22, TR_SACT = 23, TR_EACT_FIRST = 24, TR_EACT_LAST = 25,
       TR_COMB_FIRST = 26, TR_COMB_LAST = 27, TR_EXIT = 28, TR_N = 32 };

struct layer_weights {
    const float * norm_w;    // [H]
    const void  * router;    // [NE] rows of H, F16 or F32
    const char  * gate_exps; // [NE][F] rows of H weights
    const char  * up_exps;
    const char  * down_exps; // [NE][H] rows of F weights
    const char  * gate_sh;   // [FS] rows of H
    const char  * up_sh;
    const char  * down_sh;   // [H] rows of FS
};

// What one layer evaluation writes. The multi-launch path uses all of it; the persistent kernel writes logits, ids,
// weights, act, aq, dexp, sact, sq, sd (the shared expert's output alone; in the multi-launch path sd holds the shared
// output plus the routed sum, as the fused MUL_MAT+ADD writes it) and out.
struct layer_bufs {
    float      * xn;      // [H] normed input
    block_q8_1 * yq;      // [H/32] its Q8_1 copy for the routed gate/up
    float      * logits;  // [NE]
    int32_t    * ids;     // [NE] (the first KU are written)
    float      * weights; // [KU]
    float      * act;     // [KU][F] routed SwiGLU outputs
    block_q8_1 * aq;      // [KU][F/32]
    float      * dexp;    // [KU][H] routed down outputs
    float      * moe;     // [H] weighted sum (multi-launch path)
    block_q8_1 * yq2;     // [H/32] the shared expert's own copy (multi-launch path)
    float      * sact;    // [FS] shared SwiGLU output
    block_q8_1 * sq;      // [FS/32]
    float      * sd;      // [H]
    float      * out;     // [H] layer output (residual stream)
};

struct pk_args {
    layer_weights        w;
    const float        * x;            // [H] layer input (residual stream)
    layer_bufs           b;
    unsigned           * sync;         // [N_SYNC], zero before the first launch
    float              * dbg_xn;       // if set, block 0 copies its normed input here
    block_q8_1         * dbg_yq;       // and its Q8_1 copy here
    unsigned long long * trace;        // TRACE only
    const float        * expert_scale; // nullptr, as in Flint's graph
    int                  ncols;        // H; runtime values as the multi-launch kernels receive them
    float                eps;
    float                clamp_val;
    float                scale_val;
    int                  spin_limit;   // polls of a flag before giving up
    int                  fault;        // testing: 1 = never publish the top-6 (every wait on it must time out)
};

constexpr __host__ __device__ int cmax(const int a, const int b) {
    return a > b ? a : b;
}

static __device__ __forceinline__ unsigned long long pk_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

static __device__ __forceinline__ unsigned pk_load_volatile(const unsigned * p) {
    return *(const volatile unsigned *) p;
}

// The next work item for the whole warp.
static __device__ __forceinline__ unsigned pk_fetch(unsigned * sync, const int lane) {
    unsigned i = 0;
    if (lane == 0) {
        i = atomicAdd(&sync[S_NEXT], 1u);
    }
    return __shfl_sync(0xffffffff, i, 0);
}

// Count this warp's finished item into *ctr (after its results are stored). True in all lanes for the warp that brings
// the count to n; that warp then sees every other contributor's results (read them with __ldcg).
static __device__ __forceinline__ bool pk_arrive(unsigned * ctr, const unsigned n, const int lane) {
    __threadfence();
    __syncwarp();
    unsigned old = 0;
    if (lane == 0) {
        old = atomicAdd(ctr, 1u);
    }
    old = __shfl_sync(0xffffffff, old, 0);
    __syncwarp();
    const bool last = old == n - 1;
    if (last) {
        __threadfence();
    }
    return last;
}

static __device__ __forceinline__ void pk_publish(unsigned * flag, const int lane) {
    __threadfence();
    __syncwarp();
    if (lane == 0) {
        atomicExch(flag, 1u);
    }
}

// Wait until *flag is set. False (in all lanes) if the wait timed out or another warp reported a timeout.
static __device__ __forceinline__ bool pk_wait(const unsigned * flag, unsigned * err, const int spin_limit, const int lane,
                                               unsigned & spins) {
    unsigned ok = 1;
    if (lane == 0) {
        int n = 0;
        while (pk_load_volatile(flag) == 0) {
            ++n;
            if (n >= spin_limit || ((n & 63) == 0 && pk_load_volatile(err) != 0)) {
                atomicExch(err, 1u);
                ok = 0;
                break;
            }
        }
        spins += n;
    }
    ok = __shfl_sync(0xffffffff, ok, 0);
    __syncwarp();
    __threadfence();
    return ok != 0;
}

// Persistent kernel for one layer. TR: router weight type (half or float); TGU, TDN: routed gate/up and down types;
// TSH: the shared expert's type (gate, up and down); RG: routed gate/up rows per warp pass (the fork uses 1; 2 puts twice
// the bytes in flight per warp, with the same arithmetic per row).
template <typename TR, int TGU, int TDN, int TSH, int RG, bool TRACE>
__launch_bounds__(PK_THREADS, 2)
static __global__ void k_moe_layer(const pk_args a) {
    // launch shapes of the fork's kernel for each product
    constexpr int C_GU   = mmvqp_shape<TGU, true>::C;
    constexpr int LPR_GU = mmvqp_lpr_log2(TGU, H, C_GU);
    constexpr int NIT_GU = (H / 32 / C_GU) >> LPR_GU;
    constexpr int RB_GU  = H / 32 * mmvqp_traits<TGU>::bs;
    constexpr int C_SG   = mmvqp_shape<TSH, true>::C;
    constexpr int LPR_SG = mmvqp_lpr_log2(TSH, H, C_SG);
    constexpr int NIT_SG = (H / 32 / C_SG) >> LPR_SG;
    constexpr int RB_SG  = H / 32 * mmvqp_traits<TSH>::bs;
    constexpr int R_SD   = mmvqp_shape<TSH, false>::R;
    constexpr int C_SD   = mmvqp_shape<TSH, false>::C;
    constexpr int LPR_SD = mmvqp_lpr_log2(TSH, FS, C_SD);
    constexpr int NIT_SD = (FS / 32 / C_SD) >> LPR_SD;
    constexpr int RB_SD  = FS / 32 * mmvqp_traits<TSH>::bs;
    constexpr int R_ED   = mmvqp_shape<TDN, false>::R;
    constexpr int C_ED   = mmvqp_shape<TDN, false>::C;
    constexpr int LPR_ED = mmvqp_lpr_log2(TDN, F, C_ED);
    constexpr int NIT_ED = (F / 32 / C_ED) >> LPR_ED;
    constexpr int RB_ED  = F / 32 * mmvqp_traits<TDN>::bs;
    constexpr int ROWS_SD = (32 >> LPR_SD) * R_SD;   // rows per warp pass
    constexpr int ROWS_ED = (32 >> LPR_ED) * R_ED;
    static_assert(LPR_GU == 5 && mmvqp_shape<TSH, true>::R == 1 && LPR_SG == 5, "gate/up passes are one lane group per warp");
    static_assert(EG_ROWS % RG == 0 && CR % ROWS_SD == 0 && CR % ROWS_ED == 0, "items must cover whole passes and output groups");
    constexpr int STAGE = cmax(cmax(mmvqp_stage_bytes<TGU, RG, 2>(), mmvqp_stage_bytes<TSH, 1, 2>()),
                               cmax(mmvqp_stage_bytes<TDN, R_ED, 1>(), mmvqp_stage_bytes<TSH, R_SD, 1>()));

    __shared__ __align__(16) block_q8_1 s_yq[H / 32];
    __shared__ float                    s_red[32];
    __shared__ __align__(16) char       s_stage[PK_NW][STAGE];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    unsigned * sync = a.sync;
    [[maybe_unused]] unsigned long long * tr = a.trace;

    if constexpr (TRACE) {
        if (tid == 0) {
            atomicMin(&tr[TR_START], pk_now());
        }
    }

    // ---- RMS norm and Q8_1 quantization of the input, in every block -------------------------------------------------
    // k_rms_norm_mul's 512 threads: thread tid stands for threads tid and tid + 256. The normed values themselves are not
    // kept: the quantization and the router recompute each one as rms_norm_apply1(scale, x, w), the same two multiplies.
    float scale;
    {
        constexpr int VPT = 512 / PK_THREADS;
        constexpr int NQ  = H / 32 / PK_NW;  // Q8_1 blocks per warp
        float xq[NQ], mq[NQ];                // this lane's value of each, loaded before the reduction
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
            xq[k] = __ldg(a.x + (warp + k * PK_NW) * 32 + lane);
            mq[k] = __ldg(a.w.norm_w + (warp + k * PK_NW) * 32 + lane);
        }
        const float4 * x4 = (const float4 *) a.x;
#pragma unroll
        for (int v = 0; v < VPT; ++v) {
            float tmp = rms_norm_partial(__ldg(x4 + tid + v * PK_THREADS));
            tmp = warp_reduce_sum(tmp);
            if (lane == 0) {
                s_red[(tid + v * PK_THREADS) / 32] = tmp;
            }
        }
        __syncthreads();
        float tmp = 0.0f;
        if (lane < 512 / 32) {
            tmp = s_red[lane];
        }
        tmp = warp_reduce_sum(tmp);
        scale = rms_norm_scale(tmp, a.ncols, a.eps);
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
            const int ib = warp + k * PK_NW;
            const float xn = rms_norm_apply1(scale, xq[k], mq[k]);
            quantize_q8_1_lane(xn, s_yq, ib, lane);
            if (a.dbg_xn != nullptr && blockIdx.x == 0) {
                a.dbg_xn[ib * 32 + lane] = xn;
            }
        }
        __syncthreads();
        if (a.dbg_yq != nullptr && blockIdx.x == 0) {
            for (int i = tid; i < H / 32; i += PK_THREADS) {
                a.dbg_yq[i] = s_yq[i];
            }
        }
    }
    if constexpr (TRACE) {
        if (tid == 0) {
            atomicMax(&tr[TR_PROLOGUE], pk_now());
        }
    }

    // ---- work items --------------------------------------------------------------------------------------------------
    char * smw = s_stage[warp];
    bool have_topk = false;
    int32_t eid_l = 0;  // lane e < KU: expert id of slot e
    float   ew_l  = 0.f; //              and its routing weight
    bool have_sact = false;
    unsigned eact_mask = 0;
    unsigned spins = 0;
    bool failed = false;

    // the top-6, once per warp
    auto need_topk = [&]() -> bool {
        if (!have_topk) {
            if (!pk_wait(&sync[S_TOPK], &sync[S_ERR], a.spin_limit, lane, spins)) {
                return false;
            }
            if (lane < KU) {
                eid_l = __ldcg(&a.b.ids[lane]);
                ew_l  = __ldcg(&a.b.weights[lane]);
            }
            have_topk = true;
        }
        return true;
    };

    // weighted sum of the routed experts, plus the shared expert, plus the residual, for output group g
    auto combine = [&](const int g) {
        float w[KU];
#pragma unroll
        for (int e = 0; e < KU; ++e) {
            w[e] = __shfl_sync(0xffffffff, ew_l, e);
        }
        if (lane < CR) {
            const int r = g * CR + lane;
            float v[KU];
#pragma unroll
            for (int e = 0; e < KU; ++e) {
                v[e] = __ldcg(&a.b.dexp[e * H + r]);
            }
            const float moe = moe_weighted_sum<KU>(v, w, a.expert_scale, 0);
            float result = __ldcg(&a.b.sd[r]);  // shared down product, then its fused add of the routed sum
            result += moe;
            a.b.out[r] = op_add(result, __ldg(a.x + r));
        }
        if constexpr (TRACE) {
            if (lane == 0) {
                const unsigned long long t = pk_now();
                atomicMin(&tr[TR_COMB_FIRST], t);
                atomicMax(&tr[TR_COMB_LAST], t);
            }
        }
    };

    unsigned item = pk_fetch(sync, lane);
    while (item < (unsigned) N_ITEMS) {
        const unsigned next = pk_fetch(sync, lane);
        [[maybe_unused]] unsigned long long t0 = 0;
        [[maybe_unused]] const unsigned spins0 = spins;
        if constexpr (TRACE) {
            t0 = pk_now();
        }
        int cls;
        if (item < (unsigned) I_SG) {
            // router rows: one warp stands for k_router's 256-thread block
            cls = 0;
            for (int k = 0; k < RT_ROWS; ++k) {
                const int row = (item - I_RT) * RT_ROWS + k;
                const TR * xr = (const TR *) a.w.router + (int64_t) row * a.ncols;
                const float2 * x2 = (const float2 *) a.x;
                const float2 * m2 = (const float2 *) a.w.norm_w;
                const auto xn2 = [&](const int c) {
                    const float2 xi = __ldg(x2 + c);  // read-only during the kernel: cached in L1
                    const float2 m  = __ldg(m2 + c);
                    return make_float2(rms_norm_apply1(scale, xi.x, m.x), rms_norm_apply1(scale, xi.y, m.y));
                };
                float s[8];
#pragma unroll
                for (int vw = 0; vw < 8; ++vw) {
                    s[vw] = mmvf_partial<TR, H / 2, true>(xr, xn2, a.ncols / 2, vw * 32 + lane, 256);
                }
#pragma unroll
                for (int vw = 0; vw < 8; ++vw) {
                    s[vw] = warp_reduce_sum(s[vw]);
                }
                float sumf = 0.0f;  // buf_iw[lane]: the 8 warp sums, zero beyond
#pragma unroll
                for (int vw = 0; vw < 8; ++vw) {
                    if (lane == vw) {
                        sumf = s[vw];
                    }
                }
                sumf = warp_reduce_sum(sumf);
                if (lane == 0) {
                    a.b.logits[row] = sumf;
                }
            }
            if (pk_arrive(&sync[S_RT_DONE], N_RT, lane)) {
                float wt[NE / 32];
#pragma unroll
                for (int i = 0; i < NE / 32; ++i) {
                    wt[i] = __ldcg(&a.b.logits[i * 32 + lane]);
                }
                topk_moe_warp<NE>(wt, a.b.weights, a.b.ids, KU, a.clamp_val, a.scale_val);
                if (a.fault != 1) {
                    pk_publish(&sync[S_TOPK], lane);
                }
                if constexpr (TRACE) {
                    if (lane == 0) {
                        tr[TR_TOPK] = pk_now();
                    }
                }
            }
        } else if (item < (unsigned) I_EG) {
            // shared expert gate/up + SwiGLU
            cls = 1;
            for (int p = 0; p < SG_ROWS; ++p) {
                float acc[2][1];
                const int row0 = mmvqp_rows<TSH, 1, C_SG, true, 2, SRC_SHARED>(
                    a.w.up_sh, a.w.gate_sh, FS, RB_SG, LPR_SG, NIT_SG, (const char *) s_yq, (item - I_SG) * SG_ROWS + p, lane, smw, acc);
                mmvqp_store<1, 2, MMVQP_GATE>(acc, row0, lane, LPR_SG, FS, a.b.sact, nullptr);
            }
            if (pk_arrive(&sync[S_SG_DONE], N_SG, lane)) {
                float v[FS / 32];  // all loads first, then the 32 block quantizations
#pragma unroll
                for (int ib = 0; ib < FS / 32; ++ib) {
                    v[ib] = __ldcg(&a.b.sact[ib * 32 + lane]);
                }
#pragma unroll
                for (int ib = 0; ib < FS / 32; ++ib) {
                    quantize_q8_1_lane(v[ib], a.b.sq, ib, lane);
                }
                pk_publish(&sync[S_SACT], lane);
                if constexpr (TRACE) {
                    if (lane == 0) {
                        tr[TR_SACT] = pk_now();
                    }
                }
            }
        } else if (item < (unsigned) I_SD) {
            // routed expert gate/up + SwiGLU, one expert slot per item
            cls = 2;
            if (!need_topk()) {
                failed = true;
                break;
            }
            const int j = item - I_EG;
            const int e = j / N_EG1;
            const int64_t xoff = (int64_t) __shfl_sync(0xffffffff, eid_l, e) * F * RB_GU;
            for (int p = 0; p < EG_ROWS / RG; ++p) {
                float acc[2][RG];
                const int row0 = mmvqp_rows<TGU, RG, C_GU, true, 2, SRC_SHARED>(
                    a.w.up_exps + xoff, a.w.gate_exps + xoff, F, RB_GU, LPR_GU, NIT_GU, (const char *) s_yq,
                    (j % N_EG1) * (EG_ROWS / RG) + p, lane, smw, acc);
                mmvqp_store<RG, 2, MMVQP_GATE>(acc, row0, lane, LPR_GU, F, a.b.act + e * F, nullptr);
            }
            if (pk_arrive(&sync[S_EG_DONE + e], N_EG1, lane)) {
                float v[F / 32];
#pragma unroll
                for (int ib = 0; ib < F / 32; ++ib) {
                    v[ib] = __ldcg(&a.b.act[e * F + ib * 32 + lane]);
                }
#pragma unroll
                for (int ib = 0; ib < F / 32; ++ib) {
                    quantize_q8_1_lane(v[ib], a.b.aq, e * (F / 32) + ib, lane);
                }
                pk_publish(&sync[S_EACT + e], lane);
                if constexpr (TRACE) {
                    if (lane == 0) {
                        const unsigned long long t = pk_now();
                        atomicMin(&tr[TR_EACT_FIRST], t);
                        atomicMax(&tr[TR_EACT_LAST], t);
                    }
                }
            }
        } else if (item < (unsigned) I_ED) {
            // shared expert down projection, 8 rows
            cls = 3;
            if (!have_sact) {
                if (!pk_wait(&sync[S_SACT], &sync[S_ERR], a.spin_limit, lane, spins)) {
                    failed = true;
                    break;
                }
                have_sact = true;
            }
            const int g = item - I_SD;
            for (int p = 0; p < CR / ROWS_SD; ++p) {
                float acc[1][R_SD];
                const int row0 = mmvqp_rows<TSH, R_SD, C_SD, true, 1, SRC_LDCG>(
                    a.w.down_sh, nullptr, H, RB_SD, LPR_SD, NIT_SD, (const char *) a.b.sq, g * (CR / ROWS_SD) + p, lane, smw, acc);
                mmvqp_store<R_SD, 1, MMVQP_PLAIN>(acc, row0, lane, LPR_SD, H, a.b.sd, nullptr);
            }
            if (pk_arrive(&sync[S_COMB + g], KU + 1, lane)) {
                if (!need_topk()) {
                    failed = true;
                    break;
                }
                combine(g);
            }
        } else {
            // routed expert down projection, 8 rows of one expert slot
            cls = 4;
            const int j = item - I_ED;
            const int g = j / KU;
            const int e = j % KU;
            if (!need_topk()) {
                failed = true;
                break;
            }
            if (!((eact_mask >> e) & 1)) {
                if (!pk_wait(&sync[S_EACT + e], &sync[S_ERR], a.spin_limit, lane, spins)) {
                    failed = true;
                    break;
                }
                eact_mask |= 1u << e;
            }
            const int64_t xoff = (int64_t) __shfl_sync(0xffffffff, eid_l, e) * H * RB_ED;
            for (int p = 0; p < CR / ROWS_ED; ++p) {
                float acc[1][R_ED];
                const int row0 = mmvqp_rows<TDN, R_ED, C_ED, true, 1, SRC_LDCG>(
                    a.w.down_exps + xoff, nullptr, H, RB_ED, LPR_ED, NIT_ED, (const char *) (a.b.aq + e * (F / 32)),
                    g * (CR / ROWS_ED) + p, lane, smw, acc);
                mmvqp_store<R_ED, 1, MMVQP_PLAIN>(acc, row0, lane, LPR_ED, H, a.b.dexp + e * H, nullptr);
            }
            if (pk_arrive(&sync[S_COMB + g], KU + 1, lane)) {
                combine(g);
            }
        }
        if constexpr (TRACE) {
            if (lane == 0) {
                atomicMin(&tr[TR_CLASS + 4 * cls + 0], t0);
                atomicMax(&tr[TR_CLASS + 4 * cls + 1], pk_now());
                atomicAdd(&tr[TR_CLASS + 4 * cls + 2], (unsigned long long) (spins - spins0));
                atomicAdd(&tr[TR_CLASS + 4 * cls + 3], 1ull);
            }
        }
        item = next;
    }
    (void) failed;

    // ---- leave; the last warp out resets the counters ----------------------------------------------------------------
    __syncwarp();
    unsigned old = 0;
    if (lane == 0) {
        old = atomicAdd(&sync[S_EXIT], 1u);
    }
    old = __shfl_sync(0xffffffff, old, 0);
    if (old == gridDim.x * PK_NW - 1) {
        __threadfence();
        for (int i = lane; i < N_SYNC; i += 32) {
            if (i != S_ERR) {
                sync[i] = 0;
            }
        }
        __threadfence();
        if constexpr (TRACE) {
            if (lane == 0) {
                tr[TR_EXIT] = pk_now();
            }
        }
    }
}

} // namespace pk
