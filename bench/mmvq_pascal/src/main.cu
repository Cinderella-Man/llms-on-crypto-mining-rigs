// mmvq_pascal: single-token quantized matrix-vector (decode) microbenchmark on Flint's shapes.
// See ../README.md for what it measures and how to run it.

#include "common.cuh"
#include "data.cuh"
#include "llama_mmvq.cuh"
#include "memsys.cuh"
#include "pascal_mmvq.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

using namespace mb;

namespace {

constexpr int NW = 4; // warps per block of the tuned kernels
bool g_verbose = false;

// ---- shapes ----------------------------------------------------------------------------------------------------

enum shape_kind { S_DENSE, S_DENSE_GLU, S_MOE_GLU, S_MOE_DOWN };

struct shape_def {
    const char * name;
    shape_kind kind;
    int K, N;          // input width, output rows (per expert)
    int n_expert;      // experts in the tensor (MoE)
    int n_used;        // experts selected per token (MoE)
    int calls;         // calls per Flint token
    const char * what;
};

// Flint: hidden 2048, 32 query heads x 128, 4 KV heads x 128, 49 layers; 47 MoE layers with 256 experts of width 512,
// top-6, plus a shared expert of width 1024; vocabulary 150,144.
const shape_def SHAPES[] = {
    { "attn_q",        S_DENSE,     2048,   4096,   0, 0, 49, "attention Q projection" },
    { "attn_kv",       S_DENSE,     2048,    512,   0, 0, 98, "attention K or V projection (2 calls per layer)" },
    { "attn_output",   S_DENSE,     4096,   2048,   0, 0, 49, "attention output projection" },
    { "moe_gate_up",   S_MOE_GLU,   2048,    512, 256, 6, 47, "routed experts, fused gate+up+SwiGLU, 6 of 256 (MUL_MAT_ID)" },
    { "moe_down",      S_MOE_DOWN,   512,   2048, 256, 6, 47, "routed experts, down projection, 6 of 256 (MUL_MAT_ID)" },
    { "shexp_gate_up", S_DENSE_GLU, 2048,   1024,   0, 0, 47, "shared expert, fused gate+up+SwiGLU" },
    { "shexp_down",    S_DENSE,     1024,   2048,   0, 0, 47, "shared expert, down projection" },
    { "output",        S_DENSE,     2048, 150144,   0, 0,  1, "output head" },
};

bool is_moe(shape_kind k) { return k == S_MOE_GLU || k == S_MOE_DOWN; }
bool is_glu(shape_kind k) { return k == S_DENSE_GLU || k == S_MOE_GLU; }

// ---- variants --------------------------------------------------------------------------------------------------

enum variant_kind { V_LLAMA, V_LLAMA_SK, V_PR25479, V_ROWS, V_RL, V_ST, V_SOA };

struct variant_def {
    const char * name;
    variant_kind kind;
    int R, C;
    const char * what;
};

const variant_def VARIANTS[] = {
    { "llama",   V_LLAMA,   0, 0, "(a) llama.cpp d77256999 MMVQ, Pascal launch shape: 4 warps, 1 row/block (small K: 4 rows/block)" },
    { "llama_sk", V_LLAMA_SK, 0, 0, "(a) with the small-K shape also for IQ4_XS (Q47; llama.cpp excludes IQ4_XS on Pascal); IQ4_XS only" },
    { "pr25479", V_PR25479, 0, 0, "(b) same kernel, PR #25479 shape: 2 warps at one token (small K: 2 rows/block)" },
    { "p_r1c1",  V_ROWS,    1, 1, "(c) tuned, K-/I-quants: 1 row per lane group, one 256-weight super-block per lane per iteration" },
    { "p_r2c1",  V_ROWS,    2, 1, "(c) tuned, K-/I-quants: 2 rows per lane group, one super-block per lane" },
    { "p_r1c2",  V_ROWS,    1, 2, "(c) tuned: 1 row per lane group, 2 blocks per lane per iteration" },
    { "p_r2c2",  V_ROWS,    2, 2, "(c) tuned: 2 rows share the activations, 2 blocks per lane" },
    { "p_r4c2",  V_ROWS,    4, 2, "(c) tuned: 4 rows share the activations, 2 blocks per lane" },
    { "p_r2c4",  V_ROWS,    2, 4, "(c) tuned: 2 rows, 4 blocks per lane (128-bit loads for Q4_1)" },
    { "rl_r1c2", V_RL,      1, 2, "(e) row-local planes (Q8_0, Q4_1, Q5_0): codes then scales within each row, 128-bit code loads; 1 row, 2 blocks per lane" },
    { "rl_r2c2", V_RL,      2, 2, "(e) row-local planes, 2 rows, 2 blocks per lane" },
    { "rl_r4c2", V_RL,      4, 2, "(e) row-local planes, 4 rows, 2 blocks per lane" },
    { "rl_r2c4", V_RL,      2, 4, "(e) row-local planes, 2 rows, 4 blocks per lane" },
    { "st_r1c1", V_ST,      1, 1, "(f) staged, K-/I-quants: 1 row, one super-block per lane" },
    { "st_r2c1", V_ST,      2, 1, "(f) staged, K-/I-quants: 2 rows, one super-block per lane" },
    { "st_r1c2", V_ST,      1, 2, "(f) unchanged blocks staged through shared memory with coalesced 128-bit loads; 1 row, 2 blocks per lane" },
    { "st_r2c2", V_ST,      2, 2, "(f) staged, 2 rows, 2 blocks per lane" },
    { "st_r4c2", V_ST,      4, 2, "(f) staged, 4 rows, 2 blocks per lane" },
    { "st_r2c4", V_ST,      2, 4, "(f) staged, 2 rows, 4 blocks per lane" },
    { "soa_r2",  V_SOA,     2, 0, "(d) SoA repack (Q4_1, Q4_1_64 only), 128-bit coalesced loads, FP32 activation sums, 2 rows" },
    { "soa_r4",  V_SOA,     4, 0, "(d) SoA repack, 4 rows" },
};

// ---- kernel dispatch -------------------------------------------------------------------------------------------

// K-/I-quants take one super-block per lane (C = 1); the 32-weight types take 2 or 4 blocks. Q6_K's 2-byte-aligned
// super-blocks have no direct-load variant.
template <int T, bool G>
const void * rows_kernel(int R, int C) {
    if constexpr (T == T_Q6_K) {
        return nullptr;
    } else if constexpr (traits<T>::qk == QK_K) {
        if (R == 1 && C == 1) return (const void *) pascal::mmvq_rows<T, 1, 1, NW, G>;
        if (R == 2 && C == 1) return (const void *) pascal::mmvq_rows<T, 2, 1, NW, G>;
    } else {
        if (R == 1 && C == 2) return (const void *) pascal::mmvq_rows<T, 1, 2, NW, G>;
        if (R == 2 && C == 2) return (const void *) pascal::mmvq_rows<T, 2, 2, NW, G>;
        if (R == 4 && C == 2) return (const void *) pascal::mmvq_rows<T, 4, 2, NW, G>;
        if (R == 2 && C == 4) return (const void *) pascal::mmvq_rows<T, 2, 4, NW, G>;
    }
    return nullptr;
}

const void * rows_kernel_rt(int T, bool G, int R, int C) {
    switch (T) {
        case T_Q4_0:    return G ? rows_kernel<T_Q4_0, true>(R, C)    : rows_kernel<T_Q4_0, false>(R, C);
        case T_Q4_1:    return G ? rows_kernel<T_Q4_1, true>(R, C)    : rows_kernel<T_Q4_1, false>(R, C);
        case T_Q8_0:    return G ? rows_kernel<T_Q8_0, true>(R, C)    : rows_kernel<T_Q8_0, false>(R, C);
        case T_Q4_1_64: return G ? rows_kernel<T_Q4_1_64, true>(R, C) : rows_kernel<T_Q4_1_64, false>(R, C);
        case T_Q5_0:    return G ? rows_kernel<T_Q5_0, true>(R, C)    : rows_kernel<T_Q5_0, false>(R, C);
        case T_Q4_K:    return G ? rows_kernel<T_Q4_K, true>(R, C)    : rows_kernel<T_Q4_K, false>(R, C);
        case T_Q5_K:    return G ? rows_kernel<T_Q5_K, true>(R, C)    : rows_kernel<T_Q5_K, false>(R, C);
        case T_IQ4_XS:  return G ? rows_kernel<T_IQ4_XS, true>(R, C)  : rows_kernel<T_IQ4_XS, false>(R, C);
        case T_Q6_K:    return nullptr;
    }
    return nullptr;
}

// (f) instances whose shared memory (NW warps x 32 lanes x R rows x C blocks, twice with the gate except for Q6_K, which
// stages gate and up one after the other) fits in 48 KB; SU =
// bytes per staging load (16, or 8 and 4 for Q6_K's rows that are only 8- or 4-byte aligned)
template <int T, int R, int C, bool G, int SU = 16>
const void * st_instance() {
    if constexpr (NW * (32 * R * (G && T != T_Q6_K ? 2 : 1) * C * traits<T>::bs + (T == T_Q6_K ? 16 : 0)) <= 48 * 1024) {
        return (const void *) pascal::mmvq_staged<T, R, C, NW, G, SU>;
    }
    return nullptr;
}

template <int T, bool G>
const void * st_kernel(int R, int C, int SU) {
    if constexpr (T == T_Q6_K) {
        if (R == 1 && C == 1) return SU == 16 ? st_instance<T, 1, 1, G, 16>() : SU == 8 ? st_instance<T, 1, 1, G, 8>() : st_instance<T, 1, 1, G, 4>();
        if (R == 2 && C == 1) return SU == 16 ? st_instance<T, 2, 1, G, 16>() : SU == 8 ? st_instance<T, 2, 1, G, 8>() : st_instance<T, 2, 1, G, 4>();
    } else if constexpr (traits<T>::qk == QK_K) {
        if (SU != 16) return nullptr;
        if (R == 1 && C == 1) return st_instance<T, 1, 1, G>();
        if (R == 2 && C == 1) return st_instance<T, 2, 1, G>();
    } else {
        if (SU != 16) return nullptr;
        if (R == 1 && C == 2) return st_instance<T, 1, 2, G>();
        if (R == 2 && C == 2) return st_instance<T, 2, 2, G>();
        if (R == 4 && C == 2) return st_instance<T, 4, 2, G>();
        if (R == 2 && C == 4) return st_instance<T, 2, 4, G>();
    }
    return nullptr;
}

const void * st_kernel_rt(int T, bool G, int R, int C, int SU) {
    switch (T) {
        case T_Q4_0: return G ? st_kernel<T_Q4_0, true>(R, C, SU) : st_kernel<T_Q4_0, false>(R, C, SU);
        case T_Q4_1: return G ? st_kernel<T_Q4_1, true>(R, C, SU) : st_kernel<T_Q4_1, false>(R, C, SU);
        case T_Q8_0: return G ? st_kernel<T_Q8_0, true>(R, C, SU) : st_kernel<T_Q8_0, false>(R, C, SU);
        case T_Q5_0: return G ? st_kernel<T_Q5_0, true>(R, C, SU) : st_kernel<T_Q5_0, false>(R, C, SU);
        case T_Q4_K: return G ? st_kernel<T_Q4_K, true>(R, C, SU) : st_kernel<T_Q4_K, false>(R, C, SU);
        case T_Q5_K: return G ? st_kernel<T_Q5_K, true>(R, C, SU) : st_kernel<T_Q5_K, false>(R, C, SU);
        case T_IQ4_XS: return G ? st_kernel<T_IQ4_XS, true>(R, C, SU) : st_kernel<T_IQ4_XS, false>(R, C, SU);
        case T_Q6_K: return G ? st_kernel<T_Q6_K, true>(R, C, SU) : st_kernel<T_Q6_K, false>(R, C, SU);
    }
    return nullptr;
}

template <int T, bool G>
const void * rl_kernel(int R, int C) {
    if (R == 1 && C == 2) return (const void *) pascal::mmvq_rowplanes<T, 1, 2, NW, G>;
    if (R == 2 && C == 2) return (const void *) pascal::mmvq_rowplanes<T, 2, 2, NW, G>;
    if (R == 4 && C == 2) return (const void *) pascal::mmvq_rowplanes<T, 4, 2, NW, G>;
    if (R == 2 && C == 4) return (const void *) pascal::mmvq_rowplanes<T, 2, 4, NW, G>;
    return nullptr;
}

const void * rl_kernel_rt(int T, bool G, int R, int C) {
    switch (T) {
        case T_Q8_0: return G ? rl_kernel<T_Q8_0, true>(R, C) : rl_kernel<T_Q8_0, false>(R, C);
        case T_Q4_1: return G ? rl_kernel<T_Q4_1, true>(R, C) : rl_kernel<T_Q4_1, false>(R, C);
        case T_Q5_0: return G ? rl_kernel<T_Q5_0, true>(R, C) : rl_kernel<T_Q5_0, false>(R, C);
    }
    return nullptr;
}

void repack_rl_rt(int T, const uint8_t * src, uint8_t * dst, size_t nrows_total, int nb, cudaStream_t s) {
    const size_t n = nrows_total * nb;
    const unsigned g = (unsigned) ((n + 255) / 256);
    switch (T) {
        case T_Q8_0: pascal::repack_rowplanes<T_Q8_0><<<g, 256, 0, s>>>(src, dst, nrows_total, nb); break;
        case T_Q4_1: pascal::repack_rowplanes<T_Q4_1><<<g, 256, 0, s>>>(src, dst, nrows_total, nb); break;
        case T_Q5_0: pascal::repack_rowplanes<T_Q5_0><<<g, 256, 0, s>>>(src, dst, nrows_total, nb); break;
    }
}

const void * soa_kernel_rt(int S, bool G, int R) {
#define MB_SOA(S_, R_) if (S == S_ && R == R_) return G ? (const void *) pascal::mmvq_soa<S_, R_, NW, true> : (const void *) pascal::mmvq_soa<S_, R_, NW, false>;
    MB_SOA(1, 2) MB_SOA(1, 4) MB_SOA(2, 2) MB_SOA(2, 4)
#undef MB_SOA
    return nullptr;
}

template <int table>
llama::launch_shape llama_plan_rt(int T, const llama::call_args & a) {
    switch (T) {
        case T_Q4_0:    return llama::plan<T_Q4_0, table>(a);
        case T_Q4_1:    return llama::plan<T_Q4_1, table>(a);
        case T_Q8_0:    return llama::plan<T_Q8_0, table>(a);
        case T_Q5_0:    return llama::plan<T_Q5_0, table>(a);
        case T_Q4_K:    return llama::plan<T_Q4_K, table>(a);
        case T_Q5_K:    return llama::plan<T_Q5_K, table>(a);
        case T_IQ4_XS:  return llama::plan<T_IQ4_XS, table>(a);
        case T_Q6_K:    return llama::plan<T_Q6_K, table>(a);
        default:        return llama::plan<T_Q4_1_64, table>(a);
    }
}

template <int table>
void llama_launch_rt(int T, const llama::call_args & a, cudaStream_t s) {
    switch (T) {
        case T_Q4_0:    llama::launch<T_Q4_0, table>(a, s); break;
        case T_Q4_1:    llama::launch<T_Q4_1, table>(a, s); break;
        case T_Q8_0:    llama::launch<T_Q8_0, table>(a, s); break;
        case T_Q5_0:    llama::launch<T_Q5_0, table>(a, s); break;
        case T_Q4_K:    llama::launch<T_Q4_K, table>(a, s); break;
        case T_Q5_K:    llama::launch<T_Q5_K, table>(a, s); break;
        case T_IQ4_XS:  llama::launch<T_IQ4_XS, table>(a, s); break;
        case T_Q6_K:    llama::launch<T_Q6_K, table>(a, s); break;
        default:        llama::launch<T_Q4_1_64, table>(a, s); break;
    }
}

// ---- small helpers ---------------------------------------------------------------------------------------------

struct dev_buf {
    void * p = nullptr;
    size_t n = 0;
    dev_buf() = default;
    // cudaMemset runs on the legacy default stream, which does not order with the non-blocking work stream: wait for it.
    explicit dev_buf(size_t bytes) : n(bytes) {
        CUDA_CHECK(cudaMalloc(&p, bytes + 256));
        CUDA_CHECK(cudaMemset(p, 0, bytes + 256));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    ~dev_buf() { if (p) cudaFree(p); }
    dev_buf(const dev_buf &) = delete;
    dev_buf & operator=(const dev_buf &) = delete;
    dev_buf(dev_buf && o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    dev_buf & operator=(dev_buf && o) noexcept { std::swap(p, o.p); std::swap(n, o.n); return *this; }
    template <typename X> X * as() const { return (X *) p; }
};

std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> out;
    size_t a = 0;
    while (a <= s.size()) {
        size_t b = s.find(',', a);
        if (b == std::string::npos) b = s.size();
        if (b > a) out.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return out;
}

bool contains(const std::vector<std::string> & v, const std::string & x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n == 0 ? 0.0 : (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
}

__global__ void k_arch(int * out) {
#ifdef __CUDA_ARCH__
    *out = __CUDA_ARCH__;
#endif
}

__global__ void k_empty() {}

// Plain global loads (no read-only/texture path), for comparison with k_read's __ldg loads.
__global__ void k_read_plain(const uint4 * p, size_t n, unsigned * out) {
    unsigned acc = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint4 v;
        asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) {
        out[0] = acc;
    }
}

__global__ void k_read(const uint4 * __restrict__ p, size_t n, unsigned * out) {
    unsigned acc = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = __ldg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) {
        out[0] = acc; // practically never; keeps the loads alive
    }
}

// --spread: at the start of every graph launch, copy the next id slot of the pool (n ints: one set of expert ids per
// call of the graph) into the ids buffer the calls read, so every call of every launch picks fresh experts. Slot 0 is
// what the buffer holds before the first launch (the correctness check uses it).
__global__ void k_ids_refresh(const int * __restrict__ pool, int * __restrict__ buf, unsigned * counter, const int n, const int nslots) {
    __shared__ unsigned slot;
    if (threadIdx.x == 0) {
        slot = (*counter + 1) % nslots;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        buf[i] = pool[(size_t) slot * n + i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *counter = slot;
    }
}

// Run `launch(stream, j)` for j = 0..batch-1 as one CUDA graph, after `prologue` if given; time `reps` launches of the
// graph. Returns per-call times in microseconds (graph time / batch).
std::vector<double> time_graph(const std::function<void(cudaStream_t, int)> & launch, int batch, int reps, cudaStream_t s,
                               const std::function<void(cudaStream_t)> & prologue = nullptr) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    if (prologue) {
        prologue(s);
    }
    for (int j = 0; j < batch; ++j) {
        launch(s, j);
    }
    CUDA_CHECK(cudaStreamEndCapture(s, &g));
    CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
    for (int w = 0; w < 3; ++w) {
        CUDA_CHECK(cudaGraphLaunch(ge, s));
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventRecord(e0, s));
        CUDA_CHECK(cudaGraphLaunch(ge, s));
        CUDA_CHECK(cudaEventRecord(e1, s));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        t.push_back(1000.0 * ms / batch);
    }
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaGraphExecDestroy(ge));
    CUDA_CHECK(cudaGraphDestroy(g));
    return t;
}

// ---- results ---------------------------------------------------------------------------------------------------

struct result {
    std::string shape, type, variant;
    int K = 0, N = 0, experts = 0, used = 0, calls = 0;
    double bytes = 0;            // weight bytes read per call
    double us_med = 0, us_min = 0, us_max = 0;
    bool timed = false;
    bool small_k = false;        // llama.cpp's small-K launch shape (variant llama)
    double err = 0;              // max |gpu - ref| / bound
    bool pass = false;
    int regs = 0, blocks_per_sm = 0, threads = 0;
    long grid = 0;
    int ncopies = 1;             // weight sets cycled call by call
    double set_bytes = 0;        // bytes of all weight sets of the shape (the footprint the calls sweep)
};

struct options {
    std::vector<std::string> types = { "q4_1", "q4_0", "q8_0", "q4_1_64", "q5_0", "q5_K", "iq4_xs", "q6_K", "q4_K" };
    std::vector<std::string> variants;
    std::vector<std::string> shapes;
    int reps = 30;
    int batch = 64;
    bool check_only = false;
    double ref_gbs = 173.0;
    std::string csv;
    uint64_t seed = 1234;
    int spread = 0;              // weight sets per shape, cycled call by call, with fresh expert ids per call
    double spread_gb = 0.0;      // or as many sets as fill this many GB
    double spread_max_gb = 5.0;
    bool tlb_probe = false;
    double tlb_max_gb = 5.0;
};

void usage() {
    printf("usage: mmvq_pascal [options]\n"
           "  --types LIST     weight types, comma-separated (default q4_1,q4_0,q8_0,q4_1_64,q5_0,q5_K,iq4_xs,q6_K,q4_K)\n"
           "  --variants LIST  kernel variants (default all; see --list)\n"
           "  --shapes LIST    shapes (default all; see --list)\n"
           "  --reps N         timed graph launches per kernel, median reported (default 30)\n"
           "  --batch N        kernel calls per CUDA graph (default 64; 8 for the output head)\n"
           "  --check-only     correctness checks only, no timing\n"
           "  --verbose        print the largest deviations found by the correctness check\n"
           "  --ref-gbs X      bandwidth for the %%-of-peak column (default 173)\n"
           "  --csv FILE       also write all results as CSV\n"
           "  --seed N         random seed (default 1234)\n"
           "  --spread N       N weight sets per shape (each holding all experts of a MoE matrix), cycled call by call,\n"
           "                   with fresh random expert ids for every call; the sets stop at --spread-max-gb (default 5)\n"
           "                   and the free device memory. Without it: one set for MoE shapes (64 replayed id sets) and\n"
           "                   >= 64 MB, <= 1.2 GB of copies for dense shapes\n"
           "  --spread-gb X    as --spread, with as many sets as fill X GB (at least one, at most 4096 per shape), so\n"
           "                   that small and large shapes sweep the same footprint\n"
           "  --tlb-probe      pointer-chase probe of the address translation instead of the matrix-vector kernels\n"
           "                   (2 MiB and 32 MiB strides over growing footprints, random 2 MiB pages; alone and under load)\n"
           "  --tlb-max-gb X   largest chase buffer (default 5); --check-only walks every chain once, untimed\n"
           "  --list           list shapes and variants\n");
}

void list_all() {
    printf("shapes (K -> N, calls per Flint token):\n");
    for (const auto & s : SHAPES) {
        printf("  %-14s %6d -> %-6d x%-3d %s\n", s.name, s.K, s.N, s.calls, s.what);
    }
    printf("variants:\n");
    for (const auto & v : VARIANTS) {
        printf("  %-8s %s\n", v.name, v.what);
    }
}

int type_from_name(const std::string & n) {
    for (int t = 0; t < T_COUNT; ++t) {
        if (n == type_name(t)) return t;
    }
    return -1;
}

// ---- one (shape, type) case --------------------------------------------------------------------------------------

struct case_data {
    const shape_def * sh = nullptr;
    int type = 0;
    int K = 0, N = 0, nexp = 1, nused = 1, nvec = 1, nchan = 1;
    size_t rbytes = 0, mat_bytes = 0;   // row bytes, bytes of one matrix (all experts)
    int ncopies = 1, nidsets = 1;
    bool glu = false, moe = false;
    std::vector<dev_buf> w_up, w_gate;  // per copy (with --spread: one buffer per matrix holding all sets back to back)
    std::vector<uint8_t *> up, gate;    // the weights of each copy
    dev_buf x, y_q81, a_q, a_ds, ids, dst;
    std::vector<int> ids_host;          // nidsets * nused
    double bytes_per_call = 0;
    // reference
    std::vector<int> check_rows;        // rows checked (dense); all rows for MoE
    std::vector<uint8_t> w_host_up, w_host_gate; // checked rows (dense) or the selected experts (MoE), contiguous
};

void fetch_rows(const case_data & c, const uint8_t * buf, std::vector<uint8_t> & host) {
    if (c.moe) {
        host.resize((size_t) c.nused * c.N * c.rbytes);
        for (int j = 0; j < c.nused; ++j) {
            const int e = c.ids_host[j];
            CUDA_CHECK(cudaMemcpy(host.data() + (size_t) j * c.N * c.rbytes, buf + (size_t) e * c.N * c.rbytes,
                                  (size_t) c.N * c.rbytes, cudaMemcpyDeviceToHost));
        }
        return;
    }
    host.resize(c.check_rows.size() * c.rbytes);
    if ((int) c.check_rows.size() == c.N) {
        CUDA_CHECK(cudaMemcpy(host.data(), buf, (size_t) c.N * c.rbytes, cudaMemcpyDeviceToHost));
        return;
    }
    // evenly strided sample plus the last row
    const int stride = c.check_rows.size() > 1 ? c.check_rows[1] - c.check_rows[0] : 1;
    const size_t n_strided = c.check_rows.size() - 1;
    CUDA_CHECK(cudaMemcpy2D(host.data(), c.rbytes, buf, c.rbytes * stride, c.rbytes, n_strided, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host.data() + n_strided * c.rbytes, buf + (size_t) c.check_rows.back() * c.rbytes,
                          c.rbytes, cudaMemcpyDeviceToHost));
}

// Reference outputs for one activation layout. For GLU shapes: up, gate and their abs-sums.
struct ref_out {
    std::vector<double> up, upabs, gate, gateabs;
};

ref_out compute_ref(const case_data & c, const act_host & a) {
    ref_out r;
    const int nrows = c.moe ? c.nused * c.N : (int) c.check_rows.size();
    r.up.resize(nrows);
    r.upabs.resize(nrows);
    if (c.glu) {
        r.gate.resize(nrows);
        r.gateabs.resize(nrows);
    }
    for (int i = 0; i < nrows; ++i) {
        const int slot = c.moe ? i / c.N : 0;
        const int vec = c.sh->kind == S_MOE_DOWN ? slot : 0;
        r.up[i] = ref_dot(c.type, c.w_host_up.data() + (size_t) i * c.rbytes, a, vec, &r.upabs[i]);
        if (c.glu) {
            r.gate[i] = ref_dot(c.type, c.w_host_gate.data() + (size_t) i * c.rbytes, a, vec, &r.gateabs[i]);
        }
    }
    return r;
}

// Compare the GPU output (dst layout [nchan][N]) with the reference; returns max |err| / bound.
double compare(const case_data & c, const std::vector<float> & out, const ref_out & r, double tol) {
    double worst = 0.0;
    const int nrows = (int) r.up.size();
    for (int i = 0; i < nrows; ++i) {
        size_t oi;
        if (c.moe) {
            oi = (size_t) i; // slot-major [nused][N]
        } else {
            oi = (size_t) c.check_rows[i];
        }
        const double g = out[oi];
        double ref, bound;
        if (c.glu) {
            const double gv = r.gate[i];
            const double sg = gv / (1.0 + exp(-gv));
            const double dsg = 1.1; // |d silu / dx| <= 1.1
            ref = r.up[i] * sg;
            bound = tol * (r.upabs[i] * fabs(sg) + fabs(r.up[i]) * dsg * r.gateabs[i]) + 1e-5 * fabs(ref) + 1e-30;
        } else {
            ref = r.up[i];
            bound = tol * r.upabs[i] + 1e-30;
        }
        const double e = std::isfinite(g) ? fabs(g - ref) / bound : 1e30;
        if (e > worst && g_verbose) {
            printf("    worst so far: output %zu gpu %.9g ref %.9g bound %.3g\n", oi, g, ref, bound);
        }
        worst = std::max(worst, e);
    }
    return worst;
}

} // namespace

int main(int argc, char ** argv) {
    options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); exit(1); }
            return argv[++i];
        };
        if (a == "--types") opt.types = split(next());
        else if (a == "--variants") opt.variants = split(next());
        else if (a == "--shapes") opt.shapes = split(next());
        else if (a == "--reps") opt.reps = std::max(1, atoi(next().c_str()));
        else if (a == "--batch") opt.batch = std::max(1, atoi(next().c_str()));
        else if (a == "--check-only") opt.check_only = true;
        else if (a == "--verbose") g_verbose = true;
        else if (a == "--ref-gbs") opt.ref_gbs = atof(next().c_str());
        else if (a == "--csv") opt.csv = next();
        else if (a == "--seed") opt.seed = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--spread") opt.spread = std::max(0, atoi(next().c_str()));
        else if (a == "--spread-gb") opt.spread_gb = atof(next().c_str());
        else if (a == "--spread-max-gb") opt.spread_max_gb = atof(next().c_str());
        else if (a == "--tlb-probe") opt.tlb_probe = true;
        else if (a == "--tlb-max-gb") opt.tlb_max_gb = atof(next().c_str());
        else if (a == "--list") { list_all(); return 0; }
        else { usage(); return a == "--help" || a == "-h" ? 0 : 1; }
    }
    if (opt.variants.empty()) for (const auto & v : VARIANTS) opt.variants.push_back(v.name);
    if (opt.shapes.empty()) for (const auto & s : SHAPES) opt.shapes.push_back(s.name);
    for (const auto & t : opt.types) {
        if (type_from_name(t) < 0) { fprintf(stderr, "unknown type %s\n", t.c_str()); return 1; }
    }

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int code_arch = 0;
    {
        dev_buf b(sizeof(int));
        k_arch<<<1, 1>>>(b.as<int>());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&code_arch, b.p, sizeof(int), cudaMemcpyDeviceToHost));
    }
    int clock_khz = 0, mem_khz = 0, bus_bits = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
    // llama.cpp computes Q4_1 scale products in FP16 (FAST_FP16_AVAILABLE) on every arch from 6.0 up except 6.1.
    const bool llama_fast_fp16 = code_arch >= 600 && code_arch != 610;
    printf("device: %s, compute %d.%d, %d SMs, max clock %.0f MHz, memory %d-bit at %.0f MHz\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, clock_khz / 1000.0, bus_bits, mem_khz / 1000.0);
    printf("device code: compiled for compute %d.%d%s\n", code_arch / 100, (code_arch % 100) / 10,
           code_arch / 10 == prop.major * 10 + prop.minor ? "" : " (JIT from PTX)");

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    if (opt.tlb_probe) {
        const auto res = memsys::tlb_probe((size_t) (opt.tlb_max_gb * (double) (1ull << 30)), opt.check_only, s);
        bool ok = true;
        if (opt.check_only) {
            for (const auto & r : res) ok &= r.ok;
            printf("tlb probe: %zu chains walked once around, %s\n", res.size(), ok ? "all end where they started" : "FAILURES");
        } else {
            printf("\n%-7s %10s %10s %7s %7s %10s %10s %12s %12s\n", "chase", "MiB", "stride", "slots", "steps", "ns/step",
                   "cyc/step", "ns (loaded)", "cyc (loaded)");
            for (const auto & r : res) {
                printf("%-7s %10.0f %10s %7d %7d %10.1f %10.1f %12.1f %12.1f\n", r.kind.c_str(), r.footprint / 1048576.0,
                       r.kind == "stride" ? (std::to_string(r.stride >> 20) + " MiB").c_str() : "random", r.slots, r.steps,
                       r.ns, r.cycles, r.ns_loaded, r.cycles_loaded);
            }
        }
        if (!opt.csv.empty()) {
            FILE * f = fopen(opt.csv.c_str(), "w");
            if (!f) { fprintf(stderr, "cannot write %s\n", opt.csv.c_str()); return 1; }
            fprintf(f, "device,code_arch,chase,footprint_bytes,stride_bytes,slots,steps,ns_per_step,cycles_per_step,"
                       "ns_per_step_loaded,cycles_per_step_loaded,check\n");
            for (const auto & r : res) {
                fprintf(f, "\"%s\",%d,%s,%zu,%zu,%d,%d,%.3f,%.3f,%.3f,%.3f,%s\n", prop.name, code_arch, r.kind.c_str(), r.footprint,
                        r.stride, r.slots, r.steps, r.ns, r.cycles, r.ns_loaded, r.cycles_loaded, r.ok ? "PASS" : "FAIL");
            }
            fclose(f);
            printf("\nCSV written to %s\n", opt.csv.c_str());
        }
        CUDA_CHECK(cudaStreamDestroy(s));
        return ok ? 0 : 2;
    }

    // Reference points: graph node overhead and streaming read bandwidth. Streaming for ~2 s first also brings the
    // GPU to its sustained clock before anything is timed.
    double node_us = 0.0, read_gbs = 0.0, read_plain_gbs = 0.0;
    if (!opt.check_only) {
        const size_t nbytes = (size_t) 256 << 20;
        dev_buf rb(nbytes), ro(64);
        const int grid = prop.multiProcessorCount * 16;
        auto rd = [&](cudaStream_t st, int) { k_read<<<grid, 256, 0, st>>>(rb.as<uint4>(), nbytes / 16, ro.as<unsigned>()); };
        auto rdp = [&](cudaStream_t st, int) { k_read_plain<<<grid, 256, 0, st>>>(rb.as<uint4>(), nbytes / 16, ro.as<unsigned>()); };
        double warm_ms = 0.0;
        while (warm_ms < 2000.0) {
            const auto t = time_graph(rd, 8, 4, s);
            for (double x : t) warm_ms += x * 8 / 1000.0;
        }
        node_us = median(time_graph([](cudaStream_t st, int) { k_empty<<<1, 32, 0, st>>>(); }, 64, 20, s));
        read_gbs = nbytes / (median(time_graph(rd, 4, 10, s)) * 1e3);
        read_plain_gbs = nbytes / (median(time_graph(rdp, 4, 10, s)) * 1e3);
        printf("empty kernel in a CUDA graph: %.2f us per node\n", node_us);
        printf("streaming read of 256 MB with 128-bit loads: %.1f GB/s read-only path (__ldg), %.1f GB/s plain loads\n", read_gbs, read_plain_gbs);
    }
    printf("\n%-14s %-8s %-8s %10s %9s %9s %7s %6s %5s %4s %7s %9s %s\n", "shape", "type", "variant", "MB/call", "us/call",
           "us(min)", "GB/s", "%peak", "regs", "occ", "grid", "err/tol", "check");

    std::vector<result> results;
    bool any_fail = false;

    for (const auto & sh : SHAPES) {
        if (!contains(opt.shapes, sh.name)) continue;
        for (const auto & tname : opt.types) {
            const int T = type_from_name(tname);
            case_data c;
            c.sh = &sh;
            c.type = T;
            c.K = sh.K;
            c.N = sh.N;
            c.moe = is_moe(sh.kind);
            c.glu = is_glu(sh.kind);
            c.nexp = c.moe ? sh.n_expert : 1;
            c.nused = c.moe ? sh.n_used : 1;
            c.nchan = c.moe ? sh.n_used : 1;
            c.nvec = sh.kind == S_MOE_DOWN ? sh.n_used : 1;
            c.rbytes = row_bytes(T, c.K);
            c.mat_bytes = (size_t) c.nexp * c.N * c.rbytes;
            const int nmats = c.glu ? 2 : 1;
            c.bytes_per_call = (double) nmats * c.nused * c.N * c.rbytes;
            int batch = c.bytes_per_call > 32e6 ? std::min(opt.batch, 8) : opt.batch;
            int npool = 1; // --spread: id slots of batch sets each
            const bool spread = opt.spread > 0 || opt.spread_gb > 0;
            if (spread) {
                // N full weight sets, within the cap and the free memory; every call of a graph takes the next set
                size_t free_b = 0, total_b = 0;
                CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
                const double set_b = (double) nmats * c.mat_bytes;
                const double cap = std::min(opt.spread_max_gb * 1e9, (double) free_b - 0.4e9);
                const int want = opt.spread_gb > 0 ? std::min(4096, std::max(1, (int) (opt.spread_gb * 1e9 / set_b))) : opt.spread;
                c.ncopies = std::max(1, std::min(want, (int) (cap / set_b)));
                batch = std::max(batch, c.ncopies);
                c.nidsets = batch;
                npool = opt.reps + 8;
            } else if (c.moe) {
                c.ncopies = 1;
                c.nidsets = batch;
            } else {
                // enough copies that consecutive calls never find their weights in the 1.5 MB L2
                const size_t want = (size_t) 64 << 20;
                c.ncopies = (int) std::min<size_t>(batch, std::max<size_t>(2, (want + nmats * c.mat_bytes - 1) / (nmats * c.mat_bytes)));
                while (c.ncopies > 1 && (double) c.ncopies * nmats * c.mat_bytes > 1.2e9) c.ncopies--;
            }

            // weights: one buffer per copy and matrix, or with --spread all copies of a matrix back to back in one buffer,
            // so that N copies take N times the bytes and sweep one contiguous range of 2 MiB pages
            const size_t set_stride = (c.mat_bytes + 255) / 256 * 256;
            if (spread) {
                for (int m = 0; m < nmats; ++m) {
                    (m == 0 ? c.w_up : c.w_gate).push_back(dev_buf(set_stride * c.ncopies));
                }
            }
            for (int cp = 0; cp < c.ncopies; ++cp) {
                for (int m = 0; m < nmats; ++m) {
                    uint8_t * w;
                    if (spread) {
                        w = (m == 0 ? c.w_up : c.w_gate)[0].as<uint8_t>() + (size_t) cp * set_stride;
                    } else {
                        (m == 0 ? c.w_up : c.w_gate).push_back(dev_buf(c.mat_bytes));
                        w = (m == 0 ? c.w_up : c.w_gate).back().as<uint8_t>();
                    }
                    const size_t nblocks = c.mat_bytes / type_bs(T);
                    gen_blocks<<<prop.multiProcessorCount * 32, 256, 0, s>>>(w, nblocks, T,
                        opt.seed * 1000003ull + (uint64_t) (&sh - SHAPES) * 101 + T * 13 + cp * 7 + m);
                    CUDA_CHECK(cudaGetLastError());
                    (m == 0 ? c.up : c.gate).push_back(w);
                }
            }
            // activations: float input, llama Q8_1, SoA int8 + float2
            const size_t nx = (size_t) c.K * c.nvec;
            c.x = dev_buf(nx * sizeof(float));
            gen_floats<<<64, 256, 0, s>>>(c.x.as<float>(), nx, opt.seed * 7919ull + (uint64_t) (&sh - SHAPES));
            CUDA_CHECK(cudaGetLastError());
            c.y_q81 = dev_buf(nx / 32 * sizeof(block_q8_1));
            llama::quantize_row_q8_1_cuda(c.x.as<float>(), c.y_q81.p, c.K, c.nvec, s);
            CUDA_CHECK(cudaGetLastError());
            c.a_q = dev_buf(nx);
            c.a_ds = dev_buf(nx / 32 * sizeof(float2));
            pascal::quantize_soa<<<(unsigned) ((nx + 255) / 256), 256, 0, s>>>(c.x.as<float>(), c.a_q.as<int8_t>(), c.a_ds.as<float2>(), c.K, c.nvec);
            CUDA_CHECK(cudaGetLastError());
            // expert ids: nidsets sets of nused distinct experts (npool times as many with --spread: slot 0 goes into the
            // ids buffer, the others are copied there one slot per graph launch)
            dev_buf ids_pool, ids_counter;
            std::vector<int> pool_host;
            if (c.moe) {
                uint64_t h = opt.seed * 31 + 17;
                for (int st = 0; st < c.nidsets * npool; ++st) {
                    std::vector<int> pick;
                    while ((int) pick.size() < c.nused) {
                        h = h * 6364136223846793005ull + 1442695040888963407ull;
                        const int e = (int) ((h >> 33) % c.nexp);
                        if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
                    }
                    c.ids_host.insert(c.ids_host.end(), pick.begin(), pick.end());
                }
                if (npool > 1) {
                    ids_pool = dev_buf(c.ids_host.size() * sizeof(int));
                    CUDA_CHECK(cudaMemcpy(ids_pool.p, c.ids_host.data(), c.ids_host.size() * sizeof(int), cudaMemcpyHostToDevice));
                    ids_counter = dev_buf(sizeof(unsigned));
                    pool_host = c.ids_host;
                    c.ids_host.resize((size_t) c.nidsets * c.nused);
                }
                c.ids = dev_buf(c.ids_host.size() * sizeof(int));
                CUDA_CHECK(cudaMemcpy(c.ids.p, c.ids_host.data(), c.ids_host.size() * sizeof(int), cudaMemcpyHostToDevice));
            }
            std::function<void(cudaStream_t)> prologue;
            if (npool > 1 && c.moe) {
                const int n_ids = c.nidsets * c.nused;
                prologue = [&ids_pool, &ids_counter, &c, n_ids, npool](cudaStream_t st) {
                    k_ids_refresh<<<1, 256, 0, st>>>(ids_pool.as<int>(), c.ids.as<int>(), ids_counter.as<unsigned>(), n_ids, npool);
                };
            }
            if (spread) {
                printf("%-14s %-8s spread: %d weight set%s, %.2f GB swept, %s\n", sh.name, tname.c_str(), c.ncopies,
                       c.ncopies == 1 ? "" : "s", (double) c.ncopies * nmats * c.mat_bytes / 1e9,
                       c.moe ? "fresh expert ids for every call" : "one set per call in turn");
                if (prologue && opt.check_only) {
                    // one refresh must bring slot 1 of the pool into the ids buffer
                    prologue(s);
                    CUDA_CHECK(cudaGetLastError());
                    std::vector<int> got(c.ids_host.size());
                    CUDA_CHECK(cudaMemcpyAsync(got.data(), c.ids.p, got.size() * sizeof(int), cudaMemcpyDeviceToHost, s));
                    CUDA_CHECK(cudaStreamSynchronize(s));
                    const bool ok = std::equal(got.begin(), got.end(), pool_host.begin() + got.size());
                    printf("%-14s %-8s spread: expert id refresh %s\n", sh.name, tname.c_str(), ok ? "PASS" : "FAIL");
                    any_fail |= !ok;
                }
            }
            c.dst = dev_buf((size_t) c.nchan * c.N * sizeof(float));
            CUDA_CHECK(cudaStreamSynchronize(s));

            // rows to check and their weights
            if (!c.moe) {
                if (c.N <= 16384) {
                    for (int r = 0; r < c.N; ++r) c.check_rows.push_back(r);
                } else {
                    for (int r = 0; r < c.N - 1; r += 37) c.check_rows.push_back(r);
                    c.check_rows.push_back(c.N - 1);
                }
            }
            fetch_rows(c, c.up[0], c.w_host_up);
            if (c.glu) fetch_rows(c, c.gate[0], c.w_host_gate);

            // activations as the kernels see them
            std::vector<block_q8_1> yq(nx / 32);
            CUDA_CHECK(cudaMemcpy(yq.data(), c.y_q81.p, yq.size() * sizeof(block_q8_1), cudaMemcpyDeviceToHost));
            const act_host act_q81 = act_from_q8_1(yq, c.K, c.nvec);
            act_host act_soa;
            act_soa.K = c.K;
            act_soa.nvec = c.nvec;
            act_soa.q.resize(nx);
            std::vector<float2> ds(nx / 32);
            CUDA_CHECK(cudaMemcpy(act_soa.q.data(), c.a_q.p, nx, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ds.data(), c.a_ds.p, ds.size() * sizeof(float2), cudaMemcpyDeviceToHost));
            for (const auto & v : ds) { act_soa.d.push_back(v.x); act_soa.s.push_back(v.y); }

            ref_out ref_q81, ref_soa;
            bool have_q81 = false, have_soa = false;

            // SoA planes and row-local planes, built on first use
            std::vector<dev_buf> soa_qs_up, soa_dm_up, soa_qs_gate, soa_dm_gate;
            std::vector<dev_buf> rl_up, rl_gate;
            const int S = T == T_Q4_1_64 ? 2 : 1;
            const int ngroups = c.K / (32 * S);

            for (const auto & vname : opt.variants) {
                const variant_def * vd = nullptr;
                for (const auto & v : VARIANTS) if (vname == v.name) vd = &v;
                if (!vd) { fprintf(stderr, "unknown variant %s\n", vname.c_str()); return 1; }
                if (vd->kind == V_SOA && T != T_Q4_1 && T != T_Q4_1_64) continue;
                if (vd->kind == V_RL && T != T_Q8_0 && T != T_Q4_1 && T != T_Q5_0) continue;
                if (vd->kind == V_LLAMA_SK && T != T_IQ4_XS) continue;

                result res;
                res.shape = sh.name; res.type = tname; res.variant = vd->name;
                res.K = c.K; res.N = c.N; res.experts = c.moe ? c.nexp : 0; res.used = c.moe ? c.nused : 0;
                res.calls = sh.calls; res.bytes = c.bytes_per_call;
                res.ncopies = c.ncopies; res.set_bytes = (double) c.ncopies * nmats * c.mat_bytes;

                std::function<void(cudaStream_t, int)> launch;
                const void * func = nullptr;
                dim3 grid, block;
                double tol = 1e-5;

                const int qk = type_qk(T);
                if (vd->kind == V_LLAMA || vd->kind == V_LLAMA_SK || vd->kind == V_PR25479) {
                    llama::call_args a;
                    a.vy = c.y_q81.as<block_q8_1>();
                    a.ncols_x = c.K;
                    a.nrows_x = c.N;
                    a.stride_row_x = c.K / qk;
                    a.stride_channel_x = c.N * (c.K / qk);
                    if (c.moe) {
                        a.nchannels_dst = c.nused;
                        a.nchannels_y = c.nvec;                 // 1: broadcast input (gate/up), 6: per-expert input (down)
                        a.stride_col_y = c.nvec * (c.K / 32);   // s12 = ne11 * s11
                        a.stride_channel_y = c.K / 32;          // s11
                        a.stride_col_dst = c.nused * c.N;       // s2
                        a.stride_channel_dst = c.N;             // s1
                    } else {
                        a.stride_col_y = c.K / 32;
                        a.stride_channel_y = c.K / 32;
                        a.stride_col_dst = c.N;
                        a.stride_channel_dst = c.N;
                    }
                    a.dst = c.dst.as<float>();
                    a.iq_small_k = vd->kind == V_LLAMA_SK;
                    const bool pr = vd->kind == V_PR25479;
                    const auto ls = pr ? llama_plan_rt<llama::MMVQ_PARAMETERS_PASCAL_DP4A>(T, a) : llama_plan_rt<llama::MMVQ_PARAMETERS_GENERIC>(T, a);
                    func = ls.func; grid = ls.grid; block = ls.block;
                    res.small_k = !pr && grid.x != (unsigned) c.N;
                    launch = [&c, a, pr, T](cudaStream_t st, int j) {
                        llama::call_args b = a;
                        const int cp = j % c.ncopies;
                        b.vx = c.up[cp];
                        b.vgate = c.glu ? c.gate[cp] : nullptr;
                        b.ids = c.moe ? c.ids.as<int>() + (j % c.nidsets) * c.nused : nullptr;
                        if (pr) llama_launch_rt<llama::MMVQ_PARAMETERS_PASCAL_DP4A>(T, b, st);
                        else    llama_launch_rt<llama::MMVQ_PARAMETERS_GENERIC>(T, b, st);
                    };
                    if (llama_fast_fp16 && (T == T_Q4_1 || T == T_Q4_1_64)) tol = 2e-3;
                } else if (vd->kind == V_ROWS || vd->kind == V_ST) {
                    const int blocks = c.K / qk;
                    const int chunks = blocks / vd->C;
                    if (chunks < 1 || blocks % vd->C) continue;
                    int lpr = 1;
                    while (lpr < 32 && lpr * 2 <= chunks) lpr *= 2;
                    if (chunks % lpr) continue;
                    // staging loads: the widest of 16, 8, 4 bytes that divides the segment and the row (8 and 4: Q6_K only)
                    int su = 0;
                    if (vd->kind == V_ST) {
                        const size_t seg = (size_t) lpr * vd->C * type_bs(T);
                        for (int u : { 16, 8, 4 }) {
                            if (seg % u == 0 && c.rbytes % u == 0) { su = u; break; }
                        }
                        if (su == 0 || (su != 16 && T != T_Q6_K)) continue;
                    }
                    int lpr_log2 = 0;
                    while ((1 << lpr_log2) < lpr) lpr_log2++;
                    pascal::args p;
                    p.y = c.y_q81.as<uint8_t>();
                    p.dst = c.dst.as<float>();
                    p.nrows = c.N;
                    p.row_bytes = (int) c.rbytes;
                    p.expert_bytes = c.moe ? (size_t) c.N * c.rbytes : 0;
                    p.y_chan_bytes = sh.kind == S_MOE_DOWN ? (c.K / 32) * 36 : 0;
                    p.dst_chan_stride = c.N;
                    p.lpr_log2 = lpr_log2;
                    p.niter = chunks / lpr;
                    func = vd->kind == V_ST ? st_kernel_rt(T, c.glu, vd->R, vd->C, su) : rows_kernel_rt(T, c.glu, vd->R, vd->C);
                    const int rows_per_block = NW * (32 / lpr) * vd->R;
                    grid = dim3((c.N + rows_per_block - 1) / rows_per_block, c.nchan, 1);
                    block = dim3(32, NW, 1);
                    launch = [&c, p, func, grid, block](cudaStream_t st, int j) {
                        pascal::args q = p;
                        const int cp = j % c.ncopies;
                        q.w = c.up[cp];
                        q.wg = c.glu ? c.gate[cp] : nullptr;
                        q.ids = c.moe ? c.ids.as<int>() + (j % c.nidsets) * c.nused : nullptr;
                        void * params[] = { &q };
                        CUDA_CHECK(cudaLaunchKernel(func, grid, block, params, 0, st));
                    };
                } else if (vd->kind == V_RL) {
                    const int blocks = c.K / qk;
                    const int chunks = blocks / vd->C;
                    if (chunks < 1 || blocks % vd->C) continue;
                    int lpr = 1;
                    while (lpr < 32 && lpr * 2 <= chunks) lpr *= 2;
                    if (chunks % lpr) continue;
                    int lpr_log2 = 0;
                    while ((1 << lpr_log2) < lpr) lpr_log2++;
                    if (rl_up.empty()) {
                        const size_t nrows_total = (size_t) c.nexp * c.N;
                        for (int cp = 0; cp < c.ncopies; ++cp) {
                            for (int m = 0; m < nmats; ++m) {
                                dev_buf b(c.mat_bytes);
                                repack_rl_rt(T, m == 0 ? c.up[cp] : c.gate[cp], b.as<uint8_t>(), nrows_total, blocks, s);
                                CUDA_CHECK(cudaGetLastError());
                                (m == 0 ? rl_up : rl_gate).push_back(std::move(b));
                            }
                        }
                        CUDA_CHECK(cudaStreamSynchronize(s));
                    }
                    pascal::args p;
                    p.y = c.y_q81.as<uint8_t>();
                    p.dst = c.dst.as<float>();
                    p.nrows = c.N;
                    p.row_bytes = (int) c.rbytes;
                    p.expert_bytes = c.moe ? (size_t) c.N * c.rbytes : 0;
                    p.y_chan_bytes = sh.kind == S_MOE_DOWN ? (c.K / 32) * 36 : 0;
                    p.dst_chan_stride = c.N;
                    p.lpr_log2 = lpr_log2;
                    p.niter = chunks / lpr;
                    func = rl_kernel_rt(T, c.glu, vd->R, vd->C);
                    const int rows_per_block = NW * (32 / lpr) * vd->R;
                    grid = dim3((c.N + rows_per_block - 1) / rows_per_block, c.nchan, 1);
                    block = dim3(32, NW, 1);
                    launch = [&c, &rl_up, &rl_gate, p, blocks, func, grid, block](cudaStream_t st, int j) {
                        pascal::args q = p;
                        int nb = blocks;
                        const int cp = j % c.ncopies;
                        q.w = rl_up[cp].as<uint8_t>();
                        q.wg = c.glu ? rl_gate[cp].as<uint8_t>() : nullptr;
                        q.ids = c.moe ? c.ids.as<int>() + (j % c.nidsets) * c.nused : nullptr;
                        void * params[] = { &q, &nb };
                        CUDA_CHECK(cudaLaunchKernel(func, grid, block, params, 0, st));
                    };
                } else { // V_SOA
                    if (soa_qs_up.empty()) {
                        const size_t nrows_total = (size_t) c.nexp * c.N;
                        for (int cp = 0; cp < c.ncopies; ++cp) {
                            for (int m = 0; m < nmats; ++m) {
                                dev_buf qs(nrows_total * c.K / 2), dm(nrows_total * ngroups * 4);
                                const uint8_t * src = m == 0 ? c.up[cp] : c.gate[cp];
                                const size_t nb = nrows_total * ngroups;
                                const unsigned g = (unsigned) ((nb + 255) / 256);
                                if (S == 1) pascal::repack_soa<1><<<g, 256, 0, s>>>(src, qs.as<uint8_t>(), dm.as<uint8_t>(), nrows_total, ngroups);
                                else        pascal::repack_soa<2><<<g, 256, 0, s>>>(src, qs.as<uint8_t>(), dm.as<uint8_t>(), nrows_total, ngroups);
                                CUDA_CHECK(cudaGetLastError());
                                (m == 0 ? soa_qs_up : soa_qs_gate).push_back(std::move(qs));
                                (m == 0 ? soa_dm_up : soa_dm_gate).push_back(std::move(dm));
                            }
                        }
                        CUDA_CHECK(cudaStreamSynchronize(s));
                    }
                    int lpr = 1;
                    while (lpr < 32 && lpr * 2 <= ngroups) lpr *= 2;
                    if (ngroups % lpr) continue;
                    int lpr_log2 = 0;
                    while ((1 << lpr_log2) < lpr) lpr_log2++;
                    pascal::soa_args p;
                    p.aq = c.a_q.as<int8_t>();
                    p.ads = c.a_ds.as<float2>();
                    p.dst = c.dst.as<float>();
                    p.nrows = c.N;
                    p.ngroups = ngroups;
                    p.y_chan_vals = sh.kind == S_MOE_DOWN ? c.K : 0;
                    p.dst_chan_stride = c.N;
                    p.lpr_log2 = lpr_log2;
                    p.niter = ngroups / lpr;
                    func = soa_kernel_rt(S, c.glu, vd->R);
                    const int rows_per_block = NW * (32 / lpr) * vd->R;
                    grid = dim3((c.N + rows_per_block - 1) / rows_per_block, c.nchan, 1);
                    block = dim3(32, NW, 1);
                    launch = [&c, &soa_qs_up, &soa_dm_up, &soa_qs_gate, &soa_dm_gate, p, func, grid, block](cudaStream_t st, int j) {
                        pascal::soa_args q = p;
                        const int cp = j % c.ncopies;
                        q.qs = soa_qs_up[cp].as<uint8_t>();
                        q.dm = soa_dm_up[cp].as<uint8_t>();
                        if (c.glu) {
                            q.gqs = soa_qs_gate[cp].as<uint8_t>();
                            q.gdm = soa_dm_gate[cp].as<uint8_t>();
                        }
                        q.ids = c.moe ? c.ids.as<int>() + (j % c.nidsets) * c.nused : nullptr;
                        void * params[] = { &q };
                        CUDA_CHECK(cudaLaunchKernel(func, grid, block, params, 0, st));
                    };
                }
                if (!func) continue;

                cudaFuncAttributes fa;
                CUDA_CHECK(cudaFuncGetAttributes(&fa, func));
                res.regs = fa.numRegs;
                res.threads = block.x * block.y;
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&res.blocks_per_sm, func, res.threads, 0));
                res.grid = (long) grid.x * grid.y;

                // correctness: one call on copy 0 / id set 0 (--spread: the previous variant's timing moved the ids on)
                if (prologue) {
                    CUDA_CHECK(cudaMemcpyAsync(c.ids.p, c.ids_host.data(), c.ids_host.size() * sizeof(int), cudaMemcpyHostToDevice, s));
                    CUDA_CHECK(cudaMemsetAsync(ids_counter.p, 0, sizeof(unsigned), s));
                }
                CUDA_CHECK(cudaMemsetAsync(c.dst.p, 0xff, c.dst.n, s));
                launch(s, 0);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamSynchronize(s));
                std::vector<float> out((size_t) c.nchan * c.N);
                CUDA_CHECK(cudaMemcpy(out.data(), c.dst.p, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
                const bool soa = vd->kind == V_SOA;
                if (soa && !have_soa) { ref_soa = compute_ref(c, act_soa); have_soa = true; }
                if (!soa && !have_q81) { ref_q81 = compute_ref(c, act_q81); have_q81 = true; }
                res.err = compare(c, out, soa ? ref_soa : ref_q81, tol);
                res.pass = res.err <= 1.0;
                any_fail |= !res.pass;

                if (!opt.check_only) {
                    const auto t = time_graph(launch, batch, opt.reps, s, prologue);
                    res.us_med = median(t);
                    res.us_min = *std::min_element(t.begin(), t.end());
                    res.us_max = *std::max_element(t.begin(), t.end());
                    res.timed = true;
                }
                const double gbs = res.timed ? res.bytes / (res.us_med * 1e3) : 0.0;
                printf("%-14s %-8s %-8s %10.3f %9.2f %9.2f %7.1f %5.1f%% %5d %4d %7ld %9.3f %s\n", res.shape.c_str(), res.type.c_str(),
                       res.variant.c_str(), res.bytes / 1e6, res.us_med, res.us_min, gbs, 100.0 * gbs / opt.ref_gbs, res.regs,
                       res.blocks_per_sm, res.grid, res.err, res.pass ? "PASS" : "FAIL");
                fflush(stdout);
                results.push_back(res);
            }
        }
    }

    // ---- per-token totals ----
    std::vector<std::pair<std::string, std::string>> combos;
    for (const auto & r : results) {
        auto k = std::make_pair(r.type, r.variant);
        if (std::find(combos.begin(), combos.end(), k) == combos.end()) combos.push_back(k);
    }
    struct total { double us = 0, bytes = 0; int nshapes = 0; bool pass = true; };
    std::map<std::pair<std::string, std::string>, total> totals;
    int nshapes_sel = 0;
    for (const auto & sh : SHAPES) if (contains(opt.shapes, sh.name)) nshapes_sel++;
    for (const auto & r : results) {
        auto & t = totals[{ r.type, r.variant }];
        t.us += r.us_med * r.calls;
        t.bytes += r.bytes * r.calls;
        t.nshapes++;
        t.pass &= r.pass;
    }
    // Synthetic row "widek": llama.cpp with the fork's wide-rows launch shape, i.e. per shape the small-K shape where
    // llama.cpp's rule applies (variant llama) and 2 warps elsewhere (variant pr25479).
    for (const auto & tn : opt.types) {
        total t;
        for (const auto & sh : SHAPES) {
            if (!contains(opt.shapes, sh.name)) continue;
            const result * rl = nullptr;
            const result * rp = nullptr;
            for (const auto & r : results) {
                if (r.type != tn || r.shape != sh.name || !r.timed) continue;
                if (r.variant == "llama") rl = &r;
                if (r.variant == "pr25479") rp = &r;
            }
            if (!rl || !rp) continue;
            const result * pick = rl->small_k ? rl : rp;
            t.us += pick->us_med * pick->calls;
            t.bytes += pick->bytes * pick->calls;
            t.pass &= pick->pass;
            t.nshapes++;
        }
        if (t.nshapes == 0) continue;
        totals[{ tn, "widek" }] = t;
        combos.push_back({ tn, "widek" });
    }
    // Synthetic rows: per type and shape, the fastest correct tuned kernel, as a llama.cpp port could pick per shape.
    {
        const char * names[2] = { "best(c)", "best(c-f)" };
        for (int which = 0; which < 2; ++which) {
            for (const auto & tn : opt.types) {
                total t;
                for (const auto & sh : SHAPES) {
                    if (!contains(opt.shapes, sh.name)) continue;
                    const result * best = nullptr;
                    for (const auto & r : results) {
                        if (r.type != tn || r.shape != sh.name || !r.pass || !r.timed) continue;
                        const bool is_c = r.variant.rfind("p_", 0) == 0;
                        const bool is_d = r.variant.rfind("soa_", 0) == 0 || r.variant.rfind("rl_", 0) == 0 || r.variant.rfind("st_", 0) == 0;
                        if (!(is_c || (which == 1 && is_d))) continue;
                        if (!best || r.us_med < best->us_med) best = &r;
                    }
                    if (!best) continue;
                    t.us += best->us_med * best->calls;
                    t.bytes += best->bytes * best->calls;
                    t.nshapes++;
                }
                if (t.nshapes == 0) continue;
                if (which == 1) {
                    bool has_d = false;
                    for (const auto & r : results) has_d |= r.type == tn && (r.variant.rfind("soa_", 0) == 0 || r.variant.rfind("rl_", 0) == 0 || r.variant.rfind("st_", 0) == 0);
                    if (!has_d) continue;
                }
                totals[{ tn, names[which] }] = t;
                combos.push_back({ tn, names[which] });
            }
        }
        std::stable_sort(combos.begin(), combos.end(), [&](const auto & a, const auto & b) {
            auto ia = std::find(opt.types.begin(), opt.types.end(), a.first) - opt.types.begin();
            auto ib = std::find(opt.types.begin(), opt.types.end(), b.first) - opt.types.begin();
            return ia < ib;
        });
    }
    if (!opt.check_only) {
        printf("\nPer-token matvec time: sum over shapes of us/call x calls/token (49 attention layers, 47 MoE layers, one\n"
               "output head; dense layers 0-1, the router and all non-matvec ops are not included).\n"
               "best(c) / best(c-f): fastest correct tuned kernel per shape among (c) / among (c), (d), (e) and (f).\n"
               "widek: llama.cpp's kernel with the wide-rows launch shape (small-K shape where llama.cpp applies it, else 2 warps).\n");
        printf("%-8s %-8s %10s %10s %8s %10s %12s\n", "type", "variant", "ms/token", "MB/token", "GB/s", "vs llama", "vs q4_1 llama");
        const auto base_q41 = totals.find({ "q4_1", "llama" });
        for (const auto & k : combos) {
            const auto & t = totals[k];
            if (t.nshapes != nshapes_sel) {
                printf("%-8s %-8s   (only %d of %d shapes ran)\n", k.first.c_str(), k.second.c_str(), t.nshapes, nshapes_sel);
                continue;
            }
            const auto base = totals.find({ k.first, "llama" });
            char vs[32] = "-", vs41[32] = "-";
            if (base != totals.end() && base->second.nshapes == nshapes_sel) snprintf(vs, sizeof(vs), "%.2fx", base->second.us / t.us);
            if (base_q41 != totals.end() && base_q41->second.nshapes == nshapes_sel) snprintf(vs41, sizeof(vs41), "%.2fx", base_q41->second.us / t.us);
            printf("%-8s %-8s %10.2f %10.1f %8.1f %10s %12s%s\n", k.first.c_str(), k.second.c_str(), t.us / 1e3, t.bytes / 1e6,
                   t.bytes / (t.us * 1e3), vs, vs41, t.pass ? "" : "  (has FAIL)");
        }
    }

    if (!opt.csv.empty()) {
        FILE * f = fopen(opt.csv.c_str(), "w");
        if (!f) { fprintf(stderr, "cannot write %s\n", opt.csv.c_str()); return 1; }
        fprintf(f, "device,code_arch,shape,K,N,experts,used,calls_per_token,type,variant,bytes_per_call,us_median,us_min,us_max,"
                   "gbs,pct_of_ref,ref_gbs,regs,blocks_per_sm,threads_per_block,grid_blocks,err_over_tol,check,weight_sets,"
                   "swept_bytes\n");
        for (const auto & r : results) {
            const double gbs = r.timed ? r.bytes / (r.us_med * 1e3) : 0.0;
            fprintf(f, "\"%s\",%d,%s,%d,%d,%d,%d,%d,%s,%s,%.0f,%.3f,%.3f,%.3f,%.2f,%.2f,%.1f,%d,%d,%d,%ld,%.4f,%s,%d,%.0f\n", prop.name,
                    code_arch, r.shape.c_str(), r.K, r.N, r.experts, r.used, r.calls, r.type.c_str(), r.variant.c_str(), r.bytes, r.us_med,
                    r.us_min, r.us_max, gbs, 100.0 * gbs / opt.ref_gbs, opt.ref_gbs, r.regs, r.blocks_per_sm, r.threads, r.grid, r.err,
                    r.pass ? "PASS" : "FAIL", r.ncopies, r.set_bytes);
        }
        for (const auto & k : combos) {
            const auto & t = totals[k];
            if (t.nshapes != nshapes_sel || opt.check_only) continue;
            fprintf(f, "\"%s\",%d,TOKEN_TOTAL,0,0,0,0,1,%s,%s,%.0f,%.3f,,,%.2f,%.2f,%.1f,,,,,,%s,,\n", prop.name, code_arch, k.first.c_str(),
                    k.second.c_str(), t.bytes, t.us, t.bytes / (t.us * 1e3), 100.0 * t.bytes / (t.us * 1e3) / opt.ref_gbs, opt.ref_gbs,
                    t.pass ? "PASS" : "FAIL");
        }
        if (!opt.check_only) {
            fprintf(f, "\"%s\",%d,GRAPH_NODE_OVERHEAD,0,0,0,0,0,-,-,0,%.3f,,,,,,,,,,,,,\n", prop.name, code_arch, node_us);
            fprintf(f, "\"%s\",%d,STREAM_READ_LDG,0,0,0,0,0,-,-,%zu,,,,%.2f,%.2f,%.1f,,,,,,,,\n", prop.name, code_arch, (size_t) 256 << 20, read_gbs,
                    100.0 * read_gbs / opt.ref_gbs, opt.ref_gbs);
            fprintf(f, "\"%s\",%d,STREAM_READ_PLAIN,0,0,0,0,0,-,-,%zu,,,,%.2f,%.2f,%.1f,,,,,,,,\n", prop.name, code_arch, (size_t) 256 << 20,
                    read_plain_gbs, 100.0 * read_plain_gbs / opt.ref_gbs, opt.ref_gbs);
        }
        fclose(f);
        printf("\nCSV written to %s\n", opt.csv.c_str());
    }

    printf("\ncorrectness: %s\n", any_fail ? "FAILURES (see FAIL rows)" : "all variants match the CPU reference");
    CUDA_CHECK(cudaStreamDestroy(s));
    return any_fail ? 2 : 0;
}
