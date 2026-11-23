// C962 (Q1692): does the card's top-k give its candidates in the same order every time?
//
// The backend sampling chain of a request at top-k 8, top-p 1, min-p 0 and a temperature is top-k -> temperature ->
// dist. The dist step takes the candidates in the order the top-k op wrote them, makes their probabilities, sums them up
// in that order and picks the first candidate whose running sum passes the draw u. So with one u, the pick depends on
// the order. For one row of a vocabulary wider than 4,096 columns, ggml's CUDA top-k (built with CUDA 12.x, whose CCCL
// has no DeviceTopK) takes the radix path, whose gather kernel writes each candidate above the threshold at the slot an
// atomicAdd hands out: the order is the order in which the blocks reach that atomic.
//
// This probe runs that path, the four kernels and the host function copied verbatim from the fork's
// ggml/src/ggml-cuda/top-k.cu at 3dfad876f (llama.cpp, MIT licence: Copyright (c) 2023-2026 The ggml authors), with
// small stand-ins for ggml's pool and warp helpers, on one fixed row, many times, and counts the orders it gets. For
// each order it replays the dist step on the host with the draw the fork keys by (seed, position)
// (llama_sampler_pos_seed + std::mt19937 + uniform_real_distribution<double>, as src/llama-sampler.cpp) and names the
// token picked.
//
// Input file: a line "n_vocab k seed pos temp", then k lines "token_id probability" (the candidates and the
// probabilities the server reported for one position with n_probs and post_sampling_probs). The row: every other column
// a fixed pseudo-random logit in [-4, 4); each candidate 30 + temp*ln(p), so the top-k picks exactly those k columns
// and the dist step's probabilities are p.
// Build: nvcc -O2 -std=c++17 -gencode arch=compute_61,code=compute_61 -o topk_order_probe topk_order_probe.cu
// (PTX only, as the rig build: newer cards run it through the driver's JIT)
// Run:   topk_order_probe <input file> <iterations> [device] [mode]
// mode:  plain (default): the top-k alone on the stream, call after call
//        rewrite: before each call a kernel on the same stream writes the row again (as the output layer writes the
//                 logits just before the sampling chain reads them)
//        busy: a spinning kernel on a second stream occupies half of the card's SMs during each call
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <cub/version.cuh>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// stand-ins for ggml-cuda/common.cuh (NVIDIA, warp size 32)
typedef uint32_t ggml_cuda_thread_mask_t;
static constexpr __device__ int ggml_cuda_get_physical_warp_size() { return 32; }
static __device__ __forceinline__ ggml_cuda_thread_mask_t ggml_cuda_ballot(int predicate) {
    return __ballot_sync(0xffffffffu, predicate);
}
template<int width = 32>
static __device__ __forceinline__ int warp_reduce_sum(int x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, offset, width);
    }
    return x;
}
struct ggml_cuda_pool {};
template<typename T> struct ggml_cuda_pool_alloc {
    T * ptr = nullptr;
    ggml_cuda_pool_alloc(ggml_cuda_pool &, size_t n) { CUDA_CHECK(cudaMalloc(&ptr, n*sizeof(T))); }
    ~ggml_cuda_pool_alloc() { cudaFree(ptr); }
    T * get() { return ptr; }
};
static int64_t ggml_cuda_chunk_nrows(size_t, int64_t nrows) { return nrows; }

// ---- copied verbatim from ggml/src/ggml-cuda/top-k.cu at 3dfad876f (lines 85-297) ----
static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

// one state per cache line, so the gather atomics of different rows do not compete
struct alignas(128) top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int block_cols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    const int64_t col_begin = (int64_t) row_block * block_cols;
    const int64_t col_end   = min(col_begin + block_cols, (int64_t) ncols);
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int64_t col = col_begin + tid; col < col_end; col += BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int block_cols,
        int k,
        int blocks_per_row) {
    constexpr int NBINS     = 1 << RADIX_BITS;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = BLOCK_SIZE / warp_size;
    const auto popc = [](const ggml_cuda_thread_mask_t mask) {
        if constexpr (warp_size == 64) {
            return __popcll(mask);
        } else {
            return __popc(mask);
        }
    };
    __shared__ int warp_ties[nwarps];

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];
    const uint32_t threshold = state->prefix;
    const int rank = state->rank;

    // each warp reads a contiguous part of the contiguous range of the block
    const int     warp_cols  = block_cols / nwarps;
    const int64_t warp_begin = (int64_t) row_block * block_cols + warp * warp_cols;
    const int64_t warp_end   = min(warp_begin + warp_cols, (int64_t) ncols);

    int nties = 0;
    for (int64_t col = warp_begin + lane; col < warp_end; col += warp_size) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > threshold) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == threshold) {
            nties++;
        }
    }

    // the ties fill the last rank slots in column order, the lowest columns win.
    // the histograms of the last pass count the ties of each block in the bin of the threshold.
    const int * block_ties = block_histograms + (size_t) row * blocks_per_row * NBINS + (threshold & (NBINS - 1));
    int pos = 0;
    for (int b = lane; b < row_block; b += warp_size) {
        pos += block_ties[b * NBINS];
    }
    pos = warp_reduce_sum<warp_size>(pos);
    if (pos >= rank || block_ties[row_block * NBINS] == 0) {
        return; // the same result in all threads of the block
    }

    nties = warp_reduce_sum<warp_size>(nties);
    if (lane == 0) {
        warp_ties[warp] = nties;
    }
    __syncthreads();
    for (int w = 0; w < warp; ++w) {
        pos += warp_ties[w];
    }

    // pos is the same in all lanes of the warp, so the lanes leave the loop together
    const int pos_end = min(pos + nties, rank);
    for (int64_t tile = warp_begin; tile < warp_end && pos < pos_end; tile += warp_size) {
        const int64_t col    = tile + lane;
        const bool    is_tie = col < warp_end && top_k_float_to_ordered(row_src[col]) == threshold;

        const ggml_cuda_thread_mask_t ties = ggml_cuda_ballot(is_tie);
        const int tie_pos = pos + popc(ties & (((ggml_cuda_thread_mask_t) 1 << lane) - 1));
        if (is_tie && tie_pos < pos_end) {
            row_dst[k - rank + tie_pos] = col;
        }
        pos += popc(ties);
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = (int) std::min<int64_t>(((int64_t) ncols + 1023) / 1024, 64);
    // each block of a row reads a contiguous range of whole tiles
    const int block_cols = (int) (((int64_t) ncols + blocks_per_row * BLOCK_SIZE - 1) / (blocks_per_row * BLOCK_SIZE) * BLOCK_SIZE);

    // chunk the rows to bound the histogram memory to 64 MB
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) blocks_per_row * NBINS * sizeof(int), nrows);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, chunk_nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) chunk_nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        top_k_radix_init<<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows, k);

        const dim3 row_grid(blocks_per_row * iter_nrows);
        for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
            top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    src, states, histograms, ncols, block_cols, blocks_per_row, shift);
            top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
                <<<iter_nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
        }

        top_k_radix_gather<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, dst, histograms, states, ncols, block_cols, k, blocks_per_row);

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}
// ---- end of the copy ----

static __global__ void rewrite_row(const float * __restrict__ src, float * __restrict__ dst, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

static __global__ void spin(volatile int * stop, long long max_cycles) {
    const long long t0 = clock64();
    while (!*stop && clock64() - t0 < max_cycles) {
    }
}

// src/llama-sampler.cpp at 3dfad876f: llama_sampler_pos_seed and the first draw after llama_sampler_set_pos
static uint32_t pos_seed(uint32_t seed, int32_t pos) {
    uint64_t x = ((uint64_t) seed << 32) ^ (uint64_t) (uint32_t) pos;
    x += 0x9e3779b97f4a7c15ULL;
    x  = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x  = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^=  x >> 31;
    return (uint32_t) (x ^ (x >> 32));
}
static float pos_uniform(uint32_t seed, int32_t pos) {
    std::mt19937 rng(pos_seed(seed, pos));
    std::uniform_real_distribution<double> dist(0.0f, 1.0f);
    return (float) dist(rng);
}

// the dist step of the backend chain (top-k -> temperature -> dist) on the candidates in the given order:
// scale by 1/temp, softmax, running sum, index = n - count(sum > u) with the count clamped to [1, n]
static int dist_pick(const std::vector<float> & logits, float temp, float u) {
    const int n = (int) logits.size();
    std::vector<float> x(n);
    float mx = -INFINITY;
    for (int i = 0; i < n; i++) { x[i] = logits[i] * (1.0f/temp); mx = std::max(mx, x[i]); }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float cum = 0.0f;
    int above = 0;
    for (int i = 0; i < n; i++) { cum += x[i]/sum; above += (cum - u) > 0.0f; }
    above = std::min(std::max(above, 1), n);
    return n - above;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <input> <iterations> [device] [plain|rewrite|busy]\n", argv[0]); return 1; }
    const int iters = atoi(argv[2]);
    const int dev   = argc > 3 ? atoi(argv[3]) : 0;
    const std::string mode = argc > 4 ? argv[4] : "plain";
    if (mode != "plain" && mode != "rewrite" && mode != "busy") { fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 1; }
    FILE * f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }
    int n_vocab, k, pos; unsigned seed; float temp;
    if (fscanf(f, "%d %d %u %d %f", &n_vocab, &k, &seed, &pos, &temp) != 5) { fprintf(stderr, "bad header\n"); return 1; }
    std::vector<int> ids(k); std::vector<double> p(k);
    for (int i = 0; i < k; i++) {
        if (fscanf(f, "%d %lf", &ids[i], &p[i]) != 2) { fprintf(stderr, "bad candidate line %d\n", i); return 1; }
    }
    fclose(f);

    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int drv = 0, rt = 0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
    printf("device %d: %s, compute %d.%d, %d SMs; CUDA runtime %d, driver %d; CUB %d (DeviceTopK needs CCCL >= 3.4.3)\n",
           dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount, rt, drv, CUB_VERSION);

    // the row: fixed pseudo-random background, the k candidates well above it
    std::vector<float> row(n_vocab);
    uint32_t s = 12345u;
    for (int i = 0; i < n_vocab; i++) { s = s*1664525u + 1013904223u; row[i] = -4.0f + 8.0f*((s >> 8) * (1.0f/16777216.0f)); }
    for (int i = 0; i < k; i++) { row[ids[i]] = 30.0f + temp*(float) log(p[i]); }

    const float u = pos_uniform(seed, pos);
    printf("row: %d columns, k = %d, seed %u, position %d, temperature %.3f, draw u = %.9g; mode %s\n", n_vocab, k, seed,
           pos, temp, u, mode.c_str());
    {
        // reference orders: candidates by descending probability (what a sort would give) and by ascending id
        std::vector<int> byp(k), byid(k);
        for (int i = 0; i < k; i++) byp[i] = byid[i] = i;
        std::stable_sort(byp.begin(), byp.end(), [&](int a, int b) { return p[a] > p[b]; });
        std::sort(byid.begin(), byid.end(), [&](int a, int b) { return ids[a] < ids[b]; });
        for (auto * ord : { &byp, &byid }) {
            std::vector<float> lg; for (int i : *ord) lg.push_back(row[ids[i]]);
            printf("%s order: picks %d\n", ord == &byp ? "descending-probability" : "ascending-id",
                   ids[(*ord)[dist_pick(lg, temp, u)]]);
        }
    }

    float * d_src; float * d_copy; int * d_dst; int * d_stop;
    CUDA_CHECK(cudaMalloc(&d_src, n_vocab*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_copy, n_vocab*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, k*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_stop, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_src, row.data(), n_vocab*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_copy, row.data(), n_vocab*sizeof(float), cudaMemcpyHostToDevice));
    cudaStream_t stream, stream2;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
    ggml_cuda_pool pool;

    std::map<std::string, std::pair<int, int>> orders; // order -> (count, picked token)
    std::map<int, int> picks;                          // picked token -> count
    std::vector<int> out(k);
    for (int it = 0; it < iters; it++) {
        if (mode == "rewrite") {
            rewrite_row<<<prop.multiProcessorCount, 256, 0, stream>>>(d_copy, d_src, n_vocab);
        } else if (mode == "busy") {
            CUDA_CHECK(cudaMemsetAsync(d_stop, 0, sizeof(int), stream2));
            spin<<<std::max(1, prop.multiProcessorCount/2), 32, 0, stream2>>>(d_stop, 50000000LL);
        }
        top_k_radix_cuda(pool, d_src, d_dst, n_vocab, 1, k, stream);
        if (mode == "busy") {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const int one = 1;
            CUDA_CHECK(cudaMemcpy(d_stop, &one, sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaStreamSynchronize(stream2));
        }
        CUDA_CHECK(cudaMemcpyAsync(out.data(), d_dst, k*sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::string key;
        std::vector<float> lg;
        for (int i = 0; i < k; i++) { key += (i ? "," : "") + std::to_string(out[i]); lg.push_back(row[out[i]]); }
        std::vector<int> chk(out); std::sort(chk.begin(), chk.end());
        std::vector<int> want(ids); std::sort(want.begin(), want.end());
        if (chk != want) { printf("iteration %d: wrong candidate set %s\n", it, key.c_str()); return 2; }
        const int tok = out[dist_pick(lg, temp, u)];
        auto & e = orders[key]; e.first++; e.second = tok;
        picks[tok]++;
    }
    printf("%d calls: %zu distinct orders\n", iters, orders.size());
    std::vector<std::pair<int, std::string>> byc;
    for (auto & [key, e] : orders) byc.push_back({ e.first, key });
    std::sort(byc.rbegin(), byc.rend());
    for (auto & [c, key] : byc) printf("order\t%d\t%s\tpicks\t%d\n", c, key.c_str(), orders[key].second);
    for (auto & [tok, c] : picks) printf("picked\t%d\t%d\n", tok, c);
    return 0;
}
