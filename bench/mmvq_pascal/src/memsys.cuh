// --tlb-probe: pointer chases that expose the GPU's address translation (experiment Q99, test E1 of
// docs/research/notes/pascal_forks_and_quants.md, section 3).
//
// One thread follows a chain of 8-byte slots, each holding the index of the next, so every step is one dependent load
// through L2 (ld.global.cg, no L1). After a warm-up pass the slots themselves sit in L2, so the time per step is the L2
// hit latency while the translations hit in the TLBs and grows by a page walk where they miss. Two patterns:
//   - stride: the slots are S bytes apart (2 MiB, the page size of cudaMalloc, and 32 MiB, the entry size reported for
//     Pascal's L2 TLB) and visited in order, over footprints from 16 MiB to the whole buffer;
//   - pages: one slot in each of N random 2 MiB pages of the buffer, visited in a random cycle.
// Each chase runs alone and again while the other SMs stream reads from a separate buffer (the "loaded" columns), as a
// real token's walks compete with the weight stream. With --check-only every chain is walked once around and checked to
// end where it started; nothing is timed.
#pragma once

#include "common.cuh"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace mb {
namespace memsys {

static __device__ __forceinline__ uint64_t ld_cg_u64(const uint64_t * p) {
    uint64_t v;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint64_t global_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// out[0] = slot reached, out[1] = elapsed ns (globaltimer), out[2] = elapsed SM cycles; then sets *stop for the load.
static __global__ void k_chase(const uint64_t * buf, uint64_t start, int steps, uint64_t * out, int * stop) {
    uint64_t j = start;
    const uint64_t t0 = global_ns();
    const long long c0 = clock64();
    for (int s = 0; s < steps; ++s) {
        j = ld_cg_u64(buf + j);
    }
    const long long c1 = clock64();
    const uint64_t t1 = global_ns();
    out[0] = j;
    out[1] = t1 - t0;
    out[2] = (uint64_t) (c1 - c0);
    if (stop) {
        *(volatile int *) stop = 1;
        __threadfence();
    }
}

// slot idx[i] holds idx[(i + 1) % n] (indices in 8-byte slots)
static __global__ void k_link(uint64_t * buf, const uint64_t * idx, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        buf[idx[i]] = idx[(i + 1) % n];
    }
}

// Streams reads over p[0..n) with 128-bit loads, pass after pass, until *stop is set.
static __global__ void k_stream_until(const uint4 * p, size_t n, const int * stop, unsigned * sink) {
    unsigned acc = 0;
    while (*(const volatile int *) stop == 0) {
        for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
            const uint4 v = __ldg(p + i);
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    if (acc == 0x12345678u) {
        sink[0] = acc;
    }
}

struct chase_result {
    std::string kind;       // "stride" or "pages"
    size_t footprint = 0;   // bytes spanned
    size_t stride = 0;      // bytes between slots (stride) or 2 MiB (pages)
    int    slots = 0;
    int    steps = 0;
    double ns = 0, cycles = 0;           // per step, alone
    double ns_loaded = 0, cycles_loaded = 0;
    bool   ok = true;
};

// Runs the probe on the current device. max_bytes caps the chase buffer; check_only walks every chain once and checks
// it, without timing.
static std::vector<chase_result> tlb_probe(size_t max_bytes, bool check_only, cudaStream_t s) {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    const size_t page = (size_t) 2 << 20;
    const size_t stream_bytes = (size_t) 256 << 20;
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    const size_t reserve = (size_t) 400 << 20;
    size_t bytes = std::min(max_bytes, free_b > reserve + stream_bytes ? free_b - reserve - stream_bytes : 0);
    bytes = bytes / (32 * page) * (32 * page); // whole 64 MiB units
    if (bytes < 32 * page) {
        fprintf(stderr, "tlb probe: not enough free device memory (%.2f GB free)\n", free_b / 1e9);
        exit(1);
    }
    uint64_t * buf = nullptr;
    uint4 * sbuf = nullptr;
    uint64_t * out = nullptr;
    uint64_t * idx = nullptr;
    int * stop = nullptr;
    unsigned * sink = nullptr;
    const int max_slots = (int) (bytes / page);
    CUDA_CHECK(cudaMalloc(&buf, bytes));
    CUDA_CHECK(cudaMalloc(&sbuf, stream_bytes));
    CUDA_CHECK(cudaMemset(sbuf, 0x5a, stream_bytes));
    CUDA_CHECK(cudaMalloc(&out, 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&idx, (size_t) max_slots * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&stop, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sink, sizeof(unsigned)));
    cudaStream_t s_load;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_load, cudaStreamNonBlocking));
    printf("tlb probe: %.2f GiB chase buffer (2 MiB pages), %d SMs; background load: %d SMs streaming 256 MiB\n",
           bytes / (double) (1u << 30), prop.multiProcessorCount, std::max(1, prop.multiProcessorCount - 1));

    uint64_t rng = 0x9E3779B97F4A7C15ull;
    auto next_rand = [&]() {
        rng = rng * 6364136223846793005ull + 1442695040888963407ull;
        return rng >> 33;
    };

    // one chase: n slots at the given element indices, in that order
    auto run = [&](const std::vector<uint64_t> & order, chase_result & r) {
        const int n = (int) order.size();
        CUDA_CHECK(cudaMemcpyAsync(idx, order.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice, s));
        k_link<<<std::max(1, std::min(64, (n + 255) / 256)), 256, 0, s>>>(buf, idx, n);
        CUDA_CHECK(cudaGetLastError());
        uint64_t h[3];
        if (check_only) {
            k_chase<<<1, 1, 0, s>>>(buf, order[0], n, out, nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(h, out, sizeof(h), cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            r.ok = h[0] == order[0];
            r.steps = n;
            return;
        }
        const int steps = std::max(2 * n, 4096);
        r.steps = steps;
        k_chase<<<1, 1, 0, s>>>(buf, order[0], 2 * n, out, nullptr); // warm-up: slots into L2, translations into TLBs
        k_chase<<<1, 1, 0, s>>>(buf, order[0], steps, out, nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h, out, sizeof(h), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        r.ns = (double) h[1] / steps;
        r.cycles = (double) h[2] / steps;
        // the same chase while the other SMs stream reads
        CUDA_CHECK(cudaMemsetAsync(stop, 0, sizeof(int), s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        k_stream_until<<<std::max(1, prop.multiProcessorCount - 1), 1024, 0, s_load>>>(sbuf, stream_bytes / 16, stop, sink);
        CUDA_CHECK(cudaGetLastError());
        k_chase<<<1, 1, 0, s>>>(buf, order[0], 2 * n, out, nullptr);
        k_chase<<<1, 1, 0, s>>>(buf, order[0], steps, out, stop);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h, out, sizeof(h), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaStreamSynchronize(s_load));
        r.ns_loaded = (double) h[1] / steps;
        r.cycles_loaded = (double) h[2] / steps;
    };

    std::vector<chase_result> res;
    // stride chases over growing footprints
    const size_t mib = (size_t) 1 << 20;
    const size_t fps_mib[] = { 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632 };
    for (const size_t stride : { 2 * mib, 32 * mib }) {
        for (const size_t fm : fps_mib) {
            const size_t fp = fm * mib;
            if (fp > bytes || fp / stride < 2) {
                continue;
            }
            std::vector<uint64_t> order;
            for (size_t o = 0; o < fp; o += stride) {
                order.push_back(o / 8);
            }
            chase_result r;
            r.kind = "stride";
            r.footprint = fp;
            r.stride = stride;
            r.slots = (int) order.size();
            run(order, r);
            res.push_back(r);
        }
    }
    // one slot in each of N random 2 MiB pages of the whole buffer, in a random cycle
    std::vector<int> npages = { 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 2560 };
    npages.push_back(max_slots);
    for (const int n : npages) {
        if (n > max_slots || (n == max_slots && n == npages[npages.size() - 2])) {
            continue;
        }
        std::vector<int> pages(max_slots);
        for (int i = 0; i < max_slots; ++i) {
            pages[i] = i;
        }
        for (int i = max_slots - 1; i > 0; --i) {
            std::swap(pages[i], pages[next_rand() % (i + 1)]);
        }
        std::vector<uint64_t> order;
        for (int i = 0; i < n; ++i) {
            const size_t off = (size_t) pages[i] * page + (next_rand() % (page / 128)) * 128;
            order.push_back(off / 8);
        }
        chase_result r;
        r.kind = "pages";
        r.footprint = bytes;
        r.stride = page;
        r.slots = n;
        run(order, r);
        res.push_back(r);
    }

    CUDA_CHECK(cudaStreamDestroy(s_load));
    cudaFree(buf);
    cudaFree(sbuf);
    cudaFree(out);
    cudaFree(idx);
    cudaFree(stop);
    cudaFree(sink);
    return res;
}

} // namespace memsys
} // namespace mb
