// Bit-exactness check of the benchmark's baseline (src/llama_mmvq.cuh) against llama.cpp itself: the same random
// weights and activations go through ggml's CUDA backend (MUL_MAT, MUL_MAT_ID, fused gate/up/SwiGLU) and through the
// copied kernels; the float outputs must be identical bit for bit.

#include "../src/common.cuh"
#include "../src/data.cuh"
#include "../src/llama_mmvq.cuh"
#include "verify_case.h"

#include <cstring>
#include <vector>

using namespace mb;

struct vcase {
    const char * name;
    int K, N, n_expert, n_used, moe, glu;
};

static const vcase CASES[] = {
    { "attn_q        2048 -> 4096",                  2048,   4096,  0, 0, 0, 0 },
    { "attn_kv       2048 -> 512",                   2048,    512,  0, 0, 0, 0 },
    { "attn_output   4096 -> 2048",                  4096,   2048,  0, 0, 0, 0 },
    { "shexp_gate_up 2048 -> 1024, fused SwiGLU",    2048,   1024,  0, 0, 0, 1 },
    { "shexp_down    1024 -> 2048",                  1024,   2048,  0, 0, 0, 0 },
    { "moe_gate_up   2048 -> 512 x 6/32, fused",     2048,    512, 32, 6, 1, 1 },
    { "moe_down       512 -> 2048 x 6/32",            512,   2048, 32, 6, 1, 0 },
    { "output        2048 -> 150144",                2048, 150144,  0, 0, 0, 0 },
};

__global__ void vk_arch(int * out) {
#ifdef __CUDA_ARCH__
    *out = __CUDA_ARCH__;
#endif
}

template <int T, int table = llama::MMVQ_PARAMETERS_GENERIC>
static void run_mine(const vcase & vc, const void * w, const void * wg, const float * x, const int * ids, float * dst, int nvec) {
    const int qk = traits<T>::qk;
    block_q8_1 * y;
    CUDA_CHECK(cudaMalloc(&y, (size_t) nvec * vc.K / 32 * sizeof(block_q8_1)));
    llama::quantize_row_q8_1_cuda(x, y, vc.K, nvec, 0);
    llama::call_args a;
    a.vx = w;
    a.vgate = vc.glu ? wg : nullptr;
    a.vy = y;
    a.ids = vc.moe ? ids : nullptr;
    a.dst = dst;
    a.ncols_x = vc.K;
    a.nrows_x = vc.N;
    a.stride_row_x = vc.K / qk;
    a.stride_channel_x = vc.N * (vc.K / qk);
    if (vc.moe) {
        a.nchannels_dst = vc.n_used;
        a.nchannels_y = nvec;
        a.stride_col_y = nvec * (vc.K / 32);
        a.stride_channel_y = vc.K / 32;
        a.stride_col_dst = vc.n_used * vc.N;
        a.stride_channel_dst = vc.N;
    } else {
        a.stride_col_y = vc.K / 32;
        a.stride_channel_y = vc.K / 32;
        a.stride_col_dst = vc.N;
        a.stride_channel_dst = vc.N;
    }
    llama::launch<T, table>(a, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(y));
}

int main() {
    int code_arch = 0;
    {
        int * d;
        CUDA_CHECK(cudaMalloc(&d, sizeof(int)));
        vk_arch<<<1, 1>>>(d); // which code path the driver loaded for this binary
        CUDA_CHECK(cudaMemcpy(&code_arch, d, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d));
    }
    printf("baseline kernels run as compute %d.%d code\n", code_arch / 100, (code_arch % 100) / 10);

    int total = 0, failed = 0;
    for (int T = 0; T < T_COUNT; ++T) {
        for (const auto & vc : CASES) {
            const int nexp = vc.moe ? vc.n_expert : 1;
            const int nvec = vc.moe && !vc.glu ? vc.n_used : 1;
            const int nout = vc.moe ? vc.n_used : 1;
            const size_t wbytes = (size_t) nexp * vc.N * row_bytes(T, vc.K);
            const size_t nblocks = wbytes / type_bs(T);
            uint8_t *dw, *dwg = nullptr;
            float *dx, *dd;
            int * di = nullptr;
            CUDA_CHECK(cudaMalloc(&dw, wbytes));
            gen_blocks<<<256, 256>>>(dw, nblocks, T, 11 + T);
            if (vc.glu) {
                CUDA_CHECK(cudaMalloc(&dwg, wbytes));
                gen_blocks<<<256, 256>>>(dwg, nblocks, T, 22 + T);
            }
            CUDA_CHECK(cudaMalloc(&dx, (size_t) nvec * vc.K * sizeof(float)));
            gen_floats<<<64, 256>>>(dx, (size_t) nvec * vc.K, 33 + T);
            CUDA_CHECK(cudaMalloc(&dd, (size_t) nout * vc.N * sizeof(float)));
            std::vector<int> ids;
            if (vc.moe) {
                ids = { 3, 17, 0, 31, 9, 22 };
                CUDA_CHECK(cudaMalloc(&di, ids.size() * sizeof(int)));
                CUDA_CHECK(cudaMemcpy(di, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<uint8_t> hw(wbytes), hwg(vc.glu ? wbytes : 0);
            std::vector<float> hx((size_t) nvec * vc.K), ref((size_t) nout * vc.N), mine((size_t) nout * vc.N);
            CUDA_CHECK(cudaMemcpy(hw.data(), dw, wbytes, cudaMemcpyDeviceToHost));
            if (vc.glu) CUDA_CHECK(cudaMemcpy(hwg.data(), dwg, wbytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hx.data(), dx, hx.size() * sizeof(float), cudaMemcpyDeviceToHost));

            vg_case c;
            c.type = T; c.K = vc.K; c.N = vc.N; c.n_expert = nexp; c.n_used = vc.n_used; c.nvec = nvec;
            c.moe = vc.moe; c.glu = vc.glu; c.w = hw.data(); c.wg = vc.glu ? hwg.data() : nullptr; c.x = hx.data();
            c.ids = vc.moe ? ids.data() : nullptr; c.out = ref.data();
            if (vg_run(&c) != 0) return 1;

            switch (T) {
                case T_Q4_0:    run_mine<T_Q4_0>(vc, dw, dwg, dx, di, dd, nvec); break;
                case T_Q4_1:    run_mine<T_Q4_1>(vc, dw, dwg, dx, di, dd, nvec); break;
                case T_Q8_0:    run_mine<T_Q8_0>(vc, dw, dwg, dx, di, dd, nvec); break;
                default:        run_mine<T_Q4_1_64>(vc, dw, dwg, dx, di, dd, nvec); break;
            }
            CUDA_CHECK(cudaMemcpy(mine.data(), dd, mine.size() * sizeof(float), cudaMemcpyDeviceToHost));

            // negative control: PR #25479's launch shape sums in a different order, so some outputs must differ
            std::vector<float> other(mine.size());
            switch (T) {
                case T_Q4_0:    run_mine<T_Q4_0, llama::MMVQ_PARAMETERS_PASCAL_DP4A>(vc, dw, dwg, dx, di, dd, nvec); break;
                case T_Q4_1:    run_mine<T_Q4_1, llama::MMVQ_PARAMETERS_PASCAL_DP4A>(vc, dw, dwg, dx, di, dd, nvec); break;
                case T_Q8_0:    run_mine<T_Q8_0, llama::MMVQ_PARAMETERS_PASCAL_DP4A>(vc, dw, dwg, dx, di, dd, nvec); break;
                default:        run_mine<T_Q4_1_64, llama::MMVQ_PARAMETERS_PASCAL_DP4A>(vc, dw, dwg, dx, di, dd, nvec); break;
            }
            CUDA_CHECK(cudaMemcpy(other.data(), dd, other.size() * sizeof(float), cudaMemcpyDeviceToHost));
            size_t ndiff_other = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                ndiff_other += memcmp(&ref[i], &other[i], sizeof(float)) != 0;
            }

            size_t diff = 0;
            double maxd = 0.0;
            for (size_t i = 0; i < ref.size(); ++i) {
                if (memcmp(&ref[i], &mine[i], sizeof(float)) != 0) {
                    diff++;
                    maxd = std::max(maxd, (double) fabsf(ref[i] - mine[i]));
                }
            }
            total++;
            failed += diff != 0;
            printf("%-8s %-44s %8zu outputs  %s", type_name(T), vc.name, ref.size(), diff ? "DIFFER" : "bit-identical");
            if (diff) printf(" (%zu differ, max |diff| %.3g)", diff, maxd);
            printf("   [control: PR #25479 shape differs in %zu]\n", ndiff_other);
            fflush(stdout);

            cudaFree(dw); cudaFree(dwg); cudaFree(dx); cudaFree(dd); cudaFree(di);
        }
    }
    printf("\n%d of %d cases bit-identical to llama.cpp\n", total - failed, total);
    return failed ? 2 : 0;
}
