// mmvq_kcol: the k-column products of Miner Llama (src/kcol_mm.cuh, a copy of the fork's
// ggml/src/ggml-cuda/mmvq-pascal-kcol-impl.cuh) at the dense quantized shapes of Qwen3.8-Flash-Next's Swift Q2_0 file,
// at 1 to 8 columns (the tokens of a speculative-decoding verify). See ../README.md, "k-column products (Q591)".

#include "common.cuh"
#include "data.cuh"
#include "kcol_mm.cuh"
#include "kcol_ref.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace {

using namespace kcol;

// The dense products of one token of the Swift Q2_0 file (its tensor census; 48 layers): type, K -> N, calls per token.
// hc_up and hc_down are the hyper-connection projections retyped to Q8_0 (the file keeps them in BF16; --bf16 times
// those); draft_output is the MTP draft's Q6_K head, which a draft runs once per drafted token.
struct kshape { const char * name; int type; int K, N, calls; };
const kshape SHAPES[] = {
    { "hc_up",       KQ8_0,     320,  10240, 97 }, { "hc_down",     KQ8_0,   10240,    320, 97 },
    { "attn_qkv",    KQ3_K,    2560,  10240, 18 }, { "attn_qkv",    KIQ4_XS,  2560,  10240, 13 },
    { "attn_qkv",    KQ4_K,    2560,  10240,  4 }, { "attn_qkv",    KQ2_0,    2560,  10240,  1 },
    { "attn_gate",   KQ3_K,    2560,   6144, 25 }, { "attn_gate",   KIQ4_XS,  2560,   6144,  4 },
    { "attn_gate",   KQ4_K,    2560,   6144,  3 }, { "attn_gate",   KQ2_0,    2560,   6144,  4 },
    { "attn_q",      KQ3_K,    2560,  12288,  6 }, { "attn_q",      KIQ4_XS,  2560,  12288,  4 },
    { "attn_q",      KQ2_0,    2560,  12288,  2 },
    { "ssm_out",     KIQ4_XS,  6144,   2560, 17 }, { "ssm_out",     KQ4_K,    6144,   2560, 11 },
    { "ssm_out",     KQ5_K,    6144,   2560,  3 }, { "ssm_out",     KQ6_K,    6144,   2560,  2 },
    { "ssm_out",     KQ3_K,    6144,   2560,  3 },
    { "attn_output", KQ6_K,    6144,   2560,  4 }, { "attn_output", KQ5_K,    6144,   2560,  4 },
    { "attn_output", KQ4_K,    6144,   2560,  4 },
    { "shexp_gate",  KQ2_0,    2560,    640, 21 }, { "shexp_gate",  KQ3_K,    2560,    640, 13 },
    { "shexp_gate",  KQ4_K,    2560,    640,  6 }, { "shexp_gate",  KIQ4_XS,  2560,    640,  6 },
    { "shexp_up",    KQ3_K,    2560,    640, 21 }, { "shexp_up",    KQ2_0,    2560,    640, 13 },
    { "shexp_up",    KQ4_K,    2560,    640,  6 }, { "shexp_up",    KIQ4_XS,  2560,    640,  5 },
    { "shexp_down",  KQ2_0,     640,   2560, 16 }, { "ple_key",     KQ2_0,    2560,  10240,  1 },
    { "output",      KQ5_K,    2560, 248320,  1 }, { "draft_output", KQ6_K,   2560, 248320,  0 },
};

// Variants: the fork's launch shape, R = 1 or 2 forced, the smallest activation tile, a 32 KB budget (3 blocks per SM),
// and the product alone (the records quantized once beforehand), which shows what the quantizer costs.
struct kvariant { const char * name; int r, itt, smem; bool prod_only; };
const kvariant VARIANTS[] = {
    { "pascal", 0, 0, 0, false }, { "r1", 1, 0, 0, false }, { "r2", 2, 0, 0, false }, { "itt1", 0, 1, 0, false },
    { "s32k", 0, 0, 32 * 1024, false }, { "prod", 0, 0, 0, true },
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
bool contains(const std::vector<std::string> & v, const std::string & x) { return std::find(v.begin(), v.end(), x) != v.end(); }
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n == 0 ? 0.0 : (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
}

struct dev_buf {
    void * p = nullptr;
    explicit dev_buf(size_t bytes) { CUDA_CHECK(cudaMalloc(&p, bytes + 256)); CUDA_CHECK(cudaMemset(p, 0, bytes + 256)); CUDA_CHECK(cudaDeviceSynchronize()); }
    ~dev_buf() { if (p) cudaFree(p); }
    dev_buf(const dev_buf &) = delete;
    dev_buf & operator=(const dev_buf &) = delete;
    template <typename X> X * as() const { return (X *) p; }
};

__global__ void k_arch(int * out) {
#ifdef __CUDA_ARCH__
    *out = __CUDA_ARCH__;
#endif
}

// Random weight bytes with finite FP16 scales: every block's d (and dmin) set to +-[1, 2) * 2^-8.
__global__ void gen_weights(uint8_t * __restrict__ w, const size_t nblocks, const int t, const int bs, const uint64_t seed) {
    for (size_t b = (size_t) blockIdx.x * blockDim.x + threadIdx.x; b < nblocks; b += (size_t) gridDim.x * blockDim.x) {
        uint8_t * blk = w + b * bs;
        for (int i = 0; i < bs; ++i) {
            blk[i] = (uint8_t) (mb::hash_u32(seed * 0x9E3779B97F4A7C15ull + b * 977 + i) >> 13);
        }
        const uint32_t h = mb::hash_u32(seed + b * 31 + 7);
        const uint16_t d    = (uint16_t) (0x1C00 | (h & 0x3FF) | ((h >> 10) & 1) << 15);
        const uint16_t dmin = (uint16_t) (0x1C00 | ((h >> 11) & 0x3FF));
        int off = 0;
        if (t == KQ3_K) off = 108;
        if (t == KQ6_K) off = 208;
        blk[off] = d & 0xff;
        blk[off + 1] = d >> 8;
        if (t == KQ4_K || t == KQ5_K) {
            blk[2] = dmin & 0xff;
            blk[3] = dmin >> 8;
        }
    }
}

void launch_type(const int t, const params & p, cudaStream_t s) {
    switch (t) {
        case KQ8_0:   launch<KQ8_0>(p, s);   break;
        case KQ2_0:   launch<KQ2_0>(p, s);   break;
        case KQ3_K:   launch<KQ3_K>(p, s);   break;
        case KQ4_K:   launch<KQ4_K>(p, s);   break;
        case KQ5_K:   launch<KQ5_K>(p, s);   break;
        case KQ6_K:   launch<KQ6_K>(p, s);   break;
        default:      launch<KIQ4_XS>(p, s); break;
    }
}

// The product alone, from records quantized beforehand.
template <int T>
void launch_prod_t(const params & p, cudaStream_t s) {
    args a;
    a.vx = p.vx; a.rec = p.rec; a.dst = p.dst; a.nrows = p.nrows; a.row_bytes = p.row_bytes;
    a.nunits = p.K / traits<T>::UW; a.lpr_log2 = p.s.lpr_log2; a.niter = a.nunits >> p.s.lpr_log2; a.itt = p.s.itt;
    a.gen = p.s.gen; a.seg_log2 = p.s.seg_log2; a.stride_col_dst = p.stride_col_dst;
    if (p.s.R == 1) launch_cols<T, 1>(a, p.s, p.ncols, s); else launch_cols<T, 2>(a, p.s, p.ncols, s);
}
void launch_prod(const int t, const params & p, cudaStream_t s) {
    switch (t) {
        case KQ8_0:   launch_prod_t<KQ8_0>(p, s);   break;
        case KQ2_0:   launch_prod_t<KQ2_0>(p, s);   break;
        case KQ3_K:   launch_prod_t<KQ3_K>(p, s);   break;
        case KQ4_K:   launch_prod_t<KQ4_K>(p, s);   break;
        case KQ5_K:   launch_prod_t<KQ5_K>(p, s);   break;
        case KQ6_K:   launch_prod_t<KQ6_K>(p, s);   break;
        default:      launch_prod_t<KIQ4_XS>(p, s); break;
    }
}
const void * kernel_func(const int t, const int ncols, const int R) {
#define KF(T) do { \
        const void * f1[8] = { (const void *) mul_mat_kcol<T, 1, 1>, (const void *) mul_mat_kcol<T, 2, 1>, (const void *) mul_mat_kcol<T, 3, 1>, \
            (const void *) mul_mat_kcol<T, 4, 1>, (const void *) mul_mat_kcol<T, 5, 1>, (const void *) mul_mat_kcol<T, 6, 1>, \
            (const void *) mul_mat_kcol<T, 7, 1>, (const void *) mul_mat_kcol<T, 8, 1> }; \
        const void * f2[8] = { (const void *) mul_mat_kcol<T, 1, 2>, (const void *) mul_mat_kcol<T, 2, 2>, (const void *) mul_mat_kcol<T, 3, 2>, \
            (const void *) mul_mat_kcol<T, 4, 2>, (const void *) mul_mat_kcol<T, 5, 2>, (const void *) mul_mat_kcol<T, 6, 2>, \
            (const void *) mul_mat_kcol<T, 7, 2>, (const void *) mul_mat_kcol<T, 8, 2> }; \
        return R == 1 ? f1[ncols - 1] : f2[ncols - 1]; } while (0)
    switch (t) {
        case KQ8_0: KF(KQ8_0); case KQ2_0: KF(KQ2_0); case KQ3_K: KF(KQ3_K); case KQ4_K: KF(KQ4_K);
        case KQ5_K: KF(KQ5_K); case KQ6_K: KF(KQ6_K); default: KF(KIQ4_XS);
    }
#undef KF
}

// Run `launch(stream, j)` for j = 0..batch-1 as one CUDA graph; time `reps` launches. Per-call microseconds.
std::vector<double> time_graph(const std::function<void(cudaStream_t, int)> & launch, int batch, int reps, cudaStream_t s) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int j = 0; j < batch; ++j) launch(s, j);
    CUDA_CHECK(cudaStreamEndCapture(s, &g));
    CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
    for (int w = 0; w < 3; ++w) CUDA_CHECK(cudaGraphLaunch(ge, s));
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

void usage() {
    printf("usage: mmvq_kcol [options]\n"
           "  --types LIST     q8_0,q2_0,q3_K,q4_K,q5_K,q6_K,iq4_xs (default all)\n"
           "  --shapes LIST    shape names (default all; see --list)\n"
           "  --variants LIST  pascal,r1,r2,itt1,s32k,prod (default all)\n"
           "  --tokens LIST    columns per product, 1 to 8 (default 1,2,3,4,5,6,8)\n"
           "  --reps N         timed graph launches, median reported (default 30)\n"
           "  --batch N        calls per CUDA graph (default 64; 8 for the output layers)\n"
           "  --nsm N          choose the launch shapes as for a card of N SMs (default: this card's)\n"
           "  --check-only     correctness only\n"
           "  --csv FILE       also write the results as CSV\n"
           "  --seed N         random seed (default 1234)\n"
           "  --list           list the shapes\n");
}

} // namespace

int main(int argc, char ** argv) {
    std::vector<std::string> types, shapes, variants;
    std::vector<std::string> tokens = { "1", "2", "3", "4", "5", "6", "8" };
    int reps = 30, batch = 64, nsm_force = 0;
    bool check_only = false;
    std::string csv;
    uint64_t seed = 1234;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(); exit(1); } return argv[++i]; };
        if (a == "--types") types = split(next());
        else if (a == "--shapes") shapes = split(next());
        else if (a == "--variants") variants = split(next());
        else if (a == "--tokens") tokens = split(next());
        else if (a == "--reps") reps = atoi(next().c_str());
        else if (a == "--batch") batch = atoi(next().c_str());
        else if (a == "--nsm") nsm_force = atoi(next().c_str());
        else if (a == "--check-only") check_only = true;
        else if (a == "--csv") csv = next();
        else if (a == "--seed") seed = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--list") {
            for (const auto & s : SHAPES) printf("  %-13s %-7s %6d -> %-6d x%d\n", s.name, kname(s.type), s.K, s.N, s.calls);
            return 0;
        } else { usage(); return a == "--help" ? 0 : 1; }
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    dev_buf arch_buf(4);
    k_arch<<<1, 1>>>(arch_buf.as<int>());
    int code_arch = 0;
    CUDA_CHECK(cudaMemcpy(&code_arch, arch_buf.p, 4, cudaMemcpyDeviceToHost));
    const int nsm = nsm_force > 0 ? nsm_force : prop.multiProcessorCount;
    printf("device: %s, compute %d.%d, %d SMs (shapes chosen for %d), code arch %d\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, nsm, code_arch);
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    std::vector<int> cols;
    for (const auto & t : tokens) {
        const int n = atoi(t.c_str());
        if (n < 1 || n > KCOL_MAX_COLS) { fprintf(stderr, "--tokens: 1 to %d\n", KCOL_MAX_COLS); return 1; }
        cols.push_back(n);
    }
    FILE * f = nullptr;
    if (!csv.empty()) {
        f = fopen(csv.c_str(), "w");
        if (!f) { fprintf(stderr, "cannot write %s\n", csv.c_str()); return 1; }
        fprintf(f, "device,code_arch,shape,type,K,N,calls_per_token,tokens,variant,L,R,itt,gen,smem,grid,regs,bytes_per_call,"
                   "us_median,us_min,gbs,vs_1_token,err_over_tol,check\n");
    }
    printf("\n%-12s %-7s %2s %-7s %4s %2s %4s %4s %6s %6s %5s %9s %9s %7s %6s %8s %s\n", "shape", "type", "k", "variant", "L",
           "R", "itt", "gen", "smem", "grid", "regs", "us/call", "us(min)", "GB/s", "vs k1", "err/tol", "check");
    bool ok = true;
    std::map<std::pair<std::string, int>, double> us1;           // (shape/type/variant, 1) -> us at 1 column
    std::map<std::pair<std::string, int>, double> pass_us;       // (variant, k) -> us per pass over the shapes' calls
    for (const auto & sh : SHAPES) {
        if (!shapes.empty() && !contains(shapes, sh.name)) continue;
        if (!types.empty() && !contains(types, kname(sh.type))) continue;
        const int bs = kbs(sh.type), qk = kqk(sh.type);
        const size_t rbytes = (size_t) sh.K / qk * bs, mat = rbytes * sh.N;
        const int ncopies = (int) std::max<size_t>(1, ((size_t) 64 << 20) / mat);
        const int nb = mat >= ((size_t) 64 << 20) ? 8 : batch;
        dev_buf w(mat * ncopies);
        gen_weights<<<256, 256, 0, s>>>(w.as<uint8_t>(), mat * ncopies / bs, sh.type, bs, seed + sh.N + sh.type);
        dev_buf y((size_t) sh.K * KCOL_MAX_COLS * sizeof(float)), d((size_t) sh.N * KCOL_MAX_COLS * sizeof(float));
        dev_buf rec(records_bytes(sh.type, sh.K, KCOL_MAX_COLS));
        mb::gen_floats<<<64, 256, 0, s>>>(y.as<float>(), (size_t) sh.K * KCOL_MAX_COLS, seed + 7);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(s));
        // reference rows (every row up to 4096 rows, else every 37th), dequantized once
        const int row_step = sh.N > 4096 ? 37 : 1;
        std::vector<uint8_t> wh(mat);
        CUDA_CHECK(cudaMemcpy(wh.data(), w.p, mat, cudaMemcpyDeviceToHost));
        std::vector<int> ref_rows;
        for (int r = 0; r < sh.N; r += row_step) ref_rows.push_back(r);
        std::vector<double> wd((size_t) ref_rows.size() * sh.K);
        for (size_t i = 0; i < ref_rows.size(); ++i) dequant_row(sh.type, wh.data() + (size_t) ref_rows[i] * rbytes, sh.K, wd.data() + i * sh.K);
        for (const int k : cols) {
            // the reference of the k columns, from the first variant's records (every variant quantizes the same way)
            std::vector<double> refv, absv;
            for (const auto & v : VARIANTS) {
                if (!variants.empty() && !contains(variants, v.name)) continue;
                params p;
                p.vx = w.as<char>(); p.y = y.as<float>(); p.rec = rec.as<uint32_t>(); p.dst = d.as<float>();
                p.K = sh.K; p.nrows = sh.N; p.row_bytes = (int) rbytes; p.ncols = k; p.stride_col_y = sh.K; p.stride_col_dst = sh.N;
                if (!pick_shape(sh.type, sh.K, sh.N, k, (int) rbytes, (uintptr_t) w.p, nsm, v.r, v.itt, v.smem, p.s)) {
                    printf("%-12s %-7s %2d %-7s  (no launch shape)\n", sh.name, kname(sh.type), k, v.name);
                    continue;
                }
                // correctness: copy 0
                CUDA_CHECK(cudaMemsetAsync(d.p, 0xff, (size_t) sh.N * k * sizeof(float), s));
                launch_type(sh.type, p, s);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamSynchronize(s));
                std::vector<float> got((size_t) sh.N * k);
                CUDA_CHECK(cudaMemcpy(got.data(), d.p, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
                if (refv.empty()) {
                    std::vector<uint32_t> rh(records_bytes(sh.type, sh.K, k) / 4);
                    CUDA_CHECK(cudaMemcpy(rh.data(), rec.p, rh.size() * 4, cudaMemcpyDeviceToHost));
                    std::vector<double> av;
                    act_values(sh.type, rh, sh.K, k, av);
                    refv.resize((size_t) k * ref_rows.size());
                    absv.resize(refv.size());
                    for (int j = 0; j < k; ++j) {
                        for (size_t i = 0; i < ref_rows.size(); ++i) {
                            double acc = 0.0, as = 0.0;
                            for (int c = 0; c < sh.K; ++c) {
                                const double tv = wd[i * sh.K + c] * av[(size_t) j * sh.K + c];
                                acc += tv;
                                as += fabs(tv);
                            }
                            refv[(size_t) j * ref_rows.size() + i] = acc;
                            absv[(size_t) j * ref_rows.size() + i] = as;
                        }
                    }
                }
                double worst = 0.0;
                for (int j = 0; j < k; ++j) {
                    for (size_t i = 0; i < ref_rows.size(); ++i) {
                        const double acc = refv[(size_t) j * ref_rows.size() + i];
                        const double g = got[(size_t) j * sh.N + ref_rows[i]];
                        const double e = std::isfinite(g) ? fabs(g - acc) / (1e-5 * absv[(size_t) j * ref_rows.size() + i] + 1e-30) : 1e30;
                        worst = std::max(worst, e);
                    }
                }
                const bool pass = worst <= 1.0;
                ok &= pass;
                cudaFuncAttributes attr;
                CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_func(sh.type, k, p.s.R)));
                double us_med = 0.0, us_min = 0.0, gbs = 0.0, vs1 = 0.0;
                const std::string key = std::string(sh.name) + "/" + kname(sh.type) + "/" + v.name;
                if (!check_only) {
                    const auto tm = time_graph([&](cudaStream_t st, int j) {
                        params pj = p;
                        pj.vx = w.as<char>() + (size_t) (j % ncopies) * mat;
                        if (v.prod_only) launch_prod(sh.type, pj, st); else launch_type(sh.type, pj, st);
                    }, nb, reps, s);
                    us_med = median(tm);
                    us_min = *std::min_element(tm.begin(), tm.end());
                    gbs = mat / (us_med * 1e3);
                    if (k == 1) us1[{ key, 1 }] = us_med;
                    if (us1.count({ key, 1 })) vs1 = us_med / us1[{ key, 1 }];
                    pass_us[{ v.name, k }] += us_med * sh.calls;
                }
                printf("%-12s %-7s %2d %-7s %4d %2d %4d %4d %6d %6d %5d %9.2f %9.2f %7.1f %6.2f %8.3f %s\n", sh.name, kname(sh.type),
                       k, v.name, 1 << p.s.lpr_log2, p.s.R, p.s.itt, p.s.gen, p.s.smem, p.s.nblocks, attr.numRegs, us_med, us_min,
                       gbs, vs1, worst, pass ? "PASS" : "FAIL");
                if (f) {
                    fprintf(f, "\"%s\",%d,%s,%s,%d,%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d,%zu,%.3f,%.3f,%.2f,%.3f,%.4f,%s\n", prop.name,
                            code_arch, sh.name, kname(sh.type), sh.K, sh.N, sh.calls, k, v.name, 1 << p.s.lpr_log2, p.s.R, p.s.itt,
                            p.s.gen, p.s.smem, p.s.nblocks, attr.numRegs, mat, us_med, us_min, gbs, vs1, worst, pass ? "PASS" : "FAIL");
                }
            }
        }
    }
    if (!check_only) {
        printf("\nper pass of the model (calls per token x us per call over the shapes run), ms:\n%-8s", "variant");
        for (const int k : cols) printf(" %8s", ("k=" + std::to_string(k)).c_str());
        printf("\n");
        for (const auto & v : VARIANTS) {
            if (!variants.empty() && !contains(variants, v.name)) continue;
            printf("%-8s", v.name);
            for (const int k : cols) {
                const double t = pass_us[{ v.name, k }];
                printf(" %8.2f", t / 1e3);
                if (f) fprintf(f, "\"%s\",%d,PASS_TOTAL,,0,0,0,%d,%s,0,0,0,0,0,0,0,0,%.3f,,,,,\n", prop.name, code_arch, k, v.name, t);
            }
            printf("\n");
        }
    }
    if (f) {
        fclose(f);
        printf("\nCSV written to %s\n", csv.c_str());
    }
    printf("\ncorrectness: %s\n", ok ? "all variants match the CPU reference" : "FAILURES");
    return ok ? 0 : 2;
}
