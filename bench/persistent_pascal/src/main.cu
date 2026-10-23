// persistent_pascal: one Flint MoE layer at one token as one persistent kernel, against the same layer as the 13
// launches the llama.cpp fork makes. See ../README.md.
#include "persistent.cuh"
#include "ref.cuh"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace pk;

namespace {

// ---- formats -----------------------------------------------------------------------------------------------------------

struct format {
    const char * name;
    bool router_f32;
    int  tgu, tdn, tsh;  // routed gate/up, routed down, shared expert
    const char * what;
};

const format FORMATS[] = {
    { "s4l0", false, mb::T_Q5_0, mb::T_Q8_0, mb::T_Q8_0, "S4-L0: routed gate/up Q5_0, routed down Q8_0, shared expert Q8_0, router F16" },
    { "g3",   true,  mb::T_Q5_0, mb::T_Q5_0, mb::T_Q8_0, "G3, layers 9-45: routed gate/up Q5_0, routed down Q5_0, shared expert Q8_0, router F32" },
};

size_t layer_bytes(const format & f) {
    return (size_t) NE * H * (f.router_f32 ? 4 : 2) + (size_t) KU * (2 * (size_t) F * row_bytes_of(f.tgu, H) + (size_t) H * row_bytes_of(f.tdn, F)) +
           2 * (size_t) FS * row_bytes_of(f.tsh, H) + (size_t) H * row_bytes_of(f.tsh, FS);
}

// ---- the two paths, per format -------------------------------------------------------------------------------------

constexpr float EPS   = 1e-6f;               // flint.attention.layer_norm_rms_epsilon
constexpr float CLAMP = 6.103515625e-5f;     // llama-graph.cpp: clamp of the weight sum before normalisation

// Launch the multi-launch path; `steps` selects which of the 13 kernels run (bit k = kernel k + 1), for timing them alone.
template <typename TR, int TGU, int TDN, int TSH>
void baseline(const layer_weights & w, const float * x, const layer_bufs & b, cudaStream_t s, const unsigned steps = 0x1fff) {
    auto mm = [&](auto kernel, int nrows, int rows_per_block, int nch, const mmvqp_args & p) {
        kernel<<<dim3((nrows + rows_per_block - 1) / rows_per_block, nch), dim3(32, 4), 0, s>>>(p);
    };
    // 1. RMS_NORM+MUL
    if (steps & (1u << 0)) k_rms_norm_mul<<<1, 512, 0, s>>>(x, b.xn, H, EPS, w.norm_w);
    // 2. router: MUL_MAT ffn_gate_inp (mul_mat_vec_f)
    if (steps & (1u << 1)) k_router<TR><<<NE, 256, 0, s>>>((const TR *) w.router, b.xn, b.logits, H / 2, H);
    // 3. SIGMOID+ARGSORT+GET_ROWS+SUM_ROWS+CLAMP+DIV (topk_moe)
    if (steps & (1u << 2)) k_topk_moe<<<1, dim3(32, 8), 0, s>>>(b.logits, b.weights, b.ids, 1, KU, CLAMP, 1.0f);
    // 4.-5. MUL_MAT_ID+MUL_MAT_ID+SWIGLU: quantize, then the fused gate/up product
    if (steps & (1u << 3)) k_quantize_q8_1<<<dim3(H / 256, 1), 256, 0, s>>>(b.xn, b.yq, H, H, H);
    if (steps & (1u << 4)) {
        using S = mmvqp_shape<TGU, true>;
        mmvqp_args p{};
        p.vx = w.up_exps; p.gate = w.gate_exps; p.vy = (const char *) b.yq; p.ids = b.ids; p.dst = b.act;
        p.nrows = F; p.row_bytes = row_bytes_of(TGU, H); p.lpr_log2 = mmvqp_lpr_log2(TGU, H, S::C);
        p.niter = (H / 32 / S::C) >> p.lpr_log2; p.nchannels_y = 1;
        p.channel_bytes_x = (int64_t) F * p.row_bytes; p.channel_bytes_y = 0; p.stride_channel_dst = F;
        mm(k_mmvq_pascal<TGU, S::R, S::C, true, MMVQP_GATE>, F, 4 * (32 >> p.lpr_log2) * S::R, KU, p);
    }
    // 6.-7. MUL_MAT_ID ffn_down_exps
    if (steps & (1u << 5)) k_quantize_q8_1<<<dim3(F / 256, KU), 256, 0, s>>>(b.act, b.aq, F, F, F);
    if (steps & (1u << 6)) {
        using S = mmvqp_shape<TDN, false>;
        mmvqp_args p{};
        p.vx = w.down_exps; p.vy = (const char *) b.aq; p.ids = b.ids; p.dst = b.dexp;
        p.nrows = H; p.row_bytes = row_bytes_of(TDN, F); p.lpr_log2 = mmvqp_lpr_log2(TDN, F, S::C);
        p.niter = (F / 32 / S::C) >> p.lpr_log2; p.nchannels_y = KU;
        p.channel_bytes_x = (int64_t) H * p.row_bytes; p.channel_bytes_y = (F / 32) * (int64_t) sizeof(block_q8_1); p.stride_channel_dst = H;
        mm(k_mmvq_pascal<TDN, S::R, S::C, true, MMVQP_PLAIN>, H, 4 * (32 >> p.lpr_log2) * S::R, KU, p);
    }
    // 8. MUL+ADD+ADD+ADD+ADD+ADD ffn_moe_weighted
    if (steps & (1u << 7)) k_moe_weighted_reduction<KU><<<dim3(1, H / 256), 256, 0, s>>>(b.dexp, nullptr, b.weights, b.moe, H);
    // 9.-10. MUL_MAT+MUL_MAT+SWIGLU of the shared expert (with Q50's fusion)
    if (steps & (1u << 8)) k_quantize_q8_1<<<dim3(H / 256, 1), 256, 0, s>>>(b.xn, b.yq2, H, H, H);
    if (steps & (1u << 9)) {
        using S = mmvqp_shape<TSH, true>;
        mmvqp_args p{};
        p.vx = w.up_sh; p.gate = w.gate_sh; p.vy = (const char *) b.yq2; p.dst = b.sact;
        p.nrows = FS; p.row_bytes = row_bytes_of(TSH, H); p.lpr_log2 = mmvqp_lpr_log2(TSH, H, S::C);
        p.niter = (H / 32 / S::C) >> p.lpr_log2; p.nchannels_y = 1;
        mm(k_mmvq_pascal<TSH, S::R, S::C, true, MMVQP_GATE>, FS, 4 * (32 >> p.lpr_log2) * S::R, 1, p);
    }
    // 11.-12. MUL_MAT+ADD ffn_down_shexp, the add of the routed sum fused
    if (steps & (1u << 10)) k_quantize_q8_1<<<dim3(FS / 256, 1), 256, 0, s>>>(b.sact, b.sq, FS, FS, FS);
    if (steps & (1u << 11)) {
        using S = mmvqp_shape<TSH, false>;
        mmvqp_args p{};
        p.vx = w.down_sh; p.vy = (const char *) b.sq; p.bias = b.moe; p.dst = b.sd;
        p.nrows = H; p.row_bytes = row_bytes_of(TSH, FS); p.lpr_log2 = mmvqp_lpr_log2(TSH, FS, S::C);
        p.niter = (FS / 32 / S::C) >> p.lpr_log2; p.nchannels_y = 1;
        mm(k_mmvq_pascal<TSH, S::R, S::C, true, MMVQP_BIAS>, H, 4 * (32 >> p.lpr_log2) * S::R, 1, p);
    }
    // 13. ADD l_out (residual)
    if (steps & (1u << 12)) k_add<<<H / 256, 256, 0, s>>>(b.sd, x, b.out, H);
}

const char * STEP_NAMES[13] = {
    "rms_norm+mul", "router (mmvf)", "topk_moe", "quantize xn", "routed gate/up", "quantize routed", "routed down",
    "weighted sum", "quantize xn again", "shared gate/up", "quantize shared", "shared down+add", "residual add",
};

// Persistent kernel variants: routed gate/up rows per warp pass.
constexpr int NV = 2;
const char * VARIANT_NAMES[NV] = { "r1", "r2" };
const char * VARIANT_WHAT[NV] = { "1 routed gate/up row per warp pass (the fork's launch shape)", "2 routed gate/up rows per warp pass" };

struct ops {
    void (*baseline)(const layer_weights &, const float *, const layer_bufs &, cudaStream_t, unsigned);
    const void * pk[NV];        // the persistent kernel
    const void * pk_trace[NV];  // its traced build
    std::vector<std::pair<std::string, const void *>> base_kernels;
};

template <typename TR, int TGU, int TDN, int TSH>
ops make_ops() {
    ops o;
    o.baseline = baseline<TR, TGU, TDN, TSH>;
    o.pk[0]       = (const void *) k_moe_layer<TR, TGU, TDN, TSH, 1, false>;
    o.pk[1]       = (const void *) k_moe_layer<TR, TGU, TDN, TSH, 2, false>;
    o.pk_trace[0] = (const void *) k_moe_layer<TR, TGU, TDN, TSH, 1, true>;
    o.pk_trace[1] = (const void *) k_moe_layer<TR, TGU, TDN, TSH, 2, true>;
    o.base_kernels = {
        { "rms_norm+mul", (const void *) k_rms_norm_mul },
        { "router", (const void *) k_router<TR> },
        { "topk_moe", (const void *) k_topk_moe },
        { "quantize_q8_1", (const void *) k_quantize_q8_1 },
        { "routed gate/up", (const void *) k_mmvq_pascal<TGU, mmvqp_shape<TGU, true>::R, mmvqp_shape<TGU, true>::C, true, MMVQP_GATE> },
        { "routed down", (const void *) k_mmvq_pascal<TDN, mmvqp_shape<TDN, false>::R, mmvqp_shape<TDN, false>::C, true, MMVQP_PLAIN> },
        { "weighted sum", (const void *) k_moe_weighted_reduction<KU> },
        { "shared gate/up", (const void *) k_mmvq_pascal<TSH, mmvqp_shape<TSH, true>::R, mmvqp_shape<TSH, true>::C, true, MMVQP_GATE> },
        { "shared down+add", (const void *) k_mmvq_pascal<TSH, mmvqp_shape<TSH, false>::R, mmvqp_shape<TSH, false>::C, true, MMVQP_BIAS> },
        { "add", (const void *) k_add },
    };
    return o;
}

ops ops_for(const format & f) {
    if (f.tgu == mb::T_Q5_0 && f.tdn == mb::T_Q8_0 && f.tsh == mb::T_Q8_0) {
        return f.router_f32 ? make_ops<float, mb::T_Q5_0, mb::T_Q8_0, mb::T_Q8_0>() : make_ops<half, mb::T_Q5_0, mb::T_Q8_0, mb::T_Q8_0>();
    }
    if (f.tgu == mb::T_Q5_0 && f.tdn == mb::T_Q5_0 && f.tsh == mb::T_Q8_0) {
        return f.router_f32 ? make_ops<float, mb::T_Q5_0, mb::T_Q5_0, mb::T_Q8_0>() : make_ops<half, mb::T_Q5_0, mb::T_Q5_0, mb::T_Q8_0>();
    }
    fprintf(stderr, "format not instantiated\n");
    exit(1);
}

void launch_pk(const void * fn, const pk_args & a, const int grid, cudaStream_t s) {
    void * params[] = { (void *) &a };
    CUDA_CHECK(cudaLaunchKernel(fn, dim3(grid), dim3(PK_THREADS), params, 0, s));
}

// ---- data ------------------------------------------------------------------------------------------------------------

struct dev_buf {
    void * p = nullptr;
    size_t n = 0;
    explicit dev_buf(size_t bytes) : n(bytes) {
        CUDA_CHECK(cudaMalloc(&p, bytes));
        CUDA_CHECK(cudaMemset(p, 0, bytes));
    }
    ~dev_buf() { cudaFree(p); }
    dev_buf(const dev_buf &) = delete;
    template <typename T> T * as() const { return (T *) p; }
};

__global__ void k_gen_router(void * dst, const int n, const bool f32, const uint64_t seed) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const uint64_t base = seed * 0x2545F4914F6CDD1Dull + (uint64_t) i * 4;
        const float s = mb::u01(mb::hash_u32(base)) + mb::u01(mb::hash_u32(base + 1)) + mb::u01(mb::hash_u32(base + 2)) + mb::u01(mb::hash_u32(base + 3));
        const float v = (s - 2.0f) * 0.07f;  // sd ~0.02: logits with sd ~1.5 for a normed input
        if (f32) {
            ((float *) dst)[i] = v;
        } else {
            ((half *) dst)[i] = __float2half(v);
        }
    }
}

__global__ void k_gen_norm(float * dst, const int n, const uint64_t seed) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        dst[i] = 0.5f + mb::u01(mb::hash_u32(seed * 0x9E3779B97F4A7C15ull + i));
    }
}

struct layer_set {
    std::unique_ptr<dev_buf> norm, router, gate, up, down, gsh, ush, dsh;
    layer_weights w{};
};

void make_layer(layer_set & L, const format & f, const uint64_t seed) {
    const size_t gu_rows = (size_t) NE * F, dn_rows = (size_t) NE * H;
    L.norm   = std::make_unique<dev_buf>(H * sizeof(float));
    L.router = std::make_unique<dev_buf>((size_t) NE * H * (f.router_f32 ? 4 : 2));
    L.gate   = std::make_unique<dev_buf>(gu_rows * row_bytes_of(f.tgu, H));
    L.up     = std::make_unique<dev_buf>(gu_rows * row_bytes_of(f.tgu, H));
    L.down   = std::make_unique<dev_buf>(dn_rows * row_bytes_of(f.tdn, F));
    L.gsh    = std::make_unique<dev_buf>((size_t) FS * row_bytes_of(f.tsh, H));
    L.ush    = std::make_unique<dev_buf>((size_t) FS * row_bytes_of(f.tsh, H));
    L.dsh    = std::make_unique<dev_buf>((size_t) H * row_bytes_of(f.tsh, FS));
    k_gen_norm<<<8, 256>>>(L.norm->as<float>(), H, seed * 16 + 1);
    k_gen_router<<<256, 256>>>(L.router->p, NE * H, f.router_f32, seed * 16 + 2);
    auto blocks = [](dev_buf & b, int type, size_t nrows, int K, uint64_t sd) {
        mb::gen_blocks<<<1024, 256>>>(b.as<uint8_t>(), nrows * (K / mb::type_qk(type)), type, sd);
    };
    blocks(*L.gate, f.tgu, gu_rows, H, seed * 16 + 3);
    blocks(*L.up, f.tgu, gu_rows, H, seed * 16 + 4);
    blocks(*L.down, f.tdn, dn_rows, F, seed * 16 + 5);
    blocks(*L.gsh, f.tsh, FS, H, seed * 16 + 6);
    blocks(*L.ush, f.tsh, FS, H, seed * 16 + 7);
    blocks(*L.dsh, f.tsh, H, FS, seed * 16 + 8);
    CUDA_CHECK(cudaGetLastError());
    L.w = { L.norm->as<float>(), L.router->p, L.gate->as<char>(), L.up->as<char>(), L.down->as<char>(),
            L.gsh->as<char>(), L.ush->as<char>(), L.dsh->as<char>() };
}

struct bufs_owner {
    std::vector<std::unique_ptr<dev_buf>> keep;
    layer_bufs b{};
    template <typename T> T * alloc(size_t n) {
        keep.push_back(std::make_unique<dev_buf>(n * sizeof(T)));
        return keep.back()->as<T>();
    }
    bufs_owner() {
        b.xn = alloc<float>(H);           b.yq = alloc<block_q8_1>(H / 32);
        b.logits = alloc<float>(NE);      b.ids = alloc<int32_t>(NE);       b.weights = alloc<float>(KU);
        b.act = alloc<float>(KU * F);     b.aq = alloc<block_q8_1>(KU * F / 32);
        b.dexp = alloc<float>(KU * H);    b.moe = alloc<float>(H);          b.yq2 = alloc<block_q8_1>(H / 32);
        b.sact = alloc<float>(FS);        b.sq = alloc<block_q8_1>(FS / 32);
        b.sd = alloc<float>(H);           b.out = alloc<float>(H);
    }
};

// ---- options -----------------------------------------------------------------------------------------------------------

struct options {
    std::string format = "s4l0";
    std::string router;          // f16 / f32 override
    int  sets = 4;               // distinct weight sets (layers) in memory
    int  chain = 47;             // layers per timed graph (Flint has 47 MoE layers)
    int  reps = 30;
    int  trials = 4;             // random inputs per weight set in the checks
    int  grid = 0;               // persistent grid override (0: occupancy x SMs)
    int  variant = -1;           // persistent kernel variant to time (-1: both)
    bool check_only = false;
    bool no_check = false;
    bool trace = true;
    const char * csv = nullptr;
};

void usage() {
    printf("usage: persistent_pascal [options]\n"
           "  --format s4l0|g3   weight formats of the layer (default s4l0)\n"
           "  --router f16|f32   override the format's router type\n"
           "  --sets N           distinct layers (weight sets) in GPU memory, used in turn (default 4; ~0.66 GB each)\n"
           "  --chain N          layers per timed CUDA graph, each fed the previous one's output (default 47)\n"
           "  --reps N           timed graph launches, median reported (default 30)\n"
           "  --trials N         random inputs per weight set in the checks (default 4)\n"
           "  --grid N           blocks of the persistent kernel (default: resident blocks per SM x SMs)\n"
           "  --variant r1|r2    time only this persistent variant (default both; the checks always run both)\n"
           "  --check-only       correctness checks only\n"
           "  --no-check         skip the checks (timing only)\n"
           "  --no-trace         skip the traced run of the persistent kernel\n"
           "  --csv FILE         write results as CSV\n");
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

__global__ void k_read(const uint4 * __restrict__ p, size_t n, unsigned * out) {
    unsigned acc = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = __ldg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) {
        out[0] = acc;
    }
}

// Capture `launch` once as a CUDA graph, launch it 3 times to warm up, then time `reps` launches (ms each).
std::vector<double> time_graph(const std::function<void(cudaStream_t)> & launch, int reps, cudaStream_t s) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    launch(s);
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
        t.push_back(ms);
    }
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaGraphExecDestroy(ge));
    CUDA_CHECK(cudaGraphDestroy(g));
    return t;
}

bool bytes_equal(const void * a, const void * b, size_t n) {
    return memcmp(a, b, n) == 0;
}

template <typename T>
double max_abs_diff(const std::vector<T> & a, const std::vector<T> & b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, fabs((double) a[i] - (double) b[i]));
    return m;
}

} // namespace


int main(int argc, char ** argv) {
    options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { usage(); exit(1); }
            return argv[++i];
        };
        if (a == "--format") opt.format = next();
        else if (a == "--router") opt.router = next();
        else if (a == "--sets") opt.sets = atoi(next());
        else if (a == "--chain") opt.chain = atoi(next());
        else if (a == "--reps") opt.reps = atoi(next());
        else if (a == "--trials") opt.trials = atoi(next());
        else if (a == "--grid") opt.grid = atoi(next());
        else if (a == "--variant") {
            const std::string v = next();
            opt.variant = v == "r1" ? 0 : v == "r2" ? 1 : -2;
            if (opt.variant == -2) { usage(); return 1; }
        }
        else if (a == "--check-only") opt.check_only = true;
        else if (a == "--no-check") opt.no_check = true;
        else if (a == "--no-trace") opt.trace = false;
        else if (a == "--csv") opt.csv = next();
        else { usage(); return a == "--help" || a == "-h" ? 0 : 1; }
    }
    format fmt{};
    bool found = false;
    for (const auto & f : FORMATS) {
        if (opt.format == f.name) { fmt = f; found = true; }
    }
    if (!found) { usage(); return 1; }
    if (opt.router == "f16") fmt.router_f32 = false;
    else if (opt.router == "f32") fmt.router_f32 = true;
    else if (!opt.router.empty()) { usage(); return 1; }
    opt.sets = std::max(1, opt.sets);
    opt.chain = std::max(1, opt.chain);
    opt.trials = std::max(1, opt.trials);

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int code_arch = 0;
    {
        dev_buf b(sizeof(int));
        k_arch<<<1, 1>>>(b.as<int>());
        CUDA_CHECK(cudaMemcpy(&code_arch, b.p, sizeof(int), cudaMemcpyDeviceToHost));
    }
    printf("device: %s, compute %d.%d, %d SMs, %d KB shared memory and %d registers per SM\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, (int) (prop.sharedMemPerMultiprocessor / 1024), prop.regsPerMultiprocessor);
    printf("device code: compiled for compute %d.%d%s\n", code_arch / 100, (code_arch % 100) / 10,
           code_arch / 10 == prop.major * 10 + prop.minor ? "" : " (JIT from PTX)");
    printf("layer: %s%s\n", fmt.what, opt.router.empty() ? "" : (fmt.router_f32 ? "; router F32 by option" : "; router F16 by option"));
    const size_t lbytes = layer_bytes(fmt);
    printf("weights read per layer and token: %.3f MB\n", lbytes / 1e6);

    FILE * csv = opt.csv ? fopen(opt.csv, "w") : nullptr;
    if (opt.csv && !csv) { fprintf(stderr, "cannot write %s\n", opt.csv); return 1; }
    if (csv) fprintf(csv, "device,code_arch,format,router,section,name,value,unit\n");
    auto csv_row = [&](const char * section, const std::string & name, double value, const char * unit) {
        if (csv) fprintf(csv, "%s,%d,%s,%s,%s,%s,%.6g,%s\n", prop.name, code_arch, fmt.name, fmt.router_f32 ? "f32" : "f16", section,
                         name.c_str(), value, unit);
    };

    const ops op = ops_for(fmt);

    // ---- occupancy ----------------------------------------------------------------------------------------------------
    int occ[NV], grid_default[NV], grid_of[NV];
    for (int v = 0; v < NV; ++v) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ[v], op.pk[v], PK_THREADS, 0));
        cudaFuncAttributes fa;
        CUDA_CHECK(cudaFuncGetAttributes(&fa, op.pk[v]));
        grid_default[v] = std::max(1, occ[v]) * prop.multiProcessorCount;
        grid_of[v] = opt.grid > 0 ? opt.grid : grid_default[v];
        const int by_regs = prop.regsPerMultiprocessor / std::max(1, fa.numRegs * PK_THREADS);
        const int by_smem = (int) (prop.sharedMemPerMultiprocessor / std::max<size_t>(1, fa.sharedSizeBytes));
        const int by_thr  = prop.maxThreadsPerMultiProcessor / PK_THREADS;
        printf("\npersistent kernel %s (%s):\n", VARIANT_NAMES[v], VARIANT_WHAT[v]);
        printf("  %d threads per block, %d registers per thread, %d B static shared memory per block\n", PK_THREADS, fa.numRegs,
               (int) fa.sharedSizeBytes);
        printf("  resident blocks per SM: %d (registers allow about %d, shared memory %d, threads %d); %d warps per SM\n", occ[v],
               by_regs, by_smem, by_thr, occ[v] * PK_NW);
        printf("  grid: %d blocks = %d warps%s for %d work items\n", grid_of[v], grid_of[v] * PK_NW,
               opt.grid > 0 ? " (set by --grid)" : "", N_ITEMS);
        csv_row("occupancy", std::string(VARIANT_NAMES[v]) + " registers", fa.numRegs, "count");
        csv_row("occupancy", std::string(VARIANT_NAMES[v]) + " smem_per_block", (double) fa.sharedSizeBytes, "B");
        csv_row("occupancy", std::string(VARIANT_NAMES[v]) + " blocks_per_sm", occ[v], "count");
        csv_row("occupancy", std::string(VARIANT_NAMES[v]) + " grid", grid_of[v], "count");
        if (occ[v] < 1) {
            printf("  it cannot be resident on this device; the multi-launch path runs instead\n");
        }
    }
    printf("\nmulti-launch kernels (4 warps per block for the products):\n");
    for (auto & k : op.base_kernels) {
        cudaFuncAttributes ka;
        CUDA_CHECK(cudaFuncGetAttributes(&ka, k.second));
        printf("  %-16s %3d registers, %5d B shared memory\n", k.first.c_str(), ka.numRegs, (int) ka.sharedSizeBytes);
    }

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    // ---- data -------------------------------------------------------------------------------------------------------------
    std::vector<layer_set> sets(opt.sets);
    for (int l = 0; l < opt.sets; ++l) {
        make_layer(sets[l], fmt, 1000 + l);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    bufs_owner base_b, pers_b;
    dev_buf sync_buf(N_SYNC * sizeof(unsigned));
    dev_buf trace_buf(TR_N * sizeof(unsigned long long));
    dev_buf dbg_xn(H * sizeof(float)), dbg_yq(H / 32 * sizeof(block_q8_1));
    dev_buf xin(H * sizeof(float));
    unsigned * sync = sync_buf.as<unsigned>();

    auto pk_make = [&](const layer_weights & w, const float * x, const layer_bufs & b, const bool dbg) {
        pk_args a{};
        a.w = w; a.x = x; a.b = b; a.sync = sync;
        a.dbg_xn = dbg ? dbg_xn.as<float>() : nullptr;
        a.dbg_yq = dbg ? dbg_yq.as<block_q8_1>() : nullptr;
        a.trace = trace_buf.as<unsigned long long>();
        a.expert_scale = nullptr;
        a.ncols = H; a.eps = EPS; a.clamp_val = CLAMP; a.scale_val = 1.0f;
        a.spin_limit = 1 << 22; a.fault = 0;
        return a;
    };
    auto read_err = [&]() {
        unsigned e = 0;
        CUDA_CHECK(cudaMemcpy(&e, sync + S_ERR, sizeof(unsigned), cudaMemcpyDeviceToHost));
        return e;
    };
    // The persistent kernel with the fallback: if it cannot be resident or a wait timed out, reset the counters and run
    // the multi-launch path into the same buffers. Returns true if the persistent kernel produced the output.
    auto run_layer = [&](const int v, const pk_args & a, const int g) -> bool {
        if (occ[v] >= 1) {
            launch_pk(op.pk[v], a, g, s);
            CUDA_CHECK(cudaStreamSynchronize(s));
            if (read_err() == 0) {
                return true;
            }
            CUDA_CHECK(cudaMemset(sync, 0, N_SYNC * sizeof(unsigned)));
        }
        op.baseline(a.w, a.x, a.b, s, 0x1fff);
        CUDA_CHECK(cudaStreamSynchronize(s));
        return false;
    };
    auto gen_input = [&](float * dst, uint64_t seed) {
        mb::gen_floats<<<8, 256, 0, s>>>(dst, H, seed);
        CUDA_CHECK(cudaGetLastError());
    };

    bool all_ok = true;

    // ---- checks -----------------------------------------------------------------------------------------------------------
    if (!opt.no_check) {
        printf("\n== checks ==\n");
        check_report rep_base, rep_pers[NV];
        int bit_identical[NV] = {}, bit_total = 0;
        std::vector<std::string> diffs;
        double e2e_worst = 0.0, e2e_sum = 0.0, margin_min = 1.0;
        int e2e_n = 0, e2e_set_mismatch = 0;
        std::vector<float> xh(H);
        for (int l = 0; l < opt.sets; ++l) {
            host_layer HL;
            HL.load(sets[l].w, fmt.tgu, fmt.tdn, fmt.tsh, fmt.router_f32);
            for (int t = 0; t < opt.trials; ++t) {
                HL.drop_experts();
                gen_input(xin.as<float>(), 77 + 1000 * l + t);
                CUDA_CHECK(cudaMemcpyAsync(xh.data(), xin.p, H * sizeof(float), cudaMemcpyDeviceToHost, s));
                op.baseline(sets[l].w, xin.as<float>(), base_b.b, s, 0x1fff);
                CUDA_CHECK(cudaStreamSynchronize(s));
                host_out ob;
                ob.fetch(base_b.b, base_b.b.xn, base_b.b.yq);
                std::vector<block_q8_1> yq2(H / 32);
                CUDA_CHECK(cudaMemcpy(yq2.data(), base_b.b.yq2, yq2.size() * sizeof(block_q8_1), cudaMemcpyDeviceToHost));
                if (!bytes_equal(ob.yq.data(), yq2.data(), H / 32 * sizeof(block_q8_1))) {
                    diffs.push_back("multi-launch path: its two quantizations of the normed input differ");
                    all_ok = false;
                }
                check_stages(rep_base, HL, xh, ob, false);
                ++bit_total;
                for (int v = 0; v < NV; ++v) {
                    CUDA_CHECK(cudaMemsetAsync(pers_b.b.out, 0, H * 4, s));
                    if (!run_layer(v, pk_make(sets[l].w, xin.as<float>(), pers_b.b, true), grid_of[v])) {
                        printf("  %s, set %d trial %d: the persistent kernel did not complete (fallback used)\n", VARIANT_NAMES[v], l, t);
                        all_ok = false;
                    }
                    host_out opk;
                    opk.fetch(pers_b.b, dbg_xn.as<float>(), dbg_yq.as<block_q8_1>());
                    // bit for bit against the multi-launch path
                    bool same = true;
                    auto cmp = [&](const char * what, const void * a, const void * b, size_t n, double mad) {
                        if (!bytes_equal(a, b, n)) {
                            same = false;
                            char msg[200];
                            snprintf(msg, sizeof msg, "%s, set %d trial %d: %s differs (max abs diff %.3g)", VARIANT_NAMES[v], l, t, what, mad);
                            diffs.push_back(msg);
                        }
                    };
                    cmp("normed input", ob.xn.data(), opk.xn.data(), H * 4, max_abs_diff(ob.xn, opk.xn));
                    cmp("its Q8_1 copy", ob.yq.data(), opk.yq.data(), H / 32 * sizeof(block_q8_1), 0);
                    cmp("router logits", ob.logits.data(), opk.logits.data(), NE * 4, max_abs_diff(ob.logits, opk.logits));
                    cmp("expert ids", ob.ids.data(), opk.ids.data(), KU * 4, 0);
                    cmp("routing weights", ob.weights.data(), opk.weights.data(), KU * 4, max_abs_diff(ob.weights, opk.weights));
                    cmp("routed SwiGLU", ob.act.data(), opk.act.data(), KU * F * 4, max_abs_diff(ob.act, opk.act));
                    cmp("its Q8_1 copy", ob.aq.data(), opk.aq.data(), KU * F / 32 * sizeof(block_q8_1), 0);
                    cmp("routed down", ob.dexp.data(), opk.dexp.data(), (size_t) KU * H * 4, max_abs_diff(ob.dexp, opk.dexp));
                    cmp("shared SwiGLU", ob.sact.data(), opk.sact.data(), FS * 4, max_abs_diff(ob.sact, opk.sact));
                    cmp("its Q8_1 copy", ob.sq.data(), opk.sq.data(), FS / 32 * sizeof(block_q8_1), 0);
                    cmp("output", ob.out.data(), opk.out.data(), H * 4, max_abs_diff(ob.out, opk.out));
                    bit_identical[v] += same;
                    check_stages(rep_pers[v], HL, xh, opk, true);
                }
                // whole layer in double without activation quantization
                std::vector<int> ids_ref;
                double margin = 0.0;
                const std::vector<double> ref = ref_layer_exact(HL, xh, ids_ref, &margin);
                margin_min = std::min(margin_min, margin);
                std::vector<int> ia(ob.ids.begin(), ob.ids.end()), ib(ids_ref);
                std::sort(ia.begin(), ia.end());
                std::sort(ib.begin(), ib.end());
                if (ia != ib) {
                    ++e2e_set_mismatch;
                } else {
                    double num = 0.0, den = 0.0;
                    for (int i = 0; i < H; ++i) {
                        const double ffn_ref = ref[i] - xh[i];
                        const double ffn_gpu = (double) ob.out[i] - xh[i];
                        num += (ffn_gpu - ffn_ref) * (ffn_gpu - ffn_ref);
                        den += ffn_ref * ffn_ref;
                    }
                    const double rel = sqrt(num / den);
                    e2e_worst = std::max(e2e_worst, rel);
                    e2e_sum += rel;
                    ++e2e_n;
                }
            }
        }
        printf("persistent kernel against the multi-launch path, every intermediate and the output compared bit for bit:\n");
        for (int v = 0; v < NV; ++v) {
            printf("  %s: %d of %d layer evaluations identical\n", VARIANT_NAMES[v], bit_identical[v], bit_total);
            csv_row("check", std::string(VARIANT_NAMES[v]) + " bit_identical_evaluations", bit_identical[v], "count");
            if (bit_identical[v] != bit_total) all_ok = false;
        }
        csv_row("check", "evaluations", bit_total, "count");
        for (size_t i = 0; i < diffs.size() && i < 16; ++i) printf("  %s\n", diffs[i].c_str());

        auto print_rep = [&](const std::string & title, check_report & rep) {
            printf("%s, against double precision from its own stage inputs (worst error / tolerance, must be <= 1):\n", title.c_str());
            for (auto & st : rep.st) {
                const std::string note = st.notes ? ", " + std::to_string(st.notes) + " at a rounding boundary or near-tie" : "";
                printf("  %-20s %8.3f  (%ld values%s%s)\n", st.name.c_str(), st.worst, st.n, st.fails ? ", FAIL" : "", note.c_str());
                csv_row("stage", title + ": " + st.name, st.worst, "err/tol");
            }
            if (!rep.ok()) all_ok = false;
        };
        print_rep("multi-launch path", rep_base);
        for (int v = 0; v < NV; ++v) print_rep(std::string("persistent ") + VARIANT_NAMES[v], rep_pers[v]);
        printf("whole layer in double precision without activation quantization (the error of the Q8_1 activations):\n");
        printf("  relative L2 error of the layer's update (output - input): mean %.4f, worst %.4f over %d evaluations\n",
               e2e_n ? e2e_sum / e2e_n : 0.0, e2e_worst, e2e_n);
        printf("  expert set differs from the exact one in %d of %d (smallest relative gap between 6th and 7th score %.2g)\n",
               e2e_set_mismatch, bit_total, margin_min);
        csv_row("check", "e2e_rel_l2_mean", e2e_n ? e2e_sum / e2e_n : 0.0, "ratio");
        csv_row("check", "e2e_rel_l2_worst", e2e_worst, "ratio");
        if (e2e_worst > 0.05) {
            printf("  FAIL: above 0.05\n");
            all_ok = false;
        }

        // grid sizes: one block, two, one per SM, the default, and more blocks than can be resident at once
        gen_input(xin.as<float>(), 4242);
        CUDA_CHECK(cudaStreamSynchronize(s));
        std::vector<float> want(H), o(H);
        op.baseline(sets[0].w, xin.as<float>(), base_b.b, s, 0x1fff);
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaMemcpy(want.data(), base_b.b.out, H * 4, cudaMemcpyDeviceToHost));
        for (int v = 0; v < NV; ++v) {
            printf("%s, grid sizes (output against the multi-launch path's, bit for bit):\n", VARIANT_NAMES[v]);
            const int grids[] = { 1, 2, prop.multiProcessorCount, grid_default[v], 4 * grid_default[v] + 3, 64 * prop.multiProcessorCount };
            for (int g : grids) {
                CUDA_CHECK(cudaMemset(pers_b.b.out, 0, H * 4));
                const bool okp = run_layer(v, pk_make(sets[0].w, xin.as<float>(), pers_b.b, false), g);
                CUDA_CHECK(cudaMemcpy(o.data(), pers_b.b.out, H * 4, cudaMemcpyDeviceToHost));
                const bool same = bytes_equal(o.data(), want.data(), H * 4);
                printf("  %5d blocks (%6d warps): %s%s\n", g, g * PK_NW, same ? "identical" : "DIFFERENT",
                       okp ? "" : " (persistent kernel failed, fallback used)");
                if (!same || !okp) all_ok = false;
            }
            unsigned resid[N_SYNC];
            CUDA_CHECK(cudaMemcpy(resid, sync, sizeof resid, cudaMemcpyDeviceToHost));
            int nonzero = 0;
            for (int i = 0; i < N_SYNC; ++i) nonzero += resid[i] != 0;
            printf("  counters after these runs: %s\n", nonzero ? "NOT reset" : "all zero (reset by the last warp out)");
            if (nonzero) all_ok = false;

            // fault injection: the top-6 is never published, so every routed item's wait must time out and the host
            // must fall back to the multi-launch path, with the same output
            pk_args a = pk_make(sets[0].w, xin.as<float>(), pers_b.b, false);
            a.fault = 1;
            a.spin_limit = 1 << 16;
            CUDA_CHECK(cudaMemset(pers_b.b.out, 0, H * 4));
            const bool okp = run_layer(v, a, grid_default[v]);
            CUDA_CHECK(cudaMemcpy(o.data(), pers_b.b.out, H * 4, cudaMemcpyDeviceToHost));
            const bool same = bytes_equal(o.data(), want.data(), H * 4);
            printf("  fault injection (top-6 never published): %s, fallback output %s\n", okp ? "NOT DETECTED" : "timeout detected",
                   same ? "correct" : "WRONG");
            if (okp || !same) all_ok = false;
            CUDA_CHECK(cudaMemset(pers_b.b.out, 0, H * 4));
            const bool again = run_layer(v, pk_make(sets[0].w, xin.as<float>(), pers_b.b, false), grid_default[v]);
            CUDA_CHECK(cudaMemcpy(o.data(), pers_b.b.out, H * 4, cudaMemcpyDeviceToHost));
            const bool same2 = again && bytes_equal(o.data(), want.data(), H * 4);
            printf("  next launch after the fault: %s\n", same2 ? "correct" : "WRONG");
            if (!same2) all_ok = false;
        }

        // chains of layers, each fed the previous output, as in a token
        {
            const int L = std::min(opt.chain, 16);
            dev_buf xa((size_t) (L + 1) * H * 4), xb((size_t) (L + 1) * H * 4);
            gen_input(xa.as<float>(), 99);
            for (int i = 0; i < L; ++i) {
                layer_bufs bb = base_b.b;
                bb.out = xa.as<float>() + (size_t) (i + 1) * H;
                op.baseline(sets[i % opt.sets].w, xa.as<float>() + (size_t) i * H, bb, s, 0x1fff);
            }
            std::vector<float> oa((size_t) (L + 1) * H), ob2((size_t) (L + 1) * H);
            CUDA_CHECK(cudaMemcpy(oa.data(), xa.p, oa.size() * 4, cudaMemcpyDeviceToHost));
            double rms = 0.0;
            for (int i = 0; i < H; ++i) rms += (double) oa[(size_t) L * H + i] * oa[(size_t) L * H + i];
            for (int v = 0; v < NV; ++v) {
                CUDA_CHECK(cudaMemcpyAsync(xb.p, xa.p, H * 4, cudaMemcpyDeviceToDevice, s));
                bool okp = true;
                for (int i = 0; i < L; ++i) {
                    layer_bufs bp = pers_b.b;
                    bp.out = xb.as<float>() + (size_t) (i + 1) * H;
                    okp &= run_layer(v, pk_make(sets[i % opt.sets].w, xb.as<float>() + (size_t) i * H, bp, false), grid_of[v]);
                }
                CUDA_CHECK(cudaMemcpy(ob2.data(), xb.p, ob2.size() * 4, cudaMemcpyDeviceToHost));
                const bool same = bytes_equal(oa.data(), ob2.data(), oa.size() * 4);
                printf("%s, chain of %d layers: every layer's output %s (final rms %.1f)%s\n", VARIANT_NAMES[v], L,
                       same ? "identical" : "DIFFERENT", sqrt(rms / H), okp ? "" : ", persistent kernel failed");
                if (!same || !okp) all_ok = false;
            }
        }
        printf("checks: %s\n", all_ok ? "PASS" : "FAIL");
        csv_row("check", "pass", all_ok ? 1 : 0, "bool");
    }

    // ---- timing -------------------------------------------------------------------------------------------------------------
    if (!opt.check_only) {
        printf("\n== timing ==\n");
        const int L = opt.chain;
        dev_buf xs((size_t) (L + 1) * H * 4);
        gen_input(xs.as<float>(), 4321);
        CUDA_CHECK(cudaStreamSynchronize(s));
        {
            const size_t nbytes = (size_t) 256 << 20;
            dev_buf rb(nbytes), ro(64);
            const int g = prop.multiProcessorCount * 16;
            auto rd = [&](cudaStream_t st) {
                for (int j = 0; j < 4; ++j) k_read<<<g, 256, 0, st>>>(rb.as<uint4>(), nbytes / 16, ro.as<unsigned>());
            };
            double warm_ms = 0.0;
            while (warm_ms < 2000.0) {
                for (double x : time_graph(rd, 4, s)) warm_ms += x;
            }
            const double node_us = median(time_graph([](cudaStream_t st) { for (int j = 0; j < 64; ++j) k_empty<<<1, 32, 0, st>>>(); }, 20, s)) * 1000.0 / 64;
            const double read_gbs = 4.0 * nbytes / (median(time_graph(rd, 10, s)) * 1e6);
            printf("empty kernel in a CUDA graph: %.2f us per node; streaming read: %.1f GB/s\n", node_us, read_gbs);
            csv_row("reference", "graph_node_us", node_us, "us");
            csv_row("reference", "stream_read_gbs", read_gbs, "GB/s");
        }
        auto chain_base = [&](cudaStream_t st) {
            for (int i = 0; i < L; ++i) {
                layer_bufs bb = base_b.b;
                bb.out = xs.as<float>() + (size_t) (i + 1) * H;
                op.baseline(sets[i % opt.sets].w, xs.as<float>() + (size_t) i * H, bb, st, 0x1fff);
            }
        };
        CUDA_CHECK(cudaMemset(sync, 0, N_SYNC * sizeof(unsigned)));
        printf("%d layers per CUDA graph, %d weight sets in turn, median of %d graph launches:\n", L, opt.sets, opt.reps);
        const std::vector<double> tb = time_graph(chain_base, opt.reps, s);
        const double ub = median(tb) * 1000.0 / L, ubmin = *std::min_element(tb.begin(), tb.end()) * 1000.0 / L;
        printf("  %-34s %8.1f us per layer (min %.1f), %6.1f GB/s\n", "multi-launch (13 kernels)", ub, ubmin, lbytes / (ub * 1e3));
        csv_row("timing", "multi_launch us_per_layer", ub, "us");
        csv_row("timing", "multi_launch us_per_layer_min", ubmin, "us");
        for (int v = 0; v < NV; ++v) {
            if ((opt.variant >= 0 && v != opt.variant) || occ[v] < 1) continue;
            auto chain_pk = [&](cudaStream_t st) {
                for (int i = 0; i < L; ++i) {
                    layer_bufs bp = pers_b.b;
                    bp.out = xs.as<float>() + (size_t) (i + 1) * H;
                    launch_pk(op.pk[v], pk_make(sets[i % opt.sets].w, xs.as<float>() + (size_t) i * H, bp, false), grid_of[v], st);
                }
            };
            const std::vector<double> tp = time_graph(chain_pk, opt.reps, s);
            if (read_err() != 0) {
                printf("  %s reported a timeout during timing: its times are invalid\n", VARIANT_NAMES[v]);
                all_ok = false;
                CUDA_CHECK(cudaMemset(sync, 0, N_SYNC * sizeof(unsigned)));
                continue;
            }
            const double up = median(tp) * 1000.0 / L, upmin = *std::min_element(tp.begin(), tp.end()) * 1000.0 / L;
            char name[64];
            snprintf(name, sizeof name, "persistent %s (%d blocks)", VARIANT_NAMES[v], grid_of[v]);
            printf("  %-34s %8.1f us per layer (min %.1f), %6.1f GB/s; %.3fx, saves %.1f us per layer = %.2f ms per token (47 layers)\n",
                   name, up, upmin, lbytes / (up * 1e3), ub / up, ub - up, (ub - up) * 47 / 1000.0);
            csv_row("timing", std::string(VARIANT_NAMES[v]) + " us_per_layer", up, "us");
            csv_row("timing", std::string(VARIANT_NAMES[v]) + " us_per_layer_min", upmin, "us");
            csv_row("timing", std::string(VARIANT_NAMES[v]) + " speedup", ub / up, "x");
        }
        csv_row("timing", "bytes_per_layer", (double) lbytes, "B");

        // the multi-launch kernels one kind at a time (L launches each, same inputs)
        printf("multi-launch path, one kernel at a time (%d launches per graph, us per launch):\n", L);
        double sum = 0.0;
        for (int k = 0; k < 13; ++k) {
            auto one = [&](cudaStream_t st) {
                for (int i = 0; i < L; ++i) {
                    layer_bufs bb = base_b.b;
                    bb.out = xs.as<float>() + (size_t) (i + 1) * H;
                    op.baseline(sets[i % opt.sets].w, xs.as<float>() + (size_t) i * H, bb, st, 1u << k);
                }
            };
            const double us = median(time_graph(one, std::max(5, opt.reps / 3), s)) * 1000.0 / L;
            sum += us;
            printf("  %2d %-20s %7.1f\n", k + 1, STEP_NAMES[k], us);
            csv_row("multi_launch_kernel", STEP_NAMES[k], us, "us");
        }
        printf("     %-20s %7.1f (the whole layer in one graph: %.1f)\n", "sum", sum, ub);

        // traced runs: where the persistent kernel's time goes
        for (int v = 0; v < NV && opt.trace; ++v) {
            if ((opt.variant >= 0 && v != opt.variant) || occ[v] < 1) continue;
            std::vector<double> acc(TR_N, 0.0);
            unsigned long long tr[TR_N];
            const int nt = std::min(L, 16);
            for (int i = 0; i < nt; ++i) {
                for (int j = 0; j < TR_N; ++j) tr[j] = 0;
                for (int c = 0; c < 5; ++c) tr[TR_CLASS + 4 * c] = ULLONG_MAX;
                tr[TR_START] = tr[TR_EACT_FIRST] = tr[TR_COMB_FIRST] = ULLONG_MAX;
                CUDA_CHECK(cudaMemcpy(trace_buf.p, tr, sizeof tr, cudaMemcpyHostToDevice));
                layer_bufs bp = pers_b.b;
                bp.out = xs.as<float>() + (size_t) (i + 1) * H;
                launch_pk(op.pk_trace[v], pk_make(sets[i % opt.sets].w, xs.as<float>() + (size_t) i * H, bp, false), grid_of[v], s);
                CUDA_CHECK(cudaStreamSynchronize(s));
                CUDA_CHECK(cudaMemcpy(tr, trace_buf.p, sizeof tr, cudaMemcpyDeviceToHost));
                const double t0 = (double) tr[TR_START];
                for (int j = 0; j < TR_N; ++j) {
                    const bool is_count = j < 20 && (j % 4 == 2 || j % 4 == 3);
                    acc[j] += is_count ? (double) tr[j] : ((double) tr[j] - t0) / 1000.0;
                }
            }
            for (auto & x : acc) x /= nt;
            printf("persistent %s, traced (%d single launches; us from the first block's start, %%globaltimer):\n", VARIANT_NAMES[v], nt);
            printf("  prologue (norm, quantization) done by %7.1f\n", acc[TR_PROLOGUE]);
            const char * cls[5] = { "router", "shared gate/up", "routed gate/up", "shared down", "routed down" };
            for (int c = 0; c < 5; ++c) {
                printf("  %-16s %7.1f .. %7.1f   %5.0f items, %8.0f wait polls\n", cls[c], acc[TR_CLASS + 4 * c], acc[TR_CLASS + 4 * c + 1],
                       acc[TR_CLASS + 4 * c + 3], acc[TR_CLASS + 4 * c + 2]);
                const std::string pre = std::string(VARIANT_NAMES[v]) + " " + cls[c];
                csv_row("trace", pre + " start", acc[TR_CLASS + 4 * c], "us");
                csv_row("trace", pre + " end", acc[TR_CLASS + 4 * c + 1], "us");
                csv_row("trace", pre + " wait_polls", acc[TR_CLASS + 4 * c + 2], "count");
            }
            printf("  top-6 published %7.1f, shared activations %7.1f, routed activations %7.1f .. %7.1f\n", acc[TR_TOPK], acc[TR_SACT],
                   acc[TR_EACT_FIRST], acc[TR_EACT_LAST]);
            printf("  output groups combined %7.1f .. %7.1f, last warp out %7.1f\n", acc[TR_COMB_FIRST], acc[TR_COMB_LAST], acc[TR_EXIT]);
            csv_row("trace", std::string(VARIANT_NAMES[v]) + " top6", acc[TR_TOPK], "us");
            csv_row("trace", std::string(VARIANT_NAMES[v]) + " end", acc[TR_EXIT], "us");
        }
    }
    if (csv) fclose(csv);
    CUDA_CHECK(cudaStreamDestroy(s));
    return all_ok ? 0 : 2;
}
