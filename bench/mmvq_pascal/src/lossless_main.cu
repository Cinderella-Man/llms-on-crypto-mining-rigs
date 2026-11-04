// lossless_bench: Qwen3.8-Flash-Next's BF16 matrix-vector products over real weights, in plain BF16 and in the two
// lossless codes of lossless_bf16.h (Q621), at 1 to 8 tokens. See ../README.md, "Lossless BF16 (Q621)".
//
// The weights are a sample of the model's own tensors (lossless_entropy --sample): per shape class, each tensor is
// coded once on the CPU, decoded back and checked weight for weight, then copied to the card as often as it takes to
// hold >= --min-mb MB of plain BF16 per class, so consecutive calls cycle through copies far larger than the 1.5 MB L2.
// Variants per class and token count:
//   plain/pascal  Miner Llama's Pascal BF16 kernel (bf16_mv.cuh) at the launch shape the fork picks for the card;
//   plain/gG_rR   the same kernel at G lanes per row group and R rows per group;
//   x12/gG_rR     x12, the same kernel with the weights decoded in registers: at the plain kernel's own (G, R) its output
//                 must equal the plain kernel's bit for bit ("same" column);
//   x11/gG_rR     x11, 32-weight chunks;
//   read/<code>   a stream over the code's bytes per call, nothing computed: the card's read speed for that size.
// Every variant is checked on the first tensor against a double-precision product of the same BF16 weights and FP32
// activations (every row up to 4,096 rows, else every 37th; tolerance 1e-5 of the sum of the absolute terms). The
// summary adds calls per token x the fastest correct variant of each code per class, against the plain kernel's pick.

#include "common.cuh"
#include "data.cuh"
#include "lossless_bf16.h"
#include "lossless_mv.cuh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace {

using namespace mb;
using lossless::encoded;

struct dev_buf {
    void * p = nullptr;
    size_t n = 0;
    dev_buf() = default;
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

template <typename X> dev_buf upload(const std::vector<X> & v) {
    dev_buf b(std::max<size_t>(v.size() * sizeof(X), 4));
    if (!v.empty()) CUDA_CHECK(cudaMemcpy(b.p, v.data(), v.size() * sizeof(X), cudaMemcpyHostToDevice));
    return b;
}

struct tensor { std::string name; std::vector<uint16_t> w; encoded e12, e11; };
struct klass { std::string name; int K = 0, N = 0, calls = 0; std::vector<tensor> t; };

// one coded copy on the card
struct dev_code { dev_buf lo, idx, rowinfo, esc; size_t bytes = 0; };
struct dev_copy { dev_buf plain; dev_code c12, c11; };

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

__global__ void k_arch(int * out) {
#ifdef __CUDA_ARCH__
    *out = __CUDA_ARCH__;
#endif
}

bool load_sample(const std::string & path, std::vector<klass> & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { perror(path.c_str()); return false; }
    uint32_t hdr[2];
    if (fread(hdr, 4, 2, f) != 2 || hdr[0] != 0x3142424cu) { fprintf(stderr, "%s: not a lossless_entropy sample\n", path.c_str()); return false; }
    for (uint32_t i = 0; i < hdr[1]; ++i) {
        char name[65] = { 0 }, cls[17] = { 0 };
        uint32_t meta[3];
        if (fread(name, 1, 64, f) != 64 || fread(cls, 1, 16, f) != 16 || fread(meta, 4, 3, f) != 3) return false;
        klass * k = nullptr;
        for (auto & c : out) if (c.name == cls) k = &c;
        if (!k) {
            out.push_back(klass());
            k = &out.back();
            k->name = cls; k->K = (int) meta[0]; k->N = (int) meta[1]; k->calls = (int) meta[2];
        }
        if (k->K != (int) meta[0] || k->N != (int) meta[1]) { fprintf(stderr, "%s: class %s mixes shapes\n", path.c_str(), cls); return false; }
        tensor t;
        t.name = name;
        t.w.resize((size_t) meta[0] * meta[1]);
        if (fread(t.w.data(), 2, t.w.size(), f) != t.w.size()) return false;
        k->t.push_back(std::move(t));
    }
    fclose(f);
    return true;
}

dev_code to_device(const encoded & e) {
    dev_code d;
    d.lo = upload(e.lo);
    d.idx = upload(e.idx);
    d.rowinfo = upload(e.rowinfo);
    d.esc = upload(e.esc);
    d.bytes = e.bytes();
    return d;
}

void usage() {
    printf("usage: lossless_bench --sample FILE [options]\n"
           "  --sample FILE    the weight sample written by lossless_entropy --sample\n"
           "  --classes LIST   shape classes (default all in the sample)\n"
           "  --codes LIST     plain,x12,x11,read (default all)\n"
           "  --tokens LIST    tokens per product, 1 to 8 (default 1,2,3,4,5,6,8)\n"
           "  --reps N         timed graph launches, median reported (default 20)\n"
           "  --batch N        calls per CUDA graph (default 64)\n"
           "  --min-mb N       plain BF16 megabytes per class on the card, by repeating the sample (default 32)\n"
           "  --nsm N          the plain kernel's launch shape as for a card of N SMs (default: this card's)\n"
           "  --cpu-only       the codes' CPU round trip only, no GPU\n"
           "  --check-only     correctness only, no timing\n"
           "  --csv FILE       every variant as CSV\n"
           "  --summary FILE   ms per decoding step per code and token count as CSV\n"
           "  --seed N         random seed of the activations (default 1234)\n");
}

} // namespace

int main(int argc, char ** argv) {
    std::string sample_path, csv, summary_path;
    std::vector<std::string> classes, codes = { "plain", "x12", "x11", "read" }, tokens = { "1", "2", "3", "4", "5", "6", "8" };
    int reps = 20, batch = 64, min_mb = 32, nsm_force = 0;
    bool cpu_only = false, check_only = false;
    uint64_t seed = 1234;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(); exit(1); } return argv[++i]; };
        if (a == "--sample") sample_path = next();
        else if (a == "--classes") classes = split(next());
        else if (a == "--codes") codes = split(next());
        else if (a == "--tokens") tokens = split(next());
        else if (a == "--reps") reps = atoi(next().c_str());
        else if (a == "--batch") batch = atoi(next().c_str());
        else if (a == "--min-mb") min_mb = atoi(next().c_str());
        else if (a == "--nsm") nsm_force = atoi(next().c_str());
        else if (a == "--cpu-only") cpu_only = true;
        else if (a == "--check-only") check_only = true;
        else if (a == "--csv") csv = next();
        else if (a == "--summary") summary_path = next();
        else if (a == "--seed") seed = strtoull(next().c_str(), nullptr, 10);
        else { usage(); return a == "--help" ? 0 : 1; }
    }
    if (sample_path.empty()) { usage(); return 1; }
    std::vector<int> cols;
    for (const auto & t : tokens) {
        const int n = atoi(t.c_str());
        if (n < 1 || n > 8) { fprintf(stderr, "--tokens: 1 to 8\n"); return 1; }
        cols.push_back(n);
    }

    std::vector<klass> all;
    if (!load_sample(sample_path, all)) return 1;
    std::vector<klass> ks;
    for (auto & k : all) if (classes.empty() || contains(classes, k.name)) ks.push_back(std::move(k));

    // code every tensor and decode it back with the kernels' byte steps
    bool ok = true;
    printf("%-10s %6s %6s %5s %3s %9s %9s %9s %8s %8s %s\n", "class", "K", "N", "calls", "n", "x12 bits", "x11 bits",
           "x12 esc", "x11 esc", "", "round trip");
    for (auto & k : ks) {
        size_t esc12 = 0, esc11 = 0, b12 = 0, b11 = 0, bad = 0;
        std::vector<uint16_t> back(k.K);
        for (auto & t : k.t) {
            t.e12 = lossless::encode(t.w.data(), k.K, k.N, 12);
            t.e11 = lossless::encode(t.w.data(), k.K, k.N, 11);
            for (const encoded * e : { &t.e12, &t.e11 }) {
                for (int r = 0; r < k.N; ++r) {
                    lossless::decode_row(*e, r, back.data());
                    bad += memcmp(back.data(), t.w.data() + (size_t) r * k.K, (size_t) k.K * 2) != 0;
                }
            }
            esc12 += t.e12.esc.size(); esc11 += t.e11.esc.size(); b12 += t.e12.bytes(); b11 += t.e11.bytes();
        }
        const double nw = (double) k.K * k.N * k.t.size();
        printf("%-10s %6d %6d %5d %3zu %9.3f %9.3f %8.4f%% %7.3f%% %8s %s\n", k.name.c_str(), k.K, k.N, k.calls, k.t.size(),
               8.0 * b12 / nw, 8.0 * b11 / nw, 100.0 * esc12 / nw, 100.0 * esc11 / nw, "", bad ? "FAIL" : "PASS");
        ok &= bad == 0;
    }
    if (cpu_only) {
        printf("\nround trip of every weight: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int * arch_d = nullptr;
    CUDA_CHECK(cudaMalloc(&arch_d, 4));
    k_arch<<<1, 1>>>(arch_d);
    int code_arch = 0;
    CUDA_CHECK(cudaMemcpy(&code_arch, arch_d, 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(arch_d));
    const int nsm = nsm_force > 0 ? nsm_force : prop.multiProcessorCount;
    printf("\ndevice: %s, compute %d.%d, %d SMs (plain shapes chosen for %d), code arch %d\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount, nsm, code_arch);
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    FILE * f = nullptr;
    if (!csv.empty()) {
        f = fopen(csv.c_str(), "w");
        if (!f) { fprintf(stderr, "cannot write %s\n", csv.c_str()); return 1; }
        fprintf(f, "device,code_arch,class,K,N,calls_per_token,tokens,code,variant,G,R,regs,grid,bytes_per_call,bits_per_weight,"
                   "us_median,us_min,gbs,ps_per_weight,err_over_tol,check,same_as_plain\n");
    }
    printf("\n%-10s %2s %-5s %-8s %3s %2s %5s %5s %9s %6s %9s %9s %7s %7s %8s %-5s %s\n", "class", "k", "code", "variant", "G", "R",
           "regs", "grid", "bytes", "bits", "us/call", "us(min)", "GB/s", "ps/w", "err/tol", "check", "same");

    std::map<std::pair<std::string, int>, double> best;                 // (class/code, k) -> fastest correct us
    std::map<std::pair<std::string, int>, std::string> best_name;
    for (auto & k : ks) {
        const size_t mat = (size_t) k.K * k.N * 2;
        const int ncopies = (int) std::max<size_t>(1, ((size_t) min_mb << 20) / (mat * k.t.size()));
        std::vector<dev_copy> dc;
        dc.reserve((size_t) ncopies * k.t.size());
        for (int c = 0; c < ncopies; ++c) {
            for (auto & t : k.t) {
                dev_copy d;
                d.plain = upload(t.w);
                d.c12 = to_device(t.e12);
                d.c11 = to_device(t.e11);
                dc.push_back(std::move(d));
            }
        }
        const int nsets = (int) dc.size();
        dev_buf y((size_t) k.K * 8 * sizeof(float)), d((size_t) k.N * 8 * sizeof(float));
        gen_floats<<<64, 256, 0, s>>>(y.as<float>(), (size_t) k.K * 8, seed + k.K + k.N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(s));
        std::vector<float> yh((size_t) k.K * 8);
        CUDA_CHECK(cudaMemcpy(yh.data(), y.p, yh.size() * 4, cudaMemcpyDeviceToHost));

        // reference rows of the first tensor
        const int row_step = k.N > 4096 ? 37 : 1;
        std::vector<int> ref_rows;
        for (int r = 0; r < k.N; r += row_step) ref_rows.push_back(r);
        auto bf = [](uint16_t u) { uint32_t b = (uint32_t) u << 16; float x; memcpy(&x, &b, 4); return (double) x; };

        // the read floor: one buffer of each code's bytes per call, cycled like the weights
        size_t rbytes_max = 0;
        for (const auto & dd : dc) rbytes_max = std::max({ rbytes_max, mat, dd.c12.bytes, dd.c11.bytes });
        const size_t rslice = (rbytes_max + 255) / 256 * 256;
        const int rslices = (int) std::max<size_t>(2, std::min<size_t>(nsets, ((size_t) 96 << 20) / rslice));
        dev_buf rbuf(rslice * rslices), sink(64);

        for (const int T : cols) {
            std::vector<double> refv(ref_rows.size() * T), absv(ref_rows.size() * T);
            for (int t = 0; t < T; ++t) {
                for (size_t i = 0; i < ref_rows.size(); ++i) {
                    double acc = 0.0, as = 0.0;
                    const uint16_t * wr = k.t[0].w.data() + (size_t) ref_rows[i] * k.K;
                    for (int c = 0; c < k.K; ++c) {
                        const double v = bf(wr[c]) * yh[(size_t) t * k.K + c];
                        acc += v;
                        as += fabs(v);
                    }
                    refv[(size_t) t * ref_rows.size() + i] = acc;
                    absv[(size_t) t * ref_rows.size() + i] = as;
                }
            }
            int pick_G = 0, pick_R = 0;
            bf16mv::mmvfp_pick_shape(k.K / 8, k.N, T, nsm, pick_G, pick_R);

            struct variant { std::string code, name; int G, R; };
            std::vector<variant> vs;
            if (contains(codes, "plain")) {
                vs.push_back({ "plain", "pascal", pick_G, pick_R });
                for (int G : { 8, 32, 128 }) for (int R : { 1, 2, 4, 8 }) vs.push_back({ "plain", "g" + std::to_string(G) + "_r" + std::to_string(R), G, R });
            }
            if (contains(codes, "x12")) {
                for (int G : { 8, 32, 128 }) for (int R : { 1, 2, 4, 8 }) {
                    if (k.K / 8 < G) continue;
                    vs.push_back({ "x12", "g" + std::to_string(G) + "_r" + std::to_string(R), G, R });
                }
            }
            if (contains(codes, "x11")) {
                for (int G : { 2, 8, 32, 128 }) for (int R : { 1, 2, 4, 8 }) {
                    if (k.K / 32 < G) continue;
                    vs.push_back({ "x11", "g" + std::to_string(G) + "_r" + std::to_string(R), G, R });
                }
            }
            if (contains(codes, "read")) {
                vs.push_back({ "read", "plain", 0, 0 });
                vs.push_back({ "read", "x12", 0, 0 });
                vs.push_back({ "read", "x11", 0, 0 });
            }

            // the plain kernel's output at each (G, R), for x12's bit-for-bit check
            std::map<std::pair<int, int>, std::vector<float>> plain_out;
            for (const auto & v : vs) {
                const bool is_plain = v.code == "plain", is_read = v.code == "read";
                const int code = v.code == "x12" ? 12 : 11;
                const void * func = is_plain ? bf16mv::pascal_func(v.G, v.R, T) : is_read ? (const void *) lossless_mv::read_only
                                                                                          : lossless_mv::func(code, v.G, v.R, T);
                if (!func) continue;
                const int rows_per_block = is_read ? 1 : (MMVF_PASCAL_NT / v.G) * v.R;
                const dim3 grid(is_read ? nsm * 16 : (k.N + rows_per_block - 1) / rows_per_block), block(is_read ? 256 : MMVF_PASCAL_NT);
                const size_t bytes = is_plain ? mat : v.code == "x12" ? dc[0].c12.bytes : v.code == "x11" ? dc[0].c11.bytes
                                   : v.name == "plain" ? mat : v.name == "x12" ? dc[0].c12.bytes : dc[0].c11.bytes;
                auto launch = [&](cudaStream_t st, int j) {
                    const dev_copy & dd = dc[j % nsets];
                    if (is_read) {
                        const uint4 * x = (const uint4 *) (rbuf.as<char>() + (size_t) (j % rslices) * rslice);
                        size_t n16 = bytes / 16;
                        uint32_t * sk = sink.as<uint32_t>();
                        void * args[] = { &x, &n16, &sk };
                        CUDA_CHECK(cudaLaunchKernel(func, grid, block, args, 0, st));
                    } else if (is_plain) {
                        const uint4 * x = dd.plain.as<uint4>();
                        const float4 * yy = y.as<float4>();
                        const float * bias = nullptr;
                        float * dst = d.as<float>();
                        int nrows = k.N, nchunks = k.K / 8, stride_row = k.K / 8, sy = k.K / 4, sd = k.N;
                        void * args[] = { &x, &yy, &bias, &dst, &nrows, &nchunks, &stride_row, &sy, &sd };
                        CUDA_CHECK(cudaLaunchKernel(func, grid, block, args, 0, st));
                    } else {
                        const dev_code & cc = code == 12 ? dd.c12 : dd.c11;
                        lossless_mv::lparams p;
                        p.lo = cc.lo.as<uint8_t>(); p.idx = cc.idx.as<uint32_t>(); p.rowinfo = cc.rowinfo.as<uint32_t>();
                        p.esc = cc.esc.as<uint32_t>(); p.y = y.as<float4>(); p.dst = d.as<float>();
                        p.nrows = k.N; p.K = k.K; p.stride_col_y4 = k.K / 4; p.stride_col_dst = k.N;
                        void * args[] = { &p };
                        CUDA_CHECK(cudaLaunchKernel(func, grid, block, args, 0, st));
                    }
                };
                double worst = 0.0;
                std::string same = "-";
                bool pass = true;
                if (!is_read) {
                    CUDA_CHECK(cudaMemsetAsync(d.p, 0xff, (size_t) k.N * T * 4, s));
                    launch(s, 0);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaStreamSynchronize(s));
                    std::vector<float> got((size_t) k.N * T);
                    CUDA_CHECK(cudaMemcpy(got.data(), d.p, got.size() * 4, cudaMemcpyDeviceToHost));
                    for (int t = 0; t < T; ++t) {
                        for (size_t i = 0; i < ref_rows.size(); ++i) {
                            const double g = got[(size_t) t * k.N + ref_rows[i]];
                            const double e = std::isfinite(g) ? fabs(g - refv[(size_t) t * ref_rows.size() + i])
                                                                  / (1e-5 * absv[(size_t) t * ref_rows.size() + i] + 1e-30) : 1e30;
                            worst = std::max(worst, e);
                        }
                    }
                    pass = worst <= 1.0;
                    if (is_plain) {
                        plain_out[{ v.G, v.R }] = got;
                    } else if (code == 12 && plain_out.count({ v.G, v.R })) {
                        same = memcmp(got.data(), plain_out[{ v.G, v.R }].data(), got.size() * 4) == 0 ? "same" : "DIFFERS";
                        pass &= same == "same";
                    }
                    ok &= pass;
                }
                int regs = 0;
                cudaFuncAttributes attr;
                if (cudaFuncGetAttributes(&attr, func) == cudaSuccess) regs = attr.numRegs;
                double us_med = 0.0, us_min = 0.0, gbs = 0.0, psw = 0.0;
                if (!check_only) {
                    const auto tm = time_graph(launch, batch, reps, s);
                    us_med = median(tm);
                    us_min = *std::min_element(tm.begin(), tm.end());
                    gbs = bytes / (us_med * 1e3);
                    psw = us_med * 1e6 / ((double) k.K * k.N);
                    if (pass && !is_read) {
                        const auto key = std::make_pair(k.name + "/" + v.code, T);
                        if (!best.count(key) || us_med < best[key]) { best[key] = us_med; best_name[key] = v.name; }
                        if (v.name == "pascal") { best[{ k.name + "/pick", T }] = us_med; best_name[{ k.name + "/pick", T }] = v.name; }
                    }
                }
                const double bits = 8.0 * bytes / ((double) k.K * k.N);
                printf("%-10s %2d %-5s %-8s %3d %2d %5d %5u %9zu %6.3f %9.2f %9.2f %7.1f %7.2f %8.4f %-5s %s\n", k.name.c_str(), T,
                       v.code.c_str(), v.name.c_str(), v.G, v.R, regs, grid.x, bytes, bits, us_med, us_min, gbs, psw, worst,
                       pass ? "PASS" : "FAIL", same.c_str());
                if (f) {
                    fprintf(f, "\"%s\",%d,%s,%d,%d,%d,%d,%s,%s,%d,%d,%d,%u,%zu,%.4f,%.3f,%.3f,%.2f,%.3f,%.4f,%s,%s\n", prop.name,
                            code_arch, k.name.c_str(), k.K, k.N, k.calls, T, v.code.c_str(), v.name.c_str(), v.G, v.R, regs, grid.x,
                            bytes, bits, us_med, us_min, gbs, psw, worst, pass ? "PASS" : "FAIL", same.c_str());
                    fflush(f);
                }
            }
        }
    }
    if (f) fclose(f);

    if (!check_only) {
        // ms per decoding step: calls per token x the fastest correct variant of each code, summed over the classes run
        FILE * sf = summary_path.empty() ? nullptr : fopen(summary_path.c_str(), "w");
        if (sf) fprintf(sf, "tokens,code,ms_per_step,vs_plain_pick_ms,vs_plain_best_ms,classes\n");
        printf("\nms per decoding step over the classes run (calls per token x us per call):\n%-6s %10s %10s %10s %10s %10s %10s %10s\n",
               "tokens", "plain pick", "plain best", "x12", "x12 net", "x11", "x11 net", "x12/plain");
        for (const int T : cols) {
            std::map<std::string, double> ms;
            std::map<std::string, std::string> detail;
            for (const char * c : { "pick", "plain", "x12", "x11" }) {
                double sum = 0.0;
                bool all_in = true;
                for (const auto & k : ks) {
                    const auto key = std::make_pair(k.name + "/" + c, T);
                    if (!best.count(key)) { all_in = false; continue; }
                    sum += best[key] * k.calls / 1000.0;
                    detail[c] += (detail[c].empty() ? "" : " ") + k.name + "=" + best_name[key];
                }
                ms[c] = all_in ? sum : NAN;
            }
            printf("%-6d %10.3f %10.3f %10.3f %+10.3f %10.3f %+10.3f %10.3f\n", T, ms["pick"], ms["plain"], ms["x12"],
                   ms["x12"] - ms["pick"], ms["x11"], ms["x11"] - ms["pick"], ms["x12"] / ms["pick"]);
            if (sf) {
                for (const char * c : { "pick", "plain", "x12", "x11" }) {
                    fprintf(sf, "%d,%s,%.4f,%.4f,%.4f,%s\n", T, c, ms[c], ms[c] - ms["pick"], ms[c] - ms["plain"], detail[c].c_str());
                }
            }
        }
        if (sf) fclose(sf);
    }
    printf("\nall checks: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
