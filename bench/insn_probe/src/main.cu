// insn_probe: throughput of one GPU per instruction class, in operations per clock per SM (Q103).
//
// Mining cards can have per-instruction throttles (NVIDIA's CMP series clamps dp4a and FP32 issue rates), and the
// fork's decode kernels use dp4a far too lightly to reveal one. Each class gets one kernel: every thread runs U
// independent dependency chains of the instruction, R steps unrolled per loop iteration, written in inline PTX so the
// compiler keeps every instruction (build.sh counts them in the sm_61 SASS). Blocks of 256 threads, 8 per SM (full
// occupancy on compute 6.x when a kernel needs at most 32 registers). Each block's first thread reads %smid and
// clock64 at the start and the end; per SM, the operations of its blocks divided by the cycles from its first start
// to its last end give operations per SM clock, independent of the clock the card runs at.
//
// Usage: insn_probe [--device N] [--ms T] [--runs N] [--classes a,b,...] [--csv FILE] [--list]
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int THREADS = 256;
constexpr int BLOCKS_PER_SM = 8;
constexpr int U = 8;  // independent chains per thread
constexpr int R = 8;  // unrolled steps per loop iteration

struct Rec { unsigned smid; long long t0, t1; };

__device__ __forceinline__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }

template <typename T> __device__ __forceinline__ unsigned bits(T v) { unsigned u; memcpy(&u, &v, 4); return u; }
template <> __device__ __forceinline__ unsigned bits<double>(double v) {
    unsigned long long u; memcpy(&u, &v, 8); return (unsigned) (u ^ (u >> 32));
}

// one kernel per class: x[k] = OP(x[k], pa, pb) over U chains, R steps per iteration
#define PROBE(NAME, T, CONS, INIT, ASM)                                                                       \
extern "C" __global__ void __launch_bounds__(THREADS, BLOCKS_PER_SM)                                         \
NAME(int iters, T pa, T pb, unsigned * out, Rec * rec) {                                                      \
    T x[U];                                                                                                    \
    _Pragma("unroll") for (int k = 0; k < U; ++k) x[k] = (T) (INIT);                                           \
    __syncthreads();                                                                                           \
    const long long t0 = clock64();                                                                            \
    for (int i = 0; i < iters; ++i) {                                                                          \
        _Pragma("unroll") for (int r = 0; r < R; ++r) {                                                        \
            _Pragma("unroll") for (int k = 0; k < U; ++k) {                                                    \
                asm volatile(ASM : "+" CONS(x[k]) : CONS(pa), CONS(pb));                                       \
            }                                                                                                  \
        }                                                                                                      \
    }                                                                                                          \
    __syncthreads();                                                                                           \
    const long long t1 = clock64();                                                                            \
    unsigned s = 0;                                                                                            \
    _Pragma("unroll") for (int k = 0; k < U; ++k) s ^= bits(x[k]);                                             \
    out[blockIdx.x * THREADS + threadIdx.x] = s;                                                               \
    if (threadIdx.x == 0) rec[blockIdx.x] = { smid(), t0, t1 };                                                \
}

PROBE(probe_ffma,  float,    "f", 1.0f + threadIdx.x * 1e-6f, "fma.rn.f32 %0, %0, %1, %2;")
PROBE(probe_dfma,  double,   "d", 1.0 + threadIdx.x * 1e-9,   "fma.rn.f64 %0, %0, %1, %2;")
PROBE(probe_hfma2, unsigned, "r", 0x3c003c00u + threadIdx.x,  "fma.rn.f16x2 %0, %0, %1, %2;")
PROBE(probe_imad,  unsigned, "r", threadIdx.x + k,            "mad.lo.u32 %0, %0, %1, %2;")
PROBE(probe_lop3,  unsigned, "r", threadIdx.x + k,            "lop3.b32 %0, %0, %1, %2, 0x96;")
PROBE(probe_shf,   unsigned, "r", threadIdx.x + k,            "shf.l.wrap.b32 %0, %0, %0, %1;")
PROBE(probe_bfe,   unsigned, "r", threadIdx.x * 977u + k,     "bfe.u32 %0, %0, %1, %2;")
PROBE(probe_prmt,  unsigned, "r", threadIdx.x * 977u + k,     "prmt.b32 %0, %0, %1, %2;")
PROBE(probe_dp4a,  unsigned, "r", threadIdx.x + k,            "dp4a.s32.s32 %0, %1, %2, %0;")
PROBE(probe_popc,  unsigned, "r", threadIdx.x * 977u + k,     "popc.b32 %0, %0;")
PROBE(probe_i2f,   unsigned, "r", threadIdx.x + k,            "{ .reg .f32 t; cvt.rn.f32.s32 t, %0; mov.b32 %0, t; }")
PROBE(probe_f2f,   float,    "f", 1.0f + threadIdx.x * 1e-3f, "{ .reg .f16 h; cvt.rn.f16.f32 h, %0; cvt.f32.f16 %0, h; }")
PROBE(probe_shfl,  unsigned, "r", threadIdx.x + k,            "shfl.sync.bfly.b32 %0, %0, %1, 0x1f, 0xffffffff;")

// integer add: x[k] += x[k+1], since ptxas folds a chain of adds of one value into a single add
extern "C" __global__ void __launch_bounds__(THREADS, BLOCKS_PER_SM)
probe_iadd(int iters, unsigned pa, unsigned pb, unsigned * out, Rec * rec) {
    unsigned x[U];
    _Pragma("unroll") for (int k = 0; k < U; ++k) x[k] = threadIdx.x * pa + k * pb;
    __syncthreads();
    const long long t0 = clock64();
    for (int i = 0; i < iters; ++i) {
        _Pragma("unroll") for (int r = 0; r < R; ++r) {
            _Pragma("unroll") for (int k = 0; k < U; ++k) {
                asm volatile("add.u32 %0, %0, %1;" : "+r"(x[k]) : "r"(x[(k + 1) % U]));
            }
        }
    }
    __syncthreads();
    const long long t1 = clock64();
    unsigned acc = 0;
    _Pragma("unroll") for (int k = 0; k < U; ++k) acc ^= x[k];
    out[blockIdx.x * THREADS + threadIdx.x] = acc;
    if (threadIdx.x == 0) rec[blockIdx.x] = { smid(), t0, t1 };
}

// shared-memory loads: U conflict-free 32-bit loads per step (thread t, chain k reads word k*256 + t)
extern "C" __global__ void __launch_bounds__(THREADS, BLOCKS_PER_SM)
probe_lds(int iters, unsigned pa, unsigned pb, unsigned * out, Rec * rec) {
    __shared__ unsigned s[U * THREADS];
    for (int k = 0; k < U; ++k) s[k * THREADS + threadIdx.x] = threadIdx.x * pa + k * pb;
    unsigned addr[U], x[U];
    _Pragma("unroll") for (int k = 0; k < U; ++k) {
        addr[k] = (unsigned) __cvta_generic_to_shared(&s[k * THREADS + threadIdx.x]);
        x[k] = 0;
    }
    __syncthreads();
    const long long t0 = clock64();
    for (int i = 0; i < iters; ++i) {
        _Pragma("unroll") for (int r = 0; r < R; ++r) {
            _Pragma("unroll") for (int k = 0; k < U; ++k) {
                asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(x[k]) : "r"(addr[k]));
            }
        }
    }
    __syncthreads();
    const long long t1 = clock64();
    unsigned acc = 0;
    _Pragma("unroll") for (int k = 0; k < U; ++k) acc ^= x[k];
    out[blockIdx.x * THREADS + threadIdx.x] = acc;
    if (threadIdx.x == 0) rec[blockIdx.x] = { smid(), t0, t1 };
}

typedef void (*kfn_u)(int, unsigned, unsigned, unsigned *, Rec *);
typedef void (*kfn_f)(int, float, float, unsigned *, Rec *);
typedef void (*kfn_d)(int, double, double, unsigned *, Rec *);

struct Class {
    const char * name;
    const char * sass;     // the instruction build.sh counts in the kernel's SASS
    int          mult;     // operations per inline-asm step (F2F: f32->f16->f32 = 2 conversions)
    double       guess;    // rough rate in operations per clock per SM, only to size each run
    const char * note;
    kfn_u fu; kfn_f ff; kfn_d fd;
    unsigned ua, ub; float fa, fb; double da, db;
};

static std::vector<Class> classes() {
    std::vector<Class> c;
    auto U_ = [&](const char * n, const char * s, int p, double g, const char * note, kfn_u f, unsigned a, unsigned b) {
        c.push_back({n, s, p, g, note, f, nullptr, nullptr, a, b, 0, 0, 0, 0});
    };
    c.push_back({"ffma", "FFMA", 1, 128, "32-bit FP fused multiply-add: the reference rate",
                 nullptr, probe_ffma, nullptr, 0, 0, 1.0000001f, 1e-7f, 0, 0});
    c.push_back({"dfma", "DFMA", 1, 4, "64-bit FP fused multiply-add",
                 nullptr, nullptr, probe_dfma, 0, 0, 0, 0, 1.0000000001, 1e-12});
    U_("hfma2", "HFMA2", 1, 1, "16-bit FP fused multiply-add on half2 (two results per instruction)", probe_hfma2,
       0x3c013c01u, 0x00010001u);
    U_("iadd", "IADD", 1, 128, "32-bit integer add", probe_iadd, 0x9e3779b9u, 0);
    U_("imad", "XMAD", 1, 40, "32-bit integer multiply-add (several XMADs on compute 6.x)", probe_imad, 0x9e3779b9u, 7);
    U_("lop3", "LOP3", 1, 128, "three-input logic", probe_lop3, 0x5bd1e995u, 0x1b873593u);
    U_("shf", "SHF", 1, 64, "funnel shift (rotate)", probe_shf, 5, 0);
    U_("bfe", "BFE", 1, 64, "bit-field extract", probe_bfe, 3, 20);
    U_("prmt", "PRMT", 1, 64, "byte permute", probe_prmt, 0x9e3779b9u, 0x0321);
    U_("dp4a", "IDP", 1, 128, "4 x int8 dot product with 32-bit accumulate (the MMQ and MMVQ workhorse)", probe_dp4a,
       0x01ff7f80u, 0x7f01ff02u);
    U_("popc", "POPC", 1, 32, "population count", probe_popc, 0, 0);
    U_("i2f", "I2F", 1, 32, "int32 to FP32 conversion", probe_i2f, 0, 0);
    c.push_back({"f2f", "F2F", 2, 16, "FP32 to FP16 and back (two conversions per operation)",
                 nullptr, probe_f2f, nullptr, 0, 0, 0, 0, 0, 0});
    U_("shfl", "SHFL", 1, 32, "warp shuffle (butterfly)", probe_shfl, 1, 0);
    U_("lds", "LDS", 1, 32, "32-bit shared-memory load, conflict-free (a warp's load is one transaction)", probe_lds,
       0x9e3779b9u, 7);
    return c;
}

struct Result { double per_sm_clk_med, per_sm_clk_min, per_sm_clk_max, ms, ghz; };

static double clock_hz(int dev) {
    int khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev));
    return khz * 1e3;
}

static Result run(const Class & c, int iters, int nsm, unsigned * d_out, Rec * d_rec) {
    const int blocks = nsm * BLOCKS_PER_SM;
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    if (c.fu) c.fu<<<blocks, THREADS>>>(iters, c.ua, c.ub, d_out, d_rec);
    else if (c.ff) c.ff<<<blocks, THREADS>>>(iters, c.fa, c.fb, d_out, d_rec);
    else c.fd<<<blocks, THREADS>>>(iters, c.da, c.db, d_out, d_rec);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));

    std::vector<Rec> rec(blocks);
    CUDA_CHECK(cudaMemcpy(rec.data(), d_rec, blocks * sizeof(Rec), cudaMemcpyDeviceToHost));
    // per SM: blocks it ran, first start, last end
    std::vector<long long> first(256, -1), last(256, -1);
    std::vector<int> nblk(256, 0);
    for (const auto & r : rec) {
        const unsigned s = r.smid & 255;
        if (first[s] < 0 || r.t0 < first[s]) first[s] = r.t0;
        if (last[s] < 0 || r.t1 > last[s]) last[s] = r.t1;
        ++nblk[s];
    }
    const double ops_per_block = (double) THREADS * iters * R * U * c.mult;
    std::vector<double> rate;
    double span_sum = 0;
    for (int s = 0; s < 256; ++s) {
        if (!nblk[s]) continue;
        const double span = (double) (last[s] - first[s]);
        rate.push_back(nblk[s] * ops_per_block / span);
        span_sum += span;
    }
    std::sort(rate.begin(), rate.end());
    Result res;
    res.per_sm_clk_med = rate[rate.size() / 2];
    res.per_sm_clk_min = rate.front();
    res.per_sm_clk_max = rate.back();
    res.ms = ms;
    res.ghz = span_sum / rate.size() / (ms * 1e6);
    return res;
}

int main(int argc, char ** argv) {
    int dev = 0, runs = 3;
    double target_ms = 100;
    std::string only, csv;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) dev = atoi(argv[++i]);
        else if (a == "--ms" && i + 1 < argc) target_ms = atof(argv[++i]);
        else if (a == "--runs" && i + 1 < argc) runs = atoi(argv[++i]);
        else if (a == "--classes" && i + 1 < argc) only = std::string(",") + argv[++i] + ",";
        else if (a == "--csv" && i + 1 < argc) csv = argv[++i];
        else if (a == "--list") {
            for (const auto & c : classes()) printf("%-6s %-6s %s\n", c.name, c.sass, c.note);
            return 0;
        } else {
            fprintf(stderr, "usage: %s [--device N] [--ms T] [--runs N] [--classes a,b,...] [--csv FILE] [--list]\n", argv[0]);
            return 1;
        }
    }
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    const int nsm = p.multiProcessorCount;
    const double clk_hz = clock_hz(dev);
    printf("device %d: %s, compute %d.%d, %d SMs, max clock %.0f MHz\n", dev, p.name, p.major, p.minor, nsm, clk_hz / 1e6);
    printf("%d blocks of %d threads (%d per SM), %d chains x %d steps per loop iteration, best of %d runs of ~%.0f ms\n\n",
           nsm * BLOCKS_PER_SM, THREADS, BLOCKS_PER_SM, U, R, runs, target_ms);

    unsigned * d_out; Rec * d_rec;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t) nsm * BLOCKS_PER_SM * THREADS * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&d_rec, (size_t) nsm * BLOCKS_PER_SM * sizeof(Rec)));

    const auto all = classes();
    // ~2 s of FFMA first, so the card reaches its sustained clock before anything is recorded
    {
        const int iters = (int) std::max(1.0, 0.2 * 128 * clk_hz / ((double) THREADS * BLOCKS_PER_SM * R * U));
        for (int i = 0; i < 10; ++i) run(all[0], iters, nsm, d_out, d_rec);
    }

    FILE * fc = csv.empty() ? nullptr : fopen(csv.c_str(), "w");
    if (fc) fprintf(fc, "class,sass,ops_per_clk_per_sm,min_sm,max_sm,warp_instr_per_clk_per_sm,vs_ffma,occupancy_blocks,ms,ghz\n");
    printf("%-6s %-6s %10s %10s %8s %6s %5s %8s  %s\n", "class", "sass", "ops/clk/SM", "warp-ins", "vs FFMA", "occ", "ms", "GHz", "what");
    double ffma = 0;
    for (const auto & c : all) {
        if (!only.empty() && only.find(std::string(",") + c.name + ",") == std::string::npos && strcmp(c.name, "ffma")) continue;
        int occ = 0;
        const void * fn = c.fu ? (const void *) c.fu : c.ff ? (const void *) c.ff : (const void *) c.fd;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, THREADS, 0));
        int iters = (int) std::max(1.0, target_ms * 1e-3 * c.guess * clk_hz / ((double) THREADS * BLOCKS_PER_SM * R * U * c.mult));
        Result best{};
        for (int r = 0; r < runs; ++r) {
            Result x = run(c, iters, nsm, d_out, d_rec);
            if (r == 0 && x.ms < 0.5 * target_ms) {  // the guess was low: size the remaining runs from this one
                iters = (int) std::max(1.0, iters * target_ms / std::max(x.ms, 0.01));
                x = run(c, iters, nsm, d_out, d_rec);
            }
            if (x.per_sm_clk_med > best.per_sm_clk_med) best = x;
        }
        if (!strcmp(c.name, "ffma")) ffma = best.per_sm_clk_med;
        const double vs = ffma > 0 ? best.per_sm_clk_med / ffma : 0;
        printf("%-6s %-6s %10.2f %10.3f %8.3f %6d %5.0f %8.3f  %s\n", c.name, c.sass, best.per_sm_clk_med,
               best.per_sm_clk_med / 32, vs, occ, best.ms, best.ghz, c.note);
        if (fc) fprintf(fc, "%s,%s,%.3f,%.3f,%.3f,%.4f,%.4f,%d,%.2f,%.4f\n", c.name, c.sass, best.per_sm_clk_med,
                        best.per_sm_clk_min, best.per_sm_clk_max, best.per_sm_clk_med / 32, vs, occ, best.ms, best.ghz);
    }
    if (fc) { fclose(fc); printf("\nCSV written to %s\n", csv.c_str()); }
    printf("\nops/clk/SM: operations per SM clock cycle (lane operations; a warp instruction is 32), median over SMs.\n");
    CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_rec));
    return 0;
}
