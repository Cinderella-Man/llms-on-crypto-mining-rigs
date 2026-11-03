// megakernel_p106: does one persistent kernel per card beat a CUDA graph of small dependent kernels on a 5-SM P106?
//
// N dependent stages, each a small piece of a decode step (an elementwise op, a norm, a short matrix-vector product,
// or nothing), run
//   - as N kernel launches: eager on a stream, or as one CUDA graph, each with or without a CUDA event between the
//     stages (as the llama.cpp fork's GGML_CUDA_PROFILE records them, which calibrates that profiler), and
//   - as one persistent kernel running the same N stages behind grid barriers: a cooperative launch with
//     grid.sync(); a plain launch with an atomic-counter barrier, directly or as one graph node; and the same with the
//     barrier skipped where a stage reads only what the same thread wrote in the stage before.
// Every variant runs the same device function per stage on the same blocks of work, so only the boundaries between
// stages differ, and every variant's output is checked against the eager launches and a CPU reference.
//
// --hop A,B times one card-to-card hop of a residual vector the ways a layer split can make it without peer access:
// through a pinned host buffer with two copies, as one driver-staged peer copy, and as a mailbox in mapped host memory
// that a kernel on card B polls. See ../README.md.
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_CHECK(x)                                                                                        \
    do {                                                                                                     \
        cudaError_t err_ = (x);                                                                              \
        if (err_ != cudaSuccess) {                                                                           \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(err_));      \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)

namespace {

constexpr int THREADS = 256;    // threads per block, every kernel
constexpr int NMAX    = 12288;  // elements of each of the two activation buffers

enum kind : int { K_EMPTY = 0, K_EW = 1, K_NORM = 2, K_MV = 3 };

struct stage {
    int kind;
    int n;              // elements read; EW and NORM also write n, MV writes `rows`
    int rows;           // MV output rows
    int local;          // EW after an EW of the same n: each thread reads only what it wrote, so no barrier before it
    float a, b;         // EW: y = a x + b; NORM: y = a x / rms(x); MV: y = a (W x) + b
    const __half2 * w;  // MV weights, rows x n/2 half2, row-major
};

// virtual blocks of a stage, as the fork launches it: an elementwise op one block per 256 elements, a norm one block
// for the whole vector (as rms_norm does for one row), a matrix-vector product one block per output row
__host__ __device__ inline int vblocks(const stage & st) {
    switch (st.kind) {
        case K_EW: return (st.n + THREADS - 1) / THREADS;
        case K_MV: return st.rows;
        default:   return 1;
    }
}

// sum over the block; every thread returns the same value
__device__ __forceinline__ float block_sum(float v, float * sh) {
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) {
        sh[warp] = v;
    }
    __syncthreads();
    v = lane < THREADS / 32 ? sh[lane] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    __syncthreads();  // sh is reused by the next call
    return v;
}

// One virtual block of one stage. The per-stage kernels run virtual block blockIdx.x; the persistent kernels loop
// their blocks over the stage's virtual blocks. Activations are read with ld.global.cg (L2, not L1), because in the
// persistent kernels another block may have written them since this SM last read the buffer.
__device__ __forceinline__ void run_vblock(const stage & st, const int vb, const float * x, float * y, float * sh) {
    const int t = threadIdx.x;
    if (st.kind == K_EW) {
        const int i = vb * THREADS + t;
        if (i < st.n) {
            y[i] = st.a * __ldcg(x + i) + st.b;
        }
    } else if (st.kind == K_NORM) {
        float s = 0.0f;
        for (int i = t; i < st.n; i += THREADS) {
            const float v = __ldcg(x + i);
            s += v * v;
        }
        s = block_sum(s, sh);
        const float r = st.a * rsqrtf(s / st.n + 1e-6f);
        for (int i = t; i < st.n; i += THREADS) {
            y[i] = r * __ldcg(x + i);
        }
    } else if (st.kind == K_MV) {
        const __half2 * w  = st.w + (size_t) vb * (st.n / 2);
        const float2 *  x2 = reinterpret_cast<const float2 *>(x);
        float s = 0.0f;
        for (int k = t; k < st.n / 2; k += THREADS) {
            const float2 wf = __half22float2(__ldg(w + k));
            const float2 xv = __ldcg(x2 + k);
            s += wf.x * xv.x + wf.y * xv.y;
        }
        s = block_sum(s, sh);
        if (t == 0) {
            y[vb] = st.a * s + st.b;
        }
    }
}

__global__ void __launch_bounds__(THREADS) k_stage(const stage st, const float * x, float * y) {
    __shared__ float sh[THREADS / 32];
    run_vblock(st, blockIdx.x, x, y, sh);
}

// A grid-wide barrier from one monotonic counter: every block takes a ticket, and the barrier is passed when the
// counter reaches the end of the ticket's round. The counter is never reset, so the kernel can be replayed from a CUDA
// graph, as long as every launch that shares the counter has the same grid and passes only full barriers. Needs every
// block resident at once (the grid is capped at the occupancy times the SMs).
__device__ __forceinline__ void barrier_atomic(unsigned long long * count) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned long long ticket = atomicAdd(count, 1ull);
        const unsigned long long target = (ticket / gridDim.x + 1) * gridDim.x;
        while (*(volatile unsigned long long *) count < target) {
        }
        __threadfence();
    }
    __syncthreads();
}

// MODE 0: the atomic barrier before every stage; 1: only before stages that are not local; 2: cooperative grid.sync()
template <int MODE>
__global__ void __launch_bounds__(THREADS) k_persistent(const stage * stages, const int n_stages, float * buf0, float * buf1,
                                                        unsigned long long * count) {
    __shared__ float sh[THREADS / 32];
    stage st = stages[0];
    for (int s = 0; s < n_stages; ++s) {
        const stage next = stages[s + 1 < n_stages ? s + 1 : s];  // fetched while this stage runs
        const float * x = (s & 1) ? buf1 : buf0;
        float * y = (s & 1) ? buf0 : buf1;
        const int nvb = vblocks(st);
        for (int vb = blockIdx.x; vb < nvb; vb += gridDim.x) {
            run_vblock(st, vb, x, y, sh);
        }
        if (s + 1 < n_stages && !(MODE == 1 && next.local)) {
            if (MODE == 2) {
                cg::this_grid().sync();
            } else {
                barrier_atomic(count);
            }
        }
        st = next;
    }
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

// ---- the hop between two cards --------------------------------------------------------------------------------------

__global__ void k_mail_send(const float4 * src, float4 * mail, const int n4, volatile unsigned * flag, const unsigned seq) {
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        mail[i] = src[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *flag = seq;
    }
}

// polls the flag in mapped host memory (gives up after 2^22 polls, a few seconds over PCIe, and counts a timeout),
// then copies the mail in
__global__ void k_mail_recv(const float4 * mail, float4 * dst, const int n4, const volatile unsigned * flag, const unsigned seq,
                            unsigned * timeouts) {
    __shared__ int ok;
    if (threadIdx.x == 0) {
        unsigned long long polls = 0;
        while (*flag != seq && ++polls < (1ull << 22)) {
        }
        ok = *flag == seq;
        if (!ok) {
            atomicAdd(timeouts, 1u);
        }
    }
    __syncthreads();
    if (!ok) {
        return;
    }
    __threadfence_system();
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        dst[i] = __ldcv(mail + i);
    }
}

__global__ void k_touch(float * p) {
    if (threadIdx.x == 0) {
        p[0] += 1.0f;
    }
}

// ---- host side ------------------------------------------------------------------------------------------------------

struct options {
    std::string mixes = "empty,ew2560,ew10240,norm2560,mv10240x4,mv2560x48,census1,census6";
    int  stages = 0;                 // override the stage count of the simple mixes (0: the mix's default)
    std::string grids = "5,10,20,max";
    int  reps = 0;                   // chains per timed measurement (0: enough for ~20 ms)
    int  trials = 7;
    bool check_only = false;
    bool no_check = false;
    bool smoke = false;
    std::string hop;                 // "A,B"
    std::string hop_sizes = "10240,40960,163840";
    int  hop_reps = 300;
    bool hop_only = false;
    const char * csv = nullptr;
};

// One layer of Qwen3.8-Flash-Next's small launch groups on a card, from the R13 trace's census
// (docs/data/2026-11-04_megakernel_census): per layer ~34.5 launch groups without weights, of which ~6.5 reductions,
// and ~4.5 short products (the hyper-connection injects, 10,240 x 4; the recurrent alpha/beta, 2,560 x 48; the
// shared-expert gate, 2,560 x 1), in two halves (attention or recurrent mixer, then the experts).
const char * CENSUS_HALF = "norm10240,norm2560,ew2560*5,mv2560x48,ew2560*4,norm2560,ew2560*3,mv10240x4,ew10240*2";

struct mix_def {
    std::string name;
    std::string pattern;
    int repeat;  // the pattern repeated this many times
};

void usage() {
    printf("usage: megakernel_p106 [options]\n"
           "  --mixes LIST      stage mixes to run (default %s)\n"
           "                    a mix is empty, ew<n>, norm<n>, mv<K>x<rows> (N stages of one kind; N from --stages, default\n"
           "                    256 or 64 for mv), censusL (Flash-Next's small launch groups for L layers, 38 per layer), or a\n"
           "                    pattern in brackets, e.g. [norm2560,ew2560*3,mv2560x48]*4\n"
           "  --stages N        stages of the one-kind mixes\n"
           "  --grids LIST      grids of the persistent kernels, numbers or max (resident blocks per SM x SMs); default 5,10,20,max\n"
           "  --reps N          chains per timed measurement (default: enough for ~20 ms)\n"
           "  --trials N        timed measurements per variant, median reported (default 7)\n"
           "  --check-only      correctness checks only\n"
           "  --no-check        skip the checks\n"
           "  --smoke           every path once with 2 chains and 1 trial (a functional check of the timing code)\n"
           "  --hop A,B         also time a hop from card A to card B (CUDA device numbers)\n"
           "  --hop-sizes LIST  bytes per hop (default 10240,40960,163840)\n"
           "  --hop-reps N      hops per size and way (default 300)\n"
           "  --hop-only        only the hop\n"
           "  --csv FILE        results as CSV (section,mix,stages,variant,grid,key,value,unit)\n",
           options().mixes.c_str());
}

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

FILE * g_csv = nullptr;

void csv_row(const char * section, const std::string & mix, int n, const std::string & variant, int grid,
             const std::string & key, double value, const char * unit) {
    if (g_csv) {
        fprintf(g_csv, "%s,%s,%d,%s,%d,%s,%.4f,%s\n", section, mix.c_str(), n, variant.c_str(), grid, key.c_str(), value,
                unit);
        fflush(g_csv);
    }
}

std::string stage_key(const stage & st) {
    switch (st.kind) {
        case K_EW:   return "ew" + std::to_string(st.n);
        case K_NORM: return "norm" + std::to_string(st.n);
        case K_MV:   return "mv" + std::to_string(st.n) + "x" + std::to_string(st.rows);
        default:     return "empty";
    }
}

// "norm2560,ew2560*3" -> stages (weights not yet set); returns false on a malformed token
bool parse_pattern(const std::string & pat, std::vector<stage> & out) {
    size_t pos = 0;
    while (pos < pat.size()) {
        size_t end = pat.find(',', pos);
        if (end == std::string::npos) {
            end = pat.size();
        }
        std::string tok = pat.substr(pos, end - pos);
        pos = end + 1;
        if (tok.empty()) {
            continue;
        }
        int count = 1;
        const size_t star = tok.find('*');
        if (star != std::string::npos) {
            count = atoi(tok.c_str() + star + 1);
            tok = tok.substr(0, star);
        }
        stage st{};
        if (tok == "empty") {
            st.kind = K_EMPTY;
        } else if (tok.rfind("ew", 0) == 0) {
            st.kind = K_EW;
            st.n = atoi(tok.c_str() + 2);
        } else if (tok.rfind("norm", 0) == 0) {
            st.kind = K_NORM;
            st.n = atoi(tok.c_str() + 4);
        } else if (tok.rfind("mv", 0) == 0) {
            st.kind = K_MV;
            if (sscanf(tok.c_str() + 2, "%dx%d", &st.n, &st.rows) != 2) {
                return false;
            }
        } else {
            return false;
        }
        if (count < 1 || st.n < 0 || st.n > NMAX || (st.kind == K_MV && (st.n % 2 || st.rows < 1 || st.rows > NMAX)) ||
            (st.kind != K_EMPTY && st.n < 1)) {
            return false;
        }
        for (int i = 0; i < count; ++i) {
            out.push_back(st);
        }
    }
    return !out.empty();
}

bool make_mix(const std::string & name, const options & opt, mix_def & m) {
    m.name = name;
    if (name.rfind("census", 0) == 0) {
        const int layers = std::max(1, atoi(name.c_str() + 6));
        m.pattern = std::string(CENSUS_HALF) + "," + CENSUS_HALF;
        m.repeat = layers;
    } else if (!name.empty() && name[0] == '[') {
        const size_t close = name.find(']');
        if (close == std::string::npos) {
            return false;
        }
        m.pattern = name.substr(1, close - 1);
        m.repeat = close + 1 < name.size() && name[close + 1] == '*' ? std::max(1, atoi(name.c_str() + close + 2)) : 1;
    } else {
        m.pattern = name;
        m.repeat = opt.stages > 0 ? opt.stages : (name.rfind("mv", 0) == 0 ? 64 : 256);
    }
    return true;
}

struct dev_mem {
    void * p = nullptr;
    explicit dev_mem(size_t bytes) { CUDA_CHECK(cudaMalloc(&p, bytes)); }
    ~dev_mem() { cudaFree(p); }
    template <typename T> T * as() const { return (T *) p; }
};

// a mix ready to run: the stages with their own weights (distinct for every MV stage, so no stage finds another's
// weights in L2), the stage table on the card, and the host copies for the reference
struct chain {
    std::string name;
    std::vector<stage> st;
    std::vector<std::vector<__half>> w_host;  // per stage (empty unless MV)
    std::vector<dev_mem *> w_dev;
    dev_mem * table = nullptr;
    size_t weight_bytes = 0;
    ~chain() {
        for (auto * d : w_dev) {
            delete d;
        }
        delete table;
    }
};

void build_chain(const mix_def & m, chain & c) {
    std::vector<stage> one;
    if (!parse_pattern(m.pattern, one)) {
        fprintf(stderr, "bad mix '%s'\n", m.name.c_str());
        exit(1);
    }
    c.name = m.name;
    for (int r = 0; r < m.repeat; ++r) {
        c.st.insert(c.st.end(), one.begin(), one.end());
    }
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    c.w_host.resize(c.st.size());
    for (size_t s = 0; s < c.st.size(); ++s) {
        stage & st = c.st[s];
        const float r1 = 0.5f * (u(rng) + 1.0f);
        switch (st.kind) {
            case K_EW:   st.a = 0.6f + 0.35f * r1; st.b = 0.05f * u(rng); break;
            case K_NORM: st.a = 0.8f + 0.4f * r1;  st.b = 0.0f; break;
            case K_MV:   st.a = 1.0f;              st.b = 0.01f * u(rng); break;
            default:     break;
        }
        st.local = s > 0 && st.kind == K_EW && c.st[s - 1].kind == K_EW && c.st[s - 1].n == st.n;
        if (st.kind == K_MV) {
            const size_t nw = (size_t) st.rows * st.n;
            std::vector<__half> & h = c.w_host[s];
            h.resize(nw);
            const float scale = 1.0f / sqrtf((float) st.n);
            for (size_t i = 0; i < nw; ++i) {
                h[i] = __float2half(scale * u(rng));
            }
            dev_mem * d = new dev_mem(nw * sizeof(__half));
            CUDA_CHECK(cudaMemcpy(d->p, h.data(), nw * sizeof(__half), cudaMemcpyHostToDevice));
            c.w_dev.push_back(d);
            st.w = d->as<__half2>();
            c.weight_bytes += nw * sizeof(__half);
        }
    }
    c.table = new dev_mem(c.st.size() * sizeof(stage));
    CUDA_CHECK(cudaMemcpy(c.table->p, c.st.data(), c.st.size() * sizeof(stage), cudaMemcpyHostToDevice));
}

// the chain on the CPU in double precision, from (b0, b1); returns the buffer the last stage wrote
std::vector<double> reference(const chain & c, const std::vector<float> & b0, const std::vector<float> & b1) {
    std::vector<double> buf[2] = { std::vector<double>(b0.begin(), b0.end()), std::vector<double>(b1.begin(), b1.end()) };
    for (size_t s = 0; s < c.st.size(); ++s) {
        const stage & st = c.st[s];
        const std::vector<double> & x = buf[s & 1];
        std::vector<double> & y = buf[(s & 1) ^ 1];
        if (st.kind == K_EW) {
            for (int i = 0; i < st.n; ++i) {
                y[i] = (double) st.a * x[i] + st.b;
            }
        } else if (st.kind == K_NORM) {
            double ss = 0.0;
            for (int i = 0; i < st.n; ++i) {
                ss += x[i] * x[i];
            }
            const double r = st.a / sqrt(ss / st.n + 1e-6);
            for (int i = 0; i < st.n; ++i) {
                y[i] = r * x[i];
            }
        } else if (st.kind == K_MV) {
            const std::vector<__half> & w = c.w_host[s];
            for (int row = 0; row < st.rows; ++row) {
                double acc = 0.0;
                for (int k = 0; k < st.n; ++k) {
                    acc += (double) __half2float(w[(size_t) row * st.n + k]) * x[k];
                }
                y[row] = st.a * acc + st.b;
            }
        }
    }
    return buf[c.st.size() & 1];
}

struct device_ctx {
    int dev = 0;
    cudaDeviceProp prop{};
    cudaStream_t s = nullptr;
    dev_mem * buf0 = nullptr;
    dev_mem * buf1 = nullptr;
    dev_mem * count = nullptr;
    std::vector<float> init0, init1;
    int occ[3] = { 0, 0, 0 };  // resident blocks per SM of k_persistent<0>, <1>, <2>
    bool coop = false;
    bool coop_graph = false;   // a cooperative launch can be captured into a CUDA graph

    int max_grid(int v) const;
};

void reset_buffers(device_ctx & d) {
    CUDA_CHECK(cudaMemcpyAsync(d.buf0->p, d.init0.data(), NMAX * sizeof(float), cudaMemcpyHostToDevice, d.s));
    CUDA_CHECK(cudaMemcpyAsync(d.buf1->p, d.init1.data(), NMAX * sizeof(float), cudaMemcpyHostToDevice, d.s));
    CUDA_CHECK(cudaStreamSynchronize(d.s));
}

// ---- the variants ---------------------------------------------------------------------------------------------------

enum variant : int { V_STREAM, V_STREAM_EV, V_GRAPH, V_GRAPH_EV, V_COOP, V_COOP_GRAPH, V_BAR, V_BAR_GRAPH, V_LOCAL, V_COUNT };
const char * VARIANT_NAMES[V_COUNT] = { "stream", "stream+events", "graph", "graph+events", "coop",
                                        "coop-graph", "barrier", "barrier-graph", "barrier-local" };

bool per_stage(int v) {
    return v <= V_GRAPH_EV;
}

// the largest grid variant v may use: every block resident at once
int device_ctx::max_grid(int v) const {
    const int o = occ[v == V_COOP || v == V_COOP_GRAPH ? 2 : (v == V_LOCAL ? 1 : 0)];
    return o * prop.multiProcessorCount;
}

bool runnable(const device_ctx & d, int v, int grid) {
    if ((v == V_COOP && !d.coop) || (v == V_COOP_GRAPH && !d.coop_graph)) {
        return false;
    }
    return per_stage(v) || (grid >= 1 && grid <= d.max_grid(v));
}

// enqueue one chain as N launches; with `ev`, an event before the first stage and after every stage
void enqueue_stages(const chain & c, device_ctx & d, cudaStream_t s, const std::vector<cudaEvent_t> * ev, bool capturing) {
    auto record = [&](size_t k) {
        if (capturing) {
            CUDA_CHECK(cudaEventRecordWithFlags((*ev)[k], s, cudaEventRecordExternal));
        } else {
            CUDA_CHECK(cudaEventRecord((*ev)[k], s));
        }
    };
    if (ev) {
        record(0);
    }
    for (size_t i = 0; i < c.st.size(); ++i) {
        const float * x = (i & 1) ? d.buf1->as<float>() : d.buf0->as<float>();
        float * y = (i & 1) ? d.buf0->as<float>() : d.buf1->as<float>();
        k_stage<<<vblocks(c.st[i]), THREADS, 0, s>>>(c.st[i], x, y);
        if (ev) {
            record(i + 1);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void enqueue_persistent(const chain & c, device_ctx & d, cudaStream_t s, int v, int grid) {
    const stage * table = c.table->as<stage>();
    const int n = (int) c.st.size();
    float * b0 = d.buf0->as<float>();
    float * b1 = d.buf1->as<float>();
    unsigned long long * cnt = d.count->as<unsigned long long>();
    if (v == V_COOP || v == V_COOP_GRAPH) {
        void * args[] = { (void *) &table, (void *) &n, (void *) &b0, (void *) &b1, (void *) &cnt };
        CUDA_CHECK(cudaLaunchCooperativeKernel((const void *) k_persistent<2>, dim3(grid), dim3(THREADS), args, 0, s));
    } else if (v == V_LOCAL) {
        k_persistent<1><<<grid, THREADS, 0, s>>>(table, n, b0, b1, cnt);
    } else {
        k_persistent<0><<<grid, THREADS, 0, s>>>(table, n, b0, b1, cnt);
    }
    CUDA_CHECK(cudaGetLastError());
}

struct graph_handle {
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ge = nullptr;
    ~graph_handle() {
        if (ge) {
            cudaGraphExecDestroy(ge);
        }
        if (g) {
            cudaGraphDestroy(g);
        }
    }
};

void capture(graph_handle & h, cudaStream_t s, const std::function<void()> & body) {
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    body();
    CUDA_CHECK(cudaStreamEndCapture(s, &h.g));
    CUDA_CHECK(cudaGraphInstantiate(&h.ge, h.g, 0));
}

// median ms of one `enqueue` (one chain), over `trials` measurements of `reps` enqueues each
double time_ms(const std::function<void()> & enqueue, cudaStream_t s, int reps, int trials) {
    for (int i = 0; i < 3; ++i) {
        enqueue();
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    std::vector<double> v;
    for (int t = 0; t < trials; ++t) {
        CUDA_CHECK(cudaEventRecord(e0, s));
        for (int r = 0; r < reps; ++r) {
            enqueue();
        }
        CUDA_CHECK(cudaEventRecord(e1, s));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        v.push_back(ms / reps);
    }
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return median(v);
}

// what GGML_CUDA_PROFILE would record: the time between the events around each stage, collected per stage kind over
// `runs` single chains (each read after a synchronize, as the profiler does)
void event_intervals(const chain & c, device_ctx & d, const std::vector<cudaEvent_t> & ev, const std::function<void()> & run,
                     int runs, std::map<std::string, std::vector<double>> & out) {
    for (int r = 0; r < runs; ++r) {
        run();
        CUDA_CHECK(cudaStreamSynchronize(d.s));
        for (size_t i = 0; i < c.st.size(); ++i) {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev[i], ev[i + 1]));
            out[stage_key(c.st[i])].push_back(1000.0 * ms);
            out["all"].push_back(1000.0 * ms);
        }
    }
}

std::vector<int> parse_ints(const std::string & s, int max_value) {
    std::vector<int> v;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t end = s.find(',', pos);
        if (end == std::string::npos) {
            end = s.size();
        }
        const std::string tok = s.substr(pos, end - pos);
        if (tok == "max") {
            v.push_back(max_value);
        } else if (!tok.empty()) {
            v.push_back(atoi(tok.c_str()));
        }
        pos = end + 1;
    }
    return v;
}

// run the chain once from the initial buffers with variant v and return the last stage's output buffer
std::vector<float> run_once(const chain & c, device_ctx & d, int v, int grid) {
    reset_buffers(d);
    CUDA_CHECK(cudaMemsetAsync(d.count->p, 0, sizeof(unsigned long long), d.s));
    if (v == V_STREAM || v == V_STREAM_EV) {
        enqueue_stages(c, d, d.s, nullptr, false);
    } else if (v == V_GRAPH || v == V_GRAPH_EV || v == V_BAR_GRAPH || v == V_COOP_GRAPH) {
        graph_handle h;
        capture(h, d.s, [&] {
            if (per_stage(v)) {
                enqueue_stages(c, d, d.s, nullptr, true);
            } else {
                enqueue_persistent(c, d, d.s, v == V_BAR_GRAPH ? V_BAR : V_COOP_GRAPH, grid);
            }
        });
        CUDA_CHECK(cudaGraphLaunch(h.ge, d.s));
        CUDA_CHECK(cudaStreamSynchronize(d.s));
    } else {
        enqueue_persistent(c, d, d.s, v, grid);
    }
    CUDA_CHECK(cudaStreamSynchronize(d.s));
    std::vector<float> out(NMAX);
    CUDA_CHECK(cudaMemcpy(out.data(), (c.st.size() & 1) ? d.buf1->p : d.buf0->p, NMAX * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return out;
}

int check_chain(const chain & c, device_ctx & d, const std::vector<int> & grids) {
    const std::vector<double> ref = reference(c, d.init0, d.init1);
    double ref_max = 1e-30;
    for (double x : ref) {
        ref_max = std::max(ref_max, fabs(x));
    }
    const std::vector<float> base = run_once(c, d, V_STREAM, 0);
    double err = 0.0;
    for (int i = 0; i < NMAX; ++i) {
        err = std::max(err, fabs((double) base[i] - ref[i]));
    }
    int failures = 0;
    const bool ok_ref = err / ref_max < 1e-3 && std::isfinite(err);
    failures += !ok_ref;
    printf("  check %-10s stream against the CPU: max error %.2e of max |value| %.3g  %s\n", c.name.c_str(), err, ref_max,
           ok_ref ? "ok" : "FAIL");
    // the events change nothing in the stages, so the two +events variants need no check of their own
    int checked = 0, identical = 0;
    for (int v : { V_GRAPH, V_COOP, V_COOP_GRAPH, V_BAR, V_BAR_GRAPH, V_LOCAL }) {
        const std::vector<int> gl = per_stage(v) ? std::vector<int>{ 0 } : grids;
        for (int g : gl) {
            if (!runnable(d, v, g)) {
                continue;
            }
            const std::vector<float> o = run_once(c, d, v, g);
            double diff = 0.0;
            for (int i = 0; i < NMAX; ++i) {
                diff = std::max(diff, fabs((double) o[i] - base[i]));
            }
            const bool same = memcmp(o.data(), base.data(), NMAX * sizeof(float)) == 0;
            const bool ok = diff / ref_max < 1e-4 && std::isfinite(diff);
            failures += !ok;
            ++checked;
            identical += same;
            if (!same || !ok) {
                printf("  check %-10s %-14s grid %3d: %s, max difference %.2e  %s\n", c.name.c_str(), VARIANT_NAMES[v], g,
                       same ? "bit-identical" : "not bit-identical", diff, ok ? "ok" : "FAIL");
            }
        }
    }
    printf("  check %-10s %d variant/grid runs against the stream launches: %d bit-identical\n", c.name.c_str(), checked,
           identical);
    return failures;
}

// ---- timing one chain with every variant ----------------------------------------------------------------------------

void time_chain(const chain & c, device_ctx & d, const options & opt, const std::vector<int> & grids) {
    const int n = (int) c.st.size();
    // ~20 ms per measurement at an assumed ~3 us per stage, at least 5 chains
    const int reps = opt.smoke ? 2 : (opt.reps > 0 ? opt.reps : std::max(5, 20000 / std::max(1, 3 * n)));
    const int trials = opt.smoke ? 1 : opt.trials;
    std::map<std::string, int> kinds;
    for (const stage & st : c.st) {
        kinds[stage_key(st)]++;
    }
    std::string kind_list;
    for (auto & k : kinds) {
        kind_list += (kind_list.empty() ? "" : " ") + k.first + " x" + std::to_string(k.second);
    }
    printf("\n%s: %d stages (%s), %.1f KB of weights\n", c.name.c_str(), n, kind_list.c_str(), c.weight_bytes / 1024.0);
    printf("  %-16s %5s %12s %12s\n", "variant", "grid", "us/chain", "us/stage");
    auto report = [&](int v, int grid, double ms) {
        printf("  %-16s %5s %12.2f %12.3f\n", VARIANT_NAMES[v], grid ? std::to_string(grid).c_str() : "-", 1000.0 * ms,
               1000.0 * ms / n);
        csv_row("timing", c.name, n, VARIANT_NAMES[v], grid, "us_per_chain", 1000.0 * ms, "us");
        csv_row("timing", c.name, n, VARIANT_NAMES[v], grid, "us_per_stage", 1000.0 * ms / n, "us");
    };
    std::vector<cudaEvent_t> ev(n + 1);
    for (auto & e : ev) {
        CUDA_CHECK(cudaEventCreate(&e));
    }
    std::map<std::string, std::vector<double>> iv_stream, iv_graph;

    // per-stage launches
    report(V_STREAM, 0, time_ms([&] { enqueue_stages(c, d, d.s, nullptr, false); }, d.s, reps, trials));
    report(V_STREAM_EV, 0, time_ms([&] { enqueue_stages(c, d, d.s, &ev, false); }, d.s, reps, trials));
    event_intervals(c, d, ev, [&] { enqueue_stages(c, d, d.s, &ev, false); }, opt.smoke ? 1 : 20, iv_stream);
    {
        graph_handle h;
        capture(h, d.s, [&] { enqueue_stages(c, d, d.s, nullptr, true); });
        report(V_GRAPH, 0, time_ms([&] { CUDA_CHECK(cudaGraphLaunch(h.ge, d.s)); }, d.s, reps, trials));
    }
    {
        graph_handle h;
        capture(h, d.s, [&] { enqueue_stages(c, d, d.s, &ev, true); });
        report(V_GRAPH_EV, 0, time_ms([&] { CUDA_CHECK(cudaGraphLaunch(h.ge, d.s)); }, d.s, reps, trials));
        event_intervals(c, d, ev, [&] { CUDA_CHECK(cudaGraphLaunch(h.ge, d.s)); }, opt.smoke ? 1 : 20, iv_graph);
    }

    // persistent kernels
    for (int g : grids) {
        for (int v : { V_COOP, V_COOP_GRAPH, V_BAR, V_BAR_GRAPH, V_LOCAL }) {
            if (!runnable(d, v, g)) {
                continue;
            }
            CUDA_CHECK(cudaMemset(d.count->p, 0, sizeof(unsigned long long)));
            if (v == V_BAR_GRAPH || v == V_COOP_GRAPH) {
                graph_handle h;
                capture(h, d.s, [&] { enqueue_persistent(c, d, d.s, v == V_BAR_GRAPH ? V_BAR : V_COOP_GRAPH, g); });
                report(v, g, time_ms([&] { CUDA_CHECK(cudaGraphLaunch(h.ge, d.s)); }, d.s, reps, trials));
            } else {
                report(v, g, time_ms([&] { enqueue_persistent(c, d, d.s, v, g); }, d.s, reps, trials));
            }
        }
    }

    // what the event profiler records per stage
    printf("  per stage between events (what GGML_CUDA_PROFILE records), us: median [p10, p90]\n");
    for (auto * iv : { &iv_stream, &iv_graph }) {
        const char * vn = iv == &iv_stream ? "stream+events" : "graph+events";
        std::string line;
        for (auto & k : *iv) {
            char b[128];
            snprintf(b, sizeof b, "  %s %.2f [%.2f, %.2f]", k.first.c_str(), median(k.second), pctl(k.second, 0.1),
                     pctl(k.second, 0.9));
            line += b;
            csv_row("events", c.name, n, vn, 0, k.first + "_median", median(k.second), "us");
            csv_row("events", c.name, n, vn, 0, k.first + "_p10", pctl(k.second, 0.1), "us");
            csv_row("events", c.name, n, vn, 0, k.first + "_p90", pctl(k.second, 0.9), "us");
        }
        printf("    %-14s%s\n", vn, line.c_str());
    }
    for (auto & e : ev) {
        CUDA_CHECK(cudaEventDestroy(e));
    }
}

// ---- the hop ------------------------------------------------------------------------------------------------------

void run_hop(int da, int db, const options & opt) {
    int n_dev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_dev));
    if (da == db || da < 0 || db < 0 || da >= n_dev || db >= n_dev) {
        printf("\nhop: needs two different devices below %d (got %d,%d); skipped\n", n_dev, da, db);
        return;
    }
    int can_peer = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_peer, db, da));
    cudaDeviceProp pa{}, pb{};
    CUDA_CHECK(cudaGetDeviceProperties(&pa, da));
    CUDA_CHECK(cudaGetDeviceProperties(&pb, db));
    printf("\nhop: card %d (%s) -> card %d (%s); peer access possible: %s; unified addressing: %d/%d\n", da, pa.name, db,
           pb.name, can_peer ? "yes" : "no", pa.unifiedAddressing, pb.unifiedAddressing);
    std::vector<int> sizes;
    for (int b : parse_ints(opt.hop_sizes, 0)) {
        if (b >= 16) {
            sizes.push_back(b / 16 * 16);  // whole float4s
        }
    }
    if (sizes.empty()) {
        printf("  no hop size of 16 bytes or more; skipped\n");
        return;
    }
    const int max_bytes = *std::max_element(sizes.begin(), sizes.end());
    const int reps = opt.smoke ? 3 : opt.hop_reps;

    cudaStream_t sa, sb;
    void *src, *dst, *host, *mail;
    unsigned * flag;
    unsigned * timeouts;
    cudaEvent_t ev_a;
    CUDA_CHECK(cudaSetDevice(da));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&src, max_bytes));
    CUDA_CHECK(cudaMemset(src, 0, max_bytes));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_a, cudaEventDisableTiming));
    CUDA_CHECK(cudaSetDevice(db));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&dst, max_bytes));
    CUDA_CHECK(cudaMalloc(&timeouts, sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(timeouts, 0, sizeof(unsigned)));
    CUDA_CHECK(cudaHostAlloc(&host, max_bytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&mail, max_bytes, cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &flag, 64, cudaHostAllocPortable | cudaHostAllocMapped));
    *(volatile unsigned *) flag = 0;
    // the mailbox and its flag as each card addresses them (the host pointers themselves under unified addressing)
    void *mail_a, *mail_b, *flag_a, *flag_b;
    CUDA_CHECK(cudaHostGetDevicePointer(&mail_b, mail, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&flag_b, flag, 0));
    CUDA_CHECK(cudaSetDevice(da));
    CUDA_CHECK(cudaHostGetDevicePointer(&mail_a, mail, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&flag_a, flag, 0));

    using clk = std::chrono::steady_clock;
    auto us_since = [](clk::time_point t0) { return std::chrono::duration<double, std::micro>(clk::now() - t0).count(); };
    unsigned seq = 0;
    printf("  %-10s %14s %14s %14s   (us per hop, median [p10, p90] of %d)\n", "bytes", "host-staged", "peer-copy",
           "mailbox", reps);
    for (int bytes : sizes) {
        std::vector<double> t_staged, t_peer, t_mail;
        for (int r = 0; r < reps + 5; ++r) {
            // 1. two copies through a pinned host buffer, the host waiting on each (what ggml does when it has no
            //    asynchronous copy between the two backends)
            clk::time_point t0 = clk::now();
            CUDA_CHECK(cudaSetDevice(da));
            CUDA_CHECK(cudaMemcpyAsync(host, src, bytes, cudaMemcpyDeviceToHost, sa));
            CUDA_CHECK(cudaStreamSynchronize(sa));
            CUDA_CHECK(cudaSetDevice(db));
            CUDA_CHECK(cudaMemcpyAsync(dst, host, bytes, cudaMemcpyHostToDevice, sb));
            CUDA_CHECK(cudaStreamSynchronize(sb));
            if (r >= 5) {
                t_staged.push_back(us_since(t0));
            }
            // 2. one peer copy (the driver stages it through the host without peer access), card B's stream waiting
            //    on an event and running one small kernel after it (ggml's asynchronous copy between CUDA backends)
            t0 = clk::now();
            CUDA_CHECK(cudaSetDevice(da));
            CUDA_CHECK(cudaMemcpyPeerAsync(dst, db, src, da, bytes, sa));
            CUDA_CHECK(cudaEventRecord(ev_a, sa));
            CUDA_CHECK(cudaSetDevice(db));
            CUDA_CHECK(cudaStreamWaitEvent(sb, ev_a, 0));
            k_touch<<<1, 32, 0, sb>>>((float *) dst);
            CUDA_CHECK(cudaStreamSynchronize(sb));
            if (r >= 5) {
                t_peer.push_back(us_since(t0));
            }
            // 3. a mailbox in mapped host memory: card B's kernel already polls; card A's kernel writes and raises
            //    the flag
            ++seq;
            t0 = clk::now();
            CUDA_CHECK(cudaSetDevice(db));
            k_mail_recv<<<1, THREADS, 0, sb>>>((const float4 *) mail_b, (float4 *) dst, bytes / 16, (const unsigned *) flag_b,
                                               seq, timeouts);
            CUDA_CHECK(cudaSetDevice(da));
            k_mail_send<<<1, THREADS, 0, sa>>>((const float4 *) src, (float4 *) mail_a, bytes / 16, (unsigned *) flag_a, seq);
            CUDA_CHECK(cudaSetDevice(db));
            CUDA_CHECK(cudaStreamSynchronize(sb));
            if (r >= 5) {
                t_mail.push_back(us_since(t0));
            }
            CUDA_CHECK(cudaSetDevice(da));
            CUDA_CHECK(cudaStreamSynchronize(sa));
        }
        printf("  %-10d %6.1f [%.0f,%.0f] %6.1f [%.0f,%.0f] %6.1f [%.0f,%.0f]\n", bytes, median(t_staged),
               pctl(t_staged, 0.1), pctl(t_staged, 0.9), median(t_peer), pctl(t_peer, 0.1), pctl(t_peer, 0.9),
               median(t_mail), pctl(t_mail, 0.1), pctl(t_mail, 0.9));
        const std::string pair = std::to_string(da) + "-" + std::to_string(db);
        csv_row("hop", pair, bytes, "host-staged", 0, "us_median", median(t_staged), "us");
        csv_row("hop", pair, bytes, "peer-copy", 0, "us_median", median(t_peer), "us");
        csv_row("hop", pair, bytes, "mailbox", 0, "us_median", median(t_mail), "us");
        csv_row("hop", pair, bytes, "host-staged", 0, "us_p10", pctl(t_staged, 0.1), "us");
        csv_row("hop", pair, bytes, "peer-copy", 0, "us_p10", pctl(t_peer, 0.1), "us");
        csv_row("hop", pair, bytes, "mailbox", 0, "us_p10", pctl(t_mail, 0.1), "us");
    }
    unsigned to = 0;
    CUDA_CHECK(cudaSetDevice(db));
    CUDA_CHECK(cudaMemcpy(&to, timeouts, sizeof to, cudaMemcpyDeviceToHost));
    printf("  mailbox timeouts: %u%s\n", to, to ? "  (the mailbox numbers are not valid)" : "");
    CUDA_CHECK(cudaFreeHost(host));
    CUDA_CHECK(cudaFreeHost(mail));
    CUDA_CHECK(cudaFreeHost(flag));
    CUDA_CHECK(cudaFree(dst));
    CUDA_CHECK(cudaFree(timeouts));
    CUDA_CHECK(cudaStreamDestroy(sb));
    CUDA_CHECK(cudaSetDevice(da));
    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaEventDestroy(ev_a));
    CUDA_CHECK(cudaStreamDestroy(sa));
}

}  // namespace

int main(int argc, char ** argv) {
    options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                usage();
                exit(1);
            }
            return argv[++i];
        };
        if (a == "--mixes") opt.mixes = next();
        else if (a == "--stages") opt.stages = atoi(next());
        else if (a == "--grids") opt.grids = next();
        else if (a == "--reps") opt.reps = atoi(next());
        else if (a == "--trials") opt.trials = atoi(next());
        else if (a == "--check-only") opt.check_only = true;
        else if (a == "--no-check") opt.no_check = true;
        else if (a == "--smoke") opt.smoke = true;
        else if (a == "--hop") opt.hop = next();
        else if (a == "--hop-sizes") opt.hop_sizes = next();
        else if (a == "--hop-reps") opt.hop_reps = atoi(next());
        else if (a == "--hop-only") opt.hop_only = true;
        else if (a == "--csv") opt.csv = next();
        else { usage(); return a == "-h" || a == "--help" ? 0 : 1; }
    }
    if (opt.csv) {
        g_csv = fopen(opt.csv, "w");
        if (!g_csv) {
            perror(opt.csv);
            return 1;
        }
        fprintf(g_csv, "section,mix,stages,variant,grid,key,value,unit\n");
    }

    device_ctx d;
    CUDA_CHECK(cudaGetDevice(&d.dev));
    CUDA_CHECK(cudaGetDeviceProperties(&d.prop, d.dev));
    int code_arch = 0;
    {
        dev_mem a(sizeof(int));
        k_arch<<<1, 1>>>(a.as<int>());
        CUDA_CHECK(cudaMemcpy(&code_arch, a.p, sizeof(int), cudaMemcpyDeviceToHost));
    }
    int coop_attr = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&coop_attr, cudaDevAttrCooperativeLaunch, d.dev));
    d.coop = coop_attr != 0;
    printf("device %d: %s, compute %d.%d, %d SMs, running code for compute %d.%d; cooperative launch: %s\n", d.dev,
           d.prop.name, d.prop.major, d.prop.minor, d.prop.multiProcessorCount, code_arch / 100, (code_arch % 100) / 10,
           d.coop ? "yes" : "no");
    csv_row("device", d.prop.name, 0, "", 0, "code_arch", code_arch, "");
    csv_row("device", d.prop.name, 0, "", 0, "sms", d.prop.multiProcessorCount, "");

    if (!opt.hop.empty() && opt.hop_only) {
        const std::vector<int> p = parse_ints(opt.hop, 0);
        run_hop(p.size() > 0 ? p[0] : 0, p.size() > 1 ? p[1] : 1, opt);
        if (g_csv) {
            fclose(g_csv);
        }
        return 0;
    }

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ[0], k_persistent<0>, THREADS, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ[1], k_persistent<1>, THREADS, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ[2], k_persistent<2>, THREADS, 0));
    for (int v = 0; v < 3; ++v) {
        cudaFuncAttributes fa{};
        CUDA_CHECK(cudaFuncGetAttributes(&fa, v == 0 ? (const void *) k_persistent<0>
                                                     : v == 1 ? (const void *) k_persistent<1> : (const void *) k_persistent<2>));
        printf("k_persistent<%d>: %d registers, %zu B static shared memory, %d resident blocks of %d threads per SM\n", v,
               fa.numRegs, fa.sharedSizeBytes, d.occ[v], THREADS);
        csv_row("occupancy", "", 0, v == 0 ? "barrier" : v == 1 ? "barrier-local" : "coop", 0, "blocks_per_sm", d.occ[v], "");
    }
    const int max_grid = d.coop ? std::min(d.max_grid(V_BAR), d.max_grid(V_COOP)) : d.max_grid(V_BAR);
    std::vector<int> grids = parse_ints(opt.grids, max_grid);
    std::sort(grids.begin(), grids.end());
    grids.erase(std::unique(grids.begin(), grids.end()), grids.end());
    printf("persistent grids:");
    for (int g : grids) {
        printf(" %d", g);
    }
    printf(" (largest resident grid %d)\n", max_grid);

    CUDA_CHECK(cudaStreamCreateWithFlags(&d.s, cudaStreamNonBlocking));
    d.buf0 = new dev_mem(NMAX * sizeof(float));
    d.buf1 = new dev_mem(NMAX * sizeof(float));
    d.count = new dev_mem(sizeof(unsigned long long));

    // can a cooperative launch be captured into a CUDA graph? (tried on a stream of its own, errors cleared)
    if (d.coop) {
        cudaStream_t ps;
        CUDA_CHECK(cudaStreamCreateWithFlags(&ps, cudaStreamNonBlocking));
        const stage empty[2] = {};  // two empty stages: one grid.sync() between them
        dev_mem t(sizeof empty);
        CUDA_CHECK(cudaMemcpy(t.p, empty, sizeof empty, cudaMemcpyHostToDevice));
        const stage * table = t.as<stage>();
        int one = 2;
        float * b0 = d.buf0->as<float>();
        float * b1 = d.buf1->as<float>();
        unsigned long long * cnt = d.count->as<unsigned long long>();
        void * args[] = { (void *) &table, (void *) &one, (void *) &b0, (void *) &b1, (void *) &cnt };
        cudaGraph_t g = nullptr;
        cudaGraphExec_t ge = nullptr;
        cudaError_t e = cudaStreamBeginCapture(ps, cudaStreamCaptureModeThreadLocal);
        if (e == cudaSuccess) {
            e = cudaLaunchCooperativeKernel((const void *) k_persistent<2>, dim3(d.prop.multiProcessorCount), dim3(THREADS),
                                            args, 0, ps);
            const cudaError_t e2 = cudaStreamEndCapture(ps, &g);
            e = e != cudaSuccess ? e : e2;
        }
        if (e == cudaSuccess) {
            e = cudaGraphInstantiate(&ge, g, 0);
        }
        if (e == cudaSuccess) {
            e = cudaGraphLaunch(ge, ps);
        }
        if (e == cudaSuccess) {
            e = cudaStreamSynchronize(ps);
        }
        d.coop_graph = e == cudaSuccess;
        printf("cooperative launch inside a CUDA graph: %s\n", d.coop_graph ? "yes" : cudaGetErrorString(e));
        (void) cudaGetLastError();
        if (ge) {
            cudaGraphExecDestroy(ge);
        }
        if (g) {
            cudaGraphDestroy(g);
        }
        CUDA_CHECK(cudaStreamDestroy(ps));
        csv_row("device", d.prop.name, 0, "", 0, "coop_in_graph", d.coop_graph, "");
    }
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        d.init0.resize(NMAX);
        d.init1.resize(NMAX);
        for (int i = 0; i < NMAX; ++i) {
            d.init0[i] = u(rng);
            d.init1[i] = u(rng);
        }
    }

    // reference points: stream memory for ~2 s to reach the sustained clock, then an empty kernel as a graph node
    if (!opt.check_only) {
        const size_t bytes = (size_t) 512 << 20;
        dev_mem big(bytes);
        dev_mem sink(sizeof(unsigned));
        CUDA_CHECK(cudaMemset(big.p, 1, bytes));
        const int blocks = d.prop.multiProcessorCount * 8;
        auto read_once = [&] { k_read<<<blocks, THREADS, 0, d.s>>>(big.as<uint4>(), bytes / 16, sink.as<unsigned>()); };
        double read_ms = 0.0;
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < (opt.smoke ? 0.05 : 2.0)) {
            read_ms = time_ms(read_once, d.s, 3, 1);
        }
        const double gbs = bytes / (read_ms * 1e-3) / 1e9;
        graph_handle h;
        capture(h, d.s, [&] {
            for (int j = 0; j < 64; ++j) {
                k_empty<<<1, 32, 0, d.s>>>();
            }
        });
        const double node_us = 1000.0 * time_ms([&] { CUDA_CHECK(cudaGraphLaunch(h.ge, d.s)); }, d.s, opt.smoke ? 2 : 50,
                                                opt.smoke ? 1 : opt.trials) / 64;
        const double eager_us = 1000.0 * time_ms([&] {
            for (int j = 0; j < 64; ++j) {
                k_empty<<<1, 32, 0, d.s>>>();
            }
        }, d.s, opt.smoke ? 2 : 50, opt.smoke ? 1 : opt.trials) / 64;
        printf("streaming read %.1f GB/s; empty kernel: %.2f us per graph node, %.2f us per eager launch\n", gbs, node_us,
               eager_us);
        csv_row("reference", "", 0, "", 0, "stream_read_gbs", gbs, "GB/s");
        csv_row("reference", "", 0, "graph", 0, "empty_kernel_us", node_us, "us");
        csv_row("reference", "", 0, "stream", 0, "empty_kernel_us", eager_us, "us");
    }

    int failures = 0;
    std::string list = opt.mixes;
    std::vector<std::string> names;
    // split on commas that are not inside brackets
    {
        int depth = 0;
        std::string cur;
        for (char ch : list) {
            depth += ch == '[' ? 1 : ch == ']' ? -1 : 0;
            if (ch == ',' && depth == 0) {
                names.push_back(cur);
                cur.clear();
            } else {
                cur += ch;
            }
        }
        if (!cur.empty()) {
            names.push_back(cur);
        }
    }
    for (const std::string & name : names) {
        mix_def m;
        if (!make_mix(name, opt, m)) {
            fprintf(stderr, "bad mix '%s'\n", name.c_str());
            return 1;
        }
        chain c;
        build_chain(m, c);
        if (!opt.no_check) {
            failures += check_chain(c, d, grids);
        }
        if (!opt.check_only) {
            time_chain(c, d, opt, grids);
        }
    }
    if (!opt.no_check) {
        printf("\nchecks: %s\n", failures ? "FAILED" : "all passed");
        csv_row("check", "", 0, "", 0, "failures", failures, "");
    }

    if (!opt.hop.empty()) {
        const std::vector<int> p = parse_ints(opt.hop, 0);
        run_hop(p.size() > 0 ? p[0] : 0, p.size() > 1 ? p[1] : 1, opt);
        CUDA_CHECK(cudaSetDevice(d.dev));
    }
    delete d.buf0;
    delete d.buf1;
    delete d.count;
    CUDA_CHECK(cudaStreamDestroy(d.s));
    if (g_csv) {
        fclose(g_csv);
    }
    return failures ? 2 : 0;
}
