// p2p_check: does peer-to-peer between two cards deliver the right bytes, and how fast is it against a copy staged
// through host memory? Written for forcing the NVIDIA driver's peer-to-peer on cards it refuses by policy
// (`NVreg_RegistryDwords="ForceP2P=0x11"`), where a wrong copy or a hang is possible, so every test checks the data
// and prints before it starts.
//
// Usage: p2p_check [--pairs A:B,...|none] [--tests copy,store,load,time] [--mb N] [--rounds N] [--reps N]
//                  [--host-cards C,...]
//   --host-cards  host -> card and card -> host through pinned memory on each listed card, every word checked;
//                 timed too when --tests has time (no peer access needed)
//   --pairs   ordered card pairs, A sends to B (default 0:1)
//   --tests   control: a copy staged through host memory, counted on B (proves the checker)
//             copy:  cudaMemcpyPeer A -> B of a pattern, counted on B
//             store: a kernel on A writes the pattern into B's memory through the peer pointer, counted on B
//             load:  a kernel on A reads a pattern from B's memory through the peer pointer, counted on A
//             time:  host-timed cudaMemcpyPeerAsync against the staged copy (A -> pinned host -> B), median of --reps
//   --mb      test buffer in MiB (default 64); --rounds patterns per test (default 3); --reps timing repeats (200)
// Exit code 0 when every requested test delivered every word, 1 when any word was wrong, 2 on a CUDA error.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                                                    \
    do {                                                                                                         \
        cudaError_t e_ = (x);                                                                                    \
        if (e_ != cudaSuccess) {                                                                                 \
            printf("CUDA error %s at line %d (%s): %s\n", cudaGetErrorName(e_), __LINE__, #x, cudaGetErrorString(e_)); \
            fflush(stdout);                                                                                      \
            exit(2);                                                                                             \
        }                                                                                                        \
    } while (0)

__host__ __device__ __forceinline__ unsigned pattern(unsigned i, unsigned seed) {
    unsigned x = i * 2654435761u ^ seed;
    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;
    return x;
}

__global__ void k_fill(unsigned * p, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        p[i] = pattern((unsigned) i, seed);
    }
}

// cnt[0]: words that differ from the pattern; cnt[1]: of those, words still 0 (nothing arrived there)
__global__ void k_count_bad(const unsigned * p, size_t n, unsigned seed, unsigned long long * cnt) {
    unsigned long long bad = 0, zero = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const unsigned v = p[i];
        if (v != pattern((unsigned) i, seed)) {
            ++bad;
            zero += v == 0;
        }
    }
    if (bad) {
        atomicAdd(&cnt[0], bad);
        atomicAdd(&cnt[1], zero);
    }
}

static unsigned long long g_last_zero = 0;

// runs on one card; either pointer may be a peer card's memory
__global__ void k_copy(unsigned * dst, const unsigned * src, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

static const int kBlocks = 160, kThreads = 256;

static unsigned long long count_bad(int dev, const unsigned * p, size_t n, unsigned seed) {
    CK(cudaSetDevice(dev));
    unsigned long long * d_cnt;
    unsigned long long cnt[2] = { 0, 0 };
    CK(cudaMalloc(&d_cnt, sizeof cnt));
    CK(cudaMemset(d_cnt, 0, sizeof cnt));
    k_count_bad<<<kBlocks, kThreads>>>(p, n, seed, d_cnt);
    CK(cudaGetLastError());
    CK(cudaMemcpy(cnt, d_cnt, sizeof cnt, cudaMemcpyDeviceToHost));
    CK(cudaFree(d_cnt));
    const unsigned long long bad = cnt[0];
    g_last_zero = cnt[1];
    // the counter itself is checked against the host: 1,024 words at the start, middle and end
    std::vector<unsigned> h(1024);
    const size_t offs[3] = { 0, n / 2, n - 1024 };
    unsigned long long host_bad = 0;
    for (size_t o : offs) {
        CK(cudaMemcpy(h.data(), p + o, 1024 * sizeof(unsigned), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < 1024; ++i) {
            host_bad += h[i] != pattern((unsigned) (o + i), seed);
        }
    }
    if (host_bad && !bad) {
        printf("    the card's counter missed %llu wrong words the host found\n", host_bad);
        return host_bad;
    }
    return bad;
}

static std::vector<std::string> split(const std::string & s, char c) {
    std::vector<std::string> out;
    size_t a = 0;
    while (a <= s.size()) {
        size_t b = s.find(c, a);
        if (b == std::string::npos) {
            b = s.size();
        }
        if (b > a) {
            out.push_back(s.substr(a, b - a));
        }
        a = b + 1;
    }
    return out;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static double pctl(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return v[(size_t) (q * (v.size() - 1))];
}

int main(int argc, char ** argv) {
    std::string pairs_arg = "0:1", tests_arg = "copy,store,load,time", host_cards_arg;
    int mb = 64, rounds = 3, reps = 200;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--pairs") {
            pairs_arg = v;
        } else if (k == "--tests") {
            tests_arg = v;
        } else if (k == "--mb") {
            mb = atoi(v.c_str());
        } else if (k == "--rounds") {
            rounds = atoi(v.c_str());
        } else if (k == "--reps") {
            reps = atoi(v.c_str());
        } else if (k == "--host-cards") {
            host_cards_arg = v;
        } else {
            printf("unknown option %s\n", k.c_str());
            return 2;
        }
    }
    auto has = [&](const char * t) {
        auto ts = split(tests_arg, ',');
        return std::find(ts.begin(), ts.end(), std::string(t)) != ts.end();
    };
    setvbuf(stdout, nullptr, _IONBF, 0);

    int n_dev = 0;
    CK(cudaGetDeviceCount(&n_dev));
    printf("cards: %d\npeer access as the driver reports it (row: from, column: to):\n     ", n_dev);
    for (int b = 0; b < n_dev; ++b) {
        printf(" %d", b);
    }
    printf("\n");
    for (int a = 0; a < n_dev; ++a) {
        printf("  %d: ", a);
        for (int b = 0; b < n_dev; ++b) {
            int can = 0;
            if (a != b) {
                CK(cudaDeviceCanAccessPeer(&can, a, b));
            }
            printf(" %c", a == b ? '-' : (can ? 'Y' : 'n'));
        }
        printf("\n");
    }

    const size_t n = (size_t) mb * 1024 * 1024 / sizeof(unsigned);
    const size_t bytes = n * sizeof(unsigned);
    int any_bad = 0;

    // --host-cards: host <-> card copies through pinned memory on each listed card, every word checked, then timed
    for (const auto & cs : split(host_cards_arg, ',')) {
        const int c = atoi(cs.c_str());
        if (c < 0 || c >= n_dev) {
            printf("bad card %s\n", cs.c_str());
            return 2;
        }
        printf("\n== card %d: host <-> card through pinned memory\n", c);
        CK(cudaSetDevice(c));
        unsigned *d, *h_in, *h_out;
        CK(cudaMalloc(&d, bytes));
        CK(cudaHostAlloc(&h_in, bytes, cudaHostAllocPortable));
        CK(cudaHostAlloc(&h_out, bytes, cudaHostAllocPortable));
        for (int r = 0; r < rounds; ++r) {
            const unsigned seed = 0xc2b2ae35u * (unsigned) (r + 1) + (unsigned) c;
            for (size_t i = 0; i < n; ++i) {
                h_in[i] = pattern((unsigned) i, seed);
            }
            CK(cudaMemset(d, 0, bytes));
            CK(cudaMemcpy(d, h_in, bytes, cudaMemcpyHostToDevice));
            const unsigned long long bad_in = count_bad(c, d, n, seed);
            printf("  round %d host -> card %d MiB ... %s (%llu of %zu words wrong, %llu of them still 0)\n", r, mb,
                   bad_in ? "WRONG" : "ok", bad_in, n, g_last_zero);
            any_bad |= bad_in != 0;
            k_fill<<<kBlocks, kThreads>>>(d, n, seed ^ 0x165667b1u);
            CK(cudaDeviceSynchronize());
            memset(h_out, 0, bytes);
            CK(cudaMemcpy(h_out, d, bytes, cudaMemcpyDeviceToHost));
            unsigned long long bad_out = 0;
            for (size_t i = 0; i < n; ++i) {
                bad_out += h_out[i] != pattern((unsigned) i, seed ^ 0x165667b1u);
            }
            printf("  round %d card -> host %d MiB ... %s (%llu of %zu words wrong)\n", r, mb, bad_out ? "WRONG" : "ok",
                   bad_out, n);
            any_bad |= bad_out != 0;
        }
        if (has("time")) {
            printf("  timing, host clock, median [p10, p90] of %d (microseconds; MB/s from the median):\n", reps);
            printf("  %10s %26s %26s\n", "bytes", "host -> card", "card -> host");
            const size_t sizes[] = { 10240, 163840, 1 << 20, 16 << 20, 64 << 20 };
            for (size_t sz : sizes) {
                if (sz > bytes) {
                    continue;
                }
                const int rp = sz >= (16u << 20) ? std::max(5, reps / 20) : reps;
                std::vector<double> t_in, t_out;
                for (int i = 0; i < rp + 3; ++i) {
                    auto t0 = std::chrono::steady_clock::now();
                    CK(cudaMemcpy(d, h_in, sz, cudaMemcpyHostToDevice));
                    auto t1 = std::chrono::steady_clock::now();
                    CK(cudaMemcpy(h_out, d, sz, cudaMemcpyDeviceToHost));
                    auto t2 = std::chrono::steady_clock::now();
                    if (i >= 3) {
                        t_in.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                        t_out.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
                    }
                }
                const double mi = median(t_in), mo = median(t_out);
                printf("  %10zu %9.1f [%7.1f, %7.1f] %5.0f %9.1f [%7.1f, %7.1f] %5.0f\n", sz, mi, pctl(t_in, 0.1),
                       pctl(t_in, 0.9), sz / mi, mo, pctl(t_out, 0.1), pctl(t_out, 0.9), sz / mo);
            }
        }
        CK(cudaFree(d));
        CK(cudaFreeHost(h_in));
        CK(cudaFreeHost(h_out));
    }

    for (const auto & ps : split(pairs_arg == "none" ? std::string() : pairs_arg, ',')) {
        auto ab = split(ps, ':');
        if (ab.size() != 2) {
            printf("bad pair %s\n", ps.c_str());
            return 2;
        }
        const int a = atoi(ab[0].c_str()), b = atoi(ab[1].c_str());
        if (a == b || a < 0 || b < 0 || a >= n_dev || b >= n_dev) {
            printf("bad pair %s\n", ps.c_str());
            return 2;
        }
        int can_ab = 0, can_ba = 0;
        CK(cudaDeviceCanAccessPeer(&can_ab, a, b));
        CK(cudaDeviceCanAccessPeer(&can_ba, b, a));
        printf("\n== pair %d -> %d: can access peer %d->%d %s, %d->%d %s\n", a, b, a, b, can_ab ? "yes" : "no", b, a,
               can_ba ? "yes" : "no");
        if (!can_ab && !has("control")) {
            printf("  skipped: the driver gives no peer access\n");
            continue;
        }
        if (!can_ab) {
            printf("  no peer access: only the control copy runs\n");
        }
        CK(cudaSetDevice(a));
        if (can_ab) {
            cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
                CK(e);
            }
            cudaGetLastError();
        }
        unsigned *pa, *pb;
        CK(cudaMalloc(&pa, bytes));
        CK(cudaSetDevice(b));
        CK(cudaMalloc(&pb, bytes));

        for (int r = 0; r < rounds; ++r) {
            const unsigned seed = 0x9e3779b9u * (unsigned) (r + 1) + (unsigned) (a * 16 + b);
            if (has("control")) {
                // the same check on a copy staged through host memory, which must always pass: it proves the checker
                printf("  round %d control (staged through host %d -> %d, %d MiB) ... ", r, a, b, mb);
                std::vector<unsigned> h(n);
                CK(cudaSetDevice(a));
                k_fill<<<kBlocks, kThreads>>>(pa, n, seed ^ 0x85ebca6bu);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(h.data(), pa, bytes, cudaMemcpyDeviceToHost));
                CK(cudaSetDevice(b));
                CK(cudaMemset(pb, 0, bytes));
                CK(cudaMemcpy(pb, h.data(), bytes, cudaMemcpyHostToDevice));
                const unsigned long long bad = count_bad(b, pb, n, seed ^ 0x85ebca6bu);
                printf("%s (%llu of %zu words wrong, %llu of them still 0)\n", bad ? "WRONG" : "ok", bad, n, g_last_zero);
                any_bad |= bad != 0;
            }
            if (can_ab && has("copy")) {
                printf("  round %d copy  (cudaMemcpyPeer %d -> %d, %d MiB) ... ", r, a, b, mb);
                CK(cudaSetDevice(a));
                k_fill<<<kBlocks, kThreads>>>(pa, n, seed);
                CK(cudaDeviceSynchronize());
                CK(cudaSetDevice(b));
                CK(cudaMemset(pb, 0, bytes));
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpyPeer(pb, b, pa, a, bytes));
                CK(cudaDeviceSynchronize());
                const unsigned long long bad = count_bad(b, pb, n, seed);
                printf("%s (%llu of %zu words wrong, %llu of them still 0)\n", bad ? "WRONG" : "ok", bad, n, g_last_zero);
                any_bad |= bad != 0;
            }
            if (can_ab && has("store")) {
                printf("  round %d store (kernel on %d writes into %d) ... ", r, a, b);
                CK(cudaSetDevice(b));
                CK(cudaMemset(pb, 0, bytes));
                CK(cudaDeviceSynchronize());
                CK(cudaSetDevice(a));
                k_fill<<<kBlocks, kThreads>>>(pa, n, seed ^ 0x5bd1e995u);
                k_copy<<<kBlocks, kThreads>>>(pb, pa, n);
                CK(cudaGetLastError());
                CK(cudaDeviceSynchronize());
                const unsigned long long bad = count_bad(b, pb, n, seed ^ 0x5bd1e995u);
                printf("%s (%llu of %zu words wrong, %llu of them still 0)\n", bad ? "WRONG" : "ok", bad, n, g_last_zero);
                any_bad |= bad != 0;
            }
            if (can_ab && has("load")) {
                printf("  round %d load  (kernel on %d reads from %d) ... ", r, a, b);
                CK(cudaSetDevice(b));
                k_fill<<<kBlocks, kThreads>>>(pb, n, seed ^ 0x27d4eb2fu);
                CK(cudaDeviceSynchronize());
                CK(cudaSetDevice(a));
                CK(cudaMemset(pa, 0, bytes));
                k_copy<<<kBlocks, kThreads>>>(pa, pb, n);
                CK(cudaGetLastError());
                CK(cudaDeviceSynchronize());
                const unsigned long long bad = count_bad(a, pa, n, seed ^ 0x27d4eb2fu);
                printf("%s (%llu of %zu words wrong, %llu of them still 0)\n", bad ? "WRONG" : "ok", bad, n, g_last_zero);
                any_bad |= bad != 0;
            }
        }

        if (can_ab && has("time")) {
            printf("  timing, host clock, median [p10, p90] of %d (microseconds; MB/s from the median):\n", reps);
            printf("  %10s %26s %26s\n", "bytes", "peer copy", "staged through host");
            cudaStream_t sa, sb;
            CK(cudaSetDevice(a));
            CK(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking));
            CK(cudaSetDevice(b));
            CK(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking));
            void * host;
            const size_t max_t = std::min(bytes, (size_t) 16 << 20);
            CK(cudaHostAlloc(&host, max_t, cudaHostAllocPortable));
            const size_t sizes[] = { 10240, 40960, 163840, 1 << 20, 16 << 20 };
            for (size_t sz : sizes) {
                if (sz > max_t) {
                    continue;
                }
                const int rp = sz >= (16u << 20) ? std::max(10, reps / 20) : reps;
                std::vector<double> tp, ts;
                for (int i = 0; i < rp + 5; ++i) {
                    auto t0 = std::chrono::steady_clock::now();
                    CK(cudaMemcpyPeerAsync(pb, b, pa, a, sz, sa));
                    CK(cudaStreamSynchronize(sa));
                    auto t1 = std::chrono::steady_clock::now();
                    CK(cudaMemcpyAsync(host, pa, sz, cudaMemcpyDeviceToHost, sa));
                    CK(cudaStreamSynchronize(sa));
                    CK(cudaMemcpyAsync(pb, host, sz, cudaMemcpyHostToDevice, sb));
                    CK(cudaStreamSynchronize(sb));
                    auto t2 = std::chrono::steady_clock::now();
                    if (i >= 5) {
                        tp.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                        ts.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
                    }
                }
                const double mp = median(tp), ms = median(ts);
                printf("  %10zu %9.1f [%7.1f, %7.1f] %5.0f %9.1f [%7.1f, %7.1f] %5.0f\n", sz, mp, pctl(tp, 0.1),
                       pctl(tp, 0.9), sz / mp, ms, pctl(ts, 0.1), pctl(ts, 0.9), sz / ms);
            }
            CK(cudaFreeHost(host));
            CK(cudaStreamDestroy(sa));
            CK(cudaStreamDestroy(sb));
        }
        CK(cudaSetDevice(a));
        CK(cudaFree(pa));
        if (can_ab) {
            CK(cudaDeviceDisablePeerAccess(b));
        }
        CK(cudaSetDevice(b));
        CK(cudaFree(pb));
    }
    printf("\nresult: %s\n", any_bad ? "WRONG DATA" : "every word right");
    return any_bad ? 1 : 0;
}
