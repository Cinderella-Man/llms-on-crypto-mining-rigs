// C910 (2026-11-21): what --mtp-device N does to memory, on a CPU-only machine. Two local ggml-rpc-server processes,
// each serving its CPU backend, stand in for two cards (RPC devices are GPU-type devices to llama.cpp), so the output
// head's copy really goes from one "card" to the other. For a model with MTP (nextn) layers and nextn_device -1
// (unset) and 0 (the first RPC device), it prints, as tab-separated lines:
//   fit   the memory fit's view: the model loaded with no_alloc and a context made on it, as common/fit.cpp's dry run
//         does, then llama_get_memory_breakdown: model bytes per buffer type, and their total
//   load  two real loads and frees in one process: the largest single host allocation of this process during each
//         load (malloc and friends are interposed below; std::vector and operator new go through malloc), how many
//         were 64 MiB or more, and the RPC servers' resident memory before the load, after it and after the free (a
//         buffer that is never freed stays resident on its server)
//   head  the expected size of the copy: output.weight (token_embd.weight when the head is tied) and output_norm.weight
//         from the file
// Usage: mtp-device-check <model.gguf> <rpc endpoints, comma-separated> <rpc server pids, comma-separated> [ngl]
// Build: g++ -std=c++17 -O1 -rdynamic mtp_device_check.cpp -I<src>/include -I<src>/ggml/include -I<src>/src
//        -L<build>/bin -lllama -lggml -lggml-base -Wl,-rpath,<build>/bin -o mtp-device-check

#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ---- host allocation tracking: every malloc-family call of the process (libllama, libggml, libstdc++'s operator new)
// resolves to these definitions in the executable, which forward to glibc's own entry points (noexcept, as glibc
// declares them for C++)
extern "C" {
void * __libc_malloc(size_t);
void * __libc_calloc(size_t, size_t);
void * __libc_realloc(void *, size_t);
void * __libc_memalign(size_t, size_t);
}

static std::atomic<bool>   g_track{false};
static std::atomic<size_t> g_max{0};
static std::atomic<size_t> g_n_big{0};
static constexpr size_t    BIG = 64u*1024*1024;

static inline void track(size_t n) {
    if (!g_track.load(std::memory_order_relaxed)) {
        return;
    }
    size_t cur = g_max.load(std::memory_order_relaxed);
    while (n > cur && !g_max.compare_exchange_weak(cur, n, std::memory_order_relaxed)) {
    }
    if (n >= BIG) {
        g_n_big.fetch_add(1, std::memory_order_relaxed);
    }
}

extern "C" {
void * malloc(size_t n) noexcept { track(n); return __libc_malloc(n); }
void * calloc(size_t a, size_t b) noexcept { track(a*b); return __libc_calloc(a, b); }
void * realloc(void * p, size_t n) noexcept { track(n); return __libc_realloc(p, n); }
void * memalign(size_t al, size_t n) noexcept { track(n); return __libc_memalign(al, n); }
void * aligned_alloc(size_t al, size_t n) noexcept { track(n); return __libc_memalign(al, n); }
int posix_memalign(void ** p, size_t al, size_t n) noexcept {
    track(n);
    void * r = __libc_memalign(al, n);
    if (r == nullptr && n != 0) {
        return 12; // ENOMEM
    }
    *p = r;
    return 0;
}
}

// ---- helpers
static long kib_of(const std::string & status_path, const char * key) {
    std::ifstream f(status_path);
    std::string line;
    const size_t kl = strlen(key);
    while (std::getline(f, line)) {
        if (line.compare(0, kl, key) == 0) {
            return atol(line.c_str() + kl + 1);
        }
    }
    return -1;
}

static std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

static long servers_rss_kib(const std::vector<std::string> & pids) {
    long sum = 0;
    for (const auto & p : pids) {
        sum += kib_of("/proc/" + p + "/status", "VmRSS:");
    }
    return sum;
}

static size_t gguf_tensor_bytes(const char * path, const char * name) {
    gguf_init_params ip = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(path, ip);
    if (!g) {
        return 0;
    }
    const int64_t id = gguf_find_tensor(g, name);
    const size_t n = id < 0 ? 0 : gguf_get_tensor_size(g, id);
    gguf_free(g);
    return n;
}

static llama_model_params mparams_for(int nd, int ngl, bool no_alloc) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    mp.nextn_device = nd;
    mp.load_mtp     = true;
    mp.no_alloc     = no_alloc;
    if (no_alloc) {
        mp.load_mode = LLAMA_LOAD_MODE_NONE; // as common/fit.cpp's dry run
    }
    return mp;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <rpc endpoints> <rpc server pids> [ngl]\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const auto   endpoints  = split(argv[2]);
    const auto   pids       = split(argv[3]);
    const int    ngl        = argc > 4 ? atoi(argv[4]) : 999;

    ggml_backend_load_all();
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        fprintf(stderr, "no RPC backend in this build\n");
        return 1;
    }
    typedef ggml_backend_reg_t (*add_server_t)(const char *);
    auto add_server = (add_server_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (!add_server) {
        fprintf(stderr, "no ggml_backend_rpc_add_server\n");
        return 1;
    }
    for (const auto & ep : endpoints) {
        ggml_backend_register(add_server(ep.c_str()));
    }
    llama_backend_init();

    const size_t head = gguf_tensor_bytes(model_path, "output.weight");
    const size_t tied = head ? 0 : gguf_tensor_bytes(model_path, "token_embd.weight");
    const size_t norm = gguf_tensor_bytes(model_path, "output_norm.weight");
    printf("head\t%s\t%zu\toutput_norm.weight\t%zu\tsum\t%zu\n", head ? "output.weight" : "token_embd.weight (tied)",
            head ? head : tied, norm, (head ? head : tied) + norm);

    // the memory fit's view
    for (int nd : { -1, 0 }) {
        llama_model * model = llama_model_load_from_file(model_path, mparams_for(nd, ngl, true));
        if (!model) {
            printf("fit\tnd=%d\tload failed\n", nd);
            return 1;
        }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx   = 512;
        cp.n_batch = 512;
        cp.n_ubatch = 512;
        cp.n_seq_max = 1;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            printf("fit\tnd=%d\tcontext failed\n", nd);
            llama_model_free(model);
            return 1;
        }
        std::map<std::string, size_t> by_buft;
        size_t total = 0;
        for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx)) {
            by_buft[ggml_backend_buft_name(buft)] += mb.model;
            total += mb.model;
        }
        for (const auto & [name, n] : by_buft) {
            printf("fit\tnd=%d\t%s\t%zu\n", nd, name.c_str(), n);
        }
        printf("fit\tnd=%d\ttotal\t%zu\n", nd, total);
        llama_free(ctx);
        llama_model_free(model);
    }

    // two real loads and frees in one process, per setting
    printf("load\tnd\tcycle\tlargest_host_alloc\tallocs_ge_64MiB\tservers_rss_before_kib\tservers_rss_loaded_kib\tservers_rss_freed_kib\tself_rss_anon_kib\n");
    for (int nd : { -1, 0 }) {
        for (int cycle = 1; cycle <= 2; ++cycle) {
            const long before = servers_rss_kib(pids);
            g_max = 0;
            g_n_big = 0;
            g_track = true;
            llama_model * model = llama_model_load_from_file(model_path, mparams_for(nd, ngl, false));
            g_track = false;
            if (!model) {
                printf("load\t%d\t%d\tload failed\n", nd, cycle);
                return 1;
            }
            const size_t largest = g_max;
            const size_t n_big   = g_n_big;
            const long loaded = servers_rss_kib(pids);
            const long anon   = kib_of("/proc/self/status", "RssAnon:");
            llama_model_free(model);
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            const long freed = servers_rss_kib(pids);
            printf("load\t%d\t%d\t%zu\t%zu\t%ld\t%ld\t%ld\t%ld\n", nd, cycle, largest, n_big, before, loaded, freed, anon);
            fflush(stdout);
        }
    }

    llama_backend_free();
    return 0;
}
