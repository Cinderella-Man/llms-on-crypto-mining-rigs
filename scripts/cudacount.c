// cudacount: counts a process's CUDA runtime calls (synchronizations, copies, kernel and graph launches, event records
// and waits) through LD_PRELOAD and prints the totals to stderr at exit. Per-token counts come from two runs that differ
// only in the number of generated tokens. Written for Q150 (slimming the per-token host path, 2026-10-24).
// It sees only calls into a dynamically loaded libcudart.so.12 (a statically linked cudart bypasses LD_PRELOAD), and a
// process that aborts never reaches the report. Its first use (V100s through the compute_61 PTX) reported only zeros
// next to runs that aborted on a CUDA error, so it has no validated counts yet.
//   gcc -O2 -shared -fPIC -o cudacount.so cudacount.c -ldl
//   LD_PRELOAD=$PWD/cudacount.so llama-completion -m <gguf> -p hi -n 32 ... 2>&1 | grep CUDACOUNT
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>

typedef int cudaError_t;
typedef void * cudaStream_t;
typedef void * cudaEvent_t;
typedef void * cudaGraphExec_t;
typedef struct { unsigned x, y, z; } dim3_t;

static unsigned long n_stream_sync, n_event_sync, n_device_sync, n_memcpy, n_memcpy_async, n_memcpy_peer_async,
                     n_launch, n_graph_launch, n_event_record, n_stream_wait;

#define CNT(v) __atomic_add_fetch(&(v), 1, __ATOMIC_RELAXED)
#define REAL(fn) static __typeof__(&fn) real = NULL; if (!real) real = (__typeof__(&fn)) dlsym(RTLD_NEXT, #fn)

cudaError_t cudaStreamSynchronize(cudaStream_t s) {
    REAL(cudaStreamSynchronize); CNT(n_stream_sync); return real(s);
}
cudaError_t cudaEventSynchronize(cudaEvent_t e) {
    REAL(cudaEventSynchronize); CNT(n_event_sync); return real(e);
}
cudaError_t cudaDeviceSynchronize(void) {
    REAL(cudaDeviceSynchronize); CNT(n_device_sync); return real();
}
cudaError_t cudaMemcpy(void * dst, const void * src, size_t n, int kind) {
    REAL(cudaMemcpy); CNT(n_memcpy); return real(dst, src, n, kind);
}
cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t n, int kind, cudaStream_t s) {
    REAL(cudaMemcpyAsync); CNT(n_memcpy_async);
    return real(dst, src, n, kind, s);
}
cudaError_t cudaMemcpyPeerAsync(void * dst, int dd, const void * src, int sd, size_t n, cudaStream_t s) {
    REAL(cudaMemcpyPeerAsync);
    CNT(n_memcpy_peer_async); return real(dst, dd, src, sd, n, s);
}
cudaError_t cudaLaunchKernel(const void * f, dim3_t g, dim3_t b, void ** args, size_t shm, cudaStream_t s) {
    REAL(cudaLaunchKernel);
    CNT(n_launch); return real(f, g, b, args, shm, s);
}
cudaError_t cudaLaunchKernelExC(const void * cfg, const void * f, void ** args) {
    REAL(cudaLaunchKernelExC); CNT(n_launch); return real(cfg, f, args);
}
cudaError_t cudaGraphLaunch(cudaGraphExec_t g, cudaStream_t s) {
    REAL(cudaGraphLaunch); CNT(n_graph_launch); return real(g, s);
}
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s) {
    REAL(cudaEventRecord); CNT(n_event_record); return real(e, s);
}
cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned flags) {
    REAL(cudaStreamWaitEvent); CNT(n_stream_wait);
    return real(s, e, flags);
}

__attribute__((destructor)) static void cudacount_report(void) {
    fprintf(stderr, "CUDACOUNT stream_sync=%lu event_sync=%lu device_sync=%lu memcpy=%lu memcpy_async=%lu "
            "memcpy_peer_async=%lu launch=%lu graph_launch=%lu event_record=%lu stream_wait=%lu\n",
            n_stream_sync, n_event_sync, n_device_sync, n_memcpy, n_memcpy_async, n_memcpy_peer_async, n_launch,
            n_graph_launch, n_event_record, n_stream_wait);
}
