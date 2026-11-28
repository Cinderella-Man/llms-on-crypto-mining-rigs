// memops_check: do stream memory operations work on these cards, and do they signal across cards faster than events?
//
// A hop between two cards needs the receiving card's stream to wait until the sending card's bytes are in host memory.
// Today that is an event: the sender's stream records it, the receiver's stream waits on it, and the driver relays the
// signal. Stream memory operations put the signal in a word of mapped host memory instead: the sender's stream writes
// it (cuStreamWriteValue32), the receiver's stream waits for it (cuStreamWaitValue32), with no host thread in between.
// The CUDA reference says to query support per device and does not list which products have it. This program prints
// the three attributes per card and then times, between cards A and B, `reps` rounds of:
//   event   A: cuEventRecord; B: cuStreamWaitEvent, then a 4-byte cuMemsetD32Async into a mapped host word
//   memops  A: cuStreamWriteValue32 into a mapped host word; B: cuStreamWaitValue32 on it, then the same memset
// from the host's enqueue on A until the host sees B's memset land in host memory (a volatile poll, 2 s timeout).
// Card numbers are CUDA's: run with CUDA_DEVICE_ORDER=PCI_BUS_ID to match nvidia-smi.
// Build: g++ -O2 memops_check.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64/stubs -lcuda -o memops_check
// Usage: memops_check <A> <B> [reps, default 1000]. Exit 0; 4 if the stream memory operations are refused.
#include <cuda.h>
#include <time.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                    \
  do {                                                                           \
    CUresult r_ = (x);                                                           \
    if (r_ != CUDA_SUCCESS) {                                                    \
      const char* s_ = nullptr;                                                  \
      cuGetErrorString(r_, &s_);                                                 \
      fprintf(stderr, "%s:%d %s: %s (%d)\n", __FILE__, __LINE__, #x, s_, (int)r_); \
      return r_ == CUDA_ERROR_NOT_SUPPORTED ? 4 : 2;                             \
    }                                                                            \
  } while (0)

static double now_us() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <A> <B> [reps]\n", argv[0]); return 1; }
  const int a = atoi(argv[1]), b = atoi(argv[2]), reps = argc > 3 ? atoi(argv[3]) : 1000;
  CK(cuInit(0));
  int n = 0;
  CK(cuDeviceGetCount(&n));
  // 92, 93, 94: CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS(_V1), _CAN_USE_64_BIT_STREAM_MEM_OPS(_V1),
  // _CAN_USE_STREAM_WAIT_VALUE_NOR(_V1) in cuda.h; queried by number since 12.x renamed them
  for (int i = 0; i < n; i++) {
    CUdevice d;
    CK(cuDeviceGet(&d, i));
    char name[128];
    cuDeviceGetName(name, sizeof name, d);
    int bus = -1, v92 = -1, v93 = -1, v94 = -1, uva = -1, maph = -1;
    cuDeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, d);
    cuDeviceGetAttribute(&v92, (CUdevice_attribute)92, d);
    cuDeviceGetAttribute(&v93, (CUdevice_attribute)93, d);
    cuDeviceGetAttribute(&v94, (CUdevice_attribute)94, d);
    cuDeviceGetAttribute(&uva, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, d);
    cuDeviceGetAttribute(&maph, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, d);
    printf("device %d %s bus %02x: mem_ops %d, 64-bit %d, wait_nor %d, uva %d, map_host %d\n", i, name, bus, v92, v93,
           v94, uva, maph);
  }
  if (a < 0 || b < 0 || a >= n || b >= n || a == b) { fprintf(stderr, "bad A/B\n"); return 1; }
  CUdevice da, db;
  CK(cuDeviceGet(&da, a));
  CK(cuDeviceGet(&db, b));
  CUcontext ca, cb;
  CK(cuDevicePrimaryCtxRetain(&ca, da));
  CK(cuDevicePrimaryCtxRetain(&cb, db));
  CUstream sa, sb;
  CK(cuCtxSetCurrent(ca));
  CK(cuStreamCreate(&sa, CU_STREAM_NON_BLOCKING));
  CUevent ea;
  CK(cuEventCreate(&ea, CU_EVENT_DISABLE_TIMING));
  void* h = nullptr;
  CK(cuMemHostAlloc(&h, 4096, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
  volatile uint32_t* w = (volatile uint32_t*)h;
  w[0] = 0;   // the sender's word
  w[16] = 0;  // the receiver's word, another 64-byte line
  CUdeviceptr pa0, pb0, pb16;
  CK(cuMemHostGetDevicePointer(&pa0, h, 0));
  CK(cuCtxSetCurrent(cb));
  CK(cuStreamCreate(&sb, CU_STREAM_NON_BLOCKING));
  CK(cuMemHostGetDevicePointer(&pb0, h, 0));
  pb16 = pb0 + 64;

  for (int mode = 0; mode < 2; mode++) {
    std::vector<double> t;
    int timeouts = 0;
    for (int r = -5; r < reps; r++) {
      const uint32_t v = (uint32_t)(mode * 100000 + r + 10);
      double t0 = now_us();
      if (mode == 0) {
        CK(cuCtxSetCurrent(ca));
        CK(cuEventRecord(ea, sa));
        CK(cuCtxSetCurrent(cb));
        CK(cuStreamWaitEvent(sb, ea, 0));
        CK(cuMemsetD32Async(pb16, v, 1, sb));
      } else {
        CK(cuCtxSetCurrent(cb));
        CK(cuStreamWaitValue32(sb, pb0, v, CU_STREAM_WAIT_VALUE_GEQ));
        CK(cuMemsetD32Async(pb16, v, 1, sb));
        t0 = now_us();  // the round starts when A signals; B's wait was enqueued first
        CK(cuCtxSetCurrent(ca));
        CK(cuStreamWriteValue32(sa, pa0, v, CU_STREAM_WRITE_VALUE_DEFAULT));
      }
      while (w[16] != v) {
        if (now_us() - t0 > 2e6) { timeouts++; break; }
      }
      double t1 = now_us();
      if (r >= 0) t.push_back(t1 - t0);
    }
    std::sort(t.begin(), t.end());
    printf("%s: A=%d B=%d reps %d median %.2f us p10 %.2f p90 %.2f timeouts %d\n", mode == 0 ? "event" : "memops", a, b,
           reps, t[t.size() / 2], t[t.size() / 10], t[t.size() * 9 / 10], timeouts);
    fflush(stdout);
  }
  // tear down each context's objects with that context current, then release the contexts
  CK(cuCtxSetCurrent(cb));
  CK(cuStreamSynchronize(sb));
  CK(cuStreamDestroy(sb));
  CK(cuCtxSetCurrent(ca));
  CK(cuStreamSynchronize(sa));
  CK(cuStreamDestroy(sa));
  CK(cuEventDestroy(ea));
  CK(cuMemFreeHost(h));
  CK(cuCtxSetCurrent(nullptr));
  CK(cuDevicePrimaryCtxRelease(db));
  CK(cuDevicePrimaryCtxRelease(da));
  return 0;
}
