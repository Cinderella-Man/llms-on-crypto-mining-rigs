/* mock_libcuda: a stand-in for libcuda.so.1 that makes cuprobe testable without a GPU (scripts/test/cuda_preflight_test.sh).
 * Build: gcc -shared -fPIC -O1 -o libcuda.so.1 mock_libcuda.c
 * Environment:
 *   MOCK_DEVICES=<n>               how many devices it lists (default 2)
 *   MOCK_FAIL=<call>[:<dev>]=<rc>  that driver call returns <rc> (on that device only, when given), e.g. cuInit=999 or cuMemAlloc:1=2
 *   MOCK_CORRUPT=<dev>             that device returns different data than was written
 *   MOCK_HANG=1                    cuInit never returns
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef unsigned long long dptr;
static int cur;

static int failure(const char *call) {
    const char *f = getenv("MOCK_FAIL");
    char want[64];
    int rc;
    if (!f) return 0;
    snprintf(want, sizeof want, "%s=", call);
    if (!strncmp(f, want, strlen(want))) return atoi(f + strlen(want));
    snprintf(want, sizeof want, "%s:%d=", call, cur);
    if (!strncmp(f, want, strlen(want))) { rc = atoi(f + strlen(want)); return rc; }
    return 0;
}
static int ndev(void) { return getenv("MOCK_DEVICES") ? atoi(getenv("MOCK_DEVICES")) : 2; }

int cuInit(unsigned flags) { (void)flags; if (getenv("MOCK_HANG")) for (;;) sleep(1); return failure("cuInit"); }
int cuDeviceGetCount(int *n) { *n = ndev(); return failure("cuDeviceGetCount"); }
int cuDeviceGet(int *d, int i) { *d = i; return 0; }
int cuDeviceGetName(char *s, int len, int d) { snprintf(s, len, "Mock GPU %d", d); return 0; }
int cuDeviceGetPCIBusId(char *s, int len, int d) { snprintf(s, len, "0000:%02x:00.0", 4 + d); return 0; }
int cuDevicePrimaryCtxRetain(void **c, int d) { cur = d; *c = (void *)(long)(d + 1); return failure("cuDevicePrimaryCtxRetain"); }
int cuDevicePrimaryCtxRelease_v2(int d) { (void)d; return 0; }
int cuCtxSetCurrent(void *c) { cur = (int)(long)c - 1; return 0; }
int cuCtxSynchronize(void) { return failure("cuCtxSynchronize"); }
int cuMemGetInfo_v2(size_t *f, size_t *t) { *f = 20ull << 30; *t = 24ull << 30; return 0; }
int cuMemAlloc_v2(dptr *p, size_t n) { int r = failure("cuMemAlloc"); if (r) return r; *p = (dptr)malloc(n); return *p ? 0 : 2; }
int cuMemFree_v2(dptr p) { free((void *)p); return 0; }
int cuMemcpyHtoD_v2(dptr p, const void *h, size_t n) {
    int r = failure("cuMemcpyHtoD");
    if (r) return r;
    memcpy((void *)p, h, n);
    if (getenv("MOCK_CORRUPT") && atoi(getenv("MOCK_CORRUPT")) == cur) ((unsigned char *)p)[n / 2] ^= 0x55;
    return 0;
}
int cuMemcpyDtoH_v2(void *h, dptr p, size_t n) { int r = failure("cuMemcpyDtoH"); if (r) return r; memcpy(h, (void *)p, n); return 0; }
int cuGetErrorName(int r, const char **s) {
    *s = r == 999 ? "CUDA_ERROR_UNKNOWN" : r == 2 ? "CUDA_ERROR_OUT_OF_MEMORY" : r == 719 ? "CUDA_ERROR_LAUNCH_FAILED" : "CUDA_ERROR_MOCK";
    return 0;
}
