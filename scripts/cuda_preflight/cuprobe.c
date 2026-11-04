/* cuprobe: the CUDA probe of scripts/cuda_preflight.sh.
 *
 * It uses only the driver API, loaded with dlopen, so it needs no CUDA headers or libraries: any image with a C
 * compiler builds it (gcc -O1 -o cuprobe cuprobe.c -ldl), and it runs in any container whose GPU request mounts
 * libcuda.so.1. It does what a job's first CUDA call would do: cuInit, then for every device CUDA lists a primary
 * context, an allocation of PROBE_MIB MiB (default 16), a pattern copied to the card and back, and a synchronize.
 *
 * Output: one "cuda <i> bus <id> (<name>): ok|FAIL ..." line per device, then one verdict line, always the last:
 *   PREFLIGHT OK: <n> devices ...
 *   PREFLIGHT FAIL: <what failed, on which card, the likely cause>
 * Exit status: 0 all devices work, 1 anything else.
 */
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef unsigned long long dptr;
static int (*cuInit)(unsigned);
static int (*cuDeviceGetCount)(int *);
static int (*cuDeviceGet)(int *, int);
static int (*cuDeviceGetName)(char *, int, int);
static int (*cuDeviceGetPCIBusId)(char *, int, int);
static int (*cuDevicePrimaryCtxRetain)(void **, int);
static int (*cuDevicePrimaryCtxRelease)(int);
static int (*cuCtxSetCurrent)(void *);
static int (*cuCtxSynchronize)(void);
static int (*cuMemAlloc)(dptr *, size_t);
static int (*cuMemFree)(dptr);
static int (*cuMemcpyHtoD)(dptr, const void *, size_t);
static int (*cuMemcpyDtoH)(void *, dptr, size_t);
static int (*cuMemGetInfo)(size_t *, size_t *);
static int (*cuGetErrorName)(int, const char **);

static void *lib;
/* the _v2 name is the one the driver's headers map to; the plain name is the older entry point */
static void *sym(const char *name) {
    char v2[96]; void *p;
    snprintf(v2, sizeof v2, "%s_v2", name);
    p = dlsym(lib, v2);
    return p ? p : dlsym(lib, name);
}
#define LOAD(f) (*(void **)&(f) = sym(#f))

static char ebuf[128];
static const char *ename(int r) {
    const char *s = NULL;
    if (r == -1) return "a data mismatch";
    if (cuGetErrorName && cuGetErrorName(r, &s) == 0 && s) snprintf(ebuf, sizeof ebuf, "%s (%d)", s, r);
    else snprintf(ebuf, sizeof ebuf, "error %d", r);
    return ebuf;
}

static const char *cause(int r) {
    switch (r) {
    case -1: return "the card returned different data than was written: a failing card, riser or link; see the kernel log";
    case 2: return "the card's memory is full: another process holds it (not a driver fault)";
    case 3: case 999: return "the driver is in a bad state, usually after a card fell off the bus (Xid 79) or UVM hit a fatal error; the kernel log says which, and a reboot usually clears it";
    case 35: case 803: case 804: return "the driver and the container's CUDA libraries do not match";
    case 100: return "CUDA sees no GPU in this container: the GPU request did not reach it";
    case 101: return "the device is not valid for this process (the GPU request or CUDA_VISIBLE_DEVICES hides it)";
    case 214: case 700: case 702: case 710: case 715: case 719: return "the card stopped answering (Xid lines in the kernel log say why)";
    default: return "see the kernel log for Xid and NVRM lines";
    }
}

int main(void) {
    char verdict[2048], what[1024] = "";
    const char *step = "";
    int n = 0, bad = 0, r, firstbad = 0;
    size_t mib = getenv("PROBE_MIB") ? (size_t)atoi(getenv("PROBE_MIB")) : 16, bytes, k;
    if (mib < 1) mib = 1;
    bytes = mib << 20;
    setvbuf(stdout, NULL, _IOLBF, 0);

    lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) {
        printf("PREFLIGHT FAIL: libcuda.so.1 cannot be loaded in this container (%s); likely cause: the GPU request mounted no driver (use --gpus, --runtime=nvidia or a CDI device)\n", dlerror());
        return 1;
    }
    LOAD(cuInit); LOAD(cuDeviceGetCount); LOAD(cuDeviceGet); LOAD(cuDeviceGetName); LOAD(cuDeviceGetPCIBusId);
    LOAD(cuDevicePrimaryCtxRetain); LOAD(cuDevicePrimaryCtxRelease); LOAD(cuCtxSetCurrent); LOAD(cuCtxSynchronize);
    LOAD(cuMemAlloc); LOAD(cuMemFree); LOAD(cuMemcpyHtoD); LOAD(cuMemcpyDtoH); LOAD(cuMemGetInfo); LOAD(cuGetErrorName);
    if (!cuInit || !cuDeviceGetCount || !cuDeviceGet || !cuDevicePrimaryCtxRetain || !cuCtxSetCurrent || !cuMemAlloc ||
        !cuMemFree || !cuMemcpyHtoD || !cuMemcpyDtoH || !cuCtxSynchronize || !cuDevicePrimaryCtxRelease) {
        printf("PREFLIGHT FAIL: libcuda.so.1 lacks a driver API entry point this probe needs; likely cause: the driver is far older than the probe expects\n");
        return 1;
    }

    r = cuInit(0);
    if (r) {
        static const char *nodes[] = { "/dev/nvidiactl", "/dev/nvidia-uvm" };
        char dev[256] = "";
        int i, e = 0;
        for (i = 0; i < 2 && !dev[0]; i++) {
            int fd = open(nodes[i], O_RDWR);
            if (fd < 0) { e = errno; snprintf(dev, sizeof dev, "; open(%s) fails: %s", nodes[i], strerror(e)); }
            else close(fd);
        }
        snprintf(verdict, sizeof verdict, "PREFLIGHT FAIL: cuInit returned %s%s; likely cause: %s", ename(r), dev,
                 e == EIO ? "nvidia-uvm is in a fatal state (a GPU fell off the bus earlier); only a host reboot clears it"
                 : e == ENOENT ? "the container has no NVIDIA device nodes: the GPU request did not reach it"
                 : cause(r));
        puts(verdict);
        return 1;
    }
    r = cuDeviceGetCount(&n);
    if (r) { printf("PREFLIGHT FAIL: cuDeviceGetCount returned %s; likely cause: %s\n", ename(r), cause(r)); return 1; }
    if (n == 0) { printf("PREFLIGHT FAIL: cuInit worked but CUDA lists no device; likely cause: %s\n", cause(100)); return 1; }

    unsigned char *host = malloc(bytes), *back = malloc(bytes);
    if (!host || !back) { printf("PREFLIGHT FAIL: the probe cannot allocate %zu MiB of host memory\n", mib * 2); return 1; }
    for (int i = 0; i < n; i++) {
        char bus[64] = "?", name[128] = "?";
        int d = 0;
        void *ctx = NULL;
        dptr p = 0;
        size_t fr = 0, tot = 0;
        r = 0; step = "cuDeviceGet";
#define STEP(label, call) do { if (!r) { step = label; r = (call); } } while (0)
        STEP("cuDeviceGet", cuDeviceGet(&d, i));
        if (!r) {
            if (cuDeviceGetPCIBusId) cuDeviceGetPCIBusId(bus, sizeof bus, d);
            if (cuDeviceGetName) cuDeviceGetName(name, sizeof name, d);
        }
        for (k = 0; k < bytes; k++) host[k] = (unsigned char)(k * 131 + i + 7);
        memset(back, 0, bytes);
        STEP("cuDevicePrimaryCtxRetain", cuDevicePrimaryCtxRetain(&ctx, d));
        STEP("cuCtxSetCurrent", cuCtxSetCurrent(ctx));
        if (!r && cuMemGetInfo) cuMemGetInfo(&fr, &tot);
        STEP("cuMemAlloc", cuMemAlloc(&p, bytes));
        STEP("cuMemcpyHtoD", cuMemcpyHtoD(p, host, bytes));
        STEP("cuMemcpyDtoH", cuMemcpyDtoH(back, p, bytes));
        STEP("cuCtxSynchronize", cuCtxSynchronize());
        if (!r && memcmp(host, back, bytes)) { step = "the copy check"; r = -1; }
        if (p) cuMemFree(p);
        if (ctx) cuDevicePrimaryCtxRelease(d);
        if (r) {
            char one[400];
            printf("cuda %d bus %s (%s): FAIL %s returned %s\n", i, bus, name, step, ename(r));
            snprintf(one, sizeof one, "%scard %d at bus %s (%s): %s %s %s", bad ? "; " : "", i, bus, name, step,
                     r == -1 ? "found" : "returned", ename(r));
            if (strlen(what) + strlen(one) < sizeof what) strcat(what, one);
            if (!bad) firstbad = r;
            bad++;
        } else {
            printf("cuda %d bus %s (%s): ok, %zu MiB allocated and copied both ways, %zu of %zu MiB free\n", i, bus, name,
                   mib, fr >> 20, tot >> 20);
        }
    }
    if (bad) {
        printf("PREFLIGHT FAIL: %s; likely cause: %s\n", what, cause(firstbad));
        return 1;
    }
    printf("PREFLIGHT OK: %d device%s, each with a context, a %zu MiB allocation and a copy both ways\n", n, n == 1 ? "" : "s", mib);
    return 0;
}
