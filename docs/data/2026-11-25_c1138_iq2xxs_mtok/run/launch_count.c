// launch_count.c: counts the driver-API kernel launches (cuLaunchKernel) of a process by kernel name and prints them
// to stderr at exit, so that a check can show which hand-written PTX kernels ran (C1138: the PTX kernels are launched
// with cuLaunchKernel, the CUDA C kernels through the runtime, which this does not see). Use:
//   gcc -O2 -shared -fPIC -I/usr/local/cuda/include launch_count.c -o launch_count.so -ldl
//   LD_PRELOAD=$PWD/launch_count.so <program>      (stderr: "launch_count: <n> <kernel name>" per kernel)
#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXK 64
static struct { CUfunction f; unsigned long n; } g_k[MAXK];
static int g_nk;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static CUresult (*g_launch)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream,
                            void **, void **);

static void report(void) {
    CUresult (*getname)(const char **, CUfunction) = (CUresult (*)(const char **, CUfunction)) dlsym(RTLD_DEFAULT, "cuFuncGetName");
    for (int i = 0; i < g_nk; ++i) {
        const char * name = "?";
        if (getname == NULL || getname(&name, g_k[i].f) != CUDA_SUCCESS) {
            name = "?";
        }
        fprintf(stderr, "launch_count: %lu %s\n", g_k[i].n, name);
    }
    if (g_nk == 0) {
        fprintf(stderr, "launch_count: no cuLaunchKernel calls\n");
    }
}

CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                        unsigned smem, CUstream s, void ** params, void ** extra) {
    pthread_mutex_lock(&g_mu);
    if (g_launch == NULL) {
        g_launch = dlsym(RTLD_NEXT, "cuLaunchKernel");
        atexit(report);
    }
    int i = 0;
    while (i < g_nk && g_k[i].f != f) {
        ++i;
    }
    if (i == g_nk && g_nk < MAXK) {
        g_k[g_nk].f = f;
        g_k[g_nk].n = 0;
        ++g_nk;
    }
    if (i < MAXK) {
        g_k[i].n++;
    }
    pthread_mutex_unlock(&g_mu);
    return g_launch(f, gx, gy, gz, bx, by, bz, smem, s, params, extra);
}
