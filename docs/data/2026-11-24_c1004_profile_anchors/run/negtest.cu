// C1004: does cudaEventElapsedTime give a negative time when its second event came first, and does it work across
// two streams of one device? Prints each result and its error code.
#include <cstdio>
#include <cuda_runtime.h>
__global__ void spin(long long n) { long long t0 = clock64(); while (clock64() - t0 < n) {} }
int main() {
    cudaStream_t a, b;
    cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
    cudaEvent_t e1, e2, e3;
    cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
    cudaEventRecord(e1, a);
    spin<<<1, 1, 0, a>>>(200000000LL);   // ~0.1 s
    cudaEventRecord(e2, a);
    cudaEventRecord(e3, b);              // on the idle stream: reached at once, before e2
    cudaDeviceSynchronize();
    float ms = 0.0f; cudaError_t err;
    err = cudaEventElapsedTime(&ms, e1, e2); printf("e1->e2 same stream:   %10.4f ms  %s\n", ms, cudaGetErrorString(err));
    err = cudaEventElapsedTime(&ms, e2, e1); printf("e2->e1 reversed:      %10.4f ms  %s\n", ms, cudaGetErrorString(err));
    err = cudaEventElapsedTime(&ms, e3, e2); printf("e3(b)->e2(a) streams: %10.4f ms  %s\n", ms, cudaGetErrorString(err));
    err = cudaEventElapsedTime(&ms, e2, e3); printf("e2(a)->e3(b) reversed:%10.4f ms  %s\n", ms, cudaGetErrorString(err));
    err = cudaEventElapsedTime(&ms, e1, e3); printf("e1(a)->e3(b):         %10.4f ms  %s\n", ms, cudaGetErrorString(err));
    printf("last error: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
