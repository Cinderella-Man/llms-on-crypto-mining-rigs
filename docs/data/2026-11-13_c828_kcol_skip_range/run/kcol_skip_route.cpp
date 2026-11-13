// C828's routing check, without a GPU. For each weight type and each product of 1 to 8 tokens (columns), it asks the
// build's own predicate, ggml_cuda_mmvq_pascal_kcol_skips (mmvq-pascal-kcol.cu, host code), whether
// GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP leaves the product to the other kernels. The k-column hook calls this predicate and
// nothing else for the switch. For Q4_0 and Q5_0 (Flash-Next's shared-expert downs) the last column names the kernel
// that then runs the product under the Flash-Next preset (GGML_CUDA_MMVQ_PASCAL_KCOL=1, _KCOL_TABLE=0):
//   1 token: the one-token Pascal kernel (in mode 1 the k-column hook takes no 1-token Q4_0 or Q5_0 product);
//   2-4 tokens: mul_mat_vec_q_pascal_mc when the hook declines (the Pascal kernels take 1 to 4 tokens), else k-column;
//   5-8 tokens: llama.cpp's mul_mat_vec_q when the hook declines (no Pascal kernel takes them), else k-column.
// Build (in the CUDA image, against a build tree's static libraries; see checks.sh, step route):
//   g++ -std=c++17 -I<src>/ggml/include kcol_skip_route.cpp -Wl,--start-group <build>/**/libggml*.a -Wl,--end-group \
//       -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcudart -lcublas -lcublasLt -lcuda -fopenmp -ldl -lpthread
// Run with the switch set or unset; the predicate reads it once per process. Output: TSV on stdout.
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

bool ggml_cuda_mmvq_pascal_kcol_skips(ggml_type type, int64_t ncols);

int main() {
    const char * env = getenv("GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP");
    const ggml_type types[] = {
        GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, GGML_TYPE_Q2_0, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS,
    };
    printf("switch\ttype\ttokens\tskip\tkernel\n");
    for (const ggml_type t : types) {
        for (int n = 1; n <= 8; ++n) {
            const bool s = ggml_cuda_mmvq_pascal_kcol_skips(t, n);
            const char * k = "-";
            if (t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q5_0) {
                k = n == 1 ? "one-token Pascal kernel" : !s ? "k-column" : n <= 4 ? "mul_mat_vec_q_pascal_mc" : "mul_mat_vec_q";
            }
            printf("%s\t%s\t%d\t%s\t%s\n", env ? env : "(unset)", ggml_type_name(t), n, s ? "yes" : "no", k);
        }
    }
    return 0;
}
