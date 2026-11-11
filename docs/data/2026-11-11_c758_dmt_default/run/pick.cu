// pick.cu: C758's check of the host-side choice in ggml/src/ggml-cuda/mmvq-pascal-dmt.cu, without a GPU.
//
// Built against one tree's own mmvq-pascal-dmt.cu, included whole so that the functions of its anonymous namespace can
// be called, it prints for every product the read-once kernel could take (IQ4_XS and Q4_K, 2 to 4 tokens, rows of 1 to
// 64 super-blocks, K = 256 to 16384) what ggml_cuda_mmvq_pascal_dmt launches under the GGML_CUDA_MMVQ_PASCAL_DMT*
// switches in the environment: R (dmt_pick_r), the register prefetch (the launcher's own expression, PICK_PF below),
// the warps per block (dmt_nw), the lanes per row, the shared memory and the warp staging's stride, and the kernel
// instantiation mul_mat_vec_q_pascal_dmt<type, NT, R, PF>. A product is "declined" where the tree's static checks
// leave it to the k-column kernel (an odd number of 16-byte units in a lane group's segment, or not one warp in 48 KB).
// What it does not print depends on the device or on the tensor (alignment of the data, the grid from the SM count),
// and that part of the launcher is the same text in both trees.
//
// Build (in a CUDA 12.9 devel image; no GPU needed, nothing is launched): see checks.sh, step "pick". -DC758_BASE for a
// tree before C758, whose launcher's line is `const bool pf = dmt_mode() != 2;` (checks.sh greps both lines).
// Run: env [GGML_CUDA_MMVQ_PASCAL_DMT=...] [..._DMT_R=...] [..._DMT_PF=...] [..._DMT_NW=...] ./pick > arm.tsv

#include "mmvq-pascal-dmt.cu"

#include <cstdio>

#ifdef C758_BASE
#define PICK_PF(type, nt, nsb, r) (dmt_mode() != 2)
#else
#define PICK_PF(type, nt, nsb, r) dmt_pick_pf(type, nt, nsb, r)
#endif

int main() {
    printf("type\tNT\tK\tnsb\tR\tPF\twarps\tL\tsmem\tgstride\tkernel\n");
    const ggml_type types[2] = { GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K };
    for (const ggml_type type : types) {
        const char * tn = type == GGML_TYPE_IQ4_XS ? "iq4_xs" : "q4_K";
        for (int nt = 2; nt <= DMT_MAX_COLS; ++nt) {
            for (int nsb = 1; nsb <= 64; ++nsb) {
                const int K   = nsb * QK_K;
                const int lpr = dmt_lpr_log2(nsb);
                if ((dmt_sb(type) << lpr) % 16 != 0) {
                    printf("%s\t%d\t%d\t%d\tdeclined (segment not whole 16-byte units)\n", tn, nt, K, nsb);
                    continue;
                }
                const int r  = dmt_pick_r(type, nt, nsb);
                const int nw = dmt_nw(type, nt, nsb, r);
                if (nw <= 0) {
                    printf("%s\t%d\t%d\t%d\tdeclined (48 KB)\n", tn, nt, K, nsb);
                    continue;
                }
                const bool pf   = PICK_PF(type, nt, nsb, r);
                const int  smem = dmt_act_bytes(type, nt, nsb) + nw * dmt_wstage(type, r, lpr);
                printf("%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\tmul_mat_vec_q_pascal_dmt<%s, %d, %d, %s>\n", tn, nt, K, nsb,
                       r, pf ? 1 : 0, nw, 1 << lpr, smem, dmt_gstride(dmt_sb(type), r, lpr), tn, nt, r,
                       pf ? "true" : "false");
            }
        }
    }
    return 0;
}
