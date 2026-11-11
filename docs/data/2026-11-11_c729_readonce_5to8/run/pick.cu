// pick.cu: C729's check of the host-side choices of the read-once kernels, without a GPU (from C758's pick.cu).
//
// Built against one tree's own mmvq-pascal-dmt.cu (IQ4_XS and Q4_K, C722) or, with -DPICK_Q5K, mmvq-pascal-q5k.cu (Q5_K,
// C690), included whole so that the functions of its anonymous namespace can be called, it prints for every product the
// kernel could take (1 to 8 tokens, rows of 1 to 64 super-blocks, K = 256 to 16384) what the launcher would launch under
// the GGML_CUDA_MMVQ_PASCAL_* switches in the environment: whether the switches send it there at all ("off": the
// token count's switch is 0; the types _SKIP leaves out are not modelled), R, the register prefetch, the warps per
// block, the lanes per row, the shared memory, the warp staging's stride and the kernel instantiation. "declined" is
// where the tree's static
// checks leave the product to the k-column kernel (an odd number of 16-byte units in a lane group's segment, or not
// one warp in 48 KB). What it does not print depends on the device or on the tensor (alignment of the data, the grid
// from the SM count), and that part of the launchers is the same text in both trees.
//
// -DC729_BASE for the base (pascal-p106-v4-b12246-day 18d0fa309), whose host functions take no token count where the
// tip's do and which has no GGML_CUDA_MMVQ_PASCAL_RO8: the macros below repeat each tree's own expressions (checks.sh
// greps both trees for them).
//
// Build (in a CUDA 12.9 devel image; no GPU needed, nothing is launched): checks.sh, step "pick".
// Run: env [GGML_CUDA_MMVQ_PASCAL_...=...] ./pick > arm.tsv

#ifdef PICK_Q5K
#include "mmvq-pascal-q5k.cu"
#else
#include "mmvq-pascal-dmt.cu"
#endif

#include <cstdio>

#ifdef PICK_Q5K
#ifdef C729_BASE
#define PICK_ON(nt)  ((nt) == 1 ? q5kmt_one_row(248320) : (nt) <= Q5KMT_MAX_COLS && q5kmt_mode() != 0)
#define PICK_R(nt)   q5kmt_r()
#define PICK_PF(nt)  q5kmt_pf()
#else
#define PICK_ON(nt)  ((nt) == 1 ? q5kmt_one_row(248320) : (nt) <= Q5KMT_RO8_COLS && ((nt) <= Q5KMT_MAX_COLS ? q5kmt_mode() : q5kmt_ro8()) != 0)
#define PICK_R(nt)   q5kmt_r(nt)
#define PICK_PF(nt)  q5kmt_pf(nt)
#endif
#else
#ifdef C729_BASE
#define PICK_ON(nt)  ((nt) <= DMT_MAX_COLS && dmt_mode() != 0)
#else
#define PICK_ON(nt)  ((nt) <= DMT_RO8_COLS && ((nt) <= DMT_MAX_COLS ? dmt_mode() : dmt_ro8()) != 0)
#endif
#endif

int main() {
    printf("type\tNT\tK\tnsb\tR\tPF\twarps\tL\tsmem\tgstride\tkernel\n");
#ifdef PICK_Q5K
    for (int nt = 1; nt <= 8; ++nt) {
        for (int nsb = 1; nsb <= 64; ++nsb) {
            const int K = nsb * QK_K;
            if (!PICK_ON(nt)) {
                printf("q5_K\t%d\t%d\t%d\toff\n", nt, K, nsb);
                continue;
            }
            const int nw = q5kmt_nw(nt, nsb);
            if (nw <= 0) {
                printf("q5_K\t%d\t%d\t%d\tdeclined (48 KB)\n", nt, K, nsb);
                continue;
            }
            const int  r    = PICK_R(nt);
            const bool pf   = PICK_PF(nt);
            const int  lpr  = q5kmt_lpr_log2(nsb);
            const int  smem = q5kmt_act_bytes(nt, nsb) + nw * q5kmt_wstage(r, lpr);
            printf("q5_K\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\tmul_mat_vec_q5_K_pascal_mt<%d, %d, %s>\n", nt, K, nsb, r,
                   pf ? 1 : 0, nw, 1 << lpr, smem, q5kmt_gstride(r, lpr), nt, r, pf ? "true" : "false");
        }
    }
#else
    const ggml_type types[2] = { GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K };
    for (const ggml_type type : types) {
        const char * tn = type == GGML_TYPE_IQ4_XS ? "iq4_xs" : "q4_K";
        for (int nt = 2; nt <= 8; ++nt) {
            for (int nsb = 1; nsb <= 64; ++nsb) {
                const int K   = nsb * QK_K;
                const int lpr = dmt_lpr_log2(nsb);
                if (!PICK_ON(nt)) {
                    printf("%s\t%d\t%d\t%d\toff\n", tn, nt, K, nsb);
                    continue;
                }
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
                const bool pf   = dmt_pick_pf(type, nt, nsb, r);
                const int  smem = dmt_act_bytes(type, nt, nsb) + nw * dmt_wstage(type, r, lpr);
                printf("%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\tmul_mat_vec_q_pascal_dmt<%s, %d, %d, %s>\n", tn, nt, K, nsb,
                       r, pf ? 1 : 0, nw, 1 << lpr, smem, dmt_gstride(dmt_sb(type), r, lpr), tn, nt, r,
                       pf ? "true" : "false");
            }
        }
    }
#endif
    return 0;
}
