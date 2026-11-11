// kern_pick.cu: the v4 kernel merge's check of the read-once kernels' host-side choices (C796), without a GPU. From C729's
// pick.cu (docs/data/2026-11-11_c729_readonce_5to8/run/pick.cu), carried to Q6_K (C727) and the K tiles (C728).
//
// Built against one tree's own mmvq-pascal-dmt.cu, included whole so that the functions of its anonymous namespace can be
// called, it prints for every product the kernel could take (IQ4_XS, Q4_K and Q6_K; 2 to 8 tokens; rows of 1 to 64
// super-blocks, K = 256 to 16384) what ggml_cuda_mmvq_pascal_dmt would launch under the GGML_CUDA_MMVQ_PASCAL_* switches in
// the environment: "off" where the token count's and type's switch sends it nowhere (or the tree lacks the type), else
// R, the register prefetch, the warps per block, the lanes per row, the shared memory, the warp staging's stride and the
// kernel: the whole-staging kernel mul_mat_vec_q_pascal_dmt<type, NT, R, PF> or the K tiles mul_mat_vec_q_pascal_dmt_kt<
// type, NT, M, PF> with their warps. "declined" is where the tree's static checks leave the product to the k-column
// kernel (a lane group's segment not whole 16-byte units, not one warp in 48 KB, Q6_K's single warp). What it does not
// print depends on the device or the tensor (the data's alignment, which the tiles also test on the activations, and the
// grid from the SM count); that part of the launchers is the same text in every tree. GGML_CUDA_MMVQ_PASCAL_DMT_SKIP and
// _SKIP are not modelled.
//
// One tree a build: -DTREE_BASE (pascal-p106-v4 08634ce00, whose read-once code is the day merge's), -DTREE_C727,
// -DTREE_C728, -DTREE_C729 (the three branches' tips) or -DTREE_KERN (the merge). PICK_ON repeats each tree's own test of
// the switches in ggml_cuda_mmvq_pascal_dmt_takes, PICK_Q6K1 C727's one-warp rule, PICK_KT the launcher's call of the
// tiles (checks.sh greps each tree for the expressions).
//
// Build (in a CUDA 12.9 devel image; no GPU needed, nothing is launched): checks.sh, step "pick".
// Run: env [GGML_CUDA_MMVQ_PASCAL_...=...] ./pick > arm.tsv

#include "mmvq-pascal-dmt.cu"

#include <cstdio>

#if defined(TREE_BASE) || defined(TREE_C728)
#define PICK_ON(type, nt) ((type) != GGML_TYPE_Q6_K && (nt) <= DMT_MAX_COLS && dmt_mode() != 0)
#elif defined(TREE_C727)
#define PICK_ON(type, nt) ((nt) <= DMT_MAX_COLS && dmt_type_mode(type) != 0)
#elif defined(TREE_C729)
#define PICK_ON(type, nt) ((type) != GGML_TYPE_Q6_K && (nt) <= DMT_RO8_COLS && ((nt) <= DMT_MAX_COLS ? dmt_mode() : dmt_ro8()) != 0)
#elif defined(TREE_KERN)
#define PICK_ON(type, nt) ((nt) <= DMT_RO8_COLS && dmt_nt_mode(type, nt) != 0)
#else
#error "one of TREE_BASE, TREE_C727, TREE_C728, TREE_C729, TREE_KERN"
#endif

#if defined(TREE_C727) || defined(TREE_KERN)
#define PICK_Q6K1(type, nw) ((type) == GGML_TYPE_Q6_K && dmt_nw_env() != 1 && (nw) < 2)
#define PICK_TAIL(type)     dmt_tail(type)
#else
#define PICK_Q6K1(type, nw) false
#define PICK_TAIL(type)     0
#endif

#if defined(TREE_C728) || defined(TREE_KERN)
#define PICK_KT(type, nt, nsb, r, nw) ((r) == 1 ? dmt_kt_nw(type, nt, nsb, nw) : 0)
#else
#define PICK_KT(type, nt, nsb, r, nw) 0
#endif

int main() {
    printf("type\tNT\tK\tnsb\tR\tPF\twarps\tL\tsmem\tgstride\tkernel\n");
    const ggml_type types[3] = { GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K };
    for (const ggml_type type : types) {
        const char * tn = type == GGML_TYPE_IQ4_XS ? "iq4_xs" : type == GGML_TYPE_Q4_K ? "q4_K" : "q6_K";
        const int    sb = type == GGML_TYPE_IQ4_XS ? 136 : type == GGML_TYPE_Q4_K ? 144 : 210;
        for (int nt = 2; nt <= 8; ++nt) {
            for (int nsb = 1; nsb <= 64; ++nsb) {
                const int K   = nsb * QK_K;
                const int lpr = dmt_lpr_log2(nsb);
                if (!PICK_ON(type, nt)) {
                    printf("%s\t%d\t%d\t%d\toff\n", tn, nt, K, nsb);
                    continue;
                }
                if ((sb << lpr) % 16 != 0) {
                    printf("%s\t%d\t%d\t%d\tdeclined (segment not whole 16-byte units)\n", tn, nt, K, nsb);
                    continue;
                }
                const int r  = dmt_pick_r(type, nt, nsb);
                const int nw = dmt_nw(type, nt, nsb, r);
                if (nw <= 0) {
                    printf("%s\t%d\t%d\t%d\tdeclined (48 KB)\n", tn, nt, K, nsb);
                    continue;
                }
                if (PICK_Q6K1(type, nw)) {
                    printf("%s\t%d\t%d\t%d\tdeclined (one warp)\n", tn, nt, K, nsb);
                    continue;
                }
                const int ktnw = PICK_KT(type, nt, nsb, r, nw);
                if (ktnw > 0) {
#if defined(TREE_C728) || defined(TREE_KERN)
                    const bool pf   = dmt_pick_pf(type, nt, nsb, 1);
                    const int  m    = dmt_kt_m();
                    const int  smem = dmt_kt_tile_bytes(type, nt, lpr) + nt * (288 << lpr) + ktnw * dmt_wstage(type, 1, lpr);
                    printf("%s\t%d\t%d\t%d\t1\t%d\t%d\t%d\t%d\t%d\tmul_mat_vec_q_pascal_dmt_kt<%s, %d, %d, %s>\n", tn, nt, K,
                           nsb, pf ? 1 : 0, ktnw, 1 << lpr, smem, dmt_gstride(sb, 1, lpr), tn, nt, m,
                           pf ? "true" : "false");
#endif
                    continue;
                }
                const bool pf   = dmt_pick_pf(type, nt, nsb, r);
                const int  smem = dmt_act_bytes(type, nt, nsb) + nw * dmt_wstage(type, r, lpr) + PICK_TAIL(type);
                printf("%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\tmul_mat_vec_q_pascal_dmt<%s, %d, %d, %s>\n", tn, nt, K, nsb,
                       r, pf ? 1 : 0, nw, 1 << lpr, smem, dmt_gstride(sb, r, lpr), tn, nt, r, pf ? "true" : "false");
            }
        }
    }
    return 0;
}
