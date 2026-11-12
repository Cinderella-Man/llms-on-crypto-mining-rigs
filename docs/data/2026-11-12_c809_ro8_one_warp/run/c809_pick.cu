// c809_pick.cu: C809's check of the read-once kernels' host-side choices, without a GPU. From the v4 kernel merge's
// kern_pick.cu (docs/data/2026-11-11_v4_kernel_merge/run/kern_pick.cu) for IQ4_XS, Q4_K and Q6_K, and C729's pick.cu
// (docs/data/2026-11-11_c729_readonce_5to8/run/pick.cu) for Q5_K.
//
// Built against one tree's own mmvq-pascal-dmt.cu (IQ4_XS, Q4_K, Q6_K) or, with -DPICK_Q5K, mmvq-pascal-q5k.cu (Q5_K),
// included whole so that the functions of its anonymous namespace can be called, it prints for every product the kernel
// could take (IQ4_XS, Q4_K and Q6_K at 2 to 8 tokens, Q5_K at 1 to 8; rows of 1 to 64 super-blocks, K = 256 to 16384)
// what the launcher would launch under the GGML_CUDA_MMVQ_PASCAL_* switches in the environment: "off" where the token
// count's and type's switch sends it nowhere, else R, the register prefetch, the warps per block, the lanes per row, the
// shared memory, the warp staging's stride and the kernel (the whole-staging kernel, or C728's K tiles with their warps).
// "declined" is where the tree's static checks leave the product to the k-column kernel: a lane group's segment not whole
// 16-byte units, not one warp in 48 KB, or a block of one warp where the tree declines it (Q6_K's rule since C727; C809:
// the products of 5 to 8 tokens). What it does not print depends on the device or the tensor (the data's alignment, the
// grid from the SM count); that part of the launchers is the same text in both trees. _DMT_SKIP and _SKIP are not
// modelled; Q5_K's one-token products (_Q5K_1R) are printed for 248,320 rows (the output layer).
//
// One tree a build: -DTREE_BASE (pascal-p106-v4 d21820340, whose two files are the v4 kernel merge's d097d6d88) or
// -DTREE_C809 (pascal-p106-v4-b12246-c809). The macros repeat each tree's own expressions: PICK_ON the switch tests of
// the _takes functions, PICK_ONE the one-warp rule, PICK_KT the launcher's call of the tiles (checks.sh greps each tree for
// them).
//
// Build (in a CUDA 12.9 devel image; no GPU needed, nothing is launched): checks.sh, step "pick".
// Run: env [GGML_CUDA_MMVQ_PASCAL_...=...] ./pick > arm.tsv

#ifdef PICK_Q5K
#include "mmvq-pascal-q5k.cu"
#else
#include "mmvq-pascal-dmt.cu"
#endif

#include <cstdio>

#if !defined(TREE_BASE) && !defined(TREE_C809)
#error "one of TREE_BASE, TREE_C809"
#endif

#ifdef PICK_Q5K
#define PICK_ON(nt) ((nt) == 1 ? q5kmt_one_row(248320) : ((nt) <= Q5KMT_MAX_COLS ? q5kmt_mode() : q5kmt_ro8()) != 0)
#ifdef TREE_C809
#define PICK_ONE(nt, nw) ((nt) > Q5KMT_MAX_COLS && q5kmt_nw_env() != 1 && (nw) < 2)
#else
#define PICK_ONE(nt, nw) false
#endif
#else
#define PICK_ON(type, nt) ((nt) <= DMT_RO8_COLS && dmt_nt_mode(type, nt) != 0)
#ifdef TREE_C809
#define PICK_ONE(type, nt, nw) (((type) == GGML_TYPE_Q6_K || (nt) > DMT_MAX_COLS) && dmt_nw_env() != 1 && (nw) < 2)
#else
#define PICK_ONE(type, nt, nw) ((type) == GGML_TYPE_Q6_K && dmt_nw_env() != 1 && (nw) < 2)
#endif
#define PICK_KT(type, nt, nsb, r, nw) ((r) == 1 ? dmt_kt_nw(type, nt, nsb, nw) : 0)
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
            if (PICK_ONE(nt, nw)) {
                printf("q5_K\t%d\t%d\t%d\tdeclined (one warp)\n", nt, K, nsb);
                continue;
            }
            const int  r    = q5kmt_r(nt);
            const bool pf   = q5kmt_pf(nt);
            const int  lpr  = q5kmt_lpr_log2(nsb);
            const int  smem = q5kmt_act_bytes(nt, nsb) + nw * q5kmt_wstage(r, lpr);
            printf("q5_K\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\tmul_mat_vec_q5_K_pascal_mt<%d, %d, %s>\n", nt, K, nsb, r,
                   pf ? 1 : 0, nw, 1 << lpr, smem, q5kmt_gstride(r, lpr), nt, r, pf ? "true" : "false");
        }
    }
#else
    const ggml_type types[3] = { GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K };
    for (const ggml_type type : types) {
        const char * tn = type == GGML_TYPE_IQ4_XS ? "iq4_xs" : type == GGML_TYPE_Q4_K ? "q4_K" : "q6_K";
        const int    sb = dmt_sb(type);
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
                if (PICK_ONE(type, nt, nw)) {
                    printf("%s\t%d\t%d\t%d\tdeclined (one warp)\n", tn, nt, K, nsb);
                    continue;
                }
                const int ktnw = PICK_KT(type, nt, nsb, r, nw);
                if (ktnw > 0) {
                    const bool pf   = dmt_pick_pf(type, nt, nsb, 1);
                    const int  m    = dmt_kt_m();
                    const int  smem = dmt_kt_tile_bytes(type, nt, lpr) + nt * (288 << lpr) + ktnw * dmt_wstage(type, 1, lpr);
                    printf("%s\t%d\t%d\t%d\t1\t%d\t%d\t%d\t%d\t%d\tmul_mat_vec_q_pascal_dmt_kt<%s, %d, %d, %s>\n", tn, nt, K,
                           nsb, pf ? 1 : 0, ktnw, 1 << lpr, smem, dmt_gstride(sb, 1, lpr), tn, nt, m,
                           pf ? "true" : "false");
                    continue;
                }
                const bool pf   = dmt_pick_pf(type, nt, nsb, r);
                const int  smem = dmt_act_bytes(type, nt, nsb) + nw * dmt_wstage(type, r, lpr) + dmt_tail(type);
                printf("%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\tmul_mat_vec_q_pascal_dmt<%s, %d, %d, %s>\n", tn, nt, K, nsb,
                       r, pf ? 1 : 0, nw, 1 << lpr, smem, dmt_gstride(sb, r, lpr), tn, nt, r, pf ? "true" : "false");
            }
        }
    }
#endif
    return 0;
}
