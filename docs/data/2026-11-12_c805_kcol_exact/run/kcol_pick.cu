// kcol_pick.cu: C805's check of the k-column kernel's host-side choices, without a GPU. Built against one tree's own
// mmvq-pascal-kcol.cu, included whole so that the functions of its anonymous namespace (the switches, the dispatch table,
// and in the C805 tree kcol_exact and kcol_one_token_sums) and the tree's kcol::pick_shape (and kcol::pick_exact) are the
// ones called, it prints for every plain product of the k-column kernel's types what ggml_cuda_mmvq_pascal_kcol would do
// under the GGML_CUDA_MMVQ_PASCAL_KCOL* switches in the environment, on a P106-090 (5 SMs, 96 KB of shared memory per
// SM):
//   one-token  a one-column product left to the one-token kernels (mode 1: every type but Q3_K, IQ4_NL and Q8_0 in rows
//              of chunks not a multiple of 4)
//   table      left to the other kernels by the dispatch table (Q824)
//   width      the row width is not a whole number of the type's lane units (pick_shape: false)
//   declined   no launch shape (one iteration of the columns' activations does not fit the budget)
//   fallback   the default rule's fallback shape, left to the other kernels by the table's rule
//   kcol       taken: the launches, the columns per launch, R, the iterations per activation tile, the shared memory
//              and the blocks of each launch
// The decision is DECIDE below: the lines of ggml_cuda_mmvq_pascal_kcol after the read-once kernels' takes tests, which
// it repeats (checks.sh greps each tree for them). Not modelled: the read-once kernels' takes (their switches off),
// GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP, the device's register limits (the blocks function: nullptr, i.e. registers taken to
// allow 3 blocks, which only the default shape rule asks), the data's alignment (256-byte aligned rows' start).
//
// One tree a build: -DTREE_BASE (pascal-p106-v4 d097d6d88) or -DTREE_C805.
// Build (in a CUDA 12.9 devel image; no GPU needed, nothing is launched): checks.sh, step "pick".
// Run: env [GGML_CUDA_MMVQ_PASCAL_KCOL...=...] ./pick > arm.tsv

#include "mmvq-pascal-kcol.cu"

#include <cstdio>

#if !defined(TREE_BASE) && !defined(TREE_C805)
#error "one of TREE_BASE, TREE_C805"
#endif

struct pick_type { ggml_type type; const char * name; };

static const pick_type pick_types[] = {
    { GGML_TYPE_Q3_K, "q3_K" },   { GGML_TYPE_Q2_0, "q2_0" },   { GGML_TYPE_IQ4_NL, "iq4_nl" },
    { GGML_TYPE_Q8_0, "q8_0" },   { GGML_TYPE_IQ4_XS, "iq4_xs" }, { GGML_TYPE_Q4_K, "q4_K" },
    { GGML_TYPE_Q5_K, "q5_K" },   { GGML_TYPE_Q6_K, "q6_K" },   { GGML_TYPE_Q4_0, "q4_0" },
    { GGML_TYPE_Q5_0, "q5_0" },   { GGML_TYPE_IQ2_XXS, "iq2_xxs" }, { GGML_TYPE_IQ2_XS, "iq2_xs" },
    { GGML_TYPE_IQ2_S, "iq2_s" }, { GGML_TYPE_IQ3_XXS, "iq3_xxs" }, { GGML_TYPE_IQ3_S, "iq3_s" },
    { GGML_TYPE_IQ1_M, "iq1_m" },
};

struct pick_out {
    const char * what;
    int launches, cols, R, itt, smem, nblocks, fallback;
};

static pick_out decide(const ggml_type type, const int K, const int nrows, const int ncols, const int row_bytes,
                       const uintptr_t x_addr) {
    pick_out d = { "declined", 0, 0, 0, 0, 0, 0, 0 };
    const int mode = kcol_mode();
    if (mode == 0) {
        d.what = "off";
        return d;
    }
    const int t = kcol_type(type);
    if (ncols == 1 && mode == 1) {
        const bool q8_short = t == kcol::KQ8_0 && K % 64 == 0 && (K / 64) % 4 != 0;
        if (t != kcol::KQ3_K && t != kcol::KIQ4_NL && !q8_short) {
            d.what = "one-token";
            return d;
        }
    }
#if defined(TREE_C805)
    const bool exact = kcol_exact() != 0 && kcol_one_token_sums(t, K);
    const bool table = mode == 1 && ncols >= 2 && kcol_table_on() && !exact;
#else
    const bool table = mode == 1 && ncols >= 2 && kcol_table_on();
#endif
    if (table && kcol_table_declines(type, K, ncols)) {
        d.what = "table";
        return d;
    }
    const kcol_overrides & o = kcol_shape_env();
    const int nsm  = o.nsm > 0 ? o.nsm : 5;
    const int smsm = o.smsm > 0 ? o.smsm : 96 * 1024;
    kcol::shape s;
    int cols = 0;
    if (!kcol::pick_shape(t, K, nrows, ncols, row_bytes, x_addr, nsm, o.r, o.itt, o.smem, s, smsm, nullptr)) {
        const kcol::type_info ti = kcol::info(t);
        if (K % ti.UW != 0) {
            d.what = "width";
            return d;
        }
#if defined(TREE_C805)
        if (!exact || !kcol::pick_exact(t, K, nrows, ncols, row_bytes, x_addr, nsm, o.r, o.itt, o.smem, kcol_exact(), s, cols,
                                        smsm, nullptr)) {
            return d;
        }
#else
        return d;
#endif
    }
    if (table && s.fallback) {
        d.what = "fallback";
        return d;
    }
    const int step = cols > 0 && cols < ncols ? cols : ncols;
    d.what     = "kcol";
    d.launches = (ncols + step - 1) / step;
    d.cols     = step;
    d.R        = s.R;
    d.itt      = s.itt;
    d.smem     = s.smem;
    d.nblocks  = s.nblocks;
    d.fallback = s.fallback;
    return d;
}

int main() {
    // row widths: the q3k and glu harness's and Qwen3.8-Flash-Next's, and the K-quants' and Q8_0's other widths
    const int Ks[]    = { 256, 320, 512, 640, 768, 1024, 2048, 2560, 3072, 4096, 6144, 8192, 10240, 12288, 16384 };
    const int rows[]  = { 33, 64, 77, 96, 129, 200, 333, 640, 1000, 2560, 4097, 6144, 10240, 12288 };
    printf("type\tK\trows\ttokens\tdecision\tlaunches\tcols\tR\titt\tsmem\tnblocks\tfallback\n");
    for (const pick_type & pt : pick_types) {
        const kcol::type_info ti = kcol::info(kcol_type(pt.type));
        for (const int K : Ks) {
            const int row_bytes = K % ti.UW == 0 ? K / ti.UW * ti.UB : 0;
            for (const int nr : rows) {
                for (int n = 1; n <= KCOL_MAX_COLS; ++n) {
                    const pick_out d = row_bytes > 0 ? decide(pt.type, K, nr, n, row_bytes, (uintptr_t) 256)
                                                     : pick_out{ "width", 0, 0, 0, 0, 0, 0, 0 };
                    printf("%s\t%d\t%d\t%d\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", pt.name, K, nr, n, d.what, d.launches, d.cols,
                           d.R, d.itt, d.smem, d.nblocks, d.fallback);
                }
            }
        }
    }
    return 0;
}
