// fasp_check: the sparse decode attention below Turing (Miner Llama's fattn-sparse-pascal-impl.cuh, copied here) against
// an FP64 reference and against ggml's own CPU FLASH_ATTN_EXT on random caches and QSA-like selections.
//
// CPU mode (default, no GPU needed): a host model of the kernel runs the header's shared functions (K/V row decoding,
// split ranges, split count, merge, sinks) and emulates the warp step by step: the transpose-reduce butterfly with its
// lane-to-value mapping, the max and sum butterflies, the per-head rescale broadcast, the P*V order. It checks that the
// kernel's decomposition (lists split over blocks, tiles of 32 entries, online softmax in base 2, merge of the splits)
// computes FLASH_ATTN_EXT with the n_kv_max hint.
// --gpu: also runs the kernel itself on device 0 (lists built on the host, as the compaction would) and checks it the
// same way. --bench: times the kernel at Flash-Next's QSA shape over the context (lists of 2,051 cells).
// --fault n: the kernel's GGML_CUDA_FATTN_SPARSE_PASCAL_FAULT codes (1 output x 1.01, 2 mask values ignored, 3 every
// split's last entry dropped), in the host model and the kernel: every check must then fail.
// --rule: only the switch's rule (fasp_rule_takes), on the host: with _PROMPT off it takes exactly the calls the switch
// took before the prompt rule, with it on the same decode and verify calls and the prompt calls of its definition, and
// with _UNION on the decode and verify calls of its definition (per union list); then the switch points at Flash-Next's
// 2,051 cells and the kernel each test-backend-ops prompt and union case takes per replay run.
// The union kernel (GGML_CUDA_FATTN_SPARSE_PASCAL_UNION): its host model runs the header's warp map (fasp_union_map) over
// union lists built on the host (as upstream's compaction in groups of tpb would), each warp with its own token's mask
// row. Its cases are checked like the per-token ones, its output is compared with the per-token model's, and where every
// query row of a sequence keeps the same cells (and the splits are the same) the two must agree bit for bit.
// --fault 4: the union kernel scores each token with the next token's mask row (the union cases must fail).
//
// Usage: fasp_check [--gpu] [--bench] [--fault n] [--nsm n] [--threads n] | fasp_check --rule
// Exit status 0 if every case passes, 1 otherwise.

#include "fattn-sparse-pascal-impl.cuh"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct fcase {
    const char * name;
    int D, G, n_head_kv, n_q, ne3, n_kv, n_kv_max;
    bool sinks;
    int  S_force;   // 0: fasp_pick_splits
    bool uni   = false; // the union kernel (tokens per list: fasp_union_tokens)
    int  share = 0;     // selections: 0 each query row its own, 1 every row of a sequence the same cells, 2 rows share
                        // 90 % of their pools (consecutive tokens of a verify or a prompt, as QSA picks them)
};

struct problem {
    fcase c;
    ggml_type type;
    int n_head;
    std::vector<float>       q;     // [ne3][n_head][n_q][D]: ggml's contiguous [D, n_q, n_head, ne3]
    std::vector<uint8_t>     k, v;  // [ne3][n_kv][n_head_kv][row]: the cache layout
    std::vector<ggml_fp16_t> mask;  // [ne3][n_q][n_kv]
    std::vector<float>       sinks; // [n_head]
    std::vector<int32_t>     idx;   // [ne3*n_q][n_kv_max], the finite mask columns in order
    std::vector<int32_t>     counts;
    int                      tpb = 1, n_lists = 0, ucap = 0; // the union lists: tokens a list, lists a sequence, length
    std::vector<int32_t>     uidx;  // [ne3*n_lists][ucap], the columns finite in any row of the list's tokens, in order
    std::vector<int32_t>     ucounts;
    size_t row_bytes;
    float  scale;
};

static int g_fault = 0;

static size_t row_size(ggml_type t, int D) {
    return ggml_row_size(t, D);
}

// QSA-like selections: pools of 4 consecutive cells anywhere, the incomplete pool at the end, a few single cells;
// mask values 0 or small negative ones; some rows short (down to 1 entry, one row empty in the streams case)
static void make_problem(problem & p, const fcase & c, ggml_type type, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    p.c = c;
    p.type = type;
    p.n_head = c.G*c.n_head_kv;
    p.row_bytes = row_size(type, c.D);
    p.scale = 1.0f/sqrtf((float) c.D);

    p.q.resize((size_t) c.ne3*p.n_head*c.n_q*c.D);
    for (float & x : p.q) {
        x = 2.0f*u(gen);
    }

    const size_t n_rows = (size_t) c.ne3*c.n_kv*c.n_head_kv;
    std::vector<float> f((size_t) c.D);
    p.k.resize(n_rows*p.row_bytes);
    p.v.resize(n_rows*p.row_bytes);
    for (int which = 0; which < 2; ++which) {
        uint8_t * dst = which == 0 ? p.k.data() : p.v.data();
        for (size_t r = 0; r < n_rows; ++r) {
            for (float & x : f) {
                x = u(gen);
            }
            ggml_quantize_chunk(type, f.data(), dst + r*p.row_bytes, 0, 1, c.D, nullptr);
        }
    }

    p.mask.assign((size_t) c.ne3*c.n_q*c.n_kv, ggml_fp32_to_fp16(-INFINITY));
    p.idx.assign((size_t) c.ne3*c.n_q*c.n_kv_max, -1);
    p.counts.assign((size_t) c.ne3*c.n_q, 0);
    std::vector<uint8_t> sel(c.n_kv), base(c.n_kv);
    for (int i3 = 0; i3 < c.ne3; ++i3) {
        for (int iq = 0; iq < c.n_q; ++iq) {
            const int row = i3*c.n_q + iq;
            if (c.share != 0 && iq > 0) {
                // share 1: the first row's cells; share 2: its pools, a tenth of them swapped for others
                sel = base;
                if (c.share == 2) {
                    std::vector<int> pools;
                    for (int b = 0; b + 4 <= c.n_kv - 3; b += 4) {
                        if (sel[b] && sel[b + 1] && sel[b + 2] && sel[b + 3]) {
                            pools.push_back(b);
                        }
                    }
                    std::shuffle(pools.begin(), pools.end(), gen);
                    const int n_swap = (int) pools.size()/10;
                    for (int k = 0; k < n_swap; ++k) {
                        for (int t = 0; t < 4; ++t) {
                            sel[pools[k] + t] = 0;
                        }
                    }
                    std::uniform_int_distribution<int> pool(0, c.n_kv/4 - 2);
                    for (int k = 0, guard = 0; k < n_swap && guard < 100000; ++guard) {
                        const int b = 4*pool(gen);
                        if (sel[b] || sel[b + 1] || sel[b + 2] || sel[b + 3] || base[b]) {
                            continue;
                        }
                        for (int t = 0; t < 4; ++t) {
                            sel[b + t] = 1;
                        }
                        ++k;
                    }
                }
            } else {
                std::fill(sel.begin(), sel.end(), 0);
                int want = c.n_kv_max - (row*13) % std::min(c.n_kv_max, 29);
                if (row == 1) {
                    want = 1;
                }
                if (c.ne3 > 1 && row == c.ne3*c.n_q - 1) {
                    want = 0;
                }
                int n = 0;
                // the incomplete pool: the last 3 cells
                for (int t = 0; t < 3 && n < want; ++t, ++n) {
                    sel[c.n_kv - 1 - t] = 1;
                }
                std::uniform_int_distribution<int> pool(0, c.n_kv/4 - 2);
                int guard = 0;
                while (n + 4 <= want && guard++ < 100000) {
                    const int b = 4*pool(gen);
                    if (sel[b] || sel[b + 1] || sel[b + 2] || sel[b + 3]) {
                        continue;
                    }
                    for (int t = 0; t < 4; ++t) {
                        sel[b + t] = 1;
                    }
                    n += 4;
                }
                std::uniform_int_distribution<int> cell(0, c.n_kv - 1);
                while (n < want) {
                    const int x = cell(gen);
                    if (!sel[x]) {
                        sel[x] = 1;
                        ++n;
                    }
                }
                base = sel;
            }
            int cnt = 0;
            for (int x = 0; x < c.n_kv; ++x) {
                if (!sel[x]) {
                    continue;
                }
                const float m = (x + row) % 5 == 0 ? -0.03125f*(1 + (x % 7)) : 0.0f;
                p.mask[(size_t) row*c.n_kv + x] = ggml_fp32_to_fp16(m);
                if (cnt < c.n_kv_max) {
                    p.idx[(size_t) row*c.n_kv_max + cnt] = x;
                }
                ++cnt;
            }
            p.counts[row] = std::min(cnt, c.n_kv_max);
        }
    }

    p.sinks.assign(p.n_head, 0.0f);
    if (c.sinks) {
        for (float & s : p.sinks) {
            s = 3.0f*u(gen);
        }
    }

    // the union lists of tpb consecutive query rows of a sequence, at most tpb x n_kv_max cells (the n_kv_max hint holds
    // for every row, so no list is cut)
    p.tpb     = fasp_union_tokens(c.G, c.n_q);
    p.n_lists = (c.n_q + p.tpb - 1)/p.tpb;
    p.ucap    = std::min(c.n_kv, p.tpb*c.n_kv_max);
    p.uidx.assign((size_t) c.ne3*p.n_lists*p.ucap, -1);
    p.ucounts.assign((size_t) c.ne3*p.n_lists, 0);
    for (int i3 = 0; i3 < c.ne3; ++i3) {
        for (int j = 0; j < p.n_lists; ++j) {
            const int l = i3*p.n_lists + j;
            int cnt = 0;
            for (int x = 0; x < c.n_kv; ++x) {
                bool any = false;
                for (int iq = j*p.tpb; iq < std::min(c.n_q, (j + 1)*p.tpb); ++iq) {
                    any |= std::isfinite(ggml_fp16_to_fp32(p.mask[((size_t) i3*c.n_q + iq)*c.n_kv + x]));
                }
                if (any && cnt < p.ucap) {
                    p.uidx[(size_t) l*p.ucap + cnt++] = x;
                }
            }
            p.ucounts[l] = cnt;
        }
    }
}

static fasp_args make_args(const problem & p, const char * Q, const char * K, const char * V, const half * mask,
        const int32_t * idx, const int32_t * counts, const int n_list, const float * sinks, float * dst, float * part,
        unsigned * counter) {
    const fcase & c = p.c;
    fasp_args a;
    a.Q = Q;
    a.q_nb1 = (int64_t) c.D*sizeof(float);
    a.q_nb2 = (int64_t) c.n_q*c.D*sizeof(float);
    a.q_nb3 = (int64_t) p.n_head*c.n_q*c.D*sizeof(float);
    a.K = K;
    a.k_nb1 = (int64_t) c.n_head_kv*p.row_bytes;
    a.k_nb2 = (int64_t) p.row_bytes;
    a.k_nb3 = (int64_t) c.n_kv*c.n_head_kv*p.row_bytes;
    a.V = V;
    a.v_nb1 = a.k_nb1;
    a.v_nb2 = a.k_nb2;
    a.v_nb3 = a.k_nb3;
    a.kv_ne3 = c.ne3;
    a.mask = mask;
    a.m_nb1 = c.n_kv;
    a.m_nb3 = (int64_t) c.n_q*c.n_kv;
    a.m_ne3 = c.ne3;
    a.idx = idx;
    a.counts = counts;
    a.n_kv_max = n_list;
    a.sinks = c.sinks ? sinks : nullptr;
    a.dst = dst;
    a.part = part;
    a.counter = counter;
    a.n_q = c.n_q;
    a.n_head = p.n_head;
    a.G = c.G;
    a.scale_log2 = p.scale*FASP_LOG2_E;
    a.fault = g_fault;
    return a;
}

// ---- the host model of one launch: blocks (s, g, z), warps w, lanes; the same order of operations as the kernel ----

// the device's fasp_transpose_reduce, all 32 lanes at once: v[lane][i]; returns out[lane] = v[lane][0] afterwards
static void tr_host(float v[32][32], float out[32]) {
    for (int o = 16; o > 0; o >>= 1) {
        float send[32][16], keep[32][16];
        for (int lane = 0; lane < 32; ++lane) {
            const bool upper = lane & o;
            for (int i = 0; i < o; ++i) {
                send[lane][i] = upper ? v[lane][i] : v[lane][i + o];
                keep[lane][i] = upper ? v[lane][i + o] : v[lane][i];
            }
        }
        for (int lane = 0; lane < 32; ++lane) {
            for (int i = 0; i < o; ++i) {
                v[lane][i] = keep[lane][i] + send[lane ^ o][i];
            }
        }
    }
    for (int lane = 0; lane < 32; ++lane) {
        out[lane] = v[lane][0];
    }
}

// x[lane] op= x[lane ^ o] for o = 4, 8, 16, as the xor shuffles do
template <typename F>
static void butterfly_host(float x[32], F op) {
    for (int o = 4; o <= 16; o <<= 1) {
        float y[32];
        for (int lane = 0; lane < 32; ++lane) {
            y[lane] = op(x[lane], x[lane ^ o]);
        }
        std::memcpy(x, y, sizeof(y));
    }
}

// one warp of a launch: query token iq of sequence i3, heads hq0 .. hq0 + 3, over the entries [e_begin, e_end) of its
// list idx with mask row mrow; writes its partial outputs, maxima and sums to part (its split, its first head). uni: the
// union kernel's warp, which skips a tile where its token keeps no cell
template <int D, int type>
static void model_warp(const fasp_args & a, const int iq, const int i3, const int hq0, const int32_t * idx,
        const half * mrow, const int e_begin, const int e_end, const char * Kg, const char * Vg, float * part,
        const bool uni) {
    constexpr int DPL  = D/FASP_WARP;
    constexpr int PART = fasp_part_floats(D);
    static float q[32][FASP_HPW][DPL], acc[32][FASP_HPW][DPL];
    float M[32], L[32];
    for (int lane = 0; lane < 32; ++lane) {
        for (int h = 0; h < FASP_HPW; ++h) {
            const float * qh = (const float *) (a.Q + iq*a.q_nb1 + (int64_t) (hq0 + h)*a.q_nb2 + i3*a.q_nb3) + DPL*lane;
            for (int j = 0; j < DPL; ++j) {
                q[lane][h][j] = qh[j]*a.scale_log2;
                acc[lane][h][j] = 0.0f;
            }
        }
        M[lane] = -FLT_MAX/2.0f;
        L[lane] = 0.0f;
    }

    for (int e0 = e_begin; e0 < e_end; e0 += FASP_TILE) {
        int cell[32];
        float mk[32];
        for (int lane = 0; lane < 32; ++lane) {
            cell[lane] = -1;
            mk[lane] = -INFINITY;
            if (e0 + lane < e_end) {
                cell[lane] = idx[e0 + lane];
                mk[lane] = a.fault == 2 ? 0.0f : __half2float(mrow[cell[lane]])*FASP_LOG2_E;
            }
        }
        if (uni && std::all_of(mk, mk + 32, [](float x) { return x == -INFINITY; })) {
            continue;
        }
        float sc[32][FASP_TILE/FASP_KPR];
        for (int j8 = 0; j8 < FASP_TILE/FASP_KPR; ++j8) {
            static float v[32][32];
            for (int j = 0; j < FASP_KPR; ++j) {
                const int c = cell[FASP_KPR*j8 + j];
                for (int lane = 0; lane < 32; ++lane) {
                    float kf[DPL];
                    if (c >= 0) {
                        fasp_load_row<type, DPL>(Kg + (int64_t) c*a.k_nb1, lane, kf);
                    } else {
                        for (int d = 0; d < DPL; ++d) {
                            kf[d] = 0.0f;
                        }
                    }
                    for (int h = 0; h < FASP_HPW; ++h) {
                        float t = q[lane][h][0]*kf[0];
                        for (int d = 1; d < DPL; ++d) {
                            t = fmaf(q[lane][h][d], kf[d], t);
                        }
                        v[lane][j*FASP_HPW + h] = t;
                    }
                }
            }
            float r[32];
            tr_host(v, r);
            for (int lane = 0; lane < 32; ++lane) {
                sc[lane][j8] = r[lane] + mk[FASP_KPR*j8 + fasp_tr_key(lane)];
            }
        }

        float m_tile[32], scale_old[32], l_tile[32];
        for (int lane = 0; lane < 32; ++lane) {
            m_tile[lane] = sc[lane][0];
            for (int j8 = 1; j8 < FASP_TILE/FASP_KPR; ++j8) {
                m_tile[lane] = fmaxf(m_tile[lane], sc[lane][j8]);
            }
        }
        butterfly_host(m_tile, [](float x, float y) { return fmaxf(x, y); });
        float p_sm[FASP_TILE*FASP_HPW];
        for (int lane = 0; lane < 32; ++lane) {
            const float M_new = fmaxf(M[lane], m_tile[lane]);
            scale_old[lane] = exp2f(M[lane] - M_new);
            M[lane] = M_new;
            l_tile[lane] = 0.0f;
            for (int j8 = 0; j8 < FASP_TILE/FASP_KPR; ++j8) {
                const float pr = exp2f(sc[lane][j8] - M[lane]);
                l_tile[lane] += pr;
                p_sm[(FASP_KPR*j8 + fasp_tr_key(lane))*FASP_HPW + fasp_tr_head(lane)] = pr;
            }
        }
        butterfly_host(l_tile, [](float x, float y) { return x + y; });
        for (int lane = 0; lane < 32; ++lane) {
            L[lane] = L[lane]*scale_old[lane] + l_tile[lane];
        }
        for (int lane = 0; lane < 32; ++lane) {
            for (int h = 0; h < FASP_HPW; ++h) {
                for (int d = 0; d < DPL; ++d) {
                    acc[lane][h][d] *= scale_old[h];
                }
            }
        }
        for (int kk = 0; kk < FASP_TILE; ++kk) {
            const int c = cell[kk];
            if (c < 0) {
                break;
            }
            for (int lane = 0; lane < 32; ++lane) {
                float vf[DPL];
                fasp_load_row<type, DPL>(Vg + (int64_t) c*a.v_nb1, lane, vf);
                for (int h = 0; h < FASP_HPW; ++h) {
                    for (int d = 0; d < DPL; ++d) {
                        acc[lane][h][d] = fmaf(p_sm[kk*FASP_HPW + h], vf[d], acc[lane][h][d]);
                    }
                }
            }
        }
    }

    for (int lane = 0; lane < 32; ++lane) {
        for (int h = 0; h < FASP_HPW; ++h) {
            for (int j = 0; j < DPL; ++j) {
                part[h*PART + DPL*lane + j] = acc[lane][h][j];
            }
        }
        if (lane < FASP_HPW) {
            part[lane*PART + D + 0] = M[lane];
            part[lane*PART + D + 1] = L[lane];
        }
    }
}

// the last block of a column merges its splits: heads hq0 .. hq0 + 3 (of warp w) of query row z, column col
template <int D>
static void model_merge(const fasp_args & a, const int64_t col, const int64_t split_stride, const int S, const int w,
        const int hq0, const int z) {
    constexpr int DPL  = D/FASP_WARP;
    constexpr int PART = fasp_part_floats(D);
    for (int h = 0; h < FASP_HPW; ++h) {
        const int hq = hq0 + h;
        for (int lane = 0; lane < 32; ++lane) {
            float o[DPL];
            float m_all, l;
            fasp_merge_lane<DPL>(a.part + (col*a.G + FASP_HPW*w + h)*PART, split_stride, S, D, lane, o, m_all, l);
            float f = 1.0f;
            if (a.sinks) {
                f = fasp_add_sink(a.sinks[hq]*FASP_LOG2_E, m_all, l);
            }
            const float inv = l > 0.0f ? (a.fault == 1 ? 1.01f : 1.0f)*f/l : 0.0f;
            float * out = a.dst + ((int64_t) z*a.n_head + hq)*D + DPL*lane;
            for (int j = 0; j < DPL; ++j) {
                out[j] = o[j]*inv;
            }
        }
    }
}

// the per-token kernel: blocks (s, g, z), warps w = heads 4w .. 4w + 3
template <int D, int type>
static void model_launch(const fasp_args & a, const int S, const int n_head_kv, const int n_z) {
    constexpr int PART = fasp_part_floats(D);
    const int NW = a.G/FASP_HPW;
    const int n_cols = n_head_kv*n_z;
    const int64_t split_stride = (int64_t) n_cols*a.G*PART;
    std::vector<unsigned> done(n_cols, 0);

    for (int z = 0; z < n_z; ++z) {
        for (int g = 0; g < n_head_kv; ++g) {
            for (int s = 0; s < S; ++s) {
                const int iq = z % a.n_q;
                const int i3 = z / a.n_q;
                const int list = (i3 % a.m_ne3)*a.n_q + iq;
                const int32_t * idx = a.idx + (int64_t) list*a.n_kv_max;
                const half * mrow = a.mask + (i3 % a.m_ne3)*a.m_nb3 + iq*a.m_nb1;
                const int count = a.counts[list];
                int e_begin, e_end;
                fasp_split_range(count, s, S, e_begin, e_end);
                if (a.fault == 3 && e_end > e_begin) {
                    --e_end;
                }
                const int ikv = i3 % a.kv_ne3;
                const char * Kg = a.K + (int64_t) g*a.k_nb2 + ikv*a.k_nb3;
                const char * Vg = a.V + (int64_t) g*a.v_nb2 + ikv*a.v_nb3;
                const int col = z*n_head_kv + g;

                for (int w = 0; w < NW; ++w) {
                    const int hq0 = g*a.G + FASP_HPW*w;
                    float * part = a.part + (int64_t) s*split_stride + ((int64_t) col*a.G + FASP_HPW*w)*PART;
                    model_warp<D, type>(a, iq, i3, hq0, idx, mrow, e_begin, e_end, Kg, Vg, part, false);
                }

                // the last block of the column merges (blocks run in order here, so it is split S - 1)
                if (++done[col] != (unsigned) S) {
                    continue;
                }
                for (int w = 0; w < NW; ++w) {
                    model_merge<D>(a, col, split_stride, S, w, g*a.G + FASP_HPW*w, z);
                }
            }
        }
    }
}

// the union kernel: blocks (s, g, list column bz), warp slots wy = token wy / (G/4) of the list, heads of wy % (G/4)
// (fasp_union_map); a list's tokens read its cells in the same order, each with its own mask row
template <int D, int type>
static void model_launch_union(const fasp_args & a, const int tpb, const int S, const int n_head_kv, const int ne3) {
    constexpr int PART = fasp_part_floats(D);
    const int NW = a.G/FASP_HPW;
    const int n_lists = (a.n_q + tpb - 1)/tpb;
    const int n_cols = n_head_kv*a.n_q*ne3;
    const int64_t split_stride = (int64_t) n_cols*a.G*PART;
    std::vector<unsigned> done((size_t) n_lists*ne3*n_head_kv, 0);

    for (int bz = 0; bz < n_lists*ne3; ++bz) {
        for (int g = 0; g < n_head_kv; ++g) {
            for (int s = 0; s < S; ++s) {
                for (int wy = 0; wy < NW*tpb; ++wy) {
                    const fasp_union_warp uw = fasp_union_map(bz, wy, NW, tpb, a.n_q, a.m_ne3);
                    if (!uw.valid) {
                        continue;
                    }
                    const int w = wy % NW;
                    const int iq_mask = a.fault == 4 && uw.iq + 1 < a.n_q ? uw.iq + 1 : uw.iq;
                    const int32_t * idx = a.idx + (int64_t) uw.list*a.n_kv_max;
                    const half * mrow = a.mask + (uw.i3 % a.m_ne3)*a.m_nb3 + iq_mask*a.m_nb1;
                    int e_begin, e_end;
                    fasp_split_range(a.counts[uw.list], s, S, e_begin, e_end);
                    if (a.fault == 3 && e_end > e_begin) {
                        --e_end;
                    }
                    const int ikv = uw.i3 % a.kv_ne3;
                    const char * Kg = a.K + (int64_t) g*a.k_nb2 + ikv*a.k_nb3;
                    const char * Vg = a.V + (int64_t) g*a.v_nb2 + ikv*a.v_nb3;
                    const int z = uw.i3*a.n_q + uw.iq;
                    const int64_t col = (int64_t) z*n_head_kv + g;
                    float * part = a.part + (int64_t) s*split_stride + (col*a.G + FASP_HPW*w)*PART;
                    model_warp<D, type>(a, uw.iq, uw.i3, g*a.G + FASP_HPW*w, idx, mrow, e_begin, e_end, Kg, Vg, part, true);
                }

                // the last block of the list column merges every token's splits
                if (++done[(size_t) bz*n_head_kv + g] != (unsigned) S) {
                    continue;
                }
                for (int wy = 0; wy < NW*tpb; ++wy) {
                    const fasp_union_warp uw = fasp_union_map(bz, wy, NW, tpb, a.n_q, a.m_ne3);
                    if (uw.valid) {
                        const int w = wy % NW;
                        const int z = uw.i3*a.n_q + uw.iq;
                        model_merge<D>(a, (int64_t) z*n_head_kv + g, split_stride, S, w, g*a.G + FASP_HPW*w, z);
                    }
                }
            }
        }
    }
}

static void run_model(const problem & p, const int S, std::vector<float> & out, const bool uni = false) {
    const fcase & c = p.c;
    out.assign((size_t) c.ne3*c.n_q*p.n_head*c.D, 0.0f);
    const int n_cols = c.n_head_kv*c.n_q*c.ne3;
    std::vector<float> part((size_t) S*n_cols*c.G*fasp_part_floats(c.D));
    const fasp_args a = uni ?
        make_args(p, (const char *) p.q.data(), (const char *) p.k.data(), (const char *) p.v.data(),
            (const half *) p.mask.data(), p.uidx.data(), p.ucounts.data(), p.ucap, p.sinks.data(), out.data(), part.data(), nullptr) :
        make_args(p, (const char *) p.q.data(), (const char *) p.k.data(), (const char *) p.v.data(),
            (const half *) p.mask.data(), p.idx.data(), p.counts.data(), c.n_kv_max, p.sinks.data(), out.data(), part.data(), nullptr);
    const int n_z = c.n_q*c.ne3;
    if (uni) {
        if (c.D == 256) {
            p.type == GGML_TYPE_F16 ? model_launch_union<256, FASP_F16>(a, p.tpb, S, c.n_head_kv, c.ne3) : model_launch_union<256, FASP_Q8_0>(a, p.tpb, S, c.n_head_kv, c.ne3);
        } else {
            p.type == GGML_TYPE_F16 ? model_launch_union<128, FASP_F16>(a, p.tpb, S, c.n_head_kv, c.ne3) : model_launch_union<128, FASP_Q8_0>(a, p.tpb, S, c.n_head_kv, c.ne3);
        }
    } else if (c.D == 256) {
        p.type == GGML_TYPE_F16 ? model_launch<256, FASP_F16>(a, S, c.n_head_kv, n_z) : model_launch<256, FASP_Q8_0>(a, S, c.n_head_kv, n_z);
    } else {
        p.type == GGML_TYPE_F16 ? model_launch<128, FASP_F16>(a, S, c.n_head_kv, n_z) : model_launch<128, FASP_Q8_0>(a, S, c.n_head_kv, n_z);
    }
}

// ---- references ----

// FP64 softmax over each query's listed cells, K/V decoded by ggml's own to_float; out [ne3][n_q][n_head][D]
static void run_fp64(const problem & p, std::vector<double> & out) {
    const fcase & c = p.c;
    const auto * tt = ggml_get_type_traits(p.type);
    out.assign((size_t) c.ne3*c.n_q*p.n_head*c.D, 0.0);
    std::vector<float> kf(c.D), vf(c.D);
    std::vector<double> sc, o(c.D);
    for (int i3 = 0; i3 < c.ne3; ++i3) {
        for (int iq = 0; iq < c.n_q; ++iq) {
            const int row = i3*c.n_q + iq;
            const int cnt = p.counts[row];
            for (int hq = 0; hq < p.n_head; ++hq) {
                const int g = hq / c.G;
                const float * qv = p.q.data() + (((size_t) i3*p.n_head + hq)*c.n_q + iq)*c.D;
                sc.assign(cnt, 0.0);
                double m = -INFINITY;
                for (int e = 0; e < cnt; ++e) {
                    const int x = p.idx[(size_t) row*c.n_kv_max + e];
                    tt->to_float(p.k.data() + (((size_t) i3*c.n_kv + x)*c.n_head_kv + g)*p.row_bytes, kf.data(), c.D);
                    double d = 0.0;
                    for (int i = 0; i < c.D; ++i) {
                        d += (double) qv[i]*kf[i];
                    }
                    sc[e] = d*p.scale + ggml_fp16_to_fp32(p.mask[(size_t) row*c.n_kv + x]);
                    m = std::max(m, sc[e]);
                }
                if (c.sinks) {
                    m = std::max(m, (double) p.sinks[hq]);
                }
                std::fill(o.begin(), o.end(), 0.0);
                double l = 0.0;
                for (int e = 0; e < cnt; ++e) {
                    const int x = p.idx[(size_t) row*c.n_kv_max + e];
                    tt->to_float(p.v.data() + (((size_t) i3*c.n_kv + x)*c.n_head_kv + g)*p.row_bytes, vf.data(), c.D);
                    const double pr = exp(sc[e] - m);
                    l += pr;
                    for (int i = 0; i < c.D; ++i) {
                        o[i] += pr*vf[i];
                    }
                }
                if (c.sinks) {
                    l += exp((double) p.sinks[hq] - m);
                }
                double * dst = out.data() + (((size_t) i3*c.n_q + iq)*p.n_head + hq)*c.D;
                for (int i = 0; i < c.D; ++i) {
                    dst[i] = l > 0.0 ? o[i]/l : 0.0;
                }
            }
        }
    }
}

// ggml's CPU FLASH_ATTN_EXT on the whole masked cache, as test-backend-ops builds the call; out [ne3][n_q][n_head][D]
static void run_ggml(const problem & p, const int n_threads, std::vector<float> & out) {
    const fcase & c = p.c;
    const size_t mem = p.q.size()*sizeof(float) + p.k.size() + p.v.size() + p.mask.size()*sizeof(ggml_fp16_t) +
        p.q.size()*sizeof(float) + p.sinks.size()*sizeof(float) + (size_t) 256*1024*1024;
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.D, c.n_q, p.n_head, c.ne3);
    ggml_tensor * k0 = ggml_new_tensor_4d(ctx, p.type, c.D, c.n_head_kv, c.n_kv, c.ne3);
    ggml_tensor * v0 = ggml_new_tensor_4d(ctx, p.type, c.D, c.n_head_kv, c.n_kv, c.ne3);
    ggml_tensor * m  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.n_kv, c.n_q, 1, c.ne3);
    memcpy(q->data, p.q.data(), ggml_nbytes(q));
    memcpy(k0->data, p.k.data(), ggml_nbytes(k0));
    memcpy(v0->data, p.v.data(), ggml_nbytes(v0));
    memcpy(m->data, p.mask.data(), ggml_nbytes(m));
    ggml_tensor * k = ggml_permute(ctx, k0, 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(ctx, v0, 0, 2, 1, 3);

    ggml_tensor * r = ggml_flash_attn_ext(ctx, q, k, v, m, p.scale, 0.0f, 0.0f);
    if (c.sinks) {
        ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, p.n_head);
        memcpy(s->data, p.sinks.data(), ggml_nbytes(s));
        ggml_flash_attn_ext_add_sinks(r, s);
    }
    ggml_flash_attn_ext_set_n_kv_max(r, c.n_kv_max);
    ggml_prec_set_acc(r, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    // r: [D, n_head, n_q, ne3]
    out.assign((const float *) r->data, (const float *) r->data + ggml_nelements(r));
    ggml_free(ctx);
}

struct errs {
    double max_rel; // max |x - ref| / max |ref|
    double nmse;
};

template <typename T>
static errs compare(const std::vector<float> & x, const std::vector<T> & ref) {
    double num = 0.0, den = 0.0, mx = 0.0, mref = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double d = (double) x[i] - (double) ref[i];
        if (!std::isfinite(x[i])) {
            return { INFINITY, INFINITY };
        }
        num += d*d;
        den += (double) ref[i]*ref[i];
        mx = std::max(mx, fabs(d));
        mref = std::max(mref, fabs((double) ref[i]));
    }
    return { mref > 0.0 ? mx/mref : mx, den > 0.0 ? num/den : num };
}

// ---- the kernel on the GPU ----

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

struct gpu_bufs {
    char * q = nullptr, * k = nullptr, * v = nullptr;
    half * mask = nullptr;
    int32_t * idx = nullptr, * counts = nullptr;
    float * sinks = nullptr, * dst = nullptr, * part = nullptr;
    unsigned * counter = nullptr;
};

static void gpu_launch(const fasp_args & a, const problem & p, const int S, cudaStream_t st, const bool uni = false) {
    const fcase & c = p.c;
    const int n_z = c.n_q*c.ne3;
    if (uni) {
        if (c.D == 256) {
            p.type == GGML_TYPE_F16 ? fasp_launch_union<256, FASP_F16>(a, p.tpb, S, c.n_head_kv, c.ne3, st) : fasp_launch_union<256, FASP_Q8_0>(a, p.tpb, S, c.n_head_kv, c.ne3, st);
        } else {
            p.type == GGML_TYPE_F16 ? fasp_launch_union<128, FASP_F16>(a, p.tpb, S, c.n_head_kv, c.ne3, st) : fasp_launch_union<128, FASP_Q8_0>(a, p.tpb, S, c.n_head_kv, c.ne3, st);
        }
    } else if (c.D == 256) {
        p.type == GGML_TYPE_F16 ? fasp_launch<256, FASP_F16>(a, S, c.n_head_kv, n_z, st) : fasp_launch<256, FASP_Q8_0>(a, S, c.n_head_kv, n_z, st);
    } else {
        p.type == GGML_TYPE_F16 ? fasp_launch<128, FASP_F16>(a, S, c.n_head_kv, n_z, st) : fasp_launch<128, FASP_Q8_0>(a, S, c.n_head_kv, n_z, st);
    }
}

// uni: the union lists in place of the per-token ones
static fasp_args gpu_setup(const problem & p, const int S, gpu_bufs & b, const bool uni = false) {
    const fcase & c = p.c;
    const std::vector<int32_t> & idx    = uni ? p.uidx : p.idx;
    const std::vector<int32_t> & counts = uni ? p.ucounts : p.counts;
    const int n_cols = c.n_head_kv*c.n_q*c.ne3;
    CK(cudaMalloc(&b.q, p.q.size()*sizeof(float)));
    CK(cudaMalloc(&b.k, p.k.size()));
    CK(cudaMalloc(&b.v, p.v.size()));
    CK(cudaMalloc(&b.mask, p.mask.size()*sizeof(half)));
    CK(cudaMalloc(&b.idx, idx.size()*sizeof(int32_t)));
    CK(cudaMalloc(&b.counts, counts.size()*sizeof(int32_t)));
    CK(cudaMalloc(&b.sinks, p.sinks.size()*sizeof(float)));
    CK(cudaMalloc(&b.dst, p.q.size()*sizeof(float)));
    CK(cudaMalloc(&b.part, (size_t) S*n_cols*c.G*fasp_part_floats(c.D)*sizeof(float)));
    CK(cudaMalloc(&b.counter, FASP_MAX_COLS*sizeof(unsigned)));
    CK(cudaMemcpy(b.q, p.q.data(), p.q.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.k, p.k.data(), p.k.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.v, p.v.data(), p.v.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.mask, p.mask.data(), p.mask.size()*sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.idx, idx.data(), idx.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.counts, counts.data(), counts.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.sinks, p.sinks.data(), p.sinks.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemset(b.counter, 0, FASP_MAX_COLS*sizeof(unsigned)));
    CK(cudaMemset(b.dst, 0xFF, p.q.size()*sizeof(float)));
    return make_args(p, b.q, b.k, b.v, b.mask, b.idx, b.counts, uni ? p.ucap : c.n_kv_max, b.sinks, b.dst, b.part, b.counter);
}

static void gpu_free(gpu_bufs & b) {
    for (void * x : { (void *) b.q, (void *) b.k, (void *) b.v, (void *) b.mask, (void *) b.idx, (void *) b.counts,
                      (void *) b.sinks, (void *) b.dst, (void *) b.part, (void *) b.counter }) {
        CK(cudaFree(x));
    }
}

static void run_gpu(const problem & p, const int S, std::vector<float> & out, const bool uni = false) {
    gpu_bufs b;
    const fasp_args a = gpu_setup(p, S, b, uni);
    gpu_launch(a, p, S, 0, uni);
    CK(cudaGetLastError());
    // twice: the counters must be back at zero after a launch
    gpu_launch(a, p, S, 0, uni);
    CK(cudaDeviceSynchronize());
    out.resize(p.q.size());
    CK(cudaMemcpy(out.data(), b.dst, out.size()*sizeof(float), cudaMemcpyDeviceToHost));
    gpu_free(b);
}

// The switch's rule before the prompt rule (Miner Llama pascal-p106-v4 84f7bd3d2, fattn-sparse-pascal.cu): at most nq
// query tokens, and in mode 1 at least min_ratio x n_kv_max cells per query token.
static bool rule_before(const fasp_rule & r, const int64_t n_q, const int64_t n_kv, const int32_t n_kv_max) {
    if (n_q > r.nq) {
        return false;
    }
    if (r.mode == 1 && (double) n_kv < r.min_ratio*n_kv_max*n_q) {
        return false;
    }
    return true;
}

// The prompt rule as GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT defines it: more than nq and at most prompt_nq query tokens,
// in mode 1 at least prompt_min_ratio x n_kv_max cells whatever the number of query tokens, in mode 2 any cache size.
static bool rule_prompt(const fasp_rule & r, const int64_t n_q, const int64_t n_kv, const int32_t n_kv_max) {
    if (n_q <= r.nq || r.prompt == 0 || n_q > r.prompt_nq) {
        return false;
    }
    return r.mode == 2 || (double) n_kv >= (double) r.prompt_min_ratio*n_kv_max;
}

// The union rule as GGML_CUDA_FATTN_SPARSE_PASCAL_UNION defines it: a decode or verify call (at most nq query tokens)
// of tpb > 1 tokens a list, in mode 1 at least union_min_ratio x n_kv_max cells per union list (ceil(n_q/tpb) of
// them), in mode 2 any cache size.
static bool rule_union(const fasp_rule & r, const int64_t n_q, const int64_t n_kv, const int32_t n_kv_max, const int tpb) {
    if (n_q > r.nq || r.uni == 0 || tpb < 2) {
        return false;
    }
    const int64_t lists = (n_q + tpb - 1)/tpb;
    return r.mode == 2 || (double) n_kv >= (double) r.union_min_ratio*n_kv_max*lists;
}

// smallest cache (cells) from which the rule takes a call of n_q query tokens with G query heads per KV head, -1 if
// none up to 2^22
static int64_t rule_from(const fasp_rule & r, const int64_t n_q, const int32_t n_kv_max, const int G = 12) {
    for (int64_t n_kv = 0; n_kv <= (int64_t) 1 << 22; ++n_kv) {
        if (fasp_rule_takes(r, n_q, n_kv, n_kv_max, r.uni ? fasp_union_tokens(G, n_q) : 1)) {
            return n_kv;
        }
    }
    return -1;
}

static int rule_check() {
    // 1. every setting of a grid against the definitions
    const int64_t kvs[] = { 0, 1, 31, 32, 255, 256, 333, 512, 1027, 1538, 1539, 2050, 2051, 2052, 3072, 3076, 3077,
                            3078, 4096, 4102, 4352, 5127, 5128, 6153, 6154, 8192, 8204, 8448, 12288, 12306, 12544, 16384,
                            32768, 65536, 131072, 262144, 393791, 393792, 393793 };
    long n = 0, bad_off = 0, bad_dec = 0, bad_prompt = 0, n_prompt_taken = 0, bad_uoff = 0, bad_union = 0, n_union_taken = 0;
    for (int mode : { 1, 2 }) {
        for (int nq : { 1, 4, 8, 16, 128 }) {
            for (float min_ratio : { 0.25f, 1.0f, 1.5f }) {
                for (int prompt : { 0, 1 }) {
                    for (int prompt_nq : { 16, 128, 512 }) {
                        for (float prompt_min_ratio : { 0.5f, 1.0f, 2.5f }) {
                            for (int32_t n_kv_max : { 31, 333, 2051 }) {
                                for (int uni : { 0, 1 }) {
                                    for (float union_min_ratio : { 0.5f, 1.5f }) {
                                        for (int tpb : { 1, 2, 3, 4, 8 }) {
                                            if (uni == 0 && (union_min_ratio != 1.5f || (tpb != 1 && tpb != 4))) {
                                                continue;
                                            }
                                            const fasp_rule r = { mode, nq, min_ratio, prompt, prompt_nq, prompt_min_ratio,
                                                                  uni, union_min_ratio };
                                            for (int64_t n_q = 1; n_q <= 600; ++n_q) {
                                                for (int64_t n_kv : kvs) {
                                                    const bool t = fasp_rule_takes(r, n_q, n_kv, n_kv_max, tpb);
                                                    ++n;
                                                    if (n_q > nq) {
                                                        // prompt calls: the prompt rule, with or without _UNION
                                                        const bool want = prompt != 0 && rule_prompt(r, n_q, n_kv, n_kv_max);
                                                        if (prompt == 0) {
                                                            bad_off += t != rule_before(r, n_q, n_kv, n_kv_max);
                                                        }
                                                        bad_prompt += t != want;
                                                        n_prompt_taken += t;
                                                    } else if (uni == 0 || tpb < 2) {
                                                        // _UNION off (or one token a list): the rule before, any tpb
                                                        const bool d = t != rule_before(r, n_q, n_kv, n_kv_max);
                                                        (uni == 0 ? bad_uoff : bad_dec) += d;
                                                    } else {
                                                        bad_union += t != rule_union(r, n_q, n_kv, n_kv_max, tpb);
                                                        n_union_taken += t;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    printf("rule: %ld settings x calls; _PROMPT off: %ld prompt calls differ from the rule before; prompt calls %ld differ "
           "from the prompt rule (%ld taken); _UNION off: %ld decode and verify calls differ from the rule before; "
           "_UNION on: one token a list %ld differ from it, %ld calls differ from the union rule (%ld taken)\n",
           n, bad_off, bad_prompt, n_prompt_taken, bad_uoff, bad_dec, bad_union, n_union_taken);
    const bool ok = bad_off == 0 && bad_dec == 0 && bad_prompt == 0 && n_prompt_taken > 0 && bad_uoff == 0 &&
        bad_union == 0 && n_union_taken > 0;

    // 2. the switch points at Flash-Next's 2,051 cells and 12 query heads per KV head (mode 1)
    struct named { const char * name; fasp_rule r; };
    const named sets[] = {
        { "=1 (v4)",                         { 1, 8, 1.5f, 0, 128, 1.0f, 0, 1.5f } },
        { "=1 _PROMPT=1",                    { 1, 8, 1.5f, 1, 128, 1.0f, 0, 1.5f } },
        { "=1 _MIN_RATIO=1.0 _PROMPT=1",     { 1, 8, 1.0f, 1, 128, 1.0f, 0, 1.5f } },
        { "=1 _UNION=1",                     { 1, 8, 1.5f, 0, 128, 1.0f, 1, 1.5f } },
        { "=1 _UNION=1 _PROMPT=1",           { 1, 8, 1.5f, 1, 128, 1.0f, 1, 1.5f } },
        { "=1 _UNION=1 _UNION_MIN_RATIO=1.0", { 1, 8, 1.5f, 0, 128, 1.0f, 1, 1.0f } },
        { "=1 _UNION=1 _UNION_MIN_RATIO=0.5", { 1, 8, 1.5f, 0, 128, 1.0f, 1, 0.5f } },
    };
    printf("\nswitch points, n_kv_max 2051, 12 query heads per KV head (4 tokens a union list): the smallest cache "
           "(cells) from which a call of n_q query tokens goes to the kernel (never: it stays on the dense kernels)\n%-38s",
           "setting \\ n_q");
    const int64_t nqs[] = { 1, 2, 3, 4, 5, 8, 9, 16, 128, 129, 512 };
    for (int64_t q : nqs) {
        printf(" %7lld", (long long) q);
    }
    printf("\n");
    for (const named & x : sets) {
        printf("%-38s", x.name);
        for (int64_t q : nqs) {
            const int64_t f = rule_from(x.r, q, 2051);
            if (f < 0) {
                printf(" %7s", "never");
            } else {
                printf(" %7lld", (long long) f);
            }
        }
        printf("\n");
    }

    // 3. which kernel each prompt and union case of test-backend-ops (FLASH_ATTN_EXT, n_kv_max > 0, more than one token)
    // takes in the replay runs of scripts/record_checks.sh's suites qsa-prompt and qsa-union on a P106 (S: the
    // per-token sparse kernel, U: the union kernel, d: the dense ones; over 1,024 query tokens x KV heads always d)
    struct tcase { int G, n_head_kv, nb, ne3, kv, n_kv_max; };
    const tcase tcs[] = {
        { 12, 2,   9, 1,  8192, 2051 }, { 12, 2,  16, 1,  8192, 2051 }, { 12, 2,  64, 1,  8192, 2051 },
        { 12, 2, 128, 1,  8192, 2051 }, { 12, 2,   9, 1,  3072, 2051 }, { 12, 2, 128, 1,  3072, 2051 },
        { 12, 2, 128, 1, 32768, 2051 }, { 12, 2,  64, 2,  4096, 2051 }, {  8, 4,  32, 1,  4096, 1027 },
        { 12, 2, 129, 1,  4096, 2051 }, { 12, 2, 256, 1,  4096, 2051 }, { 12, 2, 512, 1,  4096, 2051 },
        // qsa-union's own cases
        { 12, 2,   2, 1,  4096, 2051 }, { 12, 2,   3, 1,  4096, 2051 }, { 12, 2,   4, 1,  4096, 2051 },
        { 12, 2,   5, 1,  4096, 2051 }, { 12, 2,   8, 1,  4096, 2051 }, { 12, 2, 128, 1,  4096, 2051 },
        { 12, 2,   2, 1, 12288, 2051 }, { 12, 2,   4, 1, 12288, 2051 }, { 12, 2,   5, 1, 12288, 2051 },
        { 12, 2,   8, 1, 12288, 2051 }, { 12, 2, 128, 1, 12288, 2051 }, { 12, 2,   2, 1, 49152, 2051 },
        { 12, 2,   4, 1, 49152, 2051 }, { 12, 2,   8, 1, 49152, 2051 }, { 12, 2, 128, 1, 49152, 2051 },
        { 12, 2,   6, 2,  8192, 2051 }, {  4, 2,   8, 1,  4096,  333 }, {  4, 2,  13, 1,  4096,  333 },
        {  8, 4,   7, 1,  4096, 1027 }, { 16, 1,   5, 1,  4096,   31 },
        // the earlier verify cases at 8,192 and the two-sequence one
        { 12, 2,   2, 1,  8192, 2051 }, { 12, 2,   3, 1,  8192, 2051 }, { 12, 2,   5, 1,  8192, 2051 },
        { 12, 2,   8, 1,  8192, 2051 }, { 12, 2,   4, 2,  4096, 2051 },
    };
    const named runs[] = {
        { "qp-sparse1",  { 1, 8, 1.5f, 0, 128, 1.0f, 0, 1.5f } },
        { "qp-prompt",   { 1, 8, 1.5f, 1, 128, 1.0f, 0, 1.5f } },
        { "qp-m2",       { 2, 8, 1.5f, 1, 512, 1.0f, 0, 1.5f } },
        { "qu-union",    { 1, 8, 1.5f, 0, 128, 1.0f, 1, 1.5f } },
        { "qu-uprompt",  { 1, 8, 1.5f, 1, 128, 1.0f, 1, 1.5f } },
        { "qu-ur05",     { 1, 8, 1.5f, 1, 512, 1.0f, 1, 0.5f } },
        { "qu-m2",       { 2, 8, 1.5f, 1, 512, 1.0f, 0, 1.5f } },
        { "qu-m2-union", { 2, 8, 1.5f, 1, 512, 1.0f, 1, 1.5f } },
    };
    printf("\nthe prompt and union cases' kernel per replay run (G: query heads per KV head; head 256, but head 128 at G 4 "
           "and 8, with sinks at 8)\n%-34s", "G, KV heads, nb, ne3, kv, n_kv_max");
    for (const named & x : runs) {
        printf(" %-11s", x.name);
    }
    printf("\n");
    for (const tcase & c : tcs) {
        char b[64];
        snprintf(b, sizeof(b), "%d, %d, %d, %d, %d, %d", c.G, c.n_head_kv, c.nb, c.ne3, c.kv, c.n_kv_max);
        printf("%-34s", b);
        for (const named & x : runs) {
            const bool fits = (int64_t) c.n_head_kv*c.nb*c.ne3 <= FASP_MAX_COLS;
            const int tpb = x.r.uni ? fasp_union_tokens(c.G, c.nb) : 1;
            const bool t = fits && fasp_rule_takes(x.r, c.nb, c.kv, c.n_kv_max, tpb);
            printf(" %-11s", !t ? "d" : tpb > 1 ? "U" : "S");
        }
        printf("\n");
    }
    printf("fasp_check --rule: %s\n", ok ? "pass" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    bool gpu = false, bench = false;
    int nsm = 5, n_threads = 4;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--gpu") {
            gpu = true;
        } else if (s == "--bench") {
            bench = true;
        } else if (s == "--fault" && i + 1 < argc) {
            g_fault = atoi(argv[++i]);
        } else if (s == "--nsm" && i + 1 < argc) {
            nsm = atoi(argv[++i]);
        } else if (s == "--threads" && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (s == "--rule") {
            return rule_check();
        } else {
            fprintf(stderr, "usage: %s [--gpu] [--bench] [--fault n] [--nsm n] [--threads n] | %s --rule\n", argv[0], argv[0]);
            return 2;
        }
    }
    ggml_cpu_init();

    const fcase cases[] = {
        { "flash-next decode",         256, 12, 2, 1, 1,  8192, 2051, false, 0 },
        { "flash-next verify 5",       256, 12, 2, 5, 1,  8192, 2051, false, 0 },
        { "flash-next 32k",            256, 12, 2, 1, 1, 32768, 2051, false, 0 },
        { "flash-next 1 split",        256, 12, 2, 1, 1,  8192, 2051, false, 1 },
        { "flash-next 32 splits",      256, 12, 2, 2, 1,  8192, 2051, false, 32 },
        { "two streams",               256, 12, 2, 3, 2,  4096, 2051, false, 0 },
        { "head 128, 8 per KV, sinks", 128,  8, 4, 1, 1,  4096, 1027, true,  0 },
        { "head 128, 4 per KV",        128,  4, 2, 7, 1,  4096,  333, false, 0 },
        { "16 per KV, short list",     256, 16, 1, 1, 1,  4096,   31, false, 0 },
        { "one tile",                  256, 12, 2, 2, 1,  4096,   32, false, 0 },
        { "flash-next prompt 16",      256, 12, 2, 16, 1, 8192, 2051, false, 0 },
        { "flash-next prompt 128",     256, 12, 2, 128, 1, 4096, 2051, false, 0 },
        { "1,024 columns (512 tokens)", 128, 4, 2, 512, 1, 2560,  333, false, 0 },
    };

    // pass: the host model (and the kernel) within 2e-5 of the FP64 reference (max error over the largest output),
    // and within test-backend-ops' FLASH_ATTN_EXT tolerance (NMSE 5e-4) of ggml's CPU kernel
    const double tol_rel = 2e-5, tol_nmse = 5e-4;
    int n_fail = 0, n_run = 0;
    printf("%-26s %-5s %3s %4s | %-21s | %-21s\n", "case", "type", "S", "lst", "model: rel64  nmse_ggml", "gpu: rel64  nmse_ggml");
    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        uint32_t seed = 1234;
        for (const fcase & c : cases) {
            problem p;
            make_problem(p, c, type, seed++);
            const int n_cols = c.n_head_kv*c.n_q*c.ne3;
            const int S = c.S_force > 0 ? c.S_force : fasp_pick_splits(c.n_kv_max, n_cols, nsm);

            std::vector<double> r64;
            std::vector<float> rg, xm, xg;
            run_fp64(p, r64);
            run_ggml(p, n_threads, rg);
            run_model(p, S, xm);
            const errs em64 = compare(xm, r64), emg = compare(xm, rg);
            bool ok = em64.max_rel < tol_rel && emg.nmse < tol_nmse;
            char gbuf[64] = "-";
            if (gpu) {
                run_gpu(p, S, xg);
                const errs eg64 = compare(xg, r64), egg = compare(xg, rg);
                ok = ok && eg64.max_rel < tol_rel && egg.nmse < tol_nmse;
                snprintf(gbuf, sizeof(gbuf), "%.2e  %.2e", eg64.max_rel, egg.nmse);
            }
            ++n_run;
            n_fail += !ok;
            printf("%-26s %-5s %3d %4d | %.2e  %.2e     | %-21s | %s\n", c.name, ggml_type_name(type), S, p.counts[0],
                   em64.max_rel, emg.nmse, gbuf, ok ? "pass" : "FAIL");
            fflush(stdout);
        }
    }

    // the union kernel (fasp_union_tokens tokens a list, splits as fattn-sparse-pascal.cu picks them): checked like the
    // per-token cases, and compared with the per-token model on the same problem (its own splits); where every row of a
    // sequence keeps the same cells (share 1) and the splits are the same, the two must agree bit for bit
    const fcase ucases[] = {
        { "union verify 2",            256, 12, 2,   2, 1,  8192, 2051, false, 0,  true },
        { "union verify 4",            256, 12, 2,   4, 1,  8192, 2051, false, 0,  true },
        { "union verify 5 (4+1)",      256, 12, 2,   5, 1, 12288, 2051, false, 0,  true },
        { "union verify 8, 32k",       256, 12, 2,   8, 1, 32768, 2051, false, 0,  true },
        { "union verify 4, shared 90", 256, 12, 2,   4, 1, 32768, 2051, false, 0,  true, 2 },
        { "union prompt 128",          256, 12, 2, 128, 1,  4096, 2051, false, 0,  true },
        { "union prompt 128, shr 90",  256, 12, 2, 128, 1, 16384, 2051, false, 0,  true, 2 },
        { "union two streams (4+2)",   256, 12, 2,   6, 2,  4096, 2051, false, 0,  true },
        { "union 1 split",             256, 12, 2,   4, 1,  8192, 2051, false, 1,  true },
        { "union 32 splits",           256, 12, 2,   3, 1,  8192, 2051, false, 32, true },
        { "union h128, 4/KV (8+5)",    128,  4, 2,  13, 1,  4096,  333, false, 0,  true },
        { "union h128, 8/KV, sinks",   128,  8, 4,   7, 1,  4096, 1027, true,  0,  true },
        { "union 16/KV (3+2), short",  256, 16, 1,   5, 1,  4096,   31, false, 0,  true },
        { "union one tile",            256, 12, 2,   4, 1,  4096,   32, false, 0,  true },
        { "union same cells 4",        256, 12, 2,   4, 1,  8192, 2051, false, 3,  true, 1 },
        { "union same cells 128",      256, 12, 2, 128, 1,  4096, 2051, false, 1,  true, 1 },
    };
    printf("\n%-26s %-5s %3s %5s %5s | %-21s | %-21s | %-8s %-5s\n", "union case", "type", "S", "tpb", "lst",
           "model: rel64  nmse_ggml", "gpu: rel64  nmse_ggml", "vs_tok", "bits");
    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        uint32_t seed = 4321;
        for (const fcase & c : ucases) {
            problem p;
            make_problem(p, c, type, seed++);
            const int n_cols = c.n_head_kv*c.n_q*c.ne3;
            const int S  = c.S_force > 0 ? c.S_force : fasp_pick_splits(p.ucap, c.n_head_kv*p.n_lists*c.ne3, nsm);
            const int St = c.share == 1 ? S : fasp_pick_splits(c.n_kv_max, n_cols, nsm);

            std::vector<double> r64;
            std::vector<float> rg, xu, xt, xg;
            run_fp64(p, r64);
            run_ggml(p, n_threads, rg);
            run_model(p, S, xu, true);
            run_model(p, St, xt);
            const errs em64 = compare(xu, r64), emg = compare(xu, rg), et = compare(xu, xt);
            bool ok = em64.max_rel < tol_rel && emg.nmse < tol_nmse;
            const char * bits = "-";
            if (c.share == 1) {
                const bool same = memcmp(xu.data(), xt.data(), xu.size()*sizeof(float)) == 0;
                ok = ok && same;
                bits = same ? "same" : "DIFF";
            }
            char gbuf[64] = "-";
            if (gpu) {
                run_gpu(p, S, xg, true);
                const errs eg64 = compare(xg, r64), egg = compare(xg, rg);
                ok = ok && eg64.max_rel < tol_rel && egg.nmse < tol_nmse;
                snprintf(gbuf, sizeof(gbuf), "%.2e  %.2e", eg64.max_rel, egg.nmse);
            }
            ++n_run;
            n_fail += !ok;
            printf("%-26s %-5s %3d %5d %5d | %.2e  %.2e     | %-21s | %.2e %-5s | %s\n", c.name, ggml_type_name(type), S,
                   p.tpb, p.ucounts[0], em64.max_rel, emg.nmse, gbuf, et.max_rel, bits, ok ? "pass" : "FAIL");
            fflush(stdout);
        }
    }
    printf("fasp_check: %d of %d cases pass (fault %d, nsm %d%s)\n", n_run - n_fail, n_run, g_fault, nsm, gpu ? ", gpu" : "");

    if (bench) {
        // Flash-Next's QSA layer: 2 KV heads of 12 query heads, head 256, 2,051 listed cells per query
        printf("\nbench: one QSA layer call, 2051 cells per query (lists built on the host)\n");
        printf("%-5s %7s %3s %3s %9s %9s\n", "type", "n_kv", "nq", "S", "us/call", "GB/s_kv");
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
            for (int n_kv : { 4096, 16384, 32768, 131072 }) {
                for (int nq : { 1, 4 }) {
                    const fcase c = { "bench", 256, 12, 2, nq, 1, n_kv, 2051, false, 0 };
                    problem p;
                    make_problem(p, c, type, 99);
                    const int S = fasp_pick_splits(c.n_kv_max, c.n_head_kv*nq, nsm);
                    gpu_bufs b;
                    const fasp_args a = gpu_setup(p, S, b);
                    for (int it = 0; it < 5; ++it) {
                        gpu_launch(a, p, S, 0);
                    }
                    const int iters = 200;
                    CK(cudaEventRecord(e0));
                    for (int it = 0; it < iters; ++it) {
                        gpu_launch(a, p, S, 0);
                    }
                    CK(cudaEventRecord(e1));
                    CK(cudaEventSynchronize(e1));
                    float ms = 0.0f;
                    CK(cudaEventElapsedTime(&ms, e0, e1));
                    const double us = 1000.0*ms/iters;
                    long cells = 0;
                    for (int x : p.counts) {
                        cells += x;
                    }
                    const double bytes = (double) cells*2*c.n_head_kv*p.row_bytes;
                    printf("%-5s %7d %3d %3d %9.1f %9.1f\n", ggml_type_name(type), n_kv, nq, S, us, bytes/(us*1e3));
                    gpu_free(b);
                }
            }
        }

        // the union kernel against the per-token one on the same call: rows that share 90 % of their pools (share 2, as
        // consecutive tokens of a verify or a prompt pick them) and rows that share nothing beyond chance (share 0)
        printf("\nbench: the union kernel against the per-token one, one QSA layer call, 2051 cells per query\n");
        printf("%-5s %5s %7s %4s %4s %7s %9s %9s %8s\n", "type", "share", "n_kv", "nq", "tpb", "u_cells", "us_tok",
               "us_union", "tok/uni");
        const auto time_us = [&](const problem & p, const int S, const bool uni) {
            gpu_bufs b;
            const fasp_args a = gpu_setup(p, S, b, uni);
            for (int it = 0; it < 5; ++it) {
                gpu_launch(a, p, S, 0, uni);
            }
            const int iters = p.c.n_q > 8 ? 20 : 200;
            CK(cudaEventRecord(e0));
            for (int it = 0; it < iters; ++it) {
                gpu_launch(a, p, S, 0, uni);
            }
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms = 0.0f;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            gpu_free(b);
            return 1000.0*ms/iters;
        };
        for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
            for (int share : { 2, 0 }) {
                for (int n_kv : { 8192, 32768, 49152 }) {
                    for (int nq : { 2, 4, 8, 128 }) {
                        if (share == 0 && nq != 4) {
                            continue;
                        }
                        const fcase c = { "bench", 256, 12, 2, nq, 1, n_kv, 2051, false, 0, true, share };
                        problem p;
                        make_problem(p, c, type, 99);
                        const int St = fasp_pick_splits(c.n_kv_max, c.n_head_kv*nq, nsm);
                        const int Su = fasp_pick_splits(p.ucap, c.n_head_kv*p.n_lists, nsm);
                        const double ut = time_us(p, St, false), uu = time_us(p, Su, true);
                        long cells = 0;
                        for (int x : p.ucounts) {
                            cells += x;
                        }
                        printf("%-5s %5d %7d %4d %4d %7ld %9.1f %9.1f %8.2f\n", ggml_type_name(type), share, n_kv, nq, p.tpb,
                               cells/p.n_lists, ut, uu, ut/uu);
                    }
                }
            }
        }
    }
    return n_fail == 0 ? 0 : 1;
}
