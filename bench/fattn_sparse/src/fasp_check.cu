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
//
// Usage: fasp_check [--gpu] [--bench] [--fault n] [--nsm n] [--threads n]
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
    int  S_force; // 0: fasp_pick_splits
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
    std::vector<uint8_t> sel(c.n_kv);
    for (int i3 = 0; i3 < c.ne3; ++i3) {
        for (int iq = 0; iq < c.n_q; ++iq) {
            const int row = i3*c.n_q + iq;
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
}

static fasp_args make_args(const problem & p, const char * Q, const char * K, const char * V, const half * mask,
        const int32_t * idx, const int32_t * counts, const float * sinks, float * dst, float * part, unsigned * counter) {
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
    a.n_kv_max = c.n_kv_max;
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

template <int D, int type>
static void model_launch(const fasp_args & a, const int S, const int n_head_kv, const int n_z) {
    constexpr int DPL  = D/FASP_WARP;
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

                    float * part = a.part + (int64_t) s*split_stride + ((int64_t) col*a.G + FASP_HPW*w)*PART;
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

                // the last block of the column merges (blocks run in order here, so it is split S - 1)
                if (++done[col] != (unsigned) S) {
                    continue;
                }
                for (int w = 0; w < NW; ++w) {
                    for (int h = 0; h < FASP_HPW; ++h) {
                        const int hq = g*a.G + FASP_HPW*w + h;
                        for (int lane = 0; lane < 32; ++lane) {
                            float o[DPL];
                            float m_all, l;
                            fasp_merge_lane<DPL>(a.part + ((int64_t) col*a.G + FASP_HPW*w + h)*PART, split_stride, S, D, lane, o, m_all, l);
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
            }
        }
    }
}

static void run_model(const problem & p, const int S, std::vector<float> & out) {
    const fcase & c = p.c;
    out.assign((size_t) c.ne3*c.n_q*p.n_head*c.D, 0.0f);
    const int n_cols = c.n_head_kv*c.n_q*c.ne3;
    std::vector<float> part((size_t) S*n_cols*c.G*fasp_part_floats(c.D));
    const fasp_args a = make_args(p, (const char *) p.q.data(), (const char *) p.k.data(), (const char *) p.v.data(),
        (const half *) p.mask.data(), p.idx.data(), p.counts.data(), p.sinks.data(), out.data(), part.data(), nullptr);
    const int n_z = c.n_q*c.ne3;
    if (c.D == 256) {
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

static void gpu_launch(const fasp_args & a, const problem & p, const int S, cudaStream_t st) {
    const fcase & c = p.c;
    const int n_z = c.n_q*c.ne3;
    if (c.D == 256) {
        p.type == GGML_TYPE_F16 ? fasp_launch<256, FASP_F16>(a, S, c.n_head_kv, n_z, st) : fasp_launch<256, FASP_Q8_0>(a, S, c.n_head_kv, n_z, st);
    } else {
        p.type == GGML_TYPE_F16 ? fasp_launch<128, FASP_F16>(a, S, c.n_head_kv, n_z, st) : fasp_launch<128, FASP_Q8_0>(a, S, c.n_head_kv, n_z, st);
    }
}

static fasp_args gpu_setup(const problem & p, const int S, gpu_bufs & b) {
    const fcase & c = p.c;
    const int n_cols = c.n_head_kv*c.n_q*c.ne3;
    CK(cudaMalloc(&b.q, p.q.size()*sizeof(float)));
    CK(cudaMalloc(&b.k, p.k.size()));
    CK(cudaMalloc(&b.v, p.v.size()));
    CK(cudaMalloc(&b.mask, p.mask.size()*sizeof(half)));
    CK(cudaMalloc(&b.idx, p.idx.size()*sizeof(int32_t)));
    CK(cudaMalloc(&b.counts, p.counts.size()*sizeof(int32_t)));
    CK(cudaMalloc(&b.sinks, p.sinks.size()*sizeof(float)));
    CK(cudaMalloc(&b.dst, p.q.size()*sizeof(float)));
    CK(cudaMalloc(&b.part, (size_t) S*n_cols*c.G*fasp_part_floats(c.D)*sizeof(float)));
    CK(cudaMalloc(&b.counter, FASP_MAX_COLS*sizeof(unsigned)));
    CK(cudaMemcpy(b.q, p.q.data(), p.q.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.k, p.k.data(), p.k.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.v, p.v.data(), p.v.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.mask, p.mask.data(), p.mask.size()*sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.idx, p.idx.data(), p.idx.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.counts, p.counts.data(), p.counts.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.sinks, p.sinks.data(), p.sinks.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemset(b.counter, 0, FASP_MAX_COLS*sizeof(unsigned)));
    CK(cudaMemset(b.dst, 0xFF, p.q.size()*sizeof(float)));
    return make_args(p, b.q, b.k, b.v, b.mask, b.idx, b.counts, b.sinks, b.dst, b.part, b.counter);
}

static void gpu_free(gpu_bufs & b) {
    for (void * x : { (void *) b.q, (void *) b.k, (void *) b.v, (void *) b.mask, (void *) b.idx, (void *) b.counts,
                      (void *) b.sinks, (void *) b.dst, (void *) b.part, (void *) b.counter }) {
        CK(cudaFree(x));
    }
}

static void run_gpu(const problem & p, const int S, std::vector<float> & out) {
    gpu_bufs b;
    const fasp_args a = gpu_setup(p, S, b);
    gpu_launch(a, p, S, 0);
    CK(cudaGetLastError());
    // twice: the counters must be back at zero after a launch
    gpu_launch(a, p, S, 0);
    CK(cudaDeviceSynchronize());
    out.resize(p.q.size());
    CK(cudaMemcpy(out.data(), b.dst, out.size()*sizeof(float), cudaMemcpyDeviceToHost));
    gpu_free(b);
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
        } else {
            fprintf(stderr, "usage: %s [--gpu] [--bench] [--fault n] [--nsm n] [--threads n]\n", argv[0]);
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
    }
    return n_fail == 0 ? 0 : 1;
}
