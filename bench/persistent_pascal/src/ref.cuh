// CPU reference for one MoE layer in double precision, and the stage-by-stage checks of a path's intermediates.
#pragma once

#include "persistent.cuh"
#include "../../mmvq_pascal/src/data.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace pk {

inline int row_bytes_of(const int type, const int K) {
    return (int) mb::row_bytes(type, K);
}

// One row of K weights (ggml block layouts) as doubles.
static void deq_row(const int type, const uint8_t * row, const int K, double * out) {
    const int bs = mb::type_bs(type);
    for (int ib = 0; ib < K / 32; ++ib) {
        const uint8_t * b = row + (size_t) ib * bs;
        double * o = out + 32 * ib;
        switch (type) {
            case mb::T_Q8_0: {
                const double d = mb::half_bits_to_float(mb::rd_u16(b));
                for (int j = 0; j < 32; ++j) o[j] = d * (int8_t) b[2 + j];
            } break;
            case mb::T_Q5_0: {
                const double d = mb::half_bits_to_float(mb::rd_u16(b));
                const uint32_t qh = (uint32_t) b[2] | ((uint32_t) b[3] << 8) | ((uint32_t) b[4] << 16) | ((uint32_t) b[5] << 24);
                for (int j = 0; j < 16; ++j) {
                    o[j]      = d * ((int) ((b[6 + j] & 15) | (((qh >> j) & 1) << 4)) - 16);
                    o[j + 16] = d * ((int) ((b[6 + j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16);
                }
            } break;
            case mb::T_Q4_0: {
                const double d = mb::half_bits_to_float(mb::rd_u16(b));
                for (int j = 0; j < 16; ++j) {
                    o[j]      = d * ((b[2 + j] & 15) - 8);
                    o[j + 16] = d * ((b[2 + j] >> 4) - 8);
                }
            } break;
            case mb::T_Q4_1: {
                const double d = mb::half_bits_to_float(mb::rd_u16(b));
                const double m = mb::half_bits_to_float(mb::rd_u16(b + 2));
                for (int j = 0; j < 16; ++j) {
                    o[j]      = d * (b[4 + j] & 15) + m;
                    o[j + 16] = d * (b[4 + j] >> 4) + m;
                }
            } break;
            default:
                fprintf(stderr, "deq_row: unsupported type\n");
                exit(1);
        }
    }
}

// The dequantization above against mmvq_pascal's ref_dot (an independent implementation, checked against the same
// kernels there), on random rows of K weights and random activations whose block sums are exact. Returns the worst
// |difference| / sum of |terms|.
static double ref_selftest(const int type, const int K, const uint64_t seed) {
    uint64_t st = seed * 0x9E3779B97F4A7C15ull + 1;
    auto rnd = [&]() { st = st * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t) (st >> 33); };
    const int bs = mb::type_bs(type);
    std::vector<uint8_t> row((size_t) K / 32 * bs);
    for (auto & b : row) b = (uint8_t) rnd();
    for (int ib = 0; ib < K / 32; ++ib) {
        uint8_t * b = row.data() + (size_t) ib * bs;
        const uint16_t d = (uint16_t) (0x2000 | (rnd() & 0x3ff));  // FP16 in [2^-7, 2^-6)
        b[0] = d & 0xff; b[1] = d >> 8;
        if (type == mb::T_Q4_1) {
            const uint16_t m = (uint16_t) (0xA400 | (rnd() & 0x3ff)); // negative minimum
            b[2] = m & 0xff; b[3] = m >> 8;
        }
        if (type == mb::T_Q8_0) {
            for (int j = 0; j < 32; ++j) if (b[2 + j] == 0x80) b[2 + j] = 0x81;
        }
    }
    mb::act_host a;
    a.K = K; a.nvec = 1;
    a.q.resize(K); a.d.resize(K / 32); a.s.resize(K / 32);
    std::vector<double> ad(K);
    for (int ib = 0; ib < K / 32; ++ib) {
        a.d[ib] = 0.01f + (rnd() & 0xff) * 1e-4f;
        int sum = 0;
        for (int j = 0; j < 32; ++j) {
            a.q[32 * ib + j] = (int8_t) ((int) (rnd() % 255) - 127);
            sum += a.q[32 * ib + j];
            ad[32 * ib + j] = (double) a.d[ib] * a.q[32 * ib + j];
        }
        a.s[ib] = a.d[ib] * (float) sum;
    }
    std::vector<double> w(K);
    deq_row(type, row.data(), K, w.data());
    double mine = 0.0, absum = 0.0;
    for (int k = 0; k < K; ++k) {
        mine += w[k] * ad[k];
        absum += fabs(w[k] * ad[k]);
    }
    double absum2 = 0.0;
    const double theirs = mb::ref_dot(type, row.data(), a, 0, &absum2);
    return fabs(mine - theirs) / std::max(absum, 1e-30);
}

// Q8_1 blocks as the activations the fork's kernels multiply with: d8 * q (the block sum is taken from the int8 values).
static std::vector<double> deq_q8_1(const block_q8_1 * b, const int n) {
    std::vector<double> out((size_t) n * 32);
    for (int ib = 0; ib < n; ++ib) {
        const double d = mb::half_bits_to_float(mb::rd_u16((const uint8_t *) &b[ib]));
        for (int j = 0; j < 32; ++j) out[(size_t) 32 * ib + j] = d * b[ib].qs[j];
    }
    return out;
}

template <typename TW>
static double dotd(const TW * w, const double * a, const int K, double * absum) {
    double s = 0.0, as = 0.0;
    for (int k = 0; k < K; ++k) {
        const double t = w[k] * a[k];
        s += t;
        as += fabs(t);
    }
    if (absum) *absum = as;
    return s;
}

static double silud(const double x) {
    return x / (1.0 + exp(-x));
}

// Worst error relative to its tolerance, per stage.
struct stage_stat {
    std::string name;
    double worst = 0.0;   // max err/tol
    long   n = 0, fails = 0, notes = 0;
    int    at0 = -1, at1 = -1;      // where the worst one is (stage-specific: expert slot / block, row / value)
    double got = 0.0, want = 0.0;   // and its values
    void add(const double err, const double tol, const int a0 = -1, const int a1 = -1, const double g = 0.0, const double w = 0.0) {
        const double r = tol > 0 ? err / tol : (err > 0 ? INFINITY : 0.0);
        if (!(r <= worst)) {
            worst = r; at0 = a0; at1 = a1; got = g; want = w;
        }
        ++n;
        if (!(r <= 1.0)) ++fails;
    }
};

struct check_report {
    std::vector<stage_stat> st;
    stage_stat & get(const std::string & name) {
        for (auto & s : st) if (s.name == name) return s;
        st.push_back(stage_stat{name});
        return st.back();
    }
    bool ok() const {
        for (auto & s : st) if (s.fails) return false;
        return true;
    }
};

// Host copies of one path's results for one layer evaluation.
struct host_out {
    std::vector<float> xn, logits, weights, act, dexp, moe, sact, sd, out;
    std::vector<int32_t> ids;
    std::vector<block_q8_1> yq, aq, sq;
    void fetch(const layer_bufs & b, const float * xn_dev, const block_q8_1 * yq_dev) {
        auto get = [](auto & v, const auto * p, size_t n) {
            v.resize(n);
            CUDA_CHECK(cudaMemcpy(v.data(), p, n * sizeof(v[0]), cudaMemcpyDeviceToHost));
        };
        get(xn, xn_dev, H);
        get(yq, yq_dev, H / 32);
        get(logits, b.logits, NE);
        get(ids, b.ids, KU);
        get(weights, b.weights, KU);
        get(act, b.act, KU * F);
        get(aq, b.aq, KU * F / 32);
        get(dexp, b.dexp, (size_t) KU * H);
        get(moe, b.moe, H);
        get(sact, b.sact, FS);
        get(sq, b.sq, FS / 32);
        get(sd, b.sd, H);
        get(out, b.out, H);
    }
};

// One layer's weights on the host, fetched from the device as needed.
struct host_layer {
    int tgu = 0, tdn = 0, tsh = 0;
    bool router_f32 = false;
    std::vector<float>  norm_w;
    std::vector<double> router;                    // NE x H
    // dequantized weights; float holds them exactly (FP16 scale times a small integer)
    std::vector<float> gate_sh, up_sh, down_sh;
    std::map<int, std::vector<float>> gate_e, up_e, down_e;
    layer_weights dev;

    void load(const layer_weights & w, const int tgu_, const int tdn_, const int tsh_, const bool router_f32_) {
        dev = w; tgu = tgu_; tdn = tdn_; tsh = tsh_; router_f32 = router_f32_;
        norm_w.resize(H);
        CUDA_CHECK(cudaMemcpy(norm_w.data(), w.norm_w, H * sizeof(float), cudaMemcpyDeviceToHost));
        router.resize((size_t) NE * H);
        if (router_f32) {
            std::vector<float> r((size_t) NE * H);
            CUDA_CHECK(cudaMemcpy(r.data(), w.router, r.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < r.size(); ++i) router[i] = r[i];
        } else {
            std::vector<uint16_t> r((size_t) NE * H);
            CUDA_CHECK(cudaMemcpy(r.data(), w.router, r.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < r.size(); ++i) router[i] = mb::half_bits_to_float(r[i]);
        }
        gate_sh = deq_matrix(w.gate_sh, tsh, FS, H);
        up_sh   = deq_matrix(w.up_sh, tsh, FS, H);
        down_sh = deq_matrix(w.down_sh, tsh, H, FS);
    }
    static std::vector<float> deq_matrix(const char * dev_ptr, const int type, const int nrows, const int K) {
        const int rb = row_bytes_of(type, K);
        std::vector<uint8_t> raw((size_t) nrows * rb);
        CUDA_CHECK(cudaMemcpy(raw.data(), dev_ptr, raw.size(), cudaMemcpyDeviceToHost));
        std::vector<float> out((size_t) nrows * K);
        std::vector<double> row(K);
        for (int r = 0; r < nrows; ++r) {
            deq_row(type, raw.data() + (size_t) r * rb, K, row.data());
            for (int k = 0; k < K; ++k) out[(size_t) r * K + k] = (float) row[k];
        }
        return out;
    }
    void drop_experts() {
        gate_e.clear();
        up_e.clear();
        down_e.clear();
    }
    void need_expert(const int id) {
        if (gate_e.count(id)) return;
        const size_t gb = (size_t) F * row_bytes_of(tgu, H);
        const size_t db = (size_t) H * row_bytes_of(tdn, F);
        gate_e[id] = deq_matrix(dev.gate_exps + id * gb, tgu, F, H);
        up_e[id]   = deq_matrix(dev.up_exps + id * gb, tgu, F, H);
        down_e[id] = deq_matrix(dev.down_exps + id * db, tdn, H, F);
    }
};

// A Q8_1 quantization of src (n values) checked against the same rounding in double: d = amax / 127 stored as FP16,
// q = round(x / d) with d before FP16 rounding; q may differ by one only where x / d lies within 1e-4 of a half-integer
// (the GPU divides approximately).
static void check_quant(stage_stat & st, const float * src, const block_q8_1 * q, const int n) {
    for (int ib = 0; ib < n / 32; ++ib) {
        double amax = 0.0;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, (double) fabsf(src[32 * ib + j]));
        const double d_ref = amax / 127.0;
        const double d_gpu = mb::half_bits_to_float(mb::rd_u16((const uint8_t *) &q[ib]));
        st.add(fabs(d_gpu - d_ref), d_ref * (1.0 / 2048.0) * 1.01 + 1e-30);
        for (int j = 0; j < 32; ++j) {
            const double t = amax == 0.0 ? 0.0 : src[32 * ib + j] / d_ref;
            const double qr = std::round(t);
            const int qg = q[ib].qs[j];
            const double diff = fabs(qg - qr);
            const bool near_half = fabs(fabs(t - std::trunc(t)) - 0.5) < 1e-4;
            if (diff == 0.0) {
                st.add(0.0, 1.0);
            } else if (diff == 1.0 && near_half) {
                st.add(0.0, 1.0);
                ++st.notes;
            } else {
                st.add(diff, 0.5);
            }
        }
    }
}

// Stage checks of one path. persistent: sd holds the shared expert alone (else the shared expert plus the routed sum,
// and moe is checked too).
static void check_stages(check_report & rep, host_layer & L, const std::vector<float> & x, const host_out & o, const bool persistent) {
    // RMS norm
    {
        auto & st = rep.get("norm");
        double ss = 0.0;
        for (int i = 0; i < H; ++i) ss += (double) x[i] * x[i];
        const double scale = 1.0 / sqrt(ss / H + 1e-6);
        for (int i = 0; i < H; ++i) {
            const double ref = x[i] * scale * L.norm_w[i];
            st.add(fabs(o.xn[i] - ref), 4e-6 * fabs(ref) + 1e-30);
        }
    }
    check_quant(rep.get("quant xn"), o.xn.data(), o.yq.data(), H);
    // router
    {
        auto & st = rep.get("router");
        for (int r = 0; r < NE; ++r) {
            double s = 0.0, as = 0.0;
            for (int k = 0; k < H; ++k) {
                const double t = L.router[(size_t) r * H + k] * o.xn[k];
                s += t;
                as += fabs(t);
            }
            st.add(fabs(o.logits[r] - s), 1e-5 * as + 1e-30, r, -1, o.logits[r], s);
        }
    }
    // top-6: the same set as a double-precision sort of the GPU's logits (order and near-ties tolerated), and the
    // normalized weights of the experts the GPU chose
    {
        auto & st = rep.get("top-6");
        std::vector<double> sig(NE);
        for (int i = 0; i < NE; ++i) sig[i] = 1.0 / (1.0 + exp(-(double) o.logits[i]));
        std::vector<int> idx(NE);
        for (int i = 0; i < NE; ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return sig[a] > sig[b]; });
        const double cut = sig[idx[KU - 1]];
        for (int k = 0; k < KU; ++k) {
            const int id = o.ids[k];
            const bool in_ref = std::find(idx.begin(), idx.begin() + KU, id) != idx.begin() + KU;
            if (in_ref) {
                st.add(0.0, 1.0);
            } else if (id >= 0 && id < NE && fabs(sig[id] - cut) <= 1e-6 * cut) {
                st.add(0.0, 1.0);
                ++st.notes;
            } else {
                st.add(1.0, 0.5);
            }
            if (k > 0 && id >= 0 && id < NE && o.ids[k - 1] >= 0 && o.ids[k - 1] < NE) {
                st.add(std::max(0.0, sig[id] - sig[o.ids[k - 1]]), 1e-6 * sig[id] + 1e-30);  // descending order
            }
        }
        auto & sw = rep.get("weights");
        double sum = 0.0;
        for (int k = 0; k < KU; ++k) sum += sig[std::clamp(o.ids[k], 0, NE - 1)];
        sum = std::max(sum, 6.103515625e-5);
        for (int k = 0; k < KU; ++k) {
            const double ref = sig[std::clamp(o.ids[k], 0, NE - 1)] / sum;
            sw.add(fabs(o.weights[k] - ref), 1e-5 * ref + 1e-30);
        }
    }
    // routed gate/up + SwiGLU, from the GPU's Q8_1 activations
    const std::vector<double> ayq = deq_q8_1(o.yq.data(), H / 32);
    {
        auto & st = rep.get("routed gate/up");
        for (int e = 0; e < KU; ++e) {
            const int id = std::clamp(o.ids[e], 0, NE - 1);
            L.need_expert(id);
            for (int r = 0; r < F; ++r) {
                double ag, au;
                const double g = dotd(L.gate_e[id].data() + (size_t) r * H, ayq.data(), H, &ag);
                const double u = dotd(L.up_e[id].data() + (size_t) r * H, ayq.data(), H, &au);
                const double ref = u * silud(g);
                const double tol = 1e-5 * (au * fabs(silud(g)) + 1.1 * ag * fabs(u)) + 1e-6 * fabs(ref) + 1e-30;
                st.add(fabs(o.act[e * F + r] - ref), tol, e, r, o.act[e * F + r], ref);
            }
        }
    }
    check_quant(rep.get("quant routed"), o.act.data(), o.aq.data(), KU * F);
    {
        auto & st = rep.get("routed down");
        for (int e = 0; e < KU; ++e) {
            const int id = std::clamp(o.ids[e], 0, NE - 1);
            const std::vector<double> a = deq_q8_1(o.aq.data() + e * (F / 32), F / 32);
            for (int r = 0; r < H; ++r) {
                double as;
                const double ref = dotd(L.down_e[id].data() + (size_t) r * F, a.data(), F, &as);
                st.add(fabs(o.dexp[(size_t) e * H + r] - ref), 1e-5 * as + 1e-30, e, r, o.dexp[(size_t) e * H + r], ref);
            }
        }
    }
    // shared expert
    {
        auto & st = rep.get("shared gate/up");
        for (int r = 0; r < FS; ++r) {
            double ag, au;
            const double g = dotd(L.gate_sh.data() + (size_t) r * H, ayq.data(), H, &ag);
            const double u = dotd(L.up_sh.data() + (size_t) r * H, ayq.data(), H, &au);
            const double ref = u * silud(g);
            const double tol = 1e-5 * (au * fabs(silud(g)) + 1.1 * ag * fabs(u)) + 1e-6 * fabs(ref) + 1e-30;
            st.add(fabs(o.sact[r] - ref), tol, -1, r, o.sact[r], ref);
        }
    }
    check_quant(rep.get("quant shared"), o.sact.data(), o.sq.data(), FS);
    std::vector<double> sd_ref(H), sd_abs(H);
    {
        const std::vector<double> a = deq_q8_1(o.sq.data(), FS / 32);
        for (int r = 0; r < H; ++r) sd_ref[r] = dotd(L.down_sh.data() + (size_t) r * FS, a.data(), FS, &sd_abs[r]);
    }
    // weighted sum, shared expert, residual
    {
        auto & ss = rep.get("shared down (+sum)");
        auto & so = rep.get("output");
        for (int r = 0; r < H; ++r) {
            double m = 0.0, ma = 0.0;
            for (int e = 0; e < KU; ++e) {
                const double t = (double) o.weights[e] * o.dexp[(size_t) e * H + r];
                m += t;
                ma += fabs(t);
            }
            if (persistent) {
                ss.add(fabs(o.sd[r] - sd_ref[r]), 1e-5 * sd_abs[r] + 1e-30);
                const double ref = (double) x[r] + o.sd[r] + m;
                so.add(fabs(o.out[r] - ref), 1e-6 * (fabs(x[r]) + fabs(o.sd[r]) + ma) + 1e-30);
            } else {
                rep.get("weighted sum").add(fabs(o.moe[r] - m), 1e-6 * ma + 1e-30);
                ss.add(fabs(o.sd[r] - (sd_ref[r] + o.moe[r])), 1e-5 * sd_abs[r] + 1e-6 * fabs(o.moe[r]) + 1e-30);
                const double ref = (double) x[r] + o.sd[r];
                so.add(fabs(o.out[r] - ref), 2.4e-7 * (fabs(x[r]) + fabs(o.sd[r])) + 1e-30);
            }
        }
    }
}

// The whole layer in double precision from x, without quantizing any activation; returns the output and the experts
// it chose. Compared with a GPU output, this measures the error of the Q8_1 activations.
static std::vector<double> ref_layer_exact(host_layer & L, const std::vector<float> & x, std::vector<int> & ids_out,
                                           double * margin) {
    std::vector<double> xn(H);
    double ss = 0.0;
    for (int i = 0; i < H; ++i) ss += (double) x[i] * x[i];
    const double scale = 1.0 / sqrt(ss / H + 1e-6);
    for (int i = 0; i < H; ++i) xn[i] = x[i] * scale * L.norm_w[i];
    std::vector<double> sig(NE);
    for (int r = 0; r < NE; ++r) sig[r] = 1.0 / (1.0 + exp(-dotd(L.router.data() + (size_t) r * H, xn.data(), H, (double *) nullptr)));
    std::vector<int> idx(NE);
    for (int i = 0; i < NE; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return sig[a] > sig[b]; });
    *margin = (sig[idx[KU - 1]] - sig[idx[KU]]) / sig[idx[KU - 1]];
    double wsum = 0.0;
    for (int k = 0; k < KU; ++k) wsum += sig[idx[k]];
    std::vector<double> out(H);
    for (int i = 0; i < H; ++i) out[i] = x[i];
    ids_out.assign(idx.begin(), idx.begin() + KU);
    std::vector<double> act(F);
    for (int k = 0; k < KU; ++k) {
        const int id = idx[k];
        L.need_expert(id);
        for (int r = 0; r < F; ++r) {
            const double g = dotd(L.gate_e[id].data() + (size_t) r * H, xn.data(), H, (double *) nullptr);
            const double u = dotd(L.up_e[id].data() + (size_t) r * H, xn.data(), H, (double *) nullptr);
            act[r] = u * silud(g);
        }
        const double w = sig[id] / wsum;
        for (int r = 0; r < H; ++r) out[r] += w * dotd(L.down_e[id].data() + (size_t) r * F, act.data(), F, (double *) nullptr);
    }
    std::vector<double> sact(FS);
    for (int r = 0; r < FS; ++r) {
        const double g = dotd(L.gate_sh.data() + (size_t) r * H, xn.data(), H, (double *) nullptr);
        const double u = dotd(L.up_sh.data() + (size_t) r * H, xn.data(), H, (double *) nullptr);
        sact[r] = u * silud(g);
    }
    for (int r = 0; r < H; ++r) out[r] += dotd(L.down_sh.data() + (size_t) r * FS, sact.data(), FS, (double *) nullptr);
    return out;
}

} // namespace pk
