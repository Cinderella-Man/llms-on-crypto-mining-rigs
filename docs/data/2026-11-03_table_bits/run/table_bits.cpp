// table_bits: Flash-Next's n-gram table (`per_layer_token_embd`, 160-wide IQ4_NL rows in 16 hash heads) on the CPU.
//   table_bits census INFO [NORMS_OUT]        per-head row-norm census (stdout: census TSV; histogram to $HIST)
//   table_bits roundtrip INFO FMT OUT_SHARD   decode each row, round-trip it through FMT, encode the result back to
//                                             IQ4_NL with ggml's own rules, and write a copy of the shard with only
//                                             the table's bytes changed (stdout: per-head error TSV)
// INFO is table_info.py's output. FMT:
//   iq4nl   no format: the decoded row re-encoded as is (the control: how far a re-encode alone moves the table)
//   zero    every row zero (the bound: what the table is worth at all)
//   q<b>    b bits per weight on a per-head codebook of 2^b levels, times a per-head column scale, times an fp16 row
//           scale: 160 b + 16 bits a row (q2: 2.1 bits per weight, q1: 1.1, the sign bits)
//   bf<r>   r sign bits a row on a per-head real basis of r vectors, times an fp16 row scale: r + 16 bits a row
//           (bf64: 0.5 bits per weight, bf32: 0.3)
//   <fmt>-q8  the same, written back in Q8_0 instead (8.5 bits, a 54.4 GB shard; the header's type field patched), so
//           that the re-encode adds ~1e-5 instead of ~1e-2 of relative error: for the 3- and 4-bit formats, where an
//           IQ4_NL re-encode would add about as much error as the format itself
// The per-head parameters (codebook and column scales, or the basis) are fitted on $SAMPLE rows of each head
// (default 262144, evenly spaced) by alternating least squares, then every row is fitted on its own. Rows outside the
// heads (the table's 90 padding rows) are copied unchanged. Build: g++ -O3 -march=native -fopenmp.
#include <immintrin.h>
#include <omp.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static const int8_t KV[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
constexpr int W = 160, NB = W / 32, RB = NB * 18, MAXR = 128, MAXL = 16;

static inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
static inline uint16_t f2h(float f) { return _cvtss_sh(f, 0); }
static inline float r16(float f) { return h2f(f2h(f)); }
static double now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static long envl(const char * k, long d) { const char * v = getenv(k); return v ? atol(v) : d; }

struct Info {
    std::string file;
    uint64_t offset = 0, rows = 0, end = 0, file_size = 0, type_pos = 0;
    std::vector<uint64_t> ho, hn;
    int head_of(uint64_t r) const {  // -1 for rows outside every head
        for (size_t h = 0; h < ho.size(); ++h) if (r >= ho[h] && r < ho[h] + hn[h]) return (int) h;
        return -1;
    }
};

static Info read_info(const char * path) {
    Info in;
    std::ifstream f(path);
    std::string line, k;
    while (std::getline(f, line)) {
        std::istringstream s(line);
        s >> k;
        if (k == "file") s >> in.file;
        else if (k == "offset") s >> in.offset;
        else if (k == "rows") s >> in.rows;
        else if (k == "end") s >> in.end;
        else if (k == "file_size") s >> in.file_size;
        else if (k == "type_pos") s >> in.type_pos;
        else if (k == "type") { std::string t; s >> t; if (t != "IQ4_NL") { fprintf(stderr, "type %s\n", t.c_str()); exit(1); } }
        else if (k == "ne0") { int n; s >> n; if (n != W) { fprintf(stderr, "ne0 %d\n", n); exit(1); } }
        else if (k == "head") { int h; uint64_t o, n; s >> h >> o >> n; in.ho.push_back(o); in.hn.push_back(n); }
    }
    if (in.end != in.offset + in.rows * RB) { fprintf(stderr, "bad info\n"); exit(1); }
    return in;
}

static const uint8_t * map_table(const Info & in) {
    int fd = open(in.file.c_str(), O_RDONLY);
    if (fd < 0) { perror(in.file.c_str()); exit(1); }
    void * p = mmap(nullptr, in.file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    madvise(p, in.file_size, MADV_SEQUENTIAL);
    return (const uint8_t *) p;
}

static void dequant_row(const uint8_t * src, float * x) {
    for (int b = 0; b < NB; ++b) {
        const uint8_t * blk = src + 18 * b;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = h2f(dh);
        for (int j = 0; j < 16; ++j) {
            x[32 * b + j]      = d * KV[blk[2 + j] & 15];
            x[32 * b + 16 + j] = d * KV[blk[2 + j] >> 4];
        }
    }
}

// ggml's best_index_int8 over the IQ4_NL values, as a count of the midpoints at or below x (ties go up, as there)
static const float MID[15] = {-115.5f, -93.5f, -74, -57, -42, -28.5f, -16, -4.5f, 7, 19, 31.5f, 45.5f, 61, 79, 101};
static inline int best_index(float x) {
    int l = 0;
    for (int i = 0; i < 15; ++i) l += x >= MID[i];
    return l;
}

// ggml's quantize_row_iq4_nl_impl for one 32-weight block, no importance matrix (weights x^2, ntry = 7)
static void quant_block(const float * x, uint8_t * out) {
    float wt[32], amax = 0, max = 0;
    uint8_t L[32];
    for (int j = 0; j < 32; ++j) {
        wt[j] = x[j] * x[j];
        const float ax = fabsf(x[j]);
        if (ax > amax) { amax = ax; max = x[j]; }
    }
    float d = 0;
    if (amax >= 1e-15f) {
        d = -max / KV[0];
        float id = 1 / d, sumqx = 0, sumq2 = 0;
        for (int j = 0; j < 32; ++j) {
            const float q = KV[best_index(id * x[j])];
            sumqx += wt[j] * q * x[j];
            sumq2 += wt[j] * q * q;
        }
        d = sumqx / sumq2;
        float best = d * sumqx;
        for (int itry = -7; itry <= 7; ++itry) {
            id = (itry + KV[0]) / max;
            sumqx = sumq2 = 0;
            for (int j = 0; j < 32; ++j) {
                const float q = KV[best_index(id * x[j])];
                sumqx += wt[j] * q * x[j];
                sumq2 += wt[j] * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) { d = sumqx / sumq2; best = d * sumqx; }
        }
    }
    const uint16_t dh = f2h(d);
    memcpy(out, &dh, 2);
    const float id = d ? 1 / d : 0;
    for (int j = 0; j < 32; ++j) L[j] = best_index(id * x[j]);
    for (int j = 0; j < 16; ++j) out[2 + j] = L[j] | (L[16 + j] << 4);
}

// ggml's quantize_row_q8_0_ref for one 32-weight block, and its decode
constexpr int RB8 = NB * 34;
static void quant_block_q8(const float * x, uint8_t * out) {
    float amax = 0;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, fabsf(x[j]));
    const float d = amax / 127, id = d ? 1 / d : 0;
    const uint16_t dh = f2h(d);
    memcpy(out, &dh, 2);
    for (int j = 0; j < 32; ++j) out[2 + j] = (uint8_t) (int8_t) roundf(x[j] * id);
}
static void dequant_row_q8(const uint8_t * src, float * x) {
    for (int b = 0; b < NB; ++b) {
        uint16_t dh;
        memcpy(&dh, src + 34 * b, 2);
        const float d = h2f(dh);
        for (int j = 0; j < 32; ++j) x[32 * b + j] = d * (int8_t) src[34 * b + 2 + j];
    }
}

// ---------------------------------------------------------------------------------------------------------- census

constexpr int HB = 6000;              // histogram of log10(row norm), 0.002 wide bins from 1e-8
constexpr double HLO = -8, HW = 0.002;

struct Census {
    uint64_t rows = 0, zero = 0;
    double n1 = 0, n2 = 0, z4 = 0, zmax = 0, maxn = 0, minn = 1e300;
    double cs[W] = {}, cq[W] = {};
    std::vector<uint64_t> hc = std::vector<uint64_t>(HB);
    std::vector<double> he = std::vector<double>(HB);
    void add(const Census & o) {
        rows += o.rows; zero += o.zero; n1 += o.n1; n2 += o.n2; z4 += o.z4; zmax += o.zmax;
        maxn = std::max(maxn, o.maxn); minn = std::min(minn, o.minn);
        for (int j = 0; j < W; ++j) { cs[j] += o.cs[j]; cq[j] += o.cq[j]; }
        for (int b = 0; b < HB; ++b) { hc[b] += o.hc[b]; he[b] += o.he[b]; }
    }
};

static int census(const Info & in, const char * norms_out) {
    const uint8_t * base = map_table(in) + in.offset;
    const int nh = (int) in.ho.size();
    std::vector<Census> tot(nh + 1);  // last: rows outside the heads
    std::vector<uint16_t> norms(norms_out ? in.rows : 0);
    const double t0 = now();
#pragma omp parallel
    {
        std::vector<Census> loc(nh + 1);
        float x[W];
#pragma omp for schedule(static, 65536)
        for (uint64_t r = 0; r < in.rows; ++r) {
            int h = in.head_of(r);
            Census & c = loc[h < 0 ? nh : h];
            dequant_row(base + r * RB, x);
            double s2 = 0, mx = 0;
            for (int j = 0; j < W; ++j) { s2 += (double) x[j] * x[j]; mx = std::max(mx, (double) fabsf(x[j])); }
            const double n = sqrt(s2);
            if (norms_out) norms[r] = f2h((float) n);
            c.rows++;
            if (s2 == 0) { c.zero++; continue; }
            c.n1 += n; c.n2 += s2; c.maxn = std::max(c.maxn, n); c.minn = std::min(c.minn, n);
            const double rms = n / sqrt((double) W);
            for (int j = 0; j < W; ++j) {
                const double z = x[j] / rms;
                c.z4 += z * z * z * z;
                c.cs[j] += x[j]; c.cq[j] += (double) x[j] * x[j];
            }
            c.zmax += mx / rms;
            int b = (int) floor((log10(n) - HLO) / HW);
            b = std::min(std::max(b, 0), HB - 1);
            c.hc[b]++; c.he[b] += s2;
        }
#pragma omp critical
        for (int h = 0; h <= nh; ++h) tot[h].add(loc[h]);
    }
    fprintf(stderr, "census: %.1f s\n", now() - t0);
    Census all;
    for (int h = 0; h < nh; ++h) all.add(tot[h]);
    tot.push_back(all);
    // per head: quantiles of the row norm (from the histogram: bin centres), share of the energy below them
    printf("head\trows\tzero_rows\tmean_norm\trms_norm\tmin_norm\tp0.1\tp1\tp10\tp50\tp90\tp99\tp99.9\tmax_norm"
           "\tenergy_below_p10\tenergy_below_p50\tkurtosis\tmean_max_z\tcol_mean_share\tcol_rms_min\tcol_rms_max\n");
    const double qs[] = {0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999};
    for (int h = 0; h <= nh + 1; ++h) {
        const Census & c = tot[h];
        if (!c.rows) continue;
        const uint64_t nz = c.rows - c.zero;
        std::string name = h < nh ? std::to_string(h) : h == nh ? "pad" : "all";
        printf("%s\t%lu\t%lu\t%.5g\t%.5g\t%.5g", name.c_str(), c.rows, c.zero, nz ? c.n1 / nz : 0, nz ? sqrt(c.n2 / nz) : 0,
               nz ? c.minn : 0);
        double eb[2] = {0, 0};
        for (double q : qs) {
            uint64_t acc = 0; double e = 0; int b = 0;
            for (; b < HB; ++b) { if (acc + c.hc[b] > q * nz) break; acc += c.hc[b]; e += c.he[b]; }
            printf("\t%.5g", pow(10, HLO + (b + 0.5) * HW));
            if (q == 0.1) eb[0] = e;
            if (q == 0.5) eb[1] = e;
        }
        double cm = 0, cmin = 1e300, cmax = 0;
        for (int j = 0; j < W; ++j) {
            cm += (c.cs[j] / nz) * (c.cs[j] / nz);
            const double cr = sqrt(c.cq[j] / nz);
            cmin = std::min(cmin, cr); cmax = std::max(cmax, cr);
        }
        const double mean_sq = c.n2 / nz;  // mean squared row norm
        printf("\t%.5g\t%.4f\t%.4f\t%.4f\t%.3f\t%.5f\t%.5g\t%.5g\n", c.maxn, eb[0] / c.n2, eb[1] / c.n2,
               c.z4 / (nz * (double) W), c.zmax / nz, cm / mean_sq, cmin, cmax);
    }
    if (const char * hp = getenv("HIST")) {
        FILE * f = fopen(hp, "w");
        fprintf(f, "head\tlog10_norm_lo\trows\tenergy_share\n");
        for (int h = 0; h <= nh + 1; ++h)
            for (int b = 0; b < HB; ++b)
                if (tot[h].hc[b])
                    fprintf(f, "%s\t%.3f\t%lu\t%.6g\n", h < nh ? std::to_string(h).c_str() : h == nh ? "pad" : "all",
                            HLO + b * HW, tot[h].hc[b], tot[h].he[b] / tot[h].n2);
        fclose(f);
    }
    if (norms_out) {
        FILE * f = fopen(norms_out, "wb");
        fwrite(norms.data(), 2, norms.size(), f);
        fclose(f);
    }
    return 0;
}

// ----------------------------------------------------------------------------------------------------- the formats

struct Format {
    virtual ~Format() {}
    virtual void fit(int h, const float * rows, size_t n) = 0;   // per-head parameters from a sample
    virtual void apply(int h, const float * w, float * out) const = 0;  // round trip of one row
    virtual void describe(FILE * f) const = 0;
};

struct Identity : Format {
    void fit(int, const float *, size_t) override {}
    void apply(int, const float * w, float * out) const override { memcpy(out, w, W * sizeof(float)); }
    void describe(FILE *) const override {}
};

struct Zero : Format {  // the bound: a table of zeros, i.e. no table at all
    void fit(int, const float *, size_t) override {}
    void apply(int, const float *, float * out) const override { memset(out, 0, W * sizeof(float)); }
    void describe(FILE *) const override {}
};

static double row_err(const float * a, const float * b) {
    double e = 0;
    for (int j = 0; j < W; ++j) { const double d = (double) a[j] - b[j]; e += d * d; }
    return e;
}

// q<b>: w_j ~ s * c_j * C[q_j], C a per-head codebook of 2^b sorted levels, c per-head column scales, s an fp16 row scale
struct Codebook : Format {
    int L;
    std::vector<std::vector<float>> C, c;  // per head
    explicit Codebook(int bits, int nh) : L(1 << bits), C(nh), c(nh) {}

    // the row's scale and levels; returns the scale (fp16-rounded), q[] the level per weight
    float fit_row(int h, const float * w, uint8_t * q) const {
        const float * Ch = C[h].data();
        const float * ch = c[h].data();
        float z[W], a[W], th[MAXL];
        for (int k = 0; k + 1 < L; ++k) th[k] = 0.5f * (Ch[k] + Ch[k + 1]);
        double sz = 0, sa = 0;
        for (int j = 0; j < W; ++j) { z[j] = w[j] / ch[j]; a[j] = ch[j] * ch[j]; sz += a[j] * z[j] * z[j]; sa += a[j]; }
        if (sz == 0) { memset(q, 0, W); return 0; }
        double cr = 0;
        for (int k = 0; k < L; ++k) cr += Ch[k] * Ch[k];
        const float s0 = (float) sqrt(sz / sa / (cr / L));
        auto assign = [&](float s, uint8_t * qq) {
            const float is = 1 / s;
            double e = 0;
            for (int j = 0; j < W; ++j) {
                const float v = z[j] * is;
                int k = 0;
                while (k + 1 < L && v > th[k]) ++k;
                qq[j] = (uint8_t) k;
                const double d = z[j] - s * Ch[k];
                e += a[j] * d * d;
            }
            return e;
        };
        uint8_t qq[W];
        float s = s0;
        double best = 1e300;
        for (int i = 0; i <= 16; ++i) {  // a coarse search for the scale, then Lloyd steps
            const float si = s0 * (0.5f + 0.0625f * i);
            const double e = assign(si, qq);
            if (e < best) { best = e; s = si; }
        }
        assign(s, q);
        for (int it = 0; it < 8; ++it) {
            double num = 0, den = 0;
            for (int j = 0; j < W; ++j) { num += a[j] * z[j] * Ch[q[j]]; den += a[j] * Ch[q[j]] * Ch[q[j]]; }
            if (den <= 0) break;
            const float sn = (float) (num / den);
            if (!(sn > 0)) break;
            const double e = assign(sn, qq);
            if (e >= best) break;
            best = e; s = sn; memcpy(q, qq, W);
        }
        s = r16(s);
        assign(s, q);
        return s;
    }
    void apply(int h, const float * w, float * out) const override {
        uint8_t q[W];
        const float s = fit_row(h, w, q);
        for (int j = 0; j < W; ++j) out[j] = s * c[h][j] * C[h][q[j]];
    }
    void fit(int h, const float * rows, size_t n) override {
        // start: column RMS of the rows scaled to unit RMS, and 1-D k-means levels of the normalized weights
        std::vector<double> cq(W, 0);
        size_t nz = 0;
        for (size_t i = 0; i < n; ++i) {
            const float * w = rows + i * W;
            double s2 = 0;
            for (int j = 0; j < W; ++j) s2 += (double) w[j] * w[j];
            if (s2 == 0) continue;
            ++nz;
            for (int j = 0; j < W; ++j) cq[j] += (double) w[j] * w[j] / (s2 / W);
        }
        c[h].assign(W, 1);
        double m = 0;
        for (int j = 0; j < W; ++j) { c[h][j] = cq[j] > 0 ? (float) sqrt(cq[j] / nz) : 1.f; m += c[h][j] * c[h][j]; }
        for (int j = 0; j < W; ++j) c[h][j] /= (float) sqrt(m / W);
        std::vector<float> zs;
        for (size_t i = 0; i < n && zs.size() < (1u << 22); i += 1) {
            const float * w = rows + i * W;
            double s2 = 0;
            for (int j = 0; j < W; ++j) s2 += (double) w[j] * w[j];
            if (s2 == 0) continue;
            const float rms = (float) sqrt(s2 / W);
            for (int j = 0; j < W; ++j) zs.push_back(w[j] / rms / c[h][j]);
        }
        std::sort(zs.begin(), zs.end());
        C[h].resize(L);
        for (int k = 0; k < L; ++k) C[h][k] = zs[(size_t) ((k + 0.5) / L * zs.size())];
        for (int it = 0; it < 30; ++it) {  // Lloyd on the pooled values
            std::vector<double> sum(L, 0), cnt(L, 0);
            size_t k = 0;
            for (float v : zs) {
                while (k + 1 < (size_t) L && v > 0.5f * (C[h][k] + C[h][k + 1])) ++k;
                sum[k] += v; cnt[k] += 1;
            }
            for (int l = 0; l < L; ++l) if (cnt[l]) C[h][l] = (float) (sum[l] / cnt[l]);
        }
        // alternate: rows given (C, c); C given rows and c; c given rows and C
        std::vector<float> s(n);
        std::vector<uint8_t> q(n * W);
        for (int round = 0; round < (int) envl("ROUNDS", 6); ++round) {
            double ew = 0, ee = 0;
#pragma omp parallel for reduction(+ : ew, ee) schedule(dynamic, 1024)
            for (size_t i = 0; i < n; ++i) {
                const float * w = rows + i * W;
                s[i] = fit_row(h, w, &q[i * W]);
                for (int j = 0; j < W; ++j) {
                    const double d = w[j] - s[i] * c[h][j] * C[h][q[i * W + j]];
                    ew += (double) w[j] * w[j]; ee += d * d;
                }
            }
            fprintf(stderr, "  head %d round %d: relative error %.5f\n", h, round, ee / ew);
            std::vector<double> num(L, 0), den(L, 0), cn(W, 0), cd(W, 0);
            for (size_t i = 0; i < n; ++i)
                for (int j = 0; j < W; ++j) {
                    const int k = q[i * W + j];
                    const double sc = (double) s[i] * c[h][j];
                    num[k] += sc * rows[i * W + j]; den[k] += sc * sc;
                }
            for (int k = 0; k < L; ++k) if (den[k] > 0) C[h][k] = (float) (num[k] / den[k]);
            for (size_t i = 0; i < n; ++i)
                for (int j = 0; j < W; ++j) {
                    const double sc = (double) s[i] * C[h][q[i * W + j]];
                    cn[j] += sc * rows[i * W + j]; cd[j] += sc * sc;
                }
            double mm = 0;
            for (int j = 0; j < W; ++j) { if (cd[j] > 0 && cn[j] > 0) c[h][j] = (float) (cn[j] / cd[j]); mm += c[h][j] * c[h][j]; }
            for (int j = 0; j < W; ++j) c[h][j] /= (float) sqrt(mm / W);
            std::sort(C[h].begin(), C[h].end());  // the next round reassigns every level
            float cmax = 0;
            for (int k = 0; k < L; ++k) cmax = std::max(cmax, fabsf(C[h][k]));
            for (int k = 0; k < L; ++k) C[h][k] /= cmax;
        }
    }
    void describe(FILE * f) const override {
        for (size_t h = 0; h < C.size(); ++h) {
            fprintf(f, "# head %zu levels", h);
            for (float v : C[h]) fprintf(f, " %.4f", v);
            float mn = 1e30f, mx = 0;
            for (float v : c[h]) { mn = std::min(mn, v); mx = std::max(mx, v); }
            fprintf(f, " column scales %.3f-%.3f\n", mn, mx);
        }
    }
};

// symmetric positive definite r x r (double, row-major): in-place Cholesky, then solves
static bool cholesky(std::vector<double> & A, int r) {
    for (int i = 0; i < r; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = A[i * r + j];
            for (int k = 0; k < j; ++k) s -= A[i * r + k] * A[j * r + k];
            if (i == j) { if (s <= 0) return false; A[i * r + i] = sqrt(s); }
            else A[i * r + j] = s / A[j * r + j];
        }
    return true;
}
static void chol_solve(const std::vector<double> & Lc, int r, double * x) {
    for (int i = 0; i < r; ++i) { double s = x[i]; for (int k = 0; k < i; ++k) s -= Lc[i * r + k] * x[k]; x[i] = s / Lc[i * r + i]; }
    for (int i = r - 1; i >= 0; --i) { double s = x[i]; for (int k = i + 1; k < r; ++k) s -= Lc[k * r + i] * x[k]; x[i] = s / Lc[i * r + i]; }
}

// symmetric eigen-decomposition by cyclic Jacobi (n x n, row-major); eigenvectors in V's columns
static void jacobi(std::vector<double> A, int n, std::vector<double> & ev, std::vector<double> & V) {
    V.assign(n * n, 0);
    for (int i = 0; i < n; ++i) V[i * n + i] = 1;
    for (int sweep = 0; sweep < 50; ++sweep) {
        double off = 0;
        for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) off += A[i * n + j] * A[i * n + j];
        if (off < 1e-22) break;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) {
                const double apq = A[p * n + q];
                if (fabs(apq) < 1e-300) continue;
                const double th = (A[q * n + q] - A[p * n + p]) / (2 * apq);
                const double t = (th >= 0 ? 1 : -1) / (fabs(th) + sqrt(th * th + 1));
                const double cs = 1 / sqrt(t * t + 1), sn = t * cs;
                for (int k = 0; k < n; ++k) {
                    const double akp = A[k * n + p], akq = A[k * n + q];
                    A[k * n + p] = cs * akp - sn * akq; A[k * n + q] = sn * akp + cs * akq;
                }
                for (int k = 0; k < n; ++k) {
                    const double apk = A[p * n + k], aqk = A[q * n + k];
                    A[p * n + k] = cs * apk - sn * aqk; A[q * n + k] = sn * apk + cs * aqk;
                }
                for (int k = 0; k < n; ++k) {
                    const double vkp = V[k * n + p], vkq = V[k * n + q];
                    V[k * n + p] = cs * vkp - sn * vkq; V[k * n + q] = sn * vkp + cs * vkq;
                }
            }
    }
    ev.resize(n);
    for (int i = 0; i < n; ++i) ev[i] = A[i * n + i];
}

// bf<r>: w ~ s * B u, u in {-1, +1}^r, B a per-head real 160 x r basis, s an fp16 row scale
struct BinaryBasis : Format {
    int r;
    std::vector<std::vector<float>> B, G, Gi;  // per head: B (W x r, row-major), G = B^T B, its inverse
    BinaryBasis(int rank, int nh) : r(rank), B(nh), G(nh), Gi(nh) {}

    void set_basis(int h, const std::vector<double> & b) {
        B[h].assign(b.begin(), b.end());
        std::vector<double> g(r * r, 0);
        for (int a = 0; a < r; ++a)
            for (int c = 0; c < r; ++c) {
                double s = 0;
                for (int j = 0; j < W; ++j) s += b[j * r + a] * b[j * r + c];
                g[a * r + c] = s;
            }
        G[h].assign(g.begin(), g.end());
        std::vector<double> lc = g;
        for (int a = 0; a < r; ++a) lc[a * r + a] *= 1 + 1e-9;
        if (!cholesky(lc, r)) { fprintf(stderr, "head %d: basis Gram not positive definite\n", h); exit(1); }
        Gi[h].assign(r * r, 0);
        for (int c = 0; c < r; ++c) {
            double e[MAXR] = {};
            e[c] = 1;
            chol_solve(lc, r, e);
            for (int a = 0; a < r; ++a) Gi[h][a * r + c] = (float) e[a];
        }
    }
    // the row's signs (u[k] = +-1) and fp16 scale
    float fit_row(int h, const float * w, float * u) const {
        const float * b = B[h].data(), * g2 = G[h].data(), * gi = Gi[h].data();
        double g[MAXR] = {}, hv[MAXR];
        for (int j = 0; j < W; ++j) { const double wj = w[j]; for (int k = 0; k < r; ++k) g[k] += wj * b[j * r + k]; }
        // two starts: signs of the least-squares coefficients and of B^T w; keep the better one
        float u1[MAXR], u2[MAXR];
        for (int k = 0; k < r; ++k) {
            double c = 0;
            for (int m = 0; m < r; ++m) c += gi[k * r + m] * g[m];
            u1[k] = c >= 0 ? 1.f : -1.f; u2[k] = g[k] >= 0 ? 1.f : -1.f;
        }
        double bestF = -1, bestp = 0, bestq = 1;
        for (int start = 0; start < 2; ++start) {
            float * uu = start ? u2 : u1;
            double p = 0, q = 0;
            for (int k = 0; k < r; ++k) {
                double s = 0;
                for (int m = 0; m < r; ++m) s += g2[k * r + m] * uu[m];
                hv[k] = s; p += g[k] * uu[k]; q += uu[k] * s;
            }
            for (int flips = 0; flips < 4 * r; ++flips) {  // steepest single-sign flips on (g.u)^2 / (u.Gu)
                double F = p * p / q, bf = F * (1 + 1e-7);
                int bk = -1;
                double bp = 0, bq = 0;
                for (int k = 0; k < r; ++k) {
                    const double pk = p - 2 * uu[k] * g[k], qk = q - 4 * uu[k] * hv[k] + 4 * g2[k * r + k];
                    if (qk <= 0) continue;
                    const double Fk = pk * pk / qk;
                    if (Fk > bf) { bf = Fk; bk = k; bp = pk; bq = qk; }
                }
                if (bk < 0) break;
                const float old = uu[bk];
                uu[bk] = -old;
                for (int m = 0; m < r; ++m) hv[m] -= 2 * old * g2[m * r + bk];
                p = bp; q = bq;
            }
            const double F = q > 0 ? p * p / q : 0;
            if (F > bestF) { bestF = F; bestp = p; bestq = q; memcpy(u, uu, r * sizeof(float)); }
        }
        return bestq > 0 ? r16((float) (bestp / bestq)) : 0;
    }
    void apply(int h, const float * w, float * out) const override {
        float u[MAXR];
        const float s = fit_row(h, w, u);
        const float * b = B[h].data();
        for (int j = 0; j < W; ++j) {
            float v = 0;
            for (int k = 0; k < r; ++k) v += b[j * r + k] * u[k];
            out[j] = s * v;
        }
    }
    void fit(int h, const float * rows, size_t n) override {
        // start: the top r principal directions of the rows scaled to unit norm, coefficients' signs, then least squares
        std::vector<double> cov(W * W, 0);
#pragma omp parallel
        {
            std::vector<double> loc(W * W, 0);
#pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                const float * w = rows + i * W;
                double s2 = 0;
                for (int j = 0; j < W; ++j) s2 += (double) w[j] * w[j];
                if (s2 == 0) continue;
                for (int a = 0; a < W; ++a) for (int c = a; c < W; ++c) loc[a * W + c] += w[a] * (double) w[c] / s2;
            }
#pragma omp critical
            for (int k = 0; k < W * W; ++k) cov[k] += loc[k];
        }
        for (int a = 0; a < W; ++a) for (int c = 0; c < a; ++c) cov[a * W + c] = cov[c * W + a];
        std::vector<double> ev, V;
        jacobi(cov, W, ev, V);
        std::vector<int> ord(W);
        for (int k = 0; k < W; ++k) ord[k] = k;
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return ev[a] > ev[b]; });
        double etot = 0, etop = 0;
        for (int k = 0; k < W; ++k) { etot += ev[k]; if (k < r) etop += ev[ord[k]]; }
        fprintf(stderr, "  head %d: top %d principal directions hold %.4f of the normalized energy\n", h, r, etop / etot);
        std::vector<float> u(n * r), s(n, 1.f);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i)
            for (int k = 0; k < r; ++k) {
                double c = 0;
                for (int j = 0; j < W; ++j) c += rows[i * W + j] * V[j * W + ord[k]];
                u[i * r + k] = c >= 0 ? 1.f : -1.f;
            }
        auto solve_basis = [&]() {  // B = (sum s w u^T)(sum s^2 u u^T)^-1
            std::vector<double> A(W * r, 0), H(r * r, 0);
#pragma omp parallel
            {
                std::vector<double> la(W * r, 0), lh(r * r, 0);
#pragma omp for schedule(static)
                for (size_t i = 0; i < n; ++i) {
                    const float * ui = &u[i * r];
                    const double si = s[i];
                    if (si == 0) continue;
                    for (int j = 0; j < W; ++j) { const double v = si * rows[i * W + j]; for (int k = 0; k < r; ++k) la[j * r + k] += v * ui[k]; }
                    for (int a = 0; a < r; ++a) for (int c = a; c < r; ++c) lh[a * r + c] += si * si * ui[a] * ui[c];
                }
#pragma omp critical
                { for (int k = 0; k < W * r; ++k) A[k] += la[k]; for (int k = 0; k < r * r; ++k) H[k] += lh[k]; }
            }
            for (int a = 0; a < r; ++a) for (int c = 0; c < a; ++c) H[a * r + c] = H[c * r + a];
            double tr = 0;
            for (int a = 0; a < r; ++a) tr += H[a * r + a];
            for (int a = 0; a < r; ++a) H[a * r + a] += 1e-9 * tr / r;
            if (!cholesky(H, r)) { fprintf(stderr, "head %d: H not positive definite\n", h); exit(1); }
            std::vector<double> b(W * r);
            for (int j = 0; j < W; ++j) {
                double x[MAXR];
                for (int k = 0; k < r; ++k) x[k] = A[j * r + k];
                chol_solve(H, r, x);
                for (int k = 0; k < r; ++k) b[j * r + k] = x[k];
            }
            set_basis(h, b);
        };
        solve_basis();
        for (int round = 0; round < (int) envl("ROUNDS", 8); ++round) {
            double ew = 0, ee = 0;
#pragma omp parallel for reduction(+ : ew, ee) schedule(dynamic, 1024)
            for (size_t i = 0; i < n; ++i) {
                const float * w = rows + i * W;
                s[i] = fit_row(h, w, &u[i * r]);
                const float * b = B[h].data();
                for (int j = 0; j < W; ++j) {
                    float v = 0;
                    for (int k = 0; k < r; ++k) v += b[j * r + k] * u[i * r + k];
                    const double d = w[j] - s[i] * v;
                    ew += (double) w[j] * w[j]; ee += d * d;
                }
            }
            fprintf(stderr, "  head %d round %d: relative error %.5f\n", h, round, ee / ew);
            solve_basis();
        }
    }
    void describe(FILE * f) const override {
        for (size_t h = 0; h < B.size(); ++h) {
            double tr = 0;
            for (int k = 0; k < r; ++k) tr += G[h][k * r + k];
            fprintf(f, "# head %zu basis rank %d, mean squared basis vector norm %.4g\n", h, r, tr / r);
        }
    }
};

// ------------------------------------------------------------------------------------------------------ round trip

struct Err { double w = 0, fmt = 0, re = 0, tot = 0; uint64_t rows = 0, same_blocks = 0, blocks = 0; };

static int roundtrip(const Info & in, std::string fmt, const char * out_path) {
    const int nh = (int) in.ho.size();
    const bool q8 = fmt.size() > 3 && fmt.compare(fmt.size() - 3, 3, "-q8") == 0;
    if (q8) fmt.resize(fmt.size() - 3);
    if (q8 && (!in.type_pos || in.file_size != in.end)) { fprintf(stderr, "-q8 needs type_pos and no tail\n"); return 1; }
    const uint64_t ORB = q8 ? RB8 : RB;  // bytes per written row
    Format * F = nullptr;
    if (fmt == "iq4nl") F = new Identity();
    else if (fmt == "zero") F = new Zero();
    else if (fmt[0] == 'q') F = new Codebook(atoi(fmt.c_str() + 1), nh);
    else if (fmt.rfind("bf", 0) == 0) F = new BinaryBasis(atoi(fmt.c_str() + 2), nh);
    if (!F) { fprintf(stderr, "unknown format %s\n", fmt.c_str()); return 1; }
    const uint8_t * file = map_table(in);
    const uint8_t * base = file + in.offset;
    double t0 = now();
    const size_t S = (size_t) envl("SAMPLE", 262144);
    if (fmt != "iq4nl" && fmt != "zero")
        for (int h = 0; h < nh; ++h) {
            const size_t n = std::min<size_t>(S, in.hn[h]);
            std::vector<float> rows(n * W);
            const double step = (double) in.hn[h] / n;
#pragma omp parallel for
            for (size_t i = 0; i < n; ++i) dequant_row(base + (in.ho[h] + (uint64_t) (i * step)) * RB, &rows[i * W]);
            F->fit(h, rows.data(), n);
        }
    fprintf(stderr, "fit: %.1f s\n", now() - t0);
    F->describe(stdout);
    int fo = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fo < 0) { perror(out_path); return 1; }
    std::vector<uint8_t> hdr(file, file + in.offset);
    if (q8) { const uint32_t t = 8; memcpy(&hdr[in.type_pos], &t, 4); }  // GGML_TYPE_Q8_0
    if (pwrite(fo, hdr.data(), in.offset, 0) != (ssize_t) in.offset) { perror("pwrite"); return 1; }
    std::vector<Err> tot(nh);
    const uint64_t CH = 1 << 20;
    std::vector<uint8_t> buf(CH * ORB);
    t0 = now();
    for (uint64_t r0 = 0; r0 < in.rows; r0 += CH) {
        const uint64_t n = std::min(CH, in.rows - r0);
#pragma omp parallel
        {
            std::vector<Err> loc(nh);
            float w[W], y[W], z[W];
#pragma omp for schedule(dynamic, 4096)
            for (uint64_t i = 0; i < n; ++i) {
                const uint64_t r = r0 + i;
                const uint8_t * src = base + r * RB;
                uint8_t * dst = &buf[i * ORB];
                const int h = in.head_of(r);
                if (h < 0 && !q8) { memcpy(dst, src, RB); continue; }
                dequant_row(src, w);
                if (h < 0) { for (int b = 0; b < NB; ++b) quant_block_q8(w + 32 * b, dst + 34 * b); continue; }
                F->apply(h, w, y);
                if (q8) {
                    for (int b = 0; b < NB; ++b) quant_block_q8(y + 32 * b, dst + 34 * b);
                    dequant_row_q8(dst, z);
                } else {
                    for (int b = 0; b < NB; ++b) quant_block(y + 32 * b, dst + 18 * b);
                    dequant_row(dst, z);
                }
                Err & e = loc[h];
                double s2 = 0;
                for (int j = 0; j < W; ++j) s2 += (double) w[j] * w[j];
                e.w += s2; e.fmt += row_err(w, y); e.re += row_err(y, z); e.tot += row_err(w, z); e.rows++;
                for (int b = 0; b < NB; ++b) { e.blocks++; e.same_blocks += !q8 && !memcmp(src + 18 * b, dst + 18 * b, 18); }
            }
#pragma omp critical
            for (int h = 0; h < nh; ++h) {
                tot[h].w += loc[h].w; tot[h].fmt += loc[h].fmt; tot[h].re += loc[h].re; tot[h].tot += loc[h].tot;
                tot[h].rows += loc[h].rows; tot[h].same_blocks += loc[h].same_blocks; tot[h].blocks += loc[h].blocks;
            }
        }
        if (pwrite(fo, buf.data(), n * ORB, in.offset + r0 * ORB) != (ssize_t) (n * ORB)) { perror("pwrite"); return 1; }
        if ((r0 / CH) % 32 == 0) fprintf(stderr, "  %lu / %lu rows, %.0f s\n", r0 + n, in.rows, now() - t0);
    }
    if (in.file_size > in.end) {
        const size_t tail = in.file_size - in.end;
        if (pwrite(fo, file + in.end, tail, in.end) != (ssize_t) tail) { perror("pwrite"); return 1; }
    }
    fsync(fo);
    close(fo);
    fprintf(stderr, "apply: %.1f s\n", now() - t0);
    // relative squared errors: of the format (against the decoded table), of the re-encode (IQ4_NL or Q8_0, against
    // the format's output), and of the written table (against the decoded table)
    printf("head\trows\tformat_rel_err\treencode_rel_err\ttotal_rel_err\tsame_blocks\n");
    Err all;
    for (int h = 0; h <= nh; ++h) {
        const Err & e = h < nh ? tot[h] : all;
        if (h < nh) {
            all.w += e.w; all.fmt += e.fmt; all.re += e.re; all.tot += e.tot; all.rows += e.rows;
            all.same_blocks += e.same_blocks; all.blocks += e.blocks;
        }
        printf("%s\t%lu\t%.6f\t%.6f\t%.6f\t%.6f\n", h < nh ? std::to_string(h).c_str() : "all", e.rows, e.fmt / e.w,
               e.re / e.w, e.tot / e.w, (double) e.same_blocks / e.blocks);
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc >= 3 && !strcmp(argv[1], "census")) return census(read_info(argv[2]), argc > 3 ? argv[3] : nullptr);
    if (argc == 5 && !strcmp(argv[1], "roundtrip")) return roundtrip(read_info(argv[2]), argv[3], argv[4]);
    fprintf(stderr, "usage: table_bits census INFO [NORMS_OUT] | table_bits roundtrip INFO FMT OUT_SHARD\n");
    return 1;
}
