// llama-act-hessian: accumulate H = sum of x x^T over the rows of chosen graph tensors while a model processes text,
// the GPTQ Hessians of the weights those tensors feed (e.g. attn_norm-9 feeds layer 9's q/k/v, kqv_out-3 feeds layer
// 3's attention output, ffn_norm-12 the shared expert's gate/up, ffn_swiglu-12 its down projection), and per routed
// expert for chosen MoE layers.
//
//   HESS_TENSORS=kqv_out-0,attn_norm-9,... HESS_EXPERTS=9,25 HESS_OUT=dir HEAD_TEXT=train.txt \
//     [HEAD_CHUNK0=0 HEAD_CHUNKS=500 HEAD_PARSE_SPECIAL=1 HESS_EXP_OUT=dir HESS_EXP_FLUSH=512] \
//     llama-act-hessian -m model.gguf -c 512 -b 512 ...
//
// The text is tokenized and cut into chunks of n_ctx tokens as llama-perplexity does it; each chunk runs from an empty
// KV cache and its first token is left out of every H. Only the last token asks for logits: layers before the last
// compute every row anyway. HEAD_PARSE_SPECIAL=1 reads chat markup in the text as special tokens.
//
// HESS_TENSORS: for every listed tensor, HESS_OUT/<name>.h64 holds int64 n, int64 rows, then n*n float64. H is built
// per ubatch from float32 dot products over the ubatch's rows, added into float64.
//
// HESS_EXPERTS: for every listed layer L, one H per routed expert for its gate/up (rows of ffn_norm-L, each added to
// the experts ffn_moe_topk-L chose for that token) and for its down projection (each row of ffn_moe_swiglu-L, to the
// expert that produced it). Rows are buffered per expert and added HESS_EXP_FLUSH at a time, as float32 dot products
// into float64. HESS_EXP_OUT/experts-L-gate_up.e64 and experts-L-down.e64 hold int64 n, int64 n_expert,
// int64 rows[n_expert], then per expert the upper triangle of H row by row (i <= j), n(n+1)/2 float64.
//
// Built with HESS_BLAS (CMake option HESS_BLAS_LIB, a static OpenBLAS), each block's float32 Gram matrix comes from
// cblas_ssyrk; otherwise from plain OpenMP loops. Either way it is added into float64.
//
// HESS_DUMP=name,...: also append the rows each listed tensor contributed (float32, first row of a chunk left out) to
// HESS_OUT/<name>.rows.f32, or the expert ids of ffn_moe_topk-L to HESS_OUT/<name>.ids.i32 (n_used per row), so the
// Hessians can be recomputed independently on a short run.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifdef HESS_BLAS
#include <cblas.h>
#endif

static void transpose(const float * rows, int64_t m, int64_t n, std::vector<float> & xt) {
    xt.resize(n * m);
    for (int64_t k = 0; k < m; ++k) {
        for (int64_t i = 0; i < n; ++i) {
            xt[i * m + k] = rows[k * n + i];
        }
    }
}

// h += x^T x over the m rows of x ([m, n], row-major), into the upper triangle of h: full n*n storage or packed
static void add_gram(const float * rows, int64_t n, int64_t m, double * h, bool packed, std::vector<float> & work) {
#ifdef HESS_BLAS
    work.resize(n * n);
    cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans, (int) n, (int) m, 1.0f, rows, (int) n, 0.0f, work.data(), (int) n);
#pragma omp parallel for schedule(dynamic, 16)
    for (int64_t i = 0; i < n; ++i) {
        double * hi = packed ? h + i * n - i * (i - 1) / 2 - i : h + i * n;
        const float * gi = work.data() + i * n;
        for (int64_t j = i; j < n; ++j) {
            hi[j] += gi[j];
        }
    }
#else
    transpose(rows, m, n, work);
    const float * xt = work.data();
#pragma omp parallel for schedule(dynamic, 16)
    for (int64_t i = 0; i < n; ++i) {
        const float * a = xt + i * m;
        double * hi = packed ? h + i * n - i * (i - 1) / 2 - i : h + i * n;
        for (int64_t j = i; j < n; ++j) {
            const float * b = xt + j * m;
            float s = 0.0f;
#pragma omp simd reduction(+:s)
            for (int64_t k = 0; k < m; ++k) {
                s += a[k] * b[k];
            }
            hi[j] += s;
        }
    }
#endif
}

struct hess {
    int64_t n = 0;
    int64_t rows = 0;
    std::vector<double> h;
};

// per-expert Hessians of one weight class of one layer
struct expert_hess {
    int64_t n = 0;
    std::vector<int64_t> rows;
    std::vector<std::vector<float>> buf; // buffered rows per expert, row-major
    std::vector<std::vector<double>> h;  // packed upper triangle per expert

    void init(int64_t n_, int64_t n_expert) {
        n = n_;
        rows.assign(n_expert, 0);
        buf.assign(n_expert, {});
        h.assign(n_expert, std::vector<double>(n * (n + 1) / 2, 0.0));
    }
    void flush(int64_t e, std::vector<float> & xt) {
        const int64_t m = (int64_t) buf[e].size() / n;
        if (m == 0) {
            return;
        }
        add_gram(buf[e].data(), n, m, h[e].data(), true, xt);
        rows[e] += m;
        buf[e].clear();
    }
    void add(int64_t e, const float * x, int64_t flush_rows, std::vector<float> & xt) {
        buf[e].insert(buf[e].end(), x, x + n);
        if ((int64_t) buf[e].size() >= flush_rows * n) {
            flush(e, xt);
        }
    }
};

struct expert_layer {
    expert_hess gate_up, down;
    std::vector<float> x;      // ffn_norm rows of the current ubatch (first row of a chunk already dropped)
    std::vector<int32_t> ids;  // ffn_moe_topk of the current ubatch, same rows
    int64_t n_tokens = 0, n_used = 0;
};

struct state {
    std::map<std::string, hess> want;
    std::map<std::string, FILE *> dump;
    std::string out;
    std::map<int, expert_layer> experts;
    int64_t n_expert = 0;
    int64_t flush_rows = 512;
    bool skip_first = false;    // the ubatch starts a chunk: leave its first row out
    std::vector<float> raw, xt; // [rows, n] as read; add_gram's work buffer
};

// "ffn_norm-12" -> ("ffn_norm", 12)
static bool split_name(const char * name, std::string & base, int & il) {
    const char * dash = strrchr(name, '-');
    if (!dash || !dash[1]) {
        return false;
    }
    char * end = nullptr;
    il = (int) strtol(dash + 1, &end, 10);
    if (*end) {
        return false;
    }
    base.assign(name, dash - name);
    return true;
}

static expert_layer * expert_target(state * st, const char * name, std::string & base) {
    int il;
    if (st->experts.empty() || !split_name(name, base, il)) {
        return nullptr;
    }
    if (base != "ffn_norm" && base != "ffn_moe_topk" && base != "ffn_moe_swiglu") {
        return nullptr;
    }
    auto it = st->experts.find(il);
    return it == st->experts.end() ? nullptr : &it->second;
}

static void read_rows(state * st, struct ggml_tensor * t) {
    const int64_t n = t->ne[0];
    const int64_t r = ggml_nrows(t);
    st->raw.resize(n * r);
    ggml_backend_tensor_get(t, st->raw.data(), 0, n * r * sizeof(float));
}

static void need_f32(struct ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        fprintf(stderr, "%s: type %s or layout not supported\n", t->name, ggml_type_name(t->type));
        exit(1);
    }
}

static void experts_cb(state * st, expert_layer & L, const std::string & base, struct ggml_tensor * t) {
    const int64_t r0 = st->skip_first ? 1 : 0;
    if (base == "ffn_norm") {
        need_f32(t);
        read_rows(st, t);
        const int64_t n = t->ne[0];
        L.n_tokens = ggml_nrows(t) - r0;
        L.x.assign(st->raw.begin() + r0 * n, st->raw.end());
        if (L.gate_up.n == 0) {
            L.gate_up.init(n, st->n_expert);
        }
        return;
    }
    if (base == "ffn_moe_topk") {
        // a view of the router's argsort: [n_used, n_tokens] with the argsort's row stride
        if (t->type != GGML_TYPE_I32 || t->nb[0] != sizeof(int32_t) || t->ne[1] - r0 != L.n_tokens) {
            fprintf(stderr, "%s: expected I32 [n_used, %lld tokens]\n", t->name, (long long) L.n_tokens);
            exit(1);
        }
        L.n_used = t->ne[0];
        std::vector<int32_t> all(ggml_nbytes(t) / sizeof(int32_t));
        ggml_backend_tensor_get(t, all.data(), 0, all.size() * sizeof(int32_t));
        const int64_t stride = t->nb[1] / sizeof(int32_t);
        L.ids.resize(L.n_tokens * L.n_used);
        for (int64_t k = 0; k < L.n_tokens; ++k) {
            for (int64_t s = 0; s < L.n_used; ++s) {
                L.ids[k * L.n_used + s] = all[(k + r0) * stride + s];
            }
        }
        const int64_t n = L.gate_up.n;
        for (int64_t k = 0; k < L.n_tokens; ++k) {
            for (int64_t s = 0; s < L.n_used; ++s) {
                const int32_t e = L.ids[k * L.n_used + s];
                if (e < 0 || e >= st->n_expert) {
                    fprintf(stderr, "%s: expert id %d out of range\n", t->name, e);
                    exit(1);
                }
                L.gate_up.add(e, L.x.data() + k * n, st->flush_rows, st->xt);
            }
        }
        return;
    }
    // ffn_moe_swiglu: [n_ff, n_used, n_tokens], row (s, k) produced by expert ids[k][s]
    need_f32(t);
    if (t->ne[1] != L.n_used || t->ne[2] - r0 != L.n_tokens) {
        fprintf(stderr, "%s: expected [n_ff, %lld, %lld tokens]\n", t->name, (long long) L.n_used, (long long) L.n_tokens);
        exit(1);
    }
    read_rows(st, t);
    const int64_t n = t->ne[0];
    if (L.down.n == 0) {
        L.down.init(n, st->n_expert);
    }
    const float * x = st->raw.data() + r0 * L.n_used * n;
    for (int64_t k = 0; k < L.n_tokens; ++k) {
        for (int64_t s = 0; s < L.n_used; ++s) {
            L.down.add(L.ids[k * L.n_used + s], x + (k * L.n_used + s) * n, st->flush_rows, st->xt);
        }
    }
}

static bool hess_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (state *) user_data;
    auto it = st->want.find(t->name);
    std::string base;
    expert_layer * L = expert_target(st, t->name, base);
    if (ask) {
        return it != st->want.end() || L != nullptr || st->dump.count(t->name) > 0;
    }
    if (L) {
        experts_cb(st, *L, base, t);
    }
    auto dit = st->dump.find(t->name);
    if (dit != st->dump.end()) {
        if (!dit->second) {
            const bool ids = t->type == GGML_TYPE_I32;
            dit->second = fopen((st->out + "/" + t->name + (ids ? ".ids.i32" : ".rows.f32")).c_str(), "wb");
        }
        if (L && base == "ffn_moe_topk") {
            fwrite(L->ids.data(), sizeof(int32_t), L->ids.size(), dit->second);
        } else {
            need_f32(t);
            read_rows(st, t);
            // rows per token: one, or n_used for ffn_moe_swiglu ([n_ff, n_used, n_tokens])
            const int64_t skip = (st->skip_first ? 1 : 0) * t->ne[0] * (L && base == "ffn_moe_swiglu" ? L->n_used : 1);
            fwrite(st->raw.data() + skip, sizeof(float), st->raw.size() - skip, dit->second);
        }
    }
    if (it == st->want.end()) {
        return true;
    }
    need_f32(t);
    const int64_t n = t->ne[0];
    hess & H = it->second;
    if (H.n == 0) {
        H.n = n;
        H.h.assign(n * n, 0.0);
    }
    read_rows(st, t);
    const int64_t r0 = st->skip_first ? 1 : 0;
    const int64_t m  = ggml_nrows(t) - r0;
    add_gram(st->raw.data() + r0 * n, n, m, H.h.data(), false, st->xt);
    H.rows += m;
    return true;
}

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

static void write_experts(const std::string & path, expert_hess & E, std::vector<float> & xt) {
    for (size_t e = 0; e < E.h.size(); ++e) {
        E.flush(e, xt);
    }
    FILE * o = fopen(path.c_str(), "wb");
    if (!o) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        exit(1);
    }
    const int64_t n_expert = (int64_t) E.h.size();
    fwrite(&E.n, sizeof(int64_t), 1, o);
    fwrite(&n_expert, sizeof(int64_t), 1, o);
    fwrite(E.rows.data(), sizeof(int64_t), n_expert, o);
    int64_t total = 0, empty = 0;
    for (int64_t e = 0; e < n_expert; ++e) {
        fwrite(E.h[e].data(), sizeof(double), E.h[e].size(), o);
        total += E.rows[e];
        empty += E.rows[e] == 0;
    }
    fclose(o);
    fprintf(stderr, "%s: n %lld, %lld experts, %lld rows, %lld experts without rows\n", path.c_str(), (long long) E.n,
            (long long) n_expert, (long long) total, (long long) empty);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    const char * names   = getenv("HESS_TENSORS");
    const char * exps    = getenv("HESS_EXPERTS");
    const char * out     = getenv("HESS_OUT");
    const char * text    = getenv("HEAD_TEXT");
    const char * exp_out = getenv("HESS_EXP_OUT");
    if ((!names && !exps) || !out || !text) {
        fprintf(stderr, "set HESS_TENSORS and/or HESS_EXPERTS, HESS_OUT and HEAD_TEXT\n");
        return 1;
    }
    state st;
    st.out = out;
    st.flush_rows = env_int("HESS_EXP_FLUSH", 512);
    if (const char * d = getenv("HESS_DUMP")) {
        std::stringstream ss(d);
        for (std::string s; std::getline(ss, s, ',');) {
            if (!s.empty()) {
                st.dump[s] = nullptr;
            }
        }
    }
    if (names) {
        std::stringstream ss(names);
        for (std::string s; std::getline(ss, s, ',');) {
            if (!s.empty()) {
                st.want[s];
            }
        }
    }
    if (exps) {
        std::stringstream ss(exps);
        for (std::string s; std::getline(ss, s, ',');) {
            if (!s.empty()) {
                st.experts[atoi(s.c_str())];
            }
        }
    }

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = hess_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) {
        fprintf(stderr, "failed to load the model\n");
        return 1;
    }
    if (!st.experts.empty()) {
        char arch[64] = {0}, val[32] = {0};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        if (llama_model_meta_val_str(model, (std::string(arch) + ".expert_count").c_str(), val, sizeof(val)) <= 0) {
            fprintf(stderr, "HESS_EXPERTS: the model has no %s.expert_count\n", arch);
            return 1;
        }
        st.n_expert = atoll(val);
    }
    const int32_t n_batch = (int32_t) llama_n_batch(ctx);
    const int32_t n_ctx   = (int32_t) llama_n_ctx(ctx);

    std::ifstream f(text);
    std::stringstream buf;
    buf << f.rdbuf();
    const std::vector<llama_token> toks = common_tokenize(ctx, buf.str(), true, env_int("HEAD_PARSE_SPECIAL", 0) != 0);
    const int n_max  = (int) toks.size() / n_ctx;
    const int chunk0 = env_int("HEAD_CHUNK0", 0);
    const int n      = std::min(env_int("HEAD_CHUNKS", n_max - chunk0), n_max - chunk0);
    fprintf(stderr, "%zu tokens, chunks %d..%d of %d tokens, %zu tensors, %zu expert layers of %lld experts\n",
            toks.size(), chunk0, chunk0 + n - 1, n_ctx, st.want.size(), st.experts.size(), (long long) st.n_expert);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (int c = chunk0; c < chunk0 + n; ++c) {
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int32_t i0 = 0; i0 < n_ctx; i0 += n_batch) {
            const int32_t m = std::min(n_batch, n_ctx - i0);
            batch.n_tokens = m;
            for (int32_t j = 0; j < m; ++j) {
                batch.token[j]     = toks[(size_t) c * n_ctx + i0 + j];
                batch.pos[j]       = i0 + j;
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = i0 + j == n_ctx - 1;
            }
            st.skip_first = i0 == 0;
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "chunk %d: decode failed\n", c);
                return 1;
            }
        }
        if ((c - chunk0) % 50 == 0) {
            fprintf(stderr, "chunk %d done\n", c);
        }
    }
    llama_batch_free(batch);

    for (auto & [name, H] : st.want) {
        if (H.n == 0) {
            fprintf(stderr, "%s: never seen in the graph\n", name.c_str());
            return 1;
        }
        for (int64_t i = 0; i < H.n; ++i) {       // mirror the upper triangle
            for (int64_t j = 0; j < i; ++j) {
                H.h[i * H.n + j] = H.h[j * H.n + i];
            }
        }
        const std::string path = std::string(out) + "/" + name + ".h64";
        FILE * o = fopen(path.c_str(), "wb");
        fwrite(&H.n, sizeof(int64_t), 1, o);
        fwrite(&H.rows, sizeof(int64_t), 1, o);
        fwrite(H.h.data(), sizeof(double), H.h.size(), o);
        fclose(o);
        fprintf(stderr, "%s: n %lld, %lld rows\n", path.c_str(), (long long) H.n, (long long) H.rows);
    }
    for (auto & [il, L] : st.experts) {
        if (L.gate_up.n == 0 || L.down.n == 0) {
            fprintf(stderr, "layer %d: ffn_norm, ffn_moe_topk or ffn_moe_swiglu never seen in the graph\n", il);
            return 1;
        }
        const std::string dir = exp_out ? exp_out : out;
        write_experts(dir + "/experts-" + std::to_string(il) + "-gate_up.e64", L.gate_up, st.xt);
        write_experts(dir + "/experts-" + std::to_string(il) + "-down.e64", L.down, st.xt);
    }
    for (auto & [name, fp] : st.dump) {
        if (fp) {
            fclose(fp);
        }
    }
    llama_backend_free();
    fprintf(stderr, "done\n");
    return 0;
}
