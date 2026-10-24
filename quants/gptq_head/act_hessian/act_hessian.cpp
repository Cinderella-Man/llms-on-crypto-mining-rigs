// llama-act-hessian: accumulate H = sum of x x^T over the rows of chosen graph tensors while a model processes text,
// the GPTQ Hessians of the weights those tensors feed (e.g. attn_norm-9 feeds layer 9's q/k/v, kqv_out-3 feeds layer
// 3's attention output, ffn_norm-12 the shared expert's gate/up, ffn_swiglu-12 its down projection).
//
//   HESS_TENSORS=kqv_out-0,attn_norm-9,... HESS_OUT=dir HEAD_TEXT=train.txt [HEAD_CHUNK0=0 HEAD_CHUNKS=500] \
//     llama-act-hessian -m model.gguf -c 512 -b 512 ...
//
// The text is tokenized and cut into chunks of n_ctx tokens as llama-perplexity does it; each chunk runs from an empty
// KV cache and its first token is left out of H. Only the last token asks for logits: layers before the last compute
// every row anyway. For every requested tensor, HESS_OUT/<name>.h64 holds int64 n, int64 rows, then n*n float64.
// H is built per ubatch from float32 dot products over the ubatch's rows, added into float64.

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

struct hess {
    int64_t n = 0;
    int64_t rows = 0;
    std::vector<double> h;
};

struct state {
    std::map<std::string, hess> want;
    bool skip_first = false;    // the ubatch starts a chunk: leave its first row out
    std::vector<float> raw, xt; // [rows, n] as read, [n, rows] transposed
};

static bool hess_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (state *) user_data;
    auto it = st->want.find(t->name);
    if (ask || it == st->want.end()) {
        return ask ? it != st->want.end() : true;
    }
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        fprintf(stderr, "%s: type %s or layout not supported\n", t->name, ggml_type_name(t->type));
        exit(1);
    }
    const int64_t n = t->ne[0];
    const int64_t r = ggml_nrows(t);
    hess & H = it->second;
    if (H.n == 0) {
        H.n = n;
        H.h.assign(n * n, 0.0);
    }
    st->raw.resize(n * r);
    ggml_backend_tensor_get(t, st->raw.data(), 0, n * r * sizeof(float));
    const int64_t r0 = st->skip_first ? 1 : 0;
    const int64_t m  = r - r0;
    st->xt.resize(n * m);
    for (int64_t k = 0; k < m; ++k) {
        for (int64_t i = 0; i < n; ++i) {
            st->xt[i * m + k] = st->raw[(k + r0) * n + i];
        }
    }
    const float * xt = st->xt.data();
    double * h = H.h.data();
#pragma omp parallel for schedule(dynamic, 16)
    for (int64_t i = 0; i < n; ++i) {
        const float * a = xt + i * m;
        for (int64_t j = i; j < n; ++j) {
            const float * b = xt + j * m;
            float s = 0.0f;
#pragma omp simd reduction(+:s)
            for (int64_t k = 0; k < m; ++k) {
                s += a[k] * b[k];
            }
            h[i * n + j] += s;
        }
    }
    H.rows += m;
    return true;
}

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    const char * names = getenv("HESS_TENSORS");
    const char * out   = getenv("HESS_OUT");
    const char * text  = getenv("HEAD_TEXT");
    if (!names || !out || !text) {
        fprintf(stderr, "set HESS_TENSORS, HESS_OUT and HEAD_TEXT\n");
        return 1;
    }
    state st;
    {
        std::stringstream ss(names);
        for (std::string s; std::getline(ss, s, ',');) {
            if (!s.empty()) {
                st.want[s];
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
    const int32_t n_batch = (int32_t) llama_n_batch(ctx);
    const int32_t n_ctx   = (int32_t) llama_n_ctx(ctx);

    std::ifstream f(text);
    std::stringstream buf;
    buf << f.rdbuf();
    const std::vector<llama_token> toks = common_tokenize(ctx, buf.str(), true);
    const int n_max  = (int) toks.size() / n_ctx;
    const int chunk0 = env_int("HEAD_CHUNK0", 0);
    const int n      = std::min(env_int("HEAD_CHUNKS", n_max - chunk0), n_max - chunk0);
    fprintf(stderr, "%zu tokens, chunks %d..%d of %d tokens, %zu tensors\n", toks.size(), chunk0, chunk0 + n - 1, n_ctx,
            st.want.size());

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
    llama_backend_free();
    fprintf(stderr, "done\n");
    return 0;
}
