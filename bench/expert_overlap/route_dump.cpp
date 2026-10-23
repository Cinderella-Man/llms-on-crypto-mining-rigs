// llama-route-dump: run token sequences through a MoE model and record, for every MoE layer and position, the
// experts the router picked (ffn_moe_topk) and their router probabilities (ffn_moe_weights, before renormalization).
//
//   ROUTE_IN=seqs.txt ROUTE_OUT=routes.bin llama-route-dump -m model.gguf -ngl 99 -c 16384 [other common flags]
//
// seqs.txt: one sequence per line, "<seq_id> <n_prompt> <tok0> <tok1> ...", token ids as the model's vocabulary has
// them (prompt tokens first, then generated tokens). Each sequence starts from an empty KV cache. Every token
// requests its logits, because llama.cpp computes the last layer only for output rows: keep -b small (512).
// routes.bin: records of six int32 {kind, seq_id, layer, pos0, n_tok, k} followed by a payload:
//   kind 0: int32 expert ids [n_tok][k]; kind 1: float32 router probabilities [n_tok][k];
//   kind 2: sequence header with layer = -1, pos0 = n_prompt, n_tok = sequence length, k = 0, no payload.

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

struct dump_state {
    FILE * out = nullptr;
    int32_t seq_id = 0;
    std::map<int, int32_t> next_pos[2]; // per kind, per layer: position of the next ubatch's first token
    std::vector<uint8_t> raw;
    std::vector<int32_t> ids;
    std::vector<float> w;
};

// layer index if `name` is exactly "<prefix><digits>", else -1
static int layer_of(const char * name, const char * prefix) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0) {
        return -1;
    }
    const char * p = name + n;
    char * end = nullptr;
    const long il = strtol(p, &end, 10);
    if (end == p || *end != '\0') {
        return -1;
    }
    return (int) il;
}

static void write_header(FILE * f, int32_t kind, int32_t seq, int32_t layer, int32_t pos0, int32_t n_tok, int32_t k) {
    const int32_t h[6] = {kind, seq, layer, pos0, n_tok, k};
    fwrite(h, sizeof(h), 1, f);
}

static bool route_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    const int il_ids = layer_of(t->name, "ffn_moe_topk-");
    const int il_w   = layer_of(t->name, "ffn_moe_weights-");
    if (ask) {
        return il_ids >= 0 || il_w >= 0;
    }
    if (il_ids < 0 && il_w < 0) {
        return true;
    }

    const int kind = il_ids >= 0 ? 0 : 1;
    const int il   = il_ids >= 0 ? il_ids : il_w;
    // ids: [k, n_tok] (a view of the argsort, rows n_expert apart); probabilities: [1, k, n_tok]
    const int64_t k     = kind == 0 ? t->ne[0] : t->ne[1];
    const int64_t n_tok = kind == 0 ? t->ne[1] : t->ne[2];
    const size_t  s_k   = kind == 0 ? t->nb[0] : t->nb[1];
    const size_t  s_tok = kind == 0 ? t->nb[1] : t->nb[2];

    const size_t nbytes = ggml_nbytes(t);
    st->raw.resize(nbytes);
    ggml_backend_tensor_get(t, st->raw.data(), 0, nbytes);

    int32_t & pos0 = st->next_pos[kind][il];
    write_header(st->out, kind, st->seq_id, il, pos0, (int32_t) n_tok, (int32_t) k);
    if (kind == 0) {
        st->ids.resize(k * n_tok);
        for (int64_t i = 0; i < n_tok; ++i) {
            for (int64_t j = 0; j < k; ++j) {
                memcpy(&st->ids[i * k + j], st->raw.data() + i * s_tok + j * s_k, sizeof(int32_t));
            }
        }
        fwrite(st->ids.data(), sizeof(int32_t), st->ids.size(), st->out);
    } else {
        st->w.resize(k * n_tok);
        for (int64_t i = 0; i < n_tok; ++i) {
            for (int64_t j = 0; j < k; ++j) {
                memcpy(&st->w[i * k + j], st->raw.data() + i * s_tok + j * s_k, sizeof(float));
            }
        }
        fwrite(st->w.data(), sizeof(float), st->w.size(), st->out);
    }
    pos0 += (int32_t) n_tok;
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    const char * in_path  = getenv("ROUTE_IN");
    const char * out_path = getenv("ROUTE_OUT");
    if (in_path == nullptr || out_path == nullptr) {
        fprintf(stderr, "set ROUTE_IN and ROUTE_OUT\n");
        return 1;
    }

    dump_state st;
    st.out = fopen(out_path, "wb");
    if (st.out == nullptr) {
        fprintf(stderr, "cannot open %s\n", out_path);
        return 1;
    }

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval = route_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "failed to load the model\n");
        return 1;
    }

    const int32_t n_batch = (int32_t) llama_n_batch(ctx);
    const int32_t n_ctx   = (int32_t) llama_n_ctx(ctx);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    std::ifstream in(in_path);
    std::string line;
    int n_seq = 0;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        int32_t seq_id = 0, n_prompt = 0;
        if (!(ss >> seq_id >> n_prompt)) {
            continue;
        }
        std::vector<llama_token> toks;
        for (llama_token t; ss >> t;) {
            toks.push_back(t);
        }
        if ((int32_t) toks.size() > n_ctx) {
            fprintf(stderr, "sequence %d: %zu tokens > n_ctx %d, cut\n", seq_id, toks.size(), n_ctx);
            toks.resize(n_ctx);
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        st.seq_id = seq_id;
        st.next_pos[0].clear();
        st.next_pos[1].clear();
        write_header(st.out, 2, seq_id, -1, n_prompt, (int32_t) toks.size(), 0);

        for (int32_t i0 = 0; i0 < (int32_t) toks.size(); i0 += n_batch) {
            const int32_t n = std::min<int32_t>(n_batch, (int32_t) toks.size() - i0);
            batch.n_tokens = n;
            for (int32_t j = 0; j < n; ++j) {
                batch.token[j]     = toks[i0 + j];
                batch.pos[j]       = i0 + j;
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = 1; // the last layer runs its FFN only for output rows, so every token is one
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "sequence %d: decode failed at %d\n", seq_id, i0);
                return 1;
            }
        }
        ++n_seq;
        fprintf(stderr, "sequence %d: %zu tokens (%d prompt)\n", seq_id, toks.size(), n_prompt);
    }

    llama_batch_free(batch);
    fclose(st.out);
    fprintf(stderr, "done: %d sequences\n", n_seq);
    llama_backend_free();
    return 0;
}
