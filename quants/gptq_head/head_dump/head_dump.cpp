// llama-head-dump: run text through a model and record the output head's input, the final hidden state after the
// output norm (graph tensor "result_norm"), for every token. GPTQ calibration data for the output head.
//
//   HEAD_OUT=x.f32 HEAD_TEXT=train.txt [HEAD_CHUNK0=0 HEAD_CHUNKS=500] llama-head-dump -m model.gguf -c 512 -b 512 ...
//   HEAD_OUT=x.f32 HEAD_SEQS=seqs.txt llama-head-dump -m model.gguf -c 16384 -b 512 ...
//
// HEAD_TEXT: the text is tokenized as llama-perplexity does it and cut into chunks of n_ctx tokens; chunks
// HEAD_CHUNK0 .. HEAD_CHUNK0 + HEAD_CHUNKS - 1 are run, each from an empty KV cache (as llama-perplexity runs them).
// HEAD_SEQS: one sequence per line, "<seq_id> <n_prompt> <tok0> <tok1> ...", each from an empty KV cache.
// Every token requests its logits, because llama.cpp computes the last layer only for output rows.
// Output: HEAD_OUT holds float32 rows of n_embd values, one per token in order; HEAD_OUT.idx holds three int32 per
// row {chunk or seq_id, position, token id}.
// HEAD_LOGITS_EVERY=k (optional): also write the model's own logits for every token at a position p with p % k == k - 1
// to HEAD_OUT.logits (float32 rows of n_vocab values) and their row numbers in HEAD_OUT to HEAD_OUT.logits.idx
// (int64), to check that the dumped rows times the model's head reproduce them (gptq_head.py checkdump).

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
#include <sstream>
#include <string>
#include <vector>

struct dump_state {
    FILE * out = nullptr;
    int64_t n_rows = 0;
    std::vector<uint8_t> raw;
};

static bool head_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    const bool hit = strcmp(t->name, "result_norm") == 0;
    if (ask || !hit) {
        return ask ? hit : true;
    }
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "result_norm has type %s, expected f32\n", ggml_type_name(t->type));
        exit(1);
    }
    // [n_embd, n_outputs]
    const size_t nbytes = ggml_nbytes(t);
    st->raw.resize(nbytes);
    ggml_backend_tensor_get(t, st->raw.data(), 0, nbytes);
    for (int64_t i = 0; i < t->ne[1]; ++i) {
        fwrite(st->raw.data() + i * t->nb[1], sizeof(float), t->ne[0], st->out);
    }
    st->n_rows += t->ne[1];
    return true;
}

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    const char * out_path  = getenv("HEAD_OUT");
    const char * text_path = getenv("HEAD_TEXT");
    const char * seqs_path = getenv("HEAD_SEQS");
    if (out_path == nullptr || (text_path == nullptr) == (seqs_path == nullptr)) {
        fprintf(stderr, "set HEAD_OUT and one of HEAD_TEXT, HEAD_SEQS\n");
        return 1;
    }

    dump_state st;
    st.out = fopen(out_path, "wb");
    FILE * idx = fopen((std::string(out_path) + ".idx").c_str(), "wb");
    if (st.out == nullptr || idx == nullptr) {
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

    params.cb_eval = head_cb;
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
    const int32_t n_embd  = llama_model_n_embd(model);

    // the sequences to run: (id, tokens)
    std::vector<std::pair<int32_t, std::vector<llama_token>>> seqs;
    if (text_path) {
        std::ifstream f(text_path);
        std::stringstream ss;
        ss << f.rdbuf();
        const std::vector<llama_token> toks = common_tokenize(ctx, ss.str(), true);
        const int chunk0 = env_int("HEAD_CHUNK0", 0);
        const int n_max  = (int) toks.size() / n_ctx;
        const int n      = std::min(env_int("HEAD_CHUNKS", n_max - chunk0), n_max - chunk0);
        fprintf(stderr, "%zu tokens, %d chunks of %d; running chunks %d..%d\n", toks.size(), n_max, n_ctx, chunk0,
                chunk0 + n - 1);
        for (int c = chunk0; c < chunk0 + n; ++c) {
            seqs.emplace_back(c, std::vector<llama_token>(toks.begin() + (size_t) c * n_ctx,
                                                          toks.begin() + (size_t) (c + 1) * n_ctx));
        }
    } else {
        std::ifstream in(seqs_path);
        std::string line;
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
            seqs.emplace_back(seq_id, std::move(toks));
        }
    }

    const int logits_every = env_int("HEAD_LOGITS_EVERY", 0);
    const int32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE * lg     = nullptr;
    FILE * lg_idx = nullptr;
    if (logits_every > 0) {
        lg     = fopen((std::string(out_path) + ".logits").c_str(), "wb");
        lg_idx = fopen((std::string(out_path) + ".logits.idx").c_str(), "wb");
        if (lg == nullptr || lg_idx == nullptr) {
            fprintf(stderr, "cannot open %s.logits\n", out_path);
            return 1;
        }
    }

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    int64_t row0 = 0; // HEAD_OUT row of the batch's first token
    for (const auto & [id, toks] : seqs) {
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int32_t i0 = 0; i0 < (int32_t) toks.size(); i0 += n_batch) {
            const int32_t n = std::min<int32_t>(n_batch, (int32_t) toks.size() - i0);
            batch.n_tokens = n;
            for (int32_t j = 0; j < n; ++j) {
                batch.token[j]     = toks[i0 + j];
                batch.pos[j]       = i0 + j;
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = 1;
                const int32_t rec[3] = {id, i0 + j, toks[i0 + j]};
                fwrite(rec, sizeof(rec), 1, idx);
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "sequence %d: decode failed at %d\n", id, i0);
                return 1;
            }
            if (st.n_rows != row0 + n) {
                fprintf(stderr, "sequence %d: %lld rows dumped, expected %lld\n", id, (long long) st.n_rows,
                        (long long) (row0 + n));
                return 1;
            }
            for (int32_t j = 0; lg && j < n; ++j) {
                if ((i0 + j) % logits_every == logits_every - 1) {
                    const int64_t row = row0 + j;
                    fwrite(llama_get_logits_ith(ctx, j), sizeof(float), n_vocab, lg);
                    fwrite(&row, sizeof(row), 1, lg_idx);
                }
            }
            row0 += n;
        }
        fprintf(stderr, "sequence %d: %zu tokens, %lld rows so far\n", id, toks.size(), (long long) st.n_rows);
    }

    llama_batch_free(batch);
    fclose(st.out);
    fclose(idx);
    if (lg) {
        fclose(lg);
        fclose(lg_idx);
    }
    fprintf(stderr, "done: %zu sequences, %lld rows of %d floats\n", seqs.size(), (long long) st.n_rows, n_embd);
    llama_backend_free();
    return 0;
}
