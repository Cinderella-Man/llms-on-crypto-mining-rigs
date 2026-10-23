// llama-logit-dump: run token sequences through a model and record, for every position whose next token is a
// generated one, the K most likely next tokens with their log-probabilities over the whole vocabulary (before any
// sampler). pipeline_sim.py applies the sampler chain to them for Gumbel-coupled guessing (Q96).
//
//   LOGIT_IN=seqs.txt LOGIT_OUT=top.bin LOGIT_K=20 llama-logit-dump -m model.gguf -ngl 99 -c 16384 [other common flags]
//
// seqs.txt: one sequence per line, "<seq_id> <n_prompt> <tok0> <tok1> ...", prompt tokens first, then generated ones
// (bench/expert_overlap/generate.py writes it). Each sequence starts from an empty KV cache.
// top.bin: per sequence a header of four int32 {seq_id, n_prompt, n_gen, k}, then n_gen records of k pairs
// {int32 token, float32 log-probability}, sorted by probability; record i predicts generated token i.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    const char * in_path  = getenv("LOGIT_IN");
    const char * out_path = getenv("LOGIT_OUT");
    const int    k        = getenv("LOGIT_K") ? atoi(getenv("LOGIT_K")) : 20;
    if (in_path == nullptr || out_path == nullptr || k <= 0) {
        fprintf(stderr, "set LOGIT_IN and LOGIT_OUT (and optionally LOGIT_K)\n");
        return 1;
    }
    FILE * out = fopen(out_path, "wb");
    if (out == nullptr) {
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
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "failed to load the model\n");
        return 1;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int32_t n_batch = (int32_t) llama_n_batch(ctx);
    const int32_t n_ctx   = (int32_t) llama_n_ctx(ctx);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    std::vector<int32_t> order(n_vocab);

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
        const int32_t n_tok = (int32_t) toks.size();
        const int32_t n_gen = n_tok - n_prompt;
        const int32_t hdr[4] = {seq_id, n_prompt, n_gen, k};
        fwrite(hdr, sizeof(hdr), 1, out);

        llama_memory_clear(llama_get_memory(ctx), true);
        // position p predicts token p + 1: the rows n_prompt - 1 .. n_tok - 2 predict the generated tokens
        for (int32_t i0 = 0; i0 < n_tok - 1; i0 += n_batch) {
            const int32_t n = std::min<int32_t>(n_batch, n_tok - 1 - i0);
            batch.n_tokens = n;
            for (int32_t j = 0; j < n; ++j) {
                batch.token[j]     = toks[i0 + j];
                batch.pos[j]       = i0 + j;
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = i0 + j >= n_prompt - 1;
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "sequence %d: decode failed at %d\n", seq_id, i0);
                return 1;
            }
            for (int32_t j = 0; j < n; ++j) {
                if (!batch.logits[j]) {
                    continue;
                }
                const float * logits = llama_get_logits_ith(ctx, j);
                const float mx = *std::max_element(logits, logits + n_vocab);
                double sum = 0.0;
                for (int32_t t = 0; t < n_vocab; ++t) {
                    sum += std::exp((double) logits[t] - mx);
                }
                const float lse = mx + (float) std::log(sum);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + k, order.end(),
                                  [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
                for (int i = 0; i < k; ++i) {
                    fwrite(&order[i], sizeof(int32_t), 1, out);
                    const float lp = logits[order[i]] - lse;
                    fwrite(&lp, sizeof(float), 1, out);
                }
            }
        }
        ++n_seq;
        fprintf(stderr, "sequence %d: %d tokens (%d prompt)\n", seq_id, n_tok, n_prompt);
    }

    llama_batch_free(batch);
    fclose(out);
    fprintf(stderr, "done: %d sequences\n", n_seq);
    llama_backend_free();
    return 0;
}
