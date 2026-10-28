// mtp_accept: how often a model's built-in MTP (NextN) layer would have its first draft accepted, teacher-forced on a
// text, through llama.cpp's own MTP graph (the one llama-server's draft-mtp speculation runs). Per chunk of N tokens:
// the target decodes the chunk, keeping the logits and the post-norm hidden state h (h_nextn) of every position; an MTP
// context then reads token x_k at position k with the target's h_(k-1), zeros for k = 0, as the server's draft-mtp
// driver pairs them. The MTP logits at k and the target's logits at k both predict x_(k+1). Positions 1 .. N-2 count:
//   acc_t1, acc_t06    expected coupled acceptance sum_v min(p_v, q_v), target p and MTP q at temperature 1 and 0.6
//   acc_chain          the same under the makers' sampling chain for both: top-k 20, top-p 0.95, then temperature 0.6
//   top1_agree         the two argmaxes agree
//   tgt_top1_true, mtp_top1_true   each argmax is the text's next token
// aggregated over all positions and per bin of the target's top probability at temperature 1.
//   mtp-accept -m MODEL -f TEXT -c CHUNK [--chunks N] -ngl 99 [-ts ...] [-fa on]   (results to stdout, TSV)
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

static const double BINS[] = {0.0, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0001};
static const int N_BINS = 6;
static const int TOP_K = 20;
static const float TOP_P = 0.95f;
static const float TEMP = 0.6f;

struct stats {
    double n = 0, acc_t1 = 0, acc_t06 = 0, acc_chain = 0, top1_agree = 0, tgt_true = 0, mtp_true = 0, pmax = 0;
    void add(const stats & o) {
        n += o.n; acc_t1 += o.acc_t1; acc_t06 += o.acc_t06; acc_chain += o.acc_chain;
        top1_agree += o.top1_agree; tgt_true += o.tgt_true; mtp_true += o.mtp_true; pmax += o.pmax;
    }
};

// softmax of z / t into p
static void softmax(const float * z, int n, float t, std::vector<float> & p) {
    const float m = *std::max_element(z, z + n);
    double s = 0;
    for (int i = 0; i < n; i++) {
        p[i] = expf((z[i] - m) / t);
        s += p[i];
    }
    const float inv = (float) (1.0 / s);
    for (int i = 0; i < n; i++) {
        p[i] *= inv;
    }
}

// the makers' chain as llama.cpp's sampler runs it: top-k, top-p over the kept tokens' probabilities, temperature last
static void sampling_chain(const float * z, int n, std::vector<int> & idx, std::vector<std::pair<int, double>> & out) {
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + TOP_K, idx.end(), [&](int a, int b) { return z[a] > z[b]; });
    const float m = z[idx[0]];
    double s = 0, c = 0;
    double pk[TOP_K];
    for (int i = 0; i < TOP_K; i++) {
        pk[i] = exp(z[idx[i]] - m);
        s += pk[i];
    }
    int keep = TOP_K;
    for (int i = 0; i < TOP_K; i++) {
        c += pk[i] / s;
        if (c >= TOP_P) {
            keep = i + 1;
            break;
        }
    }
    out.clear();
    s = 0;
    for (int i = 0; i < keep; i++) {
        const double w = exp((z[idx[i]] - m) / TEMP);
        out.emplace_back(idx[i], w);
        s += w;
    }
    for (auto & e : out) {
        e.second /= s;
    }
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx = 4096;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    const int n_ctx = params.n_ctx;
    params.n_batch = n_ctx;

    auto mparams = common_model_params_to_llama(params);
    mparams.load_mtp = true;
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", params.model.path.c_str());
        return 1;
    }
    if (llama_model_n_layer_nextn(model) < 1) {
        fprintf(stderr, "the model has no MTP layer\n");
        return 1;
    }
    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;
    cparams.n_seq_max = 1;
    llama_context * tgt = llama_init_from_model(model, cparams);
    if (!tgt) {
        return 1;
    }
    llama_set_embeddings_nextn(tgt, true, /*masked*/ false);

    auto dparams = cparams;
    dparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    dparams.n_rs_seq = 0;
    dparams.ctx_other = tgt;
    llama_context * dft = llama_init_from_model(model, dparams);
    if (!dft) {
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd = llama_model_n_embd(model);

    std::vector<llama_token> toks = common_tokenize(tgt, params.prompt, true);
    int n_chunk = (int) (toks.size() / n_ctx);
    if (params.n_chunks > 0) {
        n_chunk = std::min(n_chunk, params.n_chunks);
    }
    fprintf(stderr, "%zu tokens, %d chunks of %d, n_vocab %d, n_embd %d\n", toks.size(), n_chunk, n_ctx, n_vocab, n_embd);

    std::vector<stats> total(N_BINS);
    std::vector<float> h((size_t) n_ctx * n_embd), zeros(n_embd, 0.0f);

    for (int ic = 0; ic < n_chunk; ic++) {
        const llama_token * tok = toks.data() + (size_t) ic * n_ctx;
        llama_memory_clear(llama_get_memory(tgt), true);
        llama_memory_clear(llama_get_memory(dft), true);

        common_batch bt(tgt);
        for (int k = 0; k < n_ctx; k++) {
            bt.add(tok[k], k, 0, true);
        }
        if (llama_process(tgt, LLAMA_PROCESS_TYPE_DECODE, bt.get()) != 0) {
            fprintf(stderr, "target decode failed at chunk %d\n", ic);
            return 1;
        }
        for (int k = 0; k < n_ctx; k++) {
            const float * hk = llama_get_embeddings_nextn_ith(tgt, k);
            std::copy(hk, hk + n_embd, h.begin() + (size_t) k * n_embd);
        }

        common_batch bd(dft);
        for (int k = 0; k < n_ctx; k++) {
            const int32_t idx = bd.add(tok[k], k, 0, true);
            bd.set_embd_state(idx, { k == 0 ? zeros.data() : h.data() + (size_t) (k - 1) * n_embd, 1, (size_t) n_embd });
        }
        if (llama_process(dft, LLAMA_PROCESS_TYPE_DECODE, bd.get()) != 0) {
            fprintf(stderr, "MTP decode failed at chunk %d\n", ic);
            return 1;
        }

        std::vector<stats> chunk(N_BINS);
        #pragma omp parallel
        {
            std::vector<float> p(n_vocab), q(n_vocab);
            std::vector<int> idx(n_vocab);
            std::vector<std::pair<int, double>> pc, qc;
            std::vector<stats> local(N_BINS);
            #pragma omp for schedule(dynamic, 16)
            for (int k = 1; k < n_ctx - 1; k++) {
                const float * zt = llama_get_logits_ith(tgt, k);
                const float * zm = llama_get_logits_ith(dft, k);
                stats s;
                s.n = 1;
                softmax(zt, n_vocab, 1.0f, p);
                softmax(zm, n_vocab, 1.0f, q);
                double a = 0;
                int at = 0, am = 0;
                for (int v = 0; v < n_vocab; v++) {
                    a += std::min(p[v], q[v]);
                    if (p[v] > p[at]) at = v;
                    if (q[v] > q[am]) am = v;
                }
                const double pmax = p[at];
                s.acc_t1 = a;
                s.pmax = pmax;
                s.top1_agree = at == am;
                s.tgt_true = at == tok[k + 1];
                s.mtp_true = am == tok[k + 1];
                softmax(zt, n_vocab, TEMP, p);
                softmax(zm, n_vocab, TEMP, q);
                a = 0;
                for (int v = 0; v < n_vocab; v++) {
                    a += std::min(p[v], q[v]);
                }
                s.acc_t06 = a;
                sampling_chain(zt, n_vocab, idx, pc);
                sampling_chain(zm, n_vocab, idx, qc);
                a = 0;
                for (const auto & e : pc) {
                    for (const auto & f : qc) {
                        if (f.first == e.first) {
                            a += std::min(e.second, f.second);
                        }
                    }
                }
                s.acc_chain = a;
                int b = 0;
                while (b < N_BINS - 1 && pmax >= BINS[b + 1]) b++;
                local[b].add(s);
            }
            #pragma omp critical
            for (int b = 0; b < N_BINS; b++) chunk[b].add(local[b]);
        }
        stats all;
        for (int b = 0; b < N_BINS; b++) {
            total[b].add(chunk[b]);
            all.add(chunk[b]);
        }
        fprintf(stderr, "chunk %d/%d: acc_t1 %.4f acc_t06 %.4f acc_chain %.4f top1_agree %.4f\n", ic + 1, n_chunk,
                all.acc_t1 / all.n, all.acc_t06 / all.n, all.acc_chain / all.n, all.top1_agree / all.n);
    }

    printf("bin_lo\tbin_hi\tpositions\tshare\tacc_t1\tacc_t06\tacc_chain\ttop1_agree\ttgt_top1_true\tmtp_top1_true\tmean_pmax\n");
    stats all;
    for (int b = 0; b < N_BINS; b++) all.add(total[b]);
    auto row = [&](const char * lo, const char * hi, const stats & s) {
        const double n = std::max(1.0, s.n);
        printf("%s\t%s\t%.0f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n", lo, hi, s.n, s.n / std::max(1.0, all.n),
               s.acc_t1 / n, s.acc_t06 / n, s.acc_chain / n, s.top1_agree / n, s.tgt_true / n, s.mtp_true / n, s.pmax / n);
    };
    row("all", "", all);
    for (int b = 0; b < N_BINS; b++) {
        char lo[16], hi[16];
        snprintf(lo, sizeof(lo), "%.2f", BINS[b]);
        snprintf(hi, sizeof(hi), "%.2f", std::min(1.0, BINS[b + 1]));
        row(lo, hi, total[b]);
    }

    llama_free(dft);
    llama_free(tgt);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
