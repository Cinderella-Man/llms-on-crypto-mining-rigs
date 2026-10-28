// mtp_chain: acceptance of a model's built-in MTP (NextN) layer at draft positions 1, 2 and 3, teacher-forced on a
// text, through llama.cpp's own MTP graph. llama-server's draft-mtp driver chains drafts for a single-head model so:
// draft 1 reads the last token x_k at position k with the target's post-norm hidden state h_(k-1); draft j+1 reads
// draft j's token at the next position with the MTP layer's own post-norm hidden state from draft j, while the MTP
// layer's KV cache holds the true entries up to position k and the drafts after it. Here the draft tokens are the
// text's own (x_(k+1), x_(k+2)): each draft is scored given that the drafts before it were accepted.
// Per chunk of N tokens: the target decodes the chunk (logits and h at every position); sequence 0 of the MTP context
// reads every true entry (x_k, h_(k-1)) at once, which gives draft 1 at every k; then, S positions at a time, sequence
// s gets a copy of sequence 0's entries 0 .. k (a metadata copy in the unified KV cache) and reads draft 2's entry at
// k+1, then draft 3's at k+2. At every k from 1 to N-4, with p_(k+j-1) the target's distribution for x_(k+j):
//   a_j = sum_v min(p, q_j) under the makers' sampling chain (top-k 20, top-p 0.95, temperature 0.6) for both
//   t_j = the two argmaxes agree
//   E_n = expected drafts accepted per step with n drafts: a_1, a_1 + a_1 a_2, a_1 + a_1 a_2 + a_1 a_2 a_3
// aggregated over all positions and per bin of the target's top probability at position k (temperature 1).
//   mtp-chain -m MODEL -f TEXT -c CHUNK [--chunks N] -ngl 99 [-ts ...] [-fa on]   (results to stdout, TSV)
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
static const int S = 64; // chained positions per batch, one sequence each

struct stats {
    double n = 0, a[3] = {0, 0, 0}, t[3] = {0, 0, 0}, e[3] = {0, 0, 0};
    void add(const stats & o) {
        n += o.n;
        for (int j = 0; j < 3; j++) {
            a[j] += o.a[j]; t[j] += o.t[j]; e[j] += o.e[j];
        }
    }
};

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

// acceptance under the chain and top-1 agreement of target logits zt and draft logits zm
static void compare(const float * zt, const float * zm, int n, std::vector<int> & idx, double & acc, double & top1) {
    std::vector<std::pair<int, double>> pc, qc;
    sampling_chain(zt, n, idx, pc);
    sampling_chain(zm, n, idx, qc);
    acc = 0;
    for (const auto & e : pc) {
        for (const auto & f : qc) {
            if (f.first == e.first) {
                acc += std::min(e.second, f.second);
            }
        }
    }
    top1 = pc[0].first == qc[0].first;
}

static double top_prob(const float * z, int n) {
    const float m = *std::max_element(z, z + n);
    double s = 0;
    for (int i = 0; i < n; i++) {
        s += exp(z[i] - m);
    }
    return 1.0 / s;
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
    if (!model || llama_model_n_layer_nextn(model) < 1) {
        fprintf(stderr, "failed to load %s, or it has no MTP layer\n", params.model.path.c_str());
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
    dparams.n_seq_max = S + 1;
    dparams.kv_unified = true;
    dparams.n_ctx = n_ctx + 2 * S + 256;
    llama_context * dft = llama_init_from_model(model, dparams);
    if (!dft) {
        return 1;
    }
    llama_set_embeddings_nextn(dft, true, /*masked*/ true);
    llama_memory_t mem = llama_get_memory(dft);

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
    std::vector<float> h((size_t) n_ctx * n_embd), h1((size_t) n_ctx * n_embd), h2((size_t) S * n_embd);
    std::vector<float> zeros(n_embd, 0.0f);
    std::vector<double> a1(n_ctx), t1(n_ctx), pmax(n_ctx), a2(n_ctx), t2(n_ctx), a3(n_ctx), t3(n_ctx);

    for (int ic = 0; ic < n_chunk; ic++) {
        const llama_token * tok = toks.data() + (size_t) ic * n_ctx;
        llama_memory_clear(llama_get_memory(tgt), true);
        llama_memory_clear(mem, true);

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

        // draft 1 at every position: the true entries, sequence 0
        common_batch bd(dft);
        for (int k = 0; k < n_ctx; k++) {
            const int32_t idx = bd.add(tok[k], k, 0, true);
            bd.set_embd_state(idx, { k == 0 ? zeros.data() : h.data() + (size_t) (k - 1) * n_embd, 1, (size_t) n_embd });
        }
        if (llama_process(dft, LLAMA_PROCESS_TYPE_DECODE, bd.get()) != 0) {
            fprintf(stderr, "MTP decode failed at chunk %d\n", ic);
            return 1;
        }
        for (int k = 0; k < n_ctx; k++) {
            const float * hk = llama_get_embeddings_nextn_ith(dft, k);
            std::copy(hk, hk + n_embd, h1.begin() + (size_t) k * n_embd);
        }
        #pragma omp parallel
        {
            std::vector<int> idx(n_vocab);
            #pragma omp for schedule(dynamic, 16)
            for (int k = 1; k < n_ctx - 3; k++) {
                const float * zt = llama_get_logits_ith(tgt, k);
                compare(zt, llama_get_logits_ith(dft, k), n_vocab, idx, a1[k], t1[k]);
                pmax[k] = top_prob(zt, n_vocab);
            }
        }

        // drafts 2 and 3, S positions at a time
        for (int k0 = 1; k0 < n_ctx - 3; k0 += S) {
            const int ns = std::min(S, n_ctx - 3 - k0);
            for (int s = 0; s < ns; s++) {
                llama_memory_seq_rm(mem, s + 1, -1, -1);
                llama_memory_seq_cp(mem, 0, s + 1, 0, k0 + s + 1);
            }
            for (int depth = 2; depth <= 3; depth++) {
                common_batch bc(dft);
                for (int s = 0; s < ns; s++) {
                    const int k = k0 + s;
                    const int p = k + depth - 1; // position of this draft's input token, x_p
                    const float * hs = depth == 2 ? h1.data() + (size_t) k * n_embd : h2.data() + (size_t) s * n_embd;
                    const int32_t i = bc.add(tok[p], p, s + 1, true);
                    bc.set_embd_state(i, { hs, 1, (size_t) n_embd });
                }
                if (llama_process(dft, LLAMA_PROCESS_TYPE_DECODE, bc.get()) != 0) {
                    fprintf(stderr, "chained MTP decode failed at chunk %d, k %d\n", ic, k0);
                    return 1;
                }
                if (depth == 2) {
                    for (int s = 0; s < ns; s++) {
                        const float * hs = llama_get_embeddings_nextn_ith(dft, s);
                        std::copy(hs, hs + n_embd, h2.begin() + (size_t) s * n_embd);
                    }
                }
                #pragma omp parallel
                {
                    std::vector<int> idx(n_vocab);
                    #pragma omp for schedule(dynamic, 4)
                    for (int s = 0; s < ns; s++) {
                        const int k = k0 + s;
                        const float * zt = llama_get_logits_ith(tgt, k + depth - 1);
                        const float * zm = llama_get_logits_ith(dft, s);
                        if (depth == 2) {
                            compare(zt, zm, n_vocab, idx, a2[k], t2[k]);
                        } else {
                            compare(zt, zm, n_vocab, idx, a3[k], t3[k]);
                        }
                    }
                }
            }
            for (int s = 0; s < ns; s++) {
                llama_memory_seq_rm(mem, s + 1, -1, -1);
            }
        }

        std::vector<stats> chunk(N_BINS);
        for (int k = 1; k < n_ctx - 3; k++) {
            stats st;
            st.n = 1;
            st.a[0] = a1[k]; st.a[1] = a2[k]; st.a[2] = a3[k];
            st.t[0] = t1[k]; st.t[1] = t2[k]; st.t[2] = t3[k];
            st.e[0] = a1[k];
            st.e[1] = a1[k] + a1[k] * a2[k];
            st.e[2] = st.e[1] + a1[k] * a2[k] * a3[k];
            int b = 0;
            while (b < N_BINS - 1 && pmax[k] >= BINS[b + 1]) b++;
            chunk[b].add(st);
        }
        stats all;
        for (int b = 0; b < N_BINS; b++) {
            total[b].add(chunk[b]);
            all.add(chunk[b]);
        }
        fprintf(stderr, "chunk %d/%d: a1 %.4f a2 %.4f a3 %.4f E1 %.4f E2 %.4f E3 %.4f\n", ic + 1, n_chunk, all.a[0] / all.n,
                all.a[1] / all.n, all.a[2] / all.n, all.e[0] / all.n, all.e[1] / all.n, all.e[2] / all.n);
    }

    printf("bin_lo\tbin_hi\tpositions\tshare\ta1\ta2\ta3\ttop1_1\ttop1_2\ttop1_3\tE1\tE2\tE3\n");
    stats all;
    for (int b = 0; b < N_BINS; b++) all.add(total[b]);
    auto row = [&](const char * lo, const char * hi, const stats & s) {
        const double n = std::max(1.0, s.n);
        printf("%s\t%s\t%.0f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n", lo, hi, s.n,
               s.n / std::max(1.0, all.n), s.a[0] / n, s.a[1] / n, s.a[2] / n, s.t[0] / n, s.t[1] / n, s.t[2] / n,
               s.e[0] / n, s.e[1] / n, s.e[2] / n);
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
