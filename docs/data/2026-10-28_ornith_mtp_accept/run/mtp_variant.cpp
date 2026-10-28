// mtp_variant: run/mtp_accept.cpp's first-draft measurement with a cheaper draft path. The target is -m; the MTP draft
// can come from another GGUF (DRAFT), as llama-server's draft-mtp does with a draft model file: the MTP graph reads that
// file's MTP block, token embedding, head norm and output head, so a variant is a file with those tensors in other
// types. Environment:
//   DRAFT=<gguf>        the draft model (default: the target itself)
//   DRAFT_KV=<k=t:v>    a metadata override for the draft model only, e.g. qwen35moe.expert_used_count=int:4
//   DRAFT_CPU=<regex>   draft tensors kept in host memory (the draft file's trunk, which the MTP graph never reads)
//   SHORTLIST=<ids>     draft logits outside these token ids are dropped (FR-Spec style); the target keeps all
//   SCREEN=<N>          a screened draft head: the draft head ranks every token, and the target file's own output
//                       rows give exact logits for its top N, from the MTP hidden state (dequantized on the CPU)
//   FAST=1              only the makers'-chain acceptance, top-1 agreement and hit rates (acc_t1, acc_t06 print 0)
// Output and statistics as in run/mtp_accept.cpp.
//   mtp-variant -m MODEL -f TEXT -c CHUNK [--chunks N] -ngl 99 [-ts ...] [-fa on]   (results to stdout, TSV)
#include "arg.h"
#include "common.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"
#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
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

// the output head of a GGUF file in host memory, for exact logits of chosen rows
struct head_rows {
    std::vector<uint8_t> data;
    ggml_type type = GGML_TYPE_F32;
    size_t row_bytes = 0;
    int n_embd = 0;
    ggml_to_float_t to_float = nullptr;

    bool load(const char * path, int n_embd_) {
        gguf_init_params ip = { /*no_alloc*/ true, /*ctx*/ nullptr };
        gguf_context * g = gguf_init_from_file(path, ip);
        if (!g) {
            return false;
        }
        const int64_t id = gguf_find_tensor(g, "output.weight");
        if (id < 0) {
            gguf_free(g);
            return false;
        }
        type = gguf_get_tensor_type(g, id);
        n_embd = n_embd_;
        row_bytes = ggml_row_size(type, n_embd);
        const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
        const size_t size = gguf_get_tensor_size(g, id);
        gguf_free(g);
        FILE * f = fopen(path, "rb");
        if (!f) {
            return false;
        }
        data.resize(size);
        fseeko(f, (off_t) off, SEEK_SET);
        const bool ok = fread(data.data(), 1, size, f) == size;
        fclose(f);
        to_float = ggml_get_type_traits(type)->to_float;
        return ok && (to_float || type == GGML_TYPE_F32);
    }

    float logit(int row, const float * h, std::vector<float> & tmp) const {
        const uint8_t * r = data.data() + (size_t) row * row_bytes;
        const float * w = (const float *) r;
        if (type != GGML_TYPE_F32) {
            to_float(r, tmp.data(), n_embd);
            w = tmp.data();
        }
        double s = 0;
        for (int i = 0; i < n_embd; i++) {
            s += (double) w[i] * h[i];
        }
        return (float) s;
    }
};

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

    // the draft model: the target, or another file with its own overrides
    llama_model * model_dft = model;
    const char * draft_path = getenv("DRAFT");
    std::vector<llama_model_kv_override> kv;
    std::vector<llama_model_tensor_buft_override> bo;
    std::string cpu_regex;
    if (draft_path && *draft_path) {
        auto dmp = common_model_params_to_llama(params);
        dmp.load_mtp = true;
        if (const char * o = getenv("DRAFT_KV")) {
            if (!string_parse_kv_override(o, kv)) {
                fprintf(stderr, "bad DRAFT_KV %s\n", o);
                return 1;
            }
            kv.emplace_back();
            kv.back().key[0] = 0;
            dmp.kv_overrides = kv.data();
        }
        if (const char * c = getenv("DRAFT_CPU")) {
            cpu_regex = c;
            bo.push_back({ cpu_regex.c_str(), ggml_backend_cpu_buffer_type() });
            bo.push_back({ nullptr, nullptr });
            dmp.tensor_buft_overrides = bo.data();
        }
        model_dft = llama_model_load_from_file(draft_path, dmp);
        if (!model_dft || llama_model_n_layer_nextn(model_dft) < 1) {
            fprintf(stderr, "failed to load the draft %s, or it has no MTP layer\n", draft_path);
            return 1;
        }
    }
    auto dparams = cparams;
    dparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    dparams.n_rs_seq = 0;
    dparams.ctx_other = tgt;
    llama_context * dft = llama_init_from_model(model_dft, dparams);
    if (!dft) {
        return 1;
    }
    const int screen = getenv("SCREEN") ? atoi(getenv("SCREEN")) : 0;
    const bool fast = getenv("FAST") && atoi(getenv("FAST")) > 0;
    head_rows exact;
    if (screen > 0) {
        llama_set_embeddings_nextn(dft, true, /*masked*/ false);
        if (!exact.load(params.model.path.c_str(), llama_model_n_embd(model))) {
            fprintf(stderr, "cannot read the target's output.weight for SCREEN\n");
            return 1;
        }
        fprintf(stderr, "screened head: top %d of the draft head rescored with the target's %s rows\n", screen,
                ggml_type_name(exact.type));
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

    std::vector<char> keep;
    if (const char * sl = getenv("SHORTLIST")) {
        keep.assign(n_vocab, 0);
        FILE * f = fopen(sl, "r");
        int id, n = 0;
        while (f && fscanf(f, "%d", &id) == 1) {
            if (id >= 0 && id < n_vocab && !keep[id]) {
                keep[id] = 1;
                n++;
            }
        }
        if (f) fclose(f);
        fprintf(stderr, "shortlist %s: %d token ids\n", sl, n);
        if (n == 0) {
            return 1;
        }
    }

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
            std::vector<float> p(n_vocab), q(n_vocab), zbuf(n_vocab), tmp(n_embd);
            std::vector<int> idx(n_vocab);
            std::vector<std::pair<int, double>> pc, qc;
            std::vector<stats> local(N_BINS);
            #pragma omp for schedule(dynamic, 16)
            for (int k = 1; k < n_ctx - 1; k++) {
                const float * zt = llama_get_logits_ith(tgt, k);
                const float * zm = llama_get_logits_ith(dft, k);
                if (!keep.empty() || screen > 0) {
                    std::copy(zm, zm + n_vocab, zbuf.begin());
                    if (!keep.empty()) {
                        for (int v = 0; v < n_vocab; v++) {
                            if (!keep[v]) zbuf[v] = -INFINITY;
                        }
                    }
                    if (screen > 0) {
                        // the screen's top N get exact logits from the target's head rows
                        std::iota(idx.begin(), idx.end(), 0);
                        std::partial_sort(idx.begin(), idx.begin() + screen, idx.end(),
                                          [&](int a, int b) { return zbuf[a] > zbuf[b]; });
                        const float * hm = llama_get_embeddings_nextn_ith(dft, k);
                        for (int i = 0; i < screen; i++) {
                            zbuf[idx[i]] = exact.logit(idx[i], hm, tmp);
                        }
                    }
                    zm = zbuf.data();
                }
                stats s;
                s.n = 1;
                double a = 0, pmax;
                int at = 0, am = 0;
                if (fast) {
                    // the makers' chain only: argmaxes and the target's top probability, no full softmaxes
                    for (int v = 1; v < n_vocab; v++) {
                        if (zt[v] > zt[at]) at = v;
                        if (zm[v] > zm[am]) am = v;
                    }
                    double sum = 0;
                    for (int v = 0; v < n_vocab; v++) sum += expf(zt[v] - zt[at]);
                    pmax = 1.0 / sum;
                } else {
                    softmax(zt, n_vocab, 1.0f, p);
                    softmax(zm, n_vocab, 1.0f, q);
                    for (int v = 0; v < n_vocab; v++) {
                        a += std::min(p[v], q[v]);
                        if (p[v] > p[at]) at = v;
                        if (q[v] > q[am]) am = v;
                    }
                    pmax = p[at];
                    s.acc_t1 = a;
                    softmax(zt, n_vocab, TEMP, p);
                    softmax(zm, n_vocab, TEMP, q);
                    a = 0;
                    for (int v = 0; v < n_vocab; v++) {
                        a += std::min(p[v], q[v]);
                    }
                    s.acc_t06 = a;
                }
                s.pmax = pmax;
                s.top1_agree = at == am;
                s.tgt_true = at == tok[k + 1];
                s.mtp_true = am == tok[k + 1];
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
    if (model_dft != model) {
        llama_model_free(model_dft);
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
