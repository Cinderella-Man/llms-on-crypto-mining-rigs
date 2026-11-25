// C981 (Q1714): how far the prompt taper's cuts=1 moves a model's next-token distributions when prompt calls are read
// the way llama-server reads them with --ctx-checkpoints-incall.
//
// Each measured call is a stretch of N tokens of a text, decoded as llama-server's prompt call: one call of N tokens
// (n_batch >= N), cut inside at N - 4 - n_ubatch and N - 4 (llama_set_ubatch_cuts) with a checkpoint filled at each cut
// (llama_state_seq_store_at, read after the call), with the server's hints (head by default, tail since the call has an
// output); then D more tokens of the text, one token a call, as decoding reads them. Every position's logits are kept.
// With KLD_PREFIX = P, P tokens before the stretch are read first as a prompt call of their own (output on its last
// token only), so the measured call sits on a cached context as an agent's next turn does.
//
// Each arm (a taper spec, llama_set_ubatch_taper) reads every call on a cleared memory. Per position the tool writes
// each arm's log-probability of the text's next token and its top token, and per pair of arms (reference:test) the KL
// divergence of the test's distribution from the reference's (sum of p_ref * (log p_ref - log p_test), in double),
// whether the top tokens agree and whether the two logit rows are the same bits. Per call and arm it writes the
// micro-batches the context logged ("ubatch taper:" lines; "-" when the context ran the plain split and logged none).
//
// Build: run/build.sh (against a rig build's own build tree). Usage:
//   kld-cuts -m <gguf> -f <text> [llama.cpp common arguments: -ngl -fa -c -b -ub -ts -fit -lm --no-host -t]
// Env:
//   KLD_SIZES   call sizes, comma-separated (required); every size must leave n_batch >= size
//   KLD_CHUNKS  calls per size (default 8); calls take consecutive stretches of the text, the sizes in turn
//   KLD_DECODE  tokens decoded one a call after each call (default 16)
//   KLD_PREFIX  tokens read before each call as a prompt call of their own (default 0)
//   KLD_START   the first token of the text to use (default 0)
//   KLD_ARMS    name=spec;... (default off=off;v5=short=1400:64;v5c=short=1400:64,cuts=1;rep=short=1400:64)
//   KLD_PAIRS   ref:test,... (default v5:v5c,off:v5,off:v5c,v5:rep)
//   KLD_RS      the context's n_rs_seq (default 3: llama-server's with an MTP draft of three tokens)
//   KLD_OUT     output prefix (default kld): <OUT>.pos.tsv per position, <OUT>.calls.tsv per call and arm
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<std::string> g_taper; // the "ubatch taper:" lines the context logged since the last clear

static void log_cb(ggml_log_level level, const char * text, void *) {
    const char * p = strstr(text, "ubatch taper: ");
    if (p) {
        std::string s(p);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
            s.pop_back();
        }
        g_taper.push_back(s);
        return;
    }
    if (level != GGML_LOG_LEVEL_DEBUG) {
        fputs(text, stderr);
    }
}

static std::vector<std::string> split(const std::string & s, char d) {
    std::vector<std::string> out;
    size_t a = 0;
    while (a <= s.size()) {
        size_t b = s.find(d, a);
        if (b == std::string::npos) {
            b = s.size();
        }
        if (b > a) {
            out.push_back(s.substr(a, b - a));
        }
        a = b + 1;
    }
    return out;
}

static std::string env_str(const char * k, const char * dflt) {
    const char * v = getenv(k);
    return v && *v ? std::string(v) : std::string(dflt);
}

static long env_int(const char * k, long dflt) {
    const char * v = getenv(k);
    return v && *v ? strtol(v, nullptr, 10) : dflt;
}

struct arm_t {
    std::string name, spec;
};

struct arm_out {
    std::vector<float>       logits; // rows x n_vocab: the call's N positions, then the D decoded tokens
    std::vector<double>      lse;    // log-sum-exp of each row
    std::vector<llama_token> top;    // each row's top token
    std::vector<double>      lp;     // each row's log-probability of the text's next token
    std::vector<float>       pre;    // the prefix call's last logits (KLD_PREFIX > 0)
    std::string plan_pre, plan_call;
    int fills = 0;                   // in-call checkpoints filled and read
};

// one prompt call as llama-server makes it with --ctx-checkpoints-incall; keeps every row's logits (all) or the last
static bool prompt_call(llama_context * ctx, llama_batch_ext * batch, const llama_token * tok, uint32_t n, llama_pos pos0,
                        bool all, int n_vocab, std::vector<float> & keep, std::string & plan, int & fills) {
    const uint32_t n_ubatch = llama_n_ubatch(ctx);
    std::vector<uint32_t> cuts;
    for (uint32_t off : { 4 + n_ubatch, 4u }) {
        if (n > off) {
            cuts.push_back(n - off);
        }
    }
    llama_batch_ext_clear(batch);
    for (uint32_t i = 0; i < n; ++i) {
        const int32_t idx = llama_batch_ext_add_token(batch, 0, tok[i]);
        if (idx < 0) {
            fprintf(stderr, "kld-cuts: batch add failed (%d) at %u\n", idx, i);
            return false;
        }
        const llama_pos p = pos0 + (llama_pos) i;
        const llama_pos ps[4] = { p, p, p, 0 };
        llama_batch_ext_set_pos(batch, idx, ps);
        llama_batch_ext_set_output_logits(batch, idx, all || i == n - 1);
    }
    for (size_t k = 0; k < cuts.size(); ++k) {
        llama_state_seq_store_at(ctx, 0, (int32_t) k, cuts[k], LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    }
    llama_set_ubatch_taper_next(ctx, -1, 1);
    llama_set_ubatch_cuts(ctx, cuts.data(), cuts.size());
    g_taper.clear();
    const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    llama_set_ubatch_cuts(ctx, nullptr, 0);
    if (rc != 0) {
        fprintf(stderr, "kld-cuts: decode of a %u-token call failed (%d)\n", n, rc);
        return false;
    }
    plan = g_taper.empty() ? "-" : g_taper.back();
    for (size_t k = 0; k < cuts.size(); ++k) {
        const size_t sz = llama_state_seq_store_at_get(ctx, nullptr, 0, 0, (int32_t) k, nullptr, nullptr);
        if (sz > 0) {
            std::vector<uint8_t> blob(sz);
            if (llama_state_seq_store_at_get(ctx, blob.data(), sz, 0, (int32_t) k, nullptr, nullptr) == sz) {
                fills++;
            }
        }
        llama_state_seq_store_free(ctx, 0, (int32_t) k);
    }
    for (uint32_t i = all ? 0 : n - 1; i < n; ++i) {
        const float * l = llama_get_logits_ith(ctx, (int32_t) i);
        if (!l) {
            fprintf(stderr, "kld-cuts: no logits at %u\n", i);
            return false;
        }
        keep.insert(keep.end(), l, l + n_vocab);
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    llama_log_set(log_cb, nullptr);

    std::vector<uint32_t> sizes;
    for (const auto & s : split(env_str("KLD_SIZES", ""), ',')) {
        sizes.push_back((uint32_t) strtoul(s.c_str(), nullptr, 10));
    }
    const long K      = env_int("KLD_CHUNKS", 8);
    const long D      = env_int("KLD_DECODE", 16);
    const long P      = env_int("KLD_PREFIX", 0);
    const long START  = env_int("KLD_START", 0);
    const long RS     = env_int("KLD_RS", 3);
    const std::string out_prefix = env_str("KLD_OUT", "kld");
    std::vector<arm_t> arms;
    for (const auto & a : split(env_str("KLD_ARMS", "off=off;v5=short=1400:64;v5c=short=1400:64,cuts=1;rep=short=1400:64"), ';')) {
        const size_t e = a.find('=');
        if (e == std::string::npos) {
            fprintf(stderr, "kld-cuts: bad arm '%s'\n", a.c_str());
            return 1;
        }
        arms.push_back({ a.substr(0, e), a.substr(e + 1) });
    }
    std::vector<std::pair<int, int>> pairs;
    for (const auto & p : split(env_str("KLD_PAIRS", "v5:v5c,off:v5,off:v5c,v5:rep"), ',')) {
        const auto ab = split(p, ':');
        int ia = -1, ib = -1;
        for (size_t i = 0; ab.size() == 2 && i < arms.size(); ++i) {
            ia = arms[i].name == ab[0] ? (int) i : ia;
            ib = arms[i].name == ab[1] ? (int) i : ib;
        }
        if (ia < 0 || ib < 0) {
            fprintf(stderr, "kld-cuts: bad pair '%s'\n", p.c_str());
            return 1;
        }
        pairs.push_back({ ia, ib });
    }
    if (sizes.empty() || K < 1 || D < 0 || P < 0 || params.prompt.empty()) {
        fprintf(stderr, "kld-cuts: KLD_SIZES, a positive KLD_CHUNKS and a text (-f) are needed\n");
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    llama_model_params mp = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "kld-cuts: cannot load %s\n", params.model.path.c_str());
        return 1;
    }
    llama_context_params cp = common_context_params_to_llama(params);
    cp.n_rs_seq = (uint32_t) RS;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "kld-cuts: cannot create a context\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const uint32_t n_batch = llama_n_batch(ctx);
    for (uint32_t n : sizes) {
        if (n > n_batch || (P > 0 && (uint32_t) P > n_batch) || (uint32_t) (P + n + D + 1) > llama_n_ctx(ctx)) {
            fprintf(stderr, "kld-cuts: size %u does not fit n_batch %u / n_ctx %u\n", n, n_batch, llama_n_ctx(ctx));
            return 1;
        }
    }
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    fprintf(stderr, "kld-cuts: %zu text tokens, n_vocab %d, n_batch %u, n_ubatch %u, n_ctx %u, n_rs_seq %u, add_bos %d\n",
            tokens.size(), n_vocab, n_batch, llama_n_ubatch(ctx), llama_n_ctx(ctx), llama_n_rs_seq(ctx), (int) add_bos);

    FILE * fpos   = fopen((out_prefix + ".pos.tsv").c_str(), "w");
    FILE * fcalls = fopen((out_prefix + ".calls.tsv").c_str(), "w");
    if (!fpos || !fcalls) {
        fprintf(stderr, "kld-cuts: cannot write %s.*\n", out_prefix.c_str());
        return 1;
    }
    fprintf(fpos, "chunk\tsize\tround\trow\tkind\tnext");
    for (const auto & a : arms) {
        fprintf(fpos, "\tlp_%s\ttop_%s", a.name.c_str(), a.name.c_str());
    }
    for (const auto & pr : pairs) {
        const char * x = arms[pr.first].name.c_str();
        const char * y = arms[pr.second].name.c_str();
        fprintf(fpos, "\tkld_%s_%s\tsametop_%s_%s\tsamebits_%s_%s", x, y, x, y, x, y);
    }
    fprintf(fpos, "\n");
    fprintf(fcalls, "chunk\tsize\tround\tstart\tarm\tspec\tprefix_plan\tcall_plan\tfills\tprefix_samebits_first_arm\n");

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    size_t off = (size_t) START;
    int chunk = 0;
    for (long k = 0; k < K; ++k) {
        for (uint32_t N : sizes) {
            const size_t need = (size_t) P + N + D + 1;
            if (off + need > tokens.size()) {
                fprintf(stderr, "kld-cuts: the text ends at token %zu, chunk %d needs %zu from %zu\n", tokens.size(), chunk, need, off);
                return 1;
            }
            std::vector<llama_token> t(tokens.begin() + off, tokens.begin() + off + need);
            if (add_bos) {
                t[0] = llama_vocab_bos(vocab);
            }
            const llama_token * tc = t.data() + P; // the measured call's tokens
            const size_t rows = (size_t) N + D;
            std::vector<arm_out> res(arms.size());
            for (size_t ai = 0; ai < arms.size(); ++ai) {
                arm_out & o = res[ai];
                llama_memory_clear(llama_get_memory(ctx), true);
                if (!llama_set_ubatch_taper(ctx, arms[ai].spec.c_str())) {
                    fprintf(stderr, "kld-cuts: the context refused the taper spec '%s'\n", arms[ai].spec.c_str());
                    return 1;
                }
                o.logits.reserve(rows * n_vocab);
                if (P > 0 && !prompt_call(ctx, batch, t.data(), (uint32_t) P, 0, false, n_vocab, o.pre, o.plan_pre, o.fills)) {
                    return 1;
                }
                if (!prompt_call(ctx, batch, tc, N, (llama_pos) P, true, n_vocab, o.logits, o.plan_call, o.fills)) {
                    return 1;
                }
                for (long g = 0; g < D; ++g) {
                    llama_batch_ext_clear(batch);
                    const int32_t idx = llama_batch_ext_add_token(batch, 0, tc[N + g]);
                    const llama_pos p = (llama_pos) (P + N + g);
                    const llama_pos ps[4] = { p, p, p, 0 };
                    llama_batch_ext_set_pos(batch, idx, ps);
                    llama_batch_ext_set_output_logits(batch, idx, true);
                    llama_set_ubatch_taper_next(ctx, -1, 1);
                    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
                        fprintf(stderr, "kld-cuts: decode of token %ld failed\n", g);
                        return 1;
                    }
                    const float * l = llama_get_logits_ith(ctx, 0);
                    o.logits.insert(o.logits.end(), l, l + n_vocab);
                }
                o.lse.assign(rows, 0.0);
                o.top.assign(rows, 0);
                o.lp.assign(rows, 0.0);
                #pragma omp parallel for schedule(static)
                for (long r = 0; r < (long) rows; ++r) {
                    const float * x = o.logits.data() + (size_t) r * n_vocab;
                    int imax = 0;
                    for (int v = 1; v < n_vocab; ++v) {
                        if (x[v] > x[imax]) {
                            imax = v;
                        }
                    }
                    const double mx = x[imax];
                    double s = 0.0;
                    for (int v = 0; v < n_vocab; ++v) {
                        s += std::exp((double) x[v] - mx);
                    }
                    o.lse[r] = mx + std::log(s);
                    o.top[r] = imax;
                    o.lp[r]  = (double) x[tc[r + 1]] - o.lse[r];
                }
            }
            // pairs
            std::vector<std::vector<double>> kld(pairs.size(), std::vector<double>(rows));
            std::vector<std::vector<char>> same(pairs.size(), std::vector<char>(rows));
            for (size_t pi = 0; pi < pairs.size(); ++pi) {
                const arm_out & A = res[pairs[pi].first];
                const arm_out & B = res[pairs[pi].second];
                #pragma omp parallel for schedule(static)
                for (long r = 0; r < (long) rows; ++r) {
                    const float * xa = A.logits.data() + (size_t) r * n_vocab;
                    const float * xb = B.logits.data() + (size_t) r * n_vocab;
                    same[pi][r] = memcmp(xa, xb, sizeof(float) * n_vocab) == 0;
                    if (same[pi][r]) {
                        kld[pi][r] = 0.0;
                        continue;
                    }
                    const double la = A.lse[r], lb = B.lse[r];
                    double s = 0.0;
                    for (int v = 0; v < n_vocab; ++v) {
                        const double a = (double) xa[v] - la;
                        const double b = (double) xb[v] - lb;
                        s += std::exp(a) * (a - b);
                    }
                    kld[pi][r] = s;
                }
            }
            for (size_t r = 0; r < rows; ++r) {
                fprintf(fpos, "%d\t%u\t%ld\t%zu\t%s\t%d", chunk, N, k, r, r < N ? "p" : "d", tc[r + 1]);
                for (const auto & o : res) {
                    fprintf(fpos, "\t%.6f\t%d", o.lp[r], o.top[r]);
                }
                for (size_t pi = 0; pi < pairs.size(); ++pi) {
                    fprintf(fpos, "\t%.6e\t%d\t%d", kld[pi][r], (int) (res[pairs[pi].first].top[r] == res[pairs[pi].second].top[r]),
                            (int) same[pi][r]);
                }
                fprintf(fpos, "\n");
            }
            for (size_t ai = 0; ai < arms.size(); ++ai) {
                const int pre_same = P > 0 ? (int) (res[ai].pre == res[0].pre) : -1;
                fprintf(fcalls, "%d\t%u\t%ld\t%zu\t%s\t%s\t%s\t%s\t%d\t%d\n", chunk, N, k, off, arms[ai].name.c_str(),
                        arms[ai].spec.c_str(), res[ai].plan_pre.empty() ? "-" : res[ai].plan_pre.c_str(),
                        res[ai].plan_call.c_str(), res[ai].fills, pre_same);
            }
            fflush(fpos);
            fflush(fcalls);
            fprintf(stderr, "kld-cuts: chunk %d (size %u, round %ld, tokens %zu-%zu):", chunk, N, k, off, off + need - 1);
            for (size_t pi = 0; pi < pairs.size(); ++pi) {
                double s = 0.0;
                for (size_t r = 0; r < rows; ++r) {
                    s += kld[pi][r];
                }
                fprintf(stderr, " %s:%s %.6f", arms[pairs[pi].first].name.c_str(), arms[pairs[pi].second].name.c_str(), s / rows);
            }
            fprintf(stderr, "\n");
            off += need;
            chunk++;
        }
    }
    llama_batch_ext_free(batch);
    fclose(fpos);
    fclose(fcalls);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    fprintf(stderr, "kld-cuts: done, %d chunks\n", chunk);
    return 0;
}
