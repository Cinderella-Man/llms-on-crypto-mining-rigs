// C1128 (Q1861): how far a verify pass's logits drift from one-token passes, over whole replies.
//
// Each request is a prompt (a text file, tokenized as llama-server's /completion tokenizes a prompt: special tokens
// added and parsed) followed by a reply given as token ids. One run of the tool is one load of the model reading
// every token after the prompt KLD_STEP tokens a call, with every position's logits kept: a drafting server's verify
// pass reads four tokens a call (the accepted token and three drafted ones), plain decoding reads one.
// Two parts per request, each on a cleared memory:
//   reply  the prompt read as a llama-server prompt call (calls of n_batch tokens, the last one cut at N - 4 - n_ubatch
//          and N - 4 with a checkpoint filled at each cut, as --ctx-checkpoints-incall does, with the server's hints),
//          then the reply KLD_STEP tokens a call. Row 0 is the prompt's last position; row i (1..R) is the position of
//          reply token i - 1, so row i predicts reply token i (row R predicts nothing in the text)
//   whole  the prompt and the reply from the first token, KLD_STEP tokens a call (how llama-perplexity reads a text at
//          -ub KLD_STEP); row i is position i and predicts token i + 1
// KLD_MODE=base writes every row's logits to KLD_BASE (the reference load). KLD_MODE=test reads them back row by row
// and writes, per row, the KL divergence of this load's distribution from the reference's (sum over the whole
// vocabulary of p_ref * (log p_ref - log p_test), in double), whether the top tokens agree, whether the two logit rows
// are the same bits, the largest logit difference, and how far the gap between the reference's two most likely tokens
// moved: (p_test(a) - p_test(b)) - (p_ref(a) - p_ref(b)), the measure C964 gave its three flips, and the same move in
// logits, (x_test(a) - x_test(b)) - (x_ref(a) - x_ref(b)), which does not depend on how likely the two tokens are.
//
// Build: run/build.sh (against a rig build's own build tree). Usage:
//   verify-drift -m <gguf> [llama.cpp common arguments: -ngl -fa -c -b -ub -ts -fit -lm --no-host -t]
// Env:
//   KLD_REQS   name=prompt_file=reply_ids_file;... (required; the ids file holds comma-separated token ids)
//   KLD_PARTS  reply,whole (default both)
//   KLD_STEP   tokens per call after the prompt (default 1)
//   KLD_RS     the context's n_rs_seq (default 0, llama-server without a draft; 3 with the preset's MTP draft of three)
//   KLD_TAPER  the micro-batch taper (llama_set_ubatch_taper; default short=1400:64, the v5 preset's)
//   KLD_MODE   base or test (required)
//   KLD_BASE   the reference rows file (written in base mode, read in test mode)
//   KLD_OUT    output prefix (default drift): <OUT>.rows.tsv per row, <OUT>.reqs.tsv per request
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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

static bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

struct req_t {
    std::string name;
    std::vector<llama_token> toks; // prompt, then reply
    size_t n_prompt = 0;
};

struct row_stats {
    double lse = 0.0, p1 = 0.0, p2 = 0.0, lp = NAN;
    int top1 = 0, top2 = 0;
};

static row_stats stats(const float * x, int n_vocab, llama_token next) {
    row_stats r;
    int i1 = 0, i2 = -1;
    for (int v = 1; v < n_vocab; ++v) {
        if (x[v] > x[i1]) {
            i2 = i1;
            i1 = v;
        } else if (i2 < 0 || x[v] > x[i2]) {
            i2 = v;
        }
    }
    const double mx = x[i1];
    double s = 0.0;
    for (int v = 0; v < n_vocab; ++v) {
        s += std::exp((double) x[v] - mx);
    }
    r.lse  = mx + std::log(s);
    r.top1 = i1;
    r.top2 = i2;
    r.p1   = std::exp((double) x[i1] - r.lse);
    r.p2   = std::exp((double) x[i2] - r.lse);
    if (next >= 0) {
        r.lp = (double) x[next] - r.lse;
    }
    return r;
}

struct row_hdr {
    int32_t part, req, row, next;
};

struct state_t {
    bool   test = false;
    int    n_vocab = 0;
    FILE * base = nullptr;
    FILE * rows = nullptr;
    std::vector<float> ref;
    long n_rows = 0, n_same = 0, n_sametop = 0;
    double kld_sum = 0.0;
    bool ok = true;
};

static void process_row(state_t & st, int part, int req, int row, long pos, llama_token next, const float * x) {
    const row_stats a = stats(x, st.n_vocab, next);
    fprintf(st.rows, "%s\t%d\t%d\t%ld\t%d\t%.6f\t%d\t%.6f\t%d\t%.6f", part == 0 ? "reply" : "whole", req, row, pos,
            (int) next, a.lp, a.top1, a.p1, a.top2, a.p2);
    const row_hdr h = { part, req, row, next };
    if (!st.test) {
        if (fwrite(&h, sizeof(h), 1, st.base) != 1 || fwrite(x, sizeof(float), st.n_vocab, st.base) != (size_t) st.n_vocab) {
            fprintf(stderr, "verify-drift: write to the base failed\n");
            st.ok = false;
        }
        fprintf(st.rows, "\n");
        st.n_rows++;
        return;
    }
    row_hdr hb;
    if (fread(&hb, sizeof(hb), 1, st.base) != 1 || fread(st.ref.data(), sizeof(float), st.n_vocab, st.base) != (size_t) st.n_vocab) {
        fprintf(stderr, "verify-drift: the base ends before row %d of part %d, request %d\n", row, part, req);
        st.ok = false;
        fprintf(st.rows, "\n");
        return;
    }
    if (memcmp(&hb, &h, sizeof(h)) != 0) {
        fprintf(stderr, "verify-drift: the base's row (%d %d %d %d) is not this row (%d %d %d %d)\n", hb.part, hb.req,
                hb.row, hb.next, h.part, h.req, h.row, h.next);
        st.ok = false;
    }
    const float * xr = st.ref.data();
    const row_stats b = stats(xr, st.n_vocab, next);
    const bool same = memcmp(xr, x, sizeof(float) * st.n_vocab) == 0;
    double kld = 0.0, maxdl = 0.0;
    if (!same) {
        for (int v = 0; v < st.n_vocab; ++v) {
            const double lr = (double) xr[v] - b.lse;
            const double lt = (double) x[v] - a.lse;
            kld += std::exp(lr) * (lr - lt);
            maxdl = std::max(maxdl, std::fabs((double) x[v] - (double) xr[v]));
        }
    }
    const double pra = b.p1, prb = b.p2;
    const double pta = std::exp((double) x[b.top1] - a.lse);
    const double ptb = std::exp((double) x[b.top2] - a.lse);
    const double shift = (pta - ptb) - (pra - prb);
    const double pairshift = ((double) x[b.top1] - (double) x[b.top2]) - ((double) xr[b.top1] - (double) xr[b.top2]);
    fprintf(st.rows, "\t%.6e\t%d\t%d\t%.6e\t%.6f\t%.6f\t%.6f\t%.6f\t%.6e\t%.6e\n", kld, (int) (a.top1 == b.top1), (int) same,
            maxdl, pra, prb, pta, ptb, shift, pairshift);
    st.n_rows++;
    st.n_same += same;
    st.n_sametop += a.top1 == b.top1;
    st.kld_sum += kld;
}

static bool add_call(llama_batch_ext * batch, const llama_token * tok, uint32_t n, llama_pos pos0, bool all) {
    llama_batch_ext_clear(batch);
    for (uint32_t i = 0; i < n; ++i) {
        const int32_t idx = llama_batch_ext_add_token(batch, 0, tok[i]);
        if (idx < 0) {
            fprintf(stderr, "verify-drift: batch add failed (%d) at %u\n", idx, i);
            return false;
        }
        const llama_pos p = pos0 + (llama_pos) i;
        const llama_pos ps[4] = { p, p, p, 0 };
        llama_batch_ext_set_pos(batch, idx, ps);
        llama_batch_ext_set_output_logits(batch, idx, all || i == n - 1);
    }
    return true;
}

// one call of n tokens with every output (or none but the last); the server's hints, as C981's tool sets them
static bool decode_call(llama_context * ctx, llama_batch_ext * batch, const llama_token * tok, uint32_t n, llama_pos pos0,
                        bool all) {
    if (!add_call(batch, tok, n, pos0, all)) {
        return false;
    }
    llama_set_ubatch_taper_next(ctx, -1, 1);
    const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    if (rc != 0) {
        fprintf(stderr, "verify-drift: decode of %u tokens at %d failed (%d)\n", n, (int) pos0, rc);
        return false;
    }
    return true;
}

// the last call of a prompt as llama-server makes it with --ctx-checkpoints-incall (C981's prompt_call): cut at
// n - 4 - n_ubatch and n - 4, a checkpoint filled at each cut and read after the call, output on the last token
static bool prompt_call(llama_context * ctx, llama_batch_ext * batch, const llama_token * tok, uint32_t n, llama_pos pos0,
                        int & fills) {
    const uint32_t n_ubatch = llama_n_ubatch(ctx);
    std::vector<uint32_t> cuts;
    for (uint32_t off : { 4 + n_ubatch, 4u }) {
        if (n > off) {
            cuts.push_back(n - off);
        }
    }
    if (!add_call(batch, tok, n, pos0, false)) {
        return false;
    }
    for (size_t k = 0; k < cuts.size(); ++k) {
        llama_state_seq_store_at(ctx, 0, (int32_t) k, cuts[k], LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    }
    llama_set_ubatch_taper_next(ctx, -1, 1);
    llama_set_ubatch_cuts(ctx, cuts.data(), cuts.size());
    const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    llama_set_ubatch_cuts(ctx, nullptr, 0);
    if (rc != 0) {
        fprintf(stderr, "verify-drift: decode of a %u-token prompt call failed (%d)\n", n, rc);
        return false;
    }
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
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    const std::string mode  = env_str("KLD_MODE", "");
    const std::string basef = env_str("KLD_BASE", "");
    const std::string out   = env_str("KLD_OUT", "drift");
    const std::string taper = env_str("KLD_TAPER", "short=1400:64");
    const long STEP = env_int("KLD_STEP", 1);
    const long RS   = env_int("KLD_RS", 0);
    bool do_reply = false, do_whole = false;
    for (const auto & p : split(env_str("KLD_PARTS", "reply,whole"), ',')) {
        do_reply |= p == "reply";
        do_whole |= p == "whole";
    }
    if ((mode != "base" && mode != "test") || basef.empty() || STEP < 1 || RS < 0 || (!do_reply && !do_whole)) {
        fprintf(stderr, "verify-drift: KLD_MODE (base|test), KLD_BASE, KLD_STEP >= 1 and a part are needed\n");
        return 1;
    }
    struct spec_t { std::string name, prompt, ids; };
    std::vector<spec_t> specs;
    for (const auto & r : split(env_str("KLD_REQS", ""), ';')) {
        const auto f = split(r, '=');
        if (f.size() != 3) {
            fprintf(stderr, "verify-drift: bad request '%s' (name=prompt_file=ids_file)\n", r.c_str());
            return 1;
        }
        specs.push_back({ f[0], f[1], f[2] });
    }
    if (specs.empty()) {
        fprintf(stderr, "verify-drift: KLD_REQS is empty\n");
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    llama_model_params mp = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "verify-drift: cannot load %s\n", params.model.path.c_str());
        return 1;
    }
    llama_context_params cp = common_context_params_to_llama(params);
    cp.n_rs_seq = (uint32_t) RS;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "verify-drift: cannot create a context\n");
        return 1;
    }
    if (!llama_set_ubatch_taper(ctx, taper.c_str())) {
        fprintf(stderr, "verify-drift: the context refused the taper '%s'\n", taper.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    state_t st;
    st.test    = mode == "test";
    st.n_vocab = llama_vocab_n_tokens(vocab);
    st.ref.resize(st.n_vocab);
    const uint32_t n_batch = llama_n_batch(ctx);

    std::vector<req_t> reqs;
    for (const auto & s : specs) {
        std::string text, ids;
        if (!read_file(s.prompt, text) || !read_file(s.ids, ids)) {
            fprintf(stderr, "verify-drift: cannot read %s or %s\n", s.prompt.c_str(), s.ids.c_str());
            return 1;
        }
        req_t r;
        r.name = s.name;
        r.toks = common_tokenize(ctx, text, true, true);
        r.n_prompt = r.toks.size();
        for (const auto & t : split(ids, ',')) {
            const long v = strtol(t.c_str(), nullptr, 10);
            if (v < 0 || v >= st.n_vocab) {
                fprintf(stderr, "verify-drift: bad token id '%s' in %s\n", t.c_str(), s.ids.c_str());
                return 1;
            }
            r.toks.push_back((llama_token) v);
        }
        if (r.toks.size() + 1 > llama_n_ctx(ctx) || r.n_prompt == 0 || r.toks.size() == r.n_prompt) {
            fprintf(stderr, "verify-drift: request %s does not fit (%zu tokens, n_ctx %u)\n", r.name.c_str(), r.toks.size(), llama_n_ctx(ctx));
            return 1;
        }
        reqs.push_back(r);
    }
    fprintf(stderr, "verify-drift: mode %s, step %ld, n_rs_seq %u, taper %s, n_vocab %d, n_batch %u, n_ubatch %u, n_ctx %u, add_bos %d\n",
            mode.c_str(), STEP, llama_n_rs_seq(ctx), taper.c_str(), st.n_vocab, n_batch, llama_n_ubatch(ctx), llama_n_ctx(ctx),
            (int) llama_vocab_get_add_bos(vocab));

    st.base = fopen(basef.c_str(), st.test ? "rb" : "wb");
    st.rows = fopen((out + ".rows.tsv").c_str(), "w");
    FILE * freqs = fopen((out + ".reqs.tsv").c_str(), "w");
    if (!st.base || !st.rows || !freqs) {
        fprintf(stderr, "verify-drift: cannot open %s or %s.*\n", basef.c_str(), out.c_str());
        return 1;
    }
    const char magic[8] = { 'C', '1', '1', '2', '8', 'R', 'W', '1' };
    if (st.test) {
        char m[8];
        int32_t nv = 0;
        if (fread(m, 1, 8, st.base) != 8 || memcmp(m, magic, 8) != 0 || fread(&nv, sizeof(nv), 1, st.base) != 1 || nv != st.n_vocab) {
            fprintf(stderr, "verify-drift: %s is not a base of this vocabulary\n", basef.c_str());
            return 1;
        }
    } else {
        const int32_t nv = st.n_vocab;
        fwrite(magic, 1, 8, st.base);
        fwrite(&nv, sizeof(nv), 1, st.base);
    }
    fprintf(st.rows, "part\treq\trow\tpos\tnext\tlp\ttop1\tp1\ttop2\tp2%s\n",
            st.test ? "\tkld\tsametop\tsamebits\tmaxdl\tpref_a\tpref_b\tptest_a\tptest_b\tgapshift\tpairshift" : "");
    fprintf(freqs, "req\tname\tprompt_tokens\treply_tokens\tprompt_calls\tfills\tfirst_prompt_ids\tlast_prompt_ids\n");

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    for (size_t ri = 0; ri < reqs.size() && st.ok; ++ri) {
        const req_t & r = reqs[ri];
        const uint32_t P = (uint32_t) r.n_prompt;
        const uint32_t N = (uint32_t) r.toks.size();
        const llama_token * t = r.toks.data();
        auto next_of = [&](uint32_t pos) { return pos + 1 < N ? t[pos + 1] : (llama_token) -1; };
        int fills = 0, prompt_calls = 0;
        if (do_reply) {
            llama_memory_clear(llama_get_memory(ctx), true);
            for (uint32_t off = 0; off < P && st.ok; off += n_batch) {
                const uint32_t n = std::min(n_batch, P - off);
                prompt_calls++;
                if (off + n < P) {
                    st.ok = decode_call(ctx, batch, t + off, n, (llama_pos) off, false);
                } else {
                    st.ok = prompt_call(ctx, batch, t + off, n, (llama_pos) off, fills);
                    if (st.ok) {
                        process_row(st, 0, (int) ri, 0, P - 1, next_of(P - 1), llama_get_logits_ith(ctx, (int32_t) (n - 1)));
                    }
                }
            }
            for (uint32_t c = P; c < N && st.ok; c += (uint32_t) STEP) {
                const uint32_t n = std::min((uint32_t) STEP, N - c);
                st.ok = decode_call(ctx, batch, t + c, n, (llama_pos) c, true);
                for (uint32_t i = 0; i < n && st.ok; ++i) {
                    process_row(st, 0, (int) ri, (int) (c + i - P + 1), c + i, next_of(c + i), llama_get_logits_ith(ctx, (int32_t) i));
                }
            }
            fprintf(stderr, "verify-drift: %s reply: %u prompt tokens in %d calls, %u reply tokens %ld a call\n",
                    r.name.c_str(), P, prompt_calls, N - P, STEP);
        }
        if (do_whole && st.ok) {
            llama_memory_clear(llama_get_memory(ctx), true);
            for (uint32_t c = 0; c < N && st.ok; c += (uint32_t) STEP) {
                const uint32_t n = std::min((uint32_t) STEP, N - c);
                st.ok = decode_call(ctx, batch, t + c, n, (llama_pos) c, true);
                for (uint32_t i = 0; i < n && st.ok; ++i) {
                    process_row(st, 1, (int) ri, (int) (c + i), c + i, next_of(c + i), llama_get_logits_ith(ctx, (int32_t) i));
                }
            }
            fprintf(stderr, "verify-drift: %s whole: %u tokens %ld a call\n", r.name.c_str(), N, STEP);
        }
        std::string first, last;
        for (uint32_t i = 0; i < std::min<uint32_t>(4, P); ++i) {
            first += (i ? "," : "") + std::to_string(t[i]);
        }
        for (uint32_t i = P - std::min<uint32_t>(4, P); i < P; ++i) {
            last += (last.empty() ? "" : ",") + std::to_string(t[i]);
        }
        fprintf(freqs, "%zu\t%s\t%u\t%u\t%d\t%d\t%s\t%s\n", ri, r.name.c_str(), P, N - P, prompt_calls, fills, first.c_str(), last.c_str());
        fflush(st.rows);
        fflush(freqs);
    }
    if (st.test && st.ok) {
        row_hdr extra;
        if (fread(&extra, sizeof(extra), 1, st.base) == 1) {
            fprintf(stderr, "verify-drift: the base holds more rows than this run read\n");
            st.ok = false;
        }
    }
    llama_batch_ext_free(batch);
    fclose(st.base);
    fclose(st.rows);
    fclose(freqs);
    if (st.test) {
        fprintf(stderr, "verify-drift: %ld rows, %ld the same bits, %ld the same top token, mean KLD %.6e\n", st.n_rows,
                st.n_same, st.n_sametop, st.n_rows ? st.kld_sum / st.n_rows : 0.0);
    } else {
        fprintf(stderr, "verify-drift: %ld rows written to the base\n", st.n_rows);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    fprintf(stderr, "verify-drift: %s\n", st.ok ? "done" : "FAILED");
    return st.ok ? 0 : 1;
}
