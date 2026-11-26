// grammar_steps: the tool-call grammar's state at every token of a reply, as llama-server's grammar sampler sees it.
//
//   grammar_steps MODEL.gguf GRAMMAR.txt GEN_PROMPT.txt TOKENS.txt [lazy TRIGGER_TOKEN_ID...]
//
// MODEL      a GGUF whose vocabulary the server used (only the vocabulary is loaded)
// GRAMMAR    the grammar text the server built for the request (the "grammar" field of its slot parameters)
// GEN_PROMPT the request's generation prompt (the "generation_prompt" field); for a grammar that is not lazy the
//            server feeds its tokens to the grammar before the first sampled token (common_sampler_init's prefill),
//            and so does this tool
// TOKENS     the reply's token ids, one per line, in the order they were sampled
// lazy ...   build the grammar lazy, with these trigger tokens (the server's own lazy mode); without it, not lazy
//
// For every token of the reply, before accepting it, the grammar is applied to the whole vocabulary (every token with
// the same logit), as the sampler's grammar step does to a row, and the tool prints one line:
//   step  token  allowed  n_allowed  n_vocab  awaiting  left_out  piece
// allowed: whether the grammar lets this token through; n_allowed: how many of the vocabulary's tokens it lets
// through at this step (n_vocab of them: the grammar constrains nothing here); awaiting: a lazy grammar still waiting
// for its trigger; left_out: the ids of the tokens it does not let through, when there are at most 32 of them (else
// "n=<count>"). The first time each list of at most 1024 tokens appears, a "# left out" line names its tokens
// (id=bytes in hex). In the piece column bytes below 0x20 and 0x7f are written \xNN. Then it accepts the token, as the sampler does after sampling it. A token the grammar rejects is
// reported and not accepted (the server would have resampled), and the walk goes on.
// The last line sums the steps: how many let the whole vocabulary through, how many did not, how many rejected
// the reply's own token.
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static std::string slurp(const char * path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "grammar_steps: cannot read %s\n", path);
        exit(2);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string piece_of(const llama_vocab * vocab, llama_token t) {
    char buf[512];
    const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
    std::string s = n > 0 ? std::string(buf, n) : std::string();
    std::string out;
    for (const unsigned char c : s) {
        if (c == '\n') {
            out += "\\n";
        } else if (c == '\t') {
            out += "\\t";
        } else if (c == '\\') {
            out += "\\\\";
        } else if (c < 0x20 || c == 0x7f) {
            char h[8];
            snprintf(h, sizeof(h), "\\x%02x", c);
            out += h;
        } else {
            out += (char) c;
        }
    }
    return out;
}

// the token's bytes in hex (a piece can hold any byte, NUL too)
static std::string hex_of(const llama_vocab * vocab, llama_token t) {
    char buf[512];
    const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
    std::string out;
    char h[4];
    for (int i = 0; i < n; ++i) {
        snprintf(h, sizeof(h), "%02x", (unsigned char) buf[i]);
        out += h;
    }
    return out.empty() ? "-" : out;
}

static void quiet_log(enum ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s MODEL.gguf GRAMMAR.txt GEN_PROMPT.txt TOKENS.txt [lazy TRIGGER_TOKEN_ID...]\n", argv[0]);
        return 2;
    }
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "grammar_steps: cannot load the vocabulary of %s\n", argv[1]);
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const std::string grammar    = slurp(argv[2]);
    const std::string gen_prompt = slurp(argv[3]);

    std::vector<llama_token> reply;
    {
        std::ifstream f(argv[4]);
        long v;
        while (f >> v) {
            reply.push_back((llama_token) v);
        }
    }

    const bool lazy = argc > 5 && strcmp(argv[5], "lazy") == 0;
    std::vector<llama_token> triggers;
    for (int i = 6; lazy && i < argc; ++i) {
        triggers.push_back((llama_token) atol(argv[i]));
    }

    llama_sampler * g = lazy
        ? llama_sampler_init_grammar_lazy_patterns(vocab, grammar.c_str(), "root", nullptr, 0, triggers.data(), triggers.size())
        : llama_sampler_init_grammar(vocab, grammar.c_str(), "root");
    if (!g) {
        fprintf(stderr, "grammar_steps: the grammar does not parse\n");
        return 2;
    }

    // the server's prefill (common_sampler_init): the generation prompt's tokens, a leading space-only first piece
    // left out, fed to a grammar that is not lazy
    if (!lazy && !gen_prompt.empty()) {
        std::vector<llama_token> pre(gen_prompt.size() + 8);
        int n = llama_tokenize(vocab, gen_prompt.c_str(), (int) gen_prompt.size(), pre.data(), (int) pre.size(), false, true);
        if (n < 0) {
            fprintf(stderr, "grammar_steps: cannot tokenize the generation prompt\n");
            return 2;
        }
        pre.resize(n);
        for (int i = 0; i < n; ++i) {
            const std::string p = piece_of(vocab, pre[i]);
            if (i == 0 && !p.empty() && isspace((unsigned char) p[0]) && !isspace((unsigned char) gen_prompt[0])) {
                continue;
            }
            llama_sampler_accept(g, pre[i]);
            printf("# prefill %d '%s'\n", pre[i], p.c_str());
        }
    }

    std::vector<llama_token_data> row(n_vocab);
    int n_whole = 0, n_narrowed = 0, n_rejected = 0;
    std::set<std::string> named;
    printf("step\ttoken\tallowed\tn_allowed\tn_vocab\tawaiting\tleft_out\tpiece\n");
    for (size_t s = 0; s < reply.size(); ++s) {
        for (int t = 0; t < n_vocab; ++t) {
            row[t] = { t, 0.0f, 0.0f };
        }
        llama_token_data_array arr = { row.data(), (size_t) n_vocab, -1, false };
        const bool awaiting = llama_sampler_grammar_awaiting_trigger(g);
        llama_sampler_apply(g, &arr);
        int n_allowed = 0;
        bool allowed = false;
        std::vector<llama_token> left;
        for (size_t i = 0; i < arr.size; ++i) {
            if (arr.data[i].logit != -INFINITY) {
                n_allowed++;
                if (arr.data[i].id == reply[s]) {
                    allowed = true;
                }
            } else if (left.size() <= 1024) {
                left.push_back(arr.data[i].id);
            }
        }
        std::string ids;
        for (const auto t : left) {
            ids += (ids.empty() ? "" : ",") + std::to_string(t);
        }
        const int n_left = n_vocab - n_allowed;
        const std::string left_s = n_left == 0 ? "none" : n_left <= 32 ? ids : "n=" + std::to_string(n_left);
        if (n_left > 0 && n_left <= 1024 && named.insert(ids).second) {
            std::string names;
            for (const auto t : left) {
                names += " " + std::to_string(t) + "=" + hex_of(vocab, t);
            }
            printf("# left out at step %zu (%d):%s\n", s, n_left, names.c_str());
        }
        n_whole    += n_allowed == n_vocab;
        n_narrowed += n_allowed != n_vocab;
        printf("%zu\t%d\t%d\t%d\t%d\t%d\t%s\t%s\n", s, reply[s], allowed ? 1 : 0, n_allowed, n_vocab, awaiting ? 1 : 0,
               left_s.c_str(), piece_of(vocab, reply[s]).c_str());
        if (!allowed) {
            n_rejected++;
            continue;
        }
        llama_sampler_accept(g, reply[s]);
    }
    printf("# steps %zu: the whole vocabulary allowed at %d, fewer tokens at %d, the reply's token rejected at %d\n",
           reply.size(), n_whole, n_narrowed, n_rejected);

    llama_sampler_free(g);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
