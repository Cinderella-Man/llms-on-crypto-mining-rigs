// The fork's n-gram drafters (common/ngram-map.cpp, ngram-mod.cpp, ngram-cache.cpp, compiled unchanged), driven the way
// llama-server drives them, one transcript at a time. Prints "<transcript> <position> <drafted> <accepted>" per step,
// the same steps replay.py --dump-steps writes. Built and compared by fork_check.sh.
//
// usage: fork_check SEQS.txt simple N M | map-k N M | map-k4v N M MIN_HITS | mod N_MATCH N_MIN N_MAX | cache
//   SEQS.txt: "<transcript> <n_prompt> <prompt and generated token ids...>" per line

#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

static const int N_CTX     = 16384;  // as in Q51's generation: -c 65536 -np 4
static const int N_PREDICT = 3072;

struct drafter {
    virtual ~drafter() = default;
    virtual void begin(const llama_tokens & prompt) = 0;
    virtual void draft(const llama_tokens & hist, llama_token last, llama_tokens & out) = 0;
    virtual void accept(uint16_t n_accepted) = 0;
};

// common_speculative_impl_ngram_simple
struct drafter_simple : drafter {
    common_ngram_simple_config config;

    drafter_simple(int n, int m) : config{ (uint16_t) n, (uint16_t) m } {}

    void begin(const llama_tokens &) override {}

    void draft(const llama_tokens & hist, llama_token last, llama_tokens & out) override {
        out = common_ngram_simple_draft(config, hist, last);
    }

    void accept(uint16_t) override {}
};

// common_speculative_impl_ngram_map_k (both ngram-map-k and ngram-map-k4v)
struct drafter_map : drafter {
    common_ngram_map map;

    drafter_map(int n, int m, bool key_only, int min_hits) : map(n, m, key_only, min_hits) {}

    void begin(const llama_tokens & prompt) override { common_ngram_map_begin(map, prompt); }

    void draft(const llama_tokens & hist, llama_token last, llama_tokens & out) override {
        common_ngram_map_draft(map, hist, last, out);
    }

    void accept(uint16_t n_accepted) override { common_ngram_map_accept(map, n_accepted); }
};

// common_speculative_impl_ngram_mod, one sequence
struct drafter_mod : drafter {
    int n_min, n_max;
    common_ngram_mod mod;
    size_t i_last = 0;
    size_t n_draft_last = 0;
    int n_low = 0;

    drafter_mod(int n_match, int n_min, int n_max) : n_min(n_min), n_max(n_max), mod(n_match, 4*1024*1024) {}

    void begin(const llama_tokens & prompt) override {
        i_last = 0;
        n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }
        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }
        i_last = prompt.size() - n;

        if ((double) mod.get_used() / (double) mod.size() > 0.25) {
            mod.reset();
        }
    }

    void draft(const llama_tokens & prompt, llama_token last, llama_tokens & result) override {
        n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }
        const size_t n = mod.get_n();

        if (i_last + 32 < cur_len) {
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }
            i_last = cur_len - n;
        }

        result.resize(n + n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = last;

        for (int i = 0; i < n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < n_min) {
                    result.clear();
                    return;
                }
                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        if (n_draft_last > 0) {
            const double f_acc = (double) n_accepted / (double) n_draft_last;
            if (f_acc < 0.25) {
                n_low++;
                if (n_low >= 5) {
                    mod.reset();
                    n_low = 0;
                    i_last = 0;
                }
            } else {
                n_low = 0;
            }
        }
    }
};

// common_speculative_impl_ngram_cache without cache files, one sequence
struct drafter_cache : drafter {
    size_t cache_size = 0;
    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;

    void begin(const llama_tokens &) override { cache_size = 0; }

    void draft(const llama_tokens & prompt, llama_token last, llama_tokens & result) override {
        llama_tokens inp(prompt);
        inp.push_back(last);

        if (cache_size < inp.size()) {
            common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, inp.size() - cache_size,
                                      false);
            cache_size = inp.size();
        }

        result.push_back(last);
        common_ngram_cache_draft(inp, result, 8, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, ngram_cache_context,
                                 ngram_cache_dynamic, ngram_cache_static);
        if (!result.empty()) {
            result.erase(result.begin());
        }
    }

    void accept(uint16_t) override {}
};

static std::unique_ptr<drafter> make_drafter(int argc, char ** argv) {
    const std::string type = argv[2];
    auto arg = [&](int i) { return i < argc ? atoi(argv[i]) : 0; };
    if (type == "simple")  return std::make_unique<drafter_simple>(arg(3), arg(4));
    if (type == "map-k")   return std::make_unique<drafter_map>(arg(3), arg(4), true, 1);
    if (type == "map-k4v") return std::make_unique<drafter_map>(arg(3), arg(4), false, arg(5));
    if (type == "mod")     return std::make_unique<drafter_mod>(arg(3), arg(4), arg(5));
    if (type == "cache")   return std::make_unique<drafter_cache>();
    fprintf(stderr, "unknown drafter type %s\n", type.c_str());
    exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s SEQS.txt TYPE [PARAMS...]\n", argv[0]);
        return 1;
    }
    std::ifstream in(argv[1]);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string id;
        size_t n_prompt;
        ss >> id >> n_prompt;
        llama_tokens all;
        for (llama_token t; ss >> t;) {
            all.push_back(t);
        }
        const llama_tokens gen(all.begin() + n_prompt, all.end());
        llama_tokens hist(all.begin(), all.begin() + n_prompt);

        auto d = make_drafter(argc, argv);
        d->begin(hist);

        const size_t n_gen = gen.size();
        size_t i = 0;
        while (i + 1 < n_gen) {
            const int room = std::min(N_CTX - (int) hist.size() - 2, N_PREDICT - (int) (i + 1) - 1);
            llama_tokens draft;
            if (room > 0) {
                d->draft(hist, gen[i], draft);
                if ((int) draft.size() > room) {
                    draft.resize(room);
                }
            }
            size_t a = 0;
            if (!draft.empty()) {
                const size_t lim = std::min(draft.size(), n_gen - 1 - i);
                while (a < lim && draft[a] == gen[i + 1 + a]) {
                    a++;
                }
                d->accept((uint16_t) a);
            }
            printf("%s %zu %zu %zu\n", id.c_str(), i, draft.size(), a);
            const size_t adv = std::min(a + 1, n_gen - 1 - i);
            hist.insert(hist.end(), gen.begin() + i, gen.begin() + i + adv);
            i += adv;
        }
    }
    return 0;
}
