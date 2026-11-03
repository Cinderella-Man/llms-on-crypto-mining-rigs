// ple_tok: tokenizes the texts replay.py wrote with the model's own tokenizer (libllama, vocabulary only), as
// llama-server tokenizes a prompt (add_special and parse_special on).
//   ple_tok MODEL.gguf TEXTS.bin TOKENS.bin [THREADS] [ID...]
// TEXTS.bin and TOKENS.bin are sequences of [u32 length][payload]: UTF-8 bytes in, int32 token ids out. Each ID given
// after THREADS is printed with its text, to check the stop tokens.
#include "llama.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

static void quiet(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL.gguf TEXTS.bin TOKENS.bin [THREADS] [ID...]\n", argv[0]);
        return 1;
    }
    const int n_threads = argc > 4 ? atoi(argv[4]) : 4;
    llama_log_set(quiet, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    for (int i = 5; i < argc; ++i) {
        char buf[256];
        const int n = llama_token_to_piece(vocab, atoi(argv[i]), buf, sizeof(buf), 0, true);
        printf("token %s = '%.*s'\n", argv[i], n > 0 ? n : 0, buf);
    }

    std::vector<std::string> texts;
    {
        FILE * f = fopen(argv[2], "rb");
        if (!f) {
            fprintf(stderr, "cannot open %s\n", argv[2]);
            return 1;
        }
        uint32_t n;
        while (fread(&n, 4, 1, f) == 1) {
            std::string s(n, '\0');
            if (n && fread(&s[0], 1, n, f) != n) {
                fprintf(stderr, "short read in %s\n", argv[2]);
                return 1;
            }
            texts.push_back(std::move(s));
        }
        fclose(f);
    }

    std::vector<std::vector<llama_token>> out(texts.size());
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    auto work = [&]() {
        for (size_t i; (i = next++) < texts.size();) {
            const std::string & s = texts[i];
            std::vector<llama_token> & t = out[i];
            t.resize(s.size() + 16);
            int n = llama_tokenize(vocab, s.data(), (int32_t) s.size(), t.data(), (int32_t) t.size(), true, true);
            if (n < 0) {
                t.resize(-n);
                n = llama_tokenize(vocab, s.data(), (int32_t) s.size(), t.data(), (int32_t) t.size(), true, true);
            }
            if (n < 0) {
                failed = true;
                n = 0;
            }
            t.resize(n);
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < n_threads; ++i) {
        pool.emplace_back(work);
    }
    for (auto & th : pool) {
        th.join();
    }
    if (failed) {
        fprintf(stderr, "tokenization failed\n");
        return 1;
    }

    FILE * f = fopen(argv[3], "wb");
    size_t total = 0;
    for (const auto & t : out) {
        const uint32_t n = (uint32_t) t.size();
        fwrite(&n, 4, 1, f);
        fwrite(t.data(), sizeof(llama_token), n, f);
        total += n;
    }
    fclose(f);
    printf("%zu texts, %zu tokens\n", texts.size(), total);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
