// llama-staircase-dump: run token sequences through a model and save what the staircase guesser (Q20/D1) learns from:
// for every position the inputs of a few layers (the card boundaries of a layer split) and the final hidden state,
// and for every position whose next token is a generated one the K most likely next tokens with their
// log-probabilities over the whole vocabulary (before any sampler).
//
//   DUMP_IN=seqs.txt DUMP_OUT=feat.bin [DUMP_LAYERS=9,17,25,33,41] [DUMP_ALLPOS=9] [DUMP_K=20] [DUMP_MARGIN=8] \
//       [DUMP_THREADS=8] [DUMP_OUT2=small.bin DUMP_LAYERS2=9,17,25,33,41] \
//       llama-staircase-dump -m model.gguf -ngl 99 -c 8192 -b 2048 -ub 512 -fa on [other common flags]
//
// seqs.txt: one sequence per line, "<seq_id> <n_prompt> <tok0> <tok1> ...", prompt tokens first, then generated ones
// (generate.py's rows, written by feats.py's `seqs` command). Each sequence starts from an empty KV cache.
//
// feat.bin, per sequence:
//   int32 header[8]  = {seq_id, n_prompt, n_tok, a0, n_layers, k, n_embd, 0x51320003}
//   int32 layers[n_layers]                              the dumped layer inputs, e.g. 9 17 25 33 41
//   int32 starts[n_layers]                              first stored position of each layer (0 or a0)
//   int32 tokens[n_tok]
//   fp16  scale_full[n_tok], uint8 full[n_tok][n_embd]  final hidden state (after the output norm), FP8 E4M3
//   per dumped layer, for positions b .. n_tok-1, where b = 0 for the layers in DUMP_ALLPOS (card 1's output, which a
//   card-1 head reads for every position; default the first dumped layer) and b = a0 = max(0, n_prompt - margin) for
//   the others:
//         fp16 scale[n_tok - b], uint8 h[n_tok - b][n_embd]
//   int32 top_ids[n_gen][k], float32 top_lp[n_gen][k]   n_gen = n_tok - n_prompt; row i predicts generated token i
// DUMP_OUT2 gets a second copy of each record with only the DUMP_LAYERS2 layers (a subset of DUMP_LAYERS), e.g. a
// training subset on a fast disk next to a superset on a large one.
// FP8 is E4M3 (torch.float8_e4m3fn) with one scale per row: x = fp8 * scale, scale = max|x| / 448. Earlier prompt
// prompt positions keep only the final hidden state and the DUMP_ALLPOS layers, which is all a committed position
// contributes to the guessers.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-ext.h"   // llama_set_embeddings_layer_inp, llama_get_embeddings_layer_inp (src/)

#include <algorithm>
#include <atomic>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (e <= 0) {
        return (uint16_t) sign;   // scales are never this small
    }
    if (e >= 31) {
        return (uint16_t) (sign | 0x7bff);
    }
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rest = m & 0x1fff;
    if (rest > 0x1000 || (rest == 0x1000 && (h & 1))) {
        h++;
    }
    return (uint16_t) h;
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t x;
    if (e == 0) {
        float f = std::ldexp((float) m, -24);
        return sign ? -f : f;
    }
    x = sign | ((e - 15 + 127) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// float -> FP8 E4M3 (no infinities, max 448), round to nearest even, saturating
static uint8_t f32_to_e4m3(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint8_t sign = (uint8_t) ((u >> 24) & 0x80);
    u &= 0x7fffffff;
    if (u > 0x7f800000) {
        return sign | 0x7f;   // NaN
    }
    if (u >= 0x43e00000) {    // >= 448
        return sign | 0x7e;
    }
    if (u < 0x3c800000) {     // < 2^-6: subnormal, steps of 2^-9
        float a;
        std::memcpy(&a, &u, 4);
        const int m = (int) std::nearbyint(a * 512.0f);   // 0..8; 8 is the smallest normal
        return sign | (uint8_t) m;
    }
    u += 0x7ffff + ((u >> 20) & 1);
    const int e = (int) (u >> 23) - 127 + 7;
    const int m = (int) ((u >> 20) & 7);
    if (e > 15 || (e == 15 && m == 7)) {
        return sign | 0x7e;
    }
    return sign | (uint8_t) ((e << 3) | m);
}

static void quant_rows(const float * src, int n_rows, int n_embd, uint16_t * scales, uint8_t * dst) {
    for (int r = 0; r < n_rows; ++r) {
        const float * x = src + (size_t) r * n_embd;
        float amax = 0.0f;
        for (int i = 0; i < n_embd; ++i) {
            amax = std::max(amax, std::fabs(x[i]));
        }
        uint16_t sh = f32_to_f16(amax > 0.0f ? amax / 448.0f : 1.0f);
        float s = f16_to_f32(sh);
        if (amax / s > 448.0f) {   // the fp16 rounding of the scale must not push values past 448
            sh++;
            s = f16_to_f32(sh);
        }
        scales[r] = sh;
        const float inv = 1.0f / s;
        uint8_t * q = dst + (size_t) r * n_embd;
        for (int i = 0; i < n_embd; ++i) {
            q[i] = f32_to_e4m3(x[i] * inv);
        }
    }
}

// ids of the complete records in a dump file; a cut-off last record is removed
static std::vector<int32_t> scan_and_trim(const char * path, bool verbose) {
    std::vector<int32_t> ids;
    std::error_code ec;
    const int64_t size = (int64_t) std::filesystem::file_size(path, ec);
    if (ec) {
        return ids;
    }
    FILE * f = fopen(path, "rb");
    int32_t hdr[8];
    int64_t pos = 0;
    while (f && fread(hdr, sizeof(hdr), 1, f) == 1) {
        const int64_t n_tok = hdr[2], nl = hdr[4], kk = hdr[5], ne = hdr[6];
        const int64_t n_gen = n_tok - hdr[1];
        if (pos + 32 + nl * 8 > size) {
            break;
        }
        std::vector<int32_t> ls(nl * 2);   // layers, then their start positions
        if (fread(ls.data(), 4, nl * 2, f) != (size_t) (nl * 2)) {
            break;
        }
        int64_t body = nl * 8 + n_tok * 4 + n_tok * (2 + ne) + n_gen * kk * 8;
        for (int64_t li = 0; li < nl; ++li) {
            body += (n_tok - ls[nl + li]) * (2 + ne);
        }
        if (pos + 32 + body > size) {
            break;
        }
        fseek(f, body - nl * 8, SEEK_CUR);
        pos += 32 + body;
        ids.push_back(hdr[0]);
    }
    if (f) {
        fclose(f);
    }
    if (pos < size) {
        std::filesystem::resize_file(path, pos);
        if (verbose) {
            fprintf(stderr, "%s: dropped a cut-off record\n", path);
        }
    }
    return ids;
}

template <typename F> static void parallel_for(int n, int n_threads, F f) {
    std::atomic<int> next{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < n_threads; ++t) {
        ts.emplace_back([&]() {
            for (int i; (i = next.fetch_add(1)) < n;) {
                f(i);
            }
        });
    }
    for (auto & t : ts) {
        t.join();
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    const char * in_path  = getenv("DUMP_IN");
    const char * out_path = getenv("DUMP_OUT");
    const char * out2_path = getenv("DUMP_OUT2");
    const int    k        = getenv("DUMP_K") ? atoi(getenv("DUMP_K")) : 20;
    const int    margin   = getenv("DUMP_MARGIN") ? atoi(getenv("DUMP_MARGIN")) : 8;
    const int    n_thr    = getenv("DUMP_THREADS") ? atoi(getenv("DUMP_THREADS")) : 8;
    std::vector<int32_t> layers = {9, 17, 25, 33, 41};
    if (getenv("DUMP_LAYERS")) {
        layers.clear();
        std::stringstream ls(getenv("DUMP_LAYERS"));
        for (std::string tok; std::getline(ls, tok, ',');) {
            layers.push_back(atoi(tok.c_str()));
        }
    }
    std::vector<int32_t> layers2;
    if (getenv("DUMP_LAYERS2")) {
        std::stringstream ls(getenv("DUMP_LAYERS2"));
        for (std::string tok; std::getline(ls, tok, ',');) {
            layers2.push_back(atoi(tok.c_str()));
        }
    }
    std::vector<int32_t> allpos = {layers[0]};
    if (getenv("DUMP_ALLPOS")) {
        allpos.clear();
        std::stringstream ls(getenv("DUMP_ALLPOS"));
        for (std::string tok; std::getline(ls, tok, ',');) {
            allpos.push_back(atoi(tok.c_str()));
        }
    }
    if (in_path == nullptr || out_path == nullptr || k <= 0) {
        fprintf(stderr, "set DUMP_IN and DUMP_OUT\n");
        return 1;
    }
    FILE * out = fopen(out_path, "ab");
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
    const int32_t n_embd  = llama_model_n_embd(model);
    const int32_t n_batch = (int32_t) llama_n_batch(ctx);
    const int32_t n_ctx   = (int32_t) llama_n_ctx(ctx);
    for (int32_t l : layers) {
        llama_set_embeddings_layer_inp(ctx, (uint32_t) l, true);
    }
    llama_set_embeddings(ctx, true);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // sequences already in the output (resume after an interruption): skip their ids, drop a cut-off last record
    std::vector<int32_t> done_ids = scan_and_trim(out_path, true);
    fclose(out);
    out = fopen(out_path, "ab");
    FILE * out2 = nullptr;
    if (out2_path != nullptr) {
        scan_and_trim(out2_path, false);
        out2 = fopen(out2_path, "ab");
    }
    fprintf(stderr, "resuming after %zu sequences\n", done_ids.size());

    std::ifstream in(in_path);
    std::string line;
    int n_seq = 0;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        int32_t seq_id = 0, n_prompt = 0;
        if (!(ss >> seq_id >> n_prompt)) {
            continue;
        }
        if (std::find(done_ids.begin(), done_ids.end(), seq_id) != done_ids.end()) {
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
        const int32_t a0    = std::max(0, n_prompt - margin);
        const size_t  nl    = layers.size();

        std::vector<uint16_t> sc_full(n_tok);
        std::vector<uint8_t>  q_full((size_t) n_tok * n_embd);
        std::vector<int32_t> b_l(nl, a0);
        for (size_t li = 0; li < nl; ++li) {
            if (std::find(allpos.begin(), allpos.end(), layers[li]) != allpos.end()) {
                b_l[li] = 0;
            }
        }
        std::vector<std::vector<uint16_t>> sc_l(nl);
        std::vector<std::vector<uint8_t>>  q_l(nl);
        for (size_t li = 0; li < nl; ++li) {
            sc_l[li].resize(n_tok - b_l[li]);
            q_l[li].resize((size_t) (n_tok - b_l[li]) * n_embd);
        }
        std::vector<int32_t> top_ids((size_t) std::max(0, n_gen) * k);
        std::vector<float>   top_lp((size_t) std::max(0, n_gen) * k);

        llama_memory_clear(llama_get_memory(ctx), true);
        bool ok = true;
        for (int32_t i0 = 0; i0 < n_tok; i0 += n_batch) {
            const int32_t n = std::min<int32_t>(n_batch, n_tok - i0);
            batch.n_tokens = n;
            for (int32_t j = 0; j < n; ++j) {
                batch.token[j]     = toks[i0 + j];
                batch.pos[j]       = i0 + j;
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = true;
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "sequence %d: decode failed at %d\n", seq_id, i0);
                ok = false;
                break;
            }
            // final hidden state of every position
            const float * emb = llama_get_embeddings(ctx);
            parallel_for((n + 63) / 64, n_thr, [&](int c) {
                const int r0 = c * 64, nr = std::min(64, n - r0);
                quant_rows(emb + (size_t) r0 * n_embd, nr, n_embd, &sc_full[i0 + r0],
                           &q_full[(size_t) (i0 + r0) * n_embd]);
            });
            // layer inputs from b_l on
            for (size_t li = 0; li < nl; ++li) {
                const int32_t b  = b_l[li];
                const int32_t j0 = std::max(0, b - i0);
                if (j0 >= n) {
                    continue;
                }
                const float * h = llama_get_embeddings_layer_inp(ctx, (uint32_t) layers[li]);
                parallel_for((n - j0 + 63) / 64, n_thr, [&](int c) {
                    const int r0 = j0 + c * 64, nr = std::min(64, n - r0);
                    quant_rows(h + (size_t) r0 * n_embd, nr, n_embd, &sc_l[li][i0 + r0 - b],
                               &q_l[li][(size_t) (i0 + r0 - b) * n_embd]);
                });
            }
            // position p predicts token p + 1: rows n_prompt - 1 .. n_tok - 2 predict the generated tokens
            const int32_t g0 = std::max(0, n_prompt - 1 - i0), g1 = std::min(n, n_tok - 1 - i0);
            if (g0 < g1) {
                parallel_for(g1 - g0, n_thr, [&](int c) {
                    const int32_t j = g0 + c;
                    const float * logits = llama_get_logits_ith(ctx, j);
                    std::vector<std::pair<float, int32_t>> best;
                    best.reserve(k + 1);
                    float mx = -INFINITY;
                    for (int32_t t = 0; t < n_vocab; ++t) {
                        mx = std::max(mx, logits[t]);
                    }
                    double sum = 0.0;
                    float thr = -INFINITY;
                    for (int32_t t = 0; t < n_vocab; ++t) {
                        const float l = logits[t];
                        sum += std::exp((double) (l - mx));
                        if ((int) best.size() < k || l > thr) {
                            best.emplace_back(l, t);
                            if ((int) best.size() > k) {
                                std::sort(best.begin(), best.end(), [](auto & a, auto & b) { return a.first > b.first; });
                                best.pop_back();
                            }
                            if ((int) best.size() == k) {
                                thr = std::min_element(best.begin(), best.end())->first;
                            }
                        }
                    }
                    std::sort(best.begin(), best.end(), [](auto & a, auto & b) { return a.first > b.first; });
                    const float lse = mx + (float) std::log(sum);
                    const size_t g = (size_t) (i0 + j - (n_prompt - 1));
                    for (int i = 0; i < k; ++i) {
                        top_ids[g * k + i] = best[i].second;
                        top_lp[g * k + i]  = best[i].first - lse;
                    }
                });
            }
        }
        if (!ok) {
            continue;
        }
        auto write_record = [&](FILE * f, const std::vector<size_t> & sel) {
            const int32_t hdr[8] = {seq_id, n_prompt, n_tok, a0, (int32_t) sel.size(), k, n_embd, 0x51320003};
            fwrite(hdr, sizeof(hdr), 1, f);
            for (size_t li : sel) {
                fwrite(&layers[li], sizeof(int32_t), 1, f);
            }
            for (size_t li : sel) {
                fwrite(&b_l[li], sizeof(int32_t), 1, f);
            }
            fwrite(toks.data(), sizeof(int32_t), n_tok, f);
            fwrite(sc_full.data(), 2, n_tok, f);
            fwrite(q_full.data(), 1, q_full.size(), f);
            for (size_t li : sel) {
                fwrite(sc_l[li].data(), 2, sc_l[li].size(), f);
                fwrite(q_l[li].data(), 1, q_l[li].size(), f);
            }
            fwrite(top_ids.data(), sizeof(int32_t), top_ids.size(), f);
            fwrite(top_lp.data(), sizeof(float), top_lp.size(), f);
            fflush(f);
        };
        std::vector<size_t> sel_all(nl), sel2;
        for (size_t li = 0; li < nl; ++li) {
            sel_all[li] = li;
            if (std::find(layers2.begin(), layers2.end(), layers[li]) != layers2.end()) {
                sel2.push_back(li);
            }
        }
        write_record(out, sel_all);
        if (out2) {
            write_record(out2, sel2);
        }
        ++n_seq;
        fprintf(stderr, "sequence %d: %d tokens (%d prompt)\n", seq_id, n_tok, n_prompt);
    }

    llama_batch_free(batch);
    fclose(out);
    if (out2) {
        fclose(out2);
    }
    fprintf(stderr, "done: %d sequences\n", n_seq);
    llama_backend_free();
    return 0;
}
