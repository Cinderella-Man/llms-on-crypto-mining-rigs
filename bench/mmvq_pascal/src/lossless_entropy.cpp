// lossless_entropy: the byte entropy of a model's BF16 matrices and the size of the lossless codes of
// src/lossless_bf16.h (Q621), on the CPU. Reads the tensors that docs/data/2026-11-04_lossless_bf16/run/census.py lists
// (BF16, and F32 when every value is a BF16 value, which is then coded as BF16), encodes each with x12 and x11, decodes
// it back with the kernels' steps and checks every weight, and writes one TSV line per tensor:
//   entropies in bits a weight: the high byte (sign | exponent >> 1), the low byte (exponent bit 0 | mantissa), the
//   exponent byte, the sign + mantissa byte, the whole 16 bits, and the exponent and high byte given the row (the mean
//   of each row's own entropy, a bound for codes with a table per row); the high byte's static Huffman code (per tensor);
//   x12 and x11: escapes (weights outside the row's window) and bits a weight with everything the kernel reads.
// With --sample FILE and --take LIST it also writes the listed tensors (name=shape class:calls per token) for
// src/lossless_main.cu: "LBB1", a count, then per tensor its name[64], class[16], K, N and calls as uint32, then the
// K * N BF16 values.
//
// Usage: lossless_entropy --census census.tsv --models DIR [--out tensors.tsv] [--sample FILE --take name=class:calls,...]

#include "lossless_bf16.h"

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <map>
#include <sstream>
#include <unistd.h>

using namespace lossless;

namespace {

struct entry { std::string shard, name, type; std::vector<uint64_t> dims; uint64_t bytes, offset; };

std::vector<std::string> split(const std::string & s, char c) {
    std::vector<std::string> out;
    std::string cur;
    std::stringstream ss(s);
    while (std::getline(ss, cur, c)) {
        out.push_back(cur);
    }
    return out;
}

bool read_at(const std::string & path, uint64_t off, void * dst, uint64_t n) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }
    uint64_t done = 0;
    while (done < n) {
        const ssize_t r = pread(fd, (char *) dst + done, n - done, (off_t) (off + done));
        if (r <= 0) {
            close(fd);
            return false;
        }
        done += (uint64_t) r;
    }
    close(fd);
    return true;
}

uint64_t fnv1a(const void * p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ ((const uint8_t *) p)[i]) * 0x100000001b3ull;
    }
    return h;
}

} // namespace

int main(int argc, char ** argv) {
    std::string census, models, out_path, sample_path, take;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
        if (a == "--census") census = next();
        else if (a == "--models") models = next();
        else if (a == "--out") out_path = next();
        else if (a == "--sample") sample_path = next();
        else if (a == "--take") take = next();
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (census.empty() || models.empty()) {
        fprintf(stderr, "usage: lossless_entropy --census census.tsv --models DIR [--out tensors.tsv] [--sample FILE --take name=class:calls,...]\n");
        return 2;
    }
    std::map<std::string, std::pair<std::string, int>> take_calls; // tensor -> (shape class, calls per token)
    for (const auto & t : split(take, ',')) {
        const auto kv = split(t, '=');
        const auto cc = kv.size() == 2 ? split(kv[1], ':') : std::vector<std::string>();
        if (cc.size() != 2) { fprintf(stderr, "--take: expected name=class:calls, got %s\n", t.c_str()); return 2; }
        take_calls[kv[0]] = { cc[0], atoi(cc[1].c_str()) };
    }

    std::vector<entry> list;
    {
        FILE * f = fopen(census.c_str(), "r");
        if (!f) { perror(census.c_str()); return 1; }
        char line[4096];
        bool header = true;
        while (fgets(line, sizeof(line), f)) {
            if (header) { header = false; continue; }
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            const auto c = split(s, '\t');
            if (c.size() != 6) continue;
            entry e{ c[0], c[1], c[2], {}, std::stoull(c[4]), std::stoull(c[5]) };
            for (const auto & d : split(c[3], 'x')) e.dims.push_back(std::stoull(d));
            list.push_back(e);
        }
        fclose(f);
    }

    FILE * out = out_path.empty() ? stdout : fopen(out_path.c_str(), "w");
    if (!out) { perror(out_path.c_str()); return 1; }
    fprintf(out, "tensor\ttype\tK\tN\tweights\tzeros\tnot_bf16\tH_hi\tH_lo\tH_exp\tH_signmant\tH_u16\tH_exp_row\tH_hi_row\t"
                 "huff_hi\tx12_esc\tx12_bits\tx11_esc\tx11_bits\tcheck\tfnv1a\n");

    FILE * sample = nullptr;
    if (!sample_path.empty()) {
        sample = fopen(sample_path.c_str(), "wb");
        if (!sample) { perror(sample_path.c_str()); return 1; }
        const uint32_t hdr[2] = { 0x3142424cu /* "LBB1" */, (uint32_t) take_calls.size() };
        fwrite(hdr, 4, 2, sample);
    }
    int sampled = 0, failed = 0;

    for (const auto & e : list) {
        if (e.dims.size() != 2 || e.dims[0] % 32 != 0 || (e.type != "BF16" && e.type != "F32")) continue;
        const int K = (int) e.dims[0], N = (int) e.dims[1];
        const size_t n = (size_t) K * N;
        std::vector<uint16_t> w(n);
        uint64_t not_bf16 = 0;
        const std::string path = models + "/" + e.shard;
        if (e.type == "BF16") {
            if (!read_at(path, e.offset, w.data(), n * 2)) { fprintf(stderr, "read failed: %s %s\n", path.c_str(), e.name.c_str()); return 1; }
        } else {
            std::vector<uint32_t> f(n);
            if (!read_at(path, e.offset, f.data(), n * 4)) { fprintf(stderr, "read failed: %s %s\n", path.c_str(), e.name.c_str()); return 1; }
            for (size_t i = 0; i < n; ++i) {
                not_bf16 += (f[i] & 0xFFFF) != 0;
                w[i] = (uint16_t) (f[i] >> 16);
            }
        }

        static uint64_t hhi[256], hlo[256], hexp[256], hsm[256], hu16[65536];
        memset(hhi, 0, sizeof(hhi)); memset(hlo, 0, sizeof(hlo)); memset(hexp, 0, sizeof(hexp));
        memset(hsm, 0, sizeof(hsm)); memset(hu16, 0, sizeof(hu16));
        uint64_t zeros = 0;
        double hrow_exp = 0.0, hrow_hi = 0.0;
        for (int r = 0; r < N; ++r) {
            uint64_t rexp[256] = { 0 }, rhi[256] = { 0 };
            for (int k = 0; k < K; ++k) {
                const uint16_t u = w[(size_t) r * K + k];
                const int ex = (u >> 7) & 0xFF, sm = ((u >> 8) & 0x80) | (u & 0x7F);
                hhi[u >> 8]++; hlo[u & 0xFF]++; hexp[ex]++; hsm[sm]++; hu16[u]++;
                rexp[ex]++; rhi[u >> 8]++;
                zeros += (u & 0x7FFF) == 0;
            }
            hrow_exp += entropy(rexp, 256, K);
            hrow_hi += entropy(rhi, 256, K);
        }
        hrow_exp /= N;
        hrow_hi /= N;

        const encoded e12 = encode(w.data(), K, N, 12), e11 = encode(w.data(), K, N, 11);
        const bool exact = not_bf16 == 0;
        bool ok = true;
        std::vector<uint16_t> back(K);
        for (const encoded * c : { &e12, &e11 }) {
            for (int r = 0; r < N && ok; ++r) {
                decode_row(*c, r, back.data());
                ok = memcmp(back.data(), w.data() + (size_t) r * K, (size_t) K * 2) == 0;
            }
        }
        failed += exact && !ok;

        fprintf(out, "%s\t%s\t%d\t%d\t%zu\t%llu\t%llu\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%zu\t%.4f\t%zu\t%.4f\t%s\t%016llx\n",
                e.name.c_str(), e.type.c_str(), K, N, n, (unsigned long long) zeros, (unsigned long long) not_bf16,
                entropy(hhi, 256, n), entropy(hlo, 256, n), entropy(hexp, 256, n), entropy(hsm, 256, n), entropy(hu16, 65536, n),
                hrow_exp, hrow_hi, huffman_bits(hhi, 256) / n,
                e12.esc.size(), e12.bits_per_weight(), e11.esc.size(), e11.bits_per_weight(),
                !exact ? "not-bf16" : ok ? "PASS" : "FAIL", (unsigned long long) fnv1a(w.data(), n * 2));
        fflush(out);

        const auto it = take_calls.find(e.name);
        if (sample && it != take_calls.end()) {
            char name[64] = { 0 }, cls[16] = { 0 };
            strncpy(name, e.name.c_str(), 63);
            strncpy(cls, it->second.first.c_str(), 15);
            const uint32_t meta[3] = { (uint32_t) K, (uint32_t) N, (uint32_t) it->second.second };
            fwrite(name, 1, 64, sample);
            fwrite(cls, 1, 16, sample);
            fwrite(meta, 4, 3, sample);
            fwrite(w.data(), 2, n, sample);
            ++sampled;
        }
    }
    if (sample) {
        fclose(sample);
        if (sampled != (int) take_calls.size()) {
            fprintf(stderr, "sample: %d of %zu tensors found\n", sampled, take_calls.size());
            return 1;
        }
    }
    if (out != stdout) fclose(out);
    fprintf(stderr, "round trip: %s (%d tensors failed)\n", failed ? "FAIL" : "PASS", failed);
    return failed ? 1 : 0;
}
