// A continuation datastore for pipelined speculation (registry row Q161): a suffix array over the reversed token stream
// of many generated answers. A query reads the context backwards (the last token first), narrows the suffix array one
// token at a time, and returns, for every context length that occurs in the store, how often it occurs and its most
// frequent next token. A guesser takes the most frequent continuation of the longest (or longest frequent enough)
// context, the way SuffixDecoding (arXiv 2411.04975) scores continuations by frequency in a suffix tree.
//
// Memory: 8 bytes per stored token (the reversed tokens and the suffix array), plus a cache of the most frequent next
// token of every context that occurs more than DS_SCAN_MAX times (counted once, then reused).
//
// Build:  g++ -O2 -std=c++17 -shared -fPIC -o libdatastore.so datastore.cpp      (the library, for ctypes)
//         g++ -O2 -std=c++17 -o datastore datastore.cpp -DDS_MAIN                 (the command-line tool)
// Tool:   datastore build TOKENS.i32 MAX_TOKENS OUT.ds    (TOKENS.i32: int32 tokens, answers separated by -1; the first
//                                                          MAX_TOKENS tokens, cut at an answer boundary)
// File:   the 8 bytes "PIPESTO1", the token count n (int64), n tokens of the stream reversed, n suffix-array entries
//         (int32 each, little-endian): the format the fork's llama-server and llama-pipespec read with --pipe-store
//         (common/pipe-store.cpp), memory-mapped.
// The library counts its work (ds_stats): suffix-array probes of the binary searches and entries scanned to count next
// tokens. Each probe reads one suffix-array entry and one token at a random place in the store, so the counts give the
// lookup cost on another machine from its memory latency, without timing anything here.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

static const char    DS_MAGIC[8] = { 'P', 'I', 'P', 'E', 'S', 'T', 'O', '1' };
static const int32_t DS_SEP      = -1;    // between answers: a context never matches across it
static const int32_t DS_END      = -2;    // past the end of the stream
static const int64_t DS_SCAN_MAX = 2048;  // ranges larger than this are counted once and cached

struct datastore {
    std::vector<int32_t> rev;   // the token stream, reversed
    std::vector<int32_t> sa;    // suffix array of rev
    struct top { int32_t tok, cnt, total; };
    std::unordered_map<uint64_t, top> big;   // (depth, lo) -> top next token, its count, all next tokens

    // work counters: queries, binary-search probes, entries scanned for next-token counts, cache hits
    int64_t n_query = 0, n_probe = 0, n_scan = 0, n_cache_hit = 0;

    int32_t at(int64_t i) const { return i < (int64_t) rev.size() ? rev[i] : DS_END; }

    // the token after the occurrence whose reversed suffix starts at r (the one before it in rev), or DS_SEP
    int32_t next_of(int64_t r) const { return r > 0 ? rev[r - 1] : DS_SEP; }
};

// prefix doubling: sort by (rank[i], rank[i + k]) until every rank is unique
static void build_sa(datastore & ds) {
    const int64_t n = ds.rev.size();
    std::vector<int32_t> & sa = ds.sa;
    sa.resize(n);
    std::vector<int32_t> rank(n), tmp(n);
    std::vector<uint64_t> key(n);
    for (int64_t i = 0; i < n; i++) {
        sa[i]   = (int32_t) i;
        rank[i] = ds.rev[i] + 2;   // DS_SEP -> 1, tokens >= 2; 0 is "past the end"
    }
    for (int64_t k = 1;; k <<= 1) {
        for (int64_t i = 0; i < n; i++) {
            key[i] = ((uint64_t) (uint32_t) rank[i] << 32) | (uint32_t) (i + k < n ? rank[i + k] : 0);
        }
        std::sort(sa.begin(), sa.end(), [&](int32_t a, int32_t b) { return key[a] < key[b]; });
        tmp[sa[0]] = 1;
        for (int64_t i = 1; i < n; i++) {
            tmp[sa[i]] = tmp[sa[i - 1]] + (key[sa[i]] != key[sa[i - 1]]);
        }
        rank.swap(tmp);
        if (rank[sa[n - 1]] == n) {
            break;
        }
    }
}

// the most frequent next token in sa[lo, hi), its count, and the count of all next tokens (separators and the stream's
// end are not tokens)
static datastore::top top_next(datastore & ds, int depth, int64_t lo, int64_t hi) {
    const bool cacheable = hi - lo > DS_SCAN_MAX;
    const uint64_t ck = ((uint64_t) depth << 40) | (uint64_t) lo;
    if (cacheable) {
        auto it = ds.big.find(ck);
        if (it != ds.big.end()) {
            ds.n_cache_hit++;
            return it->second;
        }
    }
    ds.n_scan += hi - lo;
    std::unordered_map<int32_t, int32_t> cnt;
    cnt.reserve((size_t) std::min<int64_t>(hi - lo, 4096));
    int32_t best = -1, bc = 0, total = 0;
    for (int64_t i = lo; i < hi; i++) {
        const int32_t t = ds.next_of(ds.sa[i]);
        if (t < 0) {
            continue;
        }
        total++;
        const int32_t c = ++cnt[t];
        if (c > bc || (c == bc && t < best)) {
            best = t;
            bc   = c;
        }
    }
    const datastore::top r = { best, bc, total };
    if (cacheable) {
        ds.big[ck] = r;
    }
    return r;
}

extern "C" {

// tokens: answers separated by DS_SEP
void * ds_build(const int32_t * toks, int64_t n) {
    datastore * ds = new datastore();
    ds->rev.assign(toks, toks + n);
    std::reverse(ds->rev.begin(), ds->rev.end());
    build_sa(*ds);
    return ds;
}

int ds_save(void * h, const char * path) {
    const datastore * ds = (const datastore *) h;
    FILE * f = fopen(path, "wb");
    if (!f) {
        return -1;
    }
    const int64_t n = ds->rev.size();
    const bool ok = fwrite(DS_MAGIC, 1, sizeof(DS_MAGIC), f) == sizeof(DS_MAGIC) &&
                    fwrite(&n, sizeof(n), 1, f) == 1 &&
                    fwrite(ds->rev.data(), sizeof(int32_t), n, f) == (size_t) n &&
                    fwrite(ds->sa.data(), sizeof(int32_t), n, f) == (size_t) n;
    return fclose(f) == 0 && ok ? 0 : -1;
}

void * ds_load(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return nullptr;
    }
    char    magic[sizeof(DS_MAGIC)];
    int64_t n = 0;
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) || memcmp(magic, DS_MAGIC, sizeof(magic)) != 0 ||
            fread(&n, sizeof(n), 1, f) != 1 || n <= 0) {
        fclose(f);
        return nullptr;
    }
    datastore * ds = new datastore();
    ds->rev.resize(n);
    ds->sa.resize(n);
    const bool ok = fread(ds->rev.data(), sizeof(int32_t), n, f) == (size_t) n &&
                    fread(ds->sa.data(), sizeof(int32_t), n, f) == (size_t) n;
    fclose(f);
    if (!ok) {
        delete ds;
        return nullptr;
    }
    return ds;
}

void ds_free(void * h) { delete (datastore *) h; }

int64_t ds_size(void * h) { return (int64_t) ((datastore *) h)->rev.size(); }

int64_t ds_cache_entries(void * h) { return (int64_t) ((datastore *) h)->big.size(); }

// out[0..5): queries, binary-search probes, entries scanned, cache hits, cache entries
void ds_stats(void * h, int64_t * out) {
    const datastore & ds = *(const datastore *) h;
    out[0] = ds.n_query;
    out[1] = ds.n_probe;
    out[2] = ds.n_scan;
    out[3] = ds.n_cache_hit;
    out[4] = (int64_t) ds.big.size();
}

// ctx[0..n_ctx): the context, oldest token first. For L = 1 .. the longest context length that occurs (at most max_len):
// occ[L-1] = its occurrences followed by a token, tok[L-1] = their most frequent next token, cnt[L-1] = how often.
// Returns that longest length (0: not even the last token occurs).
int ds_query(void * h, const int32_t * ctx, int n_ctx, int max_len, int32_t * occ, int32_t * tok, int32_t * cnt) {
    datastore & ds = *(datastore *) h;
    int64_t lo = 0, hi = (int64_t) ds.sa.size();
    const int lmax = std::min(n_ctx, max_len);
    int longest = 0;
    ds.n_query++;
    for (int k = 0; k < lmax; k++) {
        const int32_t sym = ctx[n_ctx - 1 - k];
        // suffixes in [lo, hi) share k symbols; their k-th symbol is sorted
        const int64_t a = std::partition_point(ds.sa.begin() + lo, ds.sa.begin() + hi,
                                               [&](int32_t s) { ds.n_probe++; return ds.at((int64_t) s + k) < sym; }) - ds.sa.begin();
        const int64_t b = std::partition_point(ds.sa.begin() + a, ds.sa.begin() + hi,
                                               [&](int32_t s) { ds.n_probe++; return ds.at((int64_t) s + k) <= sym; }) - ds.sa.begin();
        if (a >= b) {
            break;
        }
        lo = a;
        hi = b;
        const auto tc = top_next(ds, k + 1, lo, hi);
        if (tc.tok < 0) {
            break;   // occurs only at the end of answers: nothing follows it
        }
        occ[k] = tc.total;
        tok[k] = tc.tok;
        cnt[k] = tc.cnt;
        longest = k + 1;
    }
    return longest;
}

// The deployable query: narrows the ranges for every length (binary searches only) and counts next tokens in the
// longest one alone (backing off while a range has no token after it, as ds_query stops). Returns that length, or 0;
// *occ, *tok, *cnt as ds_query's entries for it.
int ds_longest(void * h, const int32_t * ctx, int n_ctx, int max_len, int32_t * occ, int32_t * tok, int32_t * cnt) {
    datastore & ds = *(datastore *) h;
    const int lmax = std::min(std::min(n_ctx, max_len), 64);
    int64_t los[64], his[64];
    int64_t lo = 0, hi = (int64_t) ds.sa.size();
    int depth = 0;
    ds.n_query++;
    for (int k = 0; k < lmax; k++) {
        const int32_t sym = ctx[n_ctx - 1 - k];
        const int64_t a = std::partition_point(ds.sa.begin() + lo, ds.sa.begin() + hi,
                                               [&](int32_t s) { ds.n_probe++; return ds.at((int64_t) s + k) < sym; }) - ds.sa.begin();
        const int64_t b = std::partition_point(ds.sa.begin() + a, ds.sa.begin() + hi,
                                               [&](int32_t s) { ds.n_probe++; return ds.at((int64_t) s + k) <= sym; }) - ds.sa.begin();
        if (a >= b) {
            break;
        }
        lo = a;
        hi = b;
        los[k] = lo;
        his[k] = hi;
        depth = k + 1;
    }
    // ds_query stops at the first length whose occurrences are all at answer ends; a longer length then cannot occur
    // with a token after it either, so the longest length with a token after it is found by backing off
    while (depth > 0) {
        const auto tc = top_next(ds, depth, los[depth - 1], his[depth - 1]);
        if (tc.tok >= 0) {
            *occ = tc.total;
            *tok = tc.tok;
            *cnt = tc.cnt;
            return depth;
        }
        depth--;
    }
    return 0;
}

}  // extern "C"

#ifdef DS_MAIN
static std::vector<int32_t> read_i32(const char * path) {
    std::vector<int32_t> v;
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize(sz / sizeof(int32_t));
    if (fread(v.data(), sizeof(int32_t), v.size(), f) != v.size()) {
        fprintf(stderr, "short read %s\n", path);
        exit(1);
    }
    fclose(f);
    return v;
}

int main(int argc, char ** argv) {
    if (argc >= 5 && strcmp(argv[1], "build") == 0) {
        std::vector<int32_t> toks = read_i32(argv[2]);
        int64_t n = std::min<int64_t>(atoll(argv[3]), (int64_t) toks.size());
        while (n < (int64_t) toks.size() && n > 0 && toks[n - 1] != DS_SEP) {
            n--;   // cut at an answer boundary
        }
        void * h = ds_build(toks.data(), n);
        if (ds_save(h, argv[4]) != 0) {
            fprintf(stderr, "cannot write %s\n", argv[4]);
            return 1;
        }
        printf("stored %lld tokens in %s\n", (long long) n, argv[4]);
        ds_free(h);
        return 0;
    }
    fprintf(stderr, "usage: datastore build TOKENS.i32 MAX_TOKENS OUT.ds\n");
    return 1;
}
#endif
