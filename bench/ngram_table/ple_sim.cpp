// ple_sim: Flash-Next's n-gram table reads for a replayed token stream (replay.py stream), through caches.
//   ple_sim DIR OUTDIR [TRAIN_DIR]
// Reads DIR/params.txt, DIR/stream.bin and DIR/traces.tsv (and DIR/check_rows.txt, if replay.py check wrote it);
// TRAIN_DIR/stream.bin, if given, is a second set of sessions that the hot-row sidecar is also learned on.
// Each processed token reads 16 table rows, hashed from it and its two predecessors as llama.cpp's qwen4exp does
// (llm_graph_input_qwen4exp_ple::set_input), and one token-embedding row. Three access streams:
//   pages  the 4 KiB pages of the table rows (a row that crosses a page boundary reads both): the page cache
//   rows   the table rows themselves: a row cache that reads rows directly
//   pages+embd  the table pages and the token-embedding pages in one page cache
// LRU is computed exactly for every cache size at once from stack distances (the number of distinct items used since
// an item's previous use; a cache of C items hits when it is below C). Three scopes: the cache starts cold at every
// session, at every run (a run's sessions in the order they ran), or once for all runs. Belady's optimal policy and
// CLOCK give bounds at a few sizes, and a static hot-row sidecar is learned on half of the tasks and tested on the
// other half (and learned on TRAIN_DIR and tested on all). Writes TSV tables to OUTDIR.
#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;
static const u32 INF = UINT32_MAX;

struct Rec { int32_t trace, req, kind, t0, t1, t2; };

struct Params {
    int ngram = 0, per = 0;
    int64_t eos = 0;
    std::vector<u64> mult, offs, voc;
    u64 table_rows = 0, table_rb = 0, table_off = 0, embd_rows = 0, embd_rb = 0, embd_off = 0, page = 4096;
};

static Params read_params(const std::string & path) {
    Params p;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::string k;
        is >> k;
        std::vector<u64> v;
        for (u64 x; is >> x;) v.push_back(x);
        if (v.empty()) continue;
        if (k == "ngram_size") p.ngram = (int) v[0];
        else if (k == "heads_per_ngram") p.per = (int) v[0];
        else if (k == "eos") p.eos = (int64_t) v[0];
        else if (k == "multipliers") p.mult = v;
        else if (k == "offsets") p.offs = v;
        else if (k == "vocab") p.voc = v;
        else if (k == "table_rows") p.table_rows = v[0];
        else if (k == "table_row_bytes") p.table_rb = v[0];
        else if (k == "table_offset") p.table_off = v[0];
        else if (k == "embd_rows") p.embd_rows = v[0];
        else if (k == "embd_row_bytes") p.embd_rb = v[0];
        else if (k == "embd_offset") p.embd_off = v[0];
        else if (k == "page") p.page = v[0];
    }
    if (p.ngram < 2 || p.per < 1 || p.offs.size() != (size_t) (p.ngram - 1) * p.per || p.voc.size() != p.offs.size()) {
        fprintf(stderr, "bad params in %s\n", path.c_str());
        exit(1);
    }
    return p;
}

// the 16 rows of one token, exactly as set_input() computes them
static void hash_rows(const Params & p, const Rec & r, u32 * out) {
    int64_t ctx[8];
    ctx[0] = r.t0;
    const int32_t prev[2] = { r.t1, r.t2 };
    bool cut = false;
    for (int s = 1; s < p.ngram; ++s) {
        const int64_t t = cut ? -1 : prev[s - 1];
        cut = cut || t < 0 || t == p.eos;
        ctx[s] = cut ? p.eos : t;
    }
    for (int n = 2; n <= p.ngram; ++n) {
        u64 mixed = (u64) ctx[0] * p.mult[0];
        for (int j = 1; j < n; ++j) mixed ^= (u64) ctx[j] * p.mult[j];
        const int base = (n - 2) * p.per;
        for (int g = 0; g < p.per; ++g) {
            const int h = base + g;
            out[h] = (u32) (mixed % p.voc[h] + p.offs[h]);
        }
    }
}

struct Fenwick {
    std::vector<int32_t> t;
    void reset(size_t n) { t.assign(n + 1, 0); }
    void add(size_t i, int32_t v) { for (; i < t.size(); i += i & (0 - i)) t[i] += v; }
    int64_t sum(size_t i) const { int64_t s = 0; for (; i > 0; i -= i & (0 - i)) s += t[i]; return s; }
};

// LRU stack distance of every access; segments start at the given access indices (the cache is cold there)
static std::vector<u32> lru_distances(const std::vector<u32> & ids, const std::vector<size_t> & seg, u64 space) {
    std::vector<u32> dist(ids.size());
    std::vector<int32_t> last_t(space, 0), last_s(space, -1);
    Fenwick fw;
    for (size_t s = 0; s < seg.size(); ++s) {
        const size_t a = seg[s], b = s + 1 < seg.size() ? seg[s + 1] : ids.size();
        fw.reset(b - a);
        for (size_t i = a; i < b; ++i) {
            const u32 id = ids[i];
            const size_t t = i - a + 1;
            if (last_s[id] == (int32_t) s) {
                const size_t tp = (size_t) last_t[id];
                dist[i] = (u32) (fw.sum(t - 1) - fw.sum(tp));
                fw.add(tp, -1);
            } else {
                dist[i] = INF;
            }
            fw.add(t, 1);
            last_s[id] = (int32_t) s;
            last_t[id] = (int32_t) t;
        }
    }
    return dist;
}

// misses of Belady's optimal policy (evict the item used furthest in the future) at a capacity of C items
static std::vector<uint8_t> opt_misses(const std::vector<u32> & ids, const std::vector<size_t> & seg, u64 space, u64 C) {
    std::vector<uint8_t> miss(ids.size(), 0);
    std::vector<u32> next(ids.size(), INF);
    std::vector<u32> cur(space, INF);
    std::vector<uint8_t> in(space, 0);
    for (size_t s = 0; s < seg.size(); ++s) {
        const size_t a = seg[s], b = s + 1 < seg.size() ? seg[s + 1] : ids.size();
        {
            std::unordered_map<u32, u32> seen;
            for (size_t i = b; i-- > a;) {
                auto it = seen.find(ids[i]);
                next[i] = it == seen.end() ? INF : it->second;
                seen[ids[i]] = (u32) i;
            }
        }
        std::priority_queue<std::pair<u32, u32>> heap;  // (next use, id); stale entries skipped
        u64 size = 0;
        for (size_t i = a; i < b; ++i) {
            const u32 id = ids[i];
            if (in[id]) {
                cur[id] = next[i];
                heap.push({ next[i], id });
                continue;
            }
            miss[i] = 1;
            if (size == C) {
                while (true) {
                    auto [n, x] = heap.top();
                    heap.pop();
                    if (in[x] && cur[x] == n) { in[x] = 0; --size; break; }
                }
            }
            in[id] = 1;
            cur[id] = next[i];
            heap.push({ next[i], id });
            ++size;
        }
        for (size_t i = a; i < b; ++i) { in[ids[i]] = 0; cur[ids[i]] = INF; }
    }
    return miss;
}

// misses of a CLOCK cache (second chance, the reference bit set on insert) of C items
static std::vector<uint8_t> clock_misses(const std::vector<u32> & ids, const std::vector<size_t> & seg, u64 space, u64 C) {
    std::vector<uint8_t> miss(ids.size(), 0);
    std::vector<u32> where(space, INF);
    std::vector<u32> slot(C, INF);
    std::vector<uint8_t> ref(C, 0);
    for (size_t s = 0; s < seg.size(); ++s) {
        const size_t a = seg[s], b = s + 1 < seg.size() ? seg[s + 1] : ids.size();
        size_t hand = 0, used = 0;
        for (size_t i = a; i < b; ++i) {
            const u32 id = ids[i];
            if (where[id] != INF) { ref[where[id]] = 1; continue; }
            miss[i] = 1;
            size_t k;
            if (used < C) {
                k = used++;
            } else {
                while (ref[hand]) { ref[hand] = 0; hand = (hand + 1) % C; }
                k = hand;
                where[slot[k]] = INF;
                hand = (hand + 1) % C;
            }
            slot[k] = id;
            ref[k] = 1;
            where[id] = (u32) k;
        }
        for (size_t k = 0; k < std::min<size_t>(used, C); ++k) { where[slot[k]] = INF; slot[k] = INF; ref[k] = 0; }
    }
    return miss;
}

struct Stream {
    std::string name;
    std::vector<u32> ids;   // item per access
    std::vector<u32> tok;   // token (record) index per access
    u64 space = 0;          // ids are below this
    u64 item_bytes = 0;     // bytes one cached item takes
};

static std::vector<size_t> segments(const Stream & st, const std::vector<int> & key_of_tok) {
    std::vector<size_t> seg;
    int last = -2;
    for (size_t i = 0; i < st.tok.size(); ++i) {
        const int k = key_of_tok[st.tok[i]];
        if (k != last) { seg.push_back(i); last = k; }
    }
    return seg;
}

static std::string fmt_bytes(u64 b) {
    char buf[64];
    if (b == 95000000) return "95 MB";
    if (b >= (1ull << 30)) snprintf(buf, sizeof buf, "%g GiB", (double) b / (1ull << 30));
    else snprintf(buf, sizeof buf, "%g MiB", (double) b / (1ull << 20));
    return buf;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s DIR OUTDIR [TRAIN_DIR]\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[1], out = argv[2];
    const Params P = read_params(dir + "/params.txt");

    auto read_stream = [](const std::string & d) {
        std::vector<Rec> v;
        FILE * f = fopen((d + "/stream.bin").c_str(), "rb");
        if (!f) { fprintf(stderr, "no %s/stream.bin\n", d.c_str()); exit(1); }
        Rec r;
        while (fread(&r, sizeof r, 1, f) == 1) v.push_back(r);
        fclose(f);
        return v;
    };
    std::vector<Rec> recs = read_stream(dir);
    std::vector<u32> train_rows;  // the rows of another set of sessions, for the sidecar
    if (argc > 3) {
        for (const Rec & r : read_stream(argv[3])) {
            u32 rr[64];
            hash_rows(P, r, rr);
            train_rows.insert(train_rows.end(), rr, rr + P.offs.size());
        }
    }
    std::map<int, std::pair<int, int>> trace_run_task;  // trace -> (run, task)
    {
        std::ifstream f(dir + "/traces.tsv");
        std::string line;
        std::getline(f, line);
        while (std::getline(f, line)) {
            std::istringstream is(line);
            int tr, run, task;
            is >> tr >> run >> task;
            trace_run_task[tr] = { run, task };
        }
    }
    const size_t T = recs.size();
    const int H = (int) P.offs.size();
    std::vector<int> run_of(T), trace_of(T), task_of(T), kind_of(T);
    for (size_t i = 0; i < T; ++i) {
        trace_of[i] = recs[i].trace;
        run_of[i] = trace_run_task.at(recs[i].trace).first;
        task_of[i] = trace_run_task.at(recs[i].trace).second;
        kind_of[i] = recs[i].kind;
    }

    std::vector<u32> rows(T * H);
    for (size_t i = 0; i < T; ++i) hash_rows(P, recs[i], &rows[i * H]);
    for (u32 r : rows) if (r >= P.table_rows) { fprintf(stderr, "row %u out of range\n", r); return 1; }

    // the Python implementation's rows of the first records, if present
    {
        std::ifstream f(dir + "/check_rows.txt");
        if (f) {
            std::string line;
            size_t n = 0, bad = 0;
            while (std::getline(f, line) && n < T) {
                std::istringstream is(line);
                for (int h = 0; h < H; ++h) { u64 x; is >> x; bad += x != rows[n * H + h]; }
                ++n;
            }
            printf("check_rows: %zu tokens compared, %zu rows differ\n", n, bad);
            if (bad) return 1;
        }
    }

    const u64 PG = P.page;
    const u64 table_pages = (P.table_off + P.table_rows * P.table_rb + PG - 1) / PG;
    const u64 embd_first = P.embd_off / PG, embd_pages = (P.embd_off + P.embd_rows * P.embd_rb + PG - 1) / PG - embd_first;

    Stream pages{ "pages" }, rowst{ "rows" }, both{ "pages+embd" };
    pages.space = table_pages; pages.item_bytes = PG;
    rowst.space = P.table_rows; rowst.item_bytes = P.table_rb;
    both.space = table_pages + embd_pages; both.item_bytes = PG;
    std::vector<uint8_t> acc_group;  // per rows-stream access: 0 bigram head, 1 trigram head
    u64 straddle = 0;
    for (size_t i = 0; i < T; ++i) {
        for (int h = 0; h < H; ++h) {
            const u32 r = rows[i * H + h];
            rowst.ids.push_back(r);
            rowst.tok.push_back((u32) i);
            acc_group.push_back(h < P.per ? 0 : 1);
            const u64 o = P.table_off + (u64) r * P.table_rb;
            const u64 p1 = o / PG, p2 = (o + P.table_rb - 1) / PG;
            for (u64 p = p1; p <= p2; ++p) {
                pages.ids.push_back((u32) p); pages.tok.push_back((u32) i);
                both.ids.push_back((u32) p); both.tok.push_back((u32) i);
            }
            straddle += p2 != p1;
        }
        const u64 e = P.embd_off + (u64) recs[i].t0 * P.embd_rb;
        for (u64 p = e / PG; p <= (e + P.embd_rb - 1) / PG; ++p) {
            both.ids.push_back((u32) (table_pages + p - embd_first)); both.tok.push_back((u32) i);
        }
    }

    u64 n_kind[2] = { 0, 0 };
    for (size_t i = 0; i < T; ++i) n_kind[kind_of[i]]++;
    std::set<int> runs_set, traces_set;
    for (size_t i = 0; i < T; ++i) { runs_set.insert(run_of[i]); traces_set.insert(trace_of[i]); }

    FILE * sm = fopen((out + "/summary.tsv").c_str(), "w");
    fprintf(sm, "key\tvalue\n");
    fprintf(sm, "runs\t%zu\nsessions\t%zu\ntokens\t%zu\nprefill_tokens\t%" PRIu64 "\ndecode_tokens\t%" PRIu64 "\n",
            runs_set.size(), traces_set.size(), T, n_kind[0], n_kind[1]);
    fprintf(sm, "rows_per_token\t%d\ntable_row_bytes\t%" PRIu64 "\ntable_bytes\t%" PRIu64 "\ntable_pages\t%" PRIu64 "\n",
            H, P.table_rb, P.table_rows * P.table_rb, table_pages);
    fprintf(sm, "page_reads_per_token_uncached\t%.4f\nrows_crossing_a_page_boundary\t%.4f\n",
            (double) pages.ids.size() / T, (double) straddle / rowst.ids.size());
    fprintf(sm, "embd_row_bytes\t%" PRIu64 "\nembd_bytes\t%" PRIu64 "\n", P.embd_rb, P.embd_rows * P.embd_rb);

    // distinct table pages per 512-token prefill micro-batch (the rows of a micro-batch are prefetched together)
    {
        std::vector<u32> seen(table_pages, INF);
        std::vector<u32> ub_pages;
        size_t i = 0;
        u32 ub_id = 0;
        while (i < T) {
            if (kind_of[i] != 0) { ++i; continue; }
            size_t j = i;
            while (j < T && kind_of[j] == 0 && recs[j].req == recs[i].req && j - i < 512) ++j;
            if (j - i == 512) {
                u32 d = 0;
                for (size_t k = i; k < j; ++k)
                    for (int h = 0; h < H; ++h) {
                        const u64 o = P.table_off + (u64) rows[k * H + h] * P.table_rb;
                        for (u64 p = o / PG; p <= (o + P.table_rb - 1) / PG; ++p)
                            if (seen[p] != ub_id) { seen[p] = ub_id; ++d; }
                    }
                ub_pages.push_back(d);
                ++ub_id;
            }
            i = j;
        }
        if (!ub_pages.empty()) {
            std::sort(ub_pages.begin(), ub_pages.end());
            const double mean = std::accumulate(ub_pages.begin(), ub_pages.end(), 0.0) / ub_pages.size();
            fprintf(sm, "full_prefill_microbatches\t%zu\ndistinct_pages_per_512_microbatch_mean\t%.1f\n"
                        "distinct_pages_per_512_microbatch_median\t%u\ndistinct_pages_per_512_microbatch_min\t%u\n",
                    ub_pages.size(), mean, ub_pages[ub_pages.size() / 2], ub_pages.front());
        }
    }

    // working sets: distinct rows and pages per session, per run and in all
    {
        FILE * f = fopen((out + "/working_set.tsv").c_str(), "w");
        fprintf(f, "scope\tunits\trows_median\trows_max\tpages_median\tpages_max\tGiB_of_pages_median\tGiB_of_pages_max\ttable_share_max\n");
        for (int sc = 0; sc < 3; ++sc) {
            const std::vector<int> & key = sc == 0 ? trace_of : run_of;
            std::map<int, std::pair<u64, u64>> per;  // key -> (distinct rows, distinct pages)
            std::vector<int32_t> rseen(P.table_rows, -1), pseen(table_pages, -1);
            for (size_t a = 0; a < rowst.ids.size(); ++a) {
                const int k = sc == 2 ? 0 : key[rowst.tok[a]];
                if (rseen[rowst.ids[a]] != k) { rseen[rowst.ids[a]] = k; per[k].first++; }
            }
            for (size_t a = 0; a < pages.ids.size(); ++a) {
                const int k = sc == 2 ? 0 : key[pages.tok[a]];
                if (pseen[pages.ids[a]] != k) { pseen[pages.ids[a]] = k; per[k].second++; }
            }
            std::vector<u64> r, p;
            for (auto & kv : per) { r.push_back(kv.second.first); p.push_back(kv.second.second); }
            std::sort(r.begin(), r.end());
            std::sort(p.begin(), p.end());
            const char * name = sc == 0 ? "session" : sc == 1 ? "run" : "all";
            fprintf(f, "%s\t%zu\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%.3f\t%.3f\t%.4f\n", name, r.size(),
                    r[r.size() / 2], r.back(), p[p.size() / 2], p.back(), p[p.size() / 2] * (double) PG / (1ull << 30),
                    p.back() * (double) PG / (1ull << 30), (double) p.back() / table_pages);
        }
        fclose(f);
    }

    const std::vector<u64> caps = { 16ull << 20, 32ull << 20, 64ull << 20, 95000000ull, 128ull << 20, 256ull << 20,
                                    384ull << 20, 512ull << 20, 768ull << 20, 1ull << 30, 3ull << 29, 2ull << 30,
                                    5ull << 29, 3ull << 30, 4ull << 30, 6ull << 30, 8ull << 30, 12ull << 30,
                                    16ull << 30, 24ull << 30, 32ull << 30 };
    const char * scope_name[3] = { "session", "run", "all" };
    std::vector<int> zero(T, 0);

    FILE * cv = fopen((out + "/lru_curve.tsv").c_str(), "w");
    fprintf(cv, "stream\tscope\tkind\tpart\tcache\tcache_bytes\tcache_items\taccesses\ttokens\thit_rate\tmisses_per_token\n");
    std::vector<u32> pages_run_dist, rows_run_dist, both_run_dist;
    for (Stream * st : { &pages, &rowst, &both }) {
        for (int sc = 0; sc < 3; ++sc) {
            if (st != &pages && sc == 0) continue;
            const std::vector<size_t> seg = segments(*st, sc == 0 ? trace_of : sc == 1 ? run_of : zero);
            std::vector<u32> d = lru_distances(st->ids, seg, st->space);
            const int nparts = st == &rowst ? 3 : 1;
            for (int kind = 0; kind < 2; ++kind)
                for (int part = 0; part < nparts; ++part) {
                    std::vector<u32> v;
                    for (size_t a = 0; a < d.size(); ++a) {
                        if (kind_of[st->tok[a]] != kind) continue;
                        if (part > 0 && acc_group[a] != part - 1) continue;
                        v.push_back(d[a]);
                    }
                    std::sort(v.begin(), v.end());
                    const char * pname = part == 0 ? "all" : part == 1 ? "bigram" : "trigram";
                    for (u64 c : caps) {
                        const u64 items = c / st->item_bytes;
                        const u64 hits = std::lower_bound(v.begin(), v.end(), (u32) std::min<u64>(items, INF - 1)) - v.begin();
                        fprintf(cv, "%s\t%s\t%s\t%s\t%s\t%" PRIu64 "\t%" PRIu64 "\t%zu\t%" PRIu64 "\t%.4f\t%.3f\n",
                                st->name.c_str(), scope_name[sc], kind ? "decode" : "prefill", pname, fmt_bytes(c).c_str(),
                                c, items, v.size(), n_kind[kind], v.empty() ? 0.0 : (double) hits / v.size(),
                                (double) (v.size() - hits) / n_kind[kind]);
                    }
                }
            if (sc == 1) (st == &pages ? pages_run_dist : st == &rowst ? rows_run_dist : both_run_dist) = std::move(d);
            fflush(cv);
        }
    }
    fclose(cv);

    // per decoded token: the distribution of page reads at a few page-cache sizes (scope run)
    const std::vector<u64> tcaps = { 256ull << 20, 512ull << 20, 1ull << 30, 3ull << 29, 2ull << 30, 4ull << 30 };
    {
        FILE * f = fopen((out + "/token_reads.tsv").c_str(), "w");
        fprintf(f, "stream\tcache\tkind\tmean\tp50\tp90\tp99\tmax");
        for (int m = 0; m <= 20; ++m) fprintf(f, "\tshare_%d", m);
        fprintf(f, "\n");
        for (Stream * st : { &pages, &both }) {
            const std::vector<u32> & d = st == &pages ? pages_run_dist : both_run_dist;
            for (u64 c : tcaps) {
                const u64 items = c / PG;
                std::vector<u32> m(T, 0);
                for (size_t a = 0; a < d.size(); ++a) m[st->tok[a]] += d[a] >= items;
                for (int kind = 0; kind < 2; ++kind) {
                    std::vector<u32> v;
                    for (size_t i = 0; i < T; ++i) if (kind_of[i] == kind) v.push_back(m[i]);
                    std::sort(v.begin(), v.end());
                    const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
                    fprintf(f, "%s\t%s\t%s\t%.3f\t%u\t%u\t%u\t%u", st->name.c_str(), fmt_bytes(c).c_str(),
                            kind ? "decode" : "prefill", mean, v[v.size() / 2], v[v.size() * 9 / 10], v[v.size() * 99 / 100], v.back());
                    for (u32 k = 0; k <= 20; ++k) {
                        const size_t n = std::upper_bound(v.begin(), v.end(), k) - std::lower_bound(v.begin(), v.end(), k);
                        fprintf(f, "\t%.4f", (double) n / v.size());
                    }
                    fprintf(f, "\n");
                }
            }
        }
        fclose(f);
    }

    // per request: prompt tokens processed, decoded tokens, and table page reads at a few sizes (scope run)
    {
        FILE * f = fopen((out + "/request_reads.tsv").c_str(), "w");
        fprintf(f, "trace\treq\tprefill_tokens\tdecode_tokens");
        for (u64 c : tcaps) fprintf(f, "\tprefill_reads_%s\tdecode_reads_%s", fmt_bytes(c).c_str(), fmt_bytes(c).c_str());
        fprintf(f, "\n");
        std::map<int, std::vector<u64>> acc;  // req -> counters
        std::map<int, int> req_trace;
        const size_t K = 2 + 2 * tcaps.size();
        for (size_t i = 0; i < T; ++i) {
            auto & v = acc[recs[i].req];
            if (v.empty()) v.assign(K, 0);
            v[kind_of[i]]++;
            req_trace[recs[i].req] = recs[i].trace;
        }
        for (size_t a = 0; a < pages_run_dist.size(); ++a) {
            const size_t i = pages.tok[a];
            auto & v = acc[recs[i].req];
            for (size_t c = 0; c < tcaps.size(); ++c)
                v[2 + 2 * c + kind_of[i]] += pages_run_dist[a] >= tcaps[c] / PG;
        }
        for (auto & kv : acc) {
            fprintf(f, "%d\t%d", req_trace[kv.first], kv.first);
            for (u64 x : kv.second) fprintf(f, "\t%" PRIu64, x);
            fprintf(f, "\n");
        }
        fclose(f);
    }

    // Belady's optimal policy and CLOCK at a few sizes (scope run), beside LRU
    {
        FILE * f = fopen((out + "/policies.tsv").c_str(), "w");
        fprintf(f, "stream\tpolicy\tcache\tcache_bytes\tprefill_hit_rate\tdecode_hit_rate\tprefill_misses_per_token\tdecode_misses_per_token\n");
        const std::vector<u64> pcaps = { 95000000ull, 256ull << 20, 512ull << 20, 1ull << 30, 2ull << 30, 4ull << 30 };
        for (Stream * st : { &pages, &rowst }) {
            const std::vector<size_t> seg = segments(*st, run_of);
            const std::vector<u32> & d = st == &pages ? pages_run_dist : rows_run_dist;
            for (u64 c : pcaps) {
                const u64 items = c / st->item_bytes;
                for (int pol = 0; pol < 3; ++pol) {
                    if (pol == 2 && st != &rowst) continue;
                    std::vector<uint8_t> miss;
                    if (pol == 1) miss = opt_misses(st->ids, seg, st->space, items);
                    if (pol == 2) miss = clock_misses(st->ids, seg, st->space, items);
                    u64 acc[2] = { 0, 0 }, mis[2] = { 0, 0 };
                    for (size_t a = 0; a < st->ids.size(); ++a) {
                        const int k = kind_of[st->tok[a]];
                        acc[k]++;
                        mis[k] += pol == 0 ? d[a] >= items : miss[a];
                    }
                    fprintf(f, "%s\t%s\t%s\t%" PRIu64 "\t%.4f\t%.4f\t%.3f\t%.3f\n", st->name.c_str(),
                            pol == 0 ? "LRU" : pol == 1 ? "OPT" : "CLOCK", fmt_bytes(c).c_str(), c,
                            1.0 - (double) mis[0] / acc[0], 1.0 - (double) mis[1] / acc[1],
                            (double) mis[0] / n_kind[0], (double) mis[1] / n_kind[1]);
                    fflush(f);
                }
            }
        }
        fclose(f);
    }

    // a static hot-row sidecar: the K rows used most in one half of the tasks, packed densely and kept resident;
    // tested on the sessions of the other half, whose other rows go through an LRU page cache of the rest of the budget
    {
        FILE * f = fopen((out + "/sidecar.tsv").c_str(), "w");
        fprintf(f, "split\tsidecar_rows\tsidecar_bytes\tbudget\tbudget_bytes\tkind\ttest_tokens\tsidecar_hit_share\tpage_hits_share\treads_per_token\n");
        const std::vector<u64> ks = { 0, 1ull << 20, 2ull << 20, 4ull << 20, 8ull << 20, 16ull << 20 };
        const std::vector<u64> budgets = { 512ull << 20, 1ull << 30, 2ull << 30, 4ull << 30 };
        for (int split = 0; split < 4; ++split) {
            // split 0: train on even tasks, test on odd; 1: the reverse; 2: train and test on all (an upper bound);
            // 3: train on another set of sessions (TRAIN_DIR), test on all
            if (split == 3 && train_rows.empty()) break;
            auto is_train = [&](size_t i) { return split == 2 || (task_of[i] % 2) == split; };
            auto is_test = [&](size_t i) { return split >= 2 || (task_of[i] % 2) != split; };
            std::vector<u32> freq(P.table_rows, 0);
            if (split == 3) {
                for (u32 r : train_rows) freq[r]++;
            } else {
                for (size_t a = 0; a < rowst.ids.size(); ++a) if (is_train(rowst.tok[a])) freq[rowst.ids[a]]++;
            }
            std::vector<u32> order;
            for (u32 r = 0; r < P.table_rows; ++r) if (freq[r]) order.push_back(r);
            std::sort(order.begin(), order.end(), [&](u32 x, u32 y) { return freq[x] > freq[y]; });
            std::vector<uint8_t> hot(P.table_rows, 0);
            u64 n_test[2] = { 0, 0 };
            for (size_t i = 0; i < T; ++i) if (is_test(i)) n_test[kind_of[i]]++;
            size_t marked = 0;
            for (u64 K : ks) {
                for (; marked < std::min<size_t>(K, order.size()); ++marked) hot[order[marked]] = 1;
                Stream rest{ "rest" };
                rest.space = table_pages;
                u64 side_hits[2] = { 0, 0 }, row_acc[2] = { 0, 0 };
                for (size_t a = 0; a < rowst.ids.size(); ++a) {
                    const size_t i = rowst.tok[a];
                    if (!is_test(i)) continue;
                    row_acc[kind_of[i]]++;
                    if (hot[rowst.ids[a]]) { side_hits[kind_of[i]]++; continue; }
                    const u64 o = P.table_off + (u64) rowst.ids[a] * P.table_rb;
                    for (u64 p = o / PG; p <= (o + P.table_rb - 1) / PG; ++p) { rest.ids.push_back((u32) p); rest.tok.push_back((u32) i); }
                }
                const std::vector<u32> d = lru_distances(rest.ids, segments(rest, run_of), rest.space);
                const u64 side_bytes = (K * P.table_rb + PG - 1) / PG * PG;
                for (u64 B : budgets) {
                    if (side_bytes >= B) continue;
                    const u64 items = (B - side_bytes) / PG;
                    u64 reads[2] = { 0, 0 }, pacc[2] = { 0, 0 };
                    for (size_t a = 0; a < d.size(); ++a) {
                        const int k = kind_of[rest.tok[a]];
                        pacc[k]++;
                        reads[k] += d[a] >= items;
                    }
                    for (int k = 0; k < 2; ++k)
                        fprintf(f, "%s\t%" PRIu64 "\t%" PRIu64 "\t%s\t%" PRIu64 "\t%s\t%" PRIu64 "\t%.4f\t%.4f\t%.3f\n",
                                split == 0 ? "even->odd" : split == 1 ? "odd->even" : split == 2 ? "all->all" : "other->all",
                                K, side_bytes,
                                fmt_bytes(B).c_str(), B, k ? "decode" : "prefill", n_test[k],
                                (double) side_hits[k] / row_acc[k], pacc[k] ? 1.0 - (double) reads[k] / pacc[k] : 1.0,
                                (double) reads[k] / n_test[k]);
                }
                fflush(f);
            }
        }
        fclose(f);
    }
    fclose(sm);
    printf("done: %zu tokens, %zu page accesses, %zu row accesses\n", T, pages.ids.size(), rowst.ids.size());
    return 0;
}
