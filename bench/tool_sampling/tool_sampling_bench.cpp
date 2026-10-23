// Host-side cost of llama-server's CPU sampling path for a Flint tool request.
// Mirrors common_sampler_init / common_sampler_sample / common_sampler_accept (common/sampling.cpp)
// without a llama_context: logits come from a synthetic buffer in which the "intended" token wins.
#include "llama.h"
#include "common.h"
#include "chat.h"
#include "sampling.h"
#include "reasoning-budget.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

static common_chat_tool tool(const char * name, const char * desc, const char * params) { return { name, desc, params }; }

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [n_reasoning_tokens] [dump_grammar]\n", argv[0]); return 1; }
    const int n_reason_rep = argc > 2 ? atoi(argv[2]) : 4;
    const bool dump = argc > 3 && atoi(argv[3]) != 0;

    llama_backend_init();
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    auto mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // ---- a coding-agent tool request (OpenCode-like tool set) ----
    auto tmpls = common_chat_templates_init(model, "");
    common_chat_templates_inputs in;
    common_chat_msg sys;  sys.role = "system"; sys.content = "You are a coding agent. Use the tools.";
    common_chat_msg user; user.role = "user";  user.content = "Create src/stats.py with a function that computes running mean and variance.";
    in.messages = { sys, user };
    in.tools = {
        tool("bash", "Run a shell command", R"({"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"number"},"description":{"type":"string"}},"required":["command","description"]})"),
        tool("read", "Read a file", R"({"type":"object","properties":{"filePath":{"type":"string"},"offset":{"type":"number"},"limit":{"type":"number"}},"required":["filePath"]})"),
        tool("write", "Write a file", R"({"type":"object","properties":{"filePath":{"type":"string"},"content":{"type":"string"}},"required":["filePath","content"]})"),
        tool("edit", "Replace text in a file", R"({"type":"object","properties":{"filePath":{"type":"string"},"oldString":{"type":"string"},"newString":{"type":"string"},"replaceAll":{"type":"boolean"}},"required":["filePath","oldString","newString"]})"),
        tool("glob", "Find files", R"({"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"}},"required":["pattern"]})"),
        tool("grep", "Search files", R"({"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"include":{"type":"string"}},"required":["pattern"]})"),
        tool("list", "List a directory", R"({"type":"object","properties":{"path":{"type":"string"},"ignore":{"type":"array","items":{"type":"string"}}}})"),
        tool("todowrite", "Update the todo list", R"({"type":"object","properties":{"todos":{"type":"array","items":{"type":"object","properties":{"content":{"type":"string"},"status":{"type":"string","enum":["pending","in_progress","completed","cancelled"]},"priority":{"type":"string","enum":["high","medium","low"]},"id":{"type":"string"}},"required":["content","status","priority","id"]}}},"required":["todos"]})"),
        tool("webfetch", "Fetch a URL", R"({"type":"object","properties":{"url":{"type":"string"},"format":{"type":"string","enum":["text","markdown","html"]},"timeout":{"type":"number"}},"required":["url","format"]})"),
    };
    in.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    in.parallel_tool_calls = true; // two calls in one turn below
    in.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    in.enable_thinking = true;

    common_chat_session sess(tmpls.get(), vocab, in);

    common_params_sampling sp;
    sp.top_k = 20; sp.top_p = 0.95f; sp.min_p = 0.05f; sp.temp = 1.0f;  // Flint's generation_config + server min_p default
    sess.apply_sampling(sp);
    // what server-common.cpp passes for a thinking template
    sp.reasoning_budget_start = common_tokenize(vocab, sess.thinking_start_tag(), false, true);
    for (const auto & t : sess.thinking_end_tags()) sp.reasoning_budget_end.push_back(common_tokenize(vocab, t, false, true));
    if (!sp.reasoning_budget_end.empty()) sp.reasoning_budget_forced = sp.reasoning_budget_end.front();

    const std::string & grammar_str = common_grammar_value(sp.grammar);
    printf("n_vocab %d, grammar %zu bytes, lazy %d, triggers %zu, thinking start '%s', end tags %zu, generation prompt '%s'\n",
           n_vocab, grammar_str.size(), (int) sp.grammar_lazy, sp.grammar_triggers.size(), sess.thinking_start_tag().c_str(),
           sess.thinking_end_tags().size(), sess.generation_prompt().c_str());
    for (const auto & t : sp.grammar_triggers) printf("  trigger type %d value '%s' token %d\n", (int) t.type, t.value.c_str(), t.token);
    if (dump) { printf("---- grammar ----\n%s\n-----------------\n", grammar_str.c_str()); return 0; }

    // ---- samplers, as common_sampler_init builds them ----
    std::vector<std::string> patterns; std::vector<llama_token> trig_tokens;
    for (const auto & t : sp.grammar_triggers) {
        if (t.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) trig_tokens.push_back(t.token);
        else if (t.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) patterns.push_back(regex_escape(t.value));
        else patterns.push_back(t.value);
    }
    std::vector<const char *> pc; for (auto & p : patterns) pc.push_back(p.c_str());
    llama_sampler * grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root", pc.data(), pc.size(), trig_tokens.data(), trig_tokens.size());
    if (!grmr) { fprintf(stderr, "grammar init failed\n"); return 1; }
    llama_sampler * rbudget = common_reasoning_budget_init(vocab, { sp.reasoning_budget_start }, sp.reasoning_budget_end, sp.reasoning_budget_forced, INT_MAX);
    for (auto tok : common_tokenize(vocab, sess.generation_prompt(), false, true)) llama_sampler_accept(rbudget, tok);

    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(sp.top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(sp.top_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(sp.min_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp_ext(sp.temp, 0.0f, 1.0f));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(1234));

    auto grammar_should_apply = [&]() {
        const auto st = common_reasoning_budget_get_state(rbudget);
        return st == REASONING_BUDGET_IDLE || st == REASONING_BUDGET_DONE;
    };

    // ---- the generated text: reasoning, then a write() call with a ~2 KB file ----
    std::string reasoning;
    for (int i = 0; i < n_reason_rep; ++i) {
        reasoning += "The user wants a module that keeps a running mean and variance. Welford's algorithm is numerically stable: "
                     "keep the count, the mean and the sum of squared differences, update them per sample, and divide at the end. "
                     "I should write the file with a small class, type hints and a docstring, then show how to use it. ";
    }
    std::string code =
        "\"\"\"Running statistics with Welford's algorithm.\"\"\"\n\nfrom __future__ import annotations\n\nimport math\n"
        "from dataclasses import dataclass, field\nfrom typing import Iterable\n\n\n@dataclass\nclass RunningStats:\n"
        "    \"\"\"Accumulates count, mean and variance in one pass.\"\"\"\n\n    count: int = 0\n    mean: float = 0.0\n"
        "    _m2: float = field(default=0.0, repr=False)\n\n    def push(self, x: float) -> None:\n        self.count += 1\n"
        "        delta = x - self.mean\n        self.mean += delta / self.count\n        self._m2 += delta * (x - self.mean)\n\n"
        "    def extend(self, xs: Iterable[float]) -> None:\n        for x in xs:\n            self.push(x)\n\n    @property\n"
        "    def variance(self) -> float:\n        if self.count < 2:\n            return 0.0\n        return self._m2 / (self.count - 1)\n\n"
        "    @property\n    def stdev(self) -> float:\n        return math.sqrt(self.variance)\n\n    def merge(self, other: \"RunningStats\") -> \"RunningStats\":\n"
        "        if other.count == 0:\n            return self\n        n = self.count + other.count\n        delta = other.mean - self.mean\n"
        "        mean = self.mean + delta * other.count / n\n        m2 = self._m2 + other._m2 + delta * delta * self.count * other.count / n\n"
        "        return RunningStats(n, mean, m2)\n\n\nif __name__ == \"__main__\":\n    rs = RunningStats()\n    rs.extend([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])\n"
        "    print(rs.count, rs.mean, rs.variance, rs.stdev)\n";
    const std::string text_pre  = "<think>" + reasoning + "</think>I will create the file.";
    const std::string text_call = "<tool_call>write<arg_key>filePath</arg_key><arg_value>src/stats.py</arg_value><arg_key>content</arg_key><arg_value>" + code + "</arg_value></tool_call>";
    const std::string text_call2 = "<tool_call>todowrite<arg_key>todos</arg_key><arg_value>[{\"content\": \"Write src/stats.py\", \"status\": \"completed\", \"priority\": \"high\", \"id\": \"1\"}, {\"content\": \"Add tests\", \"status\": \"pending\", \"priority\": \"medium\", \"id\": \"2\"}]</arg_value></tool_call>";

    auto toks_pre   = common_tokenize(vocab, text_pre,   false, true);
    auto toks_call  = common_tokenize(vocab, text_call,  false, true);
    auto toks_call2 = common_tokenize(vocab, text_call2, false, true);
    std::vector<llama_token> stream = toks_pre;
    const size_t i_call = stream.size();
    stream.insert(stream.end(), toks_call.begin(), toks_call.end());
    const size_t i_call2 = stream.size();
    stream.insert(stream.end(), toks_call2.begin(), toks_call2.end());
    stream.push_back(150005); // <|observation|>, an end-of-generation token
    printf("tokens: %zu before the call, %zu in write(), %zu in todowrite(), 1 EOG\n", i_call, toks_call.size(), toks_call2.size());

    // ---- synthetic logits ----
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 2.5f);
    std::vector<float> base(n_vocab);
    for (auto & v : base) v = nd(rng);
    const float vmax = *std::max_element(base.begin(), base.end());
    std::vector<float> logits(n_vocab);
    std::vector<llama_token_data> cur(n_vocab), cur2(n_vocab);

    struct acc { double t = 0; int n = 0; void add(double x) { t += x; ++n; } double avg() const { return n ? t / n : 0; } };
    acc a_copy[2], a_fill[2], a_rb[2], a_chain[2], a_check[2], a_accept[2], a_total[2], a_full_mask, a_full_resample;
    acc a_clone, a_scan_ms, a_scan_n, a_scan_free_ms, a_scan_free_n, a_full_free, a_full_con; int n_found_256 = 0; int n_few_valid = 0, n_measured = 0; size_t valid_min = SIZE_MAX;
    std::vector<llama_token> order(n_vocab);
    for (int t = 0; t < n_vocab; ++t) order[t] = t;
    std::sort(order.begin(), order.end(), [&](llama_token a, llama_token b) { return base[a] > base[b]; });
    double full_max = 0;
    int mismatches = 0, rejections = 0;

    for (size_t i = 0; i < stream.size(); ++i) {
        const llama_token target = stream[i];
        const int ph = i >= i_call ? 1 : 0; // 0 = before the trigger, 1 = inside tool calls

        // device -> host copy stands in as a host memcpy of the same size
        auto t0 = clk::now();
        std::copy(base.begin(), base.end(), logits.begin());
        logits[target] = vmax + 25.0f;
        a_copy[ph].add(ms_since(t0));

        const auto tt = clk::now();
        t0 = clk::now();
        for (int t = 0; t < n_vocab; ++t) cur[t] = llama_token_data{ t, logits[t], 0.0f };
        llama_token_data_array cur_p = { cur.data(), (size_t) n_vocab, -1, false };
        a_fill[ph].add(ms_since(t0));

        t0 = clk::now();
        llama_sampler_apply(rbudget, &cur_p);
        a_rb[ph].add(ms_since(t0));

        t0 = clk::now();
        llama_sampler_apply(chain, &cur_p);
        llama_token id = cur_p.data[cur_p.selected].id;
        a_chain[ph].add(ms_since(t0));

        t0 = clk::now();
        const bool gapply = grammar_should_apply();
        if (gapply) {
            llama_token_data single = { id, 1.0f, 0.0f };
            llama_token_data_array one = { &single, 1, -1, false };
            llama_sampler_apply(grmr, &one);
            if (one.data[0].logit == -INFINITY) { ++rejections; }
        }
        a_check[ph].add(ms_since(t0));
        a_total[ph].add(ms_since(tt));

        if (id != target) ++mismatches;

        // the cost of a rejection: grammar over the full row, then the chain again (measured, result discarded)
        if (ph == 1 && gapply && (i % 4 == 0 || i < i_call + 40 || i >= i_call2)) {
            for (int t = 0; t < n_vocab; ++t) cur2[t] = llama_token_data{ t, logits[t], 0.0f };
            llama_token_data_array c2 = { cur2.data(), (size_t) n_vocab, -1, false };
            t0 = clk::now();
            llama_sampler_apply(grmr, &c2);
            const double tm = ms_since(t0);
            a_full_mask.add(tm); full_max = std::max(full_max, tm);
            size_t n_valid = 0;
            for (int t = 0; t < n_vocab; ++t) n_valid += std::isfinite(c2.data[t].logit) ? 1 : 0;
            ++n_measured; if (n_valid < 20) ++n_few_valid; valid_min = std::min(valid_min, n_valid);
            // top-k-aware resample: check tokens in descending logit order until top_k valid ones are found
            t0 = clk::now();
            int found = 0, checked = 0;
            for (int j = -1; j < n_vocab && found < sp.top_k; ++j) {
                const llama_token t = j < 0 ? target : order[j];
                if (j >= 0 && t == target) continue;
                llama_token_data single = { t, logits[t], 0.0f };
                llama_token_data_array one = { &single, 1, -1, false };
                llama_sampler_apply(grmr, &one);
                ++checked; if (one.data[0].logit != -INFINITY) ++found;
            }
            const double tscan = ms_since(t0);
            a_scan_ms.add(tscan); a_scan_n.add(checked);
            if (n_valid >= 20) { a_scan_free_ms.add(tscan); a_scan_free_n.add(checked); a_full_free.add(tm); } else { a_full_con.add(tm); }
            if (found >= sp.top_k && checked <= 256) ++n_found_256;
            llama_sampler * chain2 = llama_sampler_clone(chain);
            t0 = clk::now();
            llama_sampler_apply(chain2, &c2);
            a_full_resample.add(ms_since(t0));
            llama_sampler_free(chain2);
        }

        if (ph == 1) {
            t0 = clk::now();
            llama_sampler * gclone = llama_sampler_clone(grmr);
            a_clone.add(ms_since(t0));
            llama_sampler_free(gclone);
        }
        t0 = clk::now();
        const bool accept_grammar = grammar_should_apply();
        llama_sampler_accept(rbudget, target);
        if (accept_grammar) llama_sampler_accept(grmr, target);
        llama_sampler_accept(chain, target);
        a_accept[ph].add(ms_since(t0));
    }

    const char * names[2] = { "before trigger (reasoning + text)", "inside tool calls" };
    for (int ph = 0; ph < 2; ++ph) {
        printf("%-34s n=%4d  memcpy600KB %.3f  fill %.3f  rbudget %.4f  chain %.3f  grammar-check %.4f  | sample total %.3f  accept %.4f ms\n",
               names[ph], a_total[ph].n, a_copy[ph].avg(), a_fill[ph].avg(), a_rb[ph].avg(), a_chain[ph].avg(), a_check[ph].avg(),
               a_total[ph].avg(), a_accept[ph].avg());
    }
    printf("full-row grammar apply (rejection / GBNF mask): n=%d avg %.2f ms, max %.2f ms; chain on the masked row %.3f ms\n",
           a_full_mask.n, a_full_mask.avg(), full_max, a_full_resample.avg());
    printf("valid tokens per position: min %zu, positions with < %d valid: %d of %d; top-k-aware scan: avg %.0f checks, %.2f ms\n",
           valid_min, sp.top_k, n_few_valid, n_measured, a_scan_n.avg(), a_scan_ms.avg());
    printf("positions with >= 20 valid: scan avg %.1f checks %.3f ms, full apply %.2f ms; positions with < 20 valid: full apply %.2f ms; top-k found within 256 checks: %d of %d\n",
           a_scan_free_n.avg(), a_scan_free_ms.avg(), a_full_free.avg(), a_full_con.avg(), n_found_256, n_measured);
    // the server re-parses the whole accumulated output for every streamed chunk (task_result_state::update_chat_msg)
    {
        std::vector<double> tf;
        for (size_t i = 0; i < stream.size(); ++i) {
            common_chat_input chunk;
            chunk.append(common_token_to_piece(vocab, stream[i], true), stream[i]);
            const auto t0 = clk::now();
            sess.feed(chunk);
            tf.push_back(ms_since(t0));
        }
        auto avg = [&](size_t a, size_t b) { double x = 0; for (size_t k = a; k < b; ++k) x += tf[k]; return x / (b - a); };
        const size_t n = tf.size();
        printf("chat re-parse per streamed token: tokens 0-99 %.3f ms, %zu-%zu %.3f ms, last 100 %.3f ms; whole response %.0f ms\n",
               avg(0, 100), i_call, i_call + 100, avg(i_call, i_call + 100), avg(n - 100, n), avg(0, n) * n);
        printf("parsed: %zu tool calls, reasoning %zu chars\n", sess.msg().tool_calls.size(), sess.msg().reasoning_content.size());
    }
    printf("grammar sampler clone (server, once per verify step): avg %.3f ms\n", a_clone.avg());
    printf("sampled != intended: %d, single-token grammar rejections: %d\n", mismatches, rejections);
    return 0;
}
