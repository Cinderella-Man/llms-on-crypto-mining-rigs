# Q161: a store of Flint's own answers behind the copy lookup (old stores, 2026-10-24/25)

Q161 put a continuation store, a suffix array over many of Flint's answers
([datastore.cpp](../../../bench/ngram_replay/datastore.cpp)), behind the copy lookup and the request's own counts, and
replayed Q51's 38 held-out transcripts through `pipeline_sim.py`'s nine-card model. It then rebuilt the store from
licence-filtered subsets of the same answers, with a per-package licence ruling for the prompts cut from third-party
Python packages, a builder that writes a manifest per store, and an independent checker. The result is in the registry
row Q161 ([quants/experiments.md](../../../quants/experiments.md)) and in
[pipelined_speculation.md](../../research/notes/pipelined_speculation.md) ("A datastore of Flint's own answers").
**Every store here was built from the first answer set of the staircase drafter (Q20), which was deleted on licence
grounds on 2026-10-25** (it mixed in prompt sets whose licence or origin does not allow publishing, see
[docs/drafter.md](../../drafter.md)). The stores went with it and none was released, so these tables are the only
record of their scores. The method (builder, checker, allow lists, the licence ruling) applies to any new answer set.

## Results

Q51 held out, the engine's chain (copy ≥ 4, else request counts with context ≥ 2, else the store with context ≥ 2 and
top continuation ≥ 30 % of its occurrences, else request counts), `pipeline_sim.py` speed-up over plain decoding, nine
cards 6/6/6/6/5/5/5/5/5, depth 3. Every store holds about 13.8 M tokens (110 MB as a PIPESTO1 file) unless stated.

| Store | Prompt sources | Answers | Tokens | Chat | Code | Edits | Agent | All | Tool calls | Right 1 ahead | Table |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none (`cont`) | – | – | – | 1.19 | 1.36 | 1.87 | 1.30 | 1.44 | 1.18 | 39.2 % | all |
| old (`clean-14M`, `current`, `old`) | every prompt set of the first answer set | 11,597 | 13,798,889 | 1.25 | 1.43 | 1.95 | 1.45 | 1.52 | 1.74 | 43.5 % | all |
| noshare-same | old without ShareGPT | 9,761 | 11,212,145 | 1.24 | 1.43 | 1.95 | 1.44 | 1.52 | 1.77 | 43.2 % | `variants/score.md` |
| noshare-all | the same, from the larger answer pool | 14,404 | 13,798,404 | 1.24 | 1.43 | 1.95 | 1.45 | 1.52 | 1.79 | 43.4 % | `variants/score.md` |
| strict-same | maths and tool-call sets, Python stdlib and llama.cpp files | 4,506 | 3,798,402 | 1.20 | 1.38 | 1.95 | 1.43 | 1.48 | 1.80 | 41.2 % | `variants/score.md` |
| strict-all | the same, from the larger answer pool | 22,376 | 13,797,539 | 1.21 | 1.38 | 1.96 | 1.45 | 1.49 | 1.84 | 41.6 % | `variants/score.md` |
| strict (v1) | `strict.allow` | 22,383 | 13,797,929 | 1.21 | 1.38 | 1.96 | 1.45 | 1.49 | 1.84 | 41.6 % | `strict/score.md` |
| strict v2 | v1 + the allowed site-packages | 20,842 | 13,797,576 | 1.21 | 1.39 | 1.97 | 1.45 | 1.49 | 1.86 | 41.8 % | `strict/score2.md` |
| strict v2 notool | v2 without ToolACE, glaive and hermes | 13,463 | 13,798,462 | 1.20 | 1.38 | 1.97 | 1.46 | 1.49 | 1.87 | 41.7 % | `strict/score2.md` |
| core | my own edit and agent prompts only | 14,868 | 13,798,111 | 1.19 | 1.39 | 1.98 | 1.47 | 1.49 | 1.90 | 41.7 % | `strict/score3.md` |
| core + OpenR1 (`release-20261025`) | core + OpenR1-Math (default config) | 13,702 | 13,798,379 | 1.19 | 1.39 | 1.98 | 1.46 | 1.49 | 1.90 | 41.6 % | `strict/score4.md` |
| core-b | core, both Q51 tests | 14,859 | 13,797,089 | 1.19 | 1.39 | 1.98 | 1.47 | 1.49 | 1.90 | 41.7 % | `strict/score5.md` |
| release-b | `release.allow`, both Q51 tests | 13,696 | 13,798,884 | 1.19 | 1.39 | 1.98 | 1.46 | 1.49 | 1.90 | 41.6 % | `strict/score5.md` |

- **Dropping ShareGPT costs nothing measurable:** both noshare stores stay at 1.52 overall.
- **Answers to prompts of clear licence keep the edit, agent and tool-call gains** (tool calls 1.84–1.90 against 1.74)
  but lose most of the chat and code gain: 1.49 overall against 1.52, still +0.05 over no store. With the output head
  and 2 layers on the last card the same stores reach 1.50 (strict, core, release-b) against 1.54.
- **Tool-call sets and maths add nothing on Q51:** v2 notool equals v2 (the agent answers already teach the store the
  tool-call format), and OpenR1 adds nothing to core (Q51 has no maths).
- **Store guesses:** core-b answers 21.5 % of positions from the store, 23.2 % of them right; the old store 26.0 % and
  27.9 %.
- **The audit:** ToolACE, glaive-function-calling-v2 and hermes-function-calling-v1 are written by unnamed LLMs, and
  about 45 % of NuminaMath-CoT's problems are GPT-4-written, so the strict stores were not releasable. The release
  candidate was core + OpenR1-Math's default config (crawled problems).
- **The checker:** both "b" stores PASS (`strict/verify-*.txt`). release-b by licence: MIT 43.7 %, Apache-2.0 39.6 %
  (OpenR1 included), PSF-2.0 12.2 %, BSD-3-Clause 2.9 %, BSD-2-Clause 0.8 %, Apache-2.0 OR BSD-2-Clause 0.5 %.

**Q51 hold-out.** An answer is left out when its prompt shares a 12-word span with a Q51 prompt, by two tests: user and
tool messages one at a time (the first store's rule, 352 edit answers), and all messages' text joined, the rule of
[q51_overlap.py](../2026-10-24_q20_drafter/run/q51_overlap.py). The second test also catches 10 agent answers whose task
ends with ten words of a Q51 agent task followed by the file's first line; the "b" stores apply both.

**The licence ruling** (`strict/strict-sitepkgs.md`): MIT, BSD-2/3-Clause, Apache-2.0 and PSF in, judged per package and
per vendored sub-package from each package's metadata. Out: tqdm (MPL-2.0 AND MIT), pip's vendored certifi (MPL-2.0)
and resolvelib (ISC), and Arrow's vendored portable-snippets (CC0-1.0) and `strptime.h` (unclear origin): 244 answers,
0.29 M tokens; 6,481 answers, 6.70 M tokens stay in.

## Files

| File | What |
|---|---|
| `out/accuracy.md`, `.json` | one token ahead on the true prefix: 5 old stores (3, 7 and 13.8 M tokens, `all-14M` without the hold-out, `noagent-13M` without agent and tool-call answers) × store policies and chain orders |
| `out/accuracy-extras.md`, `.json` | chain orders and the oracle choice of store or request counts (a bound), `clean-14M` |
| `out/sim.md`, `.json` | `pipeline_sim.py`, both nine-card splits, depths 1/2/3/4/8, store-first and request-first chains, the 5 old stores |
| `out/work-clean-3M.json`, `work-clean-14M.json` | lookup work per query (probes, scanned entries, cache entries), counted, not timed |
| `variants/score.md`, `.json` | noshare and strict variants against the old store (`current`), depth 3 |
| `variants/variants.json` | composition of each variant and pool (tokens per source class, Q51 hold-out) |
| `strict/score.md` … `score5.md` (+ `.json`) | strict v1; v1/v2/v2-notool; core; core + OpenR1; core-b and release-b, each against the old store, depth 3 |
| `strict/<store>.json` | `store_build.py`'s report per store: allow list, cap, tokens per source and per category, Q51 hold-out counts |
| `strict/strict-sitepkgs.md` | the licence ruling per package and vendored sub-package, with answers and tokens per row |
| `strict/*.allow` | allow lists: `everything` (reproduces the first store), `strict`, `core`, `core-openr1`, `release` |
| `strict/verify-*.txt` | `verify_store.py` on core-b and release-b: tokens per source and per licence, PASS |
| `run/store_build.py` | builds a store's token stream, manifest and report from an allow list and/or an id list |
| `run/verify_store.py` | checks a store against the rules from the answer and prompt files alone, and rebuilds its stream from the manifest |
| `run/store_variants.py` | the noshare and strict variants of the first round |
| `run/*.sh` | the score runs: `run_sim.sh` (`out/sim`), `variants_score.sh`, `score.sh`, `score2.sh`, `score3.sh`, `core_openr1.sh` (`score4`), `score5.sh` |

Store names in the tables: `old`, `current` and `clean-14M` are the same first store; `-same` stores come from the
first store's answer files, `-all` stores add the answer files written later (Flint G3, UD-Q4_K_XL and NVFP4 answers).

## Made on 2026-10-24 and 2026-10-25 with

CPU only, with [bench/ngram_replay](../../../bench/ngram_replay/) (`datastore.cpp`, `datastore_corpus.py`,
`datastore_sim.py`); `libdatastore-v3.so` in the scripts is `datastore.cpp` built as below.

```bash
# the store library and the build tool
g++ -O2 -std=c++17 -shared -fPIC -o libdatastore.so datastore.cpp && g++ -O2 -std=c++17 -DDS_MAIN -o datastore datastore.cpp
# a store: token stream, manifest and report, then the PIPESTO1 file, then the independent check
export ANSWERS=<dir of the drafter's gen-*.jsonl and prompts-*.jsonl> LLAMA_SRC=<llama.cpp tree the prompts came from>
python3 -I run/store_build.py release-b --allow strict/release.allow
./datastore build release-b.i32 99999999 release-b.pstore
python3 -I run/verify_store.py release-b.manifest.tsv release-b.i32 release-b.pstore strict/strict-sitepkgs.md
# the Q51 replay (WORK holds src/ = bench/ngram_replay, the library and pstore/)
WORK=<work dir> Q51_GEN=$PWD/release/flint-v2/reproduce/data/q51/gen.jsonl bash run/score5.sh
```

Check of the builder: with `everything.allow`, `--pool same` and `--q51-rule store` it reproduces the first store's
stream byte for byte. The builder and the checker tell standard-library files by the top-level names in
`/usr/lib/python3.12` (Ubuntu's python3.12; in the `python:3.12-slim` image link it to `/usr/local/lib/python3.12`). The first stores (`out/`) were built with `datastore_corpus.py`; the Q51 transcripts are
[release/flint-v2/reproduce/data/q51](../../../release/flint-v2/reproduce/data/q51/README.md).

The tables hold counts and scores only, no prompt or answer text. The answers were Flint's outputs (Flint is
Apache-2.0); the prompt sources and their licences are listed in `strict/strict-sitepkgs.md` and the allow lists.
