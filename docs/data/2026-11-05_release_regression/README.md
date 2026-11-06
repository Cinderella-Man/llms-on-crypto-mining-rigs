# Release regression: Flint and Ornith-122B on Miner Llama v3 (2026-11-05 and 2026-11-06)

Miner Llama v3's release build is `pascal-p106-v3-b12076-ladder5`. These tests check that it does not break or slow
down the two models of the earlier releases: Flint (Unsloth's Q8_0, the v2 release model) and Ornith-1.5-122B
(valid1, the N2P/N3P file of [Q189](../2026-10-28_lowbit_rig/README.md)). All runs are on one rig of nine P106-090
cards with 4 GB of host RAM.
- **Greedy identity:** the same tokens as the old build, and pipelined speculation the same as plain decoding.
- **KL divergence:** the size of the difference.
- **The agent check** ([bench/agent_check](../../../bench/agent_check/README.md)): a coding agent fixing 12 small
  bugs with real tool runs.

Every speed below is one measured run with its file, build and settings.

## Builds and files

| Name here | Build | What |
|---|---|---|
| v2.0 | fork `37e717ff6` (tag `miner-llama-v2.0`), the rc4 rig build | Flint's v2 release engine |
| `-lowbit` | fork `d7a75deb3` (`pascal-p106-v3-lowbit`) | Q189's engine for Ornith valid1 |
| `-ladder5` `1f7215948` | fork `pascal-p106-v3-b12076-ladder5` at `1f7215948`, rig tarball md5 `6111523566e8de73012133239b9f6db5` | the v3 release build as first frozen: every feature behind a switch whose default is the old behaviour |
| `-ladder5` `1551f5fe1` | the same branch at `1551f5fe1` (the MMVQ row-guard fix), md5 `288b57025e6ff64e76b90674a0509228` | the frozen tip from 2026-11-05 ~18:20 |
| stock | upstream llama.cpp `20a51f24a`, `-ladder5`'s merge base (tag b12076 + 16 commits), no fork commits | the stock baseline |

Files: Unsloth's `flint-Q8_0.gguf` (42,301,076,928 bytes), `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`
(48,196,722,592 bytes, [its data](../2026-10-28_ornith_valid1/README.md)) and, on 2026-11-06, Flint Pascal Q8 GGUF
(p065, `flint-near-q8-p065.gguf`, 40,431,236,544 bytes, md5 `8387b2e585a7b9cc10f40cb6b4531694`).

## Results

### The agent check

Flint's line is the v2 release line: `-c 262144 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4`. Ornith's is
Q189's: `-c 49152 -b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off`, at the makers' sampling (T 0.6). Both add the
harness's flags (`-np 1 -bs --ctx-checkpoints 4 --cache-ram 0 ...`).

| Model, file | Build | Arm: settings | Tasks passed | Minutes, all tasks | Minutes per task (median) | TTFT per turn s (median, p90) | Prompt tok/s | Generation tok/s | Tool calls / failed / malformed | Runaways |
|---|---|---|---|---:|---:|---|---:|---:|---|---:|
| Flint Q8_0 | `-ladder5` `1f7215948` | `v3base`: the v2 line, every switch at its default | 12 / 12 | 19.2 | 1.2 | 2.67, 6.48 | 168 | **45.9** | 71 / 3 / 0 | 0 |
| Flint Q8_0 | `-ladder5` `1f7215948` | `v3add`: + `LLAMA_SERVER_TEXT_REUSE=1 LLAMA_SERVER_TAIL_CKPT_MIN=4096 GGML_CUDA_KEEPALIVE=decode` | 12 / 12 | 21.6 | 1.2 | 2.45, 6.89 | 168 | 44.7 | 73 / 2 / 0 | 0 |
| Flint Q8_0 | stock `20a51f24a` | `stock64k`: `-c 65536 -b 2048 -ub 512 -fa on`, same split (stock's tuned line, below); 8 tasks on 2026-11-05, the last 4 on 2026-11-06 | 10 / 12 | 52.6 | 2.9 | 4.13, 8.04 | 148 | 14.4 | 59 / 2 / 0 | 2 |
| Flint Pascal Q8 GGUF (p065) | `-ladder5` `1551f5fe1` | `v3base`, 2026-11-06 | 12 / 12 | 22.3 | 1.0 | 2.66, 6.75 | 175 | **47.4** | 74 / 2 / 0 | 0 |
| Ornith valid1 | `-ladder5` `1f7215948` | `valid1`: Q189's line, every switch at its default | 12 / 12 | 32.2 | 2.1 | 9.41, 19.85 | 59 | **14.9** | 62 / 0 / 0 | 0 |
| Ornith valid1 | `-ladder5` `1f7215948` | `valid1-ka`: + `GGML_CUDA_KEEPALIVE=decode` | 12 / 12 | 32.0 | 2.1 | 8.39, 19.38 | 58 | **16.3** | 63 / 1 / 0 | 0 |
| Ornith UD-Q2_K_XL | `-ladder5` `1551f5fe1` | `ud-v3`: R3's line `-c 49152 -b 2048 -ub 512 -ts 8,6,5,5,5,4,5,6,6 -fit off`, every switch at its default, 2026-11-06 | 12 / 12 | 28.3 | 1.8 | 7.90, 17.24 | 69 | 12.6 | 54 / 0 / 0 | 0 |
| Ornith UD-Q2_K_XL | stock `20a51f24a` | `ud-stock`: `-c 49152 -b 2048 -ub 256 -ts 8,6,5,5,5,4,5,6,6 -fit off -fa on` (stock's tuned line), 2026-11-06 | 11 / 12 | 56.5 | 3.1 | 12.57, 20.55 | 55 | 6.0 | 56 / 2 / 0 | 2 |

The earlier numbers on the same files and lines:
- **Flint Q8_0 on v2.0:** 45.9 and 44.6 tok/s in two runs (12 / 12 each;
  [release/flint-v2/results.md](../../../release/flint-v2/results.md)).
- **Ornith valid1 on `-lowbit`:** 14.5 tok/s, 32.5 minutes, 12 / 12 (Q189).

**Verdicts:**
- **No regression on either model.** Flint's `v3base` (45.9) equals v2.0's faster run. Ornith's base arm (14.9) is
  above Q189's 14.5.
- **`v3add` on Flint is negligible:** 44.7 against 45.9, inside the 1.3 tok/s between v2.0's two runs. The re-read
  fix kept cached ids on 13 requests (0 in `v3base`), and the median first token was 2.45 against 2.67 s.
- **The keep-alive on Ornith is a gain on generation:** 16.3 against 14.9 tok/s (+9.4 %), and the median first token
  8.39 against 9.41 s. The minutes for all tasks are equal (32.0 against 32.2), since answers at T 0.6 differ in
  length.
- **Stock against Miner Llama, Flint Q8_0:** 10 / 12 against 12 / 12 tasks, generation 14.4 against 45.9 tok/s
  (3.2×), and 2.9 against 1.2 minutes per task (median). Stock's two misses ran into the 12-minute cap.
- **Stock against Miner Llama, Ornith on Unsloth's UD-Q2_K_XL** (stock cannot load valid1's N2P/N3P types): 11 / 12
  against 12 / 12 tasks, generation 6.0 against 12.6 tok/s (2.1×), 3.1 against 1.8 minutes per task (median), the
  first token at 12.57 against 7.90 s.
- **valid1 against UD-Q2_K_XL on v3, no switch:** generation 14.9 against 12.6 tok/s (+18.3 %), but prompts at 59
  against 69 tok/s, the first token at 9.41 against 7.90 s, and 2.1 against 1.8 minutes per task (median; 5 against 4
  turns). At equal work the file is ~5 % faster a task (est., from each arm's mean prefill and decode seconds a task).
- **Flint Pascal Q8 GGUF against Unsloth's Q8_0, both on `-ladder5`:** 47.4 against 45.9 tok/s generation (+3.3 %),
  the first token at 2.66 against 2.67 s, 1.0 against 1.2 minutes per task (median); both 12 / 12. Its sessions
  generated more (85 against 74 s of decoding a task, mean), so all twelve took 22.3 against 19.2 minutes. Q8_0's two
  v2.0 runs were 45.9 and 44.6.

### Stock's line (Flint Q8_0)

[stock_tune/flint.md](stock_tune/flint.md): llama-bench on stock with v3base's split, `-b 2048`, `-ub` 128 / 256 /
512, flash attention off and on, a 512-token prompt and 64 generated tokens at depths 0 and 4,096.
- **The pick:** `-ub 512` with flash attention on, which costs the least for a turn of 600 new prompt tokens and 600
  generated at depth 4,096: 157.5 prompt and 17.48 generation tok/s. Flash attention off loses 40 % of generation
  speed at that depth.
- **The context:** at `-ub 512` stock keeps an attention mask of context × micro-batch on the host, so the arm ran at
  65,536 tokens, as v2's stock arm did. A bug-fix task stays far below that.
- **No speculation on stock:** stock has no pipelined speculation, and Flint has no MTP layer of its own. Stock's
  draft-free n-gram drafts were not tried.

### Greedy identity

[run/greedy_identity.sh](run/greedy_identity.sh) uses four prompts ([run/prompts.jsonl](run/prompts.jsonl): two code
tasks, a 3,406-token script review and a prose answer), 256 tokens each at temperature 0, on each build's server.
Each prompt is sent twice: on the server's own path (pipelined speculation where the line has `--pipe-depth`) and
with top-2 probabilities, which the server decodes plainly. The tables:
- [identity/flint_v2_vs_ladder5_1f7215948.md](identity/flint_v2_vs_ladder5_1f7215948.md)
- [identity/flint_1f7215948_vs_1551f5fe1.md](identity/flint_1f7215948_vs_1551f5fe1.md)
- [identity/flint_v2_vs_1551f5fe1.md](identity/flint_v2_vs_1551f5fe1.md)
- [identity/ornith_lowbit_vs_ladder5_1f7215948.md](identity/ornith_lowbit_vs_ladder5_1f7215948.md)

| Pair | Same template and prompt tokens | Same generated tokens | Pipelined = plain on each build |
|---|---|---|---|
| Flint, `-ladder5` `1f7215948` against `1551f5fe1` | yes | **yes, 8 / 8** | yes |
| Flint, `-ladder5` at defaults against `-ladder5` with `v3add`'s three switches | yes | **yes**: the same split tokens and numbers against v2.0 | yes |
| Flint, v2.0 against `-ladder5` (both commits) | yes | **no:** each prompt splits at token 123, 60, 99, 133 | yes |
| Ornith valid1, `-lowbit` against `-ladder5` `1f7215948` | yes | **no:** splits at token 24, 4, 70, 31 | yes |

At the split, the old build's top two choices differ by this much in log-probability: Flint 0.011, 0.024, 0.056 and
0.343 nats; Ornith 0.19, 0.16, 0.83 and 0.15.

### How large the difference is

**Prompt passes (Flint, [kld/flint.tsv](kld/flint.tsv)):** llama-perplexity on wikitext-2 test, 16 chunks of 512
tokens, the release line's batches (`-b 512 -ub 128`) and split. The distributions of `1551f5fe1` against v2.0:
- mean KLD −0.000000 ± 0.000000, maximum 0.000056 ([kld/flint_stats.txt](kld/flint_stats.txt));
- the same top token 100.000 %;
- PPL 24.4293 ± 1.5576 on both builds.

So prompt processing matches v2.0 to rounding. Q8_0 itself costs 0.052 KLD against BF16.

**One-token decoding ([run/decode_drift.py](run/decode_drift.py) on the probs requests):** the chosen token's
log-probability over the tokens both builds generated alike.

| Pair | Tokens | Median abs difference | Mean | p90 (per prompt) | Maximum |
|---|---:|---:|---:|---:|---:|
| Flint, v2.0 against `-ladder5` | 415 | 0.0001 | 0.0162 | 0.044–0.065 | 0.438 |
| Ornith, `-lowbit` against `-ladder5` | 129 | 0.0001 | 0.0236 | 0.055–0.130 | 0.494 |

**What this shows:**
- **Generation drifts, prompt passes don't.** Half the generated tokens move by 0.0001 nats or less, rounding-sized,
  but a tail reaches 0.4–0.5 nats.
- **The likeliest cause** is that v3 sums the one-token matrix products in a different order: two warps a block on
  compute 6.1 (`GGML_CUDA_MMVQ_2WARPS`, default 1), against v2's four. These are per-step rounding differences, and
  they compound through the attention cache over a generation. This has not been confirmed. The check is a greedy
  identity run of `-ladder5` with `GGML_CUDA_MMVQ_2WARPS=0` against v2.0.
- **No loss of function:** both models pass 12 / 12 at full speed on `-ladder5`.

**Ornith's prompt passes do differ ([kld/ornith_lowbit_vs_1551f5fe1.tsv](kld/ornith_lowbit_vs_1551f5fe1.tsv),
[the statistics](kld/ornith_lowbit_vs_1551f5fe1_stats.txt)).** The same run on the rig, 2026-11-06: valid1 on nine
P106-090, wikitext-2 test, 16 chunks of 512 tokens, Q189's line (`-b 2048 -ub 512`), `1551f5fe1` scored against
`-lowbit`. It runs on the rig because `-lowbit`'s build does not start on an RTX 3090 (its first kernel launch fails
with "invalid argument"), and on a compute 8.6 card neither build takes its Pascal kernels.
- mean KLD 0.021653 ± 0.000881, median 0.006179, 99th percentile 0.224798, maximum 1.654823;
- the same top token 93.873 ± 0.376 %;
- perplexity 5.9938 ± 0.2410 on `1551f5fe1` against 6.0735 ± 0.2449 on `-lowbit`; paired by token, ln(PPL ratio)
  −0.0126 ± 0.0043.

So on Pascal the two builds compute Ornith's prompt passes differently, far beyond Flint's rounding (0.000056 at most),
and the release build's perplexity is the lower. On the 3090s, where neither build takes its Pascal kernels, the same
two builds differ by +0.00167 ± 0.00078 in KLD against the Q8_0 reference on this text
([2026-11-05_night_quant_chain](../2026-11-05_night_quant_chain/README.md)). Which build is closer to the model on
Pascal is the next run: both against that Q8_0 reference on the P106.

## Files

| Path | What |
|---|---|
| `agent/<model>-<arm>/` | each agent-check arm's report: `compare.md` (per task, per arm, first token per turn), `table.md`, `gonogo.md`, `attempts.csv`, `turns.csv`, `validation.txt` |
| `identity/*.md` | the greedy identity tables (compare_greedy.py) |
| `identity/*-greedy-*.jsonl` | each build's greedy requests: token ids, text, timings and the top-2 log-probabilities |
| `kld/flint.tsv`, `kld/flint_stats.txt` | Flint's KLD of `1551f5fe1` against v2.0 |
| `kld/ornith_lowbit_vs_1551f5fe1.tsv`, `_stats.txt` | Ornith valid1's KLD of `1551f5fe1` against `-lowbit` on the P106 |
| `stock_tune/flint.md`, `.jsonl` | stock's llama-bench grid and pick |
| `stock_tune/ornith-ud.md`, `.jsonl` | the same for Ornith's UD-Q2_K_XL (split `8,6,5,5,5,4,5,6,6`, `-fit off`): `-ub 256`, flash attention on |
| `run/greedy_identity.sh`, `greedy_requests.py`, `compare_greedy.py`, `prompts.jsonl` | the identity check |
| `run/decode_drift.py` | the one-token drift table |
| `run/kld_builds.sh` | the KLD between builds on the rig |
| `run/kld_docker.sh` | the same in Docker on a build machine's GPUs (compute 6.1 PTX) |
| `run/agent_flint.sh` | Flint's arms (`v3base`, `v3add`, the one-change arms, `stock`, `stock64k`) |
| `run/stock_tune.sh` | stock's llama-bench grid and pick |
| `run/tk_cost.sh` | T(k) for the adaptive draft's cost table |

Ornith's arms ran through [../2026-10-27_ornith/run/agent_ornith.sh](../2026-10-27_ornith/run/agent_ornith.sh)
(candidate `valid1`, `LOWBIT` = the `-ladder5` server, and `REL_ENV` for the keep-alive; candidates `ud-v3` with
`V3` = the `-ladder5` server and `ud-stock` with `STOCK` = stock's, `UD` = the UD-Q2_K_XL file and `UD_STOCK_LINE` =
stock's tuned line).

## How it was made

From the repository root, on a machine that only runs ssh, with the agent check's machine values in the environment
(`RIG`, `CLIENT`, `RIG_PORT`, `RELAY_PORT`, `CLIENT_PORT`, `PROXY_PORT`, `CLIENT_DIR`, `RIG_LOGDIR`, as in
[release.env.example](../../../bench/agent_check/release.env.example)):

```bash
R=docs/data/2026-11-05_release_regression/run
L5=llama.cpp-20261105-1f7215948-pascal-p106-v3-b12076-ladder5   # then the 1551f5fe1 tarball's folder
# greedy identity (Flint: v2.0 then -ladder5; then -ladder5 with v3add's switches)
RIG=rig MODEL=<flint-Q8_0.gguf> BUILDS="llama.cpp-20261025-37e717ff6-pascal-p106-v2-rc4 $L5" \
  LINE="-b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4" CTX=262144 DIR=<rig folder> OUT=id_flint bash $R/greedy_identity.sh
RIG=rig MODEL=<flint-Q8_0.gguf> BUILDS="$L5" LINE="<the same>" CTX=262144 DIR=<rig folder> OUT=id_flint TAG_SUFFIX=-v3add \
  SERVER_ENV="LLAMA_SERVER_TEXT_REUSE=1 LLAMA_SERVER_TAIL_CKPT_MIN=4096 GGML_CUDA_KEEPALIVE=decode" bash $R/greedy_identity.sh
python3 $R/compare_greedy.py id_flint/greedy-v2-rc4.jsonl id_flint/greedy-v3-b12076-ladder5.jsonl id_flint/greedy-v3-b12076-ladder5-v3add.jsonl
python3 $R/decode_drift.py id_flint/greedy-v2-rc4.jsonl id_flint/greedy-v3-b12076-ladder5.jsonl
# Ornith: BUILDS="llama.cpp-20261027-d7a75deb3-pascal-p106-v3-lowbit $L5" LINE="-b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off" CTX=49152
# KLD on wikitext-2 test (Flint)
RIG=rig MODEL=<flint-Q8_0.gguf> BUILDS="llama.cpp-20261025-37e717ff6-pascal-p106-v2-rc4 <1551f5fe1 folder>" \
  ARGS="-b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6" TEXT=<wiki.test.raw on the rig> DIR=<rig folder> OUT=kld_flint bash $R/kld_builds.sh
# the agent check, Flint
MODEL=<flint-Q8_0.gguf on the rig> BIN=/opt/llama-builds/$L5/bin/llama-server CLIENT_ROOT=<client folder> OUT=agent_flint \
  ARMS_LIST="v3base v3add" bash $R/agent_flint.sh
# stock: the grid, then the arm on its pick
RIG=rig MODEL=<flint-Q8_0.gguf> BIN=<stock folder>/bin/llama-bench TS=6/6/5/5/6/5/5/6/6 DIR=<rig folder> OUT=tune bash $R/stock_tune.sh
MODEL=<flint-Q8_0.gguf> BIN=<as above> STOCK_BIN=<stock folder>/bin/llama-server STOCK_LINE="-b 2048 -ub 512 -ts 6,6,5,5,6,5,5,6,6 -fa on" \
  BUDGET_MIN_rel=36 CLIENT_ROOT=<client folder> OUT=agent_flint ARMS_LIST=stock64k bash $R/agent_flint.sh
# the agent check, Ornith (add REL_ENV=GGML_CUDA_KEEPALIVE=decode for valid1-ka, with its own CLIENT_WORK)
CLIENT_WORK=<client folder> AD=unused STOCK=unused V2=unused LOWBIT=/opt/llama-builds/$L5/bin/llama-server \
  VALID1=<valid1 on the rig> VALID1_LINE="-b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off" OUT=agent_ornith CANDS=valid1 \
  bash docs/data/2026-10-27_ornith/run/agent_ornith.sh
```

The stock arm's client was stopped at the end of the rigs' day (22:00), during task 9. Run again with the same client
folder, `release_check.sh` skips the finished attempts and runs the rest.

**Provenance and licence:** the prompts are this repository's own. The tables summarize runs of Unsloth's Flint Q8_0
and of the Ornith valid1 file on the engine builds above. The texts the agent check runs on are its Exercism-based
bug-fix tasks, built by [build_bugfix.py](../../../bench/agent_check/build_bugfix.py) under their MIT licence.
