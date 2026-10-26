# Every number behind the release

Each row: what was measured, the value, how (the command or public script), the engine build, the date and the
experiment's row in the [registry](../../quants/experiments.md#registry), which holds the full result, its conditions
and the raw-data note. Speed is on the reference rig (nine P106-090, PCIe 1.0 x1, 2-core Celeron, 4 GB RAM) unless a
row says otherwise; quality is scored on a separate machine against Flint's BF16 logits. tok/s = generated tokens per
second; "warm" = `llama-bench` with a 512-token prompt run before the 128 generated tokens in the same process.

Builds (fork commits): stock = upstream `d77256999`; v1 = `eeaa59660`; v2 base = `032b130f4`; combined = `b3052a4f8`;
persistent = `8de13900f`; v2-next = `62fa5fc3b`; rc4 = `2d53179aa`; release = rc4 `37e717ff6` (`2d53179aa` with the
defaults the one-by-one test chose, below).

The trained drafter and the store of earlier answers come later (v2.1); their measurements are in
[docs/drafter.md](../../docs/drafter.md).

## Agentic speed: a coding agent at work (2026-10-26)

The release is chosen and reported on this measure. [bench/agent_check](../../bench/agent_check/README.md) gives the
pi coding agent twelve small bug-fix tasks (six Python, six JavaScript, real test runs, five to ten tool calls each,
the context growing every turn) against a llama-server on the rig, 25 minutes for the release engine (35 for stock),
one attempt per task, and records every request. Nine P106-090, `-np 1 -bs --ctx-checkpoints 4 --cache-ram 0`, the
makers' sampling. "Tasks" = tasks finished within the time (passed); generation and prompt reading in tokens per second
over all requests of the run; first token = the median per turn; runaways = sessions stopped at the 12-minute cap.

| File, engine, server settings | Tasks | Minutes per task, median / p90 | Generation | Prompt reading | First token | Runaways |
|---|---|---|---|---|---|---|
| Flint-Pascal-Q5, release, 262k, `-b 512 -ub 128 --pipe-depth 4` | **12 (12)** in 16.7 min | 1.2 / 2.7 | **53.8** | 178 | 2.6 s | 0 |
| Flint-Pascal-Q5, release, 262k, `--pipe-depth 3` | 10 (10) | 1.0 / 12.0 | 48.6 | 180 | 2.7 s | 1 |
| Flint-Pascal-Q5, stock, 64k, `-b 2048 -ub 512` | 11 (10) | 2.1 / 6.9 | 15.6 | 143 | 4.2 s | 1 |
| UD-Q4_K_XL, release, 262k, `-b 512 -ub 128 --pipe-depth 4` | 11 (10) | 1.1 / 4.1 | 50.7 | 191 | 2.5 s | 1 |
| UD-Q4_K_XL, stock, 64k, `-b 2048 -ub 512` | 9 (8) | 3.1 / 12.0 | 14.0 | 145 | 3.7 s | 1 |
| Q8_0, release, 262k, `-b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4` (two runs) | 12 (12), 12 (12) | 1.1 / 3.5, 1.4 / 7.8 | **45.9, 44.6** | 176, 185 | 2.6 s | 0, 0 |
| Q8_0, the same at `--pipe-depth 3` (two runs) | 11 (11), 10 (9) | 1.0 / 3.6, 1.3 / 12.0 | 44.6, 44.6 | 184, 191 | 2.4, 2.6 s | 0, 1 |
| Q8_0, the same at `--pipe-depth 2` | 10 (10) | 1.0 / 10.9 | 40.1 | 179 | 2.4 s | 0 |
| Q8_0, the same without pipelined speculation | 10 (9) | 1.7 / 12.0 | 29.3 | 184 | 2.3 s | 1 |
| Q8_0, `--pipe-depth 3` with the agent prompt cache (`--prefix-cache --checkpoint-gen`) | 10 (9) | 1.0 / 12.0 | 42.2 | 173 | 2.7 s | 1 |
| Q8_0, stock, 64k, `-b 2048 -ub 512 -ts 6,6,5,5,6,5,5,6,6` | 9 (8) | 2.9 / 12.0 | 14.4 | 144 | 3.7 s | 1 |

- **The release command line:** depth 4 (Flint-Pascal-Q5 +11 % over depth 3, Q8_0 even; the owner's rule: the depth
  that wins on both, depth 4 on a tie), `-b 512 -ub 128` (with `-b 2048 -ub 512` at 262,144 tokens the host's 4 GB of
  RAM ran out while loading, on both files), no agent prompt cache (it cost 11 s of fixed time per task and gained
  nothing).
- **Stock llama.cpp at 64k:** at 262,144 tokens its host-side attention mask does not fit the 4 GB host.
- **Pipelined speculation** is worth +52 % in generation here (Q8_0, 29.3 → 44.6 tokens/s at depth 3); it changes the
  speed, not the tokens (identity rows below).
- Commands: `bench/agent_check/release_check.sh` with `ARMS`, `CTX`, `REL_ARGS` / `STOCK_ARGS` as in the table.

Prompt reading as an agent sees it, llama-server, Q8_0, the release command line at depth 3, one request each
(2026-10-26): a fresh 13,400-token prompt reads at 485 tokens/s; the next turn (+2,000 new tokens, the rest cached)
gives its first token after 5.8 s; generation at 15,000 tokens of context runs at 37.2 tokens/s.

## The headline: Flint Q8_0 on nine cards at the full context

Unsloth's Flint Q8_0 (42.3 GB) on all nine P106-090 cards at 262,144 tokens of context, layers split by memory
(`-ts 6,6,5,5,6,5,5,6,6`).

| What | Value | How | Build | Row |
|---|---|---|---|---|
| Quality against BF16, 80 × 512 tokens of wikitext-2 test: Q8_0 / UD-Q4_K_XL | same top token 93.3 % / 87.7 %; KLD 0.052 / 0.120; PPL +3.3 % / +3.9 % | `llama-perplexity --kl-divergence` | — | Q8 |
| Five-turn chat at the full context, nine cards: Q8_0 / UD-Q4_K_XL, stock → v1 | 14.78 → **29.55** / 14.20 → 27.97 tok/s (Q8_0 1.06× UD on v1) | `scripts/server_bench.sh PROMPTS=chat`, same settings and seeds on both engines | stock, v1 | Q58 |
| Generation, Q8_0, nine cards, Pascal kernels off → on | 26.92 → 35.89 tok/s (+33 %) | `llama-bench -n 128`, one binary | v1 kernels | Q38 |
| Bytes read per generated token: Q8_0 / UD-Q4_K_XL / Flint-Pascal-Q5 (recipe K2h-r16; K2h below) | 2.70 / 2.21 / 2.09 GB | `scripts/token_cost.py` | — | — |
| v2 identity with Q8_0: pipelined output = plain, depths 1 and 3 | 34 of 34 jobs the same as plain; chat / code / edit / rewrite plain 28.8 / 28.8 / 24.0 / 27.6, depth 3 34.4 / 36.2 / 31.7 / 56.9 tok/s | `bench/pipespec/rig_run.sh` with `TS=6,6,5,5,6,5,5,6,6 DEPTHS='1 3' N=256` | rc4 `2d53179aa` | — |
| v2 generation, Q8_0, nine cards, warm: stock / v1 / release | 18.3 / 32.9 / 34.0 tok/s | `llama-bench -ngl 99 -fa on -p 512 -n 128 -r 3 -ts 6/6/5/5/6/5/5/6/6` | stock, v1, release | — |

## Measured on the release engine (2026-10-25 and 26)

| What | Value | How | Build | Row |
|---|---|---|---|---|
| Identity on the P106: pipelined output = plain; server checks (pipelined with copy lookup and continuation counts, with a floor, plain at temperature 0) | 34 of 34 jobs the same as plain (Flint-Pascal-Q5, UD-Q4_K_XL); server 13 / 13, 13 / 13, 6 / 6 | `bench/pipespec/check_identity.sh`, `rig_run.sh`, `rig_server.sh` | release | — |
| Generation, Flint-Pascal-Q5, nine cards, warm | 38.9–40.5 tok/s (two sessions) | `llama-bench -m <file> -ngl 99 -fa on -p 512 -n 128 -r 3` | release | — |
| The one-by-one test of the v2 switches on rc4 (each switch off against all on, Flint-Pascal-Q5, nine cards) | the Pascal decode-attention kernel, the input layer on the card and packed input uploads slowed generation (tg128 36.8 with them, 40.3 without) and are off by default; the attention mask built on the cards slowed long prompts 3.6× (211 against 764 tok/s at 8,192 tokens) for +1.6–2.3 % in generation and is off by default | `llama-bench` per switch (`GGML_CUDA_FATTN_PASCAL`, `LLAMA_INPUT_ON_GPU`, `LLAMA_INPUT_UPLOAD`, `LLAMA_GPU_KQ_MASK`) | rc4 → release | — |
| Prompt of 8,192 tokens in whole-prompt batches (`-b 8192 -ub 512`), nine cards: Flint-Pascal-Q5 / Q8_0 | 767 / 801 tok/s (Flint-Pascal-Q5 on stock: 593–602) | `llama-bench -p 8192 -n 0` | release, stock | — |
| Prefill at 8,192 / 32,768 tokens, the prefill-attention kernel on / off, Flint-Pascal-Q5 | 766 / 461 against 699 / 447 tok/s (+10 % / +3 %) | `GGML_CUDA_FATTN_TILE_PASCAL=1/0 llama-batched-bench` | release | Q163 |
| The agent prompt cache (`--prefix-cache --checkpoint-gen`) in the agent check, Q8_0 | no gain: 42.2 against 44.6 tok/s, 11 s of fixed time per task; off in the release command line | [bench/agent_check](../../bench/agent_check/README.md) | release | Q111 |
| Small bug-fix tasks for a coding agent, stock against the release engine | the agentic table above: 14.4 → 44.6–53.8 tok/s, 2.9 → 1.0–1.2 minutes per task | [bench/agent_check](../../bench/agent_check/README.md) | stock, release | Q131 |
| `-b 2048 -ub 512` at 262,144 tokens | does not load: the host's 4 GB of RAM runs out (Flint-Pascal-Q5, the server killed while loading); Q8_0 loses pipeline parallelism (card 1 lacks room for the 561 MiB compute buffer); `-b 512 -ub 128` is the release line | llama-server at `-c 262144` | release | Q140 |
| Tool requests sampled on the GPU (the grammar checked on the host) against sampling on the host, Flint-Pascal-Q5 | 37.9 against 33.6 tok/s at temperature 0 (2,011 tokens), 45.3–46.0 against 39.6–40.0 at temperature 1 (+13–15 %); the same tool calls, 0 grammar warnings | llama-server, `LLAMA_GRAMMAR_HOST_SAMPLING=1/0` | release | Q67 |
| Prefill with eight scheduler copies against four | 765 / 462 against 766 / 461 tok/s at 8,192 / 32,768 tokens: no gain; the release ships four | builds with `SCHED_COPIES=8` and `4` | release | Q157 |

## Engine speed (2026-10-24)

| What | Value | How | Build | Row |
|---|---|---|---|---|
| Generation, UD-Q4_K_XL, nine cards, warm: stock / v1 / v2 base | 16.5 / 32.6 / 36.5 tok/s | `llama-bench -m UD -ngl 99 -fa on -p 512,2048,8192 -n 128 -r 3`, one session | stock, v1, v2 base | Q117 |
| The same on six cards | 15.2 / 31.2 / 34.7 | the same | the same | Q117 |
| Generation alone (no prompt run before it), six cards: combined / v2 base | 30.61–30.67 / 29.86 | `llama-bench -p 0 -n 128 -r 5`, three alternating rounds | combined, v2 base | Q117 |
| Persistent MoE kernel off → on, nine cards: Flint-Pascal-Q5 / UD | 39.3 → 41.5 / 37.5 → 39.8 | `llama-bench -p 512 -n 128 -r 3`, `GGML_CUDA_PERSISTENT_MOE=0/1`, two rounds | persistent | Q88 |
| The same on six cards | 37.4 → 40.7 / 35.8 → 38.2 | the same | persistent | Q88 |
| Unified build, nine cards: Flint-Pascal-Q5 / UD | 41.8 / 39.9 | the same | v2-next | Q160 |
| Same head treatment, nine cards: UD / UD + screened head / Flint-Pascal-Q5 / K2sr | 37.5 / 38.3 / 39.3 / 39.3 | `llama-bench -p 512 -n 128 -r 3` | combined | Q125 |
| v1's steps on six cards, UD: stock / CUDA graphs / 2-warp shape / Pascal kernels | 17.6 / 19.5 / 24.5 / 34.1 | `llama-bench -n 128` | stock + patches, v1 | Q38, Q46 |
| FP32 activation sums, UD | +9.6 % | `llama-bench`, `LLAMA_NO_MOE_DOWN_RESCALE` on/off | `pascal-p106-fp32sums` | Q76 |
| Shared-expert fusion | +0.7 % | `llama-bench` | `pascal-p106-shexp` | Q50 |

## Pipelined speculation (2026-10-24)

| What | Value | How | Build | Row |
|---|---|---|---|---|
| `llama-pipespec`, nine cards, 256 tokens, greedy: plain → depth 3 / depth 4, Flint-Pascal-Q5: chat, code, edit, rewrite | 34.3 → 41.8 / 44.4; 34.7 → 44.2 / 47.0; 36.3 → 45.3 / 46.2; 39.4 → 51.7 / 52.2 | `bench/pipespec/rig_run.sh` with `DEPTHS='1 2 3 4' N=256` | v2-next | Q160 |
| The same, UD-Q4_K_XL | 32.2 → 40.4 / 42.5; 35.2 → 41.3 / 43.4; 31.8 → 44.2 / 44.6; 35.9 → 60.7 / 63.1 | the same | v2-next | Q160 |
| llama-server `--pipe-depth 3 --pipe-guess cont`, median per request, Flint-Pascal-Q5 / UD | 39.2 → 48.5 / 36.6 → 47.3 tok/s; chat 1.04–1.18×, edits 1.04–1.56×, rewrites 1.44–2.79×; ten of ten answers identical | `bench/pipespec/check_server.sh` with `ARGS='-ngl 99 -fa on -c 8192 -ub 512' DEPTHS=3 GUESSES=cont KEEP_SPEED=1` | v2-next | Q160 |
| A guesser right 70 % of the time (oracle), depth 3 | 81.9–83.9 tok/s | `bench/pipespec/rig_run.sh`, oracle guesser | `pascal-p106-v2-pipespec2` | Q87 |
| Every guess wrong, nine cards (the clock effect alone) | 1.24× plain | the same, oracle with every guess wrong | Q87's runs | Q154 |
| Idle cards at base clock against boost, the same chat prompt | 34.0 against 38.5 tok/s | clocks logged per card | Q87's runs | Q154 |

## Serving, prefill and long context (2026-10-24)

| What | Value | How | Build | Row |
|---|---|---|---|---|
| Served chat, UD, nine cards, `-bs`, 1,000 tokens: non-streamed / streamed | 31.4 / 33.3 tok/s (`-t 2`) | llama-server + client, `-c 16384` | v2 base | Q118 |
| Requests with tools against chat, UD, nine cards | 29.3–29.6 against 32.9–34.0 tok/s (12–16 % slower; the release samples them on the GPU) | llama-server with one tools request at `tool_choice` none, none with `backend_sampling: false`, and auto, 3,000 tokens | combined | Q55 |
| Prefill at 512 / 2,048 / 8,192 tokens, UD, nine cards: stock / v1 / combined | 185 / 493 / 593; 187 / 494 / 593; 228 / 613 / 723 tok/s | `llama-bench -p 512,2048,8192` | stock, v1, combined | Q117 |
| One scheduler copy against four, prefill at 8k | 132–178 against 556–695 tok/s | `llama-bench -p 8192`, `SCHED_COPIES=1` | `pascal-p106-v2` sched builds | Q89 |
| Decode at 64k tokens of context, host mask → card-built mask, Flint-Pascal-Q5 | 16.0 → 17.9 tok/s (+12 %) | `llama-batched-bench` at `-npp 65536` | `pascal-p106-v2-gpumask2` | Q116 |
| Generation at 32,768 tokens of context, v1, Miner S4-L0 | 23.3 tok/s | `llama-bench -d 32768` | v1 | Q54 |
| Eight conversations staggered through one pipeline, short prompts | 255–263 tok/s in total, all exact | `bench/pipespec/rig_hostloop.sh` | `pascal-p106-v2-pipeserve` `2650855b2` | Q155 |
| v1 five-turn chat at the full context: UD / Q8_0 / S4-L0, stock → v1 | 14.20 → 27.97 / 14.78 → 29.55 / 16.06 → 31.86 tok/s | `scripts/server_bench.sh PROMPTS=chat` | stock, v1 | Q58 |

## Quality (2026-10-23/24, quant machine)

| What | Value | How | Row |
|---|---|---|---|
| Flint-Pascal-Q5 against BF16, 80 × 512 tokens of wikitext-2 test | KLD 0.1048, same top token 88.84 %, PPL +2.65 % | `llama-perplexity --kl-divergence` ([reproduce/model.md](reproduce/model.md)) | Q113 |
| Flint-Pascal-Q5 rebuilt from the BF16 conversion with the public recipe (head copied), 2026-10-25 | 779 of 779 tensors equal byte for byte | `reproduce/build_k2h_r16.sh` | — |
| UD-Q4_K_XL / Unsloth Q8_0, the same test | KLD 0.120 / 0.052; 87.7 / 93.3 %; +3.9 / +3.3 % | the same | Q8 |
| GPTQ Q5_0 head against round-to-nearest, on F3's body | +0.0023 against +0.023 KLD | `quants/gptq_head/run.sh` | Q56 |
| K2h (32-bit router) on 3,477 single-turn tool-calling tasks (BFCL v4), against UD | +0.14 points, 90 % interval −0.87 to +1.16 (non-inferior at a 1.5-point margin) | [bench/bfcl_flint](../../bench/bfcl_flint/README.md), the makers' sampling | Q41 |
| G3 (K2h's parent) on 458 multi-turn tasks, against UD | +0.87 points, 90 % interval −3.23 to +4.98 | the same, multi-turn | Q41 |
| 8k-token agent conversations, written tokens, KLD ratio to UD | K2h 0.999 | `bench/long_context_kld/` | Q106 |
| Reasoning length against UD | level (G3/UD 1.000 on 3,477 tasks) | [answer time](../../docs/research/notes/answer_time.md) | Q123 |
| The engine changes the answers? | no: kernels change only the order of additions; v2's additions byte-identical in every check; pipelined output identical to plain | operator tests, `check_identity.sh` | Q38, Q117, Q87 |

## Hardware (2026-10-24)

| What | Value | How | Row |
|---|---|---|---|
| PowerMizer "Prefer Maximum Performance", UD, six cards: from idle / served after a minute idle | 29.75 against 29.79 / 33.25 against 33.30 tok/s (no gain) | `nvidia-settings` mode 1 against 2 | Q141 |
| Power in decode, nine cards | 35–39 W per card, 317–329 W in total, ~8.9 J per token; hottest card 57 °C | `nvidia-smi` every 200 ms | Q143 |
| PCIe 2.0 | not available: the cards advertise 2.5 GT/s | `lspci -vv` | Q86 |
| Memory clock offset | not measured; not in this release | [bench/mem_offset](../../bench/mem_offset/README.md) | Q133 |
