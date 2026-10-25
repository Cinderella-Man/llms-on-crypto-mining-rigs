# Every number behind the release

Each row: what was measured, the value, how (the command or public script), the engine build, the date and the
experiment's row in the [registry](../../quants/experiments.md#registry), which holds the full result, its conditions
and the raw-data note. Speed is on the reference rig (nine P106-090, PCIe 1.0 x1, 2-core Celeron, 4 GB RAM) unless a
row says otherwise; quality is scored on a separate machine against Flint's BF16 logits. tok/s = generated tokens per
second; "warm" = `llama-bench` with a 512-token prompt run before the 128 generated tokens in the same process.

Builds (fork commits): stock = upstream `d77256999`; v1 = `eeaa59660`; v2 base = `032b130f4`; combined = `b3052a4f8`;
persistent = `8de13900f`; v2-next = `62fa5fc3b`; release = rc4 `460160e93` (rc3 `7e6ec762f` without the drafter).

## Measured on the release engine (2026-10-25)

| What | Value | How | Build | Row |
|---|---|---|---|---|
| Identity on the P106: persistent-kernel self-check, logits on/off, pipelined output = plain | RIG-TBD | `GGML_CUDA_PERSISTENT_MOE_CHECK=1 llama-perplexity … -ub 1`; `bench/pipespec/check_identity.sh` | rc4 | — |
| Generation, UD-Q4_K_XL / K2h-r16, nine cards, warm | RIG-TBD / RIG-TBD | `llama-bench -m <file> -ngl 99 -fa on -p 512 -n 128 -r 3` | rc4 | — |
| The same on six cards | RIG-TBD / RIG-TBD | `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 llama-bench …` | rc4 | — |
| Prefill at 512 / 2,048 / 8,192 tokens, nine cards | RIG-TBD | `llama-bench -p 512,2048,8192 -n 0 -r 3` | rc4 | — |
| Served chat, `--pipe-depth 3`, nine and six cards | RIG-TBD | `bench/pipespec/rig_server.sh` | rc4 | — |
| Served with the drafter (`--pipe-guess staircase --pipe-draft --pipe-store`), nine and six cards, against the copy lookup and store alone and against no guessing, output identity | RIG-TBD | `bench/pipespec/rig_server.sh` with `DEPTHS='0 3s 3ds' STORE=… DRAFT=…` | rc4 | Q20 |
| Cost of one drafter guess on one P106 | RIG-TBD | `llama-pipe-drafter-check drafter.gguf --bench 300 --split 6,6,6,6,6,6,6,5,2 --ctx 2048` | rc4 | Q20 |
| Eight conversations through one pipeline | RIG-TBD | `bench/pipespec/rig_hostloop.sh` | rc4 | Q155 |
| Prefill with eight scheduler copies against four | RIG-TBD | `llama-bench -ub 512 -p 8192`, builds with `SCHED_COPIES=4` and `8` | rc4 | Q157 |
| Prefill-attention kernel on/off, K2h-r16 at 8k and 32k | RIG-TBD | `GGML_CUDA_FATTN_TILE_PASCAL=0/1 llama-batched-bench` | rc4 | Q163 |
| Agent sessions replayed with the prompt cache on/off, outputs against a cold server | RIG-TBD | `bench/agent_cache/rig_replay.sh`, `rig_check.sh` | rc4 | Q111 |
| Stock / v1 / v2 matrix at stock clocks: generation and served chat, UD-Q4_K_XL and K2h-r16 | RIG-TBD | [bench/release_matrix](../../bench/release_matrix/README.md) | stock, v1, rc4 | Q42 |
| The shipped command lines (chat, tools, `-b 2048 -ub 512` at 262k, six cards) and the quick-start check | RIG-TBD | `SERVE=1 scripts/p106_check.sh` | rc4 | Q140 |
| Bug-fix tasks of 5–10 tool steps, stock against the release engine | RIG-TBD | [bench/agent_check](../../bench/agent_check/README.md) | stock, rc4 | Q131 |

## Engine speed (2026-10-24)

| What | Value | How | Build | Row |
|---|---|---|---|---|
| Generation, UD-Q4_K_XL, nine cards, warm: stock / v1 / v2 base | 16.5 / 32.6 / 36.5 tok/s | `llama-bench -m UD -ngl 99 -fa on -p 512,2048,8192 -n 128 -r 3`, one session | stock, v1, v2 base | Q117 |
| The same on six cards | 15.2 / 31.2 / 34.7 | the same | the same | Q117 |
| Generation alone (no prompt run before it), six cards: combined / v2 base | 30.61–30.67 / 29.86 | `llama-bench -p 0 -n 128 -r 5`, three alternating rounds | combined, v2 base | Q117 |
| Persistent MoE kernel off → on, nine cards: K2h-r16 / UD | 39.3 → 41.5 / 37.5 → 39.8 | `llama-bench -p 512 -n 128 -r 3`, `GGML_CUDA_PERSISTENT_MOE=0/1`, two rounds | persistent | Q88 |
| The same on six cards | 37.4 → 40.7 / 35.8 → 38.2 | the same | persistent | Q88 |
| Unified build, nine cards: K2h-r16 / UD | 41.8 / 39.9 | the same | v2-next | Q160 |
| Same head treatment, nine cards: UD / UD + screened head / K2h-r16 / K2sr | 37.5 / 38.3 / 39.3 / 39.3 | `llama-bench -p 512 -n 128 -r 3` | combined | Q125 |
| v1's steps on six cards, UD: stock / CUDA graphs / 2-warp shape / Pascal kernels | 17.6 / 19.5 / 24.5 / 34.1 | `llama-bench -n 128` | stock + patches, v1 | Q38, Q46 |
| FP32 activation sums, UD | +9.6 % | `llama-bench`, `LLAMA_NO_MOE_DOWN_RESCALE` on/off | `pascal-p106-fp32sums` | Q76 |
| Shared-expert fusion | +0.7 % | `llama-bench` | `pascal-p106-shexp` | Q50 |

## Pipelined speculation (2026-10-24)

| What | Value | How | Build | Row |
|---|---|---|---|---|
| `llama-pipespec`, nine cards, 256 tokens, greedy: plain → depth 3 / depth 4, K2h-r16: chat, code, edit, rewrite | 34.3 → 41.8 / 44.4; 34.7 → 44.2 / 47.0; 36.3 → 45.3 / 46.2; 39.4 → 51.7 / 52.2 | `bench/pipespec/rig_run.sh` with `DEPTHS='1 2 3 4' N=256` | v2-next | Q160 |
| The same, UD-Q4_K_XL | 32.2 → 40.4 / 42.5; 35.2 → 41.3 / 43.4; 31.8 → 44.2 / 44.6; 35.9 → 60.7 / 63.1 | the same | v2-next | Q160 |
| llama-server `--pipe-depth 3 --pipe-guess cont`, median per request, K2h-r16 / UD | 39.2 → 48.5 / 36.6 → 47.3 tok/s; chat 1.04–1.18×, edits 1.04–1.56×, rewrites 1.44–2.79×; ten of ten answers identical | `bench/pipespec/check_server.sh` with `ARGS='-ngl 99 -fa on -c 8192 -ub 512' DEPTHS=3 GUESSES=cont KEEP_SPEED=1` | v2-next | Q160 |
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
| Decode at 64k tokens of context, host mask → card-built mask, K2h-r16 | 16.0 → 17.9 tok/s (+12 %) | `llama-batched-bench` at `-npp 65536` | `pascal-p106-v2-gpumask2` | Q116 |
| Generation at 32,768 tokens of context, v1, Miner S4-L0 | 23.3 tok/s | `llama-bench -d 32768` | v1 | Q54 |
| Eight conversations staggered through one pipeline, short prompts | 255–263 tok/s in total, all exact | `bench/pipespec/rig_hostloop.sh` | `pascal-p106-v2-pipeserve` `2650855b2` | Q155 |
| v1 five-turn chat at the full context: UD / Q8_0 / S4-L0, stock → v1 | 14.20 → 27.97 / 14.78 → 29.55 / 16.06 → 31.86 tok/s | `scripts/server_bench.sh PROMPTS=chat` | stock, v1 | Q58 |

## Quality (2026-10-23/24, quant machine)

| What | Value | How | Row |
|---|---|---|---|
| K2h-r16 against BF16, 80 × 512 tokens of wikitext-2 test | KLD 0.1048, same top token 88.84 %, PPL +2.65 % | `llama-perplexity --kl-divergence` ([reproduce/model.md](reproduce/model.md)) | Q113 |
| K2h-r16 rebuilt from the BF16 conversion with the public recipe (head copied), 2026-10-25 | 779 of 779 tensors equal byte for byte | `reproduce/build_k2h_r16.sh` | — |
| UD-Q4_K_XL / Unsloth Q8_0, the same test | KLD 0.120 / 0.052; 87.7 / 93.3 %; +3.9 / +3.3 % | the same | Q8 |
| GPTQ Q5_0 head against round-to-nearest, on F3's body | +0.0023 against +0.023 KLD | `quants/gptq_head/run.sh` | Q56 |
| K2h (32-bit router) on 3,477 single-turn tool-calling tasks (BFCL v4), against UD | +0.14 points, 90 % interval −0.87 to +1.16 (non-inferior at a 1.5-point margin) | [bench/bfcl_flint](../../bench/bfcl_flint/README.md), the makers' sampling | Q41 |
| G3 (K2h's parent) on 458 multi-turn tasks, against UD | +0.87 points, 90 % interval −3.23 to +4.98 | the same, multi-turn | Q41 |
| 8k-token agent conversations, written tokens, KLD ratio to UD | K2h 0.999 | `bench/long_context_kld/` | Q106 |
| Reasoning length against UD | level (G3/UD 1.000 on 3,477 tasks) | [answer time](../../docs/research/notes/answer_time.md) | Q123 |
| The engine changes the answers? | no: kernels change only the order of additions; v2's additions byte-identical in every check; pipelined output identical to plain | operator tests, `check_identity.sh` | Q38, Q117, Q87 |

## The drafter, offline (2026-10-25, quant machine)

| What | Value | How | Row |
|---|---|---|---|
| R1 (six cards), coupled acceptance on Q51 fresh text / overall | 0.502 / 0.562 | `score.py --alpha 3` ([reproduce/drafter.md](reproduce/drafter.md)) | Q20 |
| R1, simulated speed on six cards | 1.90× plain (70.9 tok/s) | `sim.py` | Q20 |
| R2-9 (the nine-card release head), nine cards with a light last card / even split | 0.476 / 0.538 fresh / overall; 0.492 / 0.553 | `score.py` on `q51-grid.bin` | Q20 |
| R2-gen (the release head for six and other counts), six cards / nine with a light last card | 0.523 / 0.583; 0.469 / 0.531 | the same | Q20 |
| R2-gen at three / twelve cards | 0.556 / 0.624; 0.449 / 0.508 | the same | Q20 |
| The C++ head against PyTorch, 14 cases | all top ids equal on a V100 | `check_export.py case` + `llama-pipe-drafter-check` | Q20 |

## Hardware (2026-10-24)

| What | Value | How | Row |
|---|---|---|---|
| PowerMizer "Prefer Maximum Performance", UD, six cards: from idle / served after a minute idle | 29.75 against 29.79 / 33.25 against 33.30 tok/s (no gain) | `nvidia-settings` mode 1 against 2 | Q141 |
| Power in decode, nine cards | 35–39 W per card, 317–329 W in total, ~8.9 J per token; hottest card 57 °C | `nvidia-smi` every 200 ms | Q143 |
| PCIe 2.0 | not available: the cards advertise 2.5 GT/s | `lspci -vv` | Q86 |
| Memory clock offset | RIG-TBD (not in this release) | [bench/mem_offset](../../bench/mem_offset/README.md) | Q133 |
