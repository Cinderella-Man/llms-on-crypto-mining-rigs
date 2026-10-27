# The v2 release engine on the rig, switch by switch (2026-10-25)

The rig day that set Miner Llama v2.0's defaults, on nine P106-090 cards: every v2 switch turned off on its own against
all on (one binary, release candidate rc4 `2d53179aa`), the slow ones off together in plain and pipelined decoding,
decode after 32,768 tokens of context, the prompt-speed drop bisected over the builds, the card-built attention mask
alternated on the release build `37e717ff6`, Flint Q8_0's pipelined identity run, tool requests sampled on the GPU
against the host (Q67), and the GPU time of one token per op group. How the defaults followed from these numbers is in
[docs/engine_features.md](../../engine_features.md#how-the-release-defaults-were-chosen); the release's own rows are in
[results.md](../../../release/flint-v2/results.md); registry rows Q117, Q150, Q151, Q152, Q110/Q116, Q157, Q163 and Q67
in the [registry](../../../quants/experiments.md#registry). These are single-request diagnostics; the release is chosen
and reported on agentic speed ([2026-10-26_release_setup](../2026-10-26_release_setup/README.md)).

| File | What |
|---|---|
| `llama_bench.tsv` | every llama-bench and llama-batched-bench result of the day, 98 rows: time, step, build, file, the switches set, the exact arguments, test, tok/s and its standard deviation, the run's own label; the steps are listed below |
| `pipespec_q8_9c.tsv` | Flint Q8_0 on nine cards (`-ts 6,6,5,5,6,5,5,6,6`), rc4 `2d53179aa`: [bench/pipespec/rig_run.sh](../../../bench/pipespec/README.md)'s 34 jobs (plain, depth 0, the guessers at depths 1 and 3, oracle ceilings), 256 tokens each: tok/s, speed-up over plain, guesses right / wrong / dropped, and whether the output equals plain decoding (34 of 34) |
| `pipespec_combo_k2h.tsv` | the same 34 jobs with Flint-Pascal-Q5 on rc4, all switches on against decode attention, the input layer on the card and packed uploads off: tok/s per job and identity (34 of 34 both ways) |
| `tool_request_q67.tsv` | Q67: one coding-agent request sent four ways (`tool_choice` none / auto, `backend_sampling` true / false) at temperature 0 and at 1.0 with seeds 11 and 12, Flint-Pascal-Q5 on the release build; one run with tool requests sampled on the GPU (the default), one with `LLAMA_GRAMMAR_HOST_SAMPLING=1`: finish reason, tool calls, tokens, tok/s, tokens sampled on the card, grammar tokens and rejections |
| `op_profile_rc3.tsv` | GPU time of one generated token on rc3 `7e6ec762f`, Flint-Pascal-Q5, nine cards, the attention block in two launches (default) against off: the sum over cards, each card, the wall time from one token to the next on card 0, and the 15 largest op groups (ms per token, share, launches per token, µs per launch, ops, tensors) |
| `run/ablation.sh`, `run/combo.sh`, `run/mask_ab.sh`, `run/prefill_bisect.sh`, `run/prefill_big.sh` | the scripts behind the steps below |

## What it shows

| Question | Result (tok/s) |
|---|---|
| Each switch off, Flint-Pascal-Q5 tg128 (all on: 36.80, 36.80, 37.33 at the start, middle and end) | Pascal matrix-vector kernels off 31.85 (they are worth +15.5 %), persistent MoE kernel off 34.03 (+8 %), Pascal decode attention off **40.25**, attention block in one launch 36.51, packed uploads off 37.41, input layer on the host 37.75, card-built mask off 36.68; `GGML_CUDA_NORM_Q8=0` aborts llama-bench after pp512 with an assertion in the CUDA memory pool |
| Pascal Q4_K / Q6_K kernels, UD-Q4_K_XL | 39.53 on, 39.86 skipped (`GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K`): within the noise |
| Decode after 32,768 tokens of context | rc4 with every switch on 15.04; decode attention off 25.41; the release build 25.09 (25.67 with the card-built mask on) |
| The three slow switches off together | tg128 36.84 → 39.42–40.26 (v2-next `62fa5fc3b` 42.04); pipelined depth 3 chat / code / edit / rewrite 41.5 / 42.8 / 46.1 / 70.0 → 45.6 / 48.4 / 51.6 / 106.5, 34 of 34 identical; Q8_0 31.05 → 33.67 |
| One 8,192-token prompt in one batch (`-b 8192 -ub 512`) per build | stock 602, combined `b3052a4f8` 721, v2-next `62fa5fc3b` 720, `d379da309` 738, **rc1 `594d20e08` 188**; `bff28aba6` 211, with `LLAMA_GPU_KQ_MASK=0` 764: the card-built mask, built for every micro-batch, came in with rc1 |
| The release build `37e717ff6`, no switches | Flint-Pascal-Q5 pp8192 766.9, tg128 38.87; Q8_0 pp8192 800.8, tg128 33.25 |
| Card-built mask on the release build, alternated | tg128 off 40.46 / 40.27, on 41.29 / 40.77; Q8_0 off 34.02, on 34.21 |
| Whole-prompt batches, 8k / 32k | 766.3 / 461.0; prefill attention off 699.3 / 446.8; 8 scheduler copies 765.1 / 462.1 |
| Tool requests, temperature 0, 2,011 tokens | sampled on the GPU 37.93, on the host 33.62; at temperature 1.0 45.28–46.02 against 39.62–39.95; the same tool calls |
| One token on rc3, nine cards | 26.43 ms of GPU time summed over the cards, 32.03 ms wall; the MoE layers' fused group is the largest (4.94 ms) |

The `build` column gives the fork commit and its branch: `d77256999` stock llama.cpp; `eeaa59660` v1; `b3052a4f8` the
combined test build; `62fa5fc3b` and `d379da309` v2-next; `594d20e08` rc1; `7e6ec762f` rc3; `2d53179aa` rc4 with every
switch on; `bff28aba6` rc4 with Pascal decode attention, the input layer on the card and packed uploads off by default;
`37e717ff6` the release (also the card-built mask off by default). `-sched8` marks the same commit built with eight
scheduler copies. "Flint-Pascal-Q5" is the file built with recipe K2h-r16 (the labels' "K2h"); its tensors are those of
the released file.

The `step` column:

| Step | Script and build |
|---|---|
| `release-matrix` | [bench/release_matrix](../../../bench/release_matrix/README.md) at stock clocks, Q8_0: stock, v1 and rc4 `2d53179aa` |
| `one-switch-off`, `decode-32k` | `run/ablation.sh` steps `decode` and `long`, rc4 `2d53179aa` |
| `combined-off` | `run/combo.sh` steps `k2h-bench` and `q8-bench`, rc4 `2d53179aa` (the Q8_0 run on v1 was stopped before a result) |
| `defaults` | llama-bench with no switches on `bff28aba6` and on the release build `37e717ff6` |
| `kquant` | `run/ablation.sh` step `kquant` on `bff28aba6` |
| `prefill-attention` | llama-batched-bench at 8,192 and 32,768 prompt tokens, Q163's kernel on against off, on `bff28aba6` and its 8-copy build |
| `prefill-whole-batch` | `run/prefill_big.sh` on `bff28aba6` (one row) and on the release build; its Q8_0 runs were stopped before a result |
| `prefill-bisect` | `run/prefill_bisect.sh` (stopped after rc1), then the release candidate `bff28aba6` with `LLAMA_GPU_KQ_MASK=0` |
| `card-mask` | `run/mask_ab.sh`, the release build |

Made on 2026-10-25 on the rig, nothing else running on it. Each script writes its Markdown log to `OUT`; llama-bench's
own table rows are kept as they print and turned into `llama_bench.tsv`:

```bash
export RIG=rig OUT=out    # RIG: your ssh alias for the rig
STEPS='decode long' bash docs/data/2026-10-25_release_rig/run/ablation.sh
STEPS=kquant T=llama.cpp-20261025-bff28aba6-pascal-p106-v2-rc4 bash docs/data/2026-10-25_release_rig/run/ablation.sh
STEPS='k2h-bench k2h-pipe q8-bench' bash docs/data/2026-10-25_release_rig/run/combo.sh
bash docs/data/2026-10-25_release_rig/run/prefill_bisect.sh
bash docs/data/2026-10-25_release_rig/run/prefill_big.sh
bash docs/data/2026-10-25_release_rig/run/mask_ab.sh

# defaults: the release build with no switches (Q8_0 adds -ts 6/6/5/5/6/5/5/6/6)
ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 /opt/llama-builds/llama.cpp-20261025-37e717ff6-pascal-p106-v2-rc4/bin/llama-bench \
    -m <gguf> -ngl 99 -fa on -p 512,8192 -n 128 -b 8192 -ub 512 -r 2 -o md"

# prefill-attention: Q163's kernel on, then off with GGML_CUDA_FATTN_TILE_PASCAL=0; at 32,768 tokens -c 33280 -npp 32768
ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 <bin>/llama-batched-bench -m <Flint-Pascal-Q5> -ngl 99 -fa on \
    -c 8704 -b 2048 -ub 512 -npp 8192 -ntg 32 -npl 1"

# pipespec_q8_9c.tsv (summary.tsv of the run), rc4 2d53179aa
scp -r bench/pipespec "$RIG":/tmp/
ssh "$RIG" "BIN=/opt/llama-builds/llama.cpp-20261025-2d53179aa-pascal-p106-v2-rc4/bin MODEL=<flint-Q8_0.gguf> TAG=q8-9c \
    GPUS=0,1,2,3,4,5,6,7,8 TS=6,6,5,5,6,5,5,6,6 DEPTHS='1 3' N=256 GPU_LOG=0 EXTRA_ARGS='-lv 4' bash /tmp/pipespec/rig_run.sh"

# tool_request_q67.tsv (results.jsonl of both runs), the release build
B=/opt/llama-builds/llama.cpp-20261025-37e717ff6-pascal-p106-v2-rc4/bin
ssh "$RIG" "BIN=$B MODEL=<Flint-Pascal-Q5> GPUS=0,1,2,3,4,5,6,7,8 TAG=q67-k2h-9 bash -s" < bench/tool_sampling/tool_request_ab.sh
ssh "$RIG" "BIN=$B MODEL=<Flint-Pascal-Q5> GPUS=0,1,2,3,4,5,6,7,8 TAG=q67-k2h-9-host LLAMA_GRAMMAR_HOST_SAMPLING=1 bash -s" \
    < bench/tool_sampling/tool_request_ab.sh

# op_profile_rc3.tsv: rc3 7e6ec762f, once as is and once with GGML_CUDA_ATTN_BLOCK=0
scp scripts/cuda_profile_summary.py "$RIG":/tmp/
ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 GGML_CUDA_PROFILE=/tmp/prof.csv <bin>/llama-bench -m <Flint-Pascal-Q5> \
    -ngl 99 -fa on -p 0 -n 32 -r 1 > /dev/null; python3 /tmp/cuda_profile_summary.py /tmp/prof.csv --tokens 24 --by op"
```

The scripts take the build from `T` or `BIN` and the files from `K2H`, `Q8` and `UD`; the defaults are the paths used
here. The release candidates' tarballs are not published; the release itself is the fork's tag `miner-llama-v2.0`.

All numbers are this project's own measurements (MIT, like the repository); the pipespec prompts are the repository's
own (`bench/pipespec/prompts/`).
