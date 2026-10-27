# Q30: BFCL v3 tool calling of six Flint files, greedy, with repeat runs

A task-quality check of the Flint files whose KL divergence against BF16 does not tell the whole story: the makers'
Q4_1 was tuned by distillation after quantizing, so its KLD overstates what it loses on real tasks. Each file answered
the same 300 BFCL v3 single-turn tasks with greedy decoding and Flint's own chat template, and five repeat runs measure
how much a rerun alone moves the result. The outcome is the registry row Q30
([quants/experiments.md](../../../quants/experiments.md)) and the quality column of
[quants/README.md](../../../quants/README.md); the harness is [bench/bfcl_flint](../../../bench/bfcl_flint/README.md)
as of commit `4d216de`, which later became the BFCL v4 harness of Q41. These are the only task-level verdicts of BF16
(it needs five GPUs to run).

## Results

| File | Size | KLD | BFCL correct | 95 % CI | simple | multiple | parallel | par. multiple | live simple | live multiple | irrelevance | live irrel. | same call as BF16 | same decision | McNemar p vs BF16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BF16 (reference) | 79.6 GB | 0 | **80.7 %** | 75.8–84.7 | 92.0 | 94.0 | 70.0 | 77.5 | 75.0 | 67.5 | 85.0 | 80.0 | (rerun: 87.0 %) | (98.3 %) | (rerun: 1.00) |
| Makers' Q4_1 | 26.0 GB | 0.322 | **83.0 %** | 78.3–86.8 | 96.0 | 94.0 | 75.0 | 82.5 | 72.5 | 75.0 | 80.0 | 80.0 | 75.7 % | 96.7 % | 0.30 |
| UD-mirror v1, rescale-free | 27.4 GB | 0.127 | **83.0 %** | 78.3–86.8 | 96.0 | 96.0 | 82.5 | 77.5 | 82.5 | 70.0 | 70.0 | 70.0 | 79.7 % | 97.0 % | 0.32 |
| S4, rescale-free | 32.6 GB | 0.120 | **82.0 %** | 77.3–85.9 | 96.0 | 92.0 | 80.0 | 80.0 | 75.0 | 65.0 | 85.0 | 75.0 | 78.7 % | 98.3 % | 0.52 |
| S1, rescale-free | 25.6 GB | 0.219 | **78.3 %** | 73.3–82.6 | 88.0 | 92.0 | 70.0 | 75.0 | 70.0 | 70.0 | 80.0 | 75.0 | 74.0 % | 96.3 % | 0.36 |
| Makers-fast (Q4_1_64, fused QKV, F16 router) | 23.9 GB | 0.315 | **83.3 %** | 78.7–87.1 | 94.0 | 96.0 | 85.0 | 77.5 | 72.5 | 75.0 | 75.0 | 80.0 | 74.3 % | 97.3 % | 0.22 |

Each file has n = 300; per category n = 50, 50, 40, 40, 40, 40, 20, 20, so one task moves a category by 2–5 points.
The KLD values are those of [quants/README.md](../../../quants/README.md). For context, terelabs' numbers on their own
300 tasks with their own checker: BF16 79.7 %, the makers' Q4_1 81.0 % (77.0 % same call as BF16), Unsloth's
UD-Q4_K_XL 80.0 % (79.7 %), all ±4.5 points.

### How much of this is noise

The same file, run twice:

| Comparison | Tasks | Byte-identical output | Same calls | Tasks whose verdict flips | Accuracy |
|---|---|---|---|---|---|
| BF16, rerun on the same GPUs | 300 | 47.3 % | 87.0 % | 19 | 80.7 % vs 81.0 % |
| UD-mirror v1, rerun on the same GPUs | 300 | 25.7 % | 85.3 % | 22 | 83.0 % vs 81.7 % |
| Makers' Q4_1 on V100s, two requests in flight vs one at a time | 300 | 45.7 % | 89.0 % | 15 | 83.3 % vs 83.7 % |
| Makers' Q4_1, RTX 3090 pair vs V100 pair | 300 | 4.7 % | 80.0 % | 29 | 83.0 % vs 83.3 % |
| S4, RTX 3090 pair vs RTX 3090 + 2 × RTX 3080 (first 182 tasks) | 182 | 11.0 % | 84.6 % | 14 | 87.9 % vs 87.9 % |
| Makers' Q4_1 vs makers-fast (same weights, repacked) | 300 | 4.3 % | 79.3 % | 23 | 83.0 % vs 83.3 % |

### What it means

- **No file is measurably different from BF16, or from any other file.** Every file scores 78.3–83.3 %, and every
  McNemar p against BF16 is at least 0.22. Two runs of one file differ by up to 1.3 points and flip 15–29 verdicts.
- **The makers' Q4_1 keeps its task quality despite the worst KLD (0.322).** Its three runs gave 83.0, 83.3 and
  83.7 %, against 80.7 and 81.0 % for BF16; terelabs saw the same direction (81.0 against 79.7 %). This supports the
  distillation explanation. Makers-fast, the same weights repacked, scores 83.3 %.
- **"Same call as BF16" cannot rank these files.** A BF16 rerun agrees with itself on 87 % of calls, the same file on
  another GPU type on 80 %, the quants with BF16 on 74–80 %. Flint reasons in a `<think>` block before every answer
  (about 160 output tokens on average, all of it reasoning; no run hit the 4,096-token limit apart from one repetition
  loop), and any change in the arithmetic (GPU type, batching, quantization) changes that text. Only 4–47 % of outputs
  are byte-identical between two runs, and in 11–21 % of tasks the call changes too. terelabs' 76–80 % fall in the
  range of a cross-GPU repeat of one file here, so their column most likely measures the same effect.
- **S1 is the lowest (78.3 %), but not significantly.** BF16 gets 25 tasks right that S1 gets wrong, S1 18 that BF16
  gets wrong (p = 0.36): a hint at most, matching S1's worst KLD of my own mixes. Between a quant and BF16, 7–14 % of
  tasks are discordant (about 11 % on average); at that rate a 3-point difference needs about 500 paired tasks to reach
  p ≈ 0.05, a 1.5-point difference about 2,000. This sized Q41.
- **The makers-fast run had the down-projection rescale on.** Its llama.cpp tree (q41g64, the tree with the `Q4_1_64`
  type) lacks the one-line patch that reads `LLAMA_NO_MOE_DOWN_RESCALE`: the variable's name is a string in the main
  build's binaries and absent from both q41g64 binaries (`strings`). So makers-fast ran here with the rescale active,
  and so did its KLD score, although both set the variable. If rescale-free output is identical, the quality numbers
  stand; a speed test of makers-fast needs the patch in that tree first.

## Method

**Tasks.** 300 single-turn tasks from BFCL v3, from the `bfcl-eval==3.0.0` wheel on PyPI, the last release that ships
the v3 data (`data/BFCL_v3_*.json` and `data/possible_answer/`, a June 2025 snapshot; wheel sha256 `5c00f3c4…`):

| Category | Tasks |
|---|---|
| simple | 50 |
| multiple | 50 |
| parallel | 40 |
| parallel_multiple | 40 |
| live_simple | 40 |
| live_multiple | 40 |
| irrelevance | 20 |
| live_irrelevance | 20 |

That is terelabs' split, inferred from their `comparison.csv`: their per-category percentages are multiples of
2 / 2 / 2.5 / 2.5 / 2.5 / 2.5 / 5 / 5, and only this split adds up to 300 and reproduces their BF16 overall of 79.7 %.
terelabs don't publish their task IDs, so the tasks are a seeded sample per category
(`random.Random(f"0-{category}").sample`), listed in `task_ids.txt`. The task file with the rendered prompts that
`prepare.py` writes has sha256 `38f7cfba75fd694b…`.

**Prompts (BFCL's function-calling mode, rendered on the client).** Each task's functions go through BFCL's own
`func_doc_language_specific_pre_processing` and `convert_to_tool(..., GORILLA_TO_OPENAPI, ModelStyle.OpenAI)`, which
is what BFCL sends an OpenAI-compatible function-calling model: JSON-schema types, `.` in function names replaced by
`_`, and " Note that the provided function is in Python 3 syntax." appended to each description. The messages and
tools are rendered with the chat template embedded in the GGUF files: the original Flint template (sha256
`7940f4ff…`), identical in all six files and, apart from the `replace("\x00", "")` filter, to terelabs'
`chat_template_llamacpp.jinja`. Rendering uses Jinja2 the way HF transformers does (`trim_blocks`, `lstrip_blocks`,
transformers' `tojson`), so the prompt is what `apply_chat_template(messages, tools=tools, add_generation_prompt=True)`
produces. No system prompt is added beyond what the task contains.

**Generation.** Each rendered prompt goes as a string to llama-server's native `/completion` endpoint:

- `temperature 0` (greedy) and `cache_prompt false`;
- `n_predict 4096`, capped at 8,184 tokens minus the prompt (this applied to one task, n_predict 2,084, and no output
  came near it);
- BOS off, per the GGUF's `add_bos_token = false`;
- server flags `-c 16384 -np 2 -ngl 99 -fa on --cache-ram 0 -ts …`: 8,192 tokens per slot, no unified KV cache;
- two requests in flight, except in the one-at-a-time control run.

Neither `--jinja` nor server-side tool parsing is used, so llama.cpp's support for Flint's tool-call syntax does not
matter. Flint opens every answer with a `<think>…</think>` block of its own accord, and the decoder reads only the text
after `</think>`.

**Decoding.** Flint writes a call as `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>`; its
template writes string arguments raw and every other type as JSON. The decoder keeps a value verbatim when the tool
schema says `string`, maps a bare `null` to None, and parses every other value as JSON, then as a Python literal,
else keeps the raw string. Calls are taken from every `<tool_call>` block after the reasoning.

**Scoring.** BFCL v3's own `ast_checker` from the same wheel, unmodified, with `underscore_to_dot` on (BFCL's setting
for OpenAI-style function-calling models). The relevance logic is copied from BFCL's `eval_runner.py`: an irrelevance
task is correct when the output decodes to no function call. The checker's heavy imports are stubbed (every API SDK
pulled in by `model_config`, and the tree-sitter Java/JS parsers); the Python categories don't use them.

`selftest.py` writes each task's ground truth through Flint's template as an assistant tool call, then decodes and
checks it: 292 of 300 pass. Six failures come from the self-test itself (in five, BFCL's lists of acceptable values
end up as nested-dict values; in one, the self-test used the ground truth's function name, which differs from the
function's definition that the checker uses). Two are dataset quirks no output can pass: `live_multiple_595-158-1`
expects the integer 2 for a string-enum parameter, and `live_simple_106-63-0` lists no acceptable value for two
required parameters.

**Statistics.** 95 % CI: Wilson interval. Same call as BF16: the decoded calls equal BF16's (function names and all
arguments, order ignored), as terelabs define it. Same decision: both call a function, or neither does. McNemar p:
exact two-sided test on the tasks where exactly one of the two runs is correct.

### Runs

| Tag | File | Binary | Env | GPUs (`-ts`) | Requests in flight |
|---|---|---|---|---|---|
| `BF16` | `flint-BF16.gguf` | main | — | RTX 3080 10 GB, RTX 3080 12 GB, 3 × RTX 3090 (10,12,24,24,24) | 2 |
| `makers-q4_1` | `flint-int4-q4_1.gguf` (RecursiveSI/flint-int4-gguf) | main | — | RTX 3090 pair (1,1) | 2 |
| `udmirror-v1-nr` | `flint-udmirror-v1-nr.gguf` | main | NR | RTX 3090 pair (1,1) | 2 |
| `mixS4` | `flint-mixS4.gguf` | main | NR | RTX 3090 pair (1,1) | 2 |
| `mixS1` | `flint-mixS1.gguf` | main | NR | RTX 3090 pair (1,1) | 2 |
| `makers-fast` | `flint-makers-fast.gguf` | q41g64 | NR (no effect, see above) | RTX 3090 pair (1,1) | 2 |
| `BF16_rep` | BF16 | main | — | as BF16 | 2 |
| `udmirror-v1-nr_rep` | UD-mirror v1 | main | NR | RTX 3090 pair | 2 |
| `makers-q4_1_v100` | makers' Q4_1 | main | — | V100 pair (1,1) | 2 |
| `makers-q4_1_v100_c1` | makers' Q4_1 | main | — | V100 pair (1,1) | 1 |
| `mixS4_gpu012_partial` | S4 | main | NR | RTX 3090 + 2 × RTX 3080 (10,12,24) | 2; stopped after 182 tasks to free the GPUs for BF16 |

- **NR:** `LLAMA_NO_MOE_DOWN_RESCALE=1`.
- **main:** llama.cpp `d77256999` plus two local changes (the `LLAMA_NO_MOE_DOWN_RESCALE` switch in `llama-graph.cpp`,
  and `LLM_ARCH_FLINT` in `load_hparams_hadamard`): a static Release build for CUDA 12.9, sm_70;86, CUDA graphs on.
- **q41g64:** the tree with the `Q4_1_64` type, llama-server built for this check (command below).
- Every server ran in `nvidia/cuda:12.9.2-devel-ubuntu24.04` with `-e CUDA_DEVICE_ORDER=PCI_BUS_ID`; the client in
  `python:3.12-slim` (Python 3.12.15, Jinja2 3.1.6, requests 2.34.2, numpy 2.5.4).
- The V100 run of the makers' file was started first and renamed to `makers-q4_1_v100` by `phase3.sh` afterwards.

## Files

| File | What |
|---|---|
| `scores/scores.md` | per-run accuracy, Wilson CI, per category, same call as BF16, McNemar p |
| `scores/scores.json` | the same with counts, both intervals, same decision, discordant tasks per side, answers at the token limit, answers with a reasoning block |
| `scores/per_task.csv` | per task and run: correct or not, BFCL's error type, and the decoded calls as JSON |
| `scores/pairs.txt` | the repeatability comparisons (the noise table) |
| `task_ids.txt` | the 300 task IDs |
| `run/serve.sh`, `run/stop_server.sh` | start a llama-server container and wait for `/health`; keep its log and remove it |
| `run/phase2.sh`, `run/phase3.sh` | the run order: BF16 on five GPUs, then every quant on the RTX 3090 pair |

Made on 2026-10-23 on the 7-GPU quant machine with the harness at `4d216de` (`git show 4d216de:bench/bfcl_flint/…`;
`task_ids.txt` is the one there). Only quality was checked; no speed was recorded.

```bash
# client container (python:3.12-slim): the harness folder mounted at /bfcl
pip install numpy jinja2 requests tqdm pyyaml tenacity overrides
pip download --no-deps bfcl-eval==3.0.0 && python -m zipfile -e bfcl_eval-3.0.0-py3-none-any.whl v3
export BFCL_ROOT=$PWD/v3
python gguf_meta.py flint-int4-q4_1.gguf > template_from_gguf.jinja
python prepare.py template_from_gguf.jinja tasks.jsonl 0      # 300 tasks, sha256 38f7cfba…
python selftest.py template_from_gguf.jinja tasks.jsonl       # 292 of 300 pass (see Method)

# llama-server for the Q4_1_64 type: the q41g64 tree, same image and flags as the main build
cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='70;86' \
  -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF && cmake --build build-cuda -j 12 --target llama-server

# per run: start a server, run the 300 tasks, stop it (phase2.sh and phase3.sh chain these)
bash serve.sh bfcl-srv-e 18085 5,6 1,1 "$MODELS/flint-mixS4.gguf" "$BIN" LLAMA_NO_MOE_DOWN_RESCALE=1
docker exec bfcl-py bash -c "cd /bfcl && python run.py http://127.0.0.1:18085 tasks.jsonl raw/mixS4.jsonl 2 4096"
bash stop_server.sh bfcl-srv-e

# scoring
python score.py tasks.jsonl raw scores BF16 BF16 makers-q4_1 udmirror-v1-nr mixS4 mixS1 makers-fast \
  BF16_rep udmirror-v1-nr_rep makers-q4_1_v100 makers-q4_1_v100_c1 mixS4_gpu012_partial
python pairs.py tasks.jsonl raw BF16:BF16_rep udmirror-v1-nr:udmirror-v1-nr_rep \
  makers-q4_1_v100:makers-q4_1_v100_c1 makers-q4_1:makers-q4_1_v100 mixS4:mixS4_gpu012_partial makers-q4_1:makers-fast
```

## Comparability with terelabs

These match terelabs: the BFCL v3 categories and per-category counts, Flint's own chat template, greedy decoding, a
4,096-token output limit, BOS off, and the same makers' Q4_1 file (`RecursiveSI/flint-int4-gguf`,
`flint-int4-q4_1.gguf`). These make the numbers not strictly comparable:

1. **Different tasks.** Their task IDs aren't published, and the BF16 per-category scores differ a lot (parallel
   70.0 % here against their 82.5 %, live_multiple 67.5 against 80.0 %, live_irrelevance 80.0 against 65.0 %, each
   from 20–40 tasks). Per-category numbers can't be compared, overall numbers only loosely (BF16 80.7 against 79.7 %).
2. **Different checker.** This check uses BFCL's official `ast_checker`; theirs is "a simplified BFCL AST checker",
   which can be more lenient or stricter (integers where a list of floats is expected, string normalisation).
3. **Tool presentation.** This check uses BFCL's OpenAI-style conversion (JSON-schema types, `.` → `_` in names, the
   Python-syntax note); theirs is not documented.
4. **Repeatability.** These runs used `-np 2` with two requests in flight. Batching then changes the arithmetic
   slightly, so runs don't repeat exactly: a BF16 rerun gives 87 % same calls and 19 flipped verdicts. terelabs used
   one slot, and their BF16 repeat matched 100 %, so their same-call column is free of batching noise; both are still
   exposed to the GPU type (80 % same calls for one file on two GPU types).
5. **Software and hardware.** llama.cpp `d77256999` with the local patch, and the q41g64 tree for makers-fast, on RTX
   3090s (BF16 on 3080/3080/3090/3090/3090). They ran PR #30928 at `1b44e30` on undisclosed GPUs.
6. **Coverage.** My own files have no terelabs counterpart. The closest is their Unsloth UD-Q4_K_XL (80.0 %), whose
   layer-by-layer allocation UD-mirror v1 re-expresses in fast formats.

The tasks come from the Berkeley Function Calling Leaderboard (BFCL v3, `bfcl-eval` 3.0.0, Gorilla project,
Apache-2.0); their text is not included, and `prepare.py` rebuilds it from the wheel. The decoded calls in
`per_task.csv` are Flint's outputs (Flint is Apache-2.0) and carry argument values taken from the BFCL tasks. The raw
outputs and server logs are not included.
