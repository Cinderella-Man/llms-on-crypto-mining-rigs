# Q106: long-context KLD on Flint's agent conversations (2026-10-24)

The evaluation text, full analysis tables and job scripts behind registry rows Q106, Q107 and Q112 (E1)
([experiments.md](../../../quants/experiments.md)) and the Flint results in
[bench/long_context_kld](../../../bench/long_context_kld/README.md), which also explains the method. The headline:
G3's 10 % KLD advantage over UD-Q4_K_XL at 512 tokens does not survive 8,192-token agent text (ratio 1.033), because
every quant is 3–5 % worse than UD-Q4_K_XL on the tokens the model reads (tool schemas, tool output, long-range
copies), while G3 stays 11 % better on the tokens it writes. With `eval/` anyone can score another file on exactly the
same 24 conversations without the BFCL run transcripts they were cut from. Q126 and Q127 reuse the same
text ([2026-10-24_q127](../2026-10-24_q127/README.md)).

| File | What |
|---|---|
| `eval/agent/eval.txt.gz` | the agent text: 24 conversations of exactly 8,192 tokens, joined; score with `-c 8192 --parse-special --no-escape` |
| `eval/agent/chunks.jsonl.gz` | one record per chunk: producing file (`ud` or `g3`), BFCL task id and category, prompt length, token ids |
| `eval/agent12/chunks.jsonl.gz` | the 12-chunk subset of the BF16 bias check (3 per category, 6 per producing file), with each chunk's index in the full text |
| `analysis-agent.md` | `analyze.py` on the agent text against the Q8_0 reference: UD-Q4_K_XL, G3, G3g, G3h, K2h, K2hg, K1h; by position, token class, written/read, long-range copies, the first token after `</think>` |
| `analysis-wiki8k.md`, `analysis-wiki512.md` | the same on wikitext-2 test at 8,192 (12 chunks) and at 512 tokens (80 chunks) |
| `analysis-bf16-agent12.md` | the 12-chunk subset against the BF16 reference (UD-Q4_K_XL, Unsloth's Q8_0, G3, G3g, K2h, K2hg) |
| `analysis-q8-agent12.md` | the same 12 chunks against the Q8_0 reference, for comparison |
| `q112/proxy-{o08,sh916,kv916}.md` | Q112 E1: per tensor, the H-weighted output error of G3's Q8_0, round-to-nearest Q5_0 and GPTQ Q5_0 |
| `run/common.sh` | shared definitions: Docker call, flags (`-b 2048 -ub 512 -fa on`), `score` (three references) and `score1` (one) |
| `run/reference.sh` | the three Q8_0 reference runs |
| `run/score_g3.sh` | one candidate (G3) against them, with per-token dumps; the others differ in name, file and rescale switch |
| `run/bf16_reference.sh` | the BF16 reference on the 12-chunk subset (experts of layers 21–48 in system RAM) and G3 against it |
| `run/make_q107.py` | Q107: a Q8_0 donor for the 56 global-layer attention tensors from BF16 (checked against llama-quantize's bytes), G3g and K2hg composed and checked tensor for tensor |
| `run/q112_group.sh` | Q112 E1, one group: GPTQ donor, proxy table and the split file, verified |
| `run/round11/{mix,recipes,table}.py` | the byte model behind [long_context_mix.md](../../research/notes/long_context_mix.md)'s tensor-class and ranked tables |

The per-token dumps (31 files, 27 MB) and the 29.5 GB reference logits are not here: the jobs make them again.

Made on 2026-10-24 on two RTX 3090s of the quant machine, with a CUDA build of llama.cpp `d77256999` plus
[moe-down-rescale-env.diff](../../../patches/moe-down-rescale-env.diff) and
[perplexity-parse-special-kld-dump.diff](../../../patches/perplexity-parse-special-kld-dump.diff):

```bash
D=docs/data/2026-10-24_q106     # run from the repository root
# the eval text (already here; how it was made): render the stored BFCL conversations, then cut and join them
python bench/bfcl_flint/multi_turn.py render TEMPLATE.jinja rendered.jsonl ud=e2-ud-q4kxl.jsonl g3=e2-g3.jsonl \
  --min-tokens 8192
PYTHONPATH=<llama.cpp>/gguf-py python3 bench/long_context_kld/build_eval.py rendered.jsonl flint-Q8_0.gguf \
  <build>/bin/llama-tokenize agent/
PYTHONPATH=<llama.cpp>/gguf-py python3 bench/long_context_kld/subset_eval.py agent flint-Q8_0.gguf \
  <build>/bin/llama-tokenize agent12/ --chunks 0,1,3,6,9,10,12,13,15,18,21,22
# reference logits and candidates (BIN, WORK, MODELS, GPUS: see run/common.sh)
mkdir -p "$WORK/agent"; zcat $D/eval/agent/eval.txt.gz > "$WORK/agent/eval.txt"
zcat $D/eval/agent/chunks.jsonl.gz > "$WORK/agent/chunks.jsonl"
bash $D/run/reference.sh
bash $D/run/score_g3.sh        # and: score ud "$MODELS/flint-UD-Q4_K_XL.gguf" "" (rescale on), g3h, k2h, k1h
bash $D/run/bf16_reference.sh  # after subset_eval.py into $WORK/agent12; and score1 for ud, q8, k2h, g3g, k2hg
# tables
python3 bench/long_context_kld/analyze.py --ref ud --chunks "$WORK/agent/chunks.jsonl" --model flint-Q8_0.gguf \
  ud=dump-ud-agent.tsv g3=dump-g3-agent.tsv g3g=... g3h=... k2h=... k2hg=... k1h=...
# Q107 files, Q113 routers, Q112 E1
BF16=flint-BF16.gguf G3=flint-q73-G3.gguf K2H=flint-q78k-K2h.gguf OUT=q107/ python3 $D/run/make_q107.py
python3 scripts/retype_tensors.py flint-q78k-K2h.gguf flint-q113-K2h-r16.gguf 'ffn_gate_inp=F16'
python quants/gptq_head/split_compose.py base flint-q73-G3.gguf "$OUT/flint-q112-G3-base.gguf" \
  'blk\.[0-8]\.attn_output\.weight|blk\.(9|1[0-6])\.ffn_(gate|up|down)_shexp\.weight|blk\.(9|1[0-6])\.attn_(k|v)\.weight'
for g in o08 sh916 kv916; do bash $D/run/q112_group.sh $g; done
```

The Q112 Hessians come from [act_hessian](../../../quants/gptq_head/README.md) on G3 (500 × 512 tokens of wikitext-2
train). Wikitext-2 is not included (llama.cpp's `scripts/get-wikitext-2.sh` fetches it).

**Provenance and licence of `eval/`.** The conversations are Flint's own BFCL v4 multi-turn runs (12 by UD-Q4_K_XL,
12 by G3, three per category: base, missing function, missing parameter, long context; attempt 1 of each task): the
task text, tool schemas and tool results come from the
[Berkeley Function-Calling Leaderboard v4](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
(Apache-2.0), the reasoning, calls and answers are Flint's outputs (Flint is Apache-2.0), rendered with Flint's chat
template. The tables and scripts are this project's (MIT).
