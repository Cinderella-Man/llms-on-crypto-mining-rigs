# Q126 and Q127: K2-r16 and the swap probes on agent text at 8k (2026-10-24)

The full analysis tables behind registry rows Q126 and Q127 ([experiments.md](../../../quants/experiments.md)) and
their sections in [bench/long_context_kld](../../../bench/long_context_kld/README.md), plus the scripts that built the
probe files and ran them. K2-r16 (K2 with its Q8_0 head and F16 routers) writes 7 % better than UD-Q4_K_XL on
Flint's agent conversations and reads tool schemas 7 % worse; no probe (G3 with UD-Q4_K_XL's attention, with its
routed experts, or with Q8_0 for the 19 sliding-layer attention tensors UD-Q4_K_XL keeps at Q8_0) moves the schema
tokens toward UD-Q4_K_XL, so that gap is not attributable to a tensor class. All files ran on the same two V100s
against Q106's Q8_0 reference on Q106's 24 conversations ([2026-10-24_q106](../2026-10-24_q106/README.md), `eval/`).

| File | What |
|---|---|
| `analysis-q126.md` | Q126, 24 chunks: UD-Q4_K_XL, K2-r16 and K2h on the V100s, UD-Q4_K_XL and K2h from the RTX 3090 run; token-pooled ratios (written, read, schemas, tool output, user) with chunk and cluster intervals, paired differences K2-r16 − K2h and V100 − 3090. K2h is NaN on chunk 6 with the rescale off |
| `analysis-q126-no6.md` | the same on the 23 chunks without chunk 6 (`multi_turn_long_context_10`) |
| `analysis-q126-pre.md` | the same before K2h's V100 run, on 24 chunks |
| `analysis-q127-24.md` | Q127, 24 chunks: G3, SW-X (UD-Q4_K_XL's routed experts), G3a and K2-r16 on the V100s, paired to G3 |
| `analysis-q127-no6.md` | the same on 23 chunks, with SW-A (UD-Q4_K_XL's attention), which is NaN on chunk 6 with the rescale off |
| `analysis-q127-swa-no6.md` | SW-A and G3 on the V100s with G3's RTX 3090 run, 23 chunks |
| `analysis-q127-g3r.md` | G3 with the MoE down-projection rescale on against off: bit-identical dumps on all 24 chunks |
| `round11-check.md` | `analyze.py --pooled` on Q106's RTX 3090 dumps, reproducing the round-11 tables of [long_context_mix.md](../../research/notes/long_context_mix.md) |
| `run/common.sh` | shared definitions: the agent-text run with `LLAMA_KLD_DUMP` |
| `run/q126_agent_v100.sh` | Q126's three runs; Q127's probes are the same call |
| `run/make_q127.py` | builds SW-A, SW-X and G3a from G3 (G3a's Q8_0 donor from BF16, checked against llama-quantize's bytes) and checks every tensor |
| `run/mag.py` | the NaN follow-up: per layer, the largest down-projection input, rows past 1,024 and 65,504, and 32-value block sums past FP16's range, from act_hessian's dumped rows |

Made on 2026-10-24 with an sm_70 CUDA build of the Q106 tree (llama.cpp `d77256999` with
[moe-down-rescale-env.diff](../../../patches/moe-down-rescale-env.diff) and
[perplexity-parse-special-kld-dump.diff](../../../patches/perplexity-parse-special-kld-dump.diff)), run from the
repository root:

```bash
D=docs/data/2026-10-24_q127/run
python3 scripts/retype_tensors.py flint-q78k-K2.gguf flint-q126-K2-r16.gguf 'ffn_gate_inp=F16'
BF16=flint-BF16.gguf G3=flint-q73-G3.gguf UD=flint-UD-Q4_K_XL.gguf OUT=q127/ python3 $D/make_q127.py
bash $D/q126_agent_v100.sh                       # and the probes, see its header
# tables (W = Q106's work folder with the dumps; --only-chunks <list without 6> for the -no6 tables)
python3 bench/long_context_kld/analyze.py --ref ud-v100 --chunks $W/agent/chunks.jsonl --model flint-q73-G3.gguf \
  --pooled --diff k2r16-v100-k2h-v100,ud-ud-v100,k2h-k2h-v100 \
  ud-v100=$W/dump-ud-v100-agent.tsv k2r16-v100=$W/dump-k2r16-v100-agent.tsv k2h-v100=$W/dump-k2h-v100-agent.tsv \
  ud=$W/dump-ud-agent.tsv k2h=$W/dump-k2h-agent.tsv
python3 bench/long_context_kld/analyze.py --ref ud-v100 --chunks $W/agent/chunks.jsonl --model flint-q73-G3.gguf \
  --pooled --diff swa-v100-g3-v100,swx-v100-g3-v100,g3a-v100-g3-v100,g3-g3-v100 \
  ud-v100=... g3-v100=... swa-v100=... swx-v100=... g3a-v100=... k2r16-v100=... g3=$W/dump-g3-agent.tsv
# NaN follow-up: K2h's down-projection inputs in FP32 on the CPU over chunk 6, dumped by act_hessian
#   (HESS_DUMP=ffn_moe_swiglu-L,ffn_swiglu-L,...), then
python3 $D/mag.py <dump dir>
```

The magnitude table covers the first 2,048 positions of chunk 6 (the run was stopped there); the
[release_v2.md](../../research/notes/release_v2.md) §1.3 gives the conclusion. The tables and scripts are this
project's (MIT); the conversations they score are described in Q106's README.
