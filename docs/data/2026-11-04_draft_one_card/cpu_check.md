# CPU check of the two switches (2026-11-04)

`run/cpu_check.sh` on the build machine (Docker, `--cpu-shares 256`, CPU backend, 8 threads), the fork's
`8e8706e8e`, the same code as the pushed `648c720c9`. Build and unit test, then one load per arm of the real files on the CPU:
Swift's GSQ-RCO Q2_0 of Qwen3.8-Flash-Next and the Gyro MTP draft, n-max 3, `-c 4096 -np 1`. Each arm gets the same
three requests: P1 and P2 greedy (96 and 64 tokens; Q564's two prompts) and P1 sampled (temperature 1.0, top-k 20,
top-p 0.95, seed 42, 64 tokens).

## Verdicts

| Check | Verdict | Detail |
|---|---|---|
| build | OK | `llama-server` and `test-draft-mem`, CPU backend, static |
| unit | OK | `test-draft-mem` 10,885 checks, 0 failed: the cache's sizes and refusals, the replacement order (never a slot the same batch reads), and over 3,000 random batches per type the cache's rows hold the table's bytes and `get_rows` over them equals the table's (F32, Q8_0, Q3_K: 0 bad rows, 0 bad maps, 0 bad `get_rows` of 293 compared each) |
| files-identity | OK | 15 of 15 replies of the switch arms carry the tokens and draft counts of the arm without the switches: `cache`, `cache-small`, `carry` and `both` against `base`, `short-both` against `short`, on all three requests (the sampled one too) |
| files-log | OK | the 4 cache arms logged the cache's counts, the 3 carry arms the carry |
| files-plain | INFO | no drafting arm's greedy reply equals plain decoding's, `base` (no switch) included: on this build the verify batch diverges from single-token decodes at token 20 of 96 (P1) and 24 of 57 (P2), Q971 |

## Per arm ([cpu_check.tsv](cpu_check.tsv))

| Arm | Switches | P1: tokens, drafted, accepted | P2 | P1 sampled | Token ids (md5 of P1, P2, P1 sampled) |
|---|---|---|---|---|---|
| nodraft | (plain decoding) | 96, –, – | 57, –, – | 64, –, – | ded64f3a, b1143893, 6518ee18 |
| base | (the MTP draft) | 96, 72, 70 | 53, 60, 32 | 64, 58, 43 | 9c46aa2f, ce8b1b97, 6518ee18 |
| cache | `LLAMA_DRAFT_EMBD_CACHE=8192` | 96, 72, 70 | 53, 60, 32 | 64, 58, 43 | the same as base |
| cache-small | `LLAMA_DRAFT_EMBD_CACHE=16,4` | 96, 72, 70 | 53, 60, 32 | 64, 58, 43 | the same as base |
| carry | `LLAMA_DRAFT_H_CARRY=1` | 96, 72, 70 | 53, 60, 32 | 64, 58, 43 | the same as base |
| both | the cache and the carry | 96, 72, 70 | 53, 60, 32 | 64, 58, 43 | the same as base |
| short | `LLAMA_DRAFT_SHORTLIST=32768,8192,2` | 96, 78, 68 | 53, 63, 31 | 64, 58, 43 | the same as base |
| short-both | the shortlist, the cache and the carry | 96, 78, 68 | 53, 63, 31 | 64, 58, 43 | the same as base (and short) |

- The cache's counts at the end of a load: 461 token rows read from it, 117 copied from the table first (482 and 122
  with the shortlist); `cache-small` (16 rows, batches of up to 4 tokens) copied 206. In all of them the 3 prompt
  batches (37-46 tokens) read the table. Three short requests start from an empty cache. An agent session reuses its
  few thousand tokens.
- The sampled request gives the same tokens in every arm, plain decoding included. What shows the drafts there is
  their counts: 58 drafted and 43 accepted in every drafting arm.
