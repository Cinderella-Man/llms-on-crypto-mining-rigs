# <OUT>/st-w4h2.jsonl

- lines: 792 (19 without a draft, 773 generation steps); drafted 2313, accepted 1547; drafting passes beyond the drafted tokens 1 (in 1 steps); line writes: median 259 us
- typical steps: 423 of 773 (the commonest of 15 mark sequences among the steps that drafted 3 tokens, 3 drafting passes); the first 3 after each prompt left out

## Waterfall (ms between consecutive marks; the rows add up to the line)

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd → s.post | 0.007 | 0.007 | 0.005 | 0.007 |  |
| s.post → s.shift | 0.001 | 0.001 | 0.001 | 0.001 |  |
| s.shift → s.sel | 0.016 | 0.018 | 0.004 | 0.027 |  |
| s.sel → s.drf | 0.000 | 0.000 | 0.000 | 0.000 |  |
| s.drf → s.drf.in | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.drf.in → mtp.drf | 0.001 | 0.001 | 0.000 | 0.001 |  |
| mtp.drf → mtp.dp | 0.007 | 0.007 | 0.004 | 0.011 |  |
| mtp.dp → d.dec | 0.011 | 0.012 | 0.010 | 0.014 |  |
| d.dec → d.dec.bal | 0.017 | 0.017 | 0.016 | 0.019 |  |
| d.dec.bal → d.dec.mem | 0.040 | 0.041 | 0.037 | 0.044 |  |
| d.dec.mem → d.dec.out | 0.004 | 0.004 | 0.003 | 0.004 |  |
| d.dec.out → d.ub.apply | 0.369 | 0.443 | 0.064 | 0.778 |  |
| d.ub.apply → d.ub.build | 0.423 | 0.426 | 0.409 | 0.444 |  |
| d.ub.build → d.ub.in | 0.031 | 0.031 | 0.021 | 0.039 |  |
| d.ub.in → d.ub.comp | 0.555 | 0.567 | 0.546 | 0.572 |  |
| d.ub.comp → d.ub.cp | 0.022 | 0.022 | 0.021 | 0.023 |  |
| d.ub.cp → d.dec.end | 0.002 | 0.002 | 0.001 | 0.002 |  |
| d.dec.end → mtp.dp.ret | 0.005 | 0.005 | 0.004 | 0.005 |  |
| mtp.dp.ret → d.sync(6) | 0.059 | 0.119 | 0.057 | 0.254 |  |
| d.sync(6) → d.sync.end(6) | 2.878 | 2.863 | 2.477 | 2.947 |  |
| d.sync.end(6) → mtp.dp.smp | 0.006 | 0.007 | 0.005 | 0.007 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp → d.sync | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync → d.sync.end | 0.015 | 0.015 | 0.015 | 0.016 |  |
| d.sync.end → mtp.dp.row | 0.005 | 0.005 | 0.004 | 0.006 |  |
| mtp.dp.row → mtp.dp#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.dp#2 → d.dec#2 | 0.011 | 0.012 | 0.010 | 0.014 |  |
| d.dec#2 → d.dec.bal#2 | 0.017 | 0.018 | 0.016 | 0.019 |  |
| d.dec.bal#2 → d.dec.mem#2 | 0.045 | 0.045 | 0.041 | 0.049 |  |
| d.dec.mem#2 → d.dec.out#2 | 0.003 | 0.003 | 0.003 | 0.004 |  |
| d.dec.out#2 → d.ub.apply#2 | 0.063 | 0.067 | 0.047 | 0.080 |  |
| d.ub.apply#2 → d.ub.reuse | 0.045 | 0.056 | 0.042 | 0.055 |  |
| d.ub.reuse → d.ub.in#2 | 0.027 | 0.028 | 0.018 | 0.036 |  |
| d.ub.in#2 → d.ub.comp#2 | 0.479 | 0.490 | 0.465 | 0.502 |  |
| d.ub.comp#2 → d.ub.cp#2 | 0.019 | 0.019 | 0.018 | 0.020 |  |
| d.ub.cp#2 → d.dec.end#2 | 0.001 | 0.001 | 0.001 | 0.002 |  |
| d.dec.end#2 → mtp.dp.ret#2 | 0.005 | 0.005 | 0.004 | 0.006 |  |
| mtp.dp.ret#2 → d.sync(6)#2 | 0.057 | 0.111 | 0.053 | 0.244 |  |
| d.sync(6)#2 → d.sync.end(6)#2 | 2.919 | 2.859 | 2.453 | 2.976 |  |
| d.sync.end(6)#2 → mtp.dp.smp#2 | 0.004 | 0.004 | 0.003 | 0.005 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp#2 → d.sync#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync#2 → d.sync.end#2 | 0.015 | 0.015 | 0.014 | 0.015 |  |
| d.sync.end#2 → mtp.dp.row#2 | 0.004 | 0.004 | 0.003 | 0.006 |  |
| mtp.dp.row#2 → mtp.dp#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.dp#3 → d.dec#3 | 0.009 | 0.009 | 0.008 | 0.011 |  |
| d.dec#3 → d.dec.bal#3 | 0.015 | 0.015 | 0.013 | 0.017 |  |
| d.dec.bal#3 → d.dec.mem#3 | 0.040 | 0.041 | 0.036 | 0.045 |  |
| d.dec.mem#3 → d.dec.out#3 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.dec.out#3 → d.ub.apply#3 | 0.059 | 0.062 | 0.040 | 0.078 |  |
| d.ub.apply#3 → d.ub.reuse#2 | 0.039 | 0.040 | 0.035 | 0.046 |  |
| d.ub.reuse#2 → d.ub.in#3 | 0.026 | 0.026 | 0.017 | 0.034 |  |
| d.ub.in#3 → d.ub.comp#3 | 0.466 | 0.471 | 0.452 | 0.493 |  |
| d.ub.comp#3 → d.ub.cp#3 | 0.015 | 0.015 | 0.014 | 0.016 |  |
| d.ub.cp#3 → d.dec.end#3 | 0.001 | 0.001 | 0.001 | 0.001 |  |
| d.dec.end#3 → mtp.dp.ret#3 | 0.005 | 0.005 | 0.004 | 0.006 |  |
| mtp.dp.ret#3 → s.vp | 0.006 | 0.007 | 0.006 | 0.007 |  |
| s.vp → p.dec | 0.005 | 0.005 | 0.004 | 0.006 |  |
| p.dec → p.dec.bal | 0.014 | 0.015 | 0.013 | 0.016 |  |
| p.dec.bal → p.dec.mem | 0.040 | 0.040 | 0.036 | 0.043 |  |
| p.dec.mem → p.dec.out | 0.008 | 0.008 | 0.007 | 0.008 |  |
| p.dec.out → p.ub.apply | 0.458 | 0.623 | 0.064 | 1.115 |  |
| p.ub.apply → p.ub.reuse | 0.064 | 0.065 | 0.052 | 0.076 |  |
| p.ub.reuse → p.ub.in | 0.023 | 0.025 | 0.012 | 0.034 |  |
| p.ub.in → p.ub.up | 0.138 | 0.139 | 0.131 | 0.148 |  |
| p.ub.up → p.dec.prep | 0.000 | 0.000 | 0.000 | 0.001 |  |
| p.dec.prep → s.vp.end | 0.003 | 0.003 | 0.002 | 0.004 |  |
| s.vp.end → d.sync(6)#3 | 0.058 | 0.110 | 0.055 | 0.250 |  |
| d.sync(6)#3 → d.sync.end(6)#3 | 1.859 | 1.749 | 1.237 | 2.056 |  |
| d.sync.end(6)#3 → mtp.dp.smp#3 | 0.005 | 0.005 | 0.004 | 0.006 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp#3 → d.sync#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync#3 → d.sync.end#3 | 0.015 | 0.015 | 0.014 | 0.015 |  |
| d.sync.end#3 → mtp.dp.row#3 | 0.004 | 0.004 | 0.003 | 0.004 |  |
| mtp.dp.row#3 → mtp.drf.end | 0.048 | 0.090 | 0.046 | 0.210 | the drafter's end (shortlist and carry reset, length check) |
| mtp.drf.end → s.drf.out | 0.001 | 0.001 | 0.001 | 0.002 | back in the server |
| s.drf.out → s.drf.end | 0.000 | 0.000 | 0.000 | 0.000 | the yield's end: the queue's worker thread stops |
| s.drf.end → s.ckpt | 0.017 | 0.017 | 0.012 | 0.019 | draft bookkeeping, the draft context's seq_rm, checkpoints |
| s.ckpt → s.ckgen | 0.000 | 0.000 | 0.000 | 0.001 | the generation checkpoint check |
| s.ckgen → s.bat | 0.002 | 0.002 | 0.001 | 0.002 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.002 | 0.002 | 0.001 | 0.002 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.006 | 0.006 | 0.005 | 0.007 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.001 | 0.001 | 0.000 | 0.001 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.001 | into decode() |
| s.dec → s.dec.y | 0.000 | 0.000 | 0.000 | 0.001 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.001 | 0.001 | 0.001 | 0.002 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.008 | 0.008 | 0.007 | 0.010 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.001 | 0.001 | 0.000 | 0.001 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.001 | 0.001 | 0.000 | 0.001 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.002 | 0.002 | 0.001 | 0.003 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.in | 0.147 | 0.152 | 0.143 | 0.164 |  |
| v.ub.in → v.ub.comp | 3.788 | 4.141 | 3.675 | 4.680 |  |
| v.ub.comp → v.ub.cp | 0.050 | 0.054 | 0.048 | 0.063 |  |
| v.ub.cp → v.dec.end | 0.004 | 0.004 | 0.003 | 0.005 |  |
| v.dec.end → s.dec.ret | 0.030 | 0.030 | 0.027 | 0.032 |  |
| s.dec.ret → v.sync | 0.001 | 0.001 | 0.000 | 0.001 |  |
| v.sync → v.sync.end | 94.868 | 93.004 | 83.537 | 97.911 |  |
| v.sync.end → s.dec.out | 0.001 | 0.001 | 0.001 | 0.002 |  |
| s.dec.out → s.dec.end | 0.003 | 0.003 | 0.002 | 0.003 |  |
| s.dec.end → s.mpd | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.mpd → s.cu.in | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.cu.in → mtp.cu | 0.002 | 0.002 | 0.001 | 0.002 |  |
| mtp.cu → mtp.cu.dec | 0.259 | 0.260 | 0.255 | 0.265 |  |
| mtp.cu.dec → d.dec#4 | 0.006 | 0.006 | 0.006 | 0.007 |  |
| d.dec#4 → d.dec.bal#4 | 0.019 | 0.019 | 0.018 | 0.020 |  |
| d.dec.bal#4 → d.dec.mem#4 | 0.068 | 0.068 | 0.061 | 0.073 |  |
| d.dec.mem#4 → d.dec.out#4 | 0.008 | 0.008 | 0.007 | 0.008 |  |
| d.dec.out#4 → d.ub.apply#4 | 0.424 | 0.511 | 0.075 | 0.862 |  |
| d.ub.apply#4 → d.ub.reuse#3 | 0.078 | 0.080 | 0.067 | 0.091 |  |
| d.ub.reuse#3 → d.ub.in#4 | 0.026 | 0.027 | 0.016 | 0.035 |  |
| d.ub.in#4 → d.ub.comp#4 | 0.204 | 0.207 | 0.196 | 0.224 |  |
| d.ub.comp#4 → d.ub.cp#4 | 0.002 | 0.002 | 0.001 | 0.002 |  |
| d.ub.cp#4 → d.dec.end#4 | 0.002 | 0.002 | 0.002 | 0.003 |  |
| d.dec.end#4 → mtp.cu.ret | 0.005 | 0.006 | 0.005 | 0.006 |  |
| mtp.cu.ret → mtp.cu.end | 0.259 | 0.257 | 0.238 | 0.264 |  |
| mtp.cu.end → s.cu.out | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.cu.out → s.cu.end | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.cu.end → s.pd | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.pd → s.acc | 0.002 | 0.002 | 0.002 | 0.003 |  |
| s.acc → s.acc.cl | 0.015 | 0.017 | 0.013 | 0.022 |  |
| s.acc.cl → v.sync(24) | 0.002 | 0.002 | 0.001 | 0.002 |  |
| v.sync(24) → v.sync.end(24) | 0.386 | 0.390 | 0.372 | 0.417 |  |
| v.sync.end(24) → s.acc.smp | 0.001 | 0.001 | 0.001 | 0.001 |  |
| s.acc.smp → s.acc.spec | 0.002 | 0.002 | 0.001 | 0.002 |  |
| s.acc.spec → s.rm | 0.018 | 0.018 | 0.014 | 0.020 |  |
| s.rm → s.tok | 0.006 | 0.006 | 0.005 | 0.007 |  |
| s.tok → s.tok#2 | 0.001 | 0.001 | 0.001 | 0.002 |  |
| s.tok#2 → s.tok#3 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.tok#3 → s.tok#4 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.tok#4 → s.tg | 0.000 | 0.001 | 0.000 | 0.001 |  |
| s.tg → s.upd.end | 0.001 | 0.001 | 0.001 | 0.002 |  |
| **the line (s.upd → s.upd.end)** | 112.771 | 111.270 | 100.115 | 117.286 |  |

## The gap: the last drafting pass's wait → the verify pass's inputs set

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| draft card's work end → d.sync.end (last pass) | 0.194 | 0.195 | 0.188 | 0.203 | the drafted id's way back, the host's wake |
| d.sync.end#3 → mtp.dp.row#3 | 0.004 | 0.004 | 0.003 | 0.004 |  |
| mtp.dp.row#3 → mtp.drf.end | 0.048 | 0.090 | 0.046 | 0.210 | the drafter's end (shortlist and carry reset, length check) |
| mtp.drf.end → s.drf.out | 0.001 | 0.001 | 0.001 | 0.002 | back in the server |
| s.drf.out → s.drf.end | 0.000 | 0.000 | 0.000 | 0.000 | the yield's end: the queue's worker thread stops |
| s.drf.end → s.ckpt | 0.017 | 0.017 | 0.012 | 0.019 | draft bookkeeping, the draft context's seq_rm, checkpoints |
| s.ckpt → s.ckgen | 0.000 | 0.000 | 0.000 | 0.001 | the generation checkpoint check |
| s.ckgen → s.bat | 0.002 | 0.002 | 0.001 | 0.002 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.002 | 0.002 | 0.001 | 0.002 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.006 | 0.006 | 0.005 | 0.007 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.001 | 0.001 | 0.000 | 0.001 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.001 | into decode() |
| s.dec → s.dec.y | 0.000 | 0.000 | 0.000 | 0.001 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.001 | 0.001 | 0.001 | 0.002 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.008 | 0.008 | 0.007 | 0.010 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.001 | 0.001 | 0.000 | 0.001 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.001 | 0.001 | 0.000 | 0.001 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.002 | 0.002 | 0.001 | 0.003 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.in | 0.147 | 0.152 | 0.143 | 0.164 |  |
| **host: last d.sync.end → v.ub.in** | 0.242 | 0.287 | 0.231 | 0.422 |  |
| v.ub.in → first main card's graph_compute entry | 0.540 | 0.911 | 0.529 | 1.465 | the scheduler's input copies for that card |
| **the gap, as S37 counts it** | 0.975 | 1.393 | 0.955 | 2.068 |  |

## Between lines

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd.end → next line's s.upd | 0.279 | 0.279 | 0.265 | 0.294 | the line's write, the task loop |
