# <OUT>/st-w4h1.jsonl

- lines: 375 (11 without a draft, 364 generation steps); drafted 1082, accepted 684; drafting passes beyond the drafted tokens 10 (in 5 steps); line writes: median 248 us
- typical steps: 171 of 364 (the commonest of 5 mark sequences among the steps that drafted 3 tokens, 3 drafting passes); the first 3 after each prompt left out

## Waterfall (ms between consecutive marks; the rows add up to the line)

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd → s.post | 0.006 | 0.006 | 0.006 | 0.007 |  |
| s.post → s.shift | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.shift → s.sel | 0.015 | 0.016 | 0.004 | 0.028 |  |
| s.sel → s.drf | 0.000 | 0.000 | 0.000 | 0.000 |  |
| s.drf → s.drf.in | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.drf.in → mtp.drf | 0.001 | 0.001 | 0.000 | 0.001 |  |
| mtp.drf → mtp.dp | 0.010 | 0.010 | 0.009 | 0.010 |  |
| mtp.dp → d.dec | 0.006 | 0.007 | 0.006 | 0.007 |  |
| d.dec → d.dec.bal | 0.012 | 0.012 | 0.012 | 0.013 |  |
| d.dec.bal → d.dec.mem | 0.028 | 0.029 | 0.026 | 0.032 |  |
| d.dec.mem → d.dec.out | 0.004 | 0.004 | 0.003 | 0.004 |  |
| d.dec.out → d.ub.apply | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.ub.apply → d.ub.build | 2.054 | 2.533 | 1.618 | 3.398 |  |
| d.ub.build → d.ub.in | 0.027 | 0.046 | 0.016 | 0.076 |  |
| d.ub.in → d.ub.comp | 0.777 | 0.749 | 0.694 | 0.797 |  |
| d.ub.comp → d.ub.cp | 0.023 | 0.024 | 0.022 | 0.024 |  |
| d.ub.cp → d.dec.end | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.dec.end → mtp.dp.ret | 0.002 | 0.002 | 0.001 | 0.002 |  |
| mtp.dp.ret → d.sync(6) | 0.001 | 0.001 | 0.001 | 0.002 |  |
| d.sync(6) → d.sync.end(6) | 4.188 | 4.521 | 4.092 | 4.980 |  |
| d.sync.end(6) → mtp.dp.smp | 0.006 | 0.007 | 0.006 | 0.007 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp → d.sync | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync → d.sync.end | 0.015 | 0.015 | 0.014 | 0.016 |  |
| d.sync.end → mtp.dp.row | 0.004 | 0.004 | 0.003 | 0.004 |  |
| mtp.dp.row → mtp.dp#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.dp#2 → d.dec#2 | 0.006 | 0.006 | 0.006 | 0.007 |  |
| d.dec#2 → d.dec.bal#2 | 0.012 | 0.012 | 0.011 | 0.013 |  |
| d.dec.bal#2 → d.dec.mem#2 | 0.031 | 0.031 | 0.028 | 0.033 |  |
| d.dec.mem#2 → d.dec.out#2 | 0.003 | 0.003 | 0.003 | 0.004 |  |
| d.dec.out#2 → d.ub.apply#2 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.ub.apply#2 → d.ub.reuse | 0.036 | 0.036 | 0.035 | 0.038 |  |
| d.ub.reuse → d.ub.in#2 | 0.013 | 0.039 | 0.010 | 0.068 |  |
| d.ub.in#2 → d.ub.comp#2 | 0.694 | 0.697 | 0.641 | 0.764 |  |
| d.ub.comp#2 → d.ub.cp#2 | 0.022 | 0.023 | 0.021 | 0.023 |  |
| d.ub.cp#2 → d.dec.end#2 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.dec.end#2 → mtp.dp.ret#2 | 0.003 | 0.003 | 0.003 | 0.004 |  |
| mtp.dp.ret#2 → d.sync(6)#2 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.sync(6)#2 → d.sync.end(6)#2 | 4.279 | 4.653 | 4.215 | 5.096 |  |
| d.sync.end(6)#2 → mtp.dp.smp#2 | 0.004 | 0.004 | 0.003 | 0.005 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp#2 → d.sync#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync#2 → d.sync.end#2 | 0.015 | 0.015 | 0.014 | 0.015 |  |
| d.sync.end#2 → mtp.dp.row#2 | 0.004 | 0.004 | 0.003 | 0.004 |  |
| mtp.dp.row#2 → mtp.dp#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.dp#3 → d.dec#3 | 0.006 | 0.006 | 0.005 | 0.007 |  |
| d.dec#3 → d.dec.bal#3 | 0.012 | 0.013 | 0.011 | 0.013 |  |
| d.dec.bal#3 → d.dec.mem#3 | 0.031 | 0.031 | 0.026 | 0.035 |  |
| d.dec.mem#3 → d.dec.out#3 | 0.003 | 0.003 | 0.003 | 0.004 |  |
| d.dec.out#3 → d.ub.apply#3 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.ub.apply#3 → d.ub.reuse#2 | 0.036 | 0.037 | 0.034 | 0.038 |  |
| d.ub.reuse#2 → d.ub.in#3 | 0.020 | 0.040 | 0.010 | 0.070 |  |
| d.ub.in#3 → d.ub.comp#3 | 0.357 | 0.367 | 0.314 | 0.421 |  |
| d.ub.comp#3 → d.ub.cp#3 | 0.017 | 0.018 | 0.016 | 0.019 |  |
| d.ub.cp#3 → d.dec.end#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.dec.end#3 → mtp.dp.ret#3 | 0.002 | 0.002 | 0.002 | 0.003 |  |
| mtp.dp.ret#3 → d.sync(6)#3 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.sync(6)#3 → d.sync.end(6)#3 | 4.315 | 4.706 | 4.267 | 5.160 |  |
| d.sync.end(6)#3 → mtp.dp.smp#3 | 0.003 | 0.003 | 0.003 | 0.004 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp#3 → d.sync#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync#3 → d.sync.end#3 | 0.015 | 0.015 | 0.014 | 0.015 |  |
| d.sync.end#3 → mtp.dp.row#3 | 0.002 | 0.003 | 0.002 | 0.003 |  |
| mtp.dp.row#3 → mtp.drf.end | 0.001 | 0.001 | 0.001 | 0.001 | the drafter's end (shortlist and carry reset, length check) |
| mtp.drf.end → s.drf.out | 0.001 | 0.001 | 0.001 | 0.002 | back in the server |
| s.drf.out → s.drf.end | 0.000 | 0.000 | 0.000 | 0.000 | the yield's end: the queue's worker thread stops |
| s.drf.end → s.ckpt | 0.094 | 0.094 | 0.092 | 0.097 | draft bookkeeping, the draft context's seq_rm, checkpoints |
| s.ckpt → s.ckgen | 0.000 | 0.000 | 0.000 | 0.001 | the generation checkpoint check |
| s.ckgen → s.bat | 0.002 | 0.002 | 0.001 | 0.002 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.002 | 0.002 | 0.002 | 0.002 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.005 | 0.005 | 0.005 | 0.006 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.001 | 0.001 | 0.000 | 0.001 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.001 | into decode() |
| s.dec → s.dec.y | 0.000 | 0.000 | 0.000 | 0.001 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.001 | 0.001 | 0.001 | 0.002 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.004 | 0.004 | 0.004 | 0.005 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.015 | 0.015 | 0.014 | 0.016 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.039 | 0.039 | 0.035 | 0.043 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.007 | 0.008 | 0.006 | 0.009 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.apply | 0.005 | 0.005 | 0.004 | 0.005 | the ubatch, the memory context applied |
| v.ub.apply → v.ub.reuse | 0.039 | 0.040 | 0.037 | 0.045 | graph parameters, the graph reused |
| v.ub.reuse → v.ub.in | 0.016 | 0.046 | 0.012 | 0.080 | the graph's inputs set (host side) |
| v.ub.in → v.ub.comp | 5.999 | 4.796 | 2.892 | 6.211 |  |
| v.ub.comp → v.ub.cp | 0.050 | 0.052 | 0.049 | 0.059 |  |
| v.ub.cp → v.dec.end | 0.003 | 0.003 | 0.003 | 0.004 |  |
| v.dec.end → s.dec.ret | 0.005 | 0.005 | 0.004 | 0.006 |  |
| s.dec.ret → v.sync | 0.001 | 0.001 | 0.000 | 0.001 |  |
| v.sync → v.sync.end | 155.823 | 161.377 | 151.325 | 171.252 |  |
| v.sync.end → s.dec.out | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.dec.out → s.dec.end | 0.003 | 0.003 | 0.002 | 0.004 |  |
| s.dec.end → s.mpd | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.mpd → s.cu.in | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.cu.in → mtp.cu | 0.002 | 0.002 | 0.002 | 0.002 |  |
| mtp.cu → mtp.cu.dec | 0.104 | 0.104 | 0.099 | 0.108 |  |
| mtp.cu.dec → d.dec#4 | 0.007 | 0.007 | 0.007 | 0.008 |  |
| d.dec#4 → d.dec.bal#4 | 0.017 | 0.017 | 0.016 | 0.018 |  |
| d.dec.bal#4 → d.dec.mem#4 | 0.038 | 0.038 | 0.034 | 0.041 |  |
| d.dec.mem#4 → d.dec.out#4 | 0.007 | 0.007 | 0.006 | 0.007 |  |
| d.dec.out#4 → d.ub.apply#4 | 0.005 | 0.005 | 0.004 | 0.006 |  |
| d.ub.apply#4 → d.ub.build#2 | 1.581 | 1.574 | 1.472 | 1.655 |  |
| d.ub.build#2 → d.ub.in#4 | 0.020 | 0.046 | 0.013 | 0.078 |  |
| d.ub.in#4 → d.ub.comp#4 | 0.923 | 1.062 | 0.887 | 1.243 |  |
| d.ub.comp#4 → d.ub.cp#4 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.ub.cp#4 → d.dec.end#4 | 0.002 | 0.002 | 0.001 | 0.002 |  |
| d.dec.end#4 → mtp.cu.ret | 0.002 | 0.002 | 0.001 | 0.002 |  |
| mtp.cu.ret → mtp.cu.end | 0.177 | 0.176 | 0.166 | 0.185 |  |
| mtp.cu.end → s.cu.out | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.cu.out → s.cu.end | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.cu.end → s.pd | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.pd → s.acc | 0.002 | 0.002 | 0.001 | 0.002 |  |
| s.acc → s.acc.cl | 0.013 | 0.013 | 0.011 | 0.014 |  |
| s.acc.cl → v.sync(24) | 0.002 | 0.002 | 0.001 | 0.002 |  |
| v.sync(24) → v.sync.end(24) | 0.400 | 0.404 | 0.394 | 0.416 |  |
| v.sync.end(24) → s.acc.smp | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.acc.smp → s.acc.spec | 0.002 | 0.002 | 0.001 | 0.002 |  |
| s.acc.spec → s.rm | 0.182 | 0.184 | 0.181 | 0.193 |  |
| s.rm → s.tok | 0.006 | 0.022 | 0.005 | 0.006 |  |
| s.tok → s.tok#2 | 0.001 | 0.001 | 0.001 | 0.002 |  |
| s.tok#2 → s.tok#3 | 0.001 | 0.018 | 0.000 | 0.001 |  |
| s.tok#3 → s.tok#4 | 0.001 | 0.017 | 0.000 | 0.001 |  |
| s.tok#4 → s.tg | 0.000 | 0.001 | 0.000 | 0.001 |  |
| s.tg → s.upd.end | 0.001 | 0.001 | 0.001 | 0.002 |  |
| **the line (s.upd → s.upd.end)** | 179.729 | 188.991 | 174.714 | 202.749 |  |

## The gap: the last drafting pass's wait → the verify pass's inputs set

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| draft card's work end → d.sync.end (last pass) | 0.296 | 0.299 | 0.292 | 0.307 | the drafted id's way back, the host's wake |
| d.sync.end#3 → mtp.dp.row#3 | 0.002 | 0.003 | 0.002 | 0.003 |  |
| mtp.dp.row#3 → mtp.drf.end | 0.001 | 0.001 | 0.001 | 0.001 | the drafter's end (shortlist and carry reset, length check) |
| mtp.drf.end → s.drf.out | 0.001 | 0.001 | 0.001 | 0.002 | back in the server |
| s.drf.out → s.drf.end | 0.000 | 0.000 | 0.000 | 0.000 | the yield's end: the queue's worker thread stops |
| s.drf.end → s.ckpt | 0.094 | 0.094 | 0.092 | 0.097 | draft bookkeeping, the draft context's seq_rm, checkpoints |
| s.ckpt → s.ckgen | 0.000 | 0.000 | 0.000 | 0.001 | the generation checkpoint check |
| s.ckgen → s.bat | 0.002 | 0.002 | 0.001 | 0.002 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.002 | 0.002 | 0.002 | 0.002 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.005 | 0.005 | 0.005 | 0.006 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.001 | 0.001 | 0.000 | 0.001 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.001 | into decode() |
| s.dec → s.dec.y | 0.000 | 0.000 | 0.000 | 0.001 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.001 | 0.001 | 0.001 | 0.002 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.004 | 0.004 | 0.004 | 0.005 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.015 | 0.015 | 0.014 | 0.016 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.039 | 0.039 | 0.035 | 0.043 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.007 | 0.008 | 0.006 | 0.009 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.apply | 0.005 | 0.005 | 0.004 | 0.005 | the ubatch, the memory context applied |
| v.ub.apply → v.ub.reuse | 0.039 | 0.040 | 0.037 | 0.045 | graph parameters, the graph reused |
| v.ub.reuse → v.ub.in | 0.016 | 0.046 | 0.012 | 0.080 | the graph's inputs set (host side) |
| **host: last d.sync.end → v.ub.in** | 0.250 | 0.268 | 0.222 | 0.314 |  |
| v.ub.in → first main card's graph_compute entry | 0.599 | 0.508 | 0.301 | 0.664 | the scheduler's input copies for that card |
| **the gap, as S37 counts it** | 1.145 | 1.074 | 0.821 | 1.281 |  |

## Between lines

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd.end → next line's s.upd | 0.266 | 0.267 | 0.255 | 0.283 | the line's write, the task loop |
