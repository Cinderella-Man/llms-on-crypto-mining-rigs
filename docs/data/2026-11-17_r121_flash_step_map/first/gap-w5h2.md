# <OUT>/st-w5h2.jsonl

- lines: 671 (14 without a draft, 657 generation steps); drafted 2609, accepted 1668; drafting passes beyond the drafted tokens 7 (in 3 steps); line writes: median 297 us
- typical steps: 294 of 657 (the commonest of 19 mark sequences among the steps that drafted 4 tokens, 4 drafting passes); the first 3 after each prompt left out

## Waterfall (ms between consecutive marks; the rows add up to the line)

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd → s.post | 0.006 | 0.006 | 0.005 | 0.007 |  |
| s.post → s.shift | 0.001 | 0.001 | 0.001 | 0.001 |  |
| s.shift → s.sel | 0.017 | 0.018 | 0.004 | 0.027 |  |
| s.sel → s.drf | 0.000 | 0.000 | 0.000 | 0.000 |  |
| s.drf → s.drf.in | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.drf.in → mtp.drf | 0.001 | 0.001 | 0.000 | 0.001 |  |
| mtp.drf → mtp.dp | 0.007 | 0.007 | 0.004 | 0.011 |  |
| mtp.dp → d.dec | 0.011 | 0.012 | 0.010 | 0.014 |  |
| d.dec → d.dec.bal | 0.017 | 0.017 | 0.015 | 0.019 |  |
| d.dec.bal → d.dec.mem | 0.040 | 0.041 | 0.037 | 0.044 |  |
| d.dec.mem → d.dec.out | 0.003 | 0.003 | 0.003 | 0.004 |  |
| d.dec.out → d.ub.apply | 0.374 | 0.453 | 0.062 | 0.772 |  |
| d.ub.apply → d.ub.build | 0.423 | 0.426 | 0.410 | 0.445 |  |
| d.ub.build → d.ub.in | 0.032 | 0.032 | 0.020 | 0.040 |  |
| d.ub.in → d.ub.comp | 0.555 | 0.585 | 0.547 | 0.581 |  |
| d.ub.comp → d.ub.cp | 0.022 | 0.022 | 0.021 | 0.023 |  |
| d.ub.cp → d.dec.end | 0.002 | 0.002 | 0.001 | 0.002 |  |
| d.dec.end → mtp.dp.ret | 0.005 | 0.005 | 0.004 | 0.005 |  |
| mtp.dp.ret → d.sync(6) | 0.060 | 0.125 | 0.057 | 0.255 |  |
| d.sync(6) → d.sync.end(6) | 2.904 | 2.961 | 2.543 | 2.977 |  |
| d.sync.end(6) → mtp.dp.smp | 0.007 | 0.007 | 0.006 | 0.007 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp → d.sync | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync → d.sync.end | 0.015 | 0.015 | 0.015 | 0.016 |  |
| d.sync.end → mtp.dp.row | 0.005 | 0.005 | 0.004 | 0.006 |  |
| mtp.dp.row → mtp.dp#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.dp#2 → d.dec#2 | 0.011 | 0.012 | 0.010 | 0.014 |  |
| d.dec#2 → d.dec.bal#2 | 0.017 | 0.017 | 0.016 | 0.019 |  |
| d.dec.bal#2 → d.dec.mem#2 | 0.046 | 0.046 | 0.042 | 0.050 |  |
| d.dec.mem#2 → d.dec.out#2 | 0.003 | 0.003 | 0.003 | 0.004 |  |
| d.dec.out#2 → d.ub.apply#2 | 0.072 | 0.068 | 0.044 | 0.081 |  |
| d.ub.apply#2 → d.ub.reuse | 0.047 | 0.049 | 0.044 | 0.059 |  |
| d.ub.reuse → d.ub.in#2 | 0.029 | 0.029 | 0.019 | 0.037 |  |
| d.ub.in#2 → d.ub.comp#2 | 0.480 | 0.503 | 0.464 | 0.504 |  |
| d.ub.comp#2 → d.ub.cp#2 | 0.019 | 0.019 | 0.018 | 0.021 |  |
| d.ub.cp#2 → d.dec.end#2 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.dec.end#2 → mtp.dp.ret#2 | 0.005 | 0.005 | 0.004 | 0.006 |  |
| mtp.dp.ret#2 → d.sync(6)#2 | 0.057 | 0.117 | 0.053 | 0.248 |  |
| d.sync(6)#2 → d.sync.end(6)#2 | 2.874 | 2.888 | 2.543 | 2.960 |  |
| d.sync.end(6)#2 → mtp.dp.smp#2 | 0.004 | 0.004 | 0.003 | 0.005 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp#2 → d.sync#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync#2 → d.sync.end#2 | 0.015 | 0.015 | 0.014 | 0.015 |  |
| d.sync.end#2 → mtp.dp.row#2 | 0.004 | 0.004 | 0.003 | 0.006 |  |
| mtp.dp.row#2 → mtp.dp#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.dp#3 → d.dec#3 | 0.009 | 0.009 | 0.008 | 0.011 |  |
| d.dec#3 → d.dec.bal#3 | 0.015 | 0.015 | 0.013 | 0.017 |  |
| d.dec.bal#3 → d.dec.mem#3 | 0.040 | 0.041 | 0.036 | 0.046 |  |
| d.dec.mem#3 → d.dec.out#3 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.dec.out#3 → d.ub.apply#3 | 0.062 | 0.063 | 0.040 | 0.078 |  |
| d.ub.apply#3 → d.ub.reuse#2 | 0.042 | 0.042 | 0.037 | 0.049 |  |
| d.ub.reuse#2 → d.ub.in#3 | 0.027 | 0.028 | 0.018 | 0.036 |  |
| d.ub.in#3 → d.ub.comp#3 | 0.470 | 0.472 | 0.452 | 0.493 |  |
| d.ub.comp#3 → d.ub.cp#3 | 0.018 | 0.018 | 0.017 | 0.020 |  |
| d.ub.cp#3 → d.dec.end#3 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.dec.end#3 → mtp.dp.ret#3 | 0.004 | 0.005 | 0.004 | 0.005 |  |
| mtp.dp.ret#3 → d.sync(6)#3 | 0.054 | 0.108 | 0.050 | 0.241 |  |
| d.sync(6)#3 → d.sync.end(6)#3 | 2.899 | 2.916 | 2.559 | 3.001 |  |
| d.sync.end(6)#3 → mtp.dp.smp#3 | 0.004 | 0.004 | 0.003 | 0.005 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp#3 → d.sync#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync#3 → d.sync.end#3 | 0.015 | 0.015 | 0.014 | 0.015 |  |
| d.sync.end#3 → mtp.dp.row#3 | 0.004 | 0.004 | 0.003 | 0.005 |  |
| mtp.dp.row#3 → mtp.dp#4 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.dp#4 → d.dec#4 | 0.010 | 0.010 | 0.008 | 0.012 |  |
| d.dec#4 → d.dec.bal#4 | 0.015 | 0.016 | 0.013 | 0.018 |  |
| d.dec.bal#4 → d.dec.mem#4 | 0.039 | 0.040 | 0.035 | 0.046 |  |
| d.dec.mem#4 → d.dec.out#4 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.dec.out#4 → d.ub.apply#4 | 0.067 | 0.066 | 0.040 | 0.085 |  |
| d.ub.apply#4 → d.ub.reuse#3 | 0.042 | 0.043 | 0.037 | 0.051 |  |
| d.ub.reuse#3 → d.ub.in#4 | 0.028 | 0.029 | 0.018 | 0.036 |  |
| d.ub.in#4 → d.ub.comp#4 | 0.473 | 0.475 | 0.456 | 0.496 |  |
| d.ub.comp#4 → d.ub.cp#4 | 0.015 | 0.015 | 0.014 | 0.016 |  |
| d.ub.cp#4 → d.dec.end#4 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.dec.end#4 → mtp.dp.ret#4 | 0.004 | 0.005 | 0.004 | 0.005 |  |
| mtp.dp.ret#4 → s.vp | 0.007 | 0.007 | 0.006 | 0.007 |  |
| s.vp → p.dec | 0.006 | 0.006 | 0.005 | 0.007 |  |
| p.dec → p.dec.bal | 0.017 | 0.017 | 0.015 | 0.018 |  |
| p.dec.bal → p.dec.mem | 0.044 | 0.043 | 0.038 | 0.047 |  |
| p.dec.mem → p.dec.out | 0.008 | 0.008 | 0.007 | 0.008 |  |
| p.dec.out → p.ub.apply | 0.457 | 0.601 | 0.063 | 1.033 |  |
| p.ub.apply → p.ub.reuse | 0.071 | 0.071 | 0.054 | 0.086 |  |
| p.ub.reuse → p.ub.in | 0.024 | 0.026 | 0.012 | 0.035 |  |
| p.ub.in → p.ub.up | 0.136 | 0.139 | 0.130 | 0.151 |  |
| p.ub.up → p.dec.prep | 0.000 | 0.000 | 0.000 | 0.001 |  |
| p.dec.prep → s.vp.end | 0.004 | 0.004 | 0.003 | 0.005 |  |
| s.vp.end → d.sync(6)#4 | 0.060 | 0.113 | 0.056 | 0.248 |  |
| d.sync(6)#4 → d.sync.end(6)#4 | 1.851 | 1.795 | 1.273 | 2.047 |  |
| d.sync.end(6)#4 → mtp.dp.smp#4 | 0.005 | 0.005 | 0.004 | 0.006 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp#4 → d.sync#4 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync#4 → d.sync.end#4 | 0.015 | 0.015 | 0.014 | 0.015 |  |
| d.sync.end#4 → mtp.dp.row#4 | 0.003 | 0.004 | 0.003 | 0.004 |  |
| mtp.dp.row#4 → mtp.drf.end | 0.050 | 0.095 | 0.047 | 0.211 | the drafter's end (shortlist and carry reset, length check) |
| mtp.drf.end → s.drf.out | 0.001 | 0.001 | 0.001 | 0.002 | back in the server |
| s.drf.out → s.drf.end | 0.000 | 0.000 | 0.000 | 0.000 | the yield's end: the queue's worker thread stops |
| s.drf.end → s.ckpt | 0.018 | 0.018 | 0.014 | 0.021 | draft bookkeeping, the draft context's seq_rm, checkpoints |
| s.ckpt → s.ckgen | 0.000 | 0.000 | 0.000 | 0.001 | the generation checkpoint check |
| s.ckgen → s.bat | 0.002 | 0.002 | 0.001 | 0.002 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.002 | 0.002 | 0.001 | 0.002 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.006 | 0.006 | 0.004 | 0.007 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.001 | 0.001 | 0.000 | 0.001 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.001 | into decode() |
| s.dec → s.dec.y | 0.001 | 0.001 | 0.000 | 0.001 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.001 | 0.001 | 0.001 | 0.002 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.008 | 0.008 | 0.007 | 0.010 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.001 | 0.001 | 0.000 | 0.001 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.001 | 0.001 | 0.000 | 0.001 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.001 | 0.002 | 0.001 | 0.002 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.in | 0.186 | 0.193 | 0.181 | 0.218 |  |
| v.ub.in → v.ub.comp | 4.020 | 4.300 | 3.896 | 4.708 |  |
| v.ub.comp → v.ub.cp | 0.062 | 0.067 | 0.060 | 0.077 |  |
| v.ub.cp → v.dec.end | 0.004 | 0.004 | 0.004 | 0.005 |  |
| v.dec.end → s.dec.ret | 0.033 | 0.033 | 0.028 | 0.036 |  |
| s.dec.ret → v.sync | 0.001 | 0.001 | 0.000 | 0.001 |  |
| v.sync → v.sync.end | 134.788 | 136.419 | 118.812 | 148.307 |  |
| v.sync.end → s.dec.out | 0.001 | 0.001 | 0.001 | 0.002 |  |
| s.dec.out → s.dec.end | 0.003 | 0.003 | 0.002 | 0.004 |  |
| s.dec.end → s.mpd | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.mpd → s.cu.in | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.cu.in → mtp.cu | 0.002 | 0.002 | 0.002 | 0.002 |  |
| mtp.cu → mtp.cu.dec | 0.261 | 0.262 | 0.257 | 0.268 |  |
| mtp.cu.dec → d.dec#5 | 0.007 | 0.008 | 0.006 | 0.008 |  |
| d.dec#5 → d.dec.bal#5 | 0.020 | 0.020 | 0.019 | 0.021 |  |
| d.dec.bal#5 → d.dec.mem#5 | 0.074 | 0.074 | 0.062 | 0.081 |  |
| d.dec.mem#5 → d.dec.out#5 | 0.008 | 0.008 | 0.007 | 0.008 |  |
| d.dec.out#5 → d.ub.apply#5 | 0.431 | 0.528 | 0.073 | 0.866 |  |
| d.ub.apply#5 → d.ub.reuse#4 | 0.079 | 0.079 | 0.069 | 0.088 |  |
| d.ub.reuse#4 → d.ub.in#5 | 0.027 | 0.028 | 0.016 | 0.036 |  |
| d.ub.in#5 → d.ub.comp#5 | 0.210 | 0.213 | 0.202 | 0.229 |  |
| d.ub.comp#5 → d.ub.cp#5 | 0.002 | 0.002 | 0.001 | 0.002 |  |
| d.ub.cp#5 → d.dec.end#5 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.dec.end#5 → mtp.cu.ret | 0.005 | 0.006 | 0.005 | 0.006 |  |
| mtp.cu.ret → mtp.cu.end | 0.261 | 0.259 | 0.240 | 0.266 |  |
| mtp.cu.end → s.cu.out | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.cu.out → s.cu.end | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.cu.end → s.pd | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.pd → s.acc | 0.002 | 0.002 | 0.002 | 0.003 |  |
| s.acc → s.acc.cl | 0.015 | 0.017 | 0.013 | 0.021 |  |
| s.acc.cl → v.sync(30) | 0.002 | 0.002 | 0.001 | 0.002 |  |
| v.sync(30) → v.sync.end(30) | 0.473 | 0.477 | 0.457 | 0.504 |  |
| v.sync.end(30) → s.acc.smp | 0.001 | 0.001 | 0.001 | 0.002 |  |
| s.acc.smp → s.acc.spec | 0.002 | 0.002 | 0.001 | 0.002 |  |
| s.acc.spec → s.rm | 0.018 | 0.018 | 0.014 | 0.020 |  |
| s.rm → s.tok | 0.006 | 0.006 | 0.005 | 0.006 |  |
| s.tok → s.tok#2 | 0.001 | 0.001 | 0.000 | 0.002 |  |
| s.tok#2 → s.tok#3 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.tok#3 → s.tok#4 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.tok#4 → s.tok#5 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.tok#5 → s.tg | 0.000 | 0.001 | 0.000 | 0.001 |  |
| s.tg → s.upd.end | 0.001 | 0.001 | 0.001 | 0.002 |  |
| **the line (s.upd → s.upd.end)** | 156.793 | 159.012 | 139.346 | 171.720 |  |

## The gap: the last drafting pass's wait → the verify pass's inputs set

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| draft card's work end → d.sync.end (last pass) | 0.195 | 0.197 | 0.189 | 0.203 | the drafted id's way back, the host's wake |
| d.sync.end#4 → mtp.dp.row#4 | 0.003 | 0.004 | 0.003 | 0.004 |  |
| mtp.dp.row#4 → mtp.drf.end | 0.050 | 0.095 | 0.047 | 0.211 | the drafter's end (shortlist and carry reset, length check) |
| mtp.drf.end → s.drf.out | 0.001 | 0.001 | 0.001 | 0.002 | back in the server |
| s.drf.out → s.drf.end | 0.000 | 0.000 | 0.000 | 0.000 | the yield's end: the queue's worker thread stops |
| s.drf.end → s.ckpt | 0.018 | 0.018 | 0.014 | 0.021 | draft bookkeeping, the draft context's seq_rm, checkpoints |
| s.ckpt → s.ckgen | 0.000 | 0.000 | 0.000 | 0.001 | the generation checkpoint check |
| s.ckgen → s.bat | 0.002 | 0.002 | 0.001 | 0.002 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.002 | 0.002 | 0.001 | 0.002 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.006 | 0.006 | 0.004 | 0.007 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.001 | 0.001 | 0.000 | 0.001 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.001 | into decode() |
| s.dec → s.dec.y | 0.001 | 0.001 | 0.000 | 0.001 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.001 | 0.001 | 0.001 | 0.002 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.008 | 0.008 | 0.007 | 0.010 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.001 | 0.001 | 0.000 | 0.001 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.001 | 0.001 | 0.000 | 0.001 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.001 | 0.002 | 0.001 | 0.002 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.in | 0.186 | 0.193 | 0.181 | 0.218 |  |
| **host: last d.sync.end → v.ub.in** | 0.284 | 0.336 | 0.271 | 0.480 |  |
| v.ub.in → first main card's graph_compute entry | 0.665 | 0.980 | 0.654 | 1.360 | the scheduler's input copies for that card |
| **the gap, as S37 counts it** | 1.145 | 1.512 | 1.118 | 2.043 |  |

## Between lines

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd.end → next line's s.upd | 0.326 | 0.325 | 0.308 | 0.340 | the line's write, the task loop |
