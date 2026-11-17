# <OUT>/st-w5h1.jsonl

- lines: 671 (14 without a draft, 657 generation steps); drafted 2609, accepted 1668; drafting passes beyond the drafted tokens 7 (in 3 steps); line writes: median 296 us
- typical steps: 294 of 657 (the commonest of 19 mark sequences among the steps that drafted 4 tokens, 4 drafting passes); the first 3 after each prompt left out

## Waterfall (ms between consecutive marks; the rows add up to the line)

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd → s.post | 0.006 | 0.006 | 0.005 | 0.007 |  |
| s.post → s.shift | 0.001 | 0.001 | 0.001 | 0.001 |  |
| s.shift → s.sel | 0.016 | 0.018 | 0.004 | 0.027 |  |
| s.sel → s.drf | 0.000 | 0.000 | 0.000 | 0.000 |  |
| s.drf → s.drf.in | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.drf.in → mtp.drf | 0.001 | 0.001 | 0.000 | 0.001 |  |
| mtp.drf → mtp.dp | 0.007 | 0.007 | 0.004 | 0.011 |  |
| mtp.dp → d.dec | 0.011 | 0.011 | 0.010 | 0.013 |  |
| d.dec → d.dec.bal | 0.016 | 0.017 | 0.015 | 0.018 |  |
| d.dec.bal → d.dec.mem | 0.041 | 0.042 | 0.037 | 0.045 |  |
| d.dec.mem → d.dec.out | 0.003 | 0.003 | 0.003 | 0.004 |  |
| d.dec.out → d.ub.apply | 0.376 | 0.443 | 0.061 | 0.744 |  |
| d.ub.apply → d.ub.build | 0.418 | 0.421 | 0.408 | 0.437 |  |
| d.ub.build → d.ub.in | 0.031 | 0.033 | 0.021 | 0.041 |  |
| d.ub.in → d.ub.comp | 0.555 | 0.591 | 0.547 | 0.576 |  |
| d.ub.comp → d.ub.cp | 0.021 | 0.021 | 0.020 | 0.022 |  |
| d.ub.cp → d.dec.end | 0.002 | 0.002 | 0.002 | 0.003 |  |
| d.dec.end → mtp.dp.ret | 0.004 | 0.004 | 0.004 | 0.005 |  |
| mtp.dp.ret → d.sync(6) | 0.060 | 0.126 | 0.057 | 0.251 |  |
| d.sync(6) → d.sync.end(6) | 2.908 | 2.970 | 2.545 | 2.980 |  |
| d.sync.end(6) → mtp.dp.smp | 0.006 | 0.006 | 0.005 | 0.007 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp → d.sync | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync → d.sync.end | 0.015 | 0.016 | 0.015 | 0.016 |  |
| d.sync.end → mtp.dp.row | 0.005 | 0.005 | 0.004 | 0.006 |  |
| mtp.dp.row → mtp.dp#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.dp#2 → d.dec#2 | 0.011 | 0.011 | 0.010 | 0.013 |  |
| d.dec#2 → d.dec.bal#2 | 0.016 | 0.017 | 0.015 | 0.018 |  |
| d.dec.bal#2 → d.dec.mem#2 | 0.046 | 0.046 | 0.041 | 0.051 |  |
| d.dec.mem#2 → d.dec.out#2 | 0.003 | 0.003 | 0.003 | 0.004 |  |
| d.dec.out#2 → d.ub.apply#2 | 0.065 | 0.068 | 0.044 | 0.082 |  |
| d.ub.apply#2 → d.ub.reuse | 0.046 | 0.047 | 0.043 | 0.052 |  |
| d.ub.reuse → d.ub.in#2 | 0.029 | 0.029 | 0.018 | 0.037 |  |
| d.ub.in#2 → d.ub.comp#2 | 0.475 | 0.500 | 0.461 | 0.501 |  |
| d.ub.comp#2 → d.ub.cp#2 | 0.019 | 0.019 | 0.018 | 0.020 |  |
| d.ub.cp#2 → d.dec.end#2 | 0.001 | 0.001 | 0.001 | 0.002 |  |
| d.dec.end#2 → mtp.dp.ret#2 | 0.005 | 0.005 | 0.004 | 0.006 |  |
| mtp.dp.ret#2 → d.sync(6)#2 | 0.057 | 0.119 | 0.052 | 0.249 |  |
| d.sync(6)#2 → d.sync.end(6)#2 | 2.866 | 2.891 | 2.546 | 2.965 |  |
| d.sync.end(6)#2 → mtp.dp.smp#2 | 0.004 | 0.004 | 0.003 | 0.005 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp#2 → d.sync#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync#2 → d.sync.end#2 | 0.015 | 0.015 | 0.015 | 0.016 |  |
| d.sync.end#2 → mtp.dp.row#2 | 0.004 | 0.004 | 0.003 | 0.006 |  |
| mtp.dp.row#2 → mtp.dp#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.dp#3 → d.dec#3 | 0.009 | 0.009 | 0.008 | 0.011 |  |
| d.dec#3 → d.dec.bal#3 | 0.015 | 0.015 | 0.012 | 0.017 |  |
| d.dec.bal#3 → d.dec.mem#3 | 0.041 | 0.041 | 0.035 | 0.046 |  |
| d.dec.mem#3 → d.dec.out#3 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.dec.out#3 → d.ub.apply#3 | 0.062 | 0.063 | 0.041 | 0.080 |  |
| d.ub.apply#3 → d.ub.reuse#2 | 0.041 | 0.042 | 0.037 | 0.047 |  |
| d.ub.reuse#2 → d.ub.in#3 | 0.028 | 0.027 | 0.017 | 0.035 |  |
| d.ub.in#3 → d.ub.comp#3 | 0.470 | 0.469 | 0.450 | 0.489 |  |
| d.ub.comp#3 → d.ub.cp#3 | 0.018 | 0.018 | 0.017 | 0.019 |  |
| d.ub.cp#3 → d.dec.end#3 | 0.001 | 0.001 | 0.001 | 0.002 |  |
| d.dec.end#3 → mtp.dp.ret#3 | 0.005 | 0.005 | 0.004 | 0.005 |  |
| mtp.dp.ret#3 → d.sync(6)#3 | 0.054 | 0.119 | 0.049 | 0.245 |  |
| d.sync(6)#3 → d.sync.end(6)#3 | 2.905 | 2.909 | 2.552 | 3.003 |  |
| d.sync.end(6)#3 → mtp.dp.smp#3 | 0.004 | 0.004 | 0.003 | 0.005 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp#3 → d.sync#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync#3 → d.sync.end#3 | 0.015 | 0.015 | 0.015 | 0.016 |  |
| d.sync.end#3 → mtp.dp.row#3 | 0.004 | 0.004 | 0.003 | 0.005 |  |
| mtp.dp.row#3 → mtp.dp#4 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.dp#4 → d.dec#4 | 0.010 | 0.010 | 0.008 | 0.012 |  |
| d.dec#4 → d.dec.bal#4 | 0.015 | 0.015 | 0.013 | 0.018 |  |
| d.dec.bal#4 → d.dec.mem#4 | 0.041 | 0.041 | 0.035 | 0.048 |  |
| d.dec.mem#4 → d.dec.out#4 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.dec.out#4 → d.ub.apply#4 | 0.070 | 0.067 | 0.041 | 0.084 |  |
| d.ub.apply#4 → d.ub.reuse#3 | 0.042 | 0.043 | 0.037 | 0.050 |  |
| d.ub.reuse#3 → d.ub.in#4 | 0.029 | 0.028 | 0.017 | 0.036 |  |
| d.ub.in#4 → d.ub.comp#4 | 0.474 | 0.473 | 0.455 | 0.494 |  |
| d.ub.comp#4 → d.ub.cp#4 | 0.015 | 0.015 | 0.013 | 0.016 |  |
| d.ub.cp#4 → d.dec.end#4 | 0.001 | 0.001 | 0.001 | 0.002 |  |
| d.dec.end#4 → mtp.dp.ret#4 | 0.005 | 0.005 | 0.004 | 0.005 |  |
| mtp.dp.ret#4 → s.vp | 0.007 | 0.007 | 0.006 | 0.008 |  |
| s.vp → p.dec | 0.006 | 0.006 | 0.005 | 0.007 |  |
| p.dec → p.dec.bal | 0.016 | 0.016 | 0.014 | 0.017 |  |
| p.dec.bal → p.dec.mem | 0.043 | 0.044 | 0.039 | 0.048 |  |
| p.dec.mem → p.dec.out | 0.007 | 0.008 | 0.007 | 0.008 |  |
| p.dec.out → p.ub.apply | 0.483 | 0.634 | 0.065 | 1.089 |  |
| p.ub.apply → p.ub.reuse | 0.071 | 0.071 | 0.053 | 0.086 |  |
| p.ub.reuse → p.ub.in | 0.025 | 0.026 | 0.012 | 0.036 |  |
| p.ub.in → p.ub.up | 0.137 | 0.140 | 0.132 | 0.155 |  |
| p.ub.up → p.dec.prep | 0.000 | 0.000 | 0.000 | 0.001 |  |
| p.dec.prep → s.vp.end | 0.004 | 0.004 | 0.003 | 0.004 |  |
| s.vp.end → d.sync(6)#4 | 0.060 | 0.114 | 0.055 | 0.249 |  |
| d.sync(6)#4 → d.sync.end(6)#4 | 1.840 | 1.761 | 1.237 | 2.040 |  |
| d.sync.end(6)#4 → mtp.dp.smp#4 | 0.005 | 0.005 | 0.004 | 0.005 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp#4 → d.sync#4 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync#4 → d.sync.end#4 | 0.015 | 0.015 | 0.015 | 0.016 |  |
| d.sync.end#4 → mtp.dp.row#4 | 0.003 | 0.003 | 0.003 | 0.004 |  |
| mtp.dp.row#4 → mtp.drf.end | 0.050 | 0.095 | 0.047 | 0.212 | the drafter's end (shortlist and carry reset, length check) |
| mtp.drf.end → s.drf.out | 0.001 | 0.001 | 0.001 | 0.002 | back in the server |
| s.drf.out → s.drf.end | 0.000 | 0.000 | 0.000 | 0.000 | the yield's end: the queue's worker thread stops |
| s.drf.end → s.ckpt | 0.018 | 0.018 | 0.013 | 0.021 | draft bookkeeping, the draft context's seq_rm, checkpoints |
| s.ckpt → s.ckgen | 0.000 | 0.000 | 0.000 | 0.001 | the generation checkpoint check |
| s.ckgen → s.bat | 0.002 | 0.002 | 0.002 | 0.002 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.002 | 0.002 | 0.001 | 0.002 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.006 | 0.006 | 0.004 | 0.007 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.001 | 0.001 | 0.000 | 0.001 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.001 | into decode() |
| s.dec → s.dec.y | 0.000 | 0.000 | 0.000 | 0.001 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.001 | 0.001 | 0.001 | 0.001 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.008 | 0.009 | 0.007 | 0.010 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.001 | 0.001 | 0.000 | 0.001 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.001 | 0.001 | 0.000 | 0.001 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.001 | 0.002 | 0.001 | 0.002 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.in | 0.187 | 0.194 | 0.182 | 0.219 |  |
| v.ub.in → v.ub.comp | 4.038 | 4.291 | 3.897 | 4.794 |  |
| v.ub.comp → v.ub.cp | 0.061 | 0.066 | 0.059 | 0.078 |  |
| v.ub.cp → v.dec.end | 0.004 | 0.004 | 0.004 | 0.005 |  |
| v.dec.end → s.dec.ret | 0.034 | 0.033 | 0.029 | 0.036 |  |
| s.dec.ret → v.sync | 0.001 | 0.001 | 0.000 | 0.001 |  |
| v.sync → v.sync.end | 134.717 | 136.386 | 118.761 | 148.340 |  |
| v.sync.end → s.dec.out | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.dec.out → s.dec.end | 0.003 | 0.003 | 0.002 | 0.004 |  |
| s.dec.end → s.mpd | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.mpd → s.cu.in | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.cu.in → mtp.cu | 0.002 | 0.002 | 0.002 | 0.003 |  |
| mtp.cu → mtp.cu.dec | 0.262 | 0.262 | 0.258 | 0.267 |  |
| mtp.cu.dec → d.dec#5 | 0.007 | 0.007 | 0.006 | 0.007 |  |
| d.dec#5 → d.dec.bal#5 | 0.020 | 0.020 | 0.019 | 0.021 |  |
| d.dec.bal#5 → d.dec.mem#5 | 0.076 | 0.075 | 0.066 | 0.082 |  |
| d.dec.mem#5 → d.dec.out#5 | 0.008 | 0.008 | 0.007 | 0.008 |  |
| d.dec.out#5 → d.ub.apply#5 | 0.435 | 0.517 | 0.075 | 0.834 |  |
| d.ub.apply#5 → d.ub.reuse#4 | 0.082 | 0.082 | 0.070 | 0.094 |  |
| d.ub.reuse#4 → d.ub.in#5 | 0.028 | 0.029 | 0.016 | 0.036 |  |
| d.ub.in#5 → d.ub.comp#5 | 0.210 | 0.214 | 0.201 | 0.233 |  |
| d.ub.comp#5 → d.ub.cp#5 | 0.002 | 0.002 | 0.001 | 0.002 |  |
| d.ub.cp#5 → d.dec.end#5 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.dec.end#5 → mtp.cu.ret | 0.006 | 0.006 | 0.005 | 0.007 |  |
| mtp.cu.ret → mtp.cu.end | 0.261 | 0.259 | 0.240 | 0.266 |  |
| mtp.cu.end → s.cu.out | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.cu.out → s.cu.end | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.cu.end → s.pd | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.pd → s.acc | 0.002 | 0.002 | 0.002 | 0.002 |  |
| s.acc → s.acc.cl | 0.015 | 0.016 | 0.013 | 0.021 |  |
| s.acc.cl → v.sync(30) | 0.002 | 0.002 | 0.001 | 0.002 |  |
| v.sync(30) → v.sync.end(30) | 0.487 | 0.492 | 0.470 | 0.518 |  |
| v.sync.end(30) → s.acc.smp | 0.001 | 0.001 | 0.001 | 0.001 |  |
| s.acc.smp → s.acc.spec | 0.002 | 0.002 | 0.001 | 0.002 |  |
| s.acc.spec → s.rm | 0.018 | 0.018 | 0.014 | 0.020 |  |
| s.rm → s.tok | 0.006 | 0.006 | 0.005 | 0.006 |  |
| s.tok → s.tok#2 | 0.001 | 0.001 | 0.001 | 0.002 |  |
| s.tok#2 → s.tok#3 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.tok#3 → s.tok#4 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.tok#4 → s.tok#5 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.tok#5 → s.tg | 0.000 | 0.001 | 0.000 | 0.001 |  |
| s.tg → s.upd.end | 0.001 | 0.001 | 0.001 | 0.001 |  |
| **the line (s.upd → s.upd.end)** | 156.785 | 158.976 | 139.333 | 171.737 |  |

## The gap: the last drafting pass's wait → the verify pass's inputs set

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| draft card's work end → d.sync.end (last pass) | 0.199 | 0.201 | 0.193 | 0.207 | the drafted id's way back, the host's wake |
| d.sync.end#4 → mtp.dp.row#4 | 0.003 | 0.003 | 0.003 | 0.004 |  |
| mtp.dp.row#4 → mtp.drf.end | 0.050 | 0.095 | 0.047 | 0.212 | the drafter's end (shortlist and carry reset, length check) |
| mtp.drf.end → s.drf.out | 0.001 | 0.001 | 0.001 | 0.002 | back in the server |
| s.drf.out → s.drf.end | 0.000 | 0.000 | 0.000 | 0.000 | the yield's end: the queue's worker thread stops |
| s.drf.end → s.ckpt | 0.018 | 0.018 | 0.013 | 0.021 | draft bookkeeping, the draft context's seq_rm, checkpoints |
| s.ckpt → s.ckgen | 0.000 | 0.000 | 0.000 | 0.001 | the generation checkpoint check |
| s.ckgen → s.bat | 0.002 | 0.002 | 0.002 | 0.002 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.002 | 0.002 | 0.001 | 0.002 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.006 | 0.006 | 0.004 | 0.007 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.001 | 0.001 | 0.000 | 0.001 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.001 | into decode() |
| s.dec → s.dec.y | 0.000 | 0.000 | 0.000 | 0.001 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.001 | 0.001 | 0.001 | 0.001 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.008 | 0.009 | 0.007 | 0.010 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.001 | 0.001 | 0.000 | 0.001 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.001 | 0.001 | 0.000 | 0.001 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.001 | 0.002 | 0.001 | 0.002 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.in | 0.187 | 0.194 | 0.182 | 0.219 |  |
| **host: last d.sync.end → v.ub.in** | 0.284 | 0.335 | 0.271 | 0.476 |  |
| v.ub.in → first main card's graph_compute entry | 0.684 | 0.965 | 0.655 | 1.395 | the scheduler's input copies for that card |
| **the gap, as S37 counts it** | 1.162 | 1.500 | 1.125 | 2.092 |  |

## Between lines

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd.end → next line's s.upd | 0.323 | 0.323 | 0.306 | 0.338 | the line's write, the task loop |
