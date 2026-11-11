# rig/out/trace/step_trace.jsonl

- lines: 186 (7 without a draft, 179 generation steps); drafted 535, accepted 331; drafting passes beyond the drafted tokens 2 (in 1 steps); line writes: median 237 us
- typical steps: 73 of 179 (the commonest of 12 mark sequences among the steps that drafted 3 tokens, 3 drafting passes); the first 3 after each prompt left out

## Waterfall (ms between consecutive marks; the rows add up to the line)

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd → s.post | 0.008 | 0.010 | 0.006 | 0.015 |  |
| s.post → s.shift | 0.001 | 0.001 | 0.001 | 0.001 |  |
| s.shift → s.sel | 0.006 | 0.008 | 0.004 | 0.013 |  |
| s.sel → s.drf | 0.000 | 0.000 | 0.000 | 0.000 |  |
| s.drf → s.drf.in | 0.004 | 0.005 | 0.004 | 0.005 |  |
| s.drf.in → mtp.drf | 0.001 | 0.001 | 0.000 | 0.001 |  |
| mtp.drf → mtp.dp | 0.009 | 0.010 | 0.008 | 0.010 |  |
| mtp.dp → d.dec | 0.011 | 0.011 | 0.010 | 0.014 |  |
| d.dec → d.dec.bal | 0.016 | 0.017 | 0.016 | 0.020 |  |
| d.dec.bal → d.dec.mem | 0.043 | 0.044 | 0.041 | 0.047 |  |
| d.dec.mem → d.dec.out | 0.003 | 0.003 | 0.003 | 0.004 |  |
| d.dec.out → d.ub.apply | 0.074 | 0.157 | 0.050 | 0.285 |  |
| d.ub.apply → d.ub.build | 1.890 | 1.959 | 1.749 | 2.192 |  |
| d.ub.build → d.ub.in | 0.032 | 0.029 | 0.021 | 0.036 |  |
| d.ub.in → d.ub.comp | 0.822 | 0.854 | 0.751 | 0.855 |  |
| d.ub.comp → d.ub.cp | 0.019 | 0.020 | 0.019 | 0.021 |  |
| d.ub.cp → d.dec.end | 0.002 | 0.002 | 0.002 | 0.003 |  |
| d.dec.end → mtp.dp.ret | 0.005 | 0.005 | 0.004 | 0.005 |  |
| mtp.dp.ret → d.sync(6) | 0.001 | 0.001 | 0.001 | 0.002 |  |
| d.sync(6) → d.sync.end(6) | 2.796 | 3.075 | 2.675 | 3.217 |  |
| d.sync.end(6) → mtp.dp.smp | 0.006 | 0.006 | 0.006 | 0.007 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp → d.sync | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync → d.sync.end | 0.015 | 0.015 | 0.015 | 0.015 |  |
| d.sync.end → mtp.dp.row | 0.004 | 0.004 | 0.003 | 0.005 |  |
| mtp.dp.row → mtp.dp#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.dp#2 → d.dec#2 | 0.010 | 0.010 | 0.010 | 0.011 |  |
| d.dec#2 → d.dec.bal#2 | 0.017 | 0.017 | 0.016 | 0.018 |  |
| d.dec.bal#2 → d.dec.mem#2 | 0.043 | 0.043 | 0.040 | 0.045 |  |
| d.dec.mem#2 → d.dec.out#2 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.dec.out#2 → d.ub.apply#2 | 0.040 | 0.040 | 0.038 | 0.042 |  |
| d.ub.apply#2 → d.ub.reuse | 0.015 | 0.015 | 0.014 | 0.016 |  |
| d.ub.reuse → d.ub.in#2 | 0.016 | 0.020 | 0.014 | 0.026 |  |
| d.ub.in#2 → d.ub.comp#2 | 0.752 | 0.821 | 0.680 | 0.832 |  |
| d.ub.comp#2 → d.ub.cp#2 | 0.016 | 0.017 | 0.015 | 0.022 |  |
| d.ub.cp#2 → d.dec.end#2 | 0.001 | 0.001 | 0.001 | 0.002 |  |
| d.dec.end#2 → mtp.dp.ret#2 | 0.005 | 0.005 | 0.004 | 0.006 |  |
| mtp.dp.ret#2 → d.sync(6)#2 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.sync(6)#2 → d.sync.end(6)#2 | 2.941 | 3.133 | 2.738 | 3.234 |  |
| d.sync.end(6)#2 → mtp.dp.smp#2 | 0.004 | 0.004 | 0.003 | 0.005 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp#2 → d.sync#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync#2 → d.sync.end#2 | 0.015 | 0.015 | 0.014 | 0.015 |  |
| d.sync.end#2 → mtp.dp.row#2 | 0.003 | 0.003 | 0.002 | 0.004 |  |
| mtp.dp.row#2 → mtp.dp#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.dp#3 → d.dec#3 | 0.009 | 0.010 | 0.008 | 0.012 |  |
| d.dec#3 → d.dec.bal#3 | 0.013 | 0.014 | 0.012 | 0.017 |  |
| d.dec.bal#3 → d.dec.mem#3 | 0.036 | 0.038 | 0.033 | 0.048 |  |
| d.dec.mem#3 → d.dec.out#3 | 0.002 | 0.002 | 0.002 | 0.003 |  |
| d.dec.out#3 → d.ub.apply#3 | 0.038 | 0.039 | 0.033 | 0.046 |  |
| d.ub.apply#3 → d.ub.reuse#2 | 0.013 | 0.014 | 0.011 | 0.017 |  |
| d.ub.reuse#2 → d.ub.in#3 | 0.019 | 0.020 | 0.013 | 0.027 |  |
| d.ub.in#3 → d.ub.comp#3 | 0.732 | 0.735 | 0.669 | 0.803 |  |
| d.ub.comp#3 → d.ub.cp#3 | 0.016 | 0.016 | 0.015 | 0.017 |  |
| d.ub.cp#3 → d.dec.end#3 | 0.001 | 0.001 | 0.001 | 0.002 |  |
| d.dec.end#3 → mtp.dp.ret#3 | 0.004 | 0.004 | 0.003 | 0.005 |  |
| mtp.dp.ret#3 → d.sync(6)#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync(6)#3 → d.sync.end(6)#3 | 2.825 | 3.118 | 2.735 | 3.232 |  |
| d.sync.end(6)#3 → mtp.dp.smp#3 | 0.003 | 0.003 | 0.003 | 0.004 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp#3 → d.sync#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync#3 → d.sync.end#3 | 0.015 | 0.015 | 0.014 | 0.015 |  |
| d.sync.end#3 → mtp.dp.row#3 | 0.002 | 0.002 | 0.002 | 0.003 |  |
| mtp.dp.row#3 → mtp.drf.end | 0.001 | 0.001 | 0.000 | 0.001 | the drafter's end (shortlist and carry reset, length check) |
| mtp.drf.end → s.drf.out | 0.001 | 0.001 | 0.000 | 0.001 | back in the server |
| s.drf.out → s.drf.end | 0.100 | 0.114 | 0.090 | 0.148 | the yield's end: the queue's worker thread stops |
| s.drf.end → s.ckpt | 0.393 | 0.397 | 0.387 | 0.403 | draft bookkeeping, the draft context's seq_rm, checkpoints |
| s.ckpt → s.ckgen | 0.001 | 0.001 | 0.000 | 0.001 | the generation checkpoint check |
| s.ckgen → s.bat | 0.002 | 0.002 | 0.002 | 0.003 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.002 | 0.002 | 0.001 | 0.002 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.006 | 0.006 | 0.005 | 0.007 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.001 | 0.001 | 0.000 | 0.001 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.001 | into decode() |
| s.dec → s.dec.y | 0.001 | 0.001 | 0.000 | 0.001 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.004 | 0.004 | 0.003 | 0.005 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.005 | 0.005 | 0.004 | 0.006 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.017 | 0.017 | 0.016 | 0.018 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.041 | 0.042 | 0.037 | 0.046 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.010 | 0.011 | 0.009 | 0.014 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.apply | 0.084 | 0.194 | 0.062 | 0.343 | the ubatch, the memory context applied |
| v.ub.apply → v.ub.reuse | 0.069 | 0.069 | 0.061 | 0.078 | graph parameters, the graph reused |
| v.ub.reuse → v.ub.in | 0.672 | 0.633 | 0.459 | 0.764 | the graph's inputs set (host side) |
| v.ub.in → v.ub.comp | 9.970 | 13.987 | 8.201 | 11.939 |  |
| v.ub.comp → v.ub.cp | 0.060 | 0.058 | 0.049 | 0.063 |  |
| v.ub.cp → v.dec.end | 0.004 | 0.004 | 0.004 | 0.005 |  |
| v.dec.end → s.dec.ret | 0.007 | 0.008 | 0.006 | 0.008 |  |
| s.dec.ret → v.sync | 0.001 | 0.001 | 0.000 | 0.001 |  |
| v.sync → v.sync.end | 102.617 | 103.536 | 100.055 | 107.616 |  |
| v.sync.end → s.dec.out | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.dec.out → s.dec.end | 0.136 | 0.121 | 0.092 | 0.143 |  |
| s.dec.end → s.mpd | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.mpd → s.cu.in | 0.003 | 0.003 | 0.003 | 0.004 |  |
| s.cu.in → mtp.cu | 0.002 | 0.002 | 0.002 | 0.002 |  |
| mtp.cu → mtp.cu.dec | 0.270 | 0.270 | 0.261 | 0.278 |  |
| mtp.cu.dec → d.dec#4 | 0.006 | 0.007 | 0.006 | 0.007 |  |
| d.dec#4 → d.dec.bal#4 | 0.020 | 0.021 | 0.019 | 0.021 |  |
| d.dec.bal#4 → d.dec.mem#4 | 0.046 | 0.046 | 0.043 | 0.049 |  |
| d.dec.mem#4 → d.dec.out#4 | 0.006 | 0.007 | 0.006 | 0.007 |  |
| d.dec.out#4 → d.ub.apply#4 | 0.085 | 0.180 | 0.061 | 0.312 |  |
| d.ub.apply#4 → d.ub.build#2 | 0.689 | 0.690 | 0.662 | 0.719 |  |
| d.ub.build#2 → d.ub.in#4 | 0.025 | 0.023 | 0.015 | 0.029 |  |
| d.ub.in#4 → d.ub.comp#4 | 0.811 | 0.841 | 0.761 | 0.897 |  |
| d.ub.comp#4 → d.ub.cp#4 | 0.003 | 0.004 | 0.003 | 0.004 |  |
| d.ub.cp#4 → d.dec.end#4 | 0.003 | 0.003 | 0.003 | 0.004 |  |
| d.dec.end#4 → mtp.cu.ret | 0.005 | 0.005 | 0.004 | 0.006 |  |
| mtp.cu.ret → mtp.cu.end | 0.230 | 0.233 | 0.229 | 0.234 |  |
| mtp.cu.end → s.cu.out | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.cu.out → s.cu.end | 0.124 | 0.112 | 0.084 | 0.137 |  |
| s.cu.end → s.pd | 0.001 | 0.002 | 0.001 | 0.002 |  |
| s.pd → s.acc | 0.002 | 0.002 | 0.001 | 0.002 |  |
| s.acc → s.acc.cl | 0.013 | 0.035 | 0.011 | 0.061 |  |
| s.acc.cl → v.sync(24) | 0.002 | 0.002 | 0.001 | 0.002 |  |
| v.sync(24) → v.sync.end(24) | 0.400 | 0.402 | 0.389 | 0.420 |  |
| v.sync.end(24) → s.acc.smp | 0.001 | 0.001 | 0.001 | 0.002 |  |
| s.acc.smp → s.acc.spec | 0.002 | 0.002 | 0.002 | 0.003 |  |
| s.acc.spec → s.rm | 0.779 | 0.784 | 0.772 | 0.792 |  |
| s.rm → s.tok | 0.006 | 0.006 | 0.004 | 0.007 |  |
| s.tok → s.tok#2 | 0.001 | 0.001 | 0.001 | 0.002 |  |
| s.tok#2 → s.tok#3 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.tok#3 → s.tok#4 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.tok#4 → s.tg | 0.000 | 0.001 | 0.000 | 0.001 |  |
| s.tg → s.upd.end | 0.001 | 0.001 | 0.001 | 0.002 |  |
| **the line (s.upd → s.upd.end)** | 132.594 | 137.320 | 127.940 | 140.222 |  |

## The gap: the last drafting pass's wait → the verify pass's inputs set

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| draft card's work end → d.sync.end (last pass) | 0.336 | 0.337 | 0.331 | 0.344 | the drafted id's way back, the host's wake |
| d.sync.end#3 → mtp.dp.row#3 | 0.002 | 0.002 | 0.002 | 0.003 |  |
| mtp.dp.row#3 → mtp.drf.end | 0.001 | 0.001 | 0.000 | 0.001 | the drafter's end (shortlist and carry reset, length check) |
| mtp.drf.end → s.drf.out | 0.001 | 0.001 | 0.000 | 0.001 | back in the server |
| s.drf.out → s.drf.end | 0.100 | 0.114 | 0.090 | 0.148 | the yield's end: the queue's worker thread stops |
| s.drf.end → s.ckpt | 0.393 | 0.397 | 0.387 | 0.403 | draft bookkeeping, the draft context's seq_rm, checkpoints |
| s.ckpt → s.ckgen | 0.001 | 0.001 | 0.000 | 0.001 | the generation checkpoint check |
| s.ckgen → s.bat | 0.002 | 0.002 | 0.002 | 0.003 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.002 | 0.002 | 0.001 | 0.002 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.006 | 0.006 | 0.005 | 0.007 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.001 | 0.001 | 0.000 | 0.001 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.001 | into decode() |
| s.dec → s.dec.y | 0.001 | 0.001 | 0.000 | 0.001 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.004 | 0.004 | 0.003 | 0.005 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.005 | 0.005 | 0.004 | 0.006 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.017 | 0.017 | 0.016 | 0.018 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.041 | 0.042 | 0.037 | 0.046 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.010 | 0.011 | 0.009 | 0.014 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.apply | 0.084 | 0.194 | 0.062 | 0.343 | the ubatch, the memory context applied |
| v.ub.apply → v.ub.reuse | 0.069 | 0.069 | 0.061 | 0.078 | graph parameters, the graph reused |
| v.ub.reuse → v.ub.in | 0.672 | 0.633 | 0.459 | 0.764 | the graph's inputs set (host side) |
| **host: last d.sync.end → v.ub.in** | 1.483 | 1.502 | 1.289 | 1.759 |  |
| v.ub.in → first main card's graph_compute entry | 5.610 | 9.310 | 4.132 | 7.839 | the scheduler's input copies for that card |
| **the gap, as S37 counts it** | 7.512 | 11.150 | 5.938 | 9.563 |  |

## Between lines

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd.end → next line's s.upd | 0.257 | 0.258 | 0.248 | 0.268 | the line's write, the task loop |
