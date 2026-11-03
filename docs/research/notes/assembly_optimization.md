# Assembly-level optimization in LLM inference, and where it could pay off on Pascal (as of 2026-11-03)

Scope: a broad first pass, from primary sources, on hand-written or assembler-driven code below CUDA C/C++ (SASS
assemblers, inline PTX, LUT-based GEMM without tensor cores, CPU intrinsics/JIT) for LLM inference, and what of it
applies to a 5-SM NVIDIA P106-090 (Pascal, compute 6.1, no tensor cores, dp4a, ~173 GB/s). The setting: Qwen3.8-Flash-Next
(512-expert hybrid MoE) on nine P106-090s, host a 2-core Celeron 3865U (SSE4.2 only, no AVX, 4 GB RAM). Registry row
Q613 parked hand-written assembly here on the memory-bound argument; this note is the breadth pass that could reopen
or further park it, read together with [kquant_decode_pascal.md](kquant_decode_pascal.md),
[pascal_mmvq_integration.md](pascal_mmvq_integration.md), [quant_speed_b_formats_kernels.md](quant_speed_b_formats_kernels.md)
and [host_decode_loop.md](host_decode_loop.md).

Conventions:
- **(est.)** marks this note's own arithmetic or inference, never measured on this rig.
- **P**/**S** mark primary (repo, doc, paper, vendor release note) and secondary (blog, forum, write-up) sources.
- Quotes are kept under 15 words.
- Nothing here was run: no `cuobjdump`, no build, no rig or quant-machine access. Every SASS/stall-counter claim from
  this project's own binaries is carried over from the linked notes, not reproduced here.

## 1. Summary

1. **The card's own numbers already say where assembly could matter, and it is narrow.** The linked notes measure
   one-token decode as instruction-issue-bound, not memory-bound, on this 5-SM card: 3.2–8.4 SASS lane-instructions
   per weight byte against a ~6.4-per-byte issue ceiling, address arithmetic and narrow loads dominating over dp4a
   ([quant_speed_b_formats_kernels.md](quant_speed_b_formats_kernels.md) §1, §3). The task's framing of one-token
   decode as "memory-bound" (BF16 ~82 %, Q2_0 ~55 % of bandwidth) describes headroom *after* the Pascal-tuned C++
   kernels of [pascal_mmvq_integration.md](pascal_mmvq_integration.md) close most of that issue gap; Q613's own
   parking argument is about that same post-kernel state. Verify passes at k ≥ ~7 (Q2_0) or k ≥ ~13 (BF16) columns
   are compute-bound by this project's own model, which is exactly the regime every source below says hand assembly
   can still win something.
2. **No maintained SASS assembler targets Pascal with a track record on it.** maxas (Scott Gray, 2014) is a Maxwell
   assembler; its wiki and introduction name only Maxwell chips (GM107/GM204), archived 2023-01-03, Perl-based
   [[P](https://github.com/NervanaSystems/maxas)]. NVIDIA's own 8.0.61.2 cuBLAS release notes credit Gray for
   Pascal GEMM kernels, but that is NVIDIA's internal use of his method, not evidence the public maxas tool runs on
   sm_61 [[S, secondary paraphrase](https://github.com/stefan20162016/maxas-explained)]. **CuAssembler** (Yang,
   2020s) is the one maintained, actively scoped tool: its README states it "currently supports
   Pascal/Volta/Turing/Ampere instruction set (SM60/61/70/75/80/86/...)" but also calls itself "still in its
   infancy," with no correctness checker and no published kernel built with it
   [[P](https://github.com/cloudcores/CuAssembler)]. **nbassembler** (QianyanTech) lists Pascal SM60/61/62 among
   Maxwell through Ampere, is MIT-licensed with round-trip tests, but ships no benchmarks or examples
   [[P](https://github.com/qianyantech/nbassembler)]. **KeplerAs** (Zhang et al., PPoPP '17) targets Kepler only,
   reusing maxas's front end, and reported 3.1 TFLOP/s SGEMM on a K20m, ~88 % efficiency, ~15 % over cuBLAS 7.0 on
   that architecture — evidence the *method* transfers across Nvidia generations, not that any tool is Pascal-ready
   out of the box [[P, paper](https://static.aminer.org/pdf/20170130/pdfs/ppopp/c7hwcafxtjlpqeesb0wl926oskzvr1di.pdf)].
   turingas targets Volta/Turing/Ampere only, no Pascal [[P](https://github.com/daadaada/turingas)].
3. **Maxwell/Pascal's control-code model is public and well documented, from maxas's own wiki.** Each 64-bit control
   word covers three instructions: a 4-bit stall count (0–15 cycles), a 1-bit yield hint, a 3-bit write barrier and a
   3-bit read barrier (6 barriers total, shared pool), a 6-bit wait-barrier mask, and 4 register-reuse-cache bits per
   instruction [[P](https://github.com/NervanaSystems/maxas/wiki/Control-Codes)]. A stall of 0 dual-issues the next
   instruction when it uses a different functional unit; ISETP needs a stall of 13 to read its own predicate;
   variable-latency ops (memory, S2R) use barriers instead of stall counts
   [[P](https://github.com/NervanaSystems/maxas/wiki/Control-Codes)]. An NVIDIA forum thread independently confirms
   the brace-delimited dual-issue grouping appears in `cuobjdump -sass` output for cc 5.0/6.0/6.1 alike, and that the
   mechanism is otherwise undocumented by NVIDIA [[S](https://forums.developer.nvidia.com/t/in-sass-output-generated-for-cc5-0-cc6-0-and-cc6-1/111105)].
4. **Register-bank conflicts are real and ptxas does not resolve them well, per Gray's own account.** Maxwell has 4
   register banks (register number mod 4); Gray found no FFMA ordering that eliminates every conflict, reached 98 %
   of theoretical FFMA throughput on GM107 and 96 % on GM204 with a hand-chosen "swirl" operand order that gets 47 %
   reuse-cache hits (against ~4/8 bytes, i.e. 50 %, for a naive loop) and beat cuBLAS by up to 25 % on small matrices
   and 4.8 % at 4096×4096 [[P, wiki](https://github.com/NervanaSystems/maxas/wiki/SGEMM)]. He states ptxas "badly
   manages register usage" and "does a poor job of hiding memory latency," and that register-bank placement for
   memory operands is not exposed to the compiler at all [[P](https://github.com/NervanaSystems/maxas/wiki/SGEMM)].
   This is architecture-general evidence for *dense FFMA-bound* kernels (SGEMM); it says nothing about whether a
   5-SM Pascal dp4a matrix-vector kernel has comparable headroom, because that kernel is address-arithmetic- and
   load-bound, not FFMA-bound ([quant_speed_b_formats_kernels.md](quant_speed_b_formats_kernels.md) §3).
5. **DeepSeek's FFMA SASS interleaving is the closest documented precedent to a worthwhile cubin-level hack, and it
   is binary-patching, not assembly from scratch.** DeepGEMM's team noticed a yield-bit difference between NVCC
   12.2 and 12.3 compiles of the same CUTLASS FP8 kernel, wrote a script to flip the yield (and paired reuse) bit on
   compiled FFMA instructions directly in the cubin, and reported a 10%+ gain on fine-grained-scaling FP8 GEMM by
   overlapping more MMA with promotion FFMA [[S/P mix, repo + HN discussion, unverified exact figure](https://github.com/deepseek-ai/DeepGEMM)].
   NVCC 12.9 automated the same interleaving, and DeepGEMM's own README says the hand version is no longer needed
   from that compiler on [[P](https://github.com/deepseek-ai/DeepGEMM)]. This is Hopper (sm_90a) tensor-core work;
   it has no public Pascal analogue, but the *shape* of the win — a compiler leaves one scheduling bit on the table,
   a tool flips it in the cubin — is a cuobjdump/CuAssembler-reachable pattern this project could try narrowly on an
   issue-bound Pascal kernel, if a profile ever shows a specific stall-count or yield choice costing more than a few
   percent.
6. **Inline PTX intrinsics (`lop3`, `prmt`/`__byte_perm`, `dp4a`) are already how llama.cpp and the fork decode
   low-bit weights on Pascal — this is not a frontier to open, it is the path already taken.**
   `vecdotq.cuh`'s `get_int_from_table_16` already uses 8 `__byte_perm` calls per 32-bit word to do a 16-entry table
   lookup for IQ4_XS/IQ4_NL [P, this repo's own reading]; Marlin's and FasterTransformer's `lop3`-based nibble-to-fp16
   dequant (the "magic constant" trick: OR a nibble into an fp16 exponent field, then one `sub`/`fma` removes the
   bias, no int→float conversion) is the general version of the same idea
   [[P](https://github.com/IST-DASLab/marlin), [P](https://github.com/NVIDIA/FasterTransformer)]; this project's own
   SASS counts already attribute only 5–8 % of the decode loop to dp4a itself, with 41–59 % in address arithmetic
   ([quant_speed_b_formats_kernels.md](quant_speed_b_formats_kernels.md) §1). More inline-PTX dequant tricks would
   compete for a slice of a loop that is already dominated by addressing, not unpacking.
7. **LUT-based mixed-precision GEMM (T-MAC, LUT-GEMM, FLUTE, bitnet.cpp) targets CPUs or tensor-core GPUs, not a
   dp4a GEMV on 5-SM cards, and the GPU variants hit the same shared-memory bank-conflict wall this project would
   also hit.** T-MAC is CPU-only (ARM/x86), up to 4× over llama.cpp on BitNet-class models on an M2-Ultra/Raspberry Pi
   [[P](https://arxiv.org/abs/2407.00088)]; bitnet.cpp's TL1/TL2 kernels are the same CPU lineage, up to 6.25× on edge
   CPUs for ternary (BitNet b1.58) [[P](https://arxiv.org/abs/2502.11880)]; LUT-GEMM and FLUTE are GPU kernels but
   built and measured for A100-class cards at low-bit widths (2.1× over a dequant baseline for LUT-GEMM's 3-bit
   OPT-175B; 2–4× for FLUTE at batch < 32) [[P](https://arxiv.org/abs/2206.09557), [P](https://arxiv.org/abs/2407.10960)].
   A 2025 accelerator paper (FIGLUT) explicitly names the obstacle a Pascal attempt would hit: LUT-GEMM's shared-memory
   table avoids bank conflicts while building the table but hits them "frequently" during lookup, because "the
   randomness of the weight pattern" scatters reads across banks [[P](https://arxiv.org/pdf/2503.06862)]. No source
   found gives a measured LUT-vs-dequant GEMV comparison on a CUDA-core-only, non-tensor-core GPU; this is this
   note's own best negative finding, not something a source states directly.
8. **dp4a is already "full throughput, as advertised" on Pascal, and the architecture problem it has is a
   different one: fp16 compute, not int8.** NVIDIA's Pascal Tuning Guide states `dp4a`/`dp2a` "offer a throughput
   equal to that of FP32 arithmetic" on GP104, and that GP104 (6.1) has fp16 math throughput at 1/64 of FP32 — GP100
   (6.0) reaches 2× FP32 for fp16 [[P](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html)]. A separate,
   corroborated forum report adds a compiler-specific trap: ptxas can choose `HADD2` over `F2F` for an fp16→fp32
   load-convert on sm_61, which the reporter measured as 16× slower than the straightforward path, and had already
   filed as a bug [[S, single-user report with NVIDIA engagement](https://forums.developer.nvidia.com/t/nvidia-pascal-titan-xp-titan-x-geforce-gtx-1080-ti-gtx-1080-gtx-1070-gtx-1060-gtx-1050-gt-1030/42660)].
   This is a concrete, checkable "is ptxas doing something dumb on our cubin" question, cheaper than full assembly:
   a `cuobjdump -sass` grep for `HADD2` immediately after any fp16 load-and-convert in the fork's kernels.
9. **Register-bank conflicts, control codes and reuse caches (items 3–4) matter most for dense FFMA-bound SGEMM-style
   kernels; this project's decode kernels are a different animal (dp4a GEMV, address-bound).** The maxas precedent
   (item 4) is strong evidence that hand control of banks and scheduling buys several percent on a *saturated* FFMA
   pipeline. Flint's verify-pass k-column kernels, once compute-bound above ~7 (Q2_0) or ~13 (BF16) columns per this
   project's own model, are the regime where that argument starts to apply on this card; single-token decode, still
   address- and load-bound per the same notes, is not.
10. **On the CPU side, no result found changes the host's role.** llamafile's tinyBLAS reports 30–500 % prompt-eval
    gains over llama.cpp, biggest on ARMv8.2+, AVX-512 and Alder Lake-class x86 — all instruction sets the Celeron
    3865U lacks (SSE4.2 only, no AVX/AVX2/FMA/F16C, confirmed against multiple spec listings)
    [[S, press coverage of the author's post](https://hacks.mozilla.org/2024/04/llamafiles-progress-four-months-in/)].
    ik_llama.cpp's fast CPU GEMM (`iqk_mul_mat`) requires AVX2 or ARM NEON as a floor and explicitly will not fix
    AVX-only issues [[P](https://github.com/ikawrakow/ik_llama.cpp)]. KleidiAI's micro-kernels target ARM
    NEON/SVE/SME [[P](https://github.com/ARM-software/kleidiai)]. oneDNN is deprecating SSE4.1/AVX support entirely
    [[P, repo note](https://github.com/uxlfoundation/oneDNN)]. None of this touches the host's actual bottleneck,
    which per [host_decode_loop.md](host_decode_loop.md) is launch/copy/server-path overhead (est. 2–30 % of a
    token), not SIMD throughput on the one 8 KB embedding row the CPU still computes — that row is cheap however it
    is coded.
11. **Compiler flags reach much of what hand assembly would, at a fraction of the cost, and are the right first
    experiment regardless of what else is tried.** `-Xptxas -dlcm=ca` forces L1 caching of global loads on GP104,
    which defaults to L2-only on consumer Pascal (unlike GP100); `-maxrregcount` / `__launch_bounds__` trade
    occupancy against register pressure, directly relevant to the fused gate/up kernels in
    [pascal_mmvq_integration.md](pascal_mmvq_integration.md) that already hit 254–255 registers
    [[P](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html)]. None of the sources above give a measured
    `-dlcm=ca` number on an MMVQ-style kernel; it is this project's own cheapest untried lever.
12. **Tooling for finding instruction-bound kernels on Pascal is usable but dated.** `cuobjdump -sass` and
    `nvdisasm` still accept `sm_61`/`SM61` through CUDA 12.9's binary utilities, confirmed directly from the current
    docs [[P](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)] — this is exactly the tool this
    project's own notes already use. Nsight Compute dropped Pascal ("Removed support for the Pascal SM 6.x GPU
    architecture") in its 2020.1 release; the last version with Pascal support is community-reported as 2019.5 on
    Linux [[P, release notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html); S for the exact last
    version]. `nvprof`/Visual Profiler are deprecated (not yet confirmed removed) as of CUDA 12.8's profiler
    changelog and are the NVIDIA-recommended fallback for pre-Volta cards
    [[P](https://docs.nvidia.com/cuda/archive/12.4.1/profiler-users-guide/changelog.html)]. NVBit (binary
    instrumentation, dynamic instruction/opcode counting) states Pascal support directly and lists a compute-capability
    floor of 3.5, which covers sm_61 [[P](https://research.nvidia.com/publication/2019-10_nvbit-dynamic-binary-instrumentation-framework-nvidia-gpus)].
    Accel-Sim's validated hardware set includes Pascal, with roughly 15 % error against real hardware, run through
    its NVBit-based tracer [[P, paper](https://engineering.purdue.edu/tgrogers/papers/khairy.isca2020.pdf)]. This
    project already uses the one tool proven current (`cuobjdump -sass`) and already found that Nsight-style timing
    tools distort these cards (12 CUDA events per token cut S4-L0 from 37.4 to 23.4 tok/s per
    [host_decode_loop.md](host_decode_loop.md)) — a second, independent reason beyond Nsight Compute's 2020 Pascal
    drop to keep relying on static SASS counts plus NVBit-style instrumentation instead of a live profiler.
13. **Triton has no Pascal path; this closes off the "autotuned Triton kernel instead of hand SASS" alternative.**
    Upstream Triton lists "Compute Capability 7.0+" as its supported hardware floor, and a user hitting a GTX 1080 Ti
    through `torch.compile` got an explicit rejection from the Triton GPU compiler
    [[P, repo README](https://github.com/openai/triton), S for the forum report]. `tl.dot`'s tensor-core path is
    Volta+ by design. No community fork was found that restores a validated Pascal `tl.dot` path. TVM/autotuning was
    not run down to the same depth in this pass (see subtopics).

## 2. GPU, NVIDIA SASS

### 2.1 Assemblers and architecture coverage

| Tool | Author/org | Architectures claimed | Status | Pascal evidence |
|---|---|---|---|---|
| maxas | Scott Gray (Nervana) | Maxwell (its own wiki names only Maxwell/GM107/GM204) | Archived 2023-01-03, Perl, read-only | NVIDIA's cuBLAS 8.0.61.2 notes credit Gray for Pascal small-tile GEMM kernels, but as NVIDIA's internal engineering, not a statement that the public tool runs on sm_61 |
| KeplerAs | Zhang et al. (PPoPP '17) | Kepler, reusing maxas's front end | Public release referenced, paper-grade | None; Kepler only |
| CuAssembler | Yang (cloudcores) | Pascal/Volta/Turing/Ampere, "SM60/61/70/75/80/86/..." per its own README | "Still in its infancy," no correctness checker | The only maintained tool that names sm_61/SM61 explicitly; no published kernel example found |
| nbassembler | QianyanTech | Maxwell SM50/52/53, Pascal SM60/61/62, Volta, Turing, Ampere | MIT, round-trip tests, no benchmarks | Explicit SM61 in its support table; no examples |
| turingas | daadaada (HKBU-HPML) | Volta/Turing/Ampere (SM70/75/80) | Public, PPoPP '21-adjacent | Explicitly excludes Pascal |

Practical reading: if this project ever wants to hand-edit sm_61 SASS, **CuAssembler or nbassembler** are the only
two candidates that claim the target at all, and neither has a demonstrated kernel to point to — the first real
experiment would be reproducing a known-good cubin round-trip on one of the fork's own kernels before trusting either
tool for an edit that changes behavior.

### 2.2 Control codes on Maxwell/Pascal

maxas's own wiki is the primary documentation, confirmed independently by an NVIDIA forum thread discussing the same
mechanism in `cuobjdump -sass` output for cc 5.0/6.0/6.1
[[P](https://github.com/NervanaSystems/maxas/wiki/Control-Codes), [S](https://forums.developer.nvidia.com/t/in-sass-output-generated-for-cc5-0-cc6-0-and-cc6-1/111105)]:
- One 64-bit control word per three instructions; 17 control bits plus 4 reuse bits per instruction.
- Stall count: 4 bits (0–15 cycles). Yield hint: 1 bit. Write/read dependency barriers: 3 bits each, 6 shared barrier
  slots. Wait-barrier mask: 6 bits (one per barrier, maskable together).
- A stall of 0 lets the next instruction dual-issue, but only across different functional units (e.g. a CUDA core
  and a memory unit) — this matches njuffa's reading of the brace-delimited SASS groups as "dual issue from the same
  warp," which he called undocumented and speculative, on the NVIDIA forum thread above.
- Fixed latencies: most ALU ops 6 cycles; ISETP 13 cycles (reading its own predicate needs stall ≥ 13); S2R ~20
  cycles; global memory operand read 4 cycles, shared memory 2 cycles; branches/calls/exit need stall ≥ 5.
- Predicated instructions ignore barrier flags (barrier use needs an actual branch instead); stores cannot set a
  write barrier; sharing one barrier across multiple S2R instructions is unsafe since they can finish out of order.

The Pascal Tuning Guide independently confirms the architectural motivation: GP104 (6.1) has four warp schedulers
per SM managing 128 FP32 cores, and states "single-issue is now sufficient to fully utilize all CUDA Cores" — i.e.
dual issue is still legal and encoded the same way, but Pascal needs it less than Maxwell did for raw FFMA throughput
[[P](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html)]. This matters for how much control-code tuning can
still buy on a P106: a kernel bound by address arithmetic and narrow loads (this project's own measured case) is not
the FFMA-saturated case the guide is describing.

### 2.3 Register-bank conflicts

Maxwell/Pascal have 4 register banks (bank = register number mod 4). Gray's SGEMM write-up is the primary account:
no FFMA operand ordering removes every conflict; his C-register layout (R0–63) plus a "swirl" scan order for the
blocking registers (R64–95) avoids every conflict involving C registers and leaves 16 unavoidable conflicts among the
blocking registers alone, which he estimates would cost ~20 % without the reuse cache helping, though disabling reuse
in practice only cost "a couple hundred Gflops," which he attributes to operand collectors and thread-level
parallelism without confirming it [[P](https://github.com/NervanaSystems/maxas/wiki/SGEMM)]. Each operand slot has an
8-byte reuse cache; a straight loop reaches only 4 of 8 bytes (50 % reuse), the swirl order reaches 47 %. ptxas, per
Gray, "handles bank conflicts poorly, especially with vector instructions" and does not expose register-bank choice
for memory operands at all.

### 2.4 Known results

- **Scott Gray's maxas SGEMM** (Maxwell): ~98 % of theoretical FFMA throughput on GM107, ~96 % on GM204; up to 25 %
  faster than cuBLAS on small matrices, 4.8 % at 4096×4096 [[P](https://github.com/NervanaSystems/maxas/wiki/SGEMM)].
- **maxDNN / Nervana Winograd convolution** (Maxwell, built on maxas): Lavin & Gray's F(2×2,3×3) Winograd reached
  2.26× over cuDNN v3 at batch 1 and 1.48× at batch 64 on a Titan X for VGG-E fp32, peaking at 9.49 TFLOPS against
  cuDNN's 6.34 [[P, paper](https://arxiv.org/abs/1509.09308)]; a later Nervana blog post claims 2–3× over cuDNN v4
  on VGG-class 3×3 layers [[S, vendor blog](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Not-so-fast-FFT-Winograd/post/1335707)].
- **KeplerAs SGEMM** (Kepler K20m): ~3.1 TFLOP/s, ~88 % efficiency, ~15 % over cuBLAS 7.0
  [[P, paper](https://static.aminer.org/pdf/20170130/pdfs/ppopp/c7hwcafxtjlpqeesb0wl926oskzvr1di.pdf)].
- **DeepSeek DeepGEMM FFMA SASS interleaving** (Hopper, sm_90a): a cubin-patching script flips the yield bit (and
  paired reuse bit) on compiled FFMA instructions; reported 10%+ on fine-grained-scaling FP8 GEMM, obsoleted by NVCC
  12.9's automatic interleaving per the project's own README [[P + S mix](https://github.com/deepseek-ai/DeepGEMM)].
  No official DeepSeek benchmark isolating the interleaving alone was found; treat the 10 % figure as a secondary
  (HN-sourced) claim.
- **DeepSeek FlashMLA** (Hopper/Blackwell): ~3000 GB/s memory-bound and up to 660 TFLOPS compute-bound on H800,
  reported by the project itself; no instruction-level (SASS) detail was found in primary sources for this pass
  [[S, press + repo paraphrase](https://itc.ua/en/news/bypassing-sanctions-deepseek-flashmla-improves-the-performance-of-nvidia-h800-ai-chips-by-8-times/amp)].
  None of this is Pascal-relevant hardware; it is listed because the task asked for it as a known result.

## 3. GPU, inline PTX in CUDA C

- **`lop3.b32`**: a 3-input LUT instruction (`(a&b)|c` or any 3-input boolean function via an 8-bit immediate LUT).
  Marlin's 4-bit dequant uses `lop3` to OR a 4-bit nibble directly into an fp16 exponent field (e.g. mask `0x000f000f`
  and LUT constant `0x64006400` to synthesize "1024+n" in fp16), then one packed `sub.f16x2`/`fma.f16x2` removes the
  bias and the zero-point in one instruction, with **no int-to-float conversion instruction at all**
  [[P](https://github.com/IST-DASLab/marlin)]. FasterTransformer's `FastInterleavedAndBiasedNumericArrayConverter`
  generalizes this: uint8→fp16 in ~1 instruction/element via `prmt` + `sub.f16x2`; int4→fp16 in ~1.1 via `lop3` +
  `sub`/`fma`; the bf16 paths need ~2.5 and ~1.4 instructions/element respectively and are gated to sm_80+ because
  bf16 lacks the mantissa bits for the same trick [[P](https://github.com/NVIDIA/FasterTransformer)]. This is the
  generalized "lop3 dequant" trick the task names, and it is architecture-general PTX (no sm_80 gate on the fp16
  path), so it should assemble for sm_61.
- **`prmt`/`__byte_perm`**: already load-bearing in this project's own kernels. llama.cpp's `get_int_from_table_16`
  uses 8 `__byte_perm` calls per 32-bit index word to implement a 16-entry table lookup for IQ4_XS/IQ4_NL [P, this
  repo's reading of `vecdotq.cuh`]; the fork's own Pascal kernels use `PRMT` for 2-byte-aligned field extraction and
  for Q2_0's even/odd 2-bit unpacking (`pascal_mmvq_integration.md`, `quant_speed_b_formats_kernels.md`).
- **`dp4a`**: NVIDIA's own tuning guide states it runs at "throughput equal to that of FP32 arithmetic" on GP104, and
  this project's SASS counts put it at only 5–8 % of the decode loop — dp4a itself was never the bottleneck
  [[P](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html)].
- **Cache hints** (`ld.global.nc`, `.cg`/`.cs`/`.lu`/`.cv`, `-Xptxas -dlcm=ca`): GP104 defaults to L2-only caching of
  global loads (unlike GP100, which caches in L1/texture by default); `-Xptxas -dlcm=ca` opts a Pascal build into L1
  caching for all global loads, at a possible occupancy cost the driver can veto
  [[P](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html)]. `L1::no_allocate`-style PTX qualifiers are a
  newer (post-Pascal) addition; this pass could not confirm their minimum sm target from NVIDIA's own PTX ISA pages
  (the relevant PTX ISA section was not reachable in full during this search) — flagged as a subtopic.
- **`bfe`/`bfi`**: present in the Maxwell/Pascal SASS instruction set per CUDA 12.9's binary-utilities instruction
  table [[P](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-binary-utilities/index.html)]; not seen used directly
  in the llama.cpp dequant paths read for this pass (they favor mask+shift and `prmt`/LUT instead).
- **What applies without tensor cores**: all of the above (lop3, prmt, dp4a, cache-hint loads) are CUDA-core/ALU and
  load-path instructions, not tensor-core instructions, so none of them are gated by the lack of tensor cores on
  Pascal. The things that *do* need tensor cores and so do not apply — `mma.sync`, `ldmatrix`, `cp.async` with
  `createpolicy` cache hints for the async pipeline, and the Marlin/FasterTransformer bf16 dequant path (sm_80+) —
  were also catalogued from the Marlin and FasterTransformer source during this pass, as the "what to skip" list.

## 4. Compute-bound low-bit matvec without tensor cores: LUT-GEMM

| Project | Hardware | Claimed gain | Applicability to Pascal dp4a decode |
|---|---|---|---|
| T-MAC | CPU (ARM NEON, x86), M2-Ultra, Raspberry Pi 5 | up to 4× over llama.cpp on low-bit/BitNet models; up to 70 % less energy | None directly — CPU-only, and this project's host (Celeron, no AVX) is far below T-MAC's target CPUs |
| bitnet.cpp (TL1/TL2) | Edge CPUs (x86, ARM) | up to 6.25× over state-of-the-art edge baselines for ternary BitNet b1.58, lossless | Same: CPU-only, ternary-model-specific |
| LUT-GEMM | "a single GPU" (unnamed in the abstract) | 2.1× over OPTQ (dequant-based) for OPT-175B 3-bit | GPU, but tensor-core-class hardware in practice; bank-conflict problem noted below |
| FLUTE | Unspecified GPU, batch < 32, group size 128 | 2–4× over existing GEMM kernels; 1.5–2× end-to-end on LLaMA3 with NF4-style LUTs | Same caveat; 3-bit/non-power-of-2 bit widths are its niche, not Pascal-shape dp4a decode |
| LUT Tensor Core, FIGLUT | Proposed/simulated accelerator hardware | N/A (hardware co-design, not a software kernel) | Not applicable to existing silicon |

**The bank-conflict problem is the reason this project's own SASS-measured dp4a kernels likely still beat a naive
LUT-GEMV on Pascal.** FIGLUT's own framing of LUT-GEMM states shared-memory bank conflicts are frequent during the
lookup phase "because the randomness of the weight pattern" scatters accesses across banks
[[P](https://arxiv.org/pdf/2503.06862)]; CodeQuant reports the same obstacle independently, from simulation
[[P](https://arxiv.org/pdf/2604.10496)]. No source in this pass gives a measured LUT-GEMV-vs-dp4a-GEMV comparison on
a CUDA-core-only GPU — not T-MAC/bitnet.cpp (CPU), not LUT-GEMM/FLUTE (A100-class, tensor-core-era GPUs). **Verdict
for this setting (est.):** a LUT-based kernel is not a promising path for Pascal MMVQ decode without a dedicated,
rig-measured prototype first; it would need a conflict-avoiding table layout (e.g. per-warp table replication across
banks) that none of the surveyed GPU papers demonstrate on pre-Ampere hardware. The project's own survey
([quant_speed_b_formats_kernels.md](quant_speed_b_formats_kernels.md) item 12) reached the same "T-MAC is CPU-only"
conclusion independently.

## 5. CPU

| Project | Hand-tuning | ISA floor | Measured gain | Reaches this host? |
|---|---|---|---|---|
| llamafile tinyBLAS | 84 hand-written matmul kernels (C++/intrinsics, per the author's post) | AVX2/AVX-512/ARM NEON/dotprod for the big gains | 30–500 % prompt-eval vs llama.cpp; up to 10× on AVX-512 Zen4 | No — Celeron 3865U has SSE4.2 only, none of AVX/AVX2/AVX-512/FMA/F16C |
| ggml's own quant kernels | Intrinsics per ISA, dispatched via `GGML_CPU_ALL_VARIANTS` | SSE4.2 is a listed variant floor; AVX2/FMA/F16C are separate opt-in flags | Not isolated in this pass | Partially — the SSE4.2 variant exists and would be what this host's build selects |
| ik_llama.cpp (`iqk_mul_mat`) | Hand-tuned CPU GEMM, row-interleaved `_R4/_R8` quant types | "AVX2 or better, ARM_NEON or better" stated as the only fully functional floor; AVX-only is explicitly unsupported | "Much faster CPU prompt processing," no README-stated number | No — README states AVX-only issues won't be fixed |
| KleidiAI | Arm micro-kernels (C, per-feature files, not asm) | NEON/SVE/SME/SME2, FEAT_DotProd, FEAT_I8MM | Not given in the repo page itself | No — x86 host |
| oneDNN (xbyak JIT) | JIT-generated x86 assembly at runtime via xbyak | Deprecating SSE4.1/AVX support entirely | Not isolated in this pass | No — actively dropping this host's tier |

**What is possible on the Celeron, and whether it matters:** every hand-tuned CPU GEMM/dequant project surveyed sets
its useful floor at AVX2 or NEON; the Celeron 3865U has neither. The ggml SSE4.2 dispatch variant is the only one
that would even compile correctly for this host, and no source gives its throughput relative to AVX2. But per
[host_decode_loop.md](host_decode_loop.md), the host's CPU-side compute per token is one 8 KB embedding-row lookup —
not a GEMM — and the host's actual measured costs are launch/copy/synchronization/server-path overhead, not
arithmetic throughput. **Hand-tuned CPU kernels, with or without SSE4.2-specific assembly, are not a lever here**;
this matches [host_decode_loop.md](host_decode_loop.md)'s own ranked list, which contains no CPU-SIMD item.

## 6. Tools for finding instruction-bound kernels

| Tool | Pascal (sm_61) support | Notes |
|---|---|---|
| `cuobjdump -sass` / `nvdisasm` | Yes, through CUDA 12.9 (`sm_61`/`SM61` listed explicitly) | This project's own method in [quant_speed_b_formats_kernels.md](quant_speed_b_formats_kernels.md); confirmed current |
| Nsight Compute | No — "Removed support for the Pascal SM 6.x GPU architecture" in release 2020.1; last Pascal-supporting release community-reported as 2019.5 (Linux) | Matches this project's own experience (Nsight/nvprof refuse the P106-090, per [host_decode_loop.md](host_decode_loop.md) E2.2) |
| nvprof / Visual Profiler | Deprecated (CUDA 12.8 changelog), NVIDIA's own recommended fallback for pre-Volta cards, not yet confirmed fully removed | Stall-reason metrics (`stall_exec_dependency`, `stall_pipe_busy`, `stall_inst_fetch`, `stall_memory_throttle`, `stall_not_selected`, `stall_other`) exist in nvprof's metric set; this project found CUDA timing events themselves distort these cards' timing (37.4 → 23.4 tok/s with 12 events/token) |
| NVBit | Yes — compute-capability floor 3.5, explicitly names Kepler/Maxwell/Pascal/Volta support | Dynamic binary instrumentation; could give true dynamic instruction/opcode counts in place of distorting timing tools |
| Accel-Sim (NVBit-based tracer + simulator) | Yes — Pascal is in its validated hardware set, ~15 % correlation error | Research-grade; heavier setup than a profiler, but trace-driven so it does not perturb the running kernel's timing the way CUDA events do |

**Stall-reason counters on compute 6.1**: nvprof's documented stall categories (instruction fetch, exec dependency,
memory dependency, texture, sync, constant memory dependency, pipe busy, memory throttle, not selected, other) are
architecture-general in nvprof's own metric reference; this pass found no compute-6.1-specific restriction on which
apply, but also found no confirmation nvprof runs cleanly on a P106-090 today versus its listed cc8.0+ limitation
note for Visual Profiler/nvprof broadly — this is worth a direct check on the quant machine (not run in this
read-only pass) before relying on it.

**Practical conclusion**: this project is already using the one tool combination that is both current and
non-perturbing (static `cuobjdump -sass` plus its own path-counting arithmetic). NVBit is the one gap — it would
give *dynamic* per-kernel instruction/stall-reason counts (confirming which basic blocks actually execute at Flint's
shapes) without the CUDA-event timing distortion this project already measured.

## 7. Alternatives that reach similar code without assembly

| Lever | Pascal status | When hand assembly would still win |
|---|---|---|
| Triton | No Pascal path. Upstream states "Compute Capability 7.0+"; a GTX 1080 Ti user got an explicit compiler rejection under `torch.compile`. `tl.dot`'s tensor-core path is Volta+ by design | N/A on sm_61 — not a live alternative here |
| TVM / autotuning | Not run down in this pass (see subtopics) | — |
| `-Xptxas -dlcm=ca` | Confirmed applicable: opts GP104 into L1 caching of global loads, off by default (unlike GP100) | Wins nothing hand assembly couldn't also reach, but costs a compile flag instead of a rewrite — try first |
| `-maxrregcount` / `__launch_bounds__` | Confirmed applicable; Pascal's 255-register/thread ceiling and 64K-register/SM file are the same limits the fork's fused gate/up kernels already hit (254–255 registers) | Compiler-level register control is the cheap version of maxas's hand allocation; worth exhausting before any hand SASS edit |
| `--use_fast_math` | Standard nvcc flag; not specifically evaluated against this project's kernels in this pass | Irrelevant to integer dp4a decode; mainly affects transcendental/division-heavy FP code |
| ptxas's documented weaknesses (register-bank placement, FFMA/memory-op interleaving, double-buffer optimization) | Gray's account (maxas wiki) is architecture-general, not Pascal-specific, but nothing found contradicts it for sm_61 | Hand assembly wins specifically where ptxas's *documented* blind spots (bank placement for memory operands, killing double-buffered shared-memory loads) bite a *saturated* FFMA or load pipeline — not an address-arithmetic-bound integer kernel, which is a different bottleneck ptxas does not mis-schedule the same way |

## 8. Where it could pay off here

Ranked by est. gain × (low effort, low risk) for this rig, each with the measurement that would confirm it first.
All "gain" figures are est., scoped to single-token decode or k-column verify unless stated, and compared against
this project's *already-ported* Pascal C++ kernels (not stock llama.cpp), since that is the baseline any further
hand-tuning competes against.

1. **`-Xptxas -dlcm=ca` and a register-budget sweep (`-maxrregcount`/`__launch_bounds__`) on the existing Pascal
   MMVQ kernels.** Est. gain: low-to-moderate, unquantified by any source for this exact kernel shape (treat as a
   single-digit-percent-to-undetermined claim to test, not a number this note can back with a source). Effort:
   hours — compiler flags only, no kernel rewrite. Risk: very low (reversible, no correctness risk from the compiler
   itself, though occupancy could regress and the driver can silently override `-dlcm=ca` if occupancy would hit
   zero). **Measurement first:** `cuobjdump -sass` diff of the fused gate/up kernel with and without each flag
   (loads changed from 32-bit to L1-cached? register count moved off 254–255?), then a same-card microbenchmark run
   in [bench/mmvq_pascal](../../../bench/mmvq_pascal/). This is strictly prior to anything else below — it is the
   "exhaust the compiler" step items 7 and 11 both point at.
2. **A targeted `cuobjdump -sass` audit for the sm_61 fp16-load-convert trap (`HADD2` instead of `F2F`) anywhere the
   fork reads BF16/FP16 scales or router weights and converts to FP32.** Est. gain: potentially large where it
   applies (the forum report measured 16×), but unquantified for this project's specific code paths, and most of
   this project's K/I-quant and Q8_1 paths already work in integer/dp4a space rather than fp16 arithmetic, so the
   exposure may be narrow (the BF16 router, FP16 block scales/sums). Effort: hours (a grep over existing disassembly
   this project already produces). Risk: none — read-only. **Measurement first:** grep the fork's own `d77256999`
   and `pascal-p106-v2` cubins for `HADD2` adjacent to any `F2F`-eligible fp16 load; if found, test whether an
   explicit `__half2float()` call (forcing `F2F`) changes the instruction and the kernel's measured GB/s.
3. **A narrow, DeepGEMM-style cubin yield/reuse-bit patch, tried only if item 1's SASS diff shows a specific
   control-code choice (a stall count, a missed dual-issue opportunity, a yield bit) costing instructions on an
   *already address-bound* loop.** Est. gain: unknown, likely small on this shape — DeepGEMM's own 10%+ figure comes
   from a *compute-saturated* tensor-core MMA/FFMA overlap problem, which is not this card's bottleneck
   (items 1, 4, 9 above). Effort: days (CuAssembler or nbassembler round-trip has to be validated on this project's
   own cubin first, since neither tool has a demonstrated sm_61 kernel in the wild). Risk: moderate — an unvalidated
   assembler editing a production kernel's cubin needs its own correctness check (bit-identical output against the
   C++ kernel, as this project already does for every kernel variant) before it could be trusted. **Measurement
   first:** the SASS diff from item 1, plus a stall-reason count via NVBit (not nvprof/Nsight Compute, both
   Pascal-incompatible or distorting per §6) showing which specific stall reason dominates cycles in the hot loop.
   Only reopen Q613 if this shows the loop below ~70 % of bandwidth *and* a specific, nameable control-code
   inefficiency, matching Q613's own reopening criterion.
4. **LUT-based GEMV (T-MAC-style) for the k ≥ 7 (Q2_0) / k ≥ 13 (BF16) compute-bound verify-pass regime.** Est.
   gain: unknown and likely negative without a bank-conflict-avoiding table layout — no source in §4 demonstrates a
   LUT GEMV beating a dp4a GEMV on a CUDA-core-only GPU, and the one architectural detail found (frequent shared-
   memory bank conflicts during lookup) argues against it on a 5-SM card with limited shared memory per block to
   spend on table replication. Effort: weeks (a from-scratch kernel, no prior art to port). Risk: high — could lose
   to the existing dp4a k-column kernel and only be discoverable after the build. **Measurement first:** before
   building anything, a back-of-envelope bank-conflict count for a 16-entry table replicated N-ways across P106's
   shared-memory banks at the k = 7–13 column counts, compared against the existing k-column dp4a kernel's measured
   GB/s from [bench/mmvq_pascal](../../../bench/mmvq_pascal/). Lowest-ranked candidate in this note; park unless
   that envelope calculation looks favorable.

## 9. Subtopics for deeper searches

1. Does CuAssembler or nbassembler actually round-trip a real sm_61 cubin correctly today (CUDA 12.x toolchain), and
   can either reproduce this project's own `mmvq-pascal.cu` kernel bit-for-bit as a baseline test?
2. What exactly does `-Xptxas -dlcm=ca` change in the fork's compiled MMVQ kernels' SASS (load width, cache
   qualifier, register pressure), measured with a direct diff rather than inferred from the tuning guide?
3. Does the sm_61 `HADD2`-vs-`F2F` fp16-conversion trap (a single, unconfirmed forum report) actually appear
   anywhere in this project's own binaries, specifically around the BF16 router or any FP16 block scale/sum path?
4. What is TVM's actual Pascal/sm_61 autotuning status in 2026 (this pass did not run this down at all): does
   `tvm.autotune` or `meta_schedule` target sm_61 codegen, and has anyone published a Pascal GEMV/GEMM number?
5. What does NVBit's dynamic instruction/stall-reason counting actually show for this project's own MMVQ kernels at
   Flint's shapes, as a non-perturbing replacement for the Nsight Compute/nvprof timing this project's cards refuse
   or distort?
6. Is there a published, hardware-measured (not simulated) LUT-GEMV-vs-dequant-GEMV comparison on any pre-Ampere,
   non-tensor-core NVIDIA GPU, at any bit width — closing the one gap this pass could not fill from any source?
7. What register-bank-conflict structure, if any, actually exists in Pascal's dp4a execution path specifically
   (distinct from the FFMA path maxas documents) — is there a Pascal-specific primary source on dp4a operand
   bank placement, or does the maxas FFMA analysis simply not apply to integer dot-product instructions?
8. What do the PTX ISA's own pages (not web summaries) say about the sm/PTX-ISA-version floor for `ld.global.nc`,
   `L1::no_allocate`, and `prefetch.global.L2` — are any of them usable on sm_61 at all, given this pass could not
   get a clean read of the primary PTX ISA document?
9. Has anyone published a Pascal-era (sm_60/61/62) result from nbassembler specifically — a kernel, a benchmark, an
   issue thread reporting success or failure — beyond the bare architecture-support claim in its README?
10. Does ptxas's register-bank mis-scheduling (documented by Gray for FFMA) have any measurable analogue in the
    address-arithmetic-heavy integer kernels this project actually runs, or is "ptxas handles bank conflicts
    poorly" a claim that is specific to the FFMA pipeline and simply inapplicable to XMAD/IMAD-chain address code?

## Sources

GPU SASS / assemblers (P unless noted): maxas [repo](https://github.com/NervanaSystems/maxas),
[control codes](https://github.com/NervanaSystems/maxas/wiki/Control-Codes),
[SGEMM](https://github.com/NervanaSystems/maxas/wiki/SGEMM),
[introduction](https://github.com/NervanaSystems/maxas/wiki/Introduction); KeplerAs
[paper](https://static.aminer.org/pdf/20170130/pdfs/ppopp/c7hwcafxtjlpqeesb0wl926oskzvr1di.pdf); CuAssembler
[repo](https://github.com/cloudcores/CuAssembler); nbassembler [repo](https://github.com/qianyantech/nbassembler);
turingas [repo](https://github.com/daadaada/turingas); maxas-explained
[(S)](https://github.com/stefan20162016/maxas-explained); NVIDIA forum on SASS control bits
[(S)](https://forums.developer.nvidia.com/t/in-sass-output-generated-for-cc5-0-cc6-0-and-cc6-1/111105); NVIDIA forum
on Pascal fp16/XMAD/HADD2
[(S)](https://forums.developer.nvidia.com/t/nvidia-pascal-titan-xp-titan-x-geforce-gtx-1080-ti-gtx-1080-gtx-1070-gtx-1060-gtx-1050-gt-1030/42660);
NVIDIA forum on XMAD meaning [(S)](https://forums.developer.nvidia.com/t/xmad-meaning/46653); Winograd/maxDNN
[(P, arXiv 1509.09308)](https://arxiv.org/abs/1509.09308), Nervana blog
[(S)](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Not-so-fast-FFT-Winograd/post/1335707);
DeepGEMM [repo](https://github.com/deepseek-ai/DeepGEMM); FlashMLA coverage
[(S)](https://itc.ua/en/news/bypassing-sanctions-deepseek-flashmla-improves-the-performance-of-nvidia-h800-ai-chips-by-8-times/amp).

Inline PTX / dequant: Marlin [repo](https://github.com/IST-DASLab/marlin); FasterTransformer
[interleaved_numeric_conversion.h](https://github.com/NVIDIA/FasterTransformer); llama.cpp
[vecdotq.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh) (this project's
reading).

LUT-GEMM / compute-bound matvec: T-MAC [arXiv 2407.00088](https://arxiv.org/abs/2407.00088); LUT-GEMM
[arXiv 2206.09557](https://arxiv.org/abs/2206.09557); FLUTE [arXiv 2407.10960](https://arxiv.org/abs/2407.10960);
bitnet.cpp [arXiv 2502.11880](https://arxiv.org/abs/2502.11880); FIGLUT [arXiv 2503.06862](https://arxiv.org/pdf/2503.06862);
CodeQuant [arXiv 2604.10496](https://arxiv.org/pdf/2604.10496); LUT Tensor Core [arXiv 2408.06003](https://arxiv.org/pdf/2408.06003).

CPU: llamafile tinyBLAS write-up coverage
[(S)](https://hacks.mozilla.org/2024/04/llamafiles-progress-four-months-in/); ggml CPU backend
[CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/CMakeLists.txt); ik_llama.cpp
[repo](https://github.com/ikawrakow/ik_llama.cpp); KleidiAI [repo](https://github.com/ARM-software/kleidiai);
oneDNN [repo](https://github.com/uxlfoundation/oneDNN).

Tools: CUDA Binary Utilities [12.9 docs](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-binary-utilities/index.html);
Nsight Compute [release notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html); CUDA Profiler
changelog [12.4.1 archive](https://docs.nvidia.com/cuda/archive/12.4.1/profiler-users-guide/changelog.html); NVBit
[NVIDIA research page](https://research.nvidia.com/publication/2019-10_nvbit-dynamic-binary-instrumentation-framework-nvidia-gpus),
[repo](https://github.com/NVlabs/NVBit); Accel-Sim [ISCA 2020 paper](https://engineering.purdue.edu/tgrogers/papers/khairy.isca2020.pdf).

Alternatives: Triton [repo](https://github.com/openai/triton); Pascal Tuning Guide
[docs.nvidia.com](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html).

This project's own notes (read for scope and baselines, not re-measured): [kquant_decode_pascal.md](kquant_decode_pascal.md),
[pascal_mmvq_integration.md](pascal_mmvq_integration.md),
[quant_speed_b_formats_kernels.md](quant_speed_b_formats_kernels.md), [host_decode_loop.md](host_decode_loop.md),
[quants/experiments.md](../../../quants/experiments.md) (Q613).
