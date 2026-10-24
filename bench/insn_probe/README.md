# insn_probe: per-instruction throughput of a GPU (Q103)

NVIDIA throttles some instruction classes on its mining cards. The CMP 90HX, for example, runs dp4a 29× and FP32 14–15×
below its silicon's rate, as reported in [Pascal forks and quants](../../docs/research/notes/pascal_forks_and_quants.md),
section 1. No such limit is reported for Pascal mining cards. The fork's decode kernels use dp4a at only a few percent
of its rate, though, so they could not reveal a moderate throttle. Before the prompt and multi-token kernels are tuned
further (Q78, Q80), this probe measures the P106-090's rate per instruction class directly.

## What it measures

One kernel per class, each timed on its own:

| Class | Instruction (sm_61 SASS) | What it is |
|---|---|---|
| `ffma` | FFMA | 32-bit FP fused multiply-add; the reference every other class is compared with |
| `dfma` | DFMA | 64-bit FP fused multiply-add |
| `hfma2` | HFMA2 | 16-bit FP fused multiply-add on a pair (two results per instruction) |
| `iadd` | IADD | 32-bit integer add |
| `imad` | XMAD | 32-bit integer multiply-add (three XMADs on compute 6.x) |
| `lop3` | LOP3 | three-input logic |
| `shf` | SHF | funnel shift (rotate) |
| `bfe` | BFE | bit-field extract |
| `prmt` | PRMT | byte permute |
| `dp4a` | IDP | dot product of four int8 pairs into a 32-bit accumulator, the core of llama.cpp's quantized kernels |
| `popc` | POPC | population count |
| `i2f` | I2F | int32 to FP32 conversion |
| `f2f` | F2F | FP32 to FP16 and back (counted as two conversions) |
| `shfl` | SHFL | warp shuffle (butterfly) |
| `lds` | LDS | 32-bit shared-memory load, conflict-free |

**How it measures.**
- Every thread runs 8 independent dependency chains of the instruction, unrolled 8 steps per loop iteration, in inline
  PTX. Blocks of 256 threads run 8 per SM, which is full occupancy on compute 6.x since every kernel needs at most
  32 registers.
- The first thread of each block records the SM it ran on (`%smid`) and the SM clock (`clock64`) at its start and end.
  Per SM, the block's operations divided by the cycles from the first start to the last end give **operations per
  clock per SM**. That number does not depend on the clock the card happens to run at.
- Each class runs three times for ~100 ms each after ~2 s of FFMA warm-up, and the best run counts.

**Checking that it times the right thing.** `build.sh` counts each kernel's instruction in the sm_61 SASS
(`build/sass_counts.txt`): 64 per loop iteration, times the compiler's unrolling of the loop. A count below that means
the compiler merged steps. That is how the integer add got its own kernel: ptxas folds a chain of adds of one value
into a single add, so `iadd` adds a neighbouring chain's value instead.

## Reading the result

The output gives each class's operations per clock per SM (lane operations: one warp instruction is 32 of them), the
same rate in warp instructions, and the ratio to FFMA:
- NVIDIA's table for compute capability 6.1 lists 128 results per clock per SM for 32-bit FP, 4 for 64-bit FP and 2
  for 16-bit FP (one HFMA2 instruction per clock).
- NVIDIA's INT8 figure for its GP10x boards is four times their FP32 FLOPS. A dp4a does four multiply-adds where an
  FFMA does one, so on unthrottled Pascal dp4a runs at FFMA's instruction rate: ratio ~1.
- A class far below its nominal rate, while FFMA reaches ~128, is a per-class limit.

## Building and running

```bash
bench/insn_probe/build.sh                     # Docker, CUDA 12.9 image; DOCKER_ARGS adds docker run options
scp bench/insn_probe/build/insn_probe rig:/tmp/
ssh rig 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/insn_probe --csv /tmp/insn.csv' | tee insn_rig.txt
scp rig:/tmp/insn.csv . && ssh rig 'rm -f /tmp/insn_probe /tmp/insn.csv'
```

The run takes about a minute on one card. Options: `--classes dp4a,lds` (FFMA always runs as the reference),
`--ms` (time per run), `--runs`, `--device`, `--list`. `bench/rig_microbench/run_block8.sh` runs it with the other
one-card microbenchmarks. The binary has sm_61 code and compute_61 PTX; host code is built without AVX and the
CUDA runtime linked statically.
