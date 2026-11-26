# Rig tarballs built without Docker, checked against a Docker build (C1190, Q1928; 2026-11-26)

What this is: registry row Q1928 in [quants/experiments.md](../../../quants/experiments.md). The rig's programs come in
a tarball that [scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) builds inside NVIDIA's
CUDA 12.9.2 image for Ubuntu 24.04. A PC with that toolkit and system but no Docker could not build one. The script
now has `NATIVE=1`, which runs the same build steps and settings on the machine itself. Here a native build of one
Miner Llama commit (`caf27fdae`, fork branch `pascal-p106-v5-cand-b12723`) is compared with the Docker build the rigs
use, and then the first tarball made this way is built for a rig check: fork branch `pascal-p106-v5-iq1m-mmq-mtok`
(`b6e99bd11`). The native builds ran on an Ubuntu 24.04.5 PC with CUDA 12.9 (nvcc 12.9.86) and gcc 13.3.0, 8 compile
jobs at the lowest priority. Nothing here is timed on a rig.

## Result in plain words

- **The cards get the same machine code.** In `llama-server` and in `test-backend-ops`, all 10,099 kernels of the
  P106's code (compute 6.1) are identical to the Docker build's, instruction words and scheduling words included.
- **The host side matches too.** Both builds' programs need the same libraries, ask for glibc 2.38 at most, contain no
  AVX instruction (none of the 24 programs, in either build) and were compiled by the same gcc
  (13.3.0-6ubuntu2~24.04.1). Every host function of `llama-server` and `test-backend-ops` has the same size in both;
  480 names differ only because the compiler writes a hash of the source folder into them.
- **One fix was needed:** the first native build linked OpenSSL, because the build PC has OpenSSL's headers and the
  image does not. `NATIVE=1` now turns OpenSSL off, as it is in every Docker build.
- **One difference is left, and it does not reach the rig.** One fast-attention kernel's intermediate code (PTX, which
  only a card without compute-6.1 code would compile) has one load split differently; its compute-6.1 code is the same.
  Four repeat compiles on the build PC gave the native version every time, also through a source path as long as the
  container's, so it is not noise on that PC. The likely cause is a number nvcc writes into the names of that kernel's
  shared-memory variables, which looks like its process id (5339 in the container, 161637 on the PC); an ordinary user
  on that PC cannot run a compile in its own process-id space, so this was not tested.
- **Row 6e's tarball exists.** `llama.cpp-20261126-b6e99bd11-pascal-p106-v5-iq1m-mmq-mtok.tar.gz`, 1,699,698,335
  bytes, md5 `dc1fb0790803711d42306d10f482322b`, built in 620 s: IQ1_M on MMQ (`GGML_CUDA_MMQ_IQ1_M`) beside the IQ
  fast kernels (`GGML_CUDA_MTOK_IQ_FAST`), 128 IQ1_M MMQ kernels in it (the base has none); `llama-bench`,
  `llama-server` and `test-backend-ops` hold no AVX instruction and ask for glibc 2.38 at most.
- **Without the switch the Docker path is unchanged.** A stand-in `docker` that only records its arguments got the same
  arguments from the script before and after the change; the build steps differ only in lines that do nothing in the
  container (below).

## The change

`NATIVE=1 scripts/build_llamacpp_docker.sh <ref>` (the script's header has the details):

- the build steps are one text that runs either in the container or, with `NATIVE=1`, under `env -i` on the machine
  itself (only `PATH`, `HOME` and the build settings pass in, as only they reach the container);
- `$WORK` takes the place of the container's `/work`; the container's `apt-get` and `chown` and its
  `git config --global --add safe.directory` are skipped;
- the CMake flags, the CUDA architecture (61) and the CPU flags are the same, plus `-DLLAMA_OPENSSL=OFF`;
- before building, the script checks for CUDA 12.x in `CUDA_HOME` (default `/usr/local/cuda`) and for git, cmake,
  make, gcc, g++, ccache, curl and objdump, and warns when the machine's glibc is newer than Ubuntu 24.04's 2.39;
- `BUILDINFO.txt` names the system, CUDA, gcc and cmake in place of the image.

**The Docker path, before and after** ([dryrun.tsv](dryrun.tsv), [dryrun_steps.diff](dryrun_steps.diff)). Three
cases (the settings of the Docker build compared here, no settings at all, and every option with `DOCKER_ARGS` and
`IMAGE` set): in each, the 31 to 35 arguments before the build steps are byte for byte the same. The steps differ in
30 lines, the same 30 in all three cases, and each does what it did before inside the container, where `WORK` and
`NATIVE` are never set: `WORK=${WORK:-/work}` makes every `$WORK` `/work`; `apt-get`, `safe.directory` and `chown`
still run; the web page log moves from `/tmp/ui.log` to a `mktemp` file inside the container; the OpenSSL flag
expands to nothing.

## The check

The Docker tarball is the rigs' tarball of `caf27fdae` (md5 `779a6b2473045c77a643a9871ffbc96f`, built 2026-11-25,
C1118); the native one was built with the same settings (`TESTS=1 HF_UI_VERSION=b12722`).

| What | Docker | Native | Table |
|---|---|---|---|
| `BUILDINFO.txt` | | the same ref, commit, web page, check tools, 0 AVX, GLIBC_2.38, libraries; differs in the build day and time, the image line and where the loader found the libraries | [buildinfo.tsv](buildinfo.tsv) |
| Programs | 24 | the same 24; every md5 differs (build ids and paths), sizes the same or 4,096 / 8,192 bytes larger | [programs.tsv](programs.tsv) |
| AVX instructions (ymm registers / any VEX-coded instruction), 24 programs | 0 / 0 | 0 / 0 | [elf.tsv](elf.tsv) |
| Highest glibc symbol, 24 programs | GLIBC_2.38 | GLIBC_2.38 | [elf.tsv](elf.tsv) |
| Libraries needed, RUNPATH, compiler (`.comment`) | | the same in every program | [elf.tsv](elf.tsv) |
| Host functions of `llama-server` / `test-backend-ops` | 44,769 / 24,697 | 44,529 / 24,457 in both, none of another size; 240 / 240 in one build only, all of them names with a path hash, and renamed they pair up name for name and size for size | [host_totals.tsv](host_totals.tsv), [host_pairs.tsv](host_pairs.tsv), [host_llama-server.tsv.gz](host_llama-server.tsv.gz) |
| Compute-6.1 machine code (SASS), each program | 10,099 kernels | 10,099 identical | [card_llama-server.tsv](card_llama-server.tsv), [card_test-backend-ops.tsv](card_test-backend-ops.tsv) |
| Compute-6.1 PTX, each program | 10,099 functions | 10,098 identical, 1 differs | the same, [card_test-backend-ops_code.tsv](card_test-backend-ops_code.tsv) |

The card code was compared function by function ([run/card_compare.py](run/card_compare.py), after C1179's
`code_compare.py`). A linked program names no source file in its card code, so functions are matched by name and copy
number. nvcc names anonymous namespaces and internal functions after a hash of the source's path, and gives static
shared-memory variables a number that changed from build to build, so both are renamed before comparing: 15 SASS and
36 PTX names carry a hash, and no hash is the same in the two builds. The first two runs of the comparison caught too
few of these names and showed 11 PTX functions as different; the third catches nested names too.

**The one PTX function:** `flash_attn_tile_pascal<128, Q8_0, Q8_0, ...>` in `fattn-tile-pascal.cu`. The Docker build's
PTX has `.pragma "used_bytes_mask 255"` and one more `ld.shared.v2.f32` (2,349 lines, 2,069 float registers); the
native one loads the same shared memory without them (2,347 lines, 2,067). The pragma stands 3,553 times in the
Docker build's PTX and 3,552 times in the native one's, both cubins say toolkit 12.9, and the compute-6.1 machine code of this kernel is identical, so both PTX texts
compile to the same code for the P106. Recompiled four times at once on the build PC with no compiler cache, the
source gave the native version each time ([repro.tsv](repro.tsv), [repro_func.tsv](repro_func.tsv)); through a symbolic
link as long as the container's `/work/src` it did too ([repro_alias.tsv](repro_alias.tsv),
[repro_alias_func.tsv](repro_alias_func.tsv)). The known run-to-run variation of the Q4_0 and Q4_1 MMQ kernels (C1179)
did not show here: both are identical in this pair of builds.

## Files

| File | What |
|---|---|
| [builds.tsv](builds.tsv) | the three native builds and the Docker one: settings, seconds, tarball, bytes, md5 |
| [buildinfo.tsv](buildinfo.tsv) | `BUILDINFO.txt`, field by field |
| [programs.tsv](programs.tsv) | each program's size and md5 in both builds |
| [elf.tsv](elf.tsv) | per program and build: AVX counts, highest glibc symbol, libraries needed, RUNPATH, compiler |
| [host_totals.tsv](host_totals.tsv), [host_pairs.tsv](host_pairs.tsv) | host functions: counts, sizes, and the names in one build only renamed |
| [host_llama-server.tsv.gz](host_llama-server.tsv.gz) | those 480 names with their sizes (`test-backend-ops` has the same list) |
| [listings.tsv](listings.tsv) | the size of each `cuobjdump` listing compared |
| `card_<program>.tsv`, `_families.tsv`, `_pairs.tsv`, `_code.tsv` | the card code per verdict, per kernel family, constant pairs (none) and the differing function |
| [repro.tsv](repro.tsv), [repro_func.tsv](repro_func.tsv), [repro_alias.tsv](repro_alias.tsv), [repro_alias_func.tsv](repro_alias_func.tsv) | the repeat compiles of `fattn-tile-pascal.cu` |
| [dryrun.tsv](dryrun.tsv), [dryrun_steps.diff](dryrun_steps.diff) | the Docker path before and after the change |
| [b6e99bd11_binaries.tsv](b6e99bd11_binaries.tsv) | row 6e's tarball and every program's size and md5 |
| [run/dryrun.sh](run/dryrun.sh) | the stand-in `docker` run |
| [run/build_native.sh](run/build_native.sh) | one native build, detached, with its log, exit code, seconds and md5 |
| [run/compare.sh](run/compare.sh) | the whole comparison of two unpacked tarballs |
| [run/card_compare.py](run/card_compare.py), [run/host_pairs.py](run/host_pairs.py) | the card code and host symbol comparisons |
| [run/ptx_repro.sh](run/ptx_repro.sh) | the repeat compiles of one source |

## Commands

On the build PC, as an ordinary user, in `$W` with the `run/` scripts and the changed build script in `$W/scripts`. The
source clone `$W/work/src` came from an earlier clone (`git clone`, then `origin` set to
https://github.com/ggml-org/llama.cpp and the fork added as `fork`, https://github.com/Cinderella-Man/miner-llama).
The Docker tarball was unpacked in `$W/docker`.

```bash
OLD=<the script before> NEW=$W/scripts/build_llamacpp_docker.sh OUT=$W/dryrun/out bash $W/scripts/dryrun.sh
SCRIPT=$W/scripts/build_llamacpp_docker.sh WORK=$W/work OUT=$W/logs TAG=caf27fdae HF_UI_VERSION=b12722 \
  bash $W/scripts/build_native.sh fork/pascal-p106-v5-cand-b12723
A=$W/docker/llama.cpp-20261125-caf27fdae-pascal-p106-v5-cand-b12723 \
  B=$W/work/llama.cpp-20261126-caf27fdae-pascal-p106-v5-cand-b12723 OUT=$W/compare bash $W/scripts/compare.sh
python3 -I $W/scripts/host_pairs.py $W/compare/host_pairs.tsv $W/compare/host_llama-server.tsv \
  $W/compare/host_test-backend-ops.tsv
SCRIPT=$W/scripts/build_llamacpp_docker.sh WORK=$W/work OUT=$W/logs TAG=b6e99bd11 HF_UI_VERSION=b12674 \
  bash $W/scripts/build_native.sh fork/pascal-p106-v5-iq1m-mmq-mtok
```

The repeat compiles ran with the work tree at `b6e99bd11`, whose `fattn-tile-pascal.cu`, `fattn-tile-pascal.cuh`,
`fattn-common.cuh` and `common.cuh` are `caf27fdae`'s. `FUNC` is the kernel's name in the Docker build
([card_test-backend-ops_code.tsv](card_test-backend-ops_code.tsv) shows it with its hashes renamed):

```bash
/usr/local/cuda/bin/cuobjdump -ptx $W/docker/llama.cpp-20261125-caf27fdae-pascal-p106-v5-cand-b12723/bin/test-backend-ops \
  > $W/ptxpeek/full.A.ptx
/usr/local/cuda/bin/cuobjdump -ptx $W/work/llama.cpp-20261126-caf27fdae-pascal-p106-v5-cand-b12723/bin/test-backend-ops \
  > $W/ptxpeek/full.B.ptx
FUNC=_ZN53_GLOBAL__N__02ad077c_20_fattn_tile_pascal_cu_af87bd1422flash_attn_tile_pascalILi128EL9ggml_type8ELS1_8ELb1ELb0ELb0EEEvPKcS3_S3_S3_S3_PKiS5_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiill
BUILD=$W/work/src/build SRC=$W/work/src/ggml/src/ggml-cuda/fattn-tile-pascal.cu FUNC=$FUNC OUT=$W/repro \
  LISTING_A=$W/ptxpeek/full.A.ptx LISTING_B=$W/ptxpeek/full.B.ptx bash $W/scripts/ptx_repro.sh
BUILD=$W/work/src/build SRC=$W/work/src/ggml/src/ggml-cuda/fattn-tile-pascal.cu FUNC=$FUNC OUT=$W/repro-alias \
  ROOT=$W/work/src ALIAS=/tmp/wsrc N=2 LISTING_A=$W/ptxpeek/full.A.ptx bash $W/scripts/ptx_repro.sh
```

The first native build ran before `-DLLAMA_OPENSSL=OFF` was in the script; the second and third ran with the script as
committed here, apart from its header comment.

## Provenance and licence

Generated here from Miner Llama's sources (MIT, as llama.cpp) and its two tarballs; no model file was read and no card
ran. The build logs and the full code listings are not published; the tables summarize them.
