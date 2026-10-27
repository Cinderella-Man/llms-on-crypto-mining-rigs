#!/usr/bin/env bash
# Q163 (2026-10-24): the stock TILE kernels that compute 6.1 runs for prompts (head 128 and 256, 4 tokens x 8 heads per
# block), compiled alone for sm_61 (CPU only, low priority): ptxas resources and the instruction mix of each loop body
# (FFMA, LDS, STS, BAR, F2F per backward-branch loop), next to the new kernel's (run sass61.sh first). Gives the
# "bytes from shared memory per FMA" figures of docs/research/notes/pascal_attention_long_context.md.
# Usage: SRC=<llama.cpp tree, fork branch pascal-p106-v2-prefill-attn> WORK=<dir> sass_stock61.sh
SRC=$(realpath "${SRC:?SRC=<llama.cpp source tree>}")
W=${WORK:?WORK=<work dir>}
mkdir -p $W/sass
docker run --rm --name q163-sass --cpu-shares=256 -v "$SRC:/src:ro" -v "$W/sass:/o" \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -c '
    cd /src/ggml/src/ggml-cuda
    for d in 128 256; do
        nvcc -O3 -DNDEBUG -std=c++17 -arch=sm_61 -use_fast_math -extended-lambda -I/src/ggml/include -I/src/ggml/src \
            -I/src/ggml/src/ggml-cuda -Xptxas -v -c template-instances/fattn-tile-instance-dkq$d-dv$d.cu -o /o/tile$d.o > /o/ptxas-tile$d.txt 2>&1
        cuobjdump -sass /o/tile$d.o > /o/tile$d.sass
    done
'
python3 - "$W/sass" <<'EOF'
import re, sys, collections
d = sys.argv[1]
def funcs(path):
    cur, out = None, collections.OrderedDict()
    for line in open(path):
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1); out[cur] = []; continue
        if cur is None: continue
        m = re.search(r"/\*([0-9a-f]{4,})\*/\s+((?:@!?P\d\s+)?[A-Z0-9_.]+[^;]*);", line)
        if m: out[cur].append((int(m.group(1), 16), m.group(2)))
    return out
def loops(ops):
    res = []
    for a, o in ops:
        if "BRA" in o:
            t = re.search(r"0x([0-9a-f]+)", o)
            if t and int(t.group(1), 16) < a: res.append((int(t.group(1), 16), a))
    return res
def mix(ops, lo, hi):
    c = collections.Counter()
    for a, o in ops:
        if lo <= a <= hi:
            op = o.split()[1] if o.startswith("@") else o.split()[0]
            c[op.split(".")[0]] += 1
    return c
want = {"tile128": r"flash_attn_tileILi128ELi128ELi4ELi8ELb0E", "tile256": r"flash_attn_tileILi256ELi256ELi4ELi8ELb0E",
        "ftp61": r"flash_attn_tile_pascalILi(128|256)EL9ggml_type1E"}
for f, pat in want.items():
    for name, ops in funcs(f"{d}/{f}.sass").items():
        if not re.search(pat, name): continue
        print(name[:110])
        for lo, hi in loops(ops):
            c = mix(ops, lo, hi)
            if c["FFMA"] < 32: continue
            lds = c["LDS"]
            print(f"  loop {lo:x}-{hi:x}: {sum(c.values())} instr, FFMA {c['FFMA']}, LDS {lds}, STS {c['STS']}, BAR {c['BAR']}, F2F {c['F2F']}, FFMA share {c['FFMA']/sum(c.values()):.0%}")
EOF
grep -E "registers|spill" $W/sass/ptxas-tile128.txt | head -4
