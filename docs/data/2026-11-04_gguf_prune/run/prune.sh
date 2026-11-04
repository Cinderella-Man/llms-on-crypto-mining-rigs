#!/usr/bin/env bash
# Q890: usage-pruned copies of a Flash-Next file with scripts/gguf_prune.py, checked on a CPU (no GPU, no rig): the
# tool's self-test, the random qwen4exp models of ../../2026-11-04_route_once (run/cpu_checks.sh's random/ folder),
# then the real file cut to each KEEP, its verify, header copies for scripts/fit_probe.sh and a CPU load of every file.
#   selftest  scripts/gguf_prune_test.py
#   random    per random model (32 experts, 8 used, 2 layers): all ids in order must write the same bytes; all ids in
#             reverse must verify and give llama-perplexity the same logits as the model itself (the same KLD and
#             perplexity against the model's own logits: a permutation of experts and router rows is the same model),
#             while the reversed slabs with the routers in the source's order (the negative control) must not; 24 kept
#             by a seeded random ranking must verify, load, give a KLD, and pass test-route-once
#   cut       per N in KEEP: gguf_prune.py prune --keep N --experts EXPERTS into $OUT/${TAG}_e$N/, named $NAME-e$N, a
#             shard with nothing to cut linked to its source (Flash-Next's n-gram table)
#   verify    gguf_prune.py verify of each cut against SHARDS, md5 of every written shard ($OUT/${TAG}_e$N/md5.txt),
#             and $OUT/cuts.tsv (run/cuts.py: bytes and the routings the kept experts carry)
#   headers   header copies of each cut ($HEADERS/${TAG}_e$N/, sparse files of the full size), checked with
#             verify --headers-only and packed with tar -S
#   load      llama-perplexity on the CPU over the first CHUNKS chunks of 512 tokens of TEXT: the uncut file writes its
#             logits (the shards linked into $WORK/$TAG/ under one name), each cut gives its KL divergence against them
#             (the cut's own cost on this text, a small sample)
# Env: WORK, BIN (a CPU build's bin with llama-perplexity and test-route-once), RANDOM_MODELS, TEXT; SHARDS (the
#      file's shards, comma-separated, every shard of the split), NAME (the shard names' stem), TAG (folder stem),
#      EXPERTS (docs/data/2026-11-04_expert_usage/experts.tsv.gz), KEEP ("464 448 416"), OUT (default $WORK);
#      STEPS (default "selftest random cut verify headers load"), CHUNKS (4), THREADS (8), CPUS (a cpuset for every
#      step, e.g. 8-11,20-23), IMAGE, MOUNTS (more folders the containers mount, at their own paths), FREE_GB (50: the
#      space a cut must leave on OUT's disk), HEADERS (default $WORK/headers), PY_IMAGE (an image with python3 to run
#      the Python steps in, e.g. python:3.12-slim-bookworm) and PY_MEMORY (8g).
# One verdict line per check in $WORK/summary.tsv.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$here/../../../.." && pwd)
: "${WORK:?}"
OUT=${OUT:-$WORK}
HEADERS=${HEADERS:-$WORK/headers}
STEPS=${STEPS:-selftest random cut verify headers load}
KEEP=${KEEP:-464 448 416}
CHUNKS=${CHUNKS:-4}
THREADS=${THREADS:-8}
FREE_GB=${FREE_GB:-50}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
TOOL=$REPO/scripts/gguf_prune.py
mkdir -p "$WORK/logs" "$OUT"
touch "$WORK/summary.tsv"

log()     { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
verdict() { printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
mounts=""
for d in $(printf '%s\n' "$WORK" "$OUT" ${BIN:+"$(dirname "$BIN")"} ${RANDOM_MODELS:-} ${MOUNTS:-} | sort -u); do
    mounts="$mounts -v $d:$d"
done
# a command at low priority; with PY_IMAGE in a container whose memory (page cache included) is capped at PY_MEMORY,
# so that reading and writing ~150 GB per cut does not push other jobs' files out of the page cache
low() {
    if [ -n "${PY_IMAGE:-}" ]; then
        docker run --rm -i --label pascal-quants=1 --cpu-shares 256 --memory "${PY_MEMORY:-8g}" ${CPUS:+--cpuset-cpus $CPUS} \
            $mounts "$PY_IMAGE" nice -n 19 "$@"
    else
        nice -n 19 ionice -c 2 -n 7 ${CPUS:+taskset -c $CPUS} "$@"
    fi
}
py()      { low python3 -I "$@"; }
box()     { docker run --rm --label pascal-quants=1 --cpu-shares 256 ${CPUS:+--cpuset-cpus $CPUS} $mounts "$IMAGE" "$@"; }
# the lines of a llama-perplexity log that hold its result
result()  { grep -E "^(Mean    KLD|Maximum KLD|Same top p|Mean PPL|Final estimate)" "$1"; }
IFS=, read -r -a shards <<< "${SHARDS:-}"
# the -0000k-of-0000N.gguf part of a shard's name (.gguf for a file that is not split)
suffix()  { basename "$1" | grep -oE -- '-[0-9]{5}-of-[0-9]{5}\.gguf$' || echo .gguf; }

# ids <model> <all|reverse> <out>: every layer's ids, ascending or reversed
ids() {
    py - "$1" "$2" "$3" "$REPO/scripts" << 'EOF'
import sys
sys.path.insert(0, sys.argv[4])
import gguf_prune as p
hs = [p.Header(sys.argv[1])]
n = p.first_value(hs, "expert_count")
layers = sorted({l for l, _s in p.sliced(hs, n).values()})
order = list(range(n)) if sys.argv[2] == "all" else list(range(n))[::-1]
with open(sys.argv[3], "w") as f:
    f.writelines(f"{l}\t{','.join(map(str, order))}\n" for l in layers)
EOF
}

# ranking <model> <out>: a ranking table with seeded random counts per layer and expert
ranking() {
    py - "$1" "$2" "$REPO/scripts" << 'EOF'
import random, sys
sys.path.insert(0, sys.argv[3])
import gguf_prune as p
hs = [p.Header(sys.argv[1])]
n = p.first_value(hs, "expert_count")
rng = random.Random(890)
with open(sys.argv[2], "w") as f:
    f.write("layer\texpert\troutings\n")
    for l in sorted({l for l, _s in p.sliced(hs, n).values()}):
        f.writelines(f"{l}\t{e}\t{rng.randrange(1000)}\n" for e in range(n))
EOF
}

if step selftest; then
    log "selftest"
    if py "$REPO/scripts/gguf_prune_test.py" "$WORK/selftest" > "$WORK/logs/selftest.log" 2>&1; then
        verdict selftest OK "$(tail -n 1 "$WORK/logs/selftest.log")"
    else
        verdict selftest FAIL "$WORK/logs/selftest.log"
    fi
fi

if step random; then
    : "${BIN:?}" "${RANDOM_MODELS:?}" "${TEXT:?}"
    for m in qwen4exp-moe qwen4exp-moe-q5k; do
        src=$RANDOM_MODELS/$m.gguf d=$WORK/random/$m
        mkdir -p "$d"
        log "random $m"
        ids "$src" all "$d/all.tsv" && ids "$src" reverse "$d/reverse.tsv" && ranking "$src" "$d/experts.tsv"
        py "$TOOL" prune --ids "$d/all.tsv" --out-dir "$d/all" "$src" > "$d/all.log" 2>&1
        if cmp -s "$src" "$d/all/$m.gguf"; then verdict "random-all $m" OK "every id in order: the same bytes"
        else verdict "random-all $m" FAIL "$d/all.log"; fi

        ppl() { box "$BIN/llama-perplexity" -m "$1" -f "$TEXT" -c 128 -b 128 --chunks "$CHUNKS" -t "$THREADS" "${@:2}"; }
        ppl "$src" --kl-divergence-base "$d/base.kld" > "$d/base.log" 2>&1
        ppl "$src" --kl-divergence-base "$d/base.kld" --kl-divergence > "$d/self.log" 2>&1
        py "$TOOL" prune --ids "$d/reverse.tsv" --out-dir "$d/reverse" "$src" > "$d/reverse.log" 2>&1 &&
            py "$TOOL" verify "$src" --new "$d/reverse/$m.gguf" --ids "$d/reverse/kept.tsv" >> "$d/reverse.log" 2>&1
        rc=$?
        ppl "$d/reverse/$m.gguf" --kl-divergence-base "$d/base.kld" --kl-divergence > "$d/reverse.ppl.log" 2>&1
        if [ $rc -eq 0 ] && [ -n "$(result "$d/self.log")" ] && [ "$(result "$d/self.log")" = "$(result "$d/reverse.ppl.log")" ]; then
            verdict "random-reverse $m" OK "verify ok; the same KLD and PPL as the model itself: $(result "$d/self.log" | tr -s ' ' | tr '\n' ' ')"
        else
            verdict "random-reverse $m" FAIL "$d/reverse.log, $d/reverse.ppl.log against $d/self.log"
        fi
        # the negative control: the reversed slabs with the routers put back in the source's order must differ
        cp "$d/reverse/$m.gguf" "$d/mismatch.gguf"
        py - "$src" "$d/mismatch.gguf" "$REPO/scripts" << 'EOF'
import os, sys
sys.path.insert(0, sys.argv[3])
from gguf_reshard import Header
s, n = Header(sys.argv[1]), Header(sys.argv[2])
where = {t[0]: t[3] for t in n.tensors}
with open(sys.argv[1], "rb") as fs, open(sys.argv[2], "r+b") as fn:
    for name, _dims, _typ, off in s.tensors:
        if name.endswith(".ffn_gate_inp.weight"):
            os.pwrite(fn.fileno(), os.pread(fs.fileno(), s.slot[name], s.data0 + off), n.data0 + where[name])
EOF
        ppl "$d/mismatch.gguf" --kl-divergence-base "$d/base.kld" --kl-divergence > "$d/mismatch.ppl.log" 2>&1
        if grep -q "^Mean    KLD" "$d/mismatch.ppl.log" && [ "$(result "$d/self.log")" != "$(result "$d/mismatch.ppl.log")" ]; then
            verdict "random-mismatch $m" OK "the routers in the wrong order change the result: $(result "$d/mismatch.ppl.log" | tr -s ' ' | tr '\n' ' ')"
        else
            verdict "random-mismatch $m" FAIL "$d/mismatch.ppl.log is the model's own result: the check cannot see a mismatch"
        fi

        py "$TOOL" prune --keep 24 --experts "$d/experts.tsv" --out-dir "$d/cut" "$src" > "$d/cut.log" 2>&1 &&
            py "$TOOL" verify "$src" --new "$d/cut/$m.gguf" --ids "$d/cut/kept.tsv" >> "$d/cut.log" 2>&1
        rc=$?
        ppl "$d/cut/$m.gguf" --kl-divergence-base "$d/base.kld" --kl-divergence > "$d/cut.ppl.log" 2>&1
        box "$BIN/test-route-once" -m "$d/cut/$m.gguf" -n 16 -t "$THREADS" > "$d/cut.route-once.log" 2>&1
        if [ $rc -eq 0 ] && grep -q "^Mean    KLD" "$d/cut.ppl.log" && grep -q "checks, 0 failed" "$d/cut.route-once.log"; then
            verdict "random-cut24 $m" OK "verify ok; loads; $(result "$d/cut.ppl.log" | tr -s ' ' | tr '\n' ' '); test-route-once $(grep -o '[0-9]* checks, 0 failed' "$d/cut.route-once.log" | tail -n 1)"
        else
            verdict "random-cut24 $m" FAIL "$d/cut.log, $d/cut.ppl.log, $d/cut.route-once.log"
        fi
    done
fi

if step cut; then
    : "${NAME:?}" "${TAG:?}" "${EXPERTS:?}" "${SHARDS:?}"
    need=0
    for s in "${shards[@]}"; do need=$((need + $(stat -c %s "$s"))); done
    for n in $KEEP; do
        d=$OUT/${TAG}_e$n
        free=$(df -B1 --output=avail "$OUT" | tail -n 1 | tr -d ' ')
        if [ "$free" -lt $((need + FREE_GB * 1024 ** 3)) ]; then
            verdict "cut e$n" FAIL "$OUT: $((free / 1024 ** 3)) GiB free, the cut needs up to $((need / 1024 ** 3)) and $FREE_GB must stay"
            continue
        fi
        log "cut e$n"
        if py "$TOOL" prune --keep "$n" --experts "$EXPERTS" --out-dir "$d" --prefix "$NAME-e$n" --link-unchanged \
            "${shards[@]}" > "$WORK/logs/cut_e$n.log" 2>&1; then
            verdict "cut e$n" OK "$(tail -n 1 "$WORK/logs/cut_e$n.log")"
        else
            verdict "cut e$n" FAIL "$WORK/logs/cut_e$n.log"
        fi
    done
fi

if step verify; then
    : "${NAME:?}" "${TAG:?}" "${EXPERTS:?}" "${SHARDS:?}"
    dirs=()
    for n in $KEEP; do
        d=$OUT/${TAG}_e$n
        new=()
        for s in "${shards[@]}"; do new+=("$d/$NAME-e$n$(suffix "$s")"); done
        log "verify e$n"
        if py "$TOOL" verify "${shards[@]}" --new "${new[@]}" --ids "$d/kept.tsv" > "$WORK/logs/verify_e$n.log" 2>&1; then
            verdict "verify e$n" OK "$(grep -c 'every kept slab equal' "$WORK/logs/verify_e$n.log") shards"
        else
            verdict "verify e$n" FAIL "$WORK/logs/verify_e$n.log"
        fi
        for f in "${new[@]}"; do [ -L "$f" ] || low md5sum "$f" | sed 's#  .*/#  #'; done > "$d/md5.txt"
        dirs+=("$d")
    done
    py "$here/cuts.py" --experts "$EXPERTS" --shards "$SHARDS" "${dirs[@]}" > "$OUT/cuts.tsv" && log "cuts.tsv written"
fi

if step headers; then
    : "${NAME:?}" "${TAG:?}" "${EXPERTS:?}" "${SHARDS:?}"
    mkdir -p "$HEADERS"
    for n in $KEEP; do
        h=$HEADERS/${TAG}_e$n
        log "headers e$n"
        if py "$TOOL" prune --keep "$n" --experts "$EXPERTS" --out-dir "$h" --prefix "$NAME-e$n" --headers-only \
                "${shards[@]}" > "$WORK/logs/headers_e$n.log" 2>&1 &&
            py "$TOOL" verify "${shards[@]}" --new $(ls "$h"/*.gguf | sort) --ids "$h/kept.tsv" --headers-only \
                >> "$WORK/logs/headers_e$n.log" 2>&1; then
            tar -S -czf "$HEADERS/${TAG}_e$n.tar.gz" -C "$HEADERS" "${TAG}_e$n"
            verdict "headers e$n" OK "$HEADERS/${TAG}_e$n.tar.gz ($(stat -c %s "$HEADERS/${TAG}_e$n.tar.gz") bytes)"
        else
            verdict "headers e$n" FAIL "$WORK/logs/headers_e$n.log"
        fi
    done
fi

if step load; then
    : "${BIN:?}" "${TEXT:?}" "${NAME:?}" "${TAG:?}" "${SHARDS:?}"
    # the uncut file under one name in one folder, as llama.cpp looks for its shards
    u=$WORK/${TAG}
    mkdir -p "$u" "$WORK/load"
    for s in "${shards[@]}"; do ln -sfn "$(realpath "$s")" "$u/$NAME$(suffix "$s")"; done
    first() { ls "$1"/*-00001-of-*.gguf; }
    ppl() { box "$BIN/llama-perplexity" -m "$1" -f "$TEXT" -c 512 -b 512 --chunks "$CHUNKS" -t "$THREADS" "${@:2}"; }
    base=$WORK/load/base.kld
    if [ ! -f "$base" ]; then
        log "load: the uncut file's logits"
        ppl "$(first "$u")" --kl-divergence-base "$base.partial" > "$WORK/load/base.log" 2>&1 &&
            grep -q "Final estimate" "$WORK/load/base.log" && mv "$base.partial" "$base"
    fi
    [ -f "$base" ] || verdict "load uncut" FAIL "$WORK/load/base.log"
    for n in $KEEP; do
        lg=$WORK/load/e$n.log
        grep -q "^Mean    KLD" "$lg" 2> /dev/null && { log "load e$n: done before"; continue; }
        log "load e$n"
        ppl "$(first "$OUT/${TAG}_e$n")" --kl-divergence-base "$base" --kl-divergence > "$lg" 2>&1
        if grep -q "^Mean    KLD" "$lg"; then
            verdict "load e$n" OK "$(result "$lg" | tr -s ' ' | tr '\n' ' ')"
        else
            verdict "load e$n" FAIL "$lg"
        fi
    done
    # the table: llama-perplexity's summary, and the first chunk apart from the rest (its per-chunk lines are running
    # means, so chunk k's own KLD is k * mean_k - (k - 1) * mean_k-1)
    {
        printf 'keep\tmean_kld\tkld_err\tmedian_kld\tkld_99pct\tmax_kld\tsame_top_p_pct\ttop_p_err\tppl_ratio\tchunks\tchunk1_kld\tlater_chunks_kld\n'
        for n in $KEEP; do
            awk -v n="$n" '
                /^ +[0-9]+ +[0-9.]+ ±/ { k = $1; m[k] = $8 }
                /^Mean    KLD:/ { mk = $3; me = $5 }
                /^Median  KLD:/ { md = $3 }
                /^99.0%   KLD:/ { p99 = $3 }
                /^Maximum KLD:/ { mx = $3 }
                /^Same top p:/ { tp = $4; te = $6 }
                /^Mean PPL\(Q\)\/PPL\(base\)/ { r = $4 }
                END { if (mk != "") printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%.4f\t%.4f\n", n, mk, me, md, p99, mx,
                      tp, te, r, k, m[1], (k > 1 ? (k * m[k] - m[1]) / (k - 1) : 0) }' "$WORK/load/e$n.log"
        done
    } > "$WORK/cpu_kld.tsv"
fi
log "done"
