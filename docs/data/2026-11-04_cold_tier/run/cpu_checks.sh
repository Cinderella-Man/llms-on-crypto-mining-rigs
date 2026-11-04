#!/usr/bin/env bash
# Q238's checks on a CPU, no GPU needed: the cold tier of Miner Llama's -coldtier branch (run/build.sh's cpu build)
# and scripts/gguf_tier.py, every binary in a container of the build image.
#   selftest  scripts/gguf_tier_test.py, then eight planted bugs in a copy of the tool, each of which the self-test must
#             catch
#   identity  test-llama-archs -b CPU on every architecture: logits written by the base build (BASE_BIN, a CPU build
#             of the branch's parent or of a branch whose CPU graphs equal it) and compared by this build: no file has
#             a cold tier, so every architecture must match bit for bit
#   random    a random qwen4exp from test-llama-archs (32 routed experts, 8 used, 2 layers, F32), its Q4_0-expert
#             copy (llama-quantize) and a sharper one (stdev 0.6): each tiered by a seeded random ranking into hot and cold of the same type must
#             give llama-perplexity (without the repacked CPU kernels, which round by batch shape) the same
#             log-probabilities as the untiered model, byte for byte in the KL base file, at three splits (hot 24 / 31 / 8 = expert_used_count); the negative control (cold slabs in an
#             order the tier map does not say) must differ; F32 hot experts with Q4_0 cold ones (extract, llama-quantize,
#             tier) must load, run and differ from both
#   moeids    test-moe-ids (this branch's tests/) on every random model, untiered and tiered: each MUL_MAT_ID's ids
#             are distinct per token and below its expert count (CUDA's kernels need both; the CPU's do not)
#   real      a tiered real file (FILE, its first shard) on the CPU: llama-perplexity over the first CHUNKS chunks of
#             512 tokens of TEXT against BASE_KLD (the untiered file's logits on the same chunks), a preview of its
#             cost; one line per file in $WORK/cpu_kld.tsv
# Env: WORK, REPO, BIN (the cpu build's bin), IMAGE, PY_IMAGE (python:3.12-slim-bookworm), THREADS (8), CPUS (a cpuset for every container), STEPS (default
#      "selftest identity random moeids"), MOUNTS (more folders the containers mount, at their own paths), BASE_BIN for
#      identity; FILES ("<name>=<first shard> ..."), TEXT, BASE_KLD, CHUNKS (4) for real
# Writes $WORK/random, $WORK/checks/<step>/, one verdict line per check in $WORK/checks.tsv.
set -uo pipefail
: "${WORK:?}" "${REPO:?}" "${BIN:?}"
THREADS=${THREADS:-8}
CHUNKS=${CHUNKS:-4}
STEPS=${STEPS:-selftest identity random moeids}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
PY_IMAGE=${PY_IMAGE:-python:3.12-slim-bookworm}
TOOL=$REPO/scripts/gguf_tier.py
mounts=""
for d in $(printf '%s\n' "$WORK" "$REPO" "$(dirname "$BIN")" ${BASE_BIN:+"$BASE_BIN"} ${MOUNTS:-} | sort -u); do
    mounts="$mounts -v $d:$d"
done
D="--rm --cpu-shares 256 --label pascal-quants=1 ${CPUS:+--cpuset-cpus $CPUS}"
mkdir -p "$WORK/checks"
touch "$WORK/checks.tsv"

log()     { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
verdict() { printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$3" | tee -a "$WORK/checks.tsv"; }
# shellcheck disable=SC2086
box()     { docker run $D $mounts "$@"; }
py()      { box "$PY_IMAGE" nice -n 19 python3 -I "$@"; }
pyin()    { box -i "$PY_IMAGE" nice -n 19 python3 -I "$@"; } # a script on stdin
# the lines of a llama-perplexity log that hold its result
result()  { grep -E "^(Mean    KLD|Maximum KLD|Same top p|Mean PPL|Final estimate)" "$1"; }

if step selftest; then
    log "selftest"
    K=$WORK/checks/selftest
    rm -rf "$K" && mkdir -p "$K"
    out=$(py "$REPO/scripts/gguf_tier_test.py" "$K/t" 2>&1 | tail -n 1)
    verdict selftest "$([ "$out" = ok ] && echo OK || echo FAIL)" "$out"
    i=0
    while IFS='|' read -r what from to; do
        i=$((i + 1))
        m=$K/m$i
        mkdir -p "$m"
        cp "$REPO/scripts/gguf_reshard.py" "$REPO/scripts/gguf_prune.py" "$REPO/scripts/gguf_tier_test.py" "$m/"
        sed "s/$from/$to/" "$TOOL" > "$m/gguf_tier.py"
        if cmp -s "$m/gguf_tier.py" "$TOOL"; then verdict "mutation $what" FAIL "the edit did not apply"; continue; fi
        if py "$m/gguf_tier_test.py" "$m/t" > "$m/out" 2>&1; then verdict "mutation $what" FAIL "not caught"
        else verdict "mutation $what" OK "caught: $(tail -n 1 "$m/out" | cut -c1-100)"; fi
    done << 'EOF'
the tier map off by one|tier\[e\] = float(len(hot\[l\]) + s)|tier[e] = float(len(hot[l]) + s + 1)
cold slabs in id order|cpos = cold_src.slabs(name, layer, cold\[layer\])|cpos = cold_src.slabs(name, layer, sorted(cold[layer]))
ties to the higher id|order = \[e for _v, e in sorted(per\[layer\])\]|order = [e for _v, e in sorted(per[layer], key=lambda r: (r[0], -r[1]))]
the forced hot ids ignored|keep = fl . set(free)|keep = set(order[:n])
--cold-only ignored|if cold_only and not cold_only.search(name):|if False:
split.tensors.count not raised|bump = i == 0 or any|bump = False and any
the importance counts not cut|suffix in ("in_sum2", "in_sum", "counts")|suffix in ("in_sum2", "in_sum")
verify skips the slabs|assert same(src, start + p \* slab, fn, pn + k \* slab, n \* slab)|assert True or same(src, start + p * slab, fn, pn + k * slab, n * slab)
EOF
fi

if step identity; then
    log "identity"
    : "${BASE_BIN:?identity needs BASE_BIN}"
    K=$WORK/checks/identity
    rm -rf "$K" && mkdir -p "$K"
    # <name> <bin dir>: the base build writes the logits, this build compares
    archs() {
        box -e LLAMA_ARCHS_LOGITS="$K/logits-" "$IMAGE" nice -n 19 "$2/test-llama-archs" -b CPU -s 1 2>&1 \
            | sed 's/\x1b\[[0-9;]*m//g' > "$K/$1.log"
        echo "${PIPESTATUS[0]}" > "$K/$1.rc"
    }
    archs base "$BASE_BIN"
    # architectures that share a logits file compare against each other already in the base run
    grep 'logits against' "$K/base.log" | grep -v ', 0 of ' > "$K/base_self.txt"
    archs coldtier "$BIN"
    n=$(grep -c 'logits against' "$K/coldtier.log")
    bad=$(grep 'logits against' "$K/coldtier.log" | grep -v ', 0 of ' | grep -vxFf "$K/base_self.txt" | cut -d: -f1 | tr '\n' ' ')
    nb=$(echo "$bad" | wc -w)
    # the test's own summary must also be the base build's (the same tests fail in both)
    sb=$(tail -n 1 "$K/base.log") sc=$(tail -n 1 "$K/coldtier.log")
    verdict identity "$([ "$n" -gt 0 ] && [ "$nb" -eq 0 ] && [ "$sb" = "$sc" ] && echo OK || echo FAIL)" \
        "$nb of $n architectures' logits differ from the base build${bad:+: $bad}; base: $sb; this build: $sc"
    rm -f "$K"/logits-*.f32
fi

if step random; then
    log "random"
    Q4E='ffn_(gate|up|gate_up|down)_exps=q4_0'  # the random model's experts are a fused gate_up and a down
    : "${TEXT:?random needs TEXT}"
    M=$WORK/random K=$WORK/checks/random
    mkdir -p "$M" && rm -rf "$K" && mkdir -p "$K"
    if [ ! -f "$M/qwen4exp-moe.gguf" ]; then
        box -e LLAMA_TEST_ARCHS_N_VOCAB=4096 -e LLAMA_TEST_ARCHS_N_EXPERT=32,8 "$IMAGE" \
            "$BIN/test-llama-archs" -a qwen4exp -o "$M" > "$M/archs.log" 2>&1
    fi
    # a sharper copy (weights drawn with stdev 0.6, not 0.1), whose experts move the output far more than rounding
    # does: the CUDA check's model (run/cuda_check.sh)
    if [ ! -f "$M/qwen4exp-moe-sharp.gguf" ]; then
        mkdir -p "$M/sharp"
        box -e LLAMA_TEST_ARCHS_N_VOCAB=4096 -e LLAMA_TEST_ARCHS_N_EXPERT=32,8 "$IMAGE" \
            "$BIN/test-llama-archs" -a qwen4exp -o "$M/sharp" -d 0.6 > "$M/sharp/archs.log" 2>&1 &&
            mv "$M/sharp/qwen4exp-moe.gguf" "$M/qwen4exp-moe-sharp.gguf"
    fi
    if [ ! -f "$M/qwen4exp-moe-q4e.gguf" ]; then
        # llama-quantize applies --tensor-type only under a quantized default type: Q8_0, every other matrix kept F32
        box "$IMAGE" "$BIN/llama-quantize" --allow-requantize --tensor-type "$Q4E" --tensor-type '.=f32' \
            "$M/qwen4exp-moe.gguf" "$M/qwen4exp-moe-q4e.gguf" Q8_0 "$THREADS" > "$M/quantize.log" 2>&1
    fi
    n_q4=$(grep -c "converting to q4_0" "$M/quantize.log" || true)
    verdict "random q4_0 experts" "$([ "$n_q4" -eq 4 ] && echo OK || echo FAIL)" "$n_q4 of 4 expert weights converted to q4_0"
    # --no-repack: x86's repacked Q4_0 kernels compute an expert's tokens four at a time or one by one depending on how
    # many it gets, and a tier's slot 0 also gets the other tier's picks, so with them the tiered file rounds a little
    # differently (perplexity 4104.52 against 4104.34 at hot 24); without them every arm must match byte for byte
    ppl() { box "$IMAGE" nice -n 19 "$BIN/llama-perplexity" -m "$1" -f "$TEXT" -c 128 -b 128 --chunks "$CHUNKS" \
                -t "$THREADS" --no-repack --kl-divergence-base "$2" "${@:3}"; }
    # split <model> <hot> <dir>: a seeded random ranking of each layer's experts, hot.tsv and cold.tsv in <dir>
    split() {
        pyin - "$1" "$3/experts.tsv" "$REPO/scripts" "$2" << 'PY'
import random, sys
sys.path.insert(0, sys.argv[3])
from gguf_reshard import Header
h = Header(sys.argv[1])
layers = sorted({int(t[0].split(".")[1]) for t in h.tensors if t[0].endswith("_exps.weight")})
rng = random.Random(238)
with open(sys.argv[2], "w") as f:
    f.write("layer\texpert\troutings\n")
    for l in layers:
        f.writelines(f"{l}\t{e}\t{rng.randrange(1000)}\n" for e in range(32))
PY
        py "$TOOL" split --experts "$3/experts.tsv" --hot "$2" --out-dir "$3" > /dev/null
    }
    for m in qwen4exp-moe qwen4exp-moe-q4e qwen4exp-moe-sharp; do
        src=$M/$m.gguf
        ppl "$src" "$K/$m.base.kld" > "$K/$m.base.log" 2>&1
        ppl "$src" "$K/$m.base.kld" --kl-divergence > "$K/$m.self.log" 2>&1
        for hot in 24 31 8; do
            d=$K/$m-h$hot
            mkdir -p "$d"
            split "$src" "$hot" "$d"
            py "$TOOL" tier --hot-ids "$d/hot.tsv" --cold-ids "$d/cold.tsv" --cold "$src" --out-dir "$d" "$src" > "$d/tier.log" 2>&1 &&
                py "$TOOL" verify "$src" --new "$d/$m.gguf" --hot-ids "$d/hot.tsv" --cold-ids "$d/cold.tsv" --cold "$src" >> "$d/tier.log" 2>&1
            rc=$?
            ppl "$d/$m.gguf" "$d/tiered.kld" > "$d/tiered.log" 2>&1
            ppl "$d/$m.gguf" "$K/$m.base.kld" --kl-divergence > "$d/vs_base.log" 2>&1
            if [ $rc -eq 0 ] && cmp -s "$K/$m.base.kld" "$d/tiered.kld" && [ -n "$(result "$K/$m.self.log")" ] &&
                [ "$(result "$K/$m.self.log")" = "$(result "$d/vs_base.log")" ]; then
                verdict "random $m hot $hot" OK "verify ok; the KL base files are the same bytes; $(result "$d/vs_base.log" | tr -s ' ' | tr '\n' ' ')"
            else
                verdict "random $m hot $hot" FAIL "$d"
            fi
        done
        # the negative control: hot 24, with --cold-in claiming the source holds its experts in reverse, so every cold
        # slab comes from the wrong expert
        d=$K/$m-h24 c=$K/$m-mismatch
        mkdir -p "$c"
        pyin - "$d/cold.tsv" "$c/cold_in.tsv" << 'PY'
import sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
with open(sys.argv[2], "w") as f:
    f.write("layer\tids\n" + "".join(f"{l}\t{','.join(str(e) for e in reversed(range(32)))}\n" for l, _ids in rows))
PY
        py "$TOOL" tier --hot-ids "$d/hot.tsv" --cold-ids "$d/cold.tsv" --cold "$src" --cold-in "$c/cold_in.tsv" \
            --out-dir "$c" "$src" > "$c/tier.log" 2>&1
        ppl "$c/$m.gguf" "$K/$m.base.kld" --kl-divergence > "$c/vs_base.log" 2>&1
        if grep -q "^Mean    KLD" "$c/vs_base.log" && [ "$(result "$K/$m.self.log")" != "$(result "$c/vs_base.log")" ]; then
            verdict "random $m mismatch" OK "cold slabs of the wrong experts change the result: $(result "$c/vs_base.log" | tr -s ' ' | tr '\n' ' ')"
        else
            verdict "random $m mismatch" FAIL "$c"
        fi
    done
    # F32 hot experts with Q4_0 cold ones: extract the cold ids, llama-quantize them, tier
    m=qwen4exp-moe src=$M/$m.gguf d=$K/$m-h24 x=$K/$m-mixed
    mkdir -p "$x"
    py "$TOOL" extract --ids "$d/cold.tsv" --out-dir "$x" "$src" > "$x/extract.log" 2>&1
    box "$IMAGE" "$BIN/llama-quantize" --allow-requantize --tensor-type "$Q4E" --tensor-type '.=f32' \
        "$x/experts.gguf" "$x/experts-q4_0.gguf" Q8_0 "$THREADS" > "$x/quantize.log" 2>&1
    py "$TOOL" tier --hot-ids "$d/hot.tsv" --cold-ids "$d/cold.tsv" --cold "$x/experts-q4_0.gguf" --cold-in "$x/ids.tsv" \
        --out-dir "$x" "$src" > "$x/tier.log" 2>&1 &&
        py "$TOOL" verify "$src" --new "$x/$m.gguf" --hot-ids "$d/hot.tsv" --cold-ids "$d/cold.tsv" \
            --cold "$x/experts-q4_0.gguf" --cold-in "$x/ids.tsv" >> "$x/tier.log" 2>&1
    rc=$?
    ppl "$x/$m.gguf" "$K/$m.base.kld" --kl-divergence > "$x/vs_f32.log" 2>&1
    ppl "$x/$m.gguf" "$K/qwen4exp-moe-q4e.base.kld" --kl-divergence > "$x/vs_q4e.log" 2>&1
    if [ $rc -eq 0 ] && grep -q "^Mean    KLD" "$x/vs_f32.log" && grep -q "^Mean    KLD" "$x/vs_q4e.log" &&
        [ "$(result "$K/$m.self.log")" != "$(result "$x/vs_f32.log")" ] &&
        [ "$(result "$K/qwen4exp-moe-q4e.self.log")" != "$(result "$x/vs_q4e.log")" ]; then
        verdict "random mixed f32+q4_0" OK "verify ok; runs; against F32: $(result "$x/vs_f32.log" | grep -E 'Mean    KLD|Same top' | tr -s ' ' | tr '\n' ' '); against all-Q4_0: $(result "$x/vs_q4e.log" | grep -E 'Mean    KLD|Same top' | tr -s ' ' | tr '\n' ' ')"
    else
        verdict "random mixed f32+q4_0" FAIL "$x"
    fi
fi

if step moeids; then
    log "moeids"
    K=$WORK/checks/moeids R=$WORK/checks/random M=$WORK/random
    rm -rf "$K" && mkdir -p "$K"
    # the random models (run the random step first) untiered and tiered: every MUL_MAT_ID's ids distinct per token
    for f in "$M/qwen4exp-moe.gguf" "$R"/qwen4exp-moe*-h*/qwen4exp-moe*.gguf "$R/qwen4exp-moe-mixed/qwen4exp-moe.gguf"; do
        name=$(basename "$(dirname "$f")")/$(basename "$f" .gguf)
        lg=$K/$(echo "$name" | tr / _).log
        box "$IMAGE" nice -n 19 "$BIN/test-moe-ids" -m "$f" -t "$THREADS" > "$lg" 2>&1
        rc=$?
        verdict "moeids $name" "$([ $rc -eq 0 ] && echo OK || echo FAIL)" "$(grep -E '^test-moe-ids:|checks,' "$lg" | tr '\n' ' ')"
    done
fi

if step real; then
    log "real"
    : "${FILES:?real needs FILES}" "${TEXT:?}" "${BASE_KLD:?}"
    K=$WORK/checks/real
    mkdir -p "$K"
    [ -f "$WORK/cpu_kld.tsv" ] || printf 'file\tchunks\tmean_kld\tsame_top_pct\tppl_ratio\n' > "$WORK/cpu_kld.tsv"
    for f in $FILES; do
        name=${f%%=*} first=${f#*=}
        box "$IMAGE" nice -n 19 "$BIN/llama-perplexity" -m "$first" -f "$TEXT" -c 512 -b 512 --chunks "$CHUNKS" \
            -t "$THREADS" --kl-divergence-base "$BASE_KLD" --kl-divergence > "$K/$name.log" 2>&1
        kld=$(grep -E "^Mean    KLD" "$K/$name.log" | awk '{print $3 " " $4 " " $5}')
        top=$(grep -E "^Same top p" "$K/$name.log" | awk '{print $4 " " $5 " " $6}')
        ratio=$(grep -E "^Mean PPL\(Q\)/PPL\(base\)" "$K/$name.log" | awk '{print $3 " " $4 " " $5}')
        if [ -n "$kld" ]; then
            printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$CHUNKS" "$kld" "$top" "$ratio" >> "$WORK/cpu_kld.tsv"
            verdict "real $name" OK "KLD $kld, same top $top"
        else
            verdict "real $name" FAIL "$K/$name.log"
        fi
    done
fi
