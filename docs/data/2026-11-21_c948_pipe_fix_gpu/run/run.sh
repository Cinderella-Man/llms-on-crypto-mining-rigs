#!/usr/bin/env bash
# C948 (Q1687): the GPU half of C932's check (the Flash-Next pipe fix, fork branch pascal-p106-v5-pipe-logits
# 80d83fa1a, against its base pascal-p106-v5 3dfad876f), on a build machine with three newer NVIDIA cards that run the
# rig build's compute 6.1 PTX. Correctness only, nothing timed. Every check is bench/pipespec/check_identity.sh (each
# pipelined job against the plain one: the tokens, every token's logits, the KV state after generation and a probe)
# in its own container, once with REF and once with NEW:
#   random  on the first card: random Flash-Next (qwen4exp) and Flint models made by NEW's test-llama-archs (seed 1, a
#           4,096-token vocabulary), 96 tokens, depths 1 and 3, 0 and 50 % wrong guesses; REF, REF with
#           LLAMA_GDN_FUSE=0 (the unfused conv window, the control) and NEW
#   fnreal  Swift GSQ-RCO Q2_0 (the Flash-Next v5 preset's file) on the three cards with C932's real-file settings:
#           the chat template, the code prompt, 64 tokens, depth 1, 0 and 50 % wrong guesses
#   fn      the same file with the P106 check's settings: the rewrite prompt, 128 tokens, depths 1 and 3
#   fnp     Swift Qwen3.8-Flash-Next Pascal Q2 GGUF (pmix-q2-fast) with the fn settings
#   flint   Flint Pascal Q8 GGUF on the three cards with its preset's batch sizes, the code prompt, 256 tokens, depth 4
#           (the preset's), 0 and 50 % wrong guesses
# Every switch is at its default (none set), as in C932's checks.
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env:
#   WORK       a folder for the outputs (mounted read-write into every container)
#   NEW, REF   bin/ folders of the two rig builds (NEW = the fix, REF = its base)
#   PS         a copy of bench/pipespec (check_identity.sh and prompts/)
#   FN         Swift GSQ-RCO Q2_0's first shard; FNP Swift Qwen3.8-Flash-Next Pascal Q2 GGUF's first shard
#   FLINT      Flint Pascal Q8 GGUF
#   MOUNTS     folders the containers mount read-only at the same paths (the builds, the models, PS)
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT  a CUDA check run before the GPU steps as "$PREFLIGHT -g <ids>" (optional)
#   GPUS (0,1,2), TS (1,1,1), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04), STEPS ("random fnreal fn fnp flint"),
#   TIMEOUT (2400 s a check), NAME (c948: the containers' name prefix), FN_ARGS, FLINT_ARGS
# Output: WORK/out/<step>-<model>-<build>/ (check_identity's folder) and .log (its table), WORK/out/summary.tsv (step,
#   model, build, exit code, pipelined jobs, jobs differing, jobs with other tokens, plain's tokens and logits hashes);
#   WORK/DONE at the end.
set -uo pipefail
: "${WORK:?}" "${NEW:?}" "${REF:?}" "${PS:?}" "${FN:?}" "${FNP:?}" "${FLINT:?}"
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPUS=${GPUS:-0,1,2}
TS=${TS:-1,1,1}
STEPS=${STEPS:-random fnreal fn fnp flint}
TIMEOUT=${TIMEOUT:-2400}
NAME=${NAME:-c948}
PREFLIGHT=${PREFLIGHT:-}
FN_ARGS=${FN_ARGS:--ngl 99 -fa on -c 4096 -ts $TS -fit off}
FLINT_ARGS=${FLINT_ARGS:--ngl 99 -fa on -c 4096 -b 512 -ub 128 -ts $TS}
IFS=, read -ra G <<< "$GPUS"
[ ${#G[@]} = 3 ] || { echo "GPUS must name three cards"; exit 1; }

say() { echo "$(date +%H:%M:%S) $*"; }
step() { [[ " $STEPS " == *" $1 "* ]]; }
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$WORK:$WORK")
# the PTX JIT's cache, shared by every check (the cards compile the build's compute 6.1 PTX once)
JIT=(-e "CUDA_CACHE_PATH=$WORK/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)
# shellcheck disable=SC2206
DR=(docker run --rm ${DOCKER_ARGS:-} "${VOL[@]}" "${JIT[@]}")
OUT=$WORK/out
mkdir -p "$OUT" "$WORK/jitcache" "$WORK/models"
rm -f "$WORK/DONE"
md5sum "$NEW/llama-pipespec" "$NEW/test-llama-archs" "$REF/llama-pipespec" "$PS/check_identity.sh" > "$WORK/md5.txt"
[ -f "$OUT/summary.tsv" ] ||
    printf 'step\tmodel\tbuild\trc\tpipelined jobs\tjobs differing\tjobs with other tokens\tplain tokens hash\tplain logits hash\n' > "$OUT/summary.tsv"
say "start: steps '$STEPS', cards $GPUS, NEW=$NEW REF=$REF"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS" | tr '\n' ' '; echo
[ -n "$PREFLIGHT" ] && { $PREFLIGHT -g "$GPUS" > "$WORK/preflight.txt" 2>&1; say "preflight rc=$?"; }

plainfield() { grep -o 'name=plain .*' "$1/results.txt" 2> /dev/null | grep -o " $2=[^ ]*" | cut -d= -f2; }

ci() {  # <step> <model tag> <build tag> <cards> <bin> <model> [VAR=value ...]: one check_identity.sh in a container
    local s=$1 tag=$2 b=$3 cards=$4 bin=$5 m=$6; shift 6
    local o=$OUT/$s-$tag-$b c=$NAME-$s-$tag-$b
    rm -rf "$o"; mkdir -p "$o"
    docker rm -f "$c" > /dev/null 2>&1
    say "$s $tag $b: start"
    "${DR[@]}" --name "$c" --gpus "\"device=$cards\"" "$IMG" \
        env "$@" BIN="$bin" MODEL="$m" SPEED=0 TIMEOUT="$TIMEOUT" OUT="$o" bash "$PS/check_identity.sh" > "$o.log" 2>&1
    local rc=$?
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$s" "$tag" "$b" "$rc" \
        "$(grep -E ' (same|DIFF)$' "$o.log" | grep -vc '^plain ')" "$(grep -c ' DIFF$' "$o.log")" \
        "$(awk '$1 != "plain" && $1 != "run" && $8 == "DIFF"' "$o.log" | wc -l)" \
        "$(plainfield "$o" tokens)" "$(plainfield "$o" logits)" >> "$OUT/summary.tsv"
    say "$s $tag $b: rc=$rc; $(tail -n 1 "$OUT/summary.tsv" | cut -f5-7 | tr '\t' ' ') (jobs, differing, other tokens)"
}

if step random; then
    for a in qwen4exp flint; do
        d=$WORK/models/$a
        mkdir -p "$d"
        "${DR[@]}" --name "$NAME-gen-$a" --gpus "\"device=${G[0]}\"" -w "$d" "$IMG" \
            env LLAMA_TEST_ARCHS_N_VOCAB=4096 timeout 600 "$NEW/test-llama-archs" -a "^$a\$" -s 1 -o . > "$d/gen.log" 2>&1
        m=$(ls "$d/$a"*.gguf 2> /dev/null | head -n 1)
        [ -n "$m" ] || { say "random $a: no model made (see $d/gen.log)"; continue; }
        md5sum "$m" >> "$OUT/models.md5"
        for b in ref ref-fuse0 new; do
            bin=$REF; [ "$b" = new ] && bin=$NEW
            extra=X=1; [ "$b" = ref-fuse0 ] && extra=LLAMA_GDN_FUSE=0
            ci random "$a" "$b" "${G[0]}" "$bin" "$m" "$extra" CHAT=0 PROMPT="$PS/prompts/rewrite.txt" N=96 \
                DEPTHS="1 3" MISSES="0 0.5" ARGS="-ngl 99 -fa on -c 2048"
        done
    done
fi

if step fnreal; then
    for b in ref new; do
        bin=$REF; [ "$b" = new ] && bin=$NEW
        ci fnreal swift-q2 "$b" "$GPUS" "$bin" "$FN" CHAT=1 PROMPT="$PS/prompts/code.txt" N=64 DEPTHS=1 MISSES="0 0.5" \
            ARGS="$FN_ARGS"
    done
fi

if step fn; then
    for b in ref new; do
        bin=$REF; [ "$b" = new ] && bin=$NEW
        ci fn swift-q2 "$b" "$GPUS" "$bin" "$FN" CHAT=1 PROMPT="$PS/prompts/rewrite.txt" N=128 DEPTHS="1 3" \
            MISSES="0 0.5" ARGS="$FN_ARGS"
    done
fi

if step fnp; then
    for b in ref new; do
        bin=$REF; [ "$b" = new ] && bin=$NEW
        ci fnp pmix-q2-fast "$b" "$GPUS" "$bin" "$FNP" CHAT=1 PROMPT="$PS/prompts/rewrite.txt" N=128 DEPTHS="1 3" \
            MISSES="0 0.5" ARGS="$FN_ARGS"
    done
fi

if step flint; then
    for b in ref new; do
        bin=$REF; [ "$b" = new ] && bin=$NEW
        ci flint flint-q8 "$b" "$GPUS" "$bin" "$FLINT" CHAT=1 PROMPT="$PS/prompts/code.txt" N=256 DEPTHS=4 \
            MISSES="0 0.5" ARGS="$FLINT_ARGS"
    done
fi

say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | paste -sd' ')"
cat "$OUT/summary.tsv"
date -Is > "$WORK/DONE"
