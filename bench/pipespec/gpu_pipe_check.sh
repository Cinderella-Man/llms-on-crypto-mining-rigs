#!/usr/bin/env bash
# gpu_pipe_check.sh: pipelined decoding against plain decoding on the cards (C1009), the GPU twin of the pipe step of
# docs/data/2026-11-05_c565_ladder4fix/run/cpu_tests.sh (C935). That step decodes on the processor only, so a fault in a
# CUDA path alone passes it. This one runs the same random models on a CUDA build with every layer on the cards
# (-ngl 99), on one card and split over several, where the pipe's card-only paths run: each architecture's CUDA kernels
# (the gdn_conv cache fusion that held part of C932's fault among them), the hops between cards, a sampler on the device
# and the next token fed on the devices. The GPU check of every branch and merge runs it on the branch's and its base's
# TESTS=1 tarballs.
#
# For each build of BUILDS, in order:
#   models  one container on the first card: for each architecture of ARCHS (default: the four llama_pipe_init accepts
#           as recurrent, qwen4exp qwen35 qwen35moe qwen3next, and flint, a model without recurrent state) the build's
#           test-llama-archs -a '^<arch>$' -s MODEL_SEED with a 4,096-token vocabulary, its MoE model where it writes one
#           (cpu_tests.sh's models: the same files for the same seed); when an arm names q8, also a Q8_0 copy made by
#           QUANT, a llama-quantize (QUANT_ARGS keep the hyper-connection and per-layer-embedding norms F32, which the
#           graph adds and multiplies elementwise). The fork's Pascal matrix-vector kernels take quantized weights only
#           (and BF16), so only the copy reaches them
#   runs    for each layout of LAYOUTS and arm of ARMS, one container with that layout's cards; per architecture:
#             host       check_identity.sh, depths 1 and 3, 0 and 50 % wrong guesses (the oracle), the four guessers and
#                        depth 0, host sampling: every pipelined run's tokens, logits, KV state and probe hashes plain's
#             fed        the same with greedy sampling on the device in every run, plain's too (--pipe-backend-greedy),
#                        the token embedding on the first card (LLAMA_INPUT_ON_GPU=1) and FEED=1, which adds depth 0
#                        with each next token fed on the devices from the last card's sample (llama_pipe_submit_fed);
#                        the logits and probe hashes then hash token ids, while the KV state hash still reads every cell.
#                        llama_pipe_submit_fed refuses, by design, a model whose KV cells keep their token (M-RoPE, as in
#                        qwen35 and qwen35moe, or PLE, as in qwen4exp): there the fed run is left out and said so, and
#                        the other runs are judged
#             recurrent  llama-pipe-recurrent-check: the rollbacks the slot rotation rarely makes (the stash, dropped
#                        guesses, the token in flight dropped and sent again)
#             kernels    one token of llama-completion (-lv 4) on the first model: the fork's card features ggml lists
#                        (MMVQ_PASCAL, MMVF_PASCAL, COMPUTE_6_1), the devices and each one's model buffer (where the
#                        layers went), recorded, not judged
#           Each identity check also records the plain run's tokens and logits hashes, so two arms on the same model show
#           whether different kernels computed them (other logits hashes)
# Layouts: one    the first card of GPUS
#          split  every card of GPUS, the layers in equal shares (-ts 1,1,...), so every token hops between cards: the
#                 embedding on the first card, the output and the device sampler on the last
#          virt   VIRT virtual devices on the first card (GGML_CUDA_DEVICES=VIRT, -ts 1,1,...): the hops on a machine
#                 with one card
#          (default "one split" when GPUS names two or more cards, "one virt" with one)
# Arms: <models>-<setting>; models f32 (as test-llama-archs writes them) or q8 (the Q8_0 copy), setting
#          stock    no switch: on a card that is not compute 6.1, llama.cpp's own kernels
#          p106     the fork's Pascal matrix-vector kernels and its sparse attention forced on with a P106's launch shapes
#                   (P106_ENV), as a P106 runs them (on compute 6.1 they are on without a switch)
#          p106all  p106 plus the fork's other compute-6.1-only card paths (P106ALL_ENV: the attention-block fusion, the
#                   persistent MoE kernel, the Pascal flash-attention tile kernel, the F32 multi-token kernel)
#          (default "f32-stock q8-p106": the CPU step's models on llama.cpp's kernels, and the fork's kernels on the copy)
# Verdicts (summary.tsv): OK when every hash is plain's, DIFFERS when a pipelined run gives other hashes, FAIL when a
# tool stopped before comparing (its exit code and the log's last line), SKIP when the build lacks a tool, INFO for the
# kernels line.
#
# Usage (on the machine with the cards; this script beside check_identity.sh and prompts/rewrite.txt, as in
# bench/pipespec):
#   WORK=<dir> BUILDS='<tag>=<bin dir> ...' [GPUS=0,1,2] [LAYOUTS='one split'] [ARMS='f32-stock q8-p106']
#   [ARCHS='qwen4exp qwen35 qwen35moe qwen3next flint'] [CHECKS='host fed recurrent kernels'] [MODEL_SEED=1] [N=64]
#   [TIMEOUT=900] [VIRT=3] [IMG=nvidia/cuda:12.9.2-devel-ubuntu24.04] [MOUNTS='<folders mounted read-only>']
#   [DOCKER_ARGS='<labels, --cpu-shares>'] [PREFLIGHT=<cuda_preflight.sh>] [NAME=gpupipe] [QUANT_ARGS=...]
#   [P106_ENV=...] [P106ALL_ENV=...] [JITCACHE=<dir>] [QUANT=<llama-quantize>] bash gpu_pipe_check.sh
# A bin dir is a rig tarball's bin/: scripts/build_llamacpp_docker.sh with TESTS=1 packs llama-pipe-recurrent-check and
# test-llama-archs beside llama-pipespec and llama-completion (since 2026-11-24, C984). The rig tarball has no
# llama-quantize, so QUANT (default: the bin dir's own, if it has one) names one that runs in IMG, e.g. a static CPU
# build (cpu_tests.sh with SHARED=OFF and KEEP_BUILD=1); one QUANT for every build gives every build the same copies
# (models.md5). Every tool runs in a container (docker run --gpus, no network) of IMG with the build, this folder, MOUNTS
# and WORK at the same paths.
# MODEL_SEED, the random models' seed, must be a whole number (test-llama-archs -s); the script stops at once, before it
# writes anything, on any other value. The older name SEED is still read, but only when it holds a whole number and
# MODEL_SEED is unset: a caller's own SEED of another kind (a clone's path, say) is left alone and the seed stays 1.
# The PTX JIT's cache is JITCACHE (default WORK/jitcache), shared by every run: a compute 6.1 build on a newer card
# compiles its PTX on first use, so several runs of one build can share a JITCACHE.
# Output: WORK/<tag>/models/<arch>/ (gen.log, the model, q8.gguf and quant.log), WORK/<tag>/models.md5,
#   WORK/<tag>/<arm>-<layout>/ (check_identity's folders <arch>-host/ and <arch>-fed/ with .log their tables,
#   prc-<arch>.log, kernels.log, env.txt, verdicts.tsv), WORK/summary.tsv (tree, arm, layout, arch, check, verdict,
#   detail, plain tokens hash, plain logits hash); done when WORK/DONE exists. Exit status 0 when no verdict is FAIL or
#   DIFFERS.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SELF=$HERE/$(basename "$0")
ARCHS=${ARCHS-qwen4exp qwen35 qwen35moe qwen3next flint}
CHECKS=${CHECKS:-host fed recurrent kernels}
N=${N:-64}
TIMEOUT=${TIMEOUT:-900}
VIRT=${VIRT:-3}

# ---- inside a container: one arm and layout, every architecture ----
if [ "${GPU_PIPE_INNER:-}" = 1 ]; then
    : "${BIN:?}" "${MD:?}" "${KIND:?}" "${OUTD:?}" "${LARGS:?}"
    mkdir -p "$OUTD"
    V=$OUTD/verdicts.tsv
    : > "$V"
    v() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "${5:--}" "${6:--}" >> "$V"; }
    pf() { grep -ao 'name=plain .*' "$1/results.txt" 2> /dev/null | grep -o " $2=[^ ]*" | cut -d= -f2; }
    last() { grep -av '^ *$' "$1" 2> /dev/null | tail -n 1 | cut -c1-140; }
    on() { [[ " $CHECKS " == *" $1 "* ]]; }
    # <arch> <check> <model> [VAR=value ...]: check_identity.sh, judged as cpu_tests.sh's pipe step judges it
    ident() {
        local a=$1 c=$2 m=$3
        shift 3
        local o=$OUTD/$a-$c rc runs nr nd nt why vd refused=""
        rm -rf "$o"
        env ARGS="$LARGS" "$@" BIN="$BIN" MODEL="$m" CHAT=0 SPEED=0 TIMEOUT="$TIMEOUT" PROMPT="$HERE/prompts/rewrite.txt" OUT="$o" \
            N="$N" DEPTHS="1 3" MISSES="0 0.5" bash "$HERE/check_identity.sh" > "$o.log" 2>&1
        rc=$?
        # check_identity's table: one line per run, the verdict last, column 8 the tokens; plain is the reference
        runs=$(grep -aE ' (same|DIFF)$' "$o.log" | grep -av '^plain ')
        nr=$(echo "$runs" | grep -acE ' (same|DIFF)$')
        nd=$(echo "$runs" | grep -ac ' DIFF$')
        nt=$(echo "$runs" | awk '$NF == "DIFF" && $8 == "DIFF"' | wc -l)
        why="pipelined runs differing $nd of $nr"
        [ "$nd" -gt 0 ] && why="$why ($nt with other tokens): $(echo "$runs" | awk '$NF == "DIFF" {print $1}' | head -n 5 | tr '\n' ' ')"
        # llama_pipe_submit_fed refuses, by design, a model whose KV cells keep their token (M-RoPE, PLE): its fed run
        # stops at once, and the other runs (the device sampler, the embedding on the card) are still judged
        if grep -aq 'fed tokens need a model whose KV cells keep no token' "$o/run.log" 2> /dev/null; then
            refused="; pipe-d0-fed refused by design (the model's KV cells keep their token: M-RoPE or PLE)"
        fi
        if [ $rc = 0 ]; then vd=OK
        elif grep -aq '^plain ' "$o.log" && [ "$nd" -gt 0 ]; then vd=DIFFERS
        elif [ -n "$refused" ] && grep -aq "^only $((nr + 1)) of $((nr + 2)) runs finished" "$o.log"; then vd=OK
        else vd=FAIL why="$why; $(last "$o.log")"; fi
        why="$why$refused"
        v "$a" "$c" "$vd" "check_identity.sh rc=$rc: $why; N=$N, depths 1 3, misses 0 0.5${*:+; $*}" "$(pf "$o" tokens)" "$(pf "$o" logits)"
    }
    have() { [ -x "$BIN/$1" ]; }
    {
        echo "kind=$KIND args=$LARGS"
        env | grep -E '^(GGML_|LLAMA_|CUDA_VISIBLE|NVIDIA_VISIBLE)' | sort
    } > "$OUTD/env.txt"
    for a in $ARCHS; do
        if [ "$KIND" = q8 ]; then m=$MD/$a/q8.gguf; else m=$MD/$a/$(cat "$MD/$a/model.txt" 2> /dev/null); fi
        if [ ! -f "$m" ]; then
            v "$a" model FAIL "no $KIND model ($m): see $MD/$a/gen.log and quant.log"
            continue
        fi
        if on host; then
            if have llama-pipespec; then ident "$a" host "$m"; else v "$a" host SKIP "no llama-pipespec in $BIN"; fi
        fi
        if on fed; then
            if have llama-pipespec; then
                ident "$a" fed "$m" FEED=1 LLAMA_INPUT_ON_GPU=1 ARGS="$LARGS --pipe-backend-greedy"
            else
                v "$a" fed SKIP "no llama-pipespec in $BIN"
            fi
        fi
        if on recurrent; then
            if have llama-pipe-recurrent-check; then
                # shellcheck disable=SC2086
                timeout "$TIMEOUT" "$BIN/llama-pipe-recurrent-check" -m "$m" -f "$HERE/prompts/rewrite.txt" -n "$N" $LARGS \
                    > "$OUTD/prc-$a.log" 2>&1
                rc=$?
                line=$(grep -a -o 'PIPE-RECURRENT-CHECK.*' "$OUTD/prc-$a.log" | tail -n 1)
                v "$a" recurrent "$(case $rc in 0) echo OK ;; 2) echo DIFFERS ;; *) echo FAIL ;; esac)" \
                    "llama-pipe-recurrent-check rc=$rc: ${line:-$(last "$OUTD/prc-$a.log")}"
            else
                v "$a" recurrent SKIP "no llama-pipe-recurrent-check in $BIN"
            fi
        fi
    done
    if on kernels; then
        m=""
        for a in $ARCHS; do
            if [ "$KIND" = q8 ]; then f=$MD/$a/q8.gguf; else f=$MD/$a/$(cat "$MD/$a/model.txt" 2> /dev/null); fi
            [ -f "$f" ] && { m=$f; break; }
        done
        if have llama-completion && [ -n "$m" ]; then
            # shellcheck disable=SC2086
            timeout 300 "$BIN/llama-completion" -m "$m" -p 'The quick brown fox' -n 1 $LARGS -no-cnv --temp 0 -lv 4 \
                > "$OUTD/kernels.log" 2>&1 < /dev/null
            rc=$?
            f=$(grep -aoE '(MMV[QF]_PASCAL|COMPUTE_6_1) = 1' "$OUTD/kernels.log" | sort -u | paste -sd' ')
            nd=$(grep -ac 'using device CUDA' "$OUTD/kernels.log")
            mb=$(grep -aoE 'CUDA[0-9]+ model buffer size = *[0-9.]+ MiB' "$OUTD/kernels.log" | sed -E 's/ model buffer size = */ /' |
                paste -sd, | sed 's/,/, /g')
            v all kernels INFO "llama-completion rc=$rc on $(basename "$(dirname "$m")")/$(basename "$m"): ggml lists ${f:-none of MMVQ_PASCAL, MMVF_PASCAL, COMPUTE_6_1}; $nd CUDA devices, model buffers ${mb:-none}"
        else
            v all kernels SKIP "no llama-completion in $BIN, or no model"
        fi
    fi
    exit 0
fi

# ---- the driver ----
: "${WORK:?WORK=<output folder>}" "${BUILDS:?BUILDS=<tag>=<bin dir> ...}"
# the random models' seed: MODEL_SEED, else SEED only when it is a whole number, else 1 (a caller's SEED of another
# kind, once a clone's path, made every test-llama-archs -s stop in stoull: C1022)
SEEDNOTE=""
if [ -n "${MODEL_SEED:-}" ]; then
    SEEDNOTE="from MODEL_SEED"
elif [[ "${SEED:-}" =~ ^[0-9]+$ ]]; then
    MODEL_SEED=$SEED SEEDNOTE="from SEED"
else
    MODEL_SEED=1 SEEDNOTE="the default"
    [ -n "${SEED:-}" ] && SEEDNOTE="the default; SEED='$SEED' is not a whole number, so it is not taken as the seed"
fi
if ! [[ "$MODEL_SEED" =~ ^[0-9]{1,19}$ ]]; then
    echo "gpu_pipe_check.sh: MODEL_SEED='$MODEL_SEED' is not a whole number of at most 19 digits (test-llama-archs -s takes one); nothing was run" >&2
    exit 2
fi
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
GPUS=${GPUS:-0,1,2}
IFS=, read -ra G <<< "$GPUS"
NG=${#G[@]}
if [ "$NG" -ge 2 ]; then LAYOUTS=${LAYOUTS:-one split}; else LAYOUTS=${LAYOUTS:-one virt}; fi
ARMS=${ARMS:-f32-stock q8-p106}
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
NAME=${NAME:-gpupipe}
PREFLIGHT=${PREFLIGHT:-}
JITCACHE=${JITCACHE:-$WORK/jitcache}
QUANT=${QUANT:-}
mkdir -p "$JITCACHE"
QUANT_ARGS=${QUANT_ARGS:---tensor-type hc_=f32 --tensor-type ple_norm=f32}
P106_ENV=${P106_ENV:-GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_FATTN_SPARSE_PASCAL=2}
P106ALL_ENV=${P106ALL_ENV:-$P106_ENV GGML_CUDA_ATTN_BLOCK=2 GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_MMVF_F32_MT=2}
BASE_ARGS="-ngl 99 -fa on -c 2048 -fit off"
ones() { local s=1 i; for ((i = 1; i < $1; i++)); do s="$s,1"; done; echo "$s"; }

say() { echo "$(date +%H:%M:%S) $*"; }
VOL=(-v "$HERE:$HERE:ro" -v "$WORK:$WORK")
[[ "$JITCACHE" == "$WORK"/* ]] || VOL+=(-v "$JITCACHE:$JITCACHE")
[ -n "$QUANT" ] && VOL+=(-v "$(dirname "$QUANT"):$(dirname "$QUANT"):ro")
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
# shellcheck disable=SC2206
DR=(docker run --rm --network none ${DOCKER_ARGS:-} "${VOL[@]}" -e "CUDA_CACHE_PATH=$JITCACHE" -e CUDA_CACHE_MAXSIZE=4294967296)
rm -f "$WORK/DONE"
[ -f "$WORK/summary.tsv" ] ||
    printf 'tree\tarm\tlayout\tarch\tcheck\tverdict\tdetail\tplain tokens hash\tplain logits hash\n' > "$WORK/summary.tsv"
md5sum "$SELF" "$HERE/check_identity.sh" "$HERE/prompts/rewrite.txt" > "$WORK/scripts.md5"
needq8=0; [[ " $ARMS " == *" q8-"* ]] && needq8=1
say "gpu pipe check: builds '$BUILDS', cards $GPUS, layouts '$LAYOUTS', arms '$ARMS', archs '$ARCHS', checks '$CHECKS', model seed $MODEL_SEED ($SEEDNOTE)"
echo "$MODEL_SEED" > "$WORK/model_seed.txt"
if [ -n "$PREFLIGHT" ]; then
    $PREFLIGHT -g "$GPUS" > "$WORK/preflight.txt" 2>&1
    rc=$?
    say "preflight rc=$rc"
    if [ $rc != 0 ]; then
        printf '%s\t-\t-\t-\tpreflight\tFAIL\t%s\t-\t-\n' all "$PREFLIGHT -g $GPUS rc=$rc: $(tail -n 1 "$WORK/preflight.txt")" >> "$WORK/summary.tsv"
        date -Is > "$WORK/DONE"
        exit 1
    fi
fi

# <tree> <bin> <arm> <layout> <cards>: one container, every architecture
job() {
    local tag=$1 bin=$2 arm=$3 layout=$4 cards=$5
    local kind=${arm%%-*} setting=${arm#*-} senv="" largs=$BASE_ARGS c=$NAME-$1-$3-$4
    local o=$WORK/$tag/$arm-$layout
    case $setting in
    stock) senv="" ;;
    p106) senv=$P106_ENV ;;
    p106all) senv=$P106ALL_ENV ;;
    *) mkdir -p "$o"; printf 'all\tarm\tFAIL\tunknown setting %s\t-\t-\n' "$setting" > "$o/verdicts.tsv"; return ;;
    esac
    case $layout in
    one) ;;
    split) largs="$largs -ts $(ones "$NG")" ;;
    virt) largs="$largs -ts $(ones "$VIRT")"; senv="$senv GGML_CUDA_DEVICES=$VIRT" ;;
    esac
    rm -rf "$o"
    mkdir -p "$o"
    docker rm -f "$c" > /dev/null 2>&1
    # shellcheck disable=SC2086
    timeout $(( (3 * $(echo $ARCHS | wc -w) + 2) * TIMEOUT )) "${DR[@]}" --name "$c" --gpus "\"device=$cards\"" "$IMG" \
        env $senv GPU_PIPE_INNER=1 BIN="$bin" MD="$WORK/$tag/models" KIND="$kind" OUTD="$o" LARGS="$largs" \
        ARCHS="$ARCHS" CHECKS="$CHECKS" N="$N" TIMEOUT="$TIMEOUT" bash "$SELF" > "$o/container.log" 2>&1
    local rc=$?
    docker rm -f "$c" > /dev/null 2>&1
    [ $rc = 0 ] || printf 'all\tcontainer\tFAIL\trc=%s (124: timed out): %s\t-\t-\n' "$rc" "$(tail -n 1 "$o/container.log")" >> "$o/verdicts.tsv"
    say "$tag $arm $layout (cards $cards): rc=$rc; $(cut -f3 "$o/verdicts.tsv" 2> /dev/null | sort | uniq -c | awk '{printf "%s %s; ", $1, $2}')"
}

for spec in $BUILDS; do
    tag=${spec%%=*} bin=${spec#*=}
    T=$WORK/$tag
    mkdir -p "$T/models"
    say "$tag: $bin"
    (cd "$bin" && md5sum llama-pipespec llama-pipe-recurrent-check test-llama-archs llama-completion 2>&1) > "$T/bin.md5"
    quant=${QUANT:-$bin/llama-quantize}
    [ "$needq8" = 1 ] && md5sum "$quant" >> "$T/bin.md5" 2>&1
    if [ ! -x "$bin/test-llama-archs" ]; then
        printf '%s\t-\t-\t-\tmodels\tSKIP\tno test-llama-archs in %s\t-\t-\n' "$tag" "$bin" >> "$WORK/summary.tsv"
        continue
    fi
    # the models: each architecture's file from this build's test-llama-archs, and its Q8_0 copy
    # shellcheck disable=SC2086
    timeout $(( 600 * ($(echo $ARCHS | wc -w) + 1) )) "${DR[@]}" --name "$NAME-$tag-models" --gpus "\"device=${G[0]}\"" "$IMG" bash -c '
        T=$1 bin=$2 seed=$3 q8=$4 qargs=$5 quant=$6; shift 6
        for a in "$@"; do
            d=$T/models/$a; rm -rf "$d"; mkdir -p "$d"
            # the anchored name: -a takes a regular expression, and qwen35 would also match qwen35moe
            (cd "$d" && LLAMA_TEST_ARCHS_N_VOCAB=4096 timeout 600 "$bin/test-llama-archs" -a "^$a\$" -s "$seed" -o . > gen.log 2>&1)
            m=$(cd "$d" && ls "$a"-moe.gguf "$a"-dense.gguf 2> /dev/null | head -n 1)
            [ -n "$m" ] || { echo "$a: no model"; continue; }
            echo "$m" > "$d/model.txt"
            echo "$(md5sum < "$d/$m" | cut -c1-32)  $a/$m"
            if [ "$q8" = 1 ]; then
                timeout 600 "$quant" $qargs "$d/$m" "$d/q8.gguf" Q8_0 4 > "$d/quant.log" 2>&1 &&
                    echo "$(md5sum < "$d/q8.gguf" | cut -c1-32)  $a/q8.gguf" || echo "$a: no Q8_0 copy (quant.log)"
            fi
        done' _ "$T" "$bin" "$MODEL_SEED" "$needq8" "$QUANT_ARGS" "$quant" $ARCHS > "$T/models.md5" 2>&1
    say "$tag models: $(grep -c '\.gguf$' "$T/models.md5") files; $(grep -a 'no ' "$T/models.md5" | tr '\n' ' ')"
    docker rm -f "$NAME-$tag-models" > /dev/null 2>&1

    # one-card layouts: up to one container per card at a time; then the split layouts, one at a time over every card
    i=0
    for layout in $LAYOUTS; do
        case $layout in one | virt) ;; *) continue ;; esac
        for arm in $ARMS; do
            job "$tag" "$bin" "$arm" "$layout" "${G[$((i % NG))]}" &
            i=$((i + 1))
            [ $((i % NG)) = 0 ] && wait
        done
    done
    wait
    for layout in $LAYOUTS; do
        [ "$layout" = split ] || continue
        for arm in $ARMS; do job "$tag" "$bin" "$arm" split "$GPUS"; done
    done
    for layout in $LAYOUTS; do
        for arm in $ARMS; do
            f=$T/$arm-$layout/verdicts.tsv
            [ -f "$f" ] || { printf '%s\t%s\t%s\tall\tcontainer\tFAIL\tno verdicts\t-\t-\n' "$tag" "$arm" "$layout" >> "$WORK/summary.tsv"; continue; }
            awk -v t="$tag" -v a="$arm" -v l="$layout" 'BEGIN { FS = OFS = "\t" } { print t, a, l, $0 }' "$f" >> "$WORK/summary.tsv"
        done
    done
done

say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" 2> /dev/null | paste -sd' ')"
column -t -s $'\t' "$WORK/summary.tsv" 2> /dev/null | cut -c1-220 || cat "$WORK/summary.tsv"
date -Is > "$WORK/DONE"
! cut -f6 "$WORK/summary.tsv" | grep -qE '^(FAIL|DIFFERS)$'
