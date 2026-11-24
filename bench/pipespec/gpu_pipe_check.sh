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
#           (cpu_tests.sh's models: the same files for the same seed); then a quantized copy for each other kind of
#           model an arm names (below), made by QUANT, a llama-quantize (QUANT_ARGS keep the hyper-connection and
#           per-layer-embedding norms F32, which the graph adds and multiplies elementwise). The fork's Pascal
#           matrix-vector kernels take quantized weights only (and BF16), so only the copies reach them; each type has
#           its own kernels (the one-token kernel per type, the k-column kernel and the read-once multi-token kernels of
#           IQ4_XS, Q4_K and Q5_K), so each type of the release files needs its own copy. Q6_K weights stay on
#           llama.cpp's kernels in these models under every setting (C1018's profile)
#   runs    for each layout of LAYOUTS and arm of ARMS that runs in it, one container with that layout's cards; per
#           architecture, each check N tokens of prompts/rewrite.txt as plain text (cpu_tests.sh's pipe step's arms):
#             host       check_identity.sh, greedy on the host: depth 0, the floor (depth FLOOR, every guess a filler
#                        token), and at depths 1 and 3 the oracle with 0 and 50 % of its guesses wrong and the four
#                        guessers: every pipelined run's tokens, logits, KV state and probe hashes plain's
#             sampled    the same with seeded sampling on the host (SAMPLING --seed SAMPLE_SEED), where a fault that only
#                        nudges the logits changes the tokens themselves (C932's did, C1010)
#             race       the same with a race of RACE samples (seeds SAMPLE_SEED-1, SAMPLE_SEED, ...) at depths 1 and 3
#                        with 50 % wrong guesses and the floor: the plain run decodes the samples round robin, the
#                        pipelined runs keep one token of each in flight. Its stop text is the token the sampled check's
#                        plain run (the race's sample 1) first emits at or after token RACE_AT, written so the random
#                        models' dummy tokenizer reads it as that one token, so the race ends part way and the winner
#                        (sample 1, in sequence 1) goes on alone with the guessers; the winner, the race's tokens and
#                        every sample's tokens and logits are compared too. Without the sampled check first the race has
#                        no stop and runs to the token limit
#             fed        the host check with greedy sampling on the device in every run, plain's too
#                        (--pipe-backend-greedy), the token embedding on the first card (LLAMA_INPUT_ON_GPU=1) and FEED=1,
#                        which adds depth 0
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
#             profile    (not in the default CHECKS; needs NSYS) the host check's runs again under Nsight Systems
#                        (nsys profile -t cuda --cuda-graph-trace=node): every kernel the cards ran and its launch count,
#                        in prof-<arch>.tsv, recorded, not judged; the report's times are deleted unread
#           Each identity check also records the plain run's tokens and logits hashes, so two arms on the same model show
#           whether different kernels computed them (other logits hashes), and every check its seconds (times.tsv)
# Layouts: one    the first card of GPUS
#          split  every card of GPUS, the layers in equal shares (-ts 1,1,...), so every token hops between cards: the
#                 embedding on the first card, the output and the device sampler on the last
#          virt   VIRT virtual devices on the first card (GGML_CUDA_DEVICES=VIRT, -ts 1,1,...): the hops on a machine
#                 with one card
#          (default "one split" when GPUS names two or more cards, "one virt" with one)
# Arms: <models>-<setting>[-c<N>][@<layout>,...]; models
#          f32      the models as test-llama-archs writes them
#          q8       the Q8_0 copy (llama-quantize ... Q8_0)
#          <type>   a copy with every weight in that llama-quantize type where its row width allows it (llama-quantize
#                   --pure ... <TYPE>), e.g. q2_0, q4_k, q5_k, q6_k, iq4_xs: the release files' types. A K-type or
#                   IQ4_XS needs rows of 256: the expert downs (384 wide) and shared-expert downs (192) fall back
#                   (Q4_K to Q5_0, Q5_K to Q5_1, Q6_K to Q8_0, IQ4_XS to IQ4_NL), as in the release files
#          mix      one copy holding the release files' types by tensor role (MIX_ARGS): routed experts Q2_0 (as in
#                   Swift Flash-Next), shared experts Q4_K (their downs Q5_0), attention Q5_K, linear attention and
#                   dense feed-forward IQ4_XS (dense downs IQ4_NL), the output Q6_K, the rest Q8_0
#        setting
#          stock    no switch: on a card that is not compute 6.1, llama.cpp's own kernels
#          p106     the fork's Pascal matrix-vector kernels and its sparse attention forced on with a P106's launch shapes
#                   (P106_ENV), as a P106 runs them (on compute 6.1 they are on without a switch)
#          fn       p106 plus every GGML_CUDA_MMV* switch of the Flash-Next v5 preset (FN_SW, from
#                   release/flash-next-v5/settings.env): the read-once kernels of 2-4 or 2-8 tokens for IQ4_XS, Q4_K,
#                   Q3_K, Q5_K and Q6_K, the shared expert's gate, up and SwiGLU as one launch, the k-column kernel's
#                   exact mode and its shapes, the F32 multi-token kernel; GGML_CUDA_MMVF_F32_MT=2 for the preset's 1,
#                   which acts on compute 6.1 only. When this folder sits in the repo, the driver says whether FN_SW
#                   still matches the preset
#          p106all  p106 plus the fork's other compute-6.1-only card paths (P106ALL_ENV: the attention-block fusion, the
#                   persistent MoE kernel, the Pascal flash-attention tile kernel, the F32 multi-token kernel)
#        -c<N> (optional, e.g. q4_k-fn-c4): every pipespec run (host, fed, profile) decodes its prompt in calls of N
#                   tokens (llama-pipespec --pipe-chunk N), plain's too. Without it the prompt goes in one call (the
#                   large-batch kernels) and every token after it alone (pipelined runs too: each guess is its own
#                   one-token call), so only the one-token kernels run, and the two-column ones of a MoE layer with two
#                   experts a token. Calls of 2 to 8 tokens are where the fn switches and the k-column kernel act
#        @<layout>,... (optional, e.g. mix-fn-c4@one): the arm runs only in those layouts of LAYOUTS; without it, in
#                   every one
#          (default "f32-stock q8-p106 mix-fn-c4@one": the CPU step's models on llama.cpp's kernels and the fork's
#          kernels on the Q8_0 copy, in every layout; the release types under the Flash-Next preset's switches with the
#          prompt in calls of 4 tokens, on one card only (the split layout's hops are the other two arms'; the kernels
#          are the same on one card). With the default checks and containers a build takes under three minutes on three
#          cards (165 s, C1019). The per-type arms, such as "q2_0-fn-c4 q4_k-fn-c4 q5_k-fn-c4 q6_k-fn-c4 iq4_xs-fn-c4",
#          on request)
# Containers: a build's containers run JOBS at a time (0, the default: every one at once), the split layouts first, then
# the one-card layouts with the cards in turn. Every tool runs on THREADS processor threads (-t, default 2): these small
# models leave the cards mostly idle and the processor does the waiting, and with each tool's own thread count (every
# core, spinning while it waits) the host-sampled checks of five containers side by side took thirteen times as long
# as alone (C1019)
# Verdicts (summary.tsv): OK when every hash is plain's, DIFFERS when a pipelined run gives other hashes, FAIL when a
# tool stopped before comparing (its exit code and the log's last line), SKIP when the build lacks a tool, INFO for the
# kernels line.
#
# Usage (on the machine with the cards; this script beside check_identity.sh and prompts/rewrite.txt, as in
# bench/pipespec):
#   WORK=<dir> BUILDS='<tag>=<bin dir> ...' [GPUS=0,1,2] [LAYOUTS='one split'] [ARMS='f32-stock q8-p106 mix-fn-c4@one']
#   [ARCHS='qwen4exp qwen35 qwen35moe qwen3next flint'] [CHECKS='host sampled race fed recurrent kernels']
#   [MODEL_SEED=1] [N=256] [SAMPLING='--temp 1 --top-k 20 --top-p 0.95'] [SAMPLE_SEED=42] [FLOOR=3] [RACE=2]
#   [RACE_AT=N/4] [JOBS=0] [THREADS=2] [TIMEOUT=900] [VIRT=3] [IMG=nvidia/cuda:12.9.2-devel-ubuntu24.04]
#   [MOUNTS='<folders mounted read-only>']
#   [DOCKER_ARGS='<labels, --cpu-shares>'] [PREFLIGHT=<cuda_preflight.sh>] [NAME=gpupipe] [QUANT_ARGS=...]
#   [MIX_ARGS=...] [P106_ENV=...] [FN_SW=...] [P106ALL_ENV=...] [JITCACHE=<dir>] [QUANT=<llama-quantize>]
#   [NSYS=<nsys, for CHECKS with profile>] bash gpu_pipe_check.sh
# A bin dir is a rig tarball's bin/: scripts/build_llamacpp_docker.sh with TESTS=1 packs llama-pipe-recurrent-check and
# test-llama-archs (since 2026-11-24, C984) and llama-quantize (C1017) beside llama-pipespec and llama-completion, so
# each build makes its copies with its own llama-quantize. QUANT names one llama-quantize for every build instead (one
# that runs in IMG, e.g. a static CPU build: cpu_tests.sh with SHARED=OFF and KEEP_BUILD=1), which gives every build the
# same copies (models.md5); a tarball built before C1017 has none, so it needs QUANT. Every tool runs in a container
# (docker run --gpus, no network) of IMG with the build, this folder, MOUNTS and WORK at the same paths.
# MODEL_SEED, the random models' seed, must be a whole number (test-llama-archs -s); the script stops at once, before it
# writes anything, on any other value. The older name SEED is still read, but only when it holds a whole number and
# MODEL_SEED is unset: a caller's own SEED of another kind (a clone's path, say) is left alone and the seed stays 1.
# The PTX JIT's cache is JITCACHE (default WORK/jitcache), shared by every run: a compute 6.1 build on a newer card
# compiles its PTX on first use, so several runs of one build can share a JITCACHE.
# NSYS (the profile check) is an nsys binary on the host, e.g. /opt/nvidia/nsight-systems/<version>/bin/nsys: its
# install folder is mounted read-only into the containers (the CUDA images carry Nsight Compute but not Nsight Systems).
# Output: WORK/<tag>/models/<arch>/ (gen.log, the model, q8.gguf and quant.log for q8, <kind>.gguf and
#   quant-<kind>.log for the other copies), WORK/<tag>/models.md5, WORK/<tag>/types.tsv (each copy's tensors by type),
#   WORK/<tag>/<arm>-<layout>/ (check_identity's folders <arch>-<check>/ with .log their tables, prc-<arch>.log,
#   kernels.log, prof-<arch>.tsv, env.txt, verdicts.tsv, times.tsv), WORK/summary.tsv (tree, arm, layout, arch, check,
#   verdict, detail, plain tokens hash, plain logits hash), WORK/times.tsv (tree, arm, layout, arch, check, seconds),
#   WORK/jobs.tsv (tree, job, cards, rc, seconds: each container and each build's total); done when WORK/DONE exists.
#   Exit status 0 when no verdict is FAIL or DIFFERS.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SELF=$HERE/$(basename "$0")
ARCHS=${ARCHS-qwen4exp qwen35 qwen35moe qwen3next flint}
CHECKS=${CHECKS:-host sampled race fed recurrent kernels}
N=${N:-256}
SAMPLING=${SAMPLING:---temp 1 --top-k 20 --top-p 0.95}
SAMPLE_SEED=${SAMPLE_SEED:-42}
FLOOR=${FLOOR-3}
RACE=${RACE:-2}
[[ "$N" =~ ^[0-9]{1,9}$ ]] && RACE_AT=${RACE_AT:-$((N / 4))}
RACE_AT=${RACE_AT:-0}
TIMEOUT=${TIMEOUT:-900}
VIRT=${VIRT:-3}
# the random models' vocabulary (test-llama-archs); the race's stop text is hashed to a token id modulo this
VOCAB=4096

# ---- inside a container: one arm and layout, every architecture ----
if [ "${GPU_PIPE_INNER:-}" = 1 ]; then
    : "${BIN:?}" "${MD:?}" "${KIND:?}" "${OUTD:?}" "${LARGS:?}"
    # check_identity.sh reads SAMPLING, RACE and FLOOR from its environment: this script's settings of those names stay
    # in this shell, and each check passes its own
    export -n SAMPLING SAMPLE_SEED FLOOR RACE RACE_AT
    mkdir -p "$OUTD"
    V=$OUTD/verdicts.tsv
    : > "$V"
    v() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "${5:--}" "${6:--}" >> "$V"; }
    pf() { grep -ao 'name=plain .*' "$1/results.txt" 2> /dev/null | grep -o " $2=[^ ]*" | cut -d= -f2; }
    last() { grep -av '^ *$' "$1" 2> /dev/null | tail -n 1 | cut -c1-140; }
    on() { [[ " $CHECKS " == *" $1 "* ]]; }
    # <arch> <check> <start, date +%s.%N>: a line of times.tsv
    tm() { printf '%s\t%s\t%s\n' "$1" "$2" "$(awk -v a="$3" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')" >> "$OUTD/times.tsv"; }
    # <plain.tok of the sampled check, "<position> <token> <hash>" lines>: "<position> <token>" of the first token at or
    # after RACE_AT that the sample had not emitted before (else the last such token before it, from position 1);
    # cpu_tests.sh's pipe_race_tok
    race_tok() {
        awk -v at="$RACE_AT" '!($2 in seen) && $1 >= 1 { if ($1 >= at) { if (t == "") t = $1 " " $2 } else l = $1 " " $2 }
            { seen[$2] = 1 } END { print (t != "" ? t : l) }' "$1"
    }
    # <token id>: a text of two or three letters and digits that the random models' dummy tokenizer turns into that one
    # token (it cuts text into chunks of 5 bytes and hashes each, h = 31 h + byte, modulo the vocabulary); cpu_tests.sh's
    # pipe_stop_text
    stop_text() {
        awk -v t="$1" -v V="$VOCAB" 'BEGIN {
            a = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"; n = length(a)
            for (c = 32; c < 127; c++) ord[sprintf("%c", c)] = c
            for (i = 1; i <= n; i++) { ch[i] = substr(a, i, 1); o[i] = ord[ch[i]] }
            for (i = 1; i <= n; i++) for (j = 1; j <= n; j++) if ((o[i] * 31 + o[j]) % V == t) { print ch[i] ch[j]; exit }
            for (i = 1; i <= n; i++) for (j = 1; j <= n; j++) for (k = 1; k <= n; k++)
                if (((o[i] * 31 + o[j]) * 31 + o[k]) % V == t) { print ch[i] ch[j] ch[k]; exit }
        }'
    }
    # <arch> <check> <model> [VAR=value ...]: check_identity.sh, judged as cpu_tests.sh's pipe step judges it; depths 1 3,
    # misses 0 0.5 and the floor unless a VAR says otherwise
    ident() {
        local a=$1 c=$2 m=$3
        shift 3
        local o=$OUTD/$a-$c rc runs nr nd nt why vd refused="" t0 kv dp="1 3" ms="0 0.5" fl=$FLOOR race=""
        for kv in "$@"; do
            case $kv in DEPTHS=*) dp=${kv#*=} ;; MISSES=*) ms=${kv#*=} ;; FLOOR=*) fl=${kv#*=} ;; esac
        done
        rm -rf "$o"
        t0=$(date +%s.%N)
        env ARGS="$LARGS${PARGS:+ $PARGS}" DEPTHS="$dp" MISSES="$ms" FLOOR="$fl" "$@" BIN="$BIN" MODEL="$m" CHAT=0 SPEED=0 \
            TIMEOUT="$TIMEOUT" PROMPT="$HERE/prompts/rewrite.txt" OUT="$o" N="$N" bash "$HERE/check_identity.sh" > "$o.log" 2>&1
        rc=$?
        tm "$a" "$c" "$t0"
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
        # a race: who won and after how many of the samples' tokens (the plain run's; every pipelined run is compared)
        [[ " $* " == *" RACE="* ]] && race="; plain: $(grep -a 'name=plain ' "$o/results.txt" 2> /dev/null |
            grep -o 'winner=[-0-9]* race_tokens=[0-9]*' || echo 'no result')"
        v "$a" "$c" "$vd" "check_identity.sh rc=$rc: $why; N=$N, depths $dp, misses $ms${fl:+, floor $fl}$race${*:+; $*}" \
            "$(pf "$o" tokens)" "$(pf "$o" logits)"
    }
    have() { [ -x "$BIN/$1" ]; }
    # <arch>: the model file of this arm's kind
    model() { if [ "$KIND" = f32 ]; then echo "$MD/$1/$(cat "$MD/$1/model.txt" 2> /dev/null)"; else echo "$MD/$1/$KIND.gguf"; fi; }
    # <arch> <model>: the host check's runs under nsys; each kernel's launch count and name, without its parameters
    prof() {
        local a=$1 m=$2 o=$OUTD/$1-prof rc nk mm t0
        rm -rf "$o" "$o.log"
        mkdir -p "$o"
        t0=$(date +%s.%N)
        timeout "$TIMEOUT" "$NSYS" profile -t cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
            --force-overwrite=true -o "$o/rep" env ARGS="$LARGS${PARGS:+ $PARGS}" BIN="$BIN" MODEL="$m" CHAT=0 SPEED=0 TIMEOUT="$TIMEOUT" \
            PROMPT="$HERE/prompts/rewrite.txt" OUT="$o/ident" N="$N" DEPTHS="1 3" MISSES="0 0.5" FLOOR="$FLOOR" \
            bash "$HERE/check_identity.sh" > "$o.log" 2>&1
        rc=$?
        tm "$a" profile "$t0"
        "$NSYS" stats -q -r cuda_gpu_kern_sum -f csv -o "$o/kern" "$o/rep.nsys-rep" >> "$o.log" 2>&1
        # cuda_gpu_kern_sum: column 3 the launches, the last column the demangled name (quoted when it holds a comma)
        awk -F'"' 'function last(s, t,   i, j) { i = 0; while ((j = index(substr(s, i + 1), t)) > 0) i += j; return i }
            NR > 1 && NF > 0 {
                if (NF >= 3) { split($1, f, ","); n = $2 } else { split($0, f, ","); n = f[9] }
                k = last(n, ">(")
                if (k > 0) n = substr(n, 1, k); else if ((k = index(n, "(")) > 0) n = substr(n, 1, k - 1)
                sub(/^void /, "", n); gsub(/<unnamed>::/, "", n)
                print f[3] "\t" n
            }' "$o/kern_cuda_gpu_kern_sum.csv" 2> /dev/null | sort -t$'\t' -k2 > "$OUTD/prof-$a.tsv"
        # LLMRig takes no timings: the report and its times go unread
        rm -f "$o"/rep.* "$o"/kern_*.csv
        nk=$(wc -l < "$OUTD/prof-$a.tsv")
        mm=$(cut -f2 "$OUTD/prof-$a.tsv" | grep -E 'mul_mat|mmv|kcol|gemm|gemv|cutlass|_mt_|dmt|glu' |
            sed -E 's/<([^,>]*)[,>].*/<\1>/' | sort -u | paste -sd' ')
        v "$a" profile INFO "nsys rc=$rc: $nk kernels in the host check's runs; matrix products: ${mm:-none}"
    }
    {
        echo "kind=$KIND args=$LARGS pipespec args=${PARGS:-}"
        echo "checks=$CHECKS N=$N sampling=$SAMPLING sample seed=$SAMPLE_SEED floor=${FLOOR:-none} race=$RACE race at=$RACE_AT"
        env | grep -E '^(GGML_|LLAMA_|CUDA_VISIBLE|NVIDIA_VISIBLE)' | sort
    } > "$OUTD/env.txt"
    : > "$OUTD/times.tsv"
    for a in $ARCHS; do
        m=$(model "$a")
        if [ ! -f "$m" ]; then
            v "$a" model FAIL "no $KIND model ($m): see $MD/$a/gen.log and the quant logs"
            continue
        fi
        if on host; then
            if have llama-pipespec; then ident "$a" host "$m"; else v "$a" host SKIP "no llama-pipespec in $BIN"; fi
        fi
        if on sampled; then
            if have llama-pipespec; then
                ident "$a" sampled "$m" SAMPLING="$SAMPLING --seed $SAMPLE_SEED"
            else
                v "$a" sampled SKIP "no llama-pipespec in $BIN"
            fi
        fi
        if on race; then
            if have llama-pipespec; then
                # the stop: a token the sampled check's plain run (seed SAMPLE_SEED, the race's sample 1) emits part way
                pt=""
                [ -f "$OUTD/$a-sampled/plain.tok" ] && pt=$(race_tok "$OUTD/$a-sampled/plain.tok")
                if [ -n "$pt" ]; then
                    ident "$a" race "$m" MISSES=0.5 RACE="$RACE" RACE_STOP="$(stop_text "${pt#* }")" \
                        SAMPLING="$SAMPLING --seed $((SAMPLE_SEED - 1))" STOP_TOKEN="${pt#* }@${pt% *}"
                else
                    # no sampled check before it: the default stop text never comes, so the race runs to the token limit
                    ident "$a" race "$m" MISSES=0.5 RACE="$RACE" SAMPLING="$SAMPLING --seed $((SAMPLE_SEED - 1))" STOP_TOKEN=none
                fi
            else
                v "$a" race SKIP "no llama-pipespec in $BIN"
            fi
        fi
        if on fed; then
            if have llama-pipespec; then
                ident "$a" fed "$m" FEED=1 LLAMA_INPUT_ON_GPU=1 ARGS="$LARGS${PARGS:+ $PARGS} --pipe-backend-greedy"
            else
                v "$a" fed SKIP "no llama-pipespec in $BIN"
            fi
        fi
        if on recurrent; then
            if have llama-pipe-recurrent-check; then
                t0=$(date +%s.%N)
                # shellcheck disable=SC2086
                timeout "$TIMEOUT" "$BIN/llama-pipe-recurrent-check" -m "$m" -f "$HERE/prompts/rewrite.txt" -n "$N" $LARGS \
                    > "$OUTD/prc-$a.log" 2>&1
                rc=$?
                tm "$a" recurrent "$t0"
                line=$(grep -a -o 'PIPE-RECURRENT-CHECK.*' "$OUTD/prc-$a.log" | tail -n 1)
                v "$a" recurrent "$(case $rc in 0) echo OK ;; 2) echo DIFFERS ;; *) echo FAIL ;; esac)" \
                    "llama-pipe-recurrent-check rc=$rc: ${line:-$(last "$OUTD/prc-$a.log")}"
            else
                v "$a" recurrent SKIP "no llama-pipe-recurrent-check in $BIN"
            fi
        fi
        if on profile; then
            if [ -z "${NSYS:-}" ] || [ ! -x "$NSYS" ]; then v "$a" profile SKIP "no NSYS (${NSYS:-unset}) in the container"
            elif ! have llama-pipespec; then v "$a" profile SKIP "no llama-pipespec in $BIN"
            else prof "$a" "$m"; fi
        fi
    done
    if on kernels; then
        m=""
        for a in $ARCHS; do
            f=$(model "$a")
            [ -f "$f" ] && { m=$f; break; }
        done
        if have llama-completion && [ -n "$m" ]; then
            t0=$(date +%s.%N)
            # shellcheck disable=SC2086
            timeout 300 "$BIN/llama-completion" -m "$m" -p 'The quick brown fox' -n 1 $LARGS -no-cnv --temp 0 -lv 4 \
                > "$OUTD/kernels.log" 2>&1 < /dev/null
            rc=$?
            tm all kernels "$t0"
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
# the other numbers (the race's seeds are SAMPLE_SEED-1 and up, so SAMPLE_SEED is at least 1)
for kv in "N=$N" "SAMPLE_SEED=$SAMPLE_SEED" "RACE=$RACE" "RACE_AT=$RACE_AT" ${FLOOR:+"FLOOR=$FLOOR"}; do
    if ! [[ "${kv#*=}" =~ ^[0-9]{1,9}$ ]] || { [ "${kv%%=*}" != RACE_AT ] && [ "${kv#*=}" -lt 1 ]; }; then
        echo "gpu_pipe_check.sh: ${kv%%=*}='${kv#*=}' is not a whole number of at least 1 (RACE_AT may be 0); nothing was run" >&2
        exit 2
    fi
done
JOBS=${JOBS:-0}
[[ "$JOBS" =~ ^[0-9]{1,4}$ ]] || { echo "gpu_pipe_check.sh: JOBS='$JOBS' is not a whole number (0: every container at once); nothing was run" >&2; exit 2; }
THREADS=${THREADS:-2}
[[ "$THREADS" =~ ^[1-9][0-9]{0,2}$ ]] || { echo "gpu_pipe_check.sh: THREADS='$THREADS' is not a whole number of at least 1; nothing was run" >&2; exit 2; }
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
GPUS=${GPUS:-0,1,2}
IFS=, read -ra G <<< "$GPUS"
NG=${#G[@]}
if [ "$NG" -ge 2 ]; then LAYOUTS=${LAYOUTS:-one split}; else LAYOUTS=${LAYOUTS:-one virt}; fi
ARMS=${ARMS:-f32-stock q8-p106 mix-fn-c4@one}
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
NAME=${NAME:-gpupipe}
PREFLIGHT=${PREFLIGHT:-}
JITCACHE=${JITCACHE:-$WORK/jitcache}
QUANT=${QUANT:-}
NSYS=${NSYS:-}
mkdir -p "$JITCACHE"
QUANT_ARGS=${QUANT_ARGS:---tensor-type hc_=f32 --tensor-type ple_norm=f32}
# the mix copy (Q8_0 for what no pattern names); the first matching pattern wins, QUANT_ARGS' patterns first
MIX_ARGS=${MIX_ARGS:---pure --tensor-type _exps=q2_0 --tensor-type _shexp=q4_k --tensor-type attn_=q5_k --tensor-type ssm_=iq4_xs --tensor-type ffn_=iq4_xs --output-tensor-type q6_k}
P106_ENV=${P106_ENV:-GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_FATTN_SPARSE_PASCAL=2}
# every GGML_CUDA_MMV* switch of release/flash-next-v5/settings.env, GGML_CUDA_MMVF_F32_MT at 2 for the preset's 1 (C968)
FN_SW=${FN_SW:-GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 GGML_CUDA_MMVQ_PASCAL_Q3K_MT=1 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1 GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1 GGML_CUDA_MMVF_F32_MT=2 GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT=1 GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_0,q5_0}
P106ALL_ENV=${P106ALL_ENV:-$P106_ENV GGML_CUDA_ATTN_BLOCK=2 GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_MMVF_F32_MT=2}
BASE_ARGS="-ngl 99 -fa on -c 2048 -fit off -t $THREADS"
# the quantized copies the arms name (every kind but f32)
KINDS=$(for a in $ARMS; do echo "${a%%-*}"; done | grep -vx f32 | sort -u | paste -sd' ')
ones() { local s=1 i; for ((i = 1; i < $1; i++)); do s="$s,1"; done; echo "$s"; }
# <arm spec> <layout>: whether the arm runs in that layout (an arm without @ runs in every one)
inlayout() { [[ "$1" != *@* ]] || [[ ",${1#*@}," == *",$2,"* ]]; }

say() { echo "$(date +%H:%M:%S) $*"; }
VOL=(-v "$HERE:$HERE:ro" -v "$WORK:$WORK")
[[ "$JITCACHE" == "$WORK"/* ]] || VOL+=(-v "$JITCACHE:$JITCACHE")
[ -n "$QUANT" ] && VOL+=(-v "$(dirname "$QUANT"):$(dirname "$QUANT"):ro")
if [ -n "$NSYS" ]; then
    NSYS=$(readlink -f "$NSYS")
    NSYSDIR=$(dirname "$(dirname "$NSYS")")  # <install>/bin/nsys or <install>/target-linux-x64/nsys
    VOL+=(-v "$NSYSDIR:$NSYSDIR:ro")
fi
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
# shellcheck disable=SC2206
DR=(docker run --rm --network none ${DOCKER_ARGS:-} "${VOL[@]}" -e "CUDA_CACHE_PATH=$JITCACHE" -e CUDA_CACHE_MAXSIZE=4294967296)
rm -f "$WORK/DONE"
[ -f "$WORK/summary.tsv" ] ||
    printf 'tree\tarm\tlayout\tarch\tcheck\tverdict\tdetail\tplain tokens hash\tplain logits hash\n' > "$WORK/summary.tsv"
[ -f "$WORK/times.tsv" ] || printf 'tree\tarm\tlayout\tarch\tcheck\tseconds\n' > "$WORK/times.tsv"
[ -f "$WORK/jobs.tsv" ] || printf 'tree\tjob\tcards\trc\tseconds\n' > "$WORK/jobs.tsv"
# <tree> <job> <cards> <rc> <start, date +%s>: a line of jobs.tsv
jobline() { printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$(($(date +%s) - $5))" >> "$WORK/jobs.tsv"; }
md5sum "$SELF" "$HERE/check_identity.sh" "$HERE/prompts/rewrite.txt" > "$WORK/scripts.md5"
say "gpu pipe check: builds '$BUILDS', cards $GPUS, layouts '$LAYOUTS', arms '$ARMS', archs '$ARCHS', checks '$CHECKS', jobs $JOBS, threads $THREADS, model seed $MODEL_SEED ($SEEDNOTE), N=$N, sampling '$SAMPLING' seed $SAMPLE_SEED, floor ${FLOOR:-none}, race of $RACE from token $RACE_AT"
echo "$MODEL_SEED" > "$WORK/model_seed.txt"
# whether fn still matches the Flash-Next preset, when this folder sits in the repo
fnused=0
for a in $ARMS; do s=${a#*-}; [ "${s%%-*}" = fn ] && fnused=1; done
if [ "$fnused" = 1 ]; then
    PRESET=$HERE/../../release/flash-next-v5/settings.env
    if [ -f "$PRESET" ]; then
        # shellcheck disable=SC1090
        psw=$(unset PRESET_ENV; . "$PRESET" > /dev/null 2>&1
            for kv in $PRESET_ENV; do case $kv in GGML_CUDA_MMV*) echo "${kv/#GGML_CUDA_MMVF_F32_MT=1/GGML_CUDA_MMVF_F32_MT=2}" ;; esac; done | sort)
        if [ "$psw" = "$(printf '%s\n' $FN_SW | sort)" ]; then say "fn: FN_SW matches $(cd "$HERE/../.." && pwd)/release/flash-next-v5/settings.env"
        else say "fn: FN_SW differs from release/flash-next-v5/settings.env, whose GGML_CUDA_MMV* switches are: $(echo $psw)"; fi
    fi
    printf 'P106_ENV=%s\nFN_SW=%s\n' "$P106_ENV" "$FN_SW" > "$WORK/fn.txt"
fi
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
    local kind=${arm%%-*} setting=${arm#*-} opt="" pargs="" senv="" largs=$BASE_ARGS c=$NAME-$1-$3-$4
    local o=$WORK/$tag/$arm-$layout
    # <models>-<setting>[-c<N>]: -c<N> decodes every pipespec run's prompt in calls of N tokens
    [[ "$setting" == *-* ]] && opt=${setting#*-} setting=${setting%%-*}
    case $opt in
    "") ;;
    c[1-9]*) pargs="--pipe-chunk ${opt#c}" ;;
    *) mkdir -p "$o"; printf 'all\tarm\tFAIL\tunknown option %s\t-\t-\n' "$opt" > "$o/verdicts.tsv"; return ;;
    esac
    case $setting in
    stock) senv="" ;;
    p106) senv=$P106_ENV ;;
    fn) senv="$P106_ENV $FN_SW" ;;
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
    local t0
    t0=$(date +%s)
    # shellcheck disable=SC2086
    timeout $(( (6 * $(echo $ARCHS | wc -w) + 2) * TIMEOUT )) "${DR[@]}" --name "$c" --gpus "\"device=$cards\"" "$IMG" \
        env $senv GPU_PIPE_INNER=1 BIN="$bin" MD="$WORK/$tag/models" KIND="$kind" OUTD="$o" LARGS="$largs" PARGS="$pargs" \
        ARCHS="$ARCHS" CHECKS="$CHECKS" N="$N" SAMPLING="$SAMPLING" SAMPLE_SEED="$SAMPLE_SEED" FLOOR="$FLOOR" RACE="$RACE" \
        RACE_AT="$RACE_AT" TIMEOUT="$TIMEOUT" NSYS="$NSYS" bash "$SELF" > "$o/container.log" 2>&1
    local rc=$?
    docker rm -f "$c" > /dev/null 2>&1
    jobline "$tag" "$arm-$layout" "$cards" "$rc" "$t0"
    [ $rc = 0 ] || printf 'all\tcontainer\tFAIL\trc=%s (124: timed out): %s\t-\t-\n' "$rc" "$(tail -n 1 "$o/container.log")" >> "$o/verdicts.tsv"
    say "$tag $arm $layout (cards $cards): rc=$rc in $(($(date +%s) - t0)) s; $(cut -f3 "$o/verdicts.tsv" 2> /dev/null | sort | uniq -c | awk '{printf "%s %s; ", $1, $2}')"
}

for spec in $BUILDS; do
    tag=${spec%%=*} bin=${spec#*=}
    T=$WORK/$tag
    mkdir -p "$T/models"
    say "$tag: $bin"
    tb=$(date +%s)
    (cd "$bin" && md5sum llama-pipespec llama-pipe-recurrent-check test-llama-archs llama-completion 2>&1) > "$T/bin.md5"
    quant=${QUANT:-$bin/llama-quantize}
    [ -n "$KINDS" ] && md5sum "$quant" >> "$T/bin.md5" 2>&1
    [ -n "$KINDS" ] && [ ! -x "$quant" ] && say "$tag: no llama-quantize at $quant, so no $KINDS copies: pass QUANT"
    if [ ! -x "$bin/test-llama-archs" ]; then
        printf '%s\t-\t-\t-\tmodels\tSKIP\tno test-llama-archs in %s\t-\t-\n' "$tag" "$bin" >> "$WORK/summary.tsv"
        continue
    fi
    # the models: each architecture's file from this build's test-llama-archs, and its copies: q8 as before (no
    # --pure), mix by MIX_ARGS, any other kind every weight in that type (--pure)
    # shellcheck disable=SC2086
    timeout $(( 600 * ($(echo $ARCHS | wc -w) * ($(echo $KINDS | wc -w) + 1) + 1) )) "${DR[@]}" --name "$NAME-$tag-models" --gpus "\"device=${G[0]}\"" "$IMG" bash -c '
        T=$1 bin=$2 seed=$3 kinds=$4 qargs=$5 quant=$6 mixargs=$7; shift 7
        for a in "$@"; do
            d=$T/models/$a; rm -rf "$d"; mkdir -p "$d"
            # the anchored name: -a takes a regular expression, and qwen35 would also match qwen35moe
            (cd "$d" && LLAMA_TEST_ARCHS_N_VOCAB=4096 timeout 600 "$bin/test-llama-archs" -a "^$a\$" -s "$seed" -o . > gen.log 2>&1)
            m=$(cd "$d" && ls "$a"-moe.gguf "$a"-dense.gguf 2> /dev/null | head -n 1)
            [ -n "$m" ] || { echo "$a: no model"; continue; }
            echo "$m" > "$d/model.txt"
            echo "$(md5sum < "$d/$m" | cut -c1-32)  $a/$m"
            for k in $kinds; do
                case $k in
                q8) log=quant.log extra="" ty=Q8_0 ;;
                mix) log=quant-mix.log extra=$mixargs ty=Q8_0 ;;
                *) log=quant-$k.log extra=--pure ty=${k^^} ;;
                esac
                timeout 600 "$quant" $qargs $extra "$d/$m" "$d/$k.gguf" "$ty" 4 > "$d/$log" 2>&1 &&
                    echo "$(md5sum < "$d/$k.gguf" | cut -c1-32)  $a/$k.gguf" || echo "$a: no $k copy ($log)"
            done
        done' _ "$T" "$bin" "$MODEL_SEED" "$KINDS" "$QUANT_ARGS" "$quant" "$MIX_ARGS" $ARCHS > "$T/models.md5" 2>&1
    jobline "$tag" models "${G[0]}" "$?" "$tb"
    say "$tag models: $(grep -c '\.gguf$' "$T/models.md5") files; $(grep -a 'no ' "$T/models.md5" | tr '\n' ' ')"
    docker rm -f "$NAME-$tag-models" > /dev/null 2>&1
    # each copy's tensors by type, from llama-quantize's log (its last word on each tensor: the type it wrote)
    for a in $ARCHS; do
        for k in $KINDS; do
            [ "$k" = q8 ] && log=quant.log || log=quant-$k.log
            f=$T/models/$a/$log
            [ -f "$f" ] || continue
            printf '%s\t%s\t%s\n' "$a" "$k" "$(grep -aoE 'converting to [A-Za-z0-9_]+|type = +[a-z0-9_]+, size' "$f" |
                sed -E 's/converting to //; s/type = +//; s/, size//' | sort | uniq -c | awk '{printf "%s %s, ", $2, $1}' | sed 's/, $//')"
        done
    done > "$T/types.tsv"

    # the containers: the split layouts first (over every card), then the one-card layouts (the cards in turn), JOBS of
    # them at a time
    q=()
    for layout in $LAYOUTS; do
        [ "$layout" = split ] || continue
        for spec in $ARMS; do inlayout "$spec" split && q+=("${spec%%@*} split $GPUS"); done
    done
    i=0
    for layout in $LAYOUTS; do
        case $layout in one | virt) ;; *) continue ;; esac
        for spec in $ARMS; do
            inlayout "$spec" "$layout" || continue
            q+=("${spec%%@*} $layout ${G[$((i % NG))]}")
            i=$((i + 1))
        done
    done
    n=0
    for j in "${q[@]}"; do
        read -r arm layout cards <<< "$j"
        job "$tag" "$bin" "$arm" "$layout" "$cards" &
        n=$((n + 1))
        [ "$JOBS" -gt 0 ] && [ "$n" -ge "$JOBS" ] && { wait -n; n=$((n - 1)); }
    done
    wait
    for layout in $LAYOUTS; do
        for spec in $ARMS; do
            inlayout "$spec" "$layout" || continue
            arm=${spec%%@*}
            f=$T/$arm-$layout/verdicts.tsv
            [ -f "$f" ] || { printf '%s\t%s\t%s\tall\tcontainer\tFAIL\tno verdicts\t-\t-\n' "$tag" "$arm" "$layout" >> "$WORK/summary.tsv"; continue; }
            awk -v t="$tag" -v a="$arm" -v l="$layout" 'BEGIN { FS = OFS = "\t" } { print t, a, l, $0 }' "$f" >> "$WORK/summary.tsv"
            awk -v t="$tag" -v a="$arm" -v l="$layout" 'BEGIN { FS = OFS = "\t" } { print t, a, l, $0 }' \
                "$T/$arm-$layout/times.tsv" >> "$WORK/times.tsv" 2> /dev/null
        done
    done
    jobline "$tag" total "$GPUS" - "$tb"
    say "$tag: $(($(date +%s) - tb)) s for the build"
done

say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" 2> /dev/null | paste -sd' ')"
column -t -s $'\t' "$WORK/summary.tsv" 2> /dev/null | cut -c1-220 || cat "$WORK/summary.tsv"
date -Is > "$WORK/DONE"
! cut -f6 "$WORK/summary.tsv" | grep -qE '^(FAIL|DIFFERS)$'
