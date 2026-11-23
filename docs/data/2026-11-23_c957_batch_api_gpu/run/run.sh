#!/usr/bin/env bash
# C957 (Q1687 point 11): the GPU half of C880's check (fork branch pascal-p106-v5-batch-api aebff903e, one commit on
# pascal-p106-v5 3dfad876f: the fork's own tools and tests moved from llama_batch_init / llama_decode to upstream's
# llama_batch_ext / llama_process), on a build machine with three newer NVIDIA cards that run the rig build's compute
# 6.1 PTX. Correctness only, nothing timed.
#
# Both builds are rig builds (scripts/build_llamacpp_docker.sh, TESTS=1) with the same extra targets: the branch's, and
# the base's made the same way, since the base's own rig tarball holds none of the six extra tools. Each run is
# gpu_tools.sh (C880's tools.sh with the layers on the cards) with one build, one kernel setting and one set of models,
# in its own container with the three cards:
#   models  f32: C880's random models (test-llama-archs -s 1: qwen4exp MoE with a 4,096-token vocabulary, the same with
#                32 experts, 8 used, and the first with a screened head), every weight F32, as C880 ran them;
#           q8:  the first two quantized to Q8_0 (llama-quantize), so the matrix products can run the fork's Pascal
#                matrix-vector kernels, which take quantized weights only (and BF16 for the F16 twin); the
#                hyper-connection and per-layer-embedding norm tensors (hc_*, ple_norm*) stay F32 (QUANT_ARGS): the
#                graph adds or multiplies them elementwise, which no backend does with a Q8_0 operand
#   setting stock: every switch at its default (a newer card runs llama.cpp's own matrix-vector kernels);
#           p106:  the fork's Pascal kernels forced on with a P106's launch shapes (5 SMs, 96 KB of shared memory per SM),
#                  as C953 ran them: they turn themselves on only on compute 6.1 cards
# Runs (RUNS, "<models>-<setting>-<build>", build base, head, or base2 = the base again to show the runs repeat):
#   f32-stock-base f32-stock-head f32-stock-base2   every part (gpu_tools.sh PARTS: ps prc screen moe hop pool)
#   f32-p106-base f32-p106-head                     every part but pool (those three tests run on the CPU backend)
#   q8-p106-base q8-p106-head                       ps prc moe hop
#   q8-stock-base                                   ps prc moe hop: against q8-p106-base it shows whether the forced
#                                                   kernels computed anything (other logits hashes if they did)
# probe: one token of llama-completion per setting on the Q8_0 model, for ggml's CUDA feature list (the tools print
# none), which names the fork's Pascal kernels that are on.
# Then compare.py (C880's) on pairs: base against head per models and setting (the check), base against base2 (the
# runs repeat), stock against p106 (the setting's reach); the rig builds' warning lines, head against base.
ON="GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 \
GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_FATTN_SPARSE_PASCAL=2"
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env:
#   WORK       a folder for the outputs (mounted read-write into every container); the rig builds' logs, if kept as
#              WORK/logs/rig-head.log and rig-base.log, are compared for warnings
#   NEW, REF   bin/ folders of the two rig builds (NEW = the branch, REF = the base)
#   REPO       a copy of this repository (this run/, bench/pipespec, docs/data/2026-11-21_c880_batch_api/run/compare.py)
#   MD         C880's random models: qwen4exp-moe.gguf, qwen4exp-moe-screen.gguf, e32/qwen4exp-moe.gguf, prompts.txt;
#              the Q8_0 copies go to MD/q8/
#   QUANT      a llama-quantize (with its libraries beside it) and CPU_IMAGE (pq-c565-cpu:24.04) to run it in;
#              QUANT_ARGS its options ("--tensor-type hc_=f32 --tensor-type ple_norm=f32")
#   MOUNTS     more folders the containers mount read-only at the same paths
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT  a CUDA check run before the runs as "$PREFLIGHT -g <ids>" (optional)
#   GPUS (0,1,2), TS (1,1,1), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04), STEPS ("quant runs probe compare"), RUNS (above),
#   RUN_TIMEOUT (3600 s), THREADS (6)
# Output: WORK/tools/<run>/ (gpu_tools.sh's logs, rc.tsv, the pipespec results), WORK/logs/<run>.log,
# WORK/probe/<setting>.log, WORK/compare/ (compare.py's tables, summary.txt, features.tsv, warnings), WORK/md5.txt,
# WORK/DONE.
set -uo pipefail
: "${WORK:?}" "${NEW:?}" "${REF:?}" "${REPO:?}" "${MD:?}"
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
GPUS=${GPUS:-0,1,2}
TS=${TS:-1,1,1}
THREADS=${THREADS:-6}
STEPS=${STEPS:-quant runs probe compare}
QUANT_ARGS=${QUANT_ARGS:---tensor-type hc_=f32 --tensor-type ple_norm=f32}
RUNS=${RUNS:-f32-stock-base f32-stock-head f32-stock-base2 f32-p106-base f32-p106-head q8-p106-base q8-p106-head q8-stock-base}
RUN_TIMEOUT=${RUN_TIMEOUT:-3600}
PREFLIGHT=${PREFLIGHT:-}
D=$REPO/docs/data/2026-11-23_c957_batch_api_gpu/run
CMP=$REPO/docs/data/2026-11-21_c880_batch_api/run/compare.py
PS=$REPO/bench/pipespec
Q8=$MD/q8
say() { echo "$(date +%H:%M:%S) $*"; }
step() { [[ " $STEPS " == *" $1 "* ]]; }
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$WORK:$WORK" -v "$REPO:$REPO:ro" -v "$NEW:$NEW:ro" -v "$REF:$REF:ro")
[[ "$MD" == "$WORK"/* ]] || VOL+=(-v "$MD:$MD")
# the PTX JIT's cache, shared by every container (the cards compile the build's compute 6.1 PTX once)
JIT=(-e "CUDA_CACHE_PATH=$WORK/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)
# shellcheck disable=SC2206
DR=(docker run --rm ${DOCKER_ARGS:-} --network none "${VOL[@]}")
mkdir -p "$WORK/jitcache" "$WORK/tools" "$WORK/logs" "$WORK/compare"
rm -f "$WORK/DONE"
finish() { echo "$1" > "$WORK/DONE"; say "end rc=$1"; exit "$1"; }
# shellcheck disable=SC2086
(cd "$NEW" && md5sum * | sed "s|^|head |"; cd "$REF" && md5sum * | sed "s|^|base |"; md5sum "$D"/* "$CMP" "$PS"/*.sh "$PS"/prompts/*) \
    > "$WORK/md5.txt" 2>&1
say "start: steps '$STEPS', NEW=$NEW REF=$REF, p106 setting: $ON"

if step quant; then
    : "${QUANT:?QUANT=<llama-quantize>}"
    mkdir -p "$Q8/e32"
    for f in qwen4exp-moe.gguf e32/qwen4exp-moe.gguf; do
        [ -s "$Q8/$f" ] && continue
        # shellcheck disable=SC2086
        docker run --rm ${DOCKER_ARGS:-} --network none -v "$MD:$MD" -v "$(dirname "$QUANT"):$(dirname "$QUANT"):ro" \
            "$CPU_IMAGE" "$QUANT" $QUANT_ARGS "$MD/$f" "$Q8/$f" Q8_0 4 > "$WORK/logs/quant-${f//\//-}.log" 2>&1
        say "quantize $f rc=$?"
    done
    (cd "$MD" && md5sum qwen4exp-moe.gguf qwen4exp-moe-screen.gguf e32/qwen4exp-moe.gguf q8/qwen4exp-moe.gguf \
        q8/e32/qwen4exp-moe.gguf) > "$WORK/compare/models.md5"
    python3 -I "$D/model_types.py" "$MD/qwen4exp-moe.gguf" "$MD/qwen4exp-moe-screen.gguf" "$MD/e32/qwen4exp-moe.gguf" \
        "$Q8/qwen4exp-moe.gguf" "$Q8/e32/qwen4exp-moe.gguf" | sed "s|$MD/||" > "$WORK/compare/model_types.txt"
    cat "$WORK/compare/models.md5" "$WORK/compare/model_types.txt"
    [ -s "$Q8/qwen4exp-moe.gguf" ] && [ -s "$Q8/e32/qwen4exp-moe.gguf" ] || finish 1
fi

if step runs; then
    if [ -n "$PREFLIGHT" ]; then
        $PREFLIGHT -g "$GPUS" > "$WORK/preflight.txt" 2>&1
        rc=$?; say "preflight rc=$rc"; [ $rc = 0 ] || finish 1
    fi
    for r in $RUNS; do
        models=${r%%-*}; rest=${r#*-}; setting=${rest%%-*}; build=${rest#*-}
        case $build in head) bin=$NEW ;; base|base2) bin=$REF ;; *) say "$r: unknown build"; finish 1 ;; esac
        case $setting in stock) senv="" ;; p106) senv=$ON ;; *) say "$r: unknown setting"; finish 1 ;; esac
        case $models in
        f32) menv="M=$MD/qwen4exp-moe.gguf E32=$MD/e32/qwen4exp-moe.gguf SCREEN=$MD/qwen4exp-moe-screen.gguf PROMPTS=$MD/prompts.txt"
             if [ "$setting" = stock ]; then parts="ps prc screen moe hop pool"; else parts="ps prc screen moe hop"; fi ;;
        q8)  menv="M=$Q8/qwen4exp-moe.gguf E32=$Q8/e32/qwen4exp-moe.gguf"; parts="ps prc moe hop" ;;
        *) say "$r: unknown models"; finish 1 ;;
        esac
        o=$WORK/tools/$r
        rm -rf "$o" && mkdir -p "$o"
        printf 'bin: %s\nenv: %s\nmodels: %s\nparts: %s\n' "$bin" "${senv:-none}" "$menv" "$parts" > "$o/command.txt"
        say "$r: start ($parts)"
        # shellcheck disable=SC2086
        timeout "$RUN_TIMEOUT" "${DR[@]}" "${JIT[@]}" --name "c957-$r" --gpus "\"device=$GPUS\"" "$IMG" \
            bash -c "env $senv $menv BIN=$bin PS=$PS OUT=$o PARTS='$parts' TS=$TS THREADS=$THREADS bash $D/gpu_tools.sh" \
            > "$WORK/logs/$r.log" 2>&1
        say "$r: rc=$? $(cut -f2 "$o/rc.tsv" | sort | uniq -c | awk '{printf "%s rc=%s; ", $1, $2}')"
        docker rm -f "c957-$r" > /dev/null 2>&1
    done
fi

if step probe; then
    mkdir -p "$WORK/probe"
    for setting in stock p106; do
        senv=""; [ $setting = p106 ] && senv=$ON
        # shellcheck disable=SC2086
        timeout 300 "${DR[@]}" "${JIT[@]}" --name "c957-probe-$setting" --gpus "\"device=$GPUS\"" "$IMG" \
            bash -c "env $senv $NEW/llama-completion -m $Q8/qwen4exp-moe.gguf -p 'The quick brown fox' -n 1 \
                -ngl 99 -ts $TS -t $THREADS -c 256 -no-cnv --temp 0" > "$WORK/probe/$setting.log" 2>&1 < /dev/null
        say "probe $setting rc=$?: $(grep -aoE '(MMV[QF]_PASCAL|FATTN[A-Z_]*) = 1' "$WORK/probe/$setting.log" | sort -u | paste -sd' ')"
        docker rm -f "c957-probe-$setting" > /dev/null 2>&1
    done
fi

if step compare; then
    K=$WORK/compare
    pair() { # <a> <b>: compare.py's table of run a against run b
        [ -d "$WORK/tools/$1" ] && [ -d "$WORK/tools/$2" ] || return 0
        python3 -I "$CMP" "$WORK/tools/$1" "$WORK/tools/$2" "$K/$1__$2.tsv" > "$K/$1__$2.txt" 2>&1
        echo "$1 against $2 (rc=$?): $(head -n 1 "$K/$1__$2.txt")"
    }
    {
        echo "# base against head (the check)"
        pair f32-stock-base f32-stock-head
        pair f32-p106-base f32-p106-head
        pair q8-p106-base q8-p106-head
        echo "# the runs repeat"
        pair f32-stock-base f32-stock-base2
        echo "# the setting's reach"
        pair f32-stock-base f32-p106-base
        pair q8-stock-base q8-p106-base
    } | tee "$K/summary.txt"
    # which of the fork's Pascal kernels are on per setting (the probe's ggml CUDA feature list), and the cards the
    # tools' logs name
    printf 'setting\tCUDA features (the probe)\n' > "$K/features.tsv"
    for setting in stock p106; do
        [ -f "$WORK/probe/$setting.log" ] || continue
        printf '%s\t%s\n' "$setting" "$(grep -aoE 'CUDA : [^|]*(\| *[A-Z_0-9]+ = [^|]*)*' "$WORK/probe/$setting.log" | head -n 1)" \
            >> "$K/features.tsv"
    done
    printf '\nrun\tlogs\tcompute capability (lines)\n' >> "$K/features.tsv"
    for r in $RUNS; do
        o=$WORK/tools/$r
        [ -d "$o" ] || continue
        printf '%s\t%s\t%s\n' "$r" "$(find "$o" -name '*.log' | wc -l)" \
            "$(grep -rhoa 'compute capability [0-9.]*' "$o" --include='*.log' | sort | uniq -c | awk '{printf "%s×%s ", $1, $4}')" \
            >> "$K/features.tsv"
    done
    cat "$K/features.tsv"
    # the two rig builds' warning lines (C880's warnings(): each once, without line and column numbers)
    export LC_ALL=C
    warnings() { grep -aE 'warning:|warning #[0-9]|CMake Warning' "$1" \
        | sed -E 's/:[0-9]+:[0-9]+: warning/: warning/; s/\([0-9]+\): warning/: warning/; s/CMakeLists.txt:[0-9]+/CMakeLists.txt/' | sort -u; }
    if [ -f "$WORK/logs/rig-head.log" ] && [ -f "$WORK/logs/rig-base.log" ]; then
        warnings "$WORK/logs/rig-base.log" > "$K/rig-warnings-base.txt"
        warnings "$WORK/logs/rig-head.log" > "$K/rig-warnings-head.txt"
        echo "rig warnings: base $(wc -l < "$K/rig-warnings-base.txt") lines ($(grep -c 'is deprecated: use llama_' "$K/rig-warnings-base.txt") deprecated batch API)," \
            "head $(wc -l < "$K/rig-warnings-head.txt") ($(grep -c 'is deprecated: use llama_' "$K/rig-warnings-head.txt")), only in head: $(comm -13 "$K/rig-warnings-base.txt" "$K/rig-warnings-head.txt" | wc -l)" \
            | tee "$K/rig-warnings.txt"
        comm -13 "$K/rig-warnings-base.txt" "$K/rig-warnings-head.txt" | tee -a "$K/rig-warnings.txt"
    fi
fi
finish 0
