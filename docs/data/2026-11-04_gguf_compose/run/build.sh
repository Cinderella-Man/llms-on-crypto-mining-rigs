#!/usr/bin/env bash
# Q597 (FN3) and Q620: the files of this folder, built one after another as one detached job at the lowest CPU and
# disk priority, each checked: `verify` against its inputs, the md5 of every shard it wrote, llama.cpp's loader on its
# headers and tensor bounds (llama-quantize --dry-run), and for a retype its converted tensors against llama.cpp's own
# quantizer (scripts/gguf_retype_test.py with llama-quantize and libggml-base.so). A file leaves $WORK/done/<name>
# (its end time) or $WORK/failed/<name> (why), and its log in $WORK/logs/<name>.log; done files are skipped, so after
# a failure the job is simply started again. A file $WORK/STOP stops it before the next file.
#   SCRIPTS=<repo>/scripts OUT=<models folder> WORK=<work folder> \
#   Q2_0=<first shard of Swift Q2_0> Q2_0_HCQ8=<first shard of its hcq8 retype> IQ3_HCQ8=<first shard of Swift IQ3_XXS
#   hcq8> Q8_0=<first shard of Swift-1.5 Q8_0> TABLE=<a file of the n-gram table shard, -00003-of-00003> \
#   QUANTIZE=<llama-quantize> GGML=<libggml-base.so> [JOBS=6] [CPUS=10,11,22,23] [FILES="q2_0-hcq6k ..."] \
#     setsid nohup bash build.sh >> $WORK/logs/build.log 2>&1 < /dev/null &
# The files (FILES picks and orders them; the default builds the first five):
#   q2_0-hcq6k      Swift Q2_0, hyper-connections in Q6_K (the 320-wide up projections in Q8_0: Q6_K needs rows of
#                   256), router F16
#   q2_0-hcq5       Swift Q2_0, hyper-connections in Q5_0, router F16
#   iq3_xxs-spine8  Swift IQ3_XXS hcq8 with every dense matrix (attention, recurrent output, shared expert, output head)
#                   taken from the Q8_0
#   iq3_xxs-spine5  iq3_xxs-spine8 with those matrices requantized to Q5_0 (the output head stays Q8_0)
#   q2_0-hcq6k5     Swift Q2_0, hyper-connections in Q6_K with the up projections in Q5_0, router F16
#   iq3_xxs-spine4  as iq3_xxs-spine5 in Q4_0
#   q2_0-spine8     Swift Q2_0 hcq8 with the dense matrices from the Q8_0; q2_0-spine5 the same in Q5_0
# Each Q2_0 file's shard 3 is the table shard unchanged, so it is written as a link to TABLE (or, for a spine, to its
# base's table file), not copied.
# CPUS pins the job to those logical CPUs (two whole cores here), so that its workers share no core with a
# KL divergence run beside it: at the lowest priority, six unpinned workers still lengthened such a run's CPU-heavy
# agent text (its time against the wiki text's: 1.53, against 1.18 and 1.31 before; 0.94 once pinned to two cores).
set -uo pipefail
: "${SCRIPTS:?}" "${OUT:?}" "${WORK:?}" "${Q2_0:?}" "${Q2_0_HCQ8:?}" "${IQ3_HCQ8:?}" "${Q8_0:?}" "${TABLE:?}"
: "${QUANTIZE:?}" "${GGML:?}"
JOBS=${JOBS:-6}
FILES=${FILES:-"q2_0-hcq6k q2_0-hcq5 iq3_xxs-spine8 iq3_xxs-spine5 q2_0-hcq6k5"}
LOW=(nice -n 19 ionice -c3)
[ -n "${CPUS:-}" ] && LOW=(taskset -c "$CPUS" "${LOW[@]}")
DENSE='^blk\.[0-9]+\.(attn_(q|k|v|qkv|gate|output)|ssm_out|ffn_(gate|up|down)_shexp)\.weight$'
UPS='hc_(attn|ffn)_up|output_hc_up'
P2=Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0 P3=Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS
log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
shards() { ls "${1%-00001-of-*.gguf}"-0000?-of-0000?.gguf; }  # every shard, from the first one's name
first() { ls "$1/$2"-00001-of-0000?.gguf; }
mkdir -p "$WORK/logs" "$WORK/done" "$WORK/failed" "$WORK/test"

checks() {  # <folder> <prefix>: md5 of the shards written (links are named), llama.cpp's loader
    local dir=$1 prefix=$2
    ls -l "$dir"/"$prefix"-0000?-of-0000?.gguf
    (cd "$dir" && find . -maxdepth 1 -type f -name "$prefix-0000?-of-0000?.gguf" -printf '%f\n' | sort |
        xargs "${LOW[@]}" md5sum | tee "$prefix.md5") || return 1
    "${LOW[@]}" "$QUANTIZE" --dry-run --allow-requantize "$(first "$dir" "$prefix")" Q8_0 2 > "$dir/$prefix.load.log" 2>&1 ||
        { tail -n 5 "$dir/$prefix.load.log"; return 1; }
    grep -E "loaded meta data|type +[A-Za-z0-9_]+: +[0-9]+ tensors" "$dir/$prefix.load.log"
}

do_retype() {  # <name> <folder> <prefix> "<old shards>" <rule>...
    local name=$1 dir=$OUT/$2 prefix=$3 r
    local old=($4) rules=() new
    shift 4
    for r in "$@"; do rules+=(--rule "$r"); done
    "${LOW[@]}" python3 -I "$SCRIPTS/gguf_retype.py" retype "${rules[@]}" --out-dir "$dir" --prefix "$prefix" \
        --jobs "$JOBS" --link-unchanged "${old[@]}" || return 1
    new=($(ls "$dir/$prefix"-0000?-of-0000?.gguf))
    "${LOW[@]}" python3 -I "$SCRIPTS/gguf_retype.py" verify "${old[@]}" --new "${new[@]}" || return 1
    checks "$dir" "$prefix" || return 1
    "${LOW[@]}" python3 -I "$SCRIPTS/gguf_retype_test.py" "$WORK/test/$name" --quantize "$QUANTIZE" --ggml "$GGML" \
        --threads "$JOBS" --old "${old[@]}" --new "${new[@]}"
}

do_compose() {  # <name> <folder> <prefix> <base's first shard> <take>...
    local name=$1 dir=$OUT/$2 prefix=$3 t
    local base=($(shards "$4")) donor=($(shards "$Q8_0")) takes=() new
    shift 4
    for t in "$@"; do takes+=(--take "$t"); done
    "${LOW[@]}" python3 -I "$SCRIPTS/gguf_compose.py" compose --base "${base[@]}" --donor "${donor[@]}" "${takes[@]}" \
        --out-dir "$dir" --prefix "$prefix" --link-unchanged || return 1
    new=($(ls "$dir/$prefix"-0000?-of-0000?.gguf))
    "${LOW[@]}" python3 -I "$SCRIPTS/gguf_compose.py" verify --base "${base[@]}" --donor "${donor[@]}" "${takes[@]}" \
        --new "${new[@]}" || return 1
    checks "$dir" "$prefix"
}

q2_0_old() { local s=($(shards "$Q2_0")); echo "${s[@]:0:${#s[@]}-1} $TABLE"; }  # the table shard from TABLE

build() {
    case $1 in
        q2_0-hcq6k) do_retype "$1" swift_q2_0_hcq6k "$P2-hcq6k" "$(q2_0_old)" "$UPS=Q8_0@BF16" "hc_=Q6_K@BF16" \
            "ffn_gate_inp=F16@F32" ;;
        q2_0-hcq5) do_retype "$1" swift_q2_0_hcq5 "$P2-hcq5" "$(q2_0_old)" "hc_=Q5_0@BF16" "ffn_gate_inp=F16@F32" ;;
        q2_0-hcq6k5) do_retype "$1" swift_q2_0_hcq6k5 "$P2-hcq6k5" "$(q2_0_old)" "$UPS=Q5_0@BF16" "hc_=Q6_K@BF16" \
            "ffn_gate_inp=F16@F32" ;;
        iq3_xxs-spine8) do_compose "$1" swift_iq3_xxs_spine8 "$P3-hcq8-spine8" "$IQ3_HCQ8" "re:$DENSE" output.weight ;;
        iq3_xxs-spine5|iq3_xxs-spine4)
            local k=${1##*spine}
            do_retype "$1" "swift_iq3_xxs_spine$k" "$P3-hcq8-spine$k" \
                "$(shards "$(first "$OUT/swift_iq3_xxs_spine8" "$P3-hcq8-spine8")")" "$DENSE=Q${k}_0@Q8_0" ;;
        q2_0-spine8) do_compose "$1" swift_q2_0_spine8 "$P2-hcq8-spine8" "$Q2_0_HCQ8" "re:$DENSE" output.weight ;;
        q2_0-spine5)
            do_retype "$1" swift_q2_0_spine5 "$P2-hcq8-spine5" \
                "$(shards "$(first "$OUT/swift_q2_0_spine8" "$P2-hcq8-spine8")")" "$DENSE=Q5_0@Q8_0" ;;
        *) echo "unknown file $1"; return 1 ;;
    esac
}

log "build start (pid $$): $FILES"
for f in $FILES; do
    if [ -e "$WORK/done/$f" ]; then log "$f: done before"; continue; fi
    if [ -e "$WORK/STOP" ]; then log "$f: not started (STOP)"; break; fi
    rm -f "$WORK/failed/$f"
    log "$f: start"
    build "$f" >> "$WORK/logs/$f.log" 2>&1 < /dev/null
    rc=$?
    if [ $rc -eq 0 ]; then
        date -Is > "$WORK/done/$f"; log "$f: done"
    else
        echo "exit $rc at $(date -Is)" > "$WORK/failed/$f"; log "$f: FAILED (exit $rc, see logs/$f.log)"
    fi
done
log "build end: done $(ls "$WORK/done" | tr '\n' ' '); failed $(ls "$WORK/failed" | tr '\n' ' ')"
