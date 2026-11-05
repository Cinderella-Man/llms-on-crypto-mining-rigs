#!/usr/bin/env bash
# The v3 release's overnight quality and file work on a build machine with RTX 30 cards (2026-11-05): one detached
# script, four lanes side by side. Each step is skipped when its marker WORK/done/<step> exists, so a second start
# resumes; WORK/summary.tsv has one line per step (status, start, end, the kernel log's Xid count before and after,
# the result in one line). Nothing here measures speed.
#
#   lane orn    (ORN_GPUS)   B-ornith-sw, then (after build) A and B-ornith-lowbit, then the Ornith gates
#   lane flint  (FLINT_GPUS) B-flint and its gates
#   lane build  (CPU)        build
#   lane files  (CPU, disk)  C-pmix, C-valid1, C-p065
#
# build  -lowbit (d7a75deb3, Q189's build of Ornith's valid1) rebuilt for compute 6.1 only by
#        scripts/build_llamacpp_docker.sh, under the machine's build lock. Its rig tarball of 2026-10-27 also carries
#        compute 7.0 code, which an RTX 30 card's driver compiles in place of the 6.1 PTX, and a kernel launch of that
#        code fails there ("CUDA error: invalid argument" in ggml_cuda_kernel_launch).
# A      Ornith valid1's greedy identity, 1551f5fe1 against d7a75deb3, through the compute 6.1 PTX: each build's
#        llama-server with the release regression's flags (greedy_identity.sh's COMMON at CTX, then ORN_LINE) on
#        ORN_GPUS, the prompts of ../../2026-11-05_release_regression/run/prompts.jsonl at temperature 0
#        (greedy_requests.py), then compare_greedy.py (verdict IDENTICAL or DIFFERS, with the margin at each first
#        difference).
# B      each switch's KL divergence against the model's reference logits: ../../2026-11-03_flash_next_quality/run/
#        kld.sh unchanged (one wrapper per arm next to a build's llama-perplexity), then
#        ../../2026-11-04_moe_tail_kld/run/table.sh paired by chunk against the arm off, and C200k's gate per arm and
#        text: within noise when d - 2 se <= 0. Arms on 1551f5fe1: off (every switch unset) and tail
#        (LLAMA_MOE_TAIL=TAIL); for Ornith also lowbit (d7a75deb3, every switch unset), whose pairing with off
#        (t3a.tsv: off's change over lowbit) answers T3a, whether the release build is worse than Q189's build.
#        The card-hop cast (LLAMA_HOP_CAST) has no arm: in 1551f5fe1 only qwen4exp's graph calls build_hop_cast, so on
#        Flint (flint) and Ornith (qwen35moe) it changes nothing.
# C      the release model files: each file's metadata (scripts/gguf_release_header.py) and the strings that look like
#        a local path, a host or an address; a copy in REL/<folder>/ with quantize.imatrix.file reduced to the file's
#        own name and every tensor byte for byte (the tool verifies it), then its sha256. p065 is composed again first
#        (E3's recipe, ../../2026-10-27_flint_near_q8/run/near_q8.py, from P065_BASE and P065_DONOR) and must match
#        P065_MD5, the md5 of the file the rig ran.
#
# Env (the machine's values in one file, sourced by `bash chain.sh <env file>`):
#   WORK (scratch, fast disk), REPO (this repository's copy on the machine), GGUFPY (a gguf-py that knows the fork's
#   types), L5 (the 1551f5fe1 rig build's folder), LB_SRC (a llama.cpp clone with the fork's refs), LB_REF, LOCK (the
#   build lock), WAIT_FILE + WAIT_MARK (start lane orn once that file holds that line), VALID1, FLINT, ORN_BASES and
#   FLINT_BASES ("<text>=<base .kld> ..."), WIKI AGENT CODE (the texts, which kld.sh passes; the tokens come from the
#   bases), ORN_GPUS and FLINT_GPUS (PCI addresses, e.g. "09:00.0 0a:00.0"), ORN_TS and FLINT_TS (-ts in PCI order),
#   MOUNTS (folders mounted at the same path), PY_IMAGE (python3 with numpy), REL (the release copies), PMIX (shards to
#   copy), P065_BASE P065_DONOR P065_MD5; optional IMAGE, TAIL (p=0.9,renorm=1,hist=0), CTX (49152), N (256), PORT
#   (18801), JOBS (8), LANES ("orn flint build files"), FILES_AFTER, HOST_FLAGS (patterns for the machines' names, flagged in
#   the headers), PREFLIGHT (scripts/cuda_preflight.sh), MIN_FREE_GB (50)
set -uo pipefail
[ -n "${1:-}" ] && . "$1"
: "${WORK:?}" "${REPO:?}" "${GGUFPY:?}" "${L5:?}" "${LB_SRC:?}" "${LB_REF:?}" "${LOCK:?}" "${VALID1:?}" "${FLINT:?}"
: "${ORN_BASES:?}" "${FLINT_BASES:?}" "${WIKI:?}" "${AGENT:?}" "${CODE:?}" "${ORN_GPUS:?}" "${FLINT_GPUS:?}"
: "${ORN_TS:?}" "${FLINT_TS:?}" "${MOUNTS:?}" "${PY_IMAGE:?}" "${REL:?}" "${PMIX:?}" "${P065_BASE:?}" "${P065_DONOR:?}"
: "${P065_MD5:?}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
D=$REPO/docs/data
QR=$D/2026-11-03_flash_next_quality/run
RR=$D/2026-11-05_release_regression/run
TAIL=${TAIL:-p=0.9,renorm=1,hist=0}
CTX=${CTX:-49152}; N=${N:-256}; PORT=${PORT:-18801}
PREFLIGHT=${PREFLIGHT:-$REPO/scripts/cuda_preflight.sh}
export PREFLIGHT PREFLIGHT_CACHE=${PREFLIGHT_CACHE:-$WORK/preflight}
TAG=night
. "$QR/common.sh"
mkdir -p "$WORK"/{done,res,logs,cache,wrap,run}

xid() { journalctl -k -b 2> /dev/null | grep -ci xid; }
gpu_ids() { # "09:00.0 0a:00.0" -> "3,4": NVML indices (PCI order, what docker's device= takes)
    local p i out=""
    for p in $1; do
        i=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader |
            awk -F', ' -v p="$p" 'index(toupper($2), toupper(p)) { print $1 }')
        [ -n "$i" ] || { log "no GPU at $p" >&2; return 1; }
        out=${out:+$out,}$i
    done
    echo "$out"
}
step() { # <step> <function> [args]: the function's output in WORK/logs/<step>.log, its one-line result in WORK/res/<step>
    local s=$1 t0 x0 x1 rc st; shift
    S=$s; rm -f "$WORK/res/$S"
    [ -e "$WORK/done/$s" ] && { log "$s: done before"; return 0; }
    t0=$(date -Is); x0=$(xid); log "$s: start"
    "$@" > "$WORK/logs/$s.log" 2>&1; rc=$?
    x1=$(xid)
    if [ $rc -eq 0 ]; then st=ok; date -Is > "$WORK/done/$s"; else st=FAIL; fi
    [ "$x1" -gt "$x0" ] && st="$st XID"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$s" "$st" "$t0" "$(date -Is)" "$x0" "$x1" \
        "$(tr '\t\n' '  ' < "$WORK/res/$S" 2> /dev/null)" >> "$WORK/steps.tsv"
    log "$s: $st (Xid $x0 -> $x1) $(head -c 400 "$WORK/res/$S" 2> /dev/null)"
    [ $rc -eq 0 ] && [ "$x1" -eq "$x0" ]
}
lb_dir() { ls -d "$WORK"/build/llama.cpp-*-"${LB_REF:0:9}"* 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
free_gb() { df -B1000000000 --output=avail "$1" | tail -n 1 | tr -d ' '; }

# ---- build: -lowbit for compute 6.1 only
s_build() {
    local b=$WORK/build
    [ "$(free_gb "$WORK")" -ge $(( ${MIN_FREE_GB:-50} + 10 )) ] || { echo "under $(( ${MIN_FREE_GB:-50} + 10 )) GB free on $WORK" > "$WORK/res/$S"; return 1; }
    mkdir -p "$b"
    if [ ! -d "$b/src/.git" ]; then
        git clone -q --no-local --no-checkout "$LB_SRC" "$b/src" || return 1
        git -C "$b/src" fetch -q "$LB_SRC" 'refs/remotes/fork/*:refs/remotes/fork/*' || return 1
        git -C "$b/src" remote set-url origin https://github.com/ggml-org/llama.cpp
    fi
    git -C "$b/src" cat-file -e "$LB_REF^{commit}" || { echo "no $LB_REF in the clone" > "$WORK/res/$S"; return 1; }
    flock "$LOCK" env WORK="$b" JOBS="${JOBS:-8}" DOCKER_ARGS="--label pascal-quants=1 --cpu-shares 256" \
        nice -n 19 bash "$REPO/scripts/build_llamacpp_docker.sh" "$LB_REF" || return 1
    local d; d=$(lb_dir)
    [ -x "$d/bin/llama-server" ] && [ -x "$d/bin/llama-perplexity" ] || { echo "no binaries in ${d:-$b}" > "$WORK/res/$S"; return 1; }
    echo "$(basename "$d") md5 $(md5sum < "$d.tar.gz" | cut -c1-32); $(grep -i -m1 -E 'arch' "$d/BUILDINFO.txt" 2> /dev/null)" > "$WORK/res/$S"
}

# ---- A: Ornith valid1's greedy identity through the PTX
COMMON="-ngl 99 -fa on -c $CTX -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek \
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui"
s_A() {
    local g=$1 o=$WORK/A t b c i
    mkdir -p "$o"
    cp "$RR/prompts.jsonl" "$RR/greedy_requests.py" "$RR/compare_greedy.py" "$o/"
    for f in $(grep -o '"attach": "[^"]*"' "$RR/prompts.jsonl" | cut -d'"' -f4); do cp "$REPO/$f" "$o/"; done
    for t in lowbit ladder5; do
        [ -s "$o/greedy-$t.jsonl" ] && continue
        if [ $t = lowbit ]; then b=$(lb_dir); else b=$L5; fi
        c=pq-$TAG-orn-id-$t
        echo "=== $t $(date -Is): $b"
        # shellcheck disable=SC2086
        docker run -d --rm --name "$c" $DOCKER_BASE --gpus "\"device=$g\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
            -e CUDA_CACHE_PATH="$WORK/cache" -e CUDA_CACHE_MAXSIZE=4294967296 -p "127.0.0.1:$PORT:$PORT" \
            -v "$WORK:$WORK" -v "$b:$b:ro" $MOUNT_ARGS "$IMAGE" \
            "$b/bin/llama-server" -m "$VALID1" --host 0.0.0.0 --port "$PORT" $COMMON $ORN_LINE || return 1
        for i in $(seq 360); do
            curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break
            docker ps -q --filter "name=^$c\$" | grep -q . || break
            sleep 5
        done
        if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
            docker logs "$c" > "$o/server-$t.log" 2>&1; docker stop "$c" > /dev/null 2>&1
            echo "$t: the server did not start ($o/server-$t.log)" > "$WORK/res/$S"; return 1
        fi
        (cd "$o" && timeout 3600 nice -n 19 python3 -I greedy_requests.py "$PORT" prompts.jsonl "$N" "greedy-$t.jsonl.part")
        echo "requests rc=$?"
        docker logs "$c" > "$o/server-$t.log" 2>&1; docker stop "$c" > /dev/null
        [ -s "$o/greedy-$t.jsonl.part" ] && mv "$o/greedy-$t.jsonl.part" "$o/greedy-$t.jsonl"
    done
    [ -s "$o/greedy-lowbit.jsonl" ] && [ -s "$o/greedy-ladder5.jsonl" ] || { echo "no outputs" > "$WORK/res/$S"; return 1; }
    (cd "$o" && python3 -I compare_greedy.py greedy-lowbit.jsonl greedy-ladder5.jsonl > identity.md)
    tail -n 1 "$o/identity.md" > "$WORK/res/$S"
}
ORN_LINE="-b 2048 -ub 512 -ts $ORN_TS -fit off"

# ---- B: the KL divergence of each arm
wrapper() { # <arm> <build folder> <exports>: WORK/wrap/<arm>/bin/llama-perplexity, which sets them and runs the build's
    local w=$WORK/wrap/$1/bin
    mkdir -p "$w"
    printf '#!/usr/bin/env bash\n# chain.sh arm %s\nexport CUDA_CACHE_PATH=%s CUDA_CACHE_MAXSIZE=4294967296 %s\nexec %s/bin/llama-perplexity "$@"\n' \
        "$1" "$WORK/cache" "$3" "$2" > "$w/llama-perplexity"
    chmod +x "$w/llama-perplexity"
    echo "$w/llama-perplexity"
}
s_B() { # <model> <gpus> <file> <bases> <ts> <arm>...: the arms' logs in WORK/B-<model>
    local m=$1 g=$2 f=$3 bases=$4 ts=$5 o=$WORK/B-$1 texts="" kv a b e ppl failed=0
    shift 5
    mkdir -p "$o"
    for kv in $bases; do texts="$texts ${kv%%=*}"; ln -sfn "${kv#*=}" "$o/base.${kv%%=*}.kld"; done
    echo "-ts $ts" > "$o/gpu_args"
    for a in "$@"; do
        case $a in
            off) b=$L5 e="" ;;
            tail) b=$L5 e="LLAMA_MOE_TAIL=$TAIL" ;;
            lowbit) b=$(lb_dir) e="" ;;
        esac
        ppl=$(wrapper "$m-$a" "$b" "$e")
        PPL=$ppl OUT=$o GPUS=$g TAG=$TAG-$m MOUNTS="$MOUNTS $WORK $b" WIKI=$WIKI AGENT=$AGENT CODE=$CODE \
            TEXTS="${texts# }" bash "$QR/kld.sh" kld "$f" "$a" || failed=1
    done
    echo "$m: arms $* done, failed=$failed" > "$WORK/res/$S"
    return $failed
}
gate() { # <model>: WORK/B-<model>/{kld.tsv,gate.tsv}, and t3a.tsv when the arm lowbit ran; the result line
    local m=$1 o=$WORK/B-$1
    OUT=$o OFF=off CTRL=none bash "$D/2026-11-04_moe_tail_kld/run/table.sh" > "$o/kld.tsv"
    awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; print "arm\ttext\tmean_kld\td_kld_vs_off\tse\tverdict"; next }
        $(c["arm"]) != "off" && $(c["d_kld_vs_off"]) != "-" {
            d = $(c["d_kld_vs_off"]); s = $(c["d_vs_off_se"])
            printf "%s\t%s\t%s\t%s\t%s\t%s\n", $(c["arm"]), $(c["text"]), $(c["mean_kld"]), d, s,
                (d - 2 * s <= 0 ? "within noise" : "ABOVE NOISE") }' "$o/kld.tsv" > "$o/gate.tsv"
    if ls "$o"/lowbit.*.log > /dev/null 2>&1; then
        OUT=$o OFF=lowbit CTRL=none bash "$D/2026-11-04_moe_tail_kld/run/table.sh" |
            awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; print "arm\ttext\td_kld_vs_lowbit\tse\tverdict"; next }
                $(c["arm"]) == "off" && $(c["d_kld_vs_off"]) != "-" {
                    d = $(c["d_kld_vs_off"]); s = $(c["d_vs_off_se"])
                    printf "1551f5fe1\t%s\t%s\t%s\t%s\n", $(c["text"]), d, s, (d - 2 * s <= 0 ? "not worse" : "WORSE") }' > "$o/t3a.tsv"
    fi
    {
        awk -F'\t' 'NR > 1 { printf "%s %s %+.5f±%.5f %s; ", $1, $2, $4, $5, $6 }' "$o/gate.tsv"
        [ -s "$o/t3a.tsv" ] && awk -F'\t' 'NR > 1 { printf "T3a %s %+.5f±%.5f %s; ", $2, $3, $4, $5 }' "$o/t3a.tsv"
    } > "$WORK/res/$S"
    [ "$(wc -l < "$o/gate.tsv")" -gt 1 ]
}

# ---- C: the release model files
py() { # gguf-py scripts with python3 -I in PY_IMAGE, nothing but the files' folders mounted, no network
    local m="" p
    for p in "$@"; do case $p in "$WORK"/*|"$REL"/*) ;; /*) m="$m -v $(dirname "$p"):$(dirname "$p"):ro" ;; esac; done
    # shellcheck disable=SC2086
    docker run --rm $DOCKER_BASE --network none -v "$WORK:$WORK" -v "$REL:$REL" $m "$PY_IMAGE" \
        nice -n 19 ionice -c3 python3 -I "$WORK/run/pyrun.py" "$GGUFPY" "$@"
}
# strings that look like a local path, an address or (HOST_FLAGS, from the environment) one of the machines' names
FLAGS='= .(/|~)|/(srv|mnt|home|root|tmp|opt|media|var|Users)/|localhost|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'${HOST_FLAGS:+|$HOST_FLAGS}
c_file() { # <source> <folder>: the audit in WORK/C, the copy and its sha256 in REL/<folder>
    local src=$1 dir=$REL/$2 n a k v chg=()
    n=$(basename "$src") a=$WORK/C/$(basename "$src" .gguf)
    mkdir -p "$dir" "$WORK/C"
    py "$WORK/run/gguf_release_header.py" "$src" > "$a.kv" || { echo "$n: cannot read the metadata" >> "$WORK/res/$S"; return 1; }
    grep -E "$FLAGS" "$a.kv" | grep -v '^tokenizer\.' > "$a.flags"
    # the importance matrix's file and text as their own names, without the folders they were made in
    for k in quantize.imatrix.file quantize.imatrix.dataset; do
        v=$(sed -n "s/^${k//./\\.} = '\(.*\)'$/\1/p" "$a.kv")
        case $v in */*) chg+=(--set "$k=${v##*/}") ;; esac
    done
    if [ ! -s "$dir/$n" ]; then
        rm -f "$dir/$n.part"
        if [ ${#chg[@]} -eq 0 ]; then
            # nothing to change (a later shard of a split file holds only the split keys): the file as it is
            nice -n 19 ionice -c3 cp "$src" "$dir/$n.part" && cmp -s "$src" "$dir/$n.part" ||
                { echo "$n: the plain copy failed" >> "$WORK/res/$S"; return 1; }
            echo "copied unchanged, cmp equal" > "$a.write"
        else
            py "$WORK/run/gguf_release_header.py" "$src" "$dir/$n.part" "${chg[@]}" > "$a.write" 2>&1 || {
                echo "$n: the copy failed ($a.write)" >> "$WORK/res/$S"; return 1; }
            grep -q '^verified:' "$a.write" || { echo "$n: not verified ($a.write)" >> "$WORK/res/$S"; return 1; }
        fi
        mv "$dir/$n.part" "$dir/$n"
    fi
    (cd "$dir" && nice -n 19 ionice -c3 sha256sum "$n" > "$n.sha256")
    py "$WORK/run/gguf_release_header.py" "$dir/$n" > "$a.copy.kv"
    grep -E "$FLAGS" "$a.copy.kv" | grep -v '^tokenizer\.' > "$a.copy.flags"
    echo "$n: $(grep -c '^~ ' "$a.write") keys changed ($(grep '^~ ' "$a.write" | sed 's/ = .* -> / -> /' | tr '\n' ';')), flags: source $(wc -l < "$a.flags"), copy $(wc -l < "$a.copy.flags")$( [ -s "$a.copy.flags" ] && echo ' REVIEW'); sha256 $(cut -c1-16 "$dir/$n.sha256")…" >> "$WORK/res/$S"
}
s_pmix() { local f r=0; for f in $PMIX; do c_file "$f" flash-next-pmix-q2-fast || r=1; done; return $r; }
s_valid1() { c_file "$VALID1" ornith-valid1; }
s_p065() {
    local w=$REL/work out=$REL/work/flint-near-q8-p065.gguf m
    : > "$WORK/res/$S"
    mkdir -p "$w"
    if [ ! -s "$out" ]; then
        cp "$D/2026-10-27_flint_near_q8/run/near_q8.py" "$REPO/scripts/compose_gguf.py" "$WORK/run/"
        # compose_gguf.py runs in a child python with PYTHONPATH: an empty yaml module for gguf-py, as in pyrun.py
        mkdir -p "$WORK/pystub" && : > "$WORK/pystub/yaml.py"
        # shellcheck disable=SC2086
        docker run --rm $DOCKER_BASE --network none -v "$WORK:$WORK" -v "$w:$w" -v "$(dirname "$P065_BASE"):$(dirname "$P065_BASE"):ro" \
            -v "$(dirname "$P065_DONOR"):$(dirname "$P065_DONOR"):ro" -e PYTHONPATH="$GGUFPY:$WORK/pystub" "$PY_IMAGE" \
            nice -n 19 ionice -c3 python3 "$WORK/run/near_q8.py" compose "$P065_BASE" "$P065_DONOR" "$out.part" \
            q4 q5 kv4 kv5 o4 o5 sh4 sh5 gu5 head || { echo "p065: the compose failed" > "$WORK/res/$S"; return 1; }
        mv "$out.part" "$out"
    fi
    m=$(nice -n 19 ionice -c3 md5sum < "$out" | cut -c1-32)
    if [ "$m" != "$P065_MD5" ]; then echo "p065: composed md5 $m, not the rig file's $P065_MD5: no copy" > "$WORK/res/$S"; return 1; fi
    echo "p065: composed again, md5 $m = the rig file's" > "$WORK/res/$S"
    c_file "$out" flint-p065
}
s_disk() {
    mkdir -p "$REL"
    echo "WORK $(free_gb "$WORK") GB free, REL $(free_gb "$REL") GB free" > "$WORK/res/$S"
    [ "$(free_gb "$REL")" -ge 250 ]
}

# ---- the lanes
lane_orn() {
    if [ -n "${WAIT_FILE:-}" ]; then
        until grep -q "${WAIT_MARK:?}" "$WAIT_FILE" 2> /dev/null; do sleep 60; done
        log "orn: $WAIT_FILE has $WAIT_MARK"
    fi
    while docker ps --format '{{.Names}}' | grep -q '^t23kld'; do sleep 60; done
    local g; g=$(gpu_ids "$ORN_GPUS") || return 1
    wait_gpus "$g" || return 1
    step B-ornith-sw s_B ornith "$g" "$VALID1" "$ORN_BASES" "$ORN_TS" off tail
    until [ -e "$WORK/done/build" ] || grep -q $'^build\tFAIL' "$WORK/steps.tsv" 2> /dev/null; do sleep 60; done
    if [ -e "$WORK/done/build" ]; then
        step A s_A "$g"
        step B-ornith-lowbit s_B ornith "$g" "$VALID1" "$ORN_BASES" "$ORN_TS" lowbit
    fi
    step gate-ornith gate ornith
}
lane_flint() {
    local g; g=$(gpu_ids "$FLINT_GPUS") || return 1
    wait_gpus "$g" || return 1
    step B-flint s_B flint "$g" "$FLINT" "$FLINT_BASES" "$FLINT_TS" off tail
    step gate-flint gate flint
}
lane_build() { step build s_build; }
lane_files() {
    # FILES_AFTER=<step>: a second start of this lane beside a running one waits for that step's line in steps.tsv
    if [ -n "${FILES_AFTER:-}" ]; then
        until grep -q "^$FILES_AFTER"$'\t' "$WORK/steps.tsv" 2> /dev/null; do sleep 60; done
    fi
    step C-disk s_disk && { step C-pmix s_pmix; step C-valid1 s_valid1; step C-p065 s_p065; }
}

cp "$HERE/pyrun.py" "$REPO/scripts/gguf_release_header.py" "$WORK/run/" || exit 1
log "chain: start, lanes ${LANES:-orn flint build files}, Xid $(xid)"
for l in ${LANES:-orn flint build files}; do lane_$l > "$WORK/logs/lane-$l.log" 2>&1 & done
wait
{
    printf 'step\tstatus\tstart\tend\txid_before\txid_after\tresult\n'
    awk -F'\t' '!($1 in l) { k[++n] = $1 } { l[$1] = $0 } END { for (i = 1; i <= n; i++) print l[k[i]] }' "$WORK/steps.tsv"
} > "$WORK/summary.tsv"
echo "containers left: $(docker ps -q --filter name=pq-$TAG- | wc -l)"
log "chain: done, Xid $(xid)"
echo NIGHT_DONE
