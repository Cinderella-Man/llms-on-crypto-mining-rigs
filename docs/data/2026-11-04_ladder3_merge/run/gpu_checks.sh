#!/usr/bin/env bash
# C228 (Q781): GPU correctness of every switch merged into fork branch pascal-p106-v3-b12076-ladder3, one part per
# merged branch, each switch on and off, so that a failing switch is found by its part and run name. Runs natively (no
# container of its own), on one GPU, and on two for the parts that cross cards:
#   MODE=native  on P106-090s: the kernels as the rig runs them, every switch at its default (compute 6.1 turns the
#                fork's kernels on)
#   MODE=ptx     on a newer NVIDIA GPU through the build's compute_61 PTX (CUDA_FORCE_PTX_JIT=1), the fork's kernel paths
#                forced on (=2) with the P106's launch shapes (NSM=5): the kernels a P106 runs, on another card. Wrap the
#                script in the CUDA 12.9 container with the cards it may use.
# Correctness only, no timings: test-backend-ops compares the GPU with the CPU backend, the model-level parts compare
# arms of the same binary.
# Parts (PARTS, in this order; each writes OUT/<part>/ and its lines in OUT/console.txt):
#   ladder2   -ladder2's own script (docs/data/2026-11-04_ladder2_merge/run/gpu_checks.sh): BF16, Q2_0, the grid types of
#             -iqmmvq, the k-column and multi-token kernels, the GDN fusions, the persistent MoE self-check, every
#             architecture, the device checkpoint stores, the whole suite (MODE=ptx)
#   kcol      C93/C84/C85: C70's rig_kcol.sh (check, the bench's check-only run) and rig_kcol_cov.sh (Q4_0/Q5_0/IQ4_NL
#             k-column cases in every launch setting, the one-token Q2_0 small-grid rule)
#   iq2xs     C2: the IQ2_XXS/IQ2_XS/IQ3_S one-token kernels (docs/data/2026-11-04_iq2xs_kernel/run/gpu_checks.sh)
#   iqkcol    C163: the grid types and IQ1_M in the k-column and multi-token kernels (docs/data/2026-11-04_iq_kcol/run/
#             gpu_checks.sh), and mmvq_kcol --check-only on those types
#   qsa       C6: the sparse decode attention (docs/data/2026-11-04_qsa_sparse_attn/run/qsa_gpu.sh, parts check fault fasp)
#   fap       C135: FLASH_ATTN_EXT with FAP on, its three faults (2 and 3 must fail cases), its own cases with FAP off
#   hops      C200/C201: CPY at the hop shapes, test-hop-cast's round trips, and (two cards) the staged hop and the casts
#             across a real layer split, as docs/data/2026-11-04_card_hops/run/gpu_check.sh does
#   vgraphs   C151: spec_check.sh on the card (per-size graphs and schedulers, the CUDA graphs they capture, verify widths
#             1-4 on one binary: every arm the same tokens), test-llama-context at the four settings
#   ckpt      C154 with C22: server_check.sh on the card (device checkpoint stores) without and with --ctx-checkpoints-share
#   server    the host switches on the card, one llama-server per arm, the random qwen4exp drafting for itself: off,
#             LLAMA_SPEC_DUMP, LLAMA_SPEC_ADAPT_LOG, LLAMA_SPEC_ADAPT, both, LLAMA_MOE_TAIL=p=1, the four
#             GGML_CUDA_PROFILE_MODEs, --ctx-checkpoints-share, LLAMA_UBATCH_TAPER=on: greedy tokens equal to off's
#             (LLAMA_SPEC_ADAPT changes the verify widths and the taper the prompt's micro-batches, so their tokens are
#             reported, not required), the dump, log and profile files and the taper's log lines written
# Usage: BIN=<bin/ of the unpacked rig tarball, with the extra tarball's tests copied in> OUT=<dir> [MODE=native|ptx] \
#   MODELS=<dir with test-llama-archs -o models: qwen4exp-moe*, qwen35-dense*, mamba-dense*> [RUN=<docs/data of this repo>] \
#   [DEV=<card>] [DEVS2=<card,card>] [PARTS='ladder2 kcol ...'] [KCOL_BENCH=<C70's mmvq_kcol>] [MTOK_BENCH=<R7's mmvq_pascal>] \
#   [KCOL_BENCH2=<mmvq_kcol with the IQ types>] [MMVQ_PASCAL=<mmvq_pascal of the -kcol coverage bench>] [FASP=<fasp_check>] \
#   [THREADS=4] bash gpu_checks.sh
#   DEV, DEVS2  cards in CUDA_DEVICE_ORDER=PCI_BUS_ID numbering. Default: the first entry (and the first two) of
#               CUDA_VISIBLE_DEVICES when it is set, so that CUDA_VISIBLE_DEVICES=0,1,2,4,5,6 keeps every run on those
#               cards; else 0 and 0,1. An empty DEVS2 (or a single visible card) skips the two-card runs.
#   A part whose tool is not given (a bench binary, curl and jq for the server parts) prints a SKIPPED line.
# Output: OUT/console.txt (one line per run: part/run, the switch, the summary, FAIL lines, exit status, what passing
#   means), OUT/<part>/, OUT/done
set -uo pipefail
BIN=$(realpath "${BIN:?BIN=<bin/ of the unpacked tarball>}")
OUT=${OUT:?OUT=<dir>}
MODE=${MODE:-native}
RUN=${RUN:-$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)}
THREADS=${THREADS:-4}
PARTS=${PARTS:-ladder2 kcol iq2xs iqkcol qsa fap hops vgraphs ckpt server}
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
rm -f "$OUT/done"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=, read -ra POOL <<< "$CUDA_VISIBLE_DEVICES"
    DEV=${DEV:-${POOL[0]}}
    DEVS2=${DEVS2-$([ ${#POOL[@]} -ge 2 ] && echo "${POOL[0]},${POOL[1]}")}
else
    DEV=${DEV:-0}
    DEVS2=${DEVS2-0,1}
fi
if [ "$MODE" = ptx ]; then
    PTX="CUDA_FORCE_PTX_JIT=1"
    export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$OUT/nvcache} CUDA_CACHE_MAXSIZE=4294967296
else
    PTX=""
fi
C=$OUT/console.txt
part() { [[ " $PARTS " == *" $1 "* ]]; }
say() { echo "$*" | tee -a "$C"; }
summary() { # <log>: test-backend-ops', test-llama-archs' or test-llama-context's own summary lines
    grep -a -E 'tests passed|^Summary|failed +: [0-9]+|passed +: [0-9]+' "$1" | sed 's/\x1b\[[0-9;]*m//g' | sed -E 's/^ +//' \
        | sort | uniq -c | sed -E 's/^ +//' | tr '\n' ';' | cut -c1-140
}
run() { # <part/run> <the switch it tests> <what passing means> <devices> <env (K=V ...)> <command...>
    local tag=$1 what=$2 expect=$3 devs=$4 envs=$5 log
    shift 5
    log=$OUT/${tag//\//-}.log
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES="$devs" $PTX $envs timeout 7200 "$@" > "$log" 2>&1
    local rc=$?
    echo "exit $rc" >> "$log"
    local warn=""
    case "$*" in *test-backend-ops*) grep -a -q 'tests passed' "$log" || warn="NO CASES RAN; " ;; esac
    case "$*" in *test-backend-ops*|*test-llama-archs*|*test-llama-context*) grep -a -q 'CUDA0' "$log" || warn="${warn}NO CUDA0; " ;; esac
    printf '%-24s %-36s %-60s %4s FAIL  exit %-3s %s\n' "$tag" "$what" "$warn$(summary "$log")" \
        "$(grep -a -c 'FAIL' "$log")" "$rc" "$expect" | tee -a "$C"
}
sub() { # <part> <what passing means> <script> [K=V ...]: a branch's own script, its console lines folded in
    local p=$1 expect=$2 script=$3
    shift 3
    mkdir -p "$OUT/$p"
    env "$@" bash "$script" > "$OUT/$p.log" 2>&1
    local rc=$?
    if [ -s "$OUT/$p/console.txt" ]; then sed "s|^|$p/ |" "$OUT/$p/console.txt" | tee -a "$C"; else tail -n 5 "$OUT/$p.log" | sed "s|^|$p/ |" | tee -a "$C"; fi
    say "$p/ ... script exit $rc; passing: $expect"
}
skip() { say "$(printf '%-24s SKIPPED: %s' "$1" "$2")"; }
model() { ls "${MODELS:-/nonexistent}"/$1*.gguf 2>/dev/null | head -1; }
T="$BIN/test-backend-ops -b CUDA0"
NSM5=""; [ "$MODE" = ptx ] && NSM5="GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_NSM=5"
FORCE=""; [ "$MODE" = ptx ] && FORCE="GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2"
say "== $(date -Is) gpu_checks (ladder3) MODE=$MODE DEV=$DEV DEVS2=${DEVS2:-none} BIN=$BIN parts: $PARTS"
say "   $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ';')"

if part ladder2; then
    sub ladder2 "every run as -ladder2's handoff says (Q2_0's large-activation cases fail only in q2_0-off/all-off; moe-fault must differ)" \
        "$RUN/2026-11-04_ladder2_merge/run/gpu_checks.sh" \
        MODE="$MODE" BIN="$BIN" OUT="$OUT/ladder2" DEV="$DEV" DEVS2="$DEVS2" MODELS="${MODELS:-}" KCOL_BENCH="${KCOL_BENCH:-}" MTOK_BENCH="${MTOK_BENCH:-}"
fi

if part kcol; then
    if [ -n "${KCOL_BENCH2:-}" ] && [ -n "${MMVQ_PASCAL:-}" ]; then
        BA="--check-only"; [ "$MODE" = ptx ] && BA="--check-only --nsm 5"
        # shellcheck disable=SC2086
        sub kcol "every tag N/N tests passed, 0 failed; mmvq_kcol: all variants match the CPU reference" \
            "$RUN/2026-11-04_kcol_dense/run/rig_kcol.sh" \
            $PTX $FORCE $NSM5 BIN="$BIN" MMVQ="$KCOL_BENCH2" MMVQ_PASCAL="$MMVQ_PASCAL" OUT="$OUT/kcol" CARD="$DEV" PARTS="check bench" BENCH_ARGS="$BA"
        # shellcheck disable=SC2086
        sub kcolcov "cov cov_all cov_r1 cov_r2 cov_itt1 cov_off q2f q2f_r2 all passed; the --q2_0 line ends in PASS" \
            "$RUN/2026-11-04_kcol_coverage/run/rig_kcol_cov.sh" \
            $PTX $FORCE $NSM5 BIN="$BIN" MMVQ_PASCAL="$MMVQ_PASCAL" OUT="$OUT/kcolcov" CARD="$DEV" PARTS="check q2_0" Q2_BENCH_ARGS=--check-only
    else
        skip kcol "KCOL_BENCH2 and MMVQ_PASCAL not given"
    fi
fi

if part iq2xs; then
    sub iq2xs "every run N/N passed, 0 failure lines; memcheck (ptx) 15 of 15 cases with 0 errors" \
        "$RUN/2026-11-04_iq2xs_kernel/run/gpu_checks.sh" MODE="$MODE" BIN="$BIN" OUT="$OUT/iq2xs" GPU="$DEV" CUDA_VISIBLE_DEVICES="$DEV"
fi

if part iqkcol; then
    sub iqkcol "every run N/N passed, 0 failure lines; memcheck (ptx) 30 of 30 cases with 0 errors" \
        "$RUN/2026-11-04_iq_kcol/run/gpu_checks.sh" MODE="$MODE" BIN="$BIN" OUT="$OUT/iqkcol" GPU="$DEV" CUDA_VISIBLE_DEVICES="$DEV"
    if [ -n "${KCOL_BENCH2:-}" ]; then
        run iqkcol/bench "mmvq_kcol --check-only, the IQ types" "correctness: all variants match the CPU reference" "$DEV" "" \
            "$KCOL_BENCH2" --check-only $([ "$MODE" = ptx ] && echo "--nsm 5") --types iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq1_m
    else
        skip iqkcol/bench "KCOL_BENCH2 not given"
    fi
fi

if part qsa; then
    if [ -n "${FASP:-}" ]; then
        # shellcheck disable=SC2046,SC2086
        sub qsa "sparse1 sparse2 sparse0 all pass; fault1-3 have FAIL lines; fasp_gpu 20 of 20" \
            "$RUN/2026-11-04_qsa_sparse_attn/run/qsa_gpu.sh" \
            $PTX BIN="$BIN" FASP="$FASP" OUT="$OUT/qsa" PARTS="check fault fasp" CUDA_VISIBLE_DEVICES="$DEV" $([ "$MODE" = ptx ] && echo NSM=5)
    else
        skip qsa "FASP not given"
    fi
fi

if part fap; then
    FA="$T -o FLASH_ATTN_EXT"
    OWN='hsk=128,hsv=128,nh=(4|8),nr23=\[8,1\],kv=[0-9]+,nb=1,'
    ON=$([ "$MODE" = ptx ] && echo 2 || echo 1)
    run fap/on     "GGML_CUDA_FATTN_PASCAL=$ON, every FA case" "every case passes" "$DEV" "GGML_CUDA_FATTN_PASCAL=$ON" $FA
    run fap/fault1 "FAP fault 1 (output x 1.01)"   "passes (inside the NMSE limit)" "$DEV" "GGML_CUDA_FATTN_PASCAL=$ON GGML_CUDA_FATTN_PASCAL_FAULT=1" $FA -p "$OWN"
    run fap/fault2 "FAP fault 2 (sinks dropped)"   "FAIL lines present"             "$DEV" "GGML_CUDA_FATTN_PASCAL=$ON GGML_CUDA_FATTN_PASCAL_FAULT=2" $FA -p "$OWN"
    run fap/fault3 "FAP fault 3 (last tile dropped)" "FAIL lines present"           "$DEV" "GGML_CUDA_FATTN_PASCAL=$ON GGML_CUDA_FATTN_PASCAL_FAULT=3" $FA -p "$OWN"
    run fap/off    "GGML_CUDA_FATTN_PASCAL=0, FAP's cases" "every case passes"      "$DEV" "GGML_CUDA_FATTN_PASCAL=0" $FA -p "$OWN"
fi

if part hops; then
    run hops/ops  "CPY, the 18 hop-shape cases among them" "every case passes" "$DEV" "" $T -o CPY
    run hops/unit "test-hop-cast --unit-only (CPU and card)" "exit 0" "$DEV" "" "$BIN/test-hop-cast" --unit-only
    q=$(model qwen4exp-moe)
    if [ -n "$DEVS2" ] && [ "${DEVS2#*,}" != "$DEVS2" ] && [ -n "$q" ]; then
        # layer 0 on the first card, layer 1 on the second: one boundary, before layer 1 (C200g's model step)
        run hops/model "staged hop and casts across two cards" "exit 0 (staged arms bit for bit, casts within bound)" "$DEVS2" "" \
            "$BIN/test-hop-cast" -m "$q" -ngl 99 -ts 1,3 --tsv "$OUT/hops-model.tsv" \
            --arm off \
            --arm "staged exact GGML_CUDA_HOP_STAGED=2" \
            --arm "staged-c1 exact GGML_CUDA_HOP_STAGED=2 GGML_CUDA_HOP_CHUNKS=1" \
            --arm "staged-c4 exact GGML_CUDA_HOP_STAGED=2 GGML_CUDA_HOP_CHUNKS=4" \
            --arm "staged-4k exact GGML_CUDA_HOP_STAGED=2 GGML_CUDA_HOP_SLOT_KB=4 GGML_CUDA_HOP_SLOTS=2" \
            --arm "staged-s2 exact GGML_CUDA_HOP_STAGED=2 GGML_CUDA_HOP_STREAM=2" \
            --arm "staged-s0-4k exact GGML_CUDA_HOP_STAGED=2 GGML_CUDA_HOP_STREAM=0 GGML_CUDA_HOP_SLOT_KB=4 GGML_CUDA_HOP_SLOTS=2" \
            --arm "f16 LLAMA_HOP_CAST=f16" --arm "bf16 LLAMA_HOP_CAST=bf16" --arm "q8_0 LLAMA_HOP_CAST=q8_0" \
            --arm "q8_0-staged LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=2"
        say "hops/model detail: $(grep -c 'card hops in [a-z0-9_]* before layers 1 (the layer split)' "$OUT/hops-model.log") contexts cast before layer 1, $(grep -c 'staged through' "$OUT/hops-model.log") staged rings"
        run hops/archs-2cards "test-llama-archs on two cards, defaults" "Summary: all passed" "$DEVS2" "" "$BIN/test-llama-archs" -s 1
        run hops/archs-staged "test-llama-archs, GGML_CUDA_HOP_STAGED=2" "the same verdicts as ladder2/archs" "$DEVS2" "GGML_CUDA_HOP_STAGED=2" \
            "$BIN/test-llama-archs" -s 1
    else
        skip hops/model "needs two cards (DEVS2) and a qwen4exp model in MODELS"
    fi
fi

have_http() { command -v curl > /dev/null && command -v jq > /dev/null; }

if part vgraphs; then
    q=$(model qwen4exp-moe)
    if [ -n "$q" ] && have_http; then
        # shellcheck disable=SC2086
        sub vgraphs "verdict PASS: every arm base's tokens and top probabilities, two or more verify widths" \
            "$RUN/2026-11-04_vgraphs/run/spec_check.sh" \
            $PTX CUDA_VISIBLE_DEVICES="$DEV" BIN="$BIN" MODEL="$q" OUT="$OUT/vgraphs" THREADS="$THREADS" SARGS="-ngl 99 -ngld 99"
        say "vgraphs/ verdict: $(tr '\n' ' ' < "$OUT/vgraphs/verdict.txt" 2>/dev/null)"
    else
        skip vgraphs/spec "needs a qwen4exp model in MODELS, curl and jq"
    fi
    for m in qwen4exp-moe qwen35-dense; do
        f=$(model $m)
        [ -n "$f" ] || { skip "vgraphs/ctx-$m" "no $m model"; continue; }
        for s in default LLAMA_GRAPH_PER_SIZE=0 LLAMA_SCHED_PER_SIZE=1 LLAMA_GRAPH_META_CALLOC=0; do
            e=""; [ $s = default ] || e=$s
            run "vgraphs/ctx-$m-${s##*_}" "test-llama-context -ngl 99, $s" "failed : 0" "$DEV" "$e" "$BIN/test-llama-context" -m "$f" -ngl 99
        done
    done
fi

if part ckpt; then
    q=$(model qwen4exp-moe)
    if [ -n "$q" ] && have_http; then
        for a in plain: share:--ctx-checkpoints-share; do
            # shellcheck disable=SC2086
            sub "ckpt-${a%%:*}" "verdict PASS (vram, vram2, vram2s = base; spread_v2 = spread)" \
                "$RUN/2026-11-04_ckpt_cards/run/server_check.sh" \
                $PTX CUDA_VISIBLE_DEVICES="$DEV" BIN="$BIN" MODEL="$q" OUT="$OUT/ckpt-${a%%:*}/server" THREADS="$THREADS" \
                SARGS="--override-kv qwen4exp.context_length=int:16384 -ngl 99 ${a#*:}"
            say "ckpt-${a%%:*}/ verdict: $(tr '\n' ' ' < "$OUT/ckpt-${a%%:*}/server/verdict.txt" 2>/dev/null)"
        done
    else
        skip ckpt "needs a qwen4exp model in MODELS, curl and jq"
    fi
fi

if part server; then
    q=$(model qwen4exp-moe)
    if [ -n "$q" ] && have_http; then
        S=$OUT/server
        mkdir -p "$S"
        PORT=${PORT:-18095}
        toks() { awk -v n="$1" -v s="$2" -v v="$3" 'BEGIN { x = s; for (i = 0; i < n; i++) { x = (x * 16807) % 2147483647; printf "%s%d", (i ? "," : ""), 1 + x % (v - 1) } }'; }
        declare -A ENVS=(
            [off]="" [dump]="LLAMA_SPEC_DUMP=@/dump.jsonl" [adaptlog]="LLAMA_SPEC_ADAPT_LOG=@/adapt.jsonl"
            [adapt]="LLAMA_SPEC_ADAPT=1 LLAMA_SPEC_ADAPT_LOG=@/adapt.jsonl"
            [adaptdump]="LLAMA_SPEC_ADAPT=1 LLAMA_SPEC_ADAPT_LOG=@/adapt.jsonl LLAMA_SPEC_DUMP=@/dump.jsonl"
            [moep1]="LLAMA_MOE_TAIL=p=1"
            [profgroup]="GGML_CUDA_PROFILE=@/prof.csv GGML_CUDA_PROFILE_MODE=group"
            [proflayer]="GGML_CUDA_PROFILE=@/prof.csv GGML_CUDA_PROFILE_MODE=layer"
            [profgraph]="GGML_CUDA_PROFILE=@/prof.csv GGML_CUDA_PROFILE_MODE=graph"
            [profsample]="GGML_CUDA_PROFILE=@/prof.csv GGML_CUDA_PROFILE_MODE=sample=8"
            [share]="" [taper]="LLAMA_UBATCH_TAPER=on"
        )
        ARMS="off dump adaptlog adapt adaptdump moep1 profgroup proflayer profgraph profsample share taper"
        for arm in $ARMS; do
            d=$S/$arm; mkdir -p "$d"
            xa=""; [ $arm = share ] && xa="--ctx-checkpoints-share"
            # shellcheck disable=SC2086
            env CUDA_VISIBLE_DEVICES="$DEV" $PTX ${ENVS[$arm]//@/$d} "$BIN/llama-server" -m "$q" -md "$q" --spec-type draft-simple \
                --spec-draft-n-max 3 -ngl 99 -ngld 99 -t "$THREADS" -c 4096 -np 1 --cache-ram 0 --no-webui -lv 4 \
                --override-kv qwen4exp.context_length=int:16384 --host 127.0.0.1 --port "$PORT" $xa > "$d/server.log" 2>&1 &
            pid=$!
            for _ in $(seq 1 900); do curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break; sleep 1; done
            nv=$(curl -s "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].meta.n_vocab')
            for r in 1 2 3; do
                printf '{"prompt":[%s],"n_predict":64,"temperature":0,"top_k":1,"seed":42,"return_tokens":true,"cache_prompt":false,"speculative.n_max":3,"speculative.n_min":0}' \
                    "$(toks $((30 + 20 * r)) $((7 * r)) "${nv:-256}")" > "$d/R$r.req.json"
                curl -s --max-time 600 -H 'Content-Type: application/json' --data @"$d/R$r.req.json" "http://127.0.0.1:$PORT/completion" > "$d/R$r.json"
            done
            kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
            files=""; for f in dump.jsonl adapt.jsonl prof.csv; do [ -f "$d/$f" ] && files="$files $f:$(wc -l < "$d/$f")"; done
            [ $arm = taper ] && files="$files taper-lines:$(grep -c "ubatch taper:" "$d/server.log")"
            same=""
            if [ $arm != off ]; then
                same=same
                for r in 1 2 3; do cmp -s <(jq -c '.tokens' "$d/R$r.json") <(jq -c '.tokens' "$S/off/R$r.json") || same=DIFFER; done
            fi
            n=$(for r in 1 2 3; do jq -r '.tokens | length' "$d/R$r.json" 2>/dev/null; done | tr '\n' '/')
            dr=$(for r in 1 2 3; do jq -r '.timings.draft_n // 0' "$d/R$r.json" 2>/dev/null; done | tr '\n' '/')
            req="tokens as off"; [ $arm = off ] && req="64 tokens a request, drafts > 0"
            case $arm in adapt|adaptdump) req="reported only (other verify widths)";; esac
            [ $arm = taper ] && req="reported only (64-token micro-batches may round otherwise on the card); taper-lines > 0"
            case $arm in dump) req="$req; dump.jsonl lines > 0";; adaptdump) req="$req; dump.jsonl lines = the steps verified (0 when the rule cuts every draft)";; adaptlog|adapt) req="$req; adapt.jsonl lines > 0";; prof*) req="$req; prof.csv lines > 0";; esac
            say "$(printf '%-24s %-36s tokens %s drafts %s %s files:%s errors %s' "server/$arm" "${ENVS[$arm]//@\//}$xa" "$n" "$dr" "${same:-reference}" "${files:- none}" \
                "$(sed 's/\x1b\[[0-9;]*m//g' "$d/server.log" | grep -E ' E |[Ee]rror|[Aa]bort' | grep -vc set_abort_callback)")  passing: $req"
        done
    else
        skip server "needs a qwen4exp model in MODELS, curl and jq"
    fi
fi

date -Is > "$OUT/done"
say "== $(date -Is) done"
