#!/usr/bin/env bash
# C155 (Q706): what llama-server's memory fit (-fit on) projects with and without a draft model, before and after
# upstream #31610 (the draft measured with a context of the main model, so drafts that need the target's context are
# counted) and with a draft that reads the target's embeddings and head (LLAMA_DRAFT_SHARE). On a machine without a
# GPU the fit has no device to split over: it projects the host memory of the main model plus the draft
# ("projected to use N MiB of host memory"), which is the number compared here. Only the files' metadata is read
# (the measurement loads no weights); each server is stopped once its fit is done.
# Cases, each "label|binary|env|server arguments" (CASES overrides; BIN, OLD_BIN, MAIN, DRAFT, RMODEL fill them in):
#   r_main        the random qwen4exp alone
#   r_draft0      + the same file as a separate draft (draft-simple), LLAMA_FIT_DRAFT_CTX=0 (measured alone, as before)
#   r_draft       + the draft, measured with the main model's context
#   r_share       + the draft, LLAMA_DRAFT_SHARE=all: measured without the embeddings and head it reads from the target
#   r_share0      as r_share with LLAMA_FIT_DRAFT_CTX=0 (no main model to read from: as r_draft0)
#   s_main        MAIN alone (e.g. Flash-Next Q2_0), 64k context
#   s_dflash0     + DRAFT (e.g. the DFlash2 drafter, --spec-type draft-dflash), LLAMA_FIT_DRAFT_CTX=0
#   s_dflash      + the DFlash draft, measured with the main model's context
#   s_dflash_old  as s_dflash on OLD_BIN (a build without the change)
#   s_mtp0        + MTP (an MTP draft file, --spec-type draft-mtp), LLAMA_FIT_DRAFT_CTX=0
#   s_mtp         + the MTP draft, measured with the main model's context
#   s_mtp_share   + the MTP draft, LLAMA_DRAFT_SHARE=all (it reads the target's embeddings and head)
#   s_mtp_old     as s_mtp on OLD_BIN
# Usage: BIN=<dir with llama-server> RMODEL=<random qwen4exp gguf> [MAIN=<gguf> [DRAFT=<DFlash gguf>] [MTP=<MTP gguf>]]
#   [OLD_BIN=<dir>]
#   OUT=<dir> [THREADS=8] [PORT=18095] bash fit_check.sh
# Output: OUT/<label>.log (the fit's lines), OUT/fit.tsv (label, projected host MiB, the extra model measured or not)
set -uo pipefail
BIN=${BIN:?BIN=<dir with llama-server>}
OUT=${OUT:?OUT=<dir>}
RMODEL=${RMODEL:-}
MAIN=${MAIN:-}
DRAFT=${DRAFT:-}
MTP=${MTP:-}
OLD_BIN=${OLD_BIN:-}
THREADS=${THREADS:-8}
PORT=${PORT:-18095}
mkdir -p "$OUT"

if [ -z "${CASES:-}" ]; then
    CASES=""
    if [ -n "$RMODEL" ]; then
        R="-m $RMODEL -c 4096 --override-kv qwen4exp.context_length=int:16384"
        RD="$R -md $RMODEL --spec-type draft-simple"
        CASES+="r_main|$BIN||$R;"
        CASES+="r_draft0|$BIN|LLAMA_FIT_DRAFT_CTX=0|$RD;"
        CASES+="r_draft|$BIN||$RD;"
        CASES+="r_share|$BIN|LLAMA_DRAFT_SHARE=all|$RD;"
        CASES+="r_share0|$BIN|LLAMA_DRAFT_SHARE=all LLAMA_FIT_DRAFT_CTX=0|$RD;"
    fi
    if [ -n "$MAIN" ]; then
        S="-m $MAIN -c 65536 -b 2048 -ub 512"
        CASES+="s_main|$BIN||$S;"
        if [ -n "$DRAFT" ]; then
            SD="$S -md $DRAFT --spec-type draft-dflash --spec-draft-n-max 3 --spec-draft-n-ubatch 64"
            CASES+="s_dflash0|$BIN|LLAMA_FIT_DRAFT_CTX=0|$SD;"
            CASES+="s_dflash|$BIN||$SD;"
            [ -n "$OLD_BIN" ] && CASES+="s_dflash_old|$OLD_BIN||$SD;"
        fi
        if [ -n "$MTP" ]; then
            SM="$S -md $MTP --spec-type draft-mtp --spec-draft-n-max 3"
            CASES+="s_mtp0|$BIN|LLAMA_FIT_DRAFT_CTX=0|$SM;"
            CASES+="s_mtp|$BIN||$SM;"
            CASES+="s_mtp_share|$BIN|LLAMA_DRAFT_SHARE=all|$SM;"
            [ -n "$OLD_BIN" ] && CASES+="s_mtp_old|$OLD_BIN||$SM;"
        fi
    fi
fi

printf 'case\tprojected_host_MiB\textra_model\n' > "$OUT/fit.tsv"
IFS=';' read -ra CFG <<< "$CASES"
for cfg in "${CFG[@]}"; do
    [ -n "$cfg" ] || continue
    IFS='|' read -r label bin envs args <<< "$cfg"
    log="$OUT/$label.full.log"
    env $envs "$bin/llama-server" $args -t "$THREADS" -np 1 -fit on -lv 5 --ctx-checkpoints 4 --cache-ram 0 \
        --no-webui --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 < /dev/null &
    pid=$!
    for _ in $(seq 1 300); do
        grep -qE 'no changes needed|successfully fit|failed to fit|encountered an error|abort' "$log" && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    sleep 1; kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    grep -E 'projected|will leave|no changes|failed to measure|extra model|memory breakdown|\| +- |successfully|failed to fit|requires ctx_other|reading from the source|does not read shared' "$log" \
        | grep -v 'unused tensor' | cut -c1-240 > "$OUT/$label.log"
    rm -f "$log"
    proj=$(grep -m1 -oE 'projected to use [0-9]+ MiB of host memory' "$OUT/$label.log" | grep -oE '[0-9]+')
    if grep -q 'failed to measure the memory of the extra model' "$OUT/$label.log"; then
        extra="failed: $(grep -m1 -o 'fitting without it: .*' "$OUT/$label.log" | cut -c21-140)"
    elif grep -q 'extra model' "$OUT/$label.log"; then
        extra="measured"
    else
        extra="none"
    fi
    printf '%s\t%s\t%s\n' "$label" "${proj:-?}" "$extra" | tee -a "$OUT/fit.tsv"
done
