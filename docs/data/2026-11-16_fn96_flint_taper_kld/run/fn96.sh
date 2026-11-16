#!/usr/bin/env bash
# FN96 (Q1494, 2026-11-16): does the micro-batch taper change Flint's numbers? FN95's run
# (docs/data/2026-11-16_fn95_flashnext_taper_kld/run/fn95.sh) with Flint's v4 line and the tuned preset f4t's two
# variables. llama-perplexity's KL divergence on a P106 rig: Flint's micro-batch of 128 (one call of CTX tokens read
# in micro-batches of 128) writes the base, the same text with the taper (LLAMA_UBATCH_TAPER=short=600:64: each call
# of CTX <= 600 tokens read in micro-batches of 64) is scored against it, and a repeat of the base's settings gives
# the run-to-run floor. Run from the repository root on a machine that only runs ssh, with nothing else on the rig.
#
# Env: RIG (ssh alias), WORK (a rig folder with room for the base: CHUNKS x (CTX/2 - 1) x n_vocab x 2 bytes),
#   BIN (llama-perplexity on the rig), MODEL (Flint Pascal Q8 GGUF on the rig), TEXT_LOCAL (here: the agent check's
#   proxy.jsonl.bodies.gz, its proxy.jsonl beside it), OUT (here), STEP:
#     dry      print each arm's command, touch nothing
#     start    copy the text and scripts to WORK and start run/driver.sh detached (returns at once)
#     wait     return when the driver has written DONE (checks once a minute)
#     collect  copy the logs to OUT, write OUT/kld.tsv, remove WORK, record the rig's state, then the tables step
#     tables   from the logs in OUT only (no ssh): ppl_by_chunk.tsv, checks.txt, <arm>_stats.txt, verdict.tsv
#   optional TAG (the proxy tags to read, default ^t1swift/: FN93c's and FN95's text), CTX (512), CHUNKS (24),
#   TAPER (short=600:64), ARMS (base taper rep), PRESET (default release/flint-v4), ADD_ENV (variables added to the
#   preset's environment, default f4t's: GGML_CUDA_GRAPHS_FORCE=1 LLAMA_SCHED_PER_SIZE_RESERVE=2), KLD_MAX (the
#   rule's limit on the taper's mean KLD, default 0.0066), PPL_PCT (the rule's limit on the perplexity change, 0.1)
# The line is PRESET/settings.env's: PRESET_ENV as it is plus ADD_ENV, and of PRESET_ARGS the flags llama-perplexity
# takes and that act on a model's load or its arithmetic (-ngl, -fa, -ub, -ts, -fit, -lm, --no-host), with -b CTX so
# that each chunk is one call of one sequence: with -b 2048 llama-perplexity reads four chunks in one call as four
# sequences, and the taper acts only on one-sequence calls.
set -euo pipefail
: "${STEP:?}"
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
PRESET=${PRESET:-release/flint-v4}
. "$repo/$PRESET/settings.env"
ADD_ENV=${ADD_ENV-GGML_CUDA_GRAPHS_FORCE=1 LLAMA_SCHED_PER_SIZE_RESERVE=2}
PRESET_ENV="$PRESET_ENV${ADD_ENV:+ $ADD_ENV}"
TAG=${TAG:-^t1swift/}; CTX=${CTX:-512}; CHUNKS=${CHUNKS:-24}; TAPER=${TAPER:-short=600:64}; ARMS=${ARMS:-base taper rep}
BIN=${BIN:-}; MODEL=${MODEL:-}; KLD_MAX=${KLD_MAX:-0.0066}; PPL_PCT=${PPL_PCT:-0.1}

# the preset's flags that llama-perplexity takes; a flag's value is the next word unless that starts with '-'
ARGS=""
set -- $PRESET_ARGS
while [ $# -gt 0 ]; do
    f=$1; v=""
    if [ $# -gt 1 ] && [ "${2#-}" = "$2" ]; then v=$2; shift; fi
    shift
    case $f in -ngl|-fa|-ub|-ts|-fit|-lm|--no-host) ARGS="$ARGS $f${v:+ $v}" ;; esac
done
ARGS="${ARGS# } -b $CTX"

tables() {
    : "${OUT:?}"
    # each arm's running perplexity after each chunk: base.out prints [n]value, the scored arms one row a chunk
    awk -v arms="$ARMS" -v out="$OUT" 'BEGIN {
        n = split(arms, a, " ")
        for (i = 1; i <= n; i++) {
            f = out "/" a[i] ".out"
            while ((getline l < f) > 0) {
                if (a[i] == "base") {
                    while (match(l, /\[[0-9]+\][0-9.]+/)) {
                        s = substr(l, RSTART + 1, RLENGTH - 1); split(s, p, "]"); v[a[i], p[1]] = p[2]
                        if (p[1] > m) m = p[1]; l = substr(l, RSTART + RLENGTH)
                    }
                } else if (l ~ /^ *[0-9]+ +[0-9.]+ +±/) { split(l, p, " "); v[a[i], p[1]] = p[2]; if (p[1] > m) m = p[1] }
            }
        }
        printf "chunk"; for (i = 1; i <= n; i++) printf "\t%s", a[i]
        for (i = 1; i <= n; i++) if (a[i] != "base") printf "\t%s against base %%", a[i]
        printf "\n"
        for (c = 1; c <= m; c++) {
            printf "%d", c; for (i = 1; i <= n; i++) printf "\t%s", v[a[i], c]
            for (i = 1; i <= n; i++) if (a[i] != "base") printf "\t%.2f", 100 * (v[a[i], c] / v["base", c] - 1)
            printf "\n"
        }
    }' > "$OUT/ppl_by_chunk.tsv"
    # what each load logged about its line
    {
        echo "# FN96: what each llama-perplexity load logged about its line (counts of identical lines; -lv 4)"
        for arm in $ARMS; do
            echo "## $arm"
            grep -a -h -o -E 'calculating perplexity over [0-9]+ chunks.*|computing over [0-9]+ chunks.*|Final estimate: PPL = .*|LLAMA_SCHED_PER_SIZE_RESERVE.*|load time = +[0-9.]+ ms|n_batch += [0-9]+|n_ubatch += [0-9]+|n_vocab += [0-9]+|set_ubatch_taper: .*|ubatch taper: .*|switches: .*' \
                "$OUT/$arm.err" | sed -E 's/ +/ /g; s/ - ETA.*//' | sort | uniq -c | sed -E 's/^ +//'
        done
    } > "$OUT/checks.txt"
    for arm in $ARMS; do
        [ "$arm" = base ] && continue
        sed -n '/====== Perplexity statistics/,/^Same top p/p' "$OUT/$arm.out" > "$OUT/${arm}_stats.txt"
    done
    # the verdict by the rule: each scored arm against the base's own perplexity, with the paired error of the
    # tool's log ratio (the same error holds against the base's own value: the stored base only shifts the mean)
    # base_ppl is printed to 4 decimals; the repeat's PPL(Q) has 6 and equals it when the repeat is exact
    base_ppl=$(grep -a -h -o 'Final estimate: PPL = [0-9.]*' "$OUT/base.out" "$OUT/base.err" | tail -n 1 | sed 's/.*= //')
    rep_ppl=$(grep -a -s '^Mean PPL(Q) ' "$OUT/rep_stats.txt" | awk '{print $4}')
    {
        printf 'arm\tmean KLD\terror\t99.0%% KLD\tsame top p %%\tPPL\tPPL(base own)\tchange %%\tpaired error %%\twithin 2 errors\tchunks equal to base\tmax |dp| %%\tPPL(rep)\tchange against rep %%\n'
        for arm in $ARMS; do
            [ "$arm" = base ] && continue
            s="$OUT/${arm}_stats.txt"
            kl=$(grep -a '^Mean    KLD:' "$s" | awk '{print $3}'); ke=$(grep -a '^Mean    KLD:' "$s" | awk '{print $5}')
            k99=$(grep -a '^99.0%   KLD:' "$s" | awk '{print $3}'); top=$(grep -a '^Same top p:' "$s" | awk '{print $4}')
            ppl=$(grep -a '^Mean PPL(Q) ' "$s" | awk '{print $4}'); le=$(grep -a '^Mean ln(PPL(Q)/PPL(base))' "$s" | awk '{print $6}')
            dmax=$(grep -a -E '^(Maximum|Minimum) Δp:' "$s" | sed 's/%//' | awk '{x = $3 < 0 ? -$3 : $3; if (x > m) m = x} END {print m + 0}')
            eq=$(awk -F'\t' -v arm="$arm" 'NR == 1 {for (i = 1; i <= NF; i++) if ($i == arm) c = i; next} {n++; if ($c == $2) e++}
                END {printf "%d of %d", e, n}' "$OUT/ppl_by_chunk.tsv")
            awk -v a="$arm" -v kl="$kl" -v ke="$ke" -v k99="$k99" -v top="$top" -v p="$ppl" -v b="$base_ppl" -v le="$le" \
                -v eq="$eq" -v dmax="$dmax" -v r="${rep_ppl:-}" 'BEGIN {
                ch = 100 * (p / b - 1); pe = 100 * le; w = (ch <= 2 * pe && ch >= -2 * pe) ? "yes" : "no"
                cr = (r != "") ? sprintf("%+.3f", 100 * (p / r - 1)) : "-"
                printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%+.3f\t%.3f\t%s\t%s\t%s\t%s\t%s\n", a, kl, ke, k99, top, p, b, ch, pe, w, eq, dmax, (r != "" ? r : "-"), cr }'
        done
    } > "$OUT/verdict.tsv"
    awk -F'\t' -v kmax="$KLD_MAX" -v pmax="$PPL_PCT" 'NR == 1 {next}
        $1 == "taper" {k = ($2 <= kmax) ? "met" : "not met"; q = ($8 <= pmax && $8 >= -pmax) ? "met" : "not met"
            printf "taper: mean KLD %s <= %s: %s; perplexity %s%% (%s%% against the repeat) within +/-%s%%: %s; within two paired errors (+/-%.3f%%): %s\n", $2, kmax, k, $8, $14, pmax, q, 2 * $9, $10}
        $1 ~ /^rep/ {split($11, e, " "); x = (e[1] == e[3] && $5 + 0 == 100 && $2 + 0 < 0.0001) ? "exact" : "not exact"
            printf "%s: %s chunks equal to the base, same top %s%%, mean KLD %s, max |dp| %s%%: %s\n", $1, $11, $5, $2, $12, x}' \
        "$OUT/verdict.tsv" > "$OUT/verdict.txt"
    cat "$OUT/verdict.tsv" "$OUT/verdict.txt"
}

case $STEP in
dry)
    for arm in $ARMS; do
        case $arm in base) k="--kl-divergence-base $WORK/base.kld"; e="" ;;
                     taper) k="--kl-divergence-base $WORK/base.kld --kl-divergence"; e="LLAMA_UBATCH_TAPER=$TAPER" ;;
                     *) k="--kl-divergence-base $WORK/base.kld --kl-divergence"; e="" ;; esac
        echo "$arm: env $PRESET_ENV $e $BIN -m $MODEL $ARGS -f $WORK/text.txt -c $CTX --chunks $CHUNKS -lv 4 $k"
    done ;;
start)
    : "${RIG:?}" "${WORK:?}" "${BIN:?}" "${MODEL:?}" "${TEXT_LOCAL:?}"
    ssh "$RIG" "test ! -e $WORK && mkdir -p $WORK/text"
    ssh "$RIG" "cat > $WORK/text/proxy.jsonl.bodies.gz" < "$TEXT_LOCAL"
    ssh "$RIG" "cat > $WORK/text/proxy.jsonl" < "${TEXT_LOCAL%.bodies.gz}"
    ssh "$RIG" "cat > $WORK/agent_text.py" < "$here/agent_text.py"
    ssh "$RIG" "cat > $WORK/driver.sh" < "$here/driver.sh"
    ssh "$RIG" "cat > $WORK/ram_watchdog.sh" < "$repo/scripts/ram_watchdog.sh"
    for v in WORK BIN MODEL TAG CTX CHUNKS ARGS PRESET_ENV TAPER ARMS; do printf '%s=%q\n' "$v" "${!v}"; done \
        | ssh "$RIG" "cat > $WORK/run.env"
    timeout 60 ssh "$RIG" "setsid nohup bash $WORK/driver.sh > $WORK/driver.console 2>&1 < /dev/null & echo started" ;;
wait)
    : "${RIG:?}" "${WORK:?}"
    until ssh -o ConnectTimeout=20 "$RIG" "test -e $WORK/DONE" 2> /dev/null; do sleep 60; done
    ssh "$RIG" "cat $WORK/driver.log" ;;
collect)
    : "${RIG:?}" "${WORK:?}" "${OUT:?}"
    mkdir -p "$OUT"
    ssh "$RIG" "cd $WORK && ls -la . text && md5sum text.txt" > "$OUT/work_listing.txt"
    for f in driver.log driver.console text.err run.env; do ssh "$RIG" "cat $WORK/$f" > "$OUT/$f" || true; done
    for arm in $ARMS; do
        for f in "$arm.out" "$arm.err" "rw-$arm.tsv"; do ssh "$RIG" "cat $WORK/$f" > "$OUT/$f" || true; done
    done
    {
        printf 'arm\tagainst\tchunks\tmean KLD\tmedian KLD\t99.0%% KLD\tmaximum KLD\tsame top p %%\tPPL(Q)\tPPL(base)\tPPL(Q)/PPL(base)\tln(PPL(Q)/PPL(base))\ttaper lines\n'
        for arm in $ARMS; do
            o="$OUT/$arm.out"; e="$OUT/$arm.err"
            g() { grep -a -h -m1 "$1" "$o" "$e" 2> /dev/null | head -n 1 | cut -d: -f2- | xargs; }
            n=$(grep -a -c 'ubatch taper:' "$e" 2> /dev/null || true)
            if [ "$arm" = base ]; then
                ppl=$(grep -a -h -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' "$o" "$e" | sed 's/.*PPL = //' | tail -n 1)
                ch=$(grep -a -h -o 'calculating perplexity over [0-9]* chunks' "$e" "$o" | grep -o '[0-9]*' | head -n 1)
                printf '%s\t-\t%s\t-\t-\t-\t-\t-\t%s\t-\t-\t-\t%s\n' "$arm" "$ch" "$ppl" "$n"
            else
                ch=$(grep -a -h -o 'computing over [0-9]* chunks' "$e" "$o" | grep -o '[0-9]*' | head -n 1)
                printf '%s\tbase\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$ch" "$(g '^Mean    KLD:')" \
                    "$(g '^Median  KLD:')" "$(g '^99.0%   KLD:')" "$(g '^Maximum KLD:')" "$(g '^Same top p:')" \
                    "$(g '^Mean PPL(Q) ')" "$(g '^Mean PPL(base) ')" "$(g '^Mean PPL(Q)/PPL(base) ')" \
                    "$(g '^Mean ln(PPL(Q)/PPL(base)) ')" "$n"
            fi
        done
    } > "$OUT/kld.tsv"
    grep -a -h -o 'ubatch taper: .*' "$OUT"/*.err 2> /dev/null | sed 's/ - ETA.*//' | sort | uniq -c > "$OUT/taper_plans.txt" || true
    ssh "$RIG" "rm -rf $WORK"
    ssh "$RIG" "date -Is; ps -eo comm | grep -c -x -E 'llama-perplexity|llama-server' || true;
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; df -h $(dirname "$WORK") | tail -n 1" \
        > "$OUT/rig_end.txt"
    cat "$OUT/kld.tsv" "$OUT/taper_plans.txt" "$OUT/rig_end.txt"
    tables ;;
tables) tables ;;
*) echo "STEP: dry, start, wait, collect or tables" >&2; exit 2 ;;
esac
