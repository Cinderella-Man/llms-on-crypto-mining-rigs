# The stat stage of cpu_check.sh, inside its container (needs ST, SD, SD2: the target and drafters with GPT-2's
# tokenizer; PORT, N_STAT, N_STAT_G): sampled decoding with N seeds per arm, one TSV per arm in /c/stat (seed, the
# output, drafted, accepted). The arms run at once, each its own server on its own port with STAT_THREADS threads
# (default 2): a request is mostly single-threaded work (sampling, the grammar), so the arms in parallel use the
# machine's cores.
#   plain-*  the first five tokens after a text prompt, temperature 1, top-k 8 (top-p 1, min-p 0): the first comes from
#            the prompt pass, the next from drafts of three (n_predict - 2); arms nospec (plain decoding), rej
#            (token-by-token rejection) and block (block verification) with the drafter SD (accepts few: the residual
#            path), rej2 and block2 with the drafter SD2 (closer to the target: drafts accepted in part, the weights
#            between 0 and 1) and rejkeep2 (rej2 with LLAMA_SPEC_KEEP_DRAWS=1), and t08 (plain decoding at
#            temperature 0.8: a different distribution, the control the test must tell apart)
#   gram-*   a short JSON tool call under a grammar (stat_tool.gbnf: 30 outcomes), temperature 1, top-k 8; arms
#            nospec, rej, block, force (forced tokens, block verification), keep0 (forced tokens only: drafts cut
#            to nothing, verified by block verification with point-mass proposals) and rejkeep (rej with
#            LLAMA_SPEC_KEEP_DRAWS=1). Draft-simple on this hybrid target replays from a checkpoint after every
#            partial acceptance (no rollback snapshots), the case LLAMA_SPEC_KEEP_DRAWS is for
# Every arm draws its own seeds (arm k: k * 1,000,000 + 1 .. + N), so the arms' samples are independent: with shared
# seeds the first tokens, drawn before any draft, would be the same in every arm. STAT_ARMS (default: all) runs a
# subset, e.g. STAT_ARMS="plain-rej2 plain-block2"; an arm overwrites only its own table.
set -u
OUT=/c/stat
mkdir -p "$OUT"
T=$ST
DR=$SD
. /r/serve_lib.sh
DRAFT2=${DRAFT/$SD/$SD2}
want() { [ -z "${STAT_ARMS:-}" ] || [[ " $STAT_ARMS " == *" $1 "* ]]; }
THREADS=${STAT_THREADS:-2}
PORT0=$PORT

P_TEXT="The quick brown fox"
P_TOOL="Call one tool as JSON:"

plain_arm() { # <arm> <seed offset> <port> <temperature> <bin> <server args> [env...]
    local arm=$1 s0=$2 PORT=$3 temp=$4; shift 4
    want "plain-$arm" || return 0
    serve "plain-$arm" "$@" || return 1
    local t0=$SECONDS
    for s in $(seq 1 "$N_STAT"); do
        complete "$OUT/r-plain-$arm.json" "$P_TEXT" 5 "$temp" $((s0 + s))
        jq -r --argjson s "$s" '[$s, (.tokens | map(tostring) | join(",")), (.timings.draft_n // 0), (.timings.draft_n_accepted // 0)] | @tsv' "$OUT/r-plain-$arm.json"
    done > "$OUT/plain-$arm.tsv"
    echo "plain-$arm: $(wc -l < "$OUT/plain-$arm.tsv") requests in $((SECONDS - t0)) s"
    stop "plain-$arm"
}

gram_arm() { # <arm> <seed offset> <port> <bin> <server args> [env...]
    local arm=$1 s0=$2 PORT=$3; shift 3
    want "gram-$arm" || return 0
    serve "gram-$arm" "$@" || return 1
    local t0=$SECONDS
    for s in $(seq 1 "$N_STAT_G"); do
        complete "$OUT/r-gram-$arm.json" "$P_TOOL" 32 1.0 $((s0 + s)) /r/stat_tool.gbnf
        jq -r --argjson s "$s" '[$s, (.content | gsub("[\t\n]"; " ")), (.timings.draft_n // 0), (.timings.draft_n_accepted // 0)] | @tsv' "$OUT/r-gram-$arm.json"
    done > "$OUT/gram-$arm.tsv"
    echo "gram-$arm: $(wc -l < "$OUT/gram-$arm.tsv") requests in $((SECONDS - t0)) s"
    grep -h "forced tokens =" "$OUT/gram-$arm.server.log" | awk -v arm="gram-$arm" '{ for (i = 1; i <= NF; i++) { if ($i == "=") s += $(i+1); if ($i == "appended") a += $(i-5); if ($i == "accepted") c += $(i-1) } }
        END { if (s > 0) printf "forced %s: steps %d, appended %d, accepted %d\n", arm, s, a, c }'
    stop "gram-$arm"
}

B=/c/bin/llama-server
plain_arm nospec 0       $((PORT0 + 0)) 1.0 $B "" &
plain_arm rej    1000000 $((PORT0 + 1)) 1.0 $B "$DRAFT" &
plain_arm block  2000000 $((PORT0 + 2)) 1.0 $B "$DRAFT" LLAMA_SPEC_BLOCK_VERIFY=1 &
plain_arm t08    3000000 $((PORT0 + 3)) 0.8 $B "" &
plain_arm rej2   9000000 $((PORT0 + 9)) 1.0 $B "$DRAFT2" &
plain_arm block2 10000000 $((PORT0 + 10)) 1.0 $B "$DRAFT2" LLAMA_SPEC_BLOCK_VERIFY=1 &
plain_arm rejkeep2 11000000 $((PORT0 + 11)) 1.0 $B "$DRAFT2" LLAMA_SPEC_KEEP_DRAWS=1 &
gram_arm  nospec 4000000 $((PORT0 + 4)) $B "" &
gram_arm  rej    5000000 $((PORT0 + 5)) $B "$DRAFT" &
gram_arm  block  6000000 $((PORT0 + 6)) $B "$DRAFT" LLAMA_SPEC_BLOCK_VERIFY=1 &
gram_arm  force  7000000 $((PORT0 + 7)) $B "$DRAFT" LLAMA_SPEC_FORCE=1 LLAMA_SPEC_BLOCK_VERIFY=1 &
gram_arm  keep0  8000000 $((PORT0 + 8)) $B "$DRAFT" LLAMA_SPEC_FORCE=1 LLAMA_SPEC_BLOCK_VERIFY=1 LLAMA_SPEC_DRAFT_KEEP=0 &
gram_arm  rejkeep 12000000 $((PORT0 + 12)) $B "$DRAFT" LLAMA_SPEC_KEEP_DRAWS=1 &
wait
rm -f "$OUT"/r-*.json
