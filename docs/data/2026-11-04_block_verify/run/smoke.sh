# The smoke stage of cpu_check.sh, inside its container (needs T, DR, THREADS, PORT; binaries in /c/bin, /c/bin-base).
# Every arm gets the same six requests:
#   R1 greedy, plain text (48 tokens)                 R4 temperature 1, top-k 8, seed 42, plain text (48 tokens)
#   R2 greedy, a JSON tool call under a grammar       R5 temperature 1, top-k 8, seed 7, the same grammar
#   R3 greedy, a tool call in Qwen3.8's XML format    R6 temperature 1, top-k 8, seed 11, the same grammar
#      (xml_tool.gbnf: its tags, with the special token <tool_call>)
# Under temperature 1 the drafts are verified by rejection (or block verification): exact in distribution, so R4,
# R5 and R6 may differ from plain decoding; with the switches off they must equal the base build's.
set -u
OUT=/c/smoke
mkdir -p "$OUT"
rm -f "$OUT"/*
. /r/serve_lib.sh

P_TEXT="def fibonacci(n):"
P_TOOL="Call one tool as JSON:"
P_XML="Show me the README of this repository."

run_arm() { # <arm> <bin> <server args> [env...]
    local arm=$1
    serve "$@" || return 1
    complete "$OUT/$arm.R1.json" "$P_TEXT" 48 0 1
    complete "$OUT/$arm.R2.json" "$P_TOOL" 96 0 1 /r/json_tool.gbnf
    complete "$OUT/$arm.R3.json" "$P_XML" 96 0 1 /r/xml_tool.gbnf
    complete "$OUT/$arm.R4.json" "$P_TEXT" 48 1.0 42
    complete "$OUT/$arm.R5.json" "$P_TOOL" 96 1.0 7 /r/json_tool.gbnf
    complete "$OUT/$arm.R6.json" "$P_XML" 96 1.0 11 /r/xml_tool.gbnf
    for r in R1 R2 R3 R4 R5 R6; do
        echo "$arm $r $(jq -c '{n: (.tokens | length), draft_n: .timings.draft_n, draft_n_accepted: .timings.draft_n_accepted}' "$OUT/$arm.$r.json")"
    done
    grep -h "speculative verification\|forced tokens =" "$OUT/$arm.server.log" | sed "s/^/$arm: /" | head -8
    stop "$arm"
}

B=/c/bin/llama-server
run_arm nospec      $B                   ""
run_arm base        /c/bin-base/llama-server "$DRAFT"
run_arm off         $B                   "$DRAFT"
run_arm block       $B                   "$DRAFT" LLAMA_SPEC_BLOCK_VERIFY=1
run_arm force       $B                   "$DRAFT" LLAMA_SPEC_FORCE=1
run_arm force-block $B                   "$DRAFT" LLAMA_SPEC_FORCE=1 LLAMA_SPEC_BLOCK_VERIFY=1
run_arm keep0       $B                   "$DRAFT" LLAMA_SPEC_FORCE=1 LLAMA_SPEC_BLOCK_VERIFY=1 LLAMA_SPEC_DRAFT_KEEP=0
run_arm keep2       $B                   "$DRAFT" LLAMA_SPEC_FORCE=1 LLAMA_SPEC_BLOCK_VERIFY=1 LLAMA_SPEC_DRAFT_KEEP=2
echo "--- bad"
if serve bad $B "$DRAFT" LLAMA_SPEC_FORCE=1 LLAMA_SPEC_FORCE_CHARS=abc; then
    echo "bad STARTED"; stop bad
else
    echo "bad REFUSED: $(grep -m 1 "LLAMA_SPEC_FORCE_CHARS" "$OUT/bad.server.log")"
fi

toks()  { jq -c '.tokens' "$1"; }
same()  { [ "$(toks "$1")" = "$(toks "$2")" ]; }
counts() { jq -c '[.timings.draft_n, .timings.draft_n_accepted]' "$1"; }

echo "--- the switches off against the base build: tokens and draft counts"
for r in R1 R2 R3 R4 R5 R6; do
    if same "$OUT/off.$r.json" "$OUT/base.$r.json" && [ "$(counts "$OUT/off.$r.json")" = "$(counts "$OUT/base.$r.json")" ]; then
        echo "off $r identical $(counts "$OUT/off.$r.json")"
    else
        echo "off $r DIFFER $(counts "$OUT/off.$r.json") base $(counts "$OUT/base.$r.json")"
    fi
done

echo "--- greedy replies against plain decoding (nospec)"
for arm in base off block force force-block keep0 keep2; do
    for r in R1 R2 R3; do
        if same "$OUT/$arm.$r.json" "$OUT/nospec.$r.json"; then echo "$arm $r lossless"; else echo "$arm $r NOT-LOSSLESS"; fi
    done
done

echo "--- grammar replies: valid JSON (R2, R5), a whole XML tool call (R3, R6)"
xml_re='<function=(read_file|write_file|list_dir)>\n<parameter=path>\n[a-zA-Z0-9_./]+\n</parameter>\n</function>'
for arm in nospec base off block force force-block keep0 keep2; do
    for r in R2 R5; do
        if jq -r '.content' "$OUT/$arm.$r.json" | jq -e '.name and .arguments.path' > /dev/null 2>&1; then
            echo "$arm $r grammar ok"
        else
            echo "$arm $r grammar BAD: $(jq -r '.content' "$OUT/$arm.$r.json" | head -c 200)"
        fi
    done
    for r in R3 R6; do
        if jq -e --arg re "$xml_re" '.content | test($re)' "$OUT/$arm.$r.json" > /dev/null 2>&1; then
            echo "$arm $r grammar ok"
        else
            echo "$arm $r grammar BAD: $(jq -c '.content' "$OUT/$arm.$r.json" | head -c 200)"
        fi
    done
done
for arm in nospec keep0; do
    for r in R2 R3 R6; do
        echo "$arm $r: $(jq -c '{content, tokens}' "$OUT/$arm.$r.json" | head -c 600)"
    done
done

echo "--- forced tokens (summed over the arm's six requests)"
for arm in force force-block keep0 keep2; do
    awk -v arm="$arm" '/forced tokens =/ { for (i = 1; i <= NF; i++) { if ($i == "=") s += $(i+1); if ($i == "appended") a += $(i-5); if ($i == "accepted") c += $(i-1) } }
        END { printf "forced %s: steps %d, appended %d, accepted %d\n", arm, s, a, c }' "$OUT/$arm.server.log"
done

echo "--- draft acceptance, R4 R5 R6 (temperature 1): [accepted, drafted]"
for arm in off block force-block keep0 keep2; do
    echo "$arm $(jq -s -c 'map([.timings.draft_n_accepted, .timings.draft_n])' "$OUT/$arm.R4.json" "$OUT/$arm.R5.json" "$OUT/$arm.R6.json")"
done
