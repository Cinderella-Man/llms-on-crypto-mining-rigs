#!/usr/bin/env bash
# Token frequencies of a model's own agentic output, for LLAMA_DRAFT_SHORTLIST_FILE (Q592, Q400's method): the
# assistant turns (text and tool-call arguments) of an agent check's logged requests (bench/agent_check's proxy log,
# proxy.jsonl.bodies.gz), each distinct turn once, tokenized by a running llama-server of the same vocabulary
# (/tokenize), counted, most frequent first. Reasoning text is not in the logged requests, so it is not counted.
# Env: BODIES (the .gz log), URL (e.g. http://127.0.0.1:18181), OUT (the .tsv to write: token id, count). Needs jq, curl.
set -euo pipefail
: "${BODIES:?}" "${URL:?}" "${OUT:?}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

zcat "$BODIES" | jq -c '.body.messages[]? | select(.role == "assistant")
    | [(if (.content | type) == "string" then .content else "" end),
       ((.tool_calls // [])[] | (.function.arguments // "") | if type == "string" then . else tojson end)]
    | join("\n") | select(length > 0)' | sort -u > "$tmp/turns.jsonl"

while read -r s; do
    jq -nc --argjson c "$s" '{content: $c, add_special: false}' |
        curl -sf "$URL/tokenize" -H 'Content-Type: application/json' -d @- | jq -r '.tokens[]'
done < "$tmp/turns.jsonl" | sort -n | uniq -c | sort -k1,1nr -k2,2n | awk '{print $2 "\t" $1}' > "$OUT"

echo "$(wc -l < "$tmp/turns.jsonl") distinct turns, $(wc -l < "$OUT") distinct tokens, $(awk '{s += $2} END {print s}' "$OUT") tokens"
