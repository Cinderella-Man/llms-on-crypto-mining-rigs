#!/usr/bin/env bash
# Checks replay.py against the llama.cpp fork's own n-gram drafters: compiles common/ngram-{map,mod,cache}.cpp
# unchanged with stub headers (no ggml or llama library needed), replays every llama.cpp drafter setting of replay.py
# through fork_check.cpp, and compares the steps (position, drafted, accepted) one by one.
#
# usage: LLAMA_DIR=<llama.cpp checkout> bash fork_check.sh GEN.jsonl
set -euo pipefail

GEN=$(realpath "$1")
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT

cp "$LLAMA_DIR"/common/ngram-{map,mod,cache}.{h,cpp} "$W"/
cat > "$W/llama.h" <<'EOF'
#pragma once
#include <cstdint>
typedef int32_t llama_token;
#define LLAMA_TOKEN_NULL -1
EOF
cat > "$W/common.h" <<'EOF'
#pragma once
#include "llama.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
typedef std::vector<llama_token> llama_tokens;
#define GGML_ASSERT(x) do { if (!(x)) { fprintf(stderr, "GGML_ASSERT(%s) failed\n", #x); abort(); } } while (0)
#define GGML_ABORT(...) do { fprintf(stderr, __VA_ARGS__); abort(); } while (0)
static inline int64_t ggml_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
EOF
cat > "$W/log.h" <<'EOF'
#pragma once
#define LOG_DBG(...) do {} while (0)
#define LOG_INF(...) do {} while (0)
EOF
g++ -O2 -std=c++17 -I"$W" "$HERE/fork_check.cpp" "$W"/ngram-{map,mod,cache}.cpp -o "$W/fork_check"

python3 -I - "$GEN" "$W/seqs.txt" <<'EOF'
import json, sys
with open(sys.argv[2], "w") as out:
    for line in open(sys.argv[1]):
        r = json.loads(line)
        out.write(f"{r['id']} {len(r['prompt_tokens'])} " + " ".join(map(str, r["prompt_tokens"] + r["gen_tokens"])) + "\n")
EOF
python3 -I "$HERE/replay.py" "$GEN" --no-copy --only '^ngram-(simple|map|mod|cache)' --dump-steps "$W/py.tsv" > /dev/null

fail=0
while IFS='|' read -r name args; do
    # shellcheck disable=SC2086
    "$W/fork_check" "$W/seqs.txt" $args > "$W/cpp.txt"
    awk -F'\t' -v n="$name" '$1 == n { print $2, $3, $4, $5 }' "$W/py.tsv" > "$W/py.txt"
    steps=$(wc -l < "$W/cpp.txt")
    if cmp -s "$W/cpp.txt" "$W/py.txt"; then
        echo "same:      $name ($steps steps)"
    else
        echo "DIFFERENT: $name"
        diff "$W/cpp.txt" "$W/py.txt" | head -5
        fail=1
    fi
done <<'EOF'
ngram-simple 12/48 (default)|simple 12 48
ngram-simple 6/16|simple 6 16
ngram-simple 4/8|simple 4 8
ngram-simple 3/4|simple 3 4
ngram-map-k 12/48 (default)|map-k 12 48
ngram-map-k 6/16|map-k 6 16
ngram-map-k 4/8|map-k 4 8
ngram-map-k4v 12/48 (default)|map-k4v 12 48 1
ngram-map-k4v 8/8 hits 2 (docs)|map-k4v 8 8 2
ngram-map-k4v 8/24|map-k4v 8 24 1
ngram-mod 24/48/64 (default)|mod 24 48 64
ngram-mod 24/16/32|mod 24 16 32
ngram-mod 16/8/16|mod 16 8 16
ngram-mod 12/2/8|mod 12 2 8
ngram-cache (8)|cache
EOF
exit $fail
