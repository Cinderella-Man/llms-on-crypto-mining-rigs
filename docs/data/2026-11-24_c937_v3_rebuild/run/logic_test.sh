#!/usr/bin/env bash
# C937 (Q1677, 2026-11-24): checks the web page choice of scripts/build_llamacpp_docker.sh without building: C874's
# checks (host side, parse, version choice, took parser) plus the walk down to the newest release with an archive.
# Usage: T=<scratch dir> SCRIPT=<build_llamacpp_docker.sh> bash logic_test.sh
#  1. host side: a stand-in docker that prints its arguments; which -e HF_UI_VERSION reaches the container with
#     nothing set, with the environment, with DOCKER_ARGS, and with both (DOCKER_ARGS wins)
#  2. the container script parses (bash -n on the -c string the stand-in received)
#  3. the container's version choice (the block from "# the web page:" to its echo, run as is) in a repository with
#     tags b12090, b12091, b12092 below HEAD and b12100 on a branch HEAD does not contain, with a stand-in curl that
#     answers each release from a table (one answer per call, the last one repeated) and logs its calls
#  4. the "took" parser on the tree's success line, a fallback to latest, and a failed download
set -uo pipefail
T=${T:?T=<scratch dir>}
SCRIPT=${SCRIPT:?SCRIPT=<build_llamacpp_docker.sh>}
rm -rf "$T" && mkdir -p "$T/bin" "$T/work" "$T/curl"
cat > "$T/bin/docker" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do printf '%s\n' "$a"; done > "$T_OUT/args.txt"
printf '%s' "${@: -1}" > "$T_OUT/inner.sh"
EOF
# stand-in curl: the release is the path segment after resolve/; $T_OUT/curl/<release> holds its answers in order
cat > "$T/bin/curl" <<'EOF'
#!/usr/bin/env bash
url=${@: -1}; v=${url#*/resolve/}; v=${v%%/*}
echo "$v" >> "$T_OUT/curl_calls.txt"
f=$T_OUT/curl/$v
[ -f "$f" ] || { printf 404; exit 0; }
n=$(grep -c "^$v$" "$T_OUT/curl_calls.txt")
a=$(tr ' ' '\n' < "$f" | grep . | sed -n "${n}p"); [ -n "$a" ] || a=$(tr ' ' '\n' < "$f" | grep . | tail -n 1)
printf '%s' "$a"
[ "$a" = 000 ] && exit 28 || exit 0
EOF
chmod +x "$T/bin/docker" "$T/bin/curl"
fail=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1: $2"; else echo "FAIL $1: got '$2', want '$3'"; fail=1; fi; }
ui_args() { grep -x -A1 -e '-e' "$T/args.txt" | grep '^HF_UI_VERSION=' | tr '\n' ' ' | sed 's/ $//'; }

echo "== 1. host side"
export T_OUT=$T
env -u HF_UI_VERSION PATH="$T/bin:$PATH" WORK="$T/work" DOCKER_ARGS="--label x=1" bash "$SCRIPT" fork/x > /dev/null
check "nothing set" "$(ui_args)" ""
env HF_UI_VERSION=b111 PATH="$T/bin:$PATH" WORK="$T/work" DOCKER_ARGS="--label x=1" bash "$SCRIPT" fork/x > /dev/null
check "environment" "$(ui_args)" "HF_UI_VERSION=b111"
env -u HF_UI_VERSION PATH="$T/bin:$PATH" WORK="$T/work" DOCKER_ARGS="--label x=1 -e HF_UI_VERSION=b222" bash "$SCRIPT" fork/x > /dev/null
check "DOCKER_ARGS" "$(grep -c '^HF_UI_VERSION=b222$' "$T/args.txt")" "1"
env HF_UI_VERSION=b111 PATH="$T/bin:$PATH" WORK="$T/work" DOCKER_ARGS="--label x=1 -e HF_UI_VERSION=b222" bash "$SCRIPT" fork/x > /dev/null
check "both: DOCKER_ARGS only" "$(grep '^HF_UI_VERSION=' "$T/args.txt" | tr '\n' ' ' | sed 's/ $//')" "HF_UI_VERSION=b222"

echo "== 2. the container script parses"
bash -n "$T/inner.sh" && check "bash -n" "0" "0" || check "bash -n" "1" "0"

echo "== 3. the version choice in the container"
sed -n '/# the web page: the release given/,/echo "web page: HF_UI_VERSION=/p' "$T/inner.sh" > "$T/choose.sh"
check "block ends with the echo" "$(tail -n 1 "$T/choose.sh" | grep -c 'echo "web page: HF_UI_VERSION=')" "1"
R=$T/repo
g() { git -C "$R" -c user.email=t@t -c user.name=t "$@"; }
git init -q -b main "$R"
g commit -q --allow-empty -m zero
run_choose() { rm -f "$T/curl_calls.txt"; touch "$T/curl_calls.txt"
    (cd "$R" && env -u HF_UI_VERSION PATH="$T/bin:$PATH" REF=fork/x "$@" bash -euo pipefail -c "$(cat "$T/choose.sh")
echo \"result=\${HF_UI_VERSION:-unset} exported=\$(env | grep -c ^HF_UI_VERSION=)\"" 2>"$T/stderr.txt"); }
calls() { tr '\n' ' ' < "$T/curl_calls.txt" | sed 's/ $//'; }
answers() { rm -f "$T"/curl/*; while [ $# -gt 0 ]; do echo "$2" > "$T/curl/$1"; shift 2; done; }

out=$(run_choose); check "no tags" "$(echo "$out" | grep ^result=)" "result=unset exported=0"
check "no tags: warning" "$(grep -c '^WARNING: no upstream release tag' "$T/stderr.txt")" "1"
check "no tags: no request" "$(calls)" ""
g tag miner-llama-v9.9
out=$(run_choose); check "only a non-matching tag" "$(echo "$out" | grep ^result=)" "result=unset exported=0"

g commit -q --allow-empty -m b12090 && g tag b12090
g commit -q --allow-empty -m b12091 && g tag b12091 && g tag b12091-extra
g commit -q --allow-empty -m b12092 && g tag b12092
g checkout -q -b side && g commit -q --allow-empty -m b12100 && g tag b12100 && g checkout -q main
g commit -q --allow-empty -m fork && g tag miner-llama-v9.10
answers b12092 404 b12091 200 b12100 200
out=$(run_choose); check "a tree that does not download by release: own release" "$(echo "$out" | grep ^result=)" "result=b12092 exported=1"
check "  no request" "$(calls)" ""

mkdir -p "$R/tools/ui" && echo 'set(HF_UI_VERSION "$ENV{HF_UI_VERSION}")' > "$R/tools/ui/CMakeLists.txt"
g add tools/ui/CMakeLists.txt && g commit -q -m ui
answers b12092 200
out=$(run_choose); check "own release has an archive" "$(echo "$out" | grep ^result=)" "result=b12092 exported=1"
check "  printed" "$(echo "$out" | grep '^web page:')" "web page: HF_UI_VERSION=b12092 (the upstream release this tree sits on)"
check "  one request" "$(calls)" "b12092"
check "  no warning" "$(wc -c < "$T/stderr.txt")" "0"

answers b12092 404 b12091 200 b12100 200
out=$(run_choose); check "own release 404: walks down" "$(echo "$out" | grep ^result=)" "result=b12091 exported=1"
check "  printed" "$(echo "$out" | grep '^web page:')" \
    "web page: HF_UI_VERSION=b12091 (the newest upstream release with an archive at or below b12092, the one this tree sits on)"
check "  requests (b12100 is not in the tree)" "$(calls)" "b12092 b12091"
check "  no warning" "$(wc -c < "$T/stderr.txt")" "0"

answers b12092 404 b12091 404 b12090 200
out=$(run_choose); check "two 404s" "$(echo "$out" | grep ^result=)" "result=b12090 exported=1"
check "  requests" "$(calls)" "b12092 b12091 b12090"

answers b12092 404 b12091 "500 200"
out=$(run_choose); check "a 500 asked twice, then 200" "$(echo "$out" | grep ^result=)" "result=b12091 exported=1"
check "  requests" "$(calls)" "b12092 b12091 b12091"

answers b12092 404 b12091 "000 000" b12090 200
out=$(run_choose); check "no answer twice: stops, own release" "$(echo "$out" | grep ^result=)" "result=b12092 exported=1"
check "  requests" "$(calls)" "b12092 b12091 b12091"
check "  warning" "$(grep -c '^WARNING: no web page archive found for b12092 or below (last answer: HTTP 000)' "$T/stderr.txt")" "1"

answers
out=$(run_choose); check "no archive at all: own release" "$(echo "$out" | grep ^result=)" "result=b12092 exported=1"
check "  requests" "$(calls)" "b12092 b12091 b12090"
check "  warning" "$(grep -c '^WARNING: no web page archive found for b12092 or below (last answer: HTTP 404)' "$T/stderr.txt")" "1"

answers b12092 404 b12091 200
out=$(run_choose HF_UI_VERSION=b12128); check "given" "$(echo "$out" | grep ^result=)" "result=b12128 exported=1"
check "  printed" "$(echo "$out" | grep '^web page:')" "web page: HF_UI_VERSION=b12128 (given)"
check "  no request" "$(calls)" ""

echo "== 4. the took parser"
took() { printf '%s\n' "$@" > "$T/ui.log"; sed -n "s/.*, resolved: \(.*\))$/\1/p" "$T/ui.log" | tail -n 1; }
check "pinned" "$(took '-- UI: downloading from b12091: https://huggingface.co/buckets/ggml-org/llama-ui/resolve/b12091/dist.tar.gz' \
    '-- UI: archive verified and extracted' '-- UI: HF download succeeded, stamp updated (ggml-org/llama-ui|b12091, resolved: b12091)')" "b12091"
check "fell back" "$(took '-- UI: download dist.tar.gz.sha256 from b12092 failed: "HTTP response code said error"' \
    '-- UI: HF download succeeded, stamp updated (ggml-org/llama-ui|b12092, resolved: latest)')" "latest"
check "failed" "$(took '-- UI: download dist.tar.gz.sha256 from latest failed: "Couldn'"'"'t resolve host name"' '-- UI: HF download failed')" ""
echo "== $([ $fail = 0 ] && echo all passed || echo FAILED)"
exit $fail
