#!/usr/bin/env bash
# C874 (Q1577): checks the HF_UI_VERSION logic of scripts/build_llamacpp_docker.sh without building, on LLMRig's CPU.
# Usage: T=<scratch dir> SCRIPT=<build_llamacpp_docker.sh> bash logic_test.sh
#  1. host side: a stand-in docker that prints its arguments; which -e HF_UI_VERSION reaches the container with
#     nothing set, with the environment, with DOCKER_ARGS, and with both (DOCKER_ARGS wins)
#  2. the container script parses (bash -n on the -c string the stand-in received)
#  3. the container's version choice (the block from "# the web page:" to its echo, run as is) in a repo with no tags,
#     with a b<N> tag, with a non-matching tag only, and with HF_UI_VERSION given
#  4. the "took" parser on the tree's success line, a fallback to latest, and a failed download
set -uo pipefail
T=${T:?T=<scratch dir>}
SCRIPT=${SCRIPT:?SCRIPT=<build_llamacpp_docker.sh>}
rm -rf "$T" && mkdir -p "$T/bin" "$T/work"
cat > "$T/bin/docker" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do printf '%s\n' "$a"; done > "$T_OUT/args.txt"
printf '%s' "${@: -1}" > "$T_OUT/inner.sh"
EOF
chmod +x "$T/bin/docker"
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
check "block extracted (lines)" "$(wc -l < "$T/choose.sh")" "14"
R=$T/repo
git init -q "$R" && git -C "$R" -c user.email=t@t -c user.name=t commit -q --allow-empty -m one
run_choose() { (cd "$R" && env -u HF_UI_VERSION REF=fork/x "$@" bash -euo pipefail -c "$(cat "$T/choose.sh")
echo \"result=\${HF_UI_VERSION:-unset} exported=\$(env | grep -c ^HF_UI_VERSION=)\"" 2>"$T/stderr.txt"); }
out=$(run_choose); check "no tags" "$(echo "$out" | grep ^result=)" "result=unset exported=0"
check "no tags: warning" "$(grep -c '^WARNING: no upstream release tag' "$T/stderr.txt")" "1"
git -C "$R" tag miner-llama-v9.9
out=$(run_choose); check "only a non-matching tag" "$(echo "$out" | grep ^result=)" "result=unset exported=0"
git -C "$R" tag b12483 HEAD
git -C "$R" -c user.email=t@t -c user.name=t commit -q --allow-empty -m two
git -C "$R" tag miner-llama-v9.10
out=$(run_choose); check "b12483 below HEAD" "$(echo "$out" | grep ^result=)" "result=b12483 exported=1"
check "b12483: printed" "$(echo "$out" | grep '^web page:')" "web page: HF_UI_VERSION=b12483 (the upstream release this tree sits on)"
check "b12483: no warning" "$(wc -c < "$T/stderr.txt")" "0"
out=$(run_choose HF_UI_VERSION=b12360); check "given" "$(echo "$out" | grep ^result=)" "result=b12360 exported=1"
check "given: printed" "$(echo "$out" | grep '^web page:')" "web page: HF_UI_VERSION=b12360 (given)"

echo "== 4. the took parser"
took() { printf '%s\n' "$@" > "$T/ui.log"; sed -n "s/.*, resolved: \(.*\))$/\1/p" "$T/ui.log" | tail -n 1; }
check "pinned" "$(took '-- UI: downloading from b12483: https://huggingface.co/buckets/ggml-org/llama-ui/resolve/b12483/dist.tar.gz' \
    '-- UI: archive verified and extracted' '-- UI: HF download succeeded, stamp updated (ggml-org/llama-ui|b12483, resolved: b12483)')" "b12483"
check "fell back" "$(took '-- UI: download dist.tar.gz.sha256 from b99999 failed: "HTTP response code said error"' \
    '-- UI: HF download succeeded, stamp updated (ggml-org/llama-ui|b99999, resolved: latest)')" "latest"
check "failed" "$(took '-- UI: download dist.tar.gz.sha256 from latest failed: "Couldn'"'"'t resolve host name"' '-- UI: HF download failed')" ""
echo "== $([ $fail = 0 ] && echo all passed || echo FAILED)"
exit $fail
