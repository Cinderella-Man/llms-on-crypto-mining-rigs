#!/usr/bin/env bash
# p106_check.sh: is this mining rig ready for the Pascal engine? One PASS, WARN or FAIL line per check, for the quick
# start of the release (docs/research/notes/release_v2.md, section 3.4). Run on the rig as any user.
#
#   scripts/p106_check.sh                                   system checks: OS, glibc, driver, cards, host memory
#   BIN=<engine>/bin MODEL=<file.gguf> scripts/p106_check.sh
#                                                           + the engine's libraries and devices, the model's disk and
#                                                             read speed
#   BIN=... MODEL=... SERVE=1 scripts/p106_check.sh         + the release command line, started and sent one request
#                                                             each: chat, a tool request and a long prompt on all
#                                                             cards at the full context, and chat on 6 cards
#   BIN=... MODEL=<flint-Q8_0.gguf> TS=6,6,5,5,6,5,5,6,6 SERVE=1 scripts/p106_check.sh
#                                                           Flint Q8_0: its layer split (the six-card run is skipped,
#                                                             Q8_0 does not fit six cards)
#
# Env [defaults]: CARDS [9] cards expected, CARD_NAME [P106-090], SUMS (a SHA256SUMS file: also verify the model,
#   1.5-3 min for 25 GB on a 2-core Celeron), SERVE [0], CONFIGS [chat9 tools9 long9 chat6], TS (-ts for the runs on all
#   cards), DEPTH [4] (--pipe-depth), CTX6 [32768] context on six cards, LONG_TOKENS [16384] prompt length of the
#   long-context request, ENV [LLAMA_NO_MOE_DOWN_RESCALE=tg], PORT [8091], MIN_AVAIL_MB [300] (a server run whose host
#   MemAvailable falls below this fails)
# Exit status: 0 no FAIL, 1 at least one FAIL, 2 usage.
set -uo pipefail
BIN=${BIN:-}
MODEL=${MODEL:-}
CARDS=${CARDS:-9}
CARD_NAME=${CARD_NAME:-P106-090}
SUMS=${SUMS:-}
SERVE=${SERVE:-0}
CONFIGS=${CONFIGS:-chat9 tools9 long9 chat6}
TS=${TS:-}
DEPTH=${DEPTH:-4}
CTX6=${CTX6:-32768}
LONG_TOKENS=${LONG_TOKENS:-16384}
ENV=${ENV-LLAMA_NO_MOE_DOWN_RESCALE=tg}
PORT=${PORT:-8091}
MIN_AVAIL_MB=${MIN_AVAIL_MB:-300}
COMMON="-ngl 99 -fa on -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --no-webui --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0"
fails=0 warns=0
pass() { printf 'PASS  %-18s %s\n' "$1" "$2"; }
warn() { printf 'WARN  %-18s %s\n' "$1" "$2"; warns=$((warns + 1)); }
fail() { printf 'FAIL  %-18s %s\n' "$1" "$2"; fails=$((fails + 1)); }
info() { printf 'INFO  %-18s %s\n' "$1" "$2"; }
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }  # ver_ge A B: A >= B
meminfo() { awk -v k="$1:" '$1 == k { print int($2 / 1024) }' /proc/meminfo; }

# --- system ---
os=$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")
glibc=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$')
if [ -n "$glibc" ] && ver_ge "$glibc" 2.38; then pass "system" "$os, glibc $glibc"
else fail "system" "$os, glibc ${glibc:-unknown}: the release binaries need glibc 2.38 or newer (Ubuntu 24.04, Debian 13)"; fi

if ! command -v nvidia-smi > /dev/null; then
    fail "driver" "nvidia-smi not found: install nvidia-driver-580 (docs/setup.md)"
else
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ' ')
    lic=$(modinfo -F license nvidia 2>/dev/null | head -1)
    case "$drv" in
        580.*) if [ "$lic" = NVIDIA ] || [ -z "$lic" ]; then pass "driver" "${drv% }${lic:+, proprietary modules}"
               else fail "driver" "$drv with the open kernel modules ($lic): they support Turing and newer only; install nvidia-driver-580, not -open"; fi ;;
        "") fail "driver" "nvidia-smi finds no driver: $(nvidia-smi 2>&1 | head -1)" ;;
        *) fail "driver" "$drv: Pascal needs the 580 branch, the last one that supports it" ;;
    esac
    n=0 bad=""
    while IFS=',' read -r idx name total used; do
        name=$(echo "$name" | xargs) total=${total// /} used=${used// /}
        n=$((n + 1))
        case "$name" in *"$CARD_NAME"*) ;; *) bad="$bad card $idx is $name;";; esac
        [ "${total:-0}" -ge 5900 ] || bad="$bad card $idx has $total MiB;"
        [ "${used:-0}" -le 200 ] || bad="$bad card $idx already uses $used MiB;"
    done < <(nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null)
    if [ "$n" = "$CARDS" ] && [ -z "$bad" ]; then pass "cards" "$n × $CARD_NAME, ~6 GB each, free"
    elif [ "$n" != "$CARDS" ]; then fail "cards" "$n cards visible, $CARDS expected (CARDS=N if this rig has fewer)${bad:+;$bad}"
    else warn "cards" "${bad# }"; fi
    apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | tr '\n' ';')
    [ -z "$apps" ] && pass "gpu processes" "none" || warn "gpu processes" "already running: $apps"
fi

total=$(meminfo MemTotal) avail=$(meminfo MemAvailable)
if [ "$total" -lt 3700 ]; then fail "host memory" "$total MiB in total; 4 GB is the minimum"
elif [ "$avail" -lt 2000 ]; then warn "host memory" "$avail MiB available of $total: close other programs before starting the server"
else pass "host memory" "$avail MiB available of $total"; fi
info "cpu" "$(nproc) threads, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs); AVX $(grep -qw avx /proc/cpuinfo && echo yes || echo 'no (not needed)')"

# --- engine ---
if [ -n "$BIN" ]; then
    if [ ! -x "$BIN/llama-server" ]; then fail "engine" "no llama-server in $BIN"
    else
        missing=$(ldd "$BIN/llama-server" 2>/dev/null | grep 'not found' | awk '{print $1}' | tr '\n' ' ')
        if [ -n "$missing" ]; then fail "engine libraries" "not found: $missing(unpack the CUDA runtime asset next to the binaries)"
        else pass "engine libraries" "all found"; fi
        devs=$("$BIN/llama-server" --list-devices 2>&1 | grep -cE '^\s*CUDA[0-9]+:')
        if [ "$devs" = "$CARDS" ]; then pass "engine devices" "$devs CUDA devices"
        else fail "engine devices" "$devs CUDA devices seen by the engine, $CARDS expected: $("$BIN/llama-server" --list-devices 2>&1 | tail -2 | tr '\n' ' ')"; fi
    fi
fi

# --- model ---
if [ -n "$MODEL" ]; then
    if [ ! -f "$MODEL" ]; then fail "model" "$MODEL not found"
    else
        size=$(( $(stat -c %s "$MODEL") / 1000000000 ))
        free=$(df -BG --output=avail "$(dirname "$MODEL")" | tail -1 | tr -dc 0-9)
        fs=$(df --output=fstype "$(dirname "$MODEL")" | tail -1 | xargs)
        info "model" "$(basename "$MODEL"), $size GB, on $fs with $free GB free"
        [ "$fs" = vfat ] || [ "$fs" = exfat ] && warn "model filesystem" "$fs: use ext4 on USB sticks (exFAT and FAT read slowly here, and FAT has a 4 GB file limit)"
        mbs=$(dd if="$MODEL" of=/dev/null bs=16M count=128 iflag=direct 2>&1 | grep -oE '[0-9.,]+ [MG]B/s' | tail -1)
        rate=$(echo "$mbs" | awk '{ v = $1; gsub(",", ".", v); print ($2 == "GB/s") ? int(v * 1000) : int(v) }')
        if [ "${rate:-0}" -ge 400 ]; then pass "model read speed" "$mbs: a $size GB load takes about $(( size * 1000 / rate / 60 + 1 )) min"
        elif [ "${rate:-0}" -gt 0 ]; then warn "model read speed" "$mbs: a $size GB load takes about $(( size * 1000 / rate / 60 + 1 )) min; a USB 3 port and stick reach ~400 MB/s"
        else warn "model read speed" "could not measure"; fi
        if [ -n "$SUMS" ]; then
            if (cd "$(dirname "$MODEL")" && grep " $(basename "$MODEL")\$" "$SUMS" | sha256sum -c --quiet - 2>/dev/null); then
                pass "model checksum" "matches $SUMS"
            else fail "model checksum" "does not match $SUMS (or the file is not listed)"; fi
        fi
    fi
fi

# --- the shipped command lines ---
REQ_PY=$(mktemp /tmp/p106_check.XXXX.py)
trap 'rm -f "$REQ_PY"' EXIT
cat > "$REQ_PY" <<'EOF'
import json, sys, urllib.request
port, kind, long_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
body = {"messages": [{"role": "user", "content": "Name three uses of a hash table in one sentence each."}],
        "max_tokens": 256, "seed": 1}
if kind == "tools":
    body["messages"][0]["content"] = "What is the weather in Lisbon right now? Use the tool."
    body["tools"] = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    body["max_tokens"] = 1024
if kind == "long":  # about long_tokens tokens of filler, then a question about its first line
    line = "The archive keeps one ledger per shelf, and each ledger lists the boxes on that shelf in order. "
    body["messages"][0]["content"] = ("Remember: the code word is TANGERINE.\n" + line * (long_tokens // 20)
                                      + "\nWhat is the code word?")
    body["max_tokens"] = 1024
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=3600))
t, m = r.get("timings", {}), r["choices"][0]["message"]
out = (f"prompt {t.get('prompt_n')} tokens at {t.get('prompt_per_second') or 0:.0f} tok/s, "
       f"{t.get('predicted_n')} generated at {t.get('predicted_per_second') or 0:.1f} tok/s")
if kind == "tools":
    calls = m.get("tool_calls") or []
    out += f", tool calls: {', '.join(c['function']['name'] for c in calls) or 'none'}"
    if not calls:
        print(out)
        sys.exit(3)
if kind == "long" and "TANGERINE" not in ((m.get("content") or "") + (m.get("reasoning_content") or "")).upper():
    out += ", the reply does not repeat the code word"
print(out)
EOF
serve_check() {  # serve_check NAME CUDA_DEVICES SERVER_ARGS REQUEST_KIND
    local name=$1 devs=$2 args=$3 kind=$4 log srv rp minavail=999999 a ok=1 detail=""
    log=$(mktemp /tmp/p106_check.XXXX.log)
    env CUDA_VISIBLE_DEVICES="$devs" $ENV "$BIN/llama-server" -m "$MODEL" $COMMON $args --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 &
    srv=$!
    for _ in $(seq 1 300); do  # loads take ~1-3 min from a USB 3 stick
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break
        kill -0 "$srv" 2>/dev/null || break
        a=$(meminfo MemAvailable); [ "$a" -lt "$minavail" ] && minavail=$a
        [ "$a" -lt "$MIN_AVAIL_MB" ] && { kill -9 "$srv"; break; }
        sleep 2
    done
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
        python3 "$REQ_PY" "$PORT" "$kind" "$LONG_TOKENS" > "$log.req" 2>&1 &
        rp=$!
        while kill -0 "$rp" 2>/dev/null; do  # the host's memory while the request runs (long prompts pin the most)
            a=$(meminfo MemAvailable); [ "$a" -lt "$minavail" ] && minavail=$a
            [ "$a" -lt "$MIN_AVAIL_MB" ] && { kill -9 "$srv"; break; }
            sleep 1
        done
        wait "$rp"; ok=$?
        detail=$(tail -1 "$log.req"); rm -f "$log.req"
    else
        detail="the server did not start: $(grep -iE 'error|out of memory|failed' "$log" | tail -2 | tr '\n' ' ')"
    fi
    kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; sleep 3
    if [ "$minavail" -lt "$MIN_AVAIL_MB" ]; then fail "serve $name" "host MemAvailable fell to $minavail MiB (server stopped); server log: $log"; return
    elif [ "$ok" = 0 ]; then pass "serve $name" "$detail; host MemAvailable ≥ $minavail MiB"
    elif [ "$ok" = 3 ]; then warn "serve $name" "$detail: the model answered without calling the tool"
    else fail "serve $name" "${detail:-the request failed}; server log: $log"; return; fi
    rm -f "$log"
}
if [ "$SERVE" = 1 ]; then
    if [ -z "$BIN" ] || [ -z "$MODEL" ]; then echo "SERVE=1 needs BIN and MODEL"; exit 2; fi
    if pgrep -x llama-server > /dev/null; then fail "serve" "a llama-server is already running: stop it first"
    else
        all=$(seq -s, 0 $((CARDS - 1))) six=0,1,2,3,4,5
        line="-c 262144 -b 512 -ub 128${TS:+ -ts $TS} --pipe-depth $DEPTH"   # the release command line
        size_mb=$(( $(stat -Lc %s "$MODEL") / 1048576 ))
        for c in $CONFIGS; do
            case $c in
                chat9) serve_check "chat, $CARDS cards" "$all" "$line" chat ;;
                tools9) serve_check "tools, $CARDS cards" "$all" "$line" tools ;;
                long9) serve_check "long, $CARDS cards" "$all" "$line" long ;;
                chat6) if [ "$size_mb" -gt 33000 ]; then info "serve chat, 6 cards" "skipped: the model ($((size_mb / 1024)) GiB) does not fit six cards"
                       else serve_check "chat, 6 cards" "$six" "-c $CTX6 -b 512 -ub 128 --pipe-depth $DEPTH" chat; fi ;;
                *) echo "unknown config $c"; exit 2 ;;
            esac
        done
    fi
fi

echo "$fails failed, $warns warnings"
[ "$fails" = 0 ]
