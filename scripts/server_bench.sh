#!/usr/bin/env bash
# Benchmarks llama-server end to end: model load time, GPU memory per card once loaded, then a fixed set of chat
# prompts at temperature 0 with per-request speed and speculative-decoding acceptance. The server listens on
# 127.0.0.1 only and is stopped at the end. Results go to /tmp/server_bench/<TAG>/.
#
#   MODEL=/mnt/models/x.gguf TAG=base SERVER_ARGS="-c 32768" bash scripts/server_bench.sh
#   MODEL=... TAG=mtp SERVER_ARGS="-c 32768 --spec-type draft-mtp -md /mnt/models/mtp.gguf" bash scripts/server_bench.sh
#   PROMPTS=none: only load the model and record memory (for context-size fit tests)
#   PROMPTS=chat: instead of the four one-shot prompts, one conversation of five user messages that build on each other,
#     sent the way a chat client sends them: every request carries the whole history, each answer with its reasoning
#     as `reasoning_content` (KEEP_REASONING=0 resends the visible answers only), and the server reuses what it has
#     cached. Sampling: the model's GGUF defaults (general.sampling.*) or TEMP/TOP_P/TOP_K/MIN_P, fixed seeds; min-p is
#     0 unless the GGUF or MIN_P sets it (llama-server's own default is 0.05; Flint's makers use 0). Per turn it reports
#     the new prompt tokens and the time to
#     the first token, the generated tokens (reasoning + answer) and their speed; transcript in chat.json. Models with
#     sliding-window attention can reuse the cache only through context checkpoints: pass --ctx-checkpoints N (a few)
#     in SERVER_ARGS, or every turn re-processes the whole history. Use MAX_TOKENS of 2048 or more for reasoning models.
#
# Host-RAM safety: --cache-ram 0 and --ctx-checkpoints 0 are always passed (llama-server defaults to an 8 GiB prompt
# cache and 32 context checkpoints per slot in host RAM); later SERVER_ARGS override them.
# Env: GPUS (CUDA_VISIBLE_DEVICES, default all), MAX_TOKENS (default 256), PORT (default 8089), LLAMA_DIR or BIN,
# WARMUP=1 (one untimed short request before the four prompts).
set -u

BIN=${BIN:-${LLAMA_DIR:-$HOME/llama.cpp}/build/bin}
MODEL=${MODEL:?set MODEL}
TAG=${TAG:?set TAG}
SERVER_ARGS=${SERVER_ARGS:-"-c 32768"}
GPUS=${GPUS:-0,1,2,3,4,5,6,7,8}
MAX_TOKENS=${MAX_TOKENS:-256}
PORT=${PORT:-8089}
PROMPTS=${PROMPTS:-default}
OUT=/tmp/server_bench/$TAG
mkdir -p "$OUT"

t0=$(date +%s.%N)
CUDA_VISIBLE_DEVICES=$GPUS "$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on --host 127.0.0.1 --port "$PORT" \
    -np 1 --no-webui --cache-ram 0 --ctx-checkpoints 0 $SERVER_ARGS > "$OUT/server.log" 2>&1 &
srv=$!

# Watchdog: with only 4 GB of host RAM, a configuration that pins too much host memory (e.g. a very large context)
# can push the whole machine into the out-of-memory killer. scripts/ram_watchdog.sh (embedded below, byte for byte) runs
# beside the server at real-time priority, gives it oom_score_adj 1000 and kills it first on MemAvailable minus CmaFree
# below MIN_USABLE_MB (20), on memory pressure above 40 % for 10 s, on more than 600 MiB swapped out in 10 s, or on
# MemAvailable below MIN_AVAIL_MB (400); every limit is an environment variable (the script's header lists them). Its
# TSV, one row a second, ends up in $OUT/ram_watchdog.tsv.
write_watchdog() {
  cat << 'RAM_WATCHDOG'
#!/usr/bin/env bash
# ram_watchdog.sh: stops llama-server before a 4 GB host runs out of memory (Q559; from Q558's mem_guard.sh). A host that
# swaps itself to a stop does not look out of memory to MemAvailable alone: on the rigs' kernels 410-970 MiB of the 3.8 GiB
# are CMA-marked pageblocks that MemAvailable counts and that pinned buffers and kernel allocations cannot use, so it read
# ~949 MiB while the host was already out of memory (Q226). Every second this script samples
#   usable   MemAvailable minus CmaFree: what pinned buffers and the kernel's own allocations can take
#   swap-out pages written to swap in the last WINDOW_S samples (/proc/vmstat pswpout), in MiB
#   pressure /proc/pressure/memory, "some" and "full" avg10, in percent
# and kills the guarded llama-server (SIGKILL) when one of these holds, every limit an environment variable:
#   MIN_USABLE_MB  (20)   usable below it for USABLE_FLOOR_S (2) samples in a row
#   PSI_FULL_PCT   (40)   full avg10 above it for PSI_FULL_S (10) samples in a row
#   PSI_SOME_PCT   (40)   some avg10 above it for PSI_SOME_S (10) samples in a row
#   KILL_SWAPOUT_MIB (600) more than that written to swap within WINDOW_S (10) seconds
#   MIN_AVAIL_MB   (0)    MemAvailable alone below it (the agent check's old floor; 0 = off, as the floor above is the
#                         one that counts CMA; bench/agent_check/rig_server.sh and scripts/server_bench.sh pass 400)
# A limit set to "off" is disabled. The defaults sit between what runs that finished reached and what the runs that failed
# reached (docs/data/2026-11-04_ram_watchdog/thresholds.tsv has each number's source). It also
#   - gives the guarded process oom_score_adj OOM_SCORE_ADJ (1000), so the kernel's OOM killer takes it before the desktop
#     or the watchdog (which gets -1000), and reads the value back;
#   - records the scheduling policy of itself and of the guarded process: start it under a real-time policy
#     (chrt -f 10) so that it still gets a CPU while the kernel reclaims (in Q226 the old watchdog's loop got none for
#     8 minutes);
#   - uses bash builtins only in its loop (no fork per sample).
# Which process: PID=<pid> guards that process and exits when it ends; without PID it guards every process whose
# /proc/<pid>/comm is PROC (llama-server; llama-bench for Q583) and keeps running until it is killed.
# Log: LOG (a TSV, put it on tmpfs so that writing it never waits on a disk), one row a second, tab separated, the first
# nine columns those of mem_guard.sh:
#   epoch  avail_mib  cma_free_mib  usable_mib  swap_used_mib  swapout_window_mib  psi_some_avg10  psi_full_avg10  pid
# The first line names them; lines starting with # are events (the limits, the process found, its oom_score_adj and
# scheduling, KILL and its reason, the stop): read it with comment='#'. After a kill the numbers behind it are also
# appended to SERVER_LOG (if set) behind a line starting "watchdog:", as the agent check's logs have always had.
# MEMINFO, VMSTAT and PSI name other files to read from (the self-test feeds synthetic numbers).
#   PID=$srv LOG=/dev/shm/ram_watchdog.tsv SERVER_LOG=server.log setsid nohup chrt -f 10 bash ram_watchdog.sh &
#   LOG=/dev/shm/ram_watchdog.tsv setsid nohup chrt -f 10 bash ram_watchdog.sh &    # every llama-server, until killed
# Stop it with SIGTERM (it writes "stopped"). Run as root (oom_score_adj and SIGKILL of the server's user).
set -u
LOG=${LOG:?LOG=<the TSV to write, e.g. /dev/shm/ram_watchdog.tsv>}
PID=${PID:-}
PROC=${PROC:-llama-server}
SERVER_LOG=${SERVER_LOG:-}
MIN_USABLE_MB=${MIN_USABLE_MB:-20}
USABLE_FLOOR_S=${USABLE_FLOOR_S:-2}
MIN_AVAIL_MB=${MIN_AVAIL_MB:-0}
PSI_FULL_PCT=${PSI_FULL_PCT:-40}
PSI_FULL_S=${PSI_FULL_S:-10}
PSI_SOME_PCT=${PSI_SOME_PCT:-40}
PSI_SOME_S=${PSI_SOME_S:-10}
KILL_SWAPOUT_MIB=${KILL_SWAPOUT_MIB:-600}
WINDOW_S=${WINDOW_S:-10}
OOM_SCORE_ADJ=${OOM_SCORE_ADJ:-1000}
MEMINFO=${MEMINFO:-/proc/meminfo}
VMSTAT=${VMSTAT:-/proc/vmstat}
PSI=${PSI:-/proc/pressure/memory}

for v in MIN_USABLE_MB MIN_AVAIL_MB; do [ "${!v}" = off ] && printf -v "$v" %s -1000000; done
for v in PSI_FULL_PCT PSI_SOME_PCT KILL_SWAPOUT_MIB; do [ "${!v}" = off ] && printf -v "$v" %s 1000000; done
for v in MIN_USABLE_MB USABLE_FLOOR_S MIN_AVAIL_MB PSI_FULL_PCT PSI_FULL_S PSI_SOME_PCT PSI_SOME_S KILL_SWAPOUT_MIB \
  WINDOW_S OOM_SCORE_ADJ; do
  [[ ${!v} =~ ^-?[0-9]+$ ]] || { echo "ram_watchdog: $v must be an integer or off, not '${!v}'" >&2; exit 2; }
done
[ "$WINDOW_S" -ge 1 ] || { echo "ram_watchdog: WINDOW_S must be at least 1" >&2; exit 2; }
if [ -n "$PID" ]; then
  [[ $PID =~ ^[0-9]+$ ]] && kill -0 "$PID" 2> /dev/null || { echo "ram_watchdog: no process '$PID'" >&2; exit 2; }
fi
exec {lf}>> "$LOG" || { echo "ram_watchdog: cannot write $LOG" >&2; exit 1; }
exec {tick}<> <(:)  # read -t on this fd is a sleep without a fork
page=$(getconf PAGESIZE 2> /dev/null || echo 4096)
have_psi=0
[ -r "$PSI" ] && have_psi=1

now=0
ev() { printf '# %s %s\n' "$now" "$*" >&"$lf"; }
sched_of() {  # sched_of PID: SCHED = "SCHED_FIFO rt 10", "SCHED_OTHER nice 0", ...
  local s f
  s=$(< "/proc/$1/stat") 2> /dev/null || { SCHED=unknown; return; }
  s=${s##*) }
  read -ra f <<< "$s"
  case ${f[38]:-x} in
    0) SCHED="SCHED_OTHER nice ${f[16]}" ;;
    1) SCHED="SCHED_FIFO rt ${f[37]}" ;;
    2) SCHED="SCHED_RR rt ${f[37]}" ;;
    3) SCHED="SCHED_BATCH nice ${f[16]}" ;;
    5) SCHED="SCHED_IDLE" ;;
    6) SCHED="SCHED_DEADLINE" ;;
    *) SCHED="unknown" ;;
  esac
}
find_pid() {  # the first process whose comm is PROC, in pid
  local d c
  pid=""
  for d in /proc/[0-9]*; do
    { read -r c < "$d/comm"; } 2> /dev/null && [ "$c" = "$PROC" ] && { pid=${d#/proc/}; return; }
  done
}
dump() {  # dump REASON: the numbers behind a kill, appended to the server's log
  [ -n "$SERVER_LOG" ] || return 0
  {
    echo "watchdog: $1, killing $PROC"
    while read -r k v u; do
      case $k in MemAvailable: | MemFree: | CmaFree: | Shmem: | Mlocked: | Unevictable: | SwapFree:) echo "  $k $v $u" ;; esac
    done < "$MEMINFO"
    [ "$have_psi" = 1 ] && while read -r l; do echo "  psi $l"; done < "$PSI"
  } >> "$SERVER_LOG" 2> /dev/null
}
trap 'ev stopped; exit 0' TERM INT

printf -v now '%(%s)T' -1
printf '#epoch\tavail_mib\tcma_free_mib\tusable_mib\tswap_used_mib\tswapout_window_mib\tpsi_some_avg10\tpsi_full_avg10\tpid\n' >&"$lf"
printf -v when '%(%Y-%m-%dT%H:%M:%S%z)T' -1
target="every $PROC"
[ -n "$PID" ] && target="pid $PID"
ev "$when ram_watchdog: usable < $MIN_USABLE_MB MiB for $USABLE_FLOOR_S s; MemAvailable < $MIN_AVAIL_MB MiB; psi full > $PSI_FULL_PCT % for $PSI_FULL_S s, some > $PSI_SOME_PCT % for $PSI_SOME_S s; swap-out > $KILL_SWAPOUT_MIB MiB in $WINDOW_S s; guards $target"
echo -1000 > /proc/$$/oom_score_adj 2> /dev/null || ev "could not set the watchdog's own oom_score_adj"
sched_of $$
ev "watchdog pid $$: oom_score_adj $({ read -r x < /proc/$$/oom_score_adj; } 2> /dev/null && echo "$x"), $SCHED"
[ "$have_psi" = 1 ] || ev "no $PSI: the pressure limits are off"

declare -a ring
i=0 nu=0 nf=0 ns=0 adj_pid=""
pid=$PID
while :; do
  printf -v now '%(%s)T' -1
  if [ -n "$PID" ]; then
    kill -0 "$pid" 2> /dev/null || { ev "$PROC $pid ended"; break; }
  elif [ -z "$pid" ] || [ ! -d "/proc/$pid" ]; then
    find_pid
  fi
  if [ -n "$pid" ] && [ "$pid" != "$adj_pid" ]; then
    adj_pid=$pid
    echo "$OOM_SCORE_ADJ" > "/proc/$pid/oom_score_adj" 2> /dev/null
    back=$({ read -r x < "/proc/$pid/oom_score_adj"; } 2> /dev/null && echo "$x")
    sched_of "$pid"
    ev "$PROC $pid: oom_score_adj ${back:-unreadable} (set $OOM_SCORE_ADJ), $SCHED"
  fi
  avail="" cma=0 st=0 sf=0 out="" some=0.00 full=0.00
  while read -r k v _; do
    case $k in
      MemAvailable:) avail=$((v / 1024)) ;;
      CmaFree:) cma=$((v / 1024)) ;;
      SwapTotal:) st=$v ;;
      SwapFree:) sf=$v ;;
    esac
  done < "$MEMINFO"
  while read -r k v; do [ "$k" = pswpout ] && { out=$v; break; }; done < "$VMSTAT"
  if [ "$have_psi" = 1 ]; then
    while read -r kind a10 _; do
      a10=${a10#avg10=}
      if [ "$kind" = some ]; then some=$a10; else full=$a10; fi
    done < "$PSI"
  fi
  if [ -z "$avail" ] || [ -z "$out" ]; then  # a sample that could not be read is not a number to act on
    ev "unreadable sample (MemAvailable '${avail}', pswpout '${out}')"
    read -rt 1 -u "$tick"
    continue
  fi
  usable=$((avail - cma))
  slot=$((i % WINDOW_S))
  prev=${ring[$slot]:-$out}
  ring[$slot]=$out
  win=$(((out - prev) * page / 1048576))
  s100=$((10#${some/./})) f100=$((10#${full/./}))
  why=""
  if [ -n "$pid" ]; then
    [ "$usable" -lt "$MIN_USABLE_MB" ] && nu=$((nu + 1)) || nu=0
    [ "$s100" -gt $((PSI_SOME_PCT * 100)) ] && ns=$((ns + 1)) || ns=0
    [ "$f100" -gt $((PSI_FULL_PCT * 100)) ] && nf=$((nf + 1)) || nf=0
    [ "$avail" -lt "$MIN_AVAIL_MB" ] && why="MemAvailable $avail MiB, below $MIN_AVAIL_MB"
    [ "$nu" -ge "$USABLE_FLOOR_S" ] && why="${why:+$why; }MemAvailable minus CmaFree $usable MiB (MemAvailable $avail, CmaFree $cma), below $MIN_USABLE_MB for $nu s"
    [ "$ns" -ge "$PSI_SOME_S" ] && why="${why:+$why; }memory pressure some avg10 $some %, above $PSI_SOME_PCT % for $ns s"
    [ "$nf" -ge "$PSI_FULL_S" ] && why="${why:+$why; }memory pressure full avg10 $full %, above $PSI_FULL_PCT % for $nf s"
    [ "$win" -gt "$KILL_SWAPOUT_MIB" ] && why="${why:+$why; }swap-out $win MiB in $WINDOW_S s, above $KILL_SWAPOUT_MIB"
    if [ -n "$why" ]; then
      kill -9 "$pid" 2> /dev/null  # first, before anything that can wait on a disk
    fi
  else
    nu=0 nf=0 ns=0
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$now" "$avail" "$cma" "$usable" "$(((st - sf) / 1024))" "$win" "$some" \
    "$full" "${pid:--}" >&"$lf"
  if [ -n "$why" ]; then
    ev "KILL $PROC $pid: $why"
    dump "$why"
    [ -n "$PID" ] && break
    pid="" nu=0 nf=0 ns=0
  fi
  i=$((i + 1))
  read -rt 1 -u "$tick"
done
RAM_WATCHDOG
}
WD_SH=/dev/shm/ram_watchdog-$TAG.sh WD_TSV=/dev/shm/ram_watchdog-$TAG.tsv
write_watchdog > "$WD_SH"
rm -f "$WD_TSV"
if command -v chrt > /dev/null; then pre="chrt -f 10"; else pre="nice -n -20"; fi
PID=$srv LOG=$WD_TSV SERVER_LOG="$OUT/server.log" MIN_AVAIL_MB=${MIN_AVAIL_MB:-400} $pre bash "$WD_SH" > "$OUT/watchdog.out" 2>&1 < /dev/null &
wd=$!
wd_finish() {  # stop the watchdog and keep its TSV
  kill "$wd" 2> /dev/null
  for _ in 1 2 3 4 5 6; do kill -0 "$wd" 2> /dev/null || break; sleep 0.5; done
  kill -9 "$wd" 2> /dev/null
  [ -f "$WD_TSV" ] && cp "$WD_TSV" "$OUT/ram_watchdog.tsv"
  rm -f "$WD_TSV" "$WD_SH"
  [ -s "$OUT/watchdog.out" ] || rm -f "$OUT/watchdog.out"
}
for _ in 1 2 3 4 5 6 7 8 9 10; do grep -qs "^[0-9]" "$WD_TSV" && break; sleep 0.5; done
if ! grep -qs "^[0-9]" "$WD_TSV"; then
  echo "the RAM watchdog did not come up; it said:"; cat "$OUT/watchdog.out"; kill $srv; wd_finish; exit 1
fi

until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
    if ! kill -0 $srv 2> /dev/null; then echo "server exited during load:"; tail -20 "$OUT/server.log"; wd_finish; exit 1; fi
    sleep 2
done
load=$(echo "$(date +%s.%N) - $t0" | bc)
printf "load time: %.0f s\n" "$load" | tee "$OUT/summary.txt"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tee "$OUT/memory.csv" \
    | awk -F', ' '{gsub(/ MiB/,"",$2); s+=$2; printf "  GPU%s %s MiB\n", $1, $2} END {printf "  total %d MiB\n", s}' | tee -a "$OUT/summary.txt"

if [ "$PROMPTS" = chat ]; then
    python3 -u - "$PORT" "$MAX_TOKENS" "$OUT" "$MODEL" <<'EOF' | tee -a "$OUT/summary.txt"
import json, os, struct, sys, time, urllib.request
port, max_tokens, out, model = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]

def gguf_sampling(path):
    # general.sampling.* from the GGUF header (pure Python, header only)
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    st = lambda: f.read(rd("Q")).decode("utf-8", "replace")
    SZ = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    def val(t):
        if t == 8: return st()
        if t == 9:
            et, n = rd("I"), rd("Q")
            return [val(et) for _ in range(n)] if et in (8, 9) else [rd(SZ[et]) for _ in range(n)]
        return rd(SZ[t])
    if f.read(4) != b"GGUF": return {}
    rd("I"); rd("Q"); nkv = rd("Q"); out = {}
    for _ in range(nkv):
        k = st(); v = val(rd("I"))
        if k.startswith("general.sampling."): out[k.split(".")[-1]] = v
    return out

g = gguf_sampling(model)
sampling = {"temperature": float(os.environ.get("TEMP", g.get("temp", 0.8))),
            "top_p": float(os.environ.get("TOP_P", g.get("top_p", 0.95))),
            "top_k": int(os.environ.get("TOP_K", g.get("top_k", 40))),
            "min_p": float(os.environ.get("MIN_P", g.get("min_p", 0.0)))}
keep_reasoning = os.environ.get("KEEP_REASONING", "1") != "0"
print("sampling: " + ", ".join(f"{k} {v:g}" for k, v in sampling.items()) + f"; up to {max_tokens} tokens per reply; "
      + ("reasoning kept in the history" if keep_reasoning else "history without reasoning"))

turns = [
    "I have an old crypto-mining rig: nine NVIDIA P106 cards with 6 GB each, all on PCIe x1 risers, a 2-core Celeron "
    "and 4 GB of RAM. I'd like to run a local LLM on it for chatting and some coding help. What should I check or set "
    "up first?",
    "Write a bash script that prints each GPU's memory use, temperature and power draw every 5 seconds, and prints a "
    "warning line when a card goes above 80 °C.",
    "Why do the x1 risers matter so much when loading a model, but so little when generating text once it is loaded?",
    "Turn the checklist from your first answer into a short table with three columns: item, why it matters, how to "
    "check it.",
    "Thanks! Summarize the most important points of our conversation in three bullet points.",
]

def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))

messages, log = [], []
gen_n = gen_ms = new_n = new_ms = 0
t_chat = time.time()
for i, text in enumerate(turns, 1):
    messages.append({"role": "user", "content": text})
    t0 = time.time()
    r = post("/v1/chat/completions", {"messages": messages, "max_tokens": max_tokens, "seed": 1000 + i,
                                      "cache_prompt": True, **sampling})
    wall = time.time() - t0
    msg, t = r["choices"][0]["message"], r.get("timings", {})
    answer = msg.get("content") or ""
    n_answer = len(post("/tokenize", {"content": answer})["tokens"]) if answer else 0
    reasoning = msg.get("reasoning_content") or ""
    messages.append({"role": "assistant", "content": answer,
                     **({"reasoning_content": reasoning} if keep_reasoning and reasoning else {})})
    log.append({"turn": i, "wall_s": wall, "response": r})
    gen_n += t.get("predicted_n", 0); gen_ms += t.get("predicted_ms", 0)
    new_n += t.get("prompt_n", 0); new_ms += t.get("prompt_ms", 0)
    print(f"turn {i}: context {t.get('cache_n', 0) + t.get('prompt_n', 0):5d} tok, {t.get('prompt_n', 0):4d} new at "
          f"{t.get('prompt_per_second', 0):5.1f} t/s, first token after {t.get('prompt_ms', 0) / 1000:4.1f} s | generated "
          f"{t.get('predicted_n', 0):4d} tok ({n_answer} answer) at {t.get('predicted_per_second', 0):5.2f} t/s, "
          f"{r['choices'][0].get('finish_reason')} | {wall:5.1f} s")
json.dump({"sampling": sampling, "keep_reasoning": keep_reasoning, "messages": messages, "turns": log}, open(f"{out}/chat.json", "w"), indent=1)
print(f"conversation: {time.time() - t_chat:.0f} s; generated {gen_n} tokens at {1000 * gen_n / gen_ms:.2f} t/s; "
      f"{new_n} new prompt tokens at {1000 * new_n / new_ms:.1f} t/s, {new_ms / 1000:.1f} s waiting for first tokens")
EOF
elif [ "$PROMPTS" != none ]; then
    python3 - "$PORT" "$MAX_TOKENS" "$OUT" <<'EOF' | tee -a "$OUT/summary.txt"
import json, os, sys, urllib.request
port, max_tokens, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
prompts = {
    "code": "Write a Python function that merges overlapping intervals, with type hints and three unit tests.",
    "explain": "Explain to a physicist how the attention mechanism in a transformer works, in about 300 words.",
    "math": "A tank fills in 6 hours through pipe A and empties in 9 hours through pipe B. Both are open. How long until it is full? Show your reasoning.",
    "rewrite": "Rewrite this paragraph in plain English, keeping every fact: 'The P106-090 is a cut-down GP106 made for "
               "cryptocurrency mining. It has no display outputs and a narrow PCIe interface. The reference design has "
               "3 GB of memory, but this board carries 6 GB. Each card has five streaming multiprocessors and reads its "
               "memory at about 173 GB/s.'",
}
if os.environ.get("WARMUP", "0") != "0":  # one untimed request first: CUDA graph capture, first-request allocations
    body = json.dumps({"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 32, "temperature": 0,
                       "cache_prompt": False}).encode()
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                                  {"Content-Type": "application/json"}), timeout=1800).read()
tot_n = tot_ms = 0
for name, text in prompts.items():
    body = json.dumps({"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                       "temperature": 0, "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    t = r.get("timings", {})
    json.dump(r, open(f"{out}/{name}.json", "w"))
    acc = f" | draft accepted {t['draft_n_accepted']}/{t['draft_n']}" if t.get("draft_n") else ""
    print(f"{name:8s} prompt {t.get('prompt_n')} tok at {t.get('prompt_per_second', 0):6.1f} t/s | "
          f"generated {t.get('predicted_n')} tok at {t.get('predicted_per_second', 0):5.2f} t/s{acc}")
    tot_n += t.get("predicted_n", 0); tot_ms += t.get("predicted_ms", 0)
if tot_ms: print(f"overall generation: {tot_n} tokens at {1000 * tot_n / tot_ms:.2f} t/s")
EOF
fi

kill $srv; wait $srv 2> /dev/null
wd_finish
grep -E "model buffer size|KV buffer size|compute buffer size|graph nodes|graph splits|n_ctx|CUDA graphs|checkpoint|re-processing|backend sampling" "$OUT/server.log" \
    | sed 's/^[0-9.: ]*[IWD] //' | sort | uniq -c | sort -rn | head -40 > "$OUT/buffers.txt"
echo "== done; buffer sizes in $OUT/buffers.txt"
