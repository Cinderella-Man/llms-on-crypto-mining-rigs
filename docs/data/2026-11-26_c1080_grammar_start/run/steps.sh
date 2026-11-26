#!/usr/bin/env bash
# The tool-call grammar's state at every token of each chat reply run.sh recorded, with grammar-steps (run/build.sh).
# For each build folder in WORK and each reply of a chat arm with a grammar (chat, chat2): the slot's parameters from the
# server's verbose log (the grammar, lazy or not, its trigger tokens, the generation prompt), the reply's token ids
# from the response, then grammar-steps over them in the build's own image (only the vocabulary is loaded; a card is
# requested because the tool is linked with CUDA). Writes <build>/<request>.<arm>.steps.tsv and $WORK/steps.tsv (one
# line per reply).
# The server prints a slot's parameters when the slot is launched for the next request ("launching slot : {...}",
# the state it leaves), so the parameters of request k are in the k+1-th such line; the script checks that the text
# the line says was generated is the response's text.
# Env: WORK (run.sh's output folder), TOOL (the grammar-steps binary), MODEL (the GGUF whose vocabulary the server
#      read), IMG (the image the tool was built in), MOUNTS (folders mounted read-only at the same paths),
#      DOCKER_ARGS (labels, --cpu-shares), GPU (a card for the container, default 0)
set -uo pipefail
: "${WORK:?}" "${TOOL:?}" "${MODEL:?}"
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPU=${GPU:-0}
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$WORK:$WORK")

for o in "$WORK"/*/; do
    o=${o%/}
    [ -f "$o/order.txt" ] && [ -f "$o/server.log" ] || continue
    python3 -I - "$o" <<'PY' || exit 1
import json, os, sys
o = sys.argv[1]
order = [l.split() for l in open(f"{o}/order.txt") if l.strip()]
slots = []
for line in open(f"{o}/server.log", errors="replace"):
    k = line.find("launching slot : ")
    if k < 0:
        continue
    j = json.loads(line[k + len("launching slot : "):])
    slots.append(j)
# slots[k + 1] holds request k's parameters (the first line is the idle slot)
jobs = []
for k, (n, arm) in enumerate(order):
    if k + 1 >= len(slots) or "params" not in slots[k + 1]:
        if arm in ("chat", "chat2"):
            print(f"{o}: no parameters logged for request {k} ({n} {arm})"); sys.exit(1)
        continue
    s = slots[k + 1]
    p = s["params"]
    r = json.load(open(f"{o}/{n}.{arm}.json"))
    t = r["timings"]
    # the line belongs to this request: its grammar is there exactly for the arms with tools, and the slot holds
    # this request's prompt and reply (but the last sampled token, when the reply did not end on an end of turn)
    if bool(p.get("grammar")) != (arm in ("chat", "chat2")) or \
            not 0 <= t["prompt_n"] + t["cache_n"] + t["predicted_n"] - s.get("n_prompt_tokens", -9) <= 1:
        print(f"{o}: request {k} ({n} {arm}): the logged slot is not this request's"); sys.exit(1)
    if arm not in ("chat", "chat2"):
        continue
    v = r["__verbose"]
    base = f"{o}/{n}.{arm}"
    open(base + ".grammar.txt", "w").write(p["grammar"])
    open(base + ".genprompt.txt", "w").write(p.get("generation_prompt", ""))
    open(base + ".tokens.txt", "w").write("".join(f"{t}\n" for t in v["tokens"]))
    trig = [str(t["token"]) for t in p.get("grammar_triggers", []) if "token" in t]
    other = [t for t in p.get("grammar_triggers", []) if "token" not in t]
    if other:
        print(f"{o}: request {k}: triggers that are not tokens, not handled: {other}"); sys.exit(1)
    lazy = "lazy " + " ".join(trig) if p.get("grammar_lazy") else ""
    jobs.append(f"{base} {lazy}".strip())
    json.dump({"grammar_lazy": p.get("grammar_lazy"), "grammar_triggers": p.get("grammar_triggers"),
               "preserved_tokens": p.get("preserved_tokens"), "generation_prompt": p.get("generation_prompt"),
               "chat_format": p.get("chat_format"), "backend_sampling": p.get("backend_sampling")},
              open(base + ".params.json", "w"), indent=1)
open(f"{o}/steps.jobs", "w").write("".join(j + "\n" for j in jobs))
PY
    while read -r base lazy; do
        # shellcheck disable=SC2086
        docker run --rm ${DOCKER_ARGS:-} --network none --gpus "\"device=$GPU\"" "${VOL[@]}" --entrypoint "" "$IMG" \
            timeout 1200 "$TOOL" "$MODEL" "$base.grammar.txt" "$base.genprompt.txt" "$base.tokens.txt" $lazy \
            > "$base.steps.tsv" 2> "$base.steps.err"
        echo "$(basename "$o") $(basename "$base") rc=$? $(tail -n 1 "$base.steps.tsv")"
    done < "$o/steps.jobs"
done | tee "$WORK/steps.log"

# one line per reply: where the first <tool_call> token is, and what the grammar allowed before and after it
python3 -I - "$WORK" <<'PY'
import csv, glob, json, os, sys
w = sys.argv[1]
out = open(f"{w}/steps.tsv", "w")
out.write("build\trequest\tarm\tlazy\ttokens\tfirst_call_at\tbefore_call_whole_vocab\tbefore_call_narrowed\t"
          "after_call_whole_vocab\tafter_call_narrowed\trejected\n")
for f in sorted(glob.glob(f"{w}/*/*.steps.tsv")):
    build = os.path.basename(os.path.dirname(f))
    req, arm = os.path.basename(f)[:-len(".steps.tsv")].rsplit(".", 1)
    lazy = json.load(open(f[:-len(".steps.tsv")] + ".params.json"))["grammar_lazy"]
    rows = [r for r in csv.DictReader((l for l in open(f, errors="replace") if not l.startswith("#")), delimiter="\t",
                                      quoting=csv.QUOTE_NONE)]
    call = next((i for i, r in enumerate(rows) if r["piece"] == "<tool_call>"), len(rows))
    b, a = rows[:call], rows[call:]
    whole = lambda rs: sum(r["n_allowed"] == r["n_vocab"] for r in rs)
    out.write(f"{build}\t{req}\t{arm}\t{int(bool(lazy))}\t{len(rows)}\t{call if call < len(rows) else 'none'}\t"
              f"{whole(b)}\t{len(b) - whole(b)}\t{whole(a)}\t{len(a) - whole(a)}\t"
              f"{sum(r['allowed'] == '0' for r in rows)}\n")
out.close()
print(open(f"{w}/steps.tsv").read())
PY
