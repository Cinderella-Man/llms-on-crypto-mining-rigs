"""BFCL v4 multi-turn for Flint: the 800 tasks of base, miss_func, miss_param and long_context (200 scenarios each),
run as conversations against llama-server and scored with BFCL's own executor and state/response checker.

The conversation loop is BFCL's inference_multi_turn_FC (base_handler.py) with Flint in place of an API:
- each step renders the whole history with Flint's chat template (tools as an OpenAI client sends them, previous
  assistant turns with their reasoning_content, tool results as <|observation|> blocks) and samples one assistant
  turn on /completion with the makers' settings;
- the turn's calls are parsed with rec's parser and executed on BFCL's Python backends; a turn ends when the model
  answers without a call or its calls cannot be decoded; more than 20 steps in one turn ends the task (failed);
- miss_func turns add the held-out functions and BFCL's "I have updated some more functions" message.
Scoring re-decodes the stored raw text, as BFCL's eval_runner does, so a parser change needs no new generation.

usage:
  python multi_turn.py run URL TEMPLATE.jinja OUT.jsonl [--attempts 1-5] [--conc 8] [--n-predict 4096]
                          [--scenarios N [--seed S] | --ids FILE] [--shuffle SEED] [--part I/N]
  python multi_turn.py score OUT_DIR FILE=RAW.jsonl [FILE=RAW.jsonl ...]
  python multi_turn.py selftest TEMPLATE.jinja      # ground truth replayed through the loop; all 800 must pass
  python multi_turn.py render TEMPLATE.jinja OUT.jsonl FILE=RAW.jsonl [...] [--min-tokens N]
      # stored conversations as the token stream the model saw (long-context KLD material, Q106)
"""
import argparse
import ast
import copy
import hashlib
import inspect
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import bfcl_common as bc
from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING  # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils  # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker  # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (  # noqa: E402
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from score import write  # noqa: E402

EXTRA = ["n_steps", "force_quit", "max_context"]
EXCLUDED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "excluded_multi_turn.txt")


def read_ids(path):
    return {line.split()[0] for line in open(path) if line.strip() and not line.startswith("#")} if path else set()


def load_entries(ids=None, exclude=()):
    entries = []
    for cat in bc.MULTI_TURN:
        gt = {g["id"]: g["ground_truth"] for g in bc.load_ground_truth_entry(cat)}
        for e in bc.load_dataset_entry(cat):
            if (ids is None or e["id"] in ids) and e["id"] not in exclude:
                e["ground_truth"] = gt[e["id"]]
                entries.append(e)
    return entries


def tools_at(entry, turn):
    """The tool list offered at a turn: the base functions plus every held-out function released so far."""
    functions = list(entry["function"])
    for t, held in entry.get("missed_function", {}).items():
        if int(t) <= turn:
            functions += held
    return bc.build_tools(functions)


def is_long(entry):
    return "long_context" in entry["id"].rsplit("_", 1)[0]


def drop_instances(key, entry_id):
    """Backend instances live as globals of multi_turn_utils, keyed by model name and task; free them."""
    for suffix in ("", "_eval", "_ground_truth_eval"):
        prefix = re.sub(r"[-./:]", "_", f"{key}{suffix}_{entry_id}_")
        for name in [n for n in vars(multi_turn_utils) if n.startswith(prefix)]:
            del vars(multi_turn_utils)[name]


def converse(entry, tmpl, generate, key):
    """One attempt at one task. generate(prompt, messages, tools) returns a step record with the raw "content"
    (or an "error"). Returns the raw steps per turn and whether the task was force-terminated."""
    e = copy.deepcopy(entry)
    execute_multi_turn_func_call([], e["initial_config"], e["involved_classes"], key, e["id"],
                                 long_context=is_long(e))
    messages, turns, force_quit = [], [], False
    try:
        for ti, turn_messages in enumerate(e["question"]):
            tools = tools_at(e, ti)
            if str(ti) in e.get("missed_function", {}):
                assert not turn_messages, e["id"]
                turn_messages = [{"role": "user", "content": bc.DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC}]
            messages += turn_messages
            steps = []
            while True:
                prompt = tmpl.render(messages=messages, tools=tools, add_generation_prompt=True)
                step = generate(prompt, messages, tools)
                steps.append(step)
                if step.get("error"):
                    force_quit = True
                    break
                parsed = bc.split_output(step["content"], tools)
                messages.append(bc.assistant_message(parsed))
                try:
                    calls = bc.convert_to_function_call(bc.decode_ast(parsed))
                except ValueError:
                    break
                if is_empty_execute_response(calls):
                    break
                results, _ = execute_multi_turn_func_call(calls, e["initial_config"], e["involved_classes"], key,
                                                          e["id"], long_context=is_long(e))
                messages += [{"role": "tool", "content": r} for r in results]
                if len(steps) > bc.MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    break
            turns.append(steps)
            if force_quit:
                break
    finally:
        drop_instances(key, e["id"])
    return {"turns": turns, "force_quit": force_quit}


def evaluate(entry, result, key="flint_score"):
    """BFCL's verdict for one attempt, with the process flags summed over its steps."""
    flags = {"malformed": 0, "runaway": 0, "undecodable": 0, "unknown_function": 0, "unknown_arg": 0,
             "call_in_think": 0, "extra_think_close": 0, "n_calls": 0, "reasoning_chars": 0}
    decoded = []
    for ti, steps in enumerate(result["turns"]):
        tools = tools_at(entry, ti)
        turn = []
        for st in steps:
            flags["runaway"] |= st.get("stop_type") == "limit"
            if st.get("error"):
                continue
            p = bc.split_output(st["content"], tools)
            flags["malformed"] |= bool(p["malformed"] or p["unknown_function"] or p["unknown_arg"]
                                       or p["call_in_think"])
            for k in ("undecodable", "unknown_function", "unknown_arg", "call_in_think", "extra_think_close"):
                flags[k] += int(p["malformed"] if k == "undecodable" else p[k])
            flags["n_calls"] += len(p["calls"])
            flags["reasoning_chars"] += len(p["reasoning"])
            try:
                calls = bc.convert_to_function_call(bc.decode_ast(p))
            except ValueError:
                continue
            if not is_empty_execute_response(calls):
                turn.append(calls)
        decoded.append(turn)
    gt = entry["ground_truth"]
    if result["force_quit"] or len(decoded) != len(gt):
        return False, "multi_turn:force_terminated", flags
    try:
        r = multi_turn_checker(decoded, gt, copy.deepcopy(entry), entry["id"].rsplit("_", 1)[0], key)
    finally:
        drop_instances(key, entry["id"])
    return bool(r["valid"]), r.get("error_type"), flags


# ---------------------------------------------------------------- run
def cmd_run(args):
    tmpl = bc.make_template(open(args.template).read())
    ids = read_ids(args.ids) or None
    if args.scenarios:  # the same N scenarios in all four categories (pilots)
        picked = random.Random(args.seed).sample(range(200), args.scenarios)
        ids = {f"{cat}_{n}" for cat in bc.MULTI_TURN for n in picked}
    entries = load_entries(ids, read_ids(args.exclude))
    if args.shuffle is not None:  # a seeded random order, sliced: see run.py
        random.Random(args.shuffle).shuffle(entries)
    i, n = map(int, args.part.split("/"))
    entries = entries[(i - 1) * len(entries) // n:i * len(entries) // n]
    attempts = bc.parse_attempts(args.attempts)
    done = set()
    try:
        done = {(r["id"], r["attempt"]) for r in map(json.loads, open(args.out))}
    except FileNotFoundError:
        pass
    todo = [(e, a) for a in attempts for e in entries if (e["id"], a) not in done]
    slot_ctx = bc.server_props(args.url)["n_ctx_slot"]
    json.dump(bc.manifest(args.url, {
        "lane": "bfcl_multi_turn", "tasks_sha256": bc.sha256_of(e["id"] for e in entries), "n_tasks": len(entries),
        "template_sha256": hashlib.sha256(open(args.template, "rb").read()).hexdigest(), "attempts": attempts,
        "seed_offset": args.seed_offset, "n_predict": args.n_predict, "cache_prompt": not args.no_cache_prompt,
        "max_steps_per_turn": bc.MAXIMUM_STEP_LIMIT, "shuffle": args.shuffle, "part": args.part}),
        open(args.out + ".manifest.json", "w"), indent=1)
    print(f"slot n_ctx={slot_ctx}, todo={len(todo)}/{len(entries) * len(attempts)}", flush=True)
    lock = threading.Lock()
    out = open(args.out, "a")

    def one(item):
        e, a = item
        seed = a + args.seed_offset

        def generate(prompt, messages, tools):
            n_prompt = bc.count_tokens(args.url, prompt)
            n_predict = min(args.n_predict, slot_ctx - n_prompt - 8)
            if n_predict < 256:
                return {"error": "prompt too long", "n_prompt": n_prompt, "content": ""}
            return {"n_prompt": n_prompt, **bc.complete(args.url, prompt, n_predict, seed, not args.no_cache_prompt)}

        res = converse(e, tmpl, generate, f"flint_a{a}")
        rec = {"id": e["id"], "attempt": a, "seed": seed, **res}
        with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
        return rec

    n = 0
    with ThreadPoolExecutor(args.conc) as ex:
        for rec in ex.map(one, todo):
            n += 1
            print(f"{n}/{len(todo)} {rec['id']} attempt={rec['attempt']} steps="
                  f"{sum(len(t) for t in rec['turns'])} force_quit={rec['force_quit']}", flush=True)
    print("DONE", args.out, flush=True)


# ---------------------------------------------------------------- render
def cmd_render(args):
    """Each stored attempt replayed through converse() with its recorded outputs, so that BFCL's backends regenerate
    the tool results: the last step's prompt (the whole history with every reasoning block, as the model saw it) and
    the last step's output. n_prompt is the server's token count of that prompt, for the caller to check the
    rendering against."""
    tmpl = bc.make_template(open(args.template).read())
    entries = {e["id"]: e for e in load_entries(exclude=read_ids(args.exclude))}
    out = open(args.out, "w")
    n = 0
    for spec in args.raw:
        name, path = spec.split("=", 1)
        for line in open(path):
            rec = json.loads(line)
            steps = [s for t in rec["turns"] for s in t]
            if rec["id"] not in entries or not steps or any(s.get("error") for s in steps):
                continue
            if steps[-1]["n_prompt"] + steps[-1]["tokens_predicted"] < args.min_tokens:
                continue
            stored, prompts = iter(steps), []

            def generate(prompt, messages, tools):
                prompts.append(prompt)
                return dict(next(stored))  # StopIteration: the replay took a step the run did not

            res = converse(entries[rec["id"]], tmpl, generate, f"render_{name}_a{rec['attempt']}")
            if next(stored, None) is not None or [len(t) for t in res["turns"]] != [len(t) for t in rec["turns"]]:
                sys.exit(f"{name} {rec['id']} attempt {rec['attempt']}: the replay does not follow the stored steps")
            out.write(json.dumps({"file": name, "id": rec["id"], "attempt": rec["attempt"],
                                  "n_steps": len(steps), "n_prompt": steps[-1]["n_prompt"],
                                  "tokens_predicted": steps[-1]["tokens_predicted"],
                                  "prompt": prompts[-1], "output": steps[-1]["content"]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} conversations rendered to {args.out}")


# ---------------------------------------------------------------- score
def row(entry, file, rec):
    ok, err, flags = evaluate(entry, rec)
    steps = [s for t in rec["turns"] for s in t]
    return {"lane": "bfcl_multi_turn", "task_id": entry["id"], "cluster": entry["id"].rsplit("_", 1)[1],
            "category": entry["id"].rsplit("_", 1)[0], "file": file, "attempt": rec["attempt"], "seed": rec["seed"],
            "pass": int(ok), "malformed": int(flags["malformed"]), "runaway": int(flags["runaway"]),
            "error_type": err or "", "prompt_tokens": sum(s.get("tokens_evaluated") or 0 for s in steps),
            "output_tokens": sum(s.get("tokens_predicted") or 0 for s in steps),
            "reasoning_chars": flags["reasoning_chars"], "stop_type": steps[-1].get("stop_type") or "" if steps else "",
            "n_calls": flags["n_calls"], **{k: flags[k] for k in ("undecodable", "unknown_function", "unknown_arg",
                                                                  "call_in_think", "extra_think_close")},
            "n_steps": len(steps), "force_quit": int(rec["force_quit"]),
            "max_context": max(((s.get("n_prompt") or 0) + (s.get("tokens_predicted") or 0) for s in steps),
                               default=0)}


def cmd_score(args):
    entries = {e["id"]: e for e in load_entries(exclude=read_ids(args.exclude))}
    rows = []
    for spec in args.raw:
        file, path = spec.split("=", 1)
        for rec in map(json.loads, open(path)):
            if rec["id"] in entries:
                rows.append(row(entries[rec["id"]], file, rec))
    write(args.out_dir, rows, order=bc.MULTI_TURN, extra=EXTRA)


# ---------------------------------------------------------------- selftest
def gt_call(entry, call):
    """A ground-truth call string such as "sort('final_report.pdf')" as a tool call with named arguments
    (positional arguments are named from the backend method's signature)."""
    node = ast.parse(call, mode="eval").body
    name = node.func.id
    args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    if node.args:
        for cls in entry["involved_classes"]:
            module = __import__(CLASS_FILE_PATH_MAPPING[cls], fromlist=[cls])
            method = getattr(getattr(module, cls), name, None)
            if method is not None:
                params = [p for p in inspect.signature(method).parameters if p != "self"]
                break
        args = {**{p: ast.literal_eval(v) for p, v in zip(params, node.args)}, **args}
    return {"type": "function", "function": {"name": name, "arguments": args}}


def cmd_selftest(args):
    """Replay every task's ground truth through converse(): in each turn the 'model' makes the turn's ground-truth
    calls one per step (several per step when a turn has more than 20), written by Flint's template with a
    reasoning block, then answers in plain text. A task passes if BFCL's checker accepts it and no process flag
    fires; the tasks in excluded_multi_turn.txt are expected to fail."""
    tmpl = bc.make_template(open(args.template).read())
    entries = load_entries()
    fails, longest = [], (0, None)
    for e in entries:
        plan = []
        for turn in e["ground_truth"]:
            calls = [gt_call(e, c) for c in turn]
            k = -(-len(calls) // bc.MAXIMUM_STEP_LIMIT) or 1
            plan.append([calls[i:i + k] for i in range(0, len(calls), k)])
        state = {"turn": -1, "step": 0}

        def generate(prompt, messages, tools):
            # a new turn starts when the last message is a user message
            if messages[-1]["role"] == "user":
                state["turn"] += 1
                state["step"] = 0
            todo = plan[state["turn"]]
            msg = {"role": "assistant", "reasoning_content": f"Self-test step {state['step']}.", "content": ""}
            if state["step"] < len(todo):
                msg["tool_calls"] = todo[state["step"]]
            else:
                msg["content"] = "Done."
            state["step"] += 1
            full = tmpl.render(messages=messages + [msg], tools=tools, add_generation_prompt=False)
            assert full.startswith(prompt), e["id"]
            return {"content": full[len(prompt):], "stop_type": "eos", "n_prompt": len(prompt)}

        res = converse(e, tmpl, generate, "flint_selftest")
        ok, err, flags = evaluate(e, {**res, "attempt": 1, "seed": 0})
        ok = ok and not flags["malformed"]
        chars = max(s["n_prompt"] for t in res["turns"] for s in t)
        longest = max(longest, (chars, e["id"]))
        if not ok:
            fails.append(e["id"])
            print("FAIL", e["id"], err, flags)
    expected = read_ids(EXCLUDED)
    print(f"multi-turn selftest: {len(entries) - len(fails)} pass, {len(fails)} fail "
          f"({len(set(fails) & expected)} of them listed in excluded_multi_turn.txt); "
          f"longest prompt {longest[0]:,} characters ({longest[1]})")
    sys.exit(0 if set(fails) == expected else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("url")
    r.add_argument("template")
    r.add_argument("out")
    r.add_argument("--attempts", default="1-5")
    r.add_argument("--seed-offset", type=int, default=0)
    r.add_argument("--conc", type=int, default=8, help="conversations in flight; match llama-server's -np")
    r.add_argument("--n-predict", type=int, default=4096)
    r.add_argument("--no-cache-prompt", action="store_true")
    r.add_argument("--ids", help="file with the task IDs to run (one per line)")
    r.add_argument("--scenarios", type=int, default=0, help="a seeded sample of N of the 200 scenarios")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--shuffle", type=int)
    r.add_argument("--part", default="1/1")
    r.add_argument("--exclude", default=EXCLUDED)
    s = sub.add_parser("score")
    s.add_argument("out_dir")
    s.add_argument("raw", nargs="+", help="FILE=RAW.jsonl")
    s.add_argument("--exclude", default=EXCLUDED)
    t = sub.add_parser("selftest")
    t.add_argument("template")
    d = sub.add_parser("render")
    d.add_argument("template")
    d.add_argument("out")
    d.add_argument("raw", nargs="+", help="FILE=RAW.jsonl")
    d.add_argument("--min-tokens", type=int, default=0, help="only conversations at least this long (prompt + output)")
    d.add_argument("--exclude", default=EXCLUDED)
    args = ap.parse_args()
    {"run": cmd_run, "score": cmd_score, "selftest": cmd_selftest, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
