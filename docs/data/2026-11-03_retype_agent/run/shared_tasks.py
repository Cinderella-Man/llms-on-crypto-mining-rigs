#!/usr/bin/env python3
"""Q218 on the rig: the arms of separate agent checks on the tasks every arm scored, with the speed at equal work.

Reads the attempts.csv that compare_refs.sh's report writes (bench/agent_check/report.py table's columns) and prints
a Markdown table, one row per arm, over the tasks every arm scored (with and without the tasks any arm ran away on):
tasks passed, minutes, tokens generated, decode seconds and tok/s, new prompt tokens, prefill seconds and tok/s, and
the rest (fixed cost per request, tools, the agent's own time). Sampling at temperature 1.0 makes the arms generate
different amounts per task, so the last column puts each arm's rates to the reference's work: the reference's new
prompt tokens at the arm's prompt tok/s, its generated tokens at the arm's generation tok/s, plus the reference's own
rest, against the reference's minutes.

Usage: python3 shared_tasks.py ATTEMPTS_CSV REF_ARM ARM [ARM ...]
"""
import csv
import math
import sys


def summary(rows):
    s = {k: sum(float(r[k]) for r in rows) for k in ("wall_s", "prefill_s", "decode_s", "output_tokens", "prompt_new")}
    s["rest_s"] = s["wall_s"] - s["prefill_s"] - s["decode_s"]
    s["passed"] = sum(int(r["pass"]) for r in rows)
    s["n"] = len(rows)
    return s


def table(by, arms, tasks, ref, title):
    print(f"\n{title}: {len(tasks)} tasks ({', '.join(t.replace('bugfix-', '') for t in tasks)})\n")
    print("| arm | passed | minutes | per task vs " + ref + " (geomean) | generated tokens | decode s | gen tok/s | "
          "new prompt tokens | prefill s | prompt tok/s | rest s | " + ref + "'s work at this arm's rates |")
    print("|---" * 12 + "|")
    sref = summary([by[(ref, t)] for t in tasks])
    for arm in arms:
        rows = [by[(arm, t)] for t in tasks]
        s = summary(rows)
        geo = math.exp(sum(math.log(float(by[(ref, t)]["wall_s"]) / float(by[(arm, t)]["wall_s"])) for t in tasks)
                       / len(tasks))
        gen = s["output_tokens"] / s["decode_s"]
        pp = s["prompt_new"] / s["prefill_s"]
        eq = sref["prompt_new"] / pp + sref["output_tokens"] / gen + sref["rest_s"]
        print(f"| {arm} | {s['passed']} / {s['n']} | {s['wall_s'] / 60:.1f} | {geo:.2f}× | {s['output_tokens']:.0f} | "
              f"{s['decode_s']:.0f} | {gen:.2f} | {s['prompt_new']:.0f} | {s['prefill_s']:.0f} | {pp:.1f} | "
              f"{s['rest_s']:.0f} | {eq / 60:.1f} min, {sref['wall_s'] / eq:.2f}× |")


def main():
    path, ref, arms = sys.argv[1], sys.argv[2], sys.argv[2:]
    rows = list(csv.DictReader(open(path)))
    by = {(r["file"], r["task_id"]): r for r in rows if r["file"] in arms}
    tasks = [t for t in dict.fromkeys(r["task_id"] for r in rows) if all((a, t) in by for a in arms)]
    table(by, arms, tasks, ref, "Tasks every arm scored")
    calm = [t for t in tasks if not any(int(by[(a, t)]["runaway"]) for a in arms)]
    table(by, arms, calm, ref, "The same without the tasks any arm ran away on")


if __name__ == "__main__":
    main()
