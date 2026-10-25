"""Feed several llama-staircase-dump instances from one pool of answers (Q20/D1).

Every --every seconds, new complete lines of the generation files (read incrementally, so files may still grow) join
the pool unless their answer id is in an --exclude map file or already assigned. Then every instance whose queue
(answers in its map file that its log does not list as dumped) holds fewer than --low answer tokens gets the next pool
answers, up to --chunk answer tokens: appended to its map file, then to its sequence file, which the dump follows with
`tail -F`. Fast instances empty their queues sooner and so get more. Categories in --last go after all others (tool
calls: most of their tokens are prompt, which the dump replays without training on it).

A dump's log must be appended across restarts; it lists "sequence <id>: ..." for every finished sequence.

usage: python feed_split.py --gen 'DIR/gen-*.jsonl' --exclude map-d1.jsonl ... --inst NAME:SEQS:MAP:LOG [--inst ...]
                            [--every 120] [--low 300000] [--chunk 600000] [--last tool] [--max-tok 8192] [--stop FILE]
                            [--only IDS_FILE]
"""
import argparse
import collections
import glob
import json
import os
import time


def dumped_ids(log):
    if not os.path.exists(log):
        return set()
    return {int(line.split()[1].rstrip(":")) for line in open(log, errors="replace") if line.startswith("sequence ")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="glob of generation files (snapshots and finals are skipped)")
    ap.add_argument("--exclude", nargs="*", default=[], help="map files whose answers are dumped already")
    ap.add_argument("--inst", action="append", required=True, help="NAME:SEQS:MAP:LOG")
    ap.add_argument("--every", type=float, default=120)
    ap.add_argument("--low", type=int, default=300_000)
    ap.add_argument("--chunk", type=int, default=600_000)
    ap.add_argument("--last", nargs="*", default=["tool"])
    ap.add_argument("--max-tok", type=int, default=8192)
    ap.add_argument("--stop", default="")
    ap.add_argument("--only", help="file of answer ids (one per line): feed only these (e.g. licence-clean answers)")
    args = ap.parse_args()
    only = {x.strip() for x in open(args.only) if x.strip()} if args.only else None
    inst = [dict(zip(("name", "seqs", "map", "log"), x.split(":"))) for x in args.inst]
    taken = set()
    for path in args.exclude + [i["map"] for i in inst if os.path.exists(i["map"])]:
        taken |= {json.loads(line)["id"] for line in open(path)}
    offsets = collections.defaultdict(int)          # bytes of each generation file already read
    pool = (collections.deque(), collections.deque())   # (path, offset, length, id, n_gen): general, then --last
    while not (args.stop and os.path.exists(args.stop)):
        for path in sorted(glob.glob(args.gen)):
            if "snap" in path or "final" in path:
                continue
            with open(path, "rb") as f:
                f.seek(offsets[path])
                data = f.read()
            end = data.rfind(b"\n") + 1               # complete lines only
            pos = 0
            for line in data[:end].splitlines(keepends=True):
                off, pos = offsets[path] + pos, pos + len(line)
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                n_gen, n_all = len(r["gen_tokens"]), len(r["prompt_tokens"]) + len(r["gen_tokens"])
                if r["id"] in taken or n_gen < 2 or n_all > args.max_tok or (only is not None and r["id"] not in only):
                    continue
                taken.add(r["id"])
                pool[r["category"] in args.last].append((path, off, len(line), r["id"], n_gen))
            offsets[path] += end
        for i in inst:
            done = dumped_ids(i["log"])
            queued = 0
            if os.path.exists(i["map"]):
                for line in open(i["map"]):
                    r = json.loads(line)
                    queued += 0 if r["seq_id"] in done else r["n_gen"]
            if queued >= args.low:
                continue
            seqs, maps, added = [], [], 0
            while added < args.chunk and (pool[0] or pool[1]):
                path, off, length, rid, n_gen = (pool[0] or pool[1]).popleft()
                with open(path, "rb") as f:
                    f.seek(off)
                    r = json.loads(f.read(length))
                sid = int(rid.split("_")[-1])
                seqs.append(f"{sid} {len(r['prompt_tokens'])} " + " ".join(map(str, r["prompt_tokens"] + r["gen_tokens"])))
                maps.append(json.dumps({"seq_id": sid, "id": rid, "category": r["category"],
                                        "n_prompt": len(r["prompt_tokens"]), "n_gen": n_gen}))
                added += n_gen
            if seqs:
                with open(i["map"], "a") as f:
                    f.write("".join(x + "\n" for x in maps))
                with open(i["seqs"], "a") as f:
                    f.write("".join(x + "\n" for x in seqs))
                print(f"{time.strftime('%T')} [{i['name']}] queue {queued / 1e6:.2f} M -> +{len(seqs)} answers, "
                      f"{added / 1e6:.2f} M answer tokens; pool {len(pool[0])} + {len(pool[1])} later", flush=True)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
