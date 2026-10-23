"""Download LiveCodeBench code_generation_lite (release_v6, pinned revision) and keep only the problems in a date
window, as they are (hidden tests stay compressed). The files are streamed, so only the kept rows reach the disk:
454 problems and 1.9 GB for the default window (12 from test3.jsonl, 100 from test4, all 167 of test5 and 175 of
test6; test.jsonl and test2.jsonl end before 2024-08).

usage: python -I fetch.py OUT.jsonl [--start 2024-08-01] [--end 2025-05-01]
The window is start <= contest_date < end. Run it outside the repository; the data must not be committed.
"""
import argparse
import json
import urllib.request

REPO = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve"
REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"  # release_v6, last modified 2025-06-05
FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--start", default="2024-08-01")
ap.add_argument("--end", default="2025-05-01")
ap.add_argument("--files", default=",".join(FILES[2:]))
args = ap.parse_args()

kept = 0
with open(args.out, "w") as out:
    for name in args.files.split(","):
        n = 0
        with urllib.request.urlopen(f"{REPO}/{REVISION}/{name}") as f:
            for line in f:
                row = json.loads(line)
                n += 1
                if args.start <= row["contest_date"][:10] < args.end:
                    out.write(line.decode())
                    kept += 1
        print(f"{name}: {n} problems read, {kept} kept so far", flush=True)
print(f"{kept} problems -> {args.out}")
