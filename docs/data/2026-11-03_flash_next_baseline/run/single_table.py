#!/usr/bin/env python3
# single_table.py OUT.csv LINE.out...: fn_line.sh's outputs as one table. One row per engine line and request: load
# seconds, memory used per card after the load (MiB, nvidia-smi), prompt and cached tokens, prompt tok/s, time to first
# token at the client (s), generated tokens and decode tok/s (llama-server's own timings), SSD reads and MiB during
# prompt processing, SSD reads, KiB and major page faults per generated token, and MemAvailable after the request.
# The identity rows carry the first 12 hex digits of the output's MD5. Standard library only.
import csv, json, re, sys

FIELDS = ["engine", "request", "load_s", "cards_mib", "prompt_n", "cache_n", "prompt_tps", "ttft_s", "predicted_n",
          "decode_tps", "prompt_ssd_reads", "prompt_ssd_mib", "decode_ssd_reads_per_token", "decode_ssd_kib_per_token",
          "decode_majfault_per_token", "mem_available_mib", "ident_md5"]


def main():
    out, rows = sys.argv[1], []
    for path in sys.argv[2:]:
        engine, load, cards = "", "", ""
        for line in open(path, encoding="utf-8", errors="replace"):
            m = re.match(r"== (\S+): (\S+) ", line)
            if m and not line.startswith("== end"):
                engine = f"{m.group(1)} ({m.group(2)})"
            m = re.match(r"load (\d+) s", line)
            if m:
                load = m.group(1)
            if line.startswith("cards: "):
                cards = "/".join(x.split(",")[1].strip().split()[0] for x in line[7:].strip().strip(";").split(";"))
            if line.startswith("JSON "):
                for r in json.loads(line[5:]):
                    rows.append({"engine": engine, "request": r["name"], "load_s": load, "cards_mib": cards,
                                 "prompt_n": r["prompt_n"], "cache_n": r["cache_n"], "prompt_tps": r["prompt_tps"],
                                 "ttft_s": r["ttft_s"], "predicted_n": r["predicted_n"], "decode_tps": r["decode_tps"],
                                 "prompt_ssd_reads": r["prompt_ssd_reads"], "prompt_ssd_mib": r["prompt_ssd_mib"],
                                 "decode_ssd_reads_per_token": r["decode_ssd_reads_per_token"],
                                 "decode_ssd_kib_per_token": r["decode_ssd_kib_per_token"],
                                 "decode_majfault_per_token": r["decode_majfault_per_token"],
                                 "mem_available_mib": r["mem_available_mib"], "ident_md5": r.get("md5", "")})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
