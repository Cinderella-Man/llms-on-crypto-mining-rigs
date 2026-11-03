#!/usr/bin/env python3
# fn_requests.py PORT MODE [SSD_DEV] [USB_DEV]: the single-request part of Q215 against a llama-server on this machine
# (the rig), so no prompt crosses the network. Prompts are Python standard-library source (PSF) from this machine.
#   speed: decode at short context, then fresh prompts of ~1k, ~8k and ~32k tokens, 128 tokens out each (prompt
#          tok/s at each size and decode at each context), a request that diverges from the 32k prompt near its end
#          (the server restores a context checkpoint and logs how long that took), and two identity prompts at T 0
#          whose output hash compares builds.
#   diag:  a short decode and an ~8k prompt with a diverging follow-up, for a server run with the kernel profiler
#          and trace logging (checkpoint creation times); its speeds are not results.
# Around every request it reads /proc/vmstat's major page faults and the read counters of the SSD and USB disks
# (/sys/block/<dev>/stat) at three points: before, at the first streamed token (end of prompt processing) and at the
# end, so prompt and decode reads are counted apart: the n-gram table is memory-mapped from the SSD and every token
# reads some of its rows. Prints one line per request and a JSON summary. Standard library only.
import glob, hashlib, json, sys, time, urllib.request

port, mode = sys.argv[1], sys.argv[2]
ssd, usb = (sys.argv[3] if len(sys.argv) > 3 else "sda"), (sys.argv[4] if len(sys.argv) > 4 else "sdb")
files = sorted(glob.glob("/usr/lib/python3*/*.py"))
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0, seed=1)  # the GGUF's general.sampling


def text(chars, skip):
    out, n = [], 0
    for f in files[skip:] + files[:skip]:
        t = open(f, errors="replace").read()
        out.append("# file " + f.split("/")[-1] + "\n" + t)
        n += len(t)
        if n >= chars:
            break
    return "\n".join(out)[:chars]


def counters():
    c = {}
    for line in open("/proc/vmstat"):
        k, v = line.split()
        if k in ("pgmajfault", "pgpgin"):
            c[k] = int(v)
    for dev in (ssd, usb):
        try:
            s = open(f"/sys/block/{dev}/stat").read().split()
            c[f"{dev}_reads"], c[f"{dev}_sectors"] = int(s[0]), int(s[2])
        except OSError:
            pass
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        if k in ("MemAvailable", "Cached"):
            c[k + "_kB"] = int(v.split()[0])
    return c


def delta(a, b):
    return {k: b[k] - a[k] for k in a if not k.endswith("_kB")}


def ask(content, n, **kw):
    """Streams one chat request; returns (server timings, content, client times, counters at three points)."""
    body = dict(messages=[{"role": "user", "content": content}], max_tokens=n, cache_prompt=True, stream=True,
                **kw)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    c0, t0 = counters(), time.time()
    c1 = t1 = None
    timings, out = {}, []
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            timings = ev.get("timings", timings)
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if piece and c1 is None:
                    c1, t1 = counters(), time.time()
                out.append(piece)
    c2, t2 = counters(), time.time()
    c1, t1 = (c1, t1) if c1 else (c2, t2)
    return timings, "".join(out), (t1 - t0, t2 - t1), (c0, c1, c2)


def report(name, res):
    t, out, (ttft, dec), (c0, c1, c2) = res
    pre, gen = delta(c0, c1), delta(c1, c2)
    n_gen = max(t.get("predicted_n", 0), 1)
    row = dict(name=name, prompt_n=t.get("prompt_n"), cache_n=t.get("cache_n"),
               prompt_tps=round(t.get("prompt_per_second", 0), 2), prompt_ms=round(t.get("prompt_ms", 0)),
               predicted_n=t.get("predicted_n"), decode_tps=round(t.get("predicted_per_second", 0), 2),
               ttft_s=round(ttft, 2), decode_wall_s=round(dec, 2),
               prompt_majfault=pre.get("pgmajfault"), prompt_ssd_reads=pre.get(f"{ssd}_reads"),
               prompt_ssd_mib=round(pre.get(f"{ssd}_sectors", 0) * 512 / 2**20, 1),
               decode_majfault_per_token=round(gen.get("pgmajfault", 0) / n_gen, 2),
               decode_ssd_reads_per_token=round(gen.get(f"{ssd}_reads", 0) / n_gen, 2),
               decode_ssd_kib_per_token=round(gen.get(f"{ssd}_sectors", 0) * 512 / 1024 / n_gen, 1),
               usb_reads=pre.get(f"{usb}_reads", 0) + gen.get(f"{usb}_reads", 0),
               mem_available_mib=c2.get("MemAvailable_kB", 0) // 1024, cached_mib=c2.get("Cached_kB", 0) // 1024)
    print(f"{name:10s} prompt {row['prompt_n']} (+{row['cache_n']} cached) at {row['prompt_tps']} tok/s, first token "
          f"{row['ttft_s']} s; decode {row['predicted_n']} at {row['decode_tps']} tok/s; SSD reads: prompt "
          f"{row['prompt_ssd_reads']} ({row['prompt_ssd_mib']} MiB), decode {row['decode_ssd_reads_per_token']}/token "
          f"({row['decode_ssd_kib_per_token']} KiB), major faults/token {row['decode_majfault_per_token']}; "
          f"MemAvailable {row['mem_available_mib']} MiB", flush=True)
    return row


q = "\n\nIn one sentence: which of these files is the longest?"
rows = []
if mode == "speed":
    rows.append(report("short", ask("Write a Python function that checks whether a string is a palindrome.", 128,
                                    **SAMPLING)))
    for name, chars, skip in (("p1k", 4000, 10), ("p8k", 32000, 40), ("p32k", 128000, 80)):
        rows.append(report(name, ask(text(chars, skip) + q, 128, **SAMPLING)))
    # diverge near the end of the 32k prompt: the server restores the last checkpoint before the change
    rows.append(report("restore32k", ask(text(128000, 80)[:-6000] + "\n\nList the first three file names.", 64,
                                         **SAMPLING)))
    src = open(files[3], errors="replace").read()[:3000]
    for name, p in (("chat", "Explain in a short paragraph how a hash map handles collisions, then give a tiny "
                             "Python example."),
                    ("rewrite", "Rewrite this code with every function name in camelCase, changing nothing else:\n\n"
                                + src)):
        t, out, _times, _c = res = ask(p, 192, temperature=0)
        row = report("ident-" + name, res)
        row["md5"] = hashlib.md5(out.encode()).hexdigest()[:12]
        print(f"ident {name}: {row['md5']} | {out[:70]!r}", flush=True)
        rows.append(row)
else:
    rows.append(report("short", ask("Write a Python function that checks whether a string is a palindrome.", 64,
                                    **SAMPLING)))
    rows.append(report("p8k", ask(text(32000, 40) + q, 16, **SAMPLING)))
    rows.append(report("restore8k", ask(text(32000, 40)[:-3000] + "\n\nList the first three file names.", 16,
                                        **SAMPLING)))
print("JSON " + json.dumps(rows))
