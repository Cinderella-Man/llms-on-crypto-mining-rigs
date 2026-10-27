"""How much of Flint's generated text a shortlist of the head's draft vocabulary covers: the share of generated tokens
whose draft id (vocab.npy is sorted by frequency, so draft id = rank) is below K, per K, over the sessions listed in
--ids. A head whose output layer keeps only its first K draft ids (the engine's LLAMA_DRAFT_VOCAB=K) can guess no other
token, so 1 - coverage bounds what the shortlist costs in acceptance (the rare tokens are also the ones the head guesses
worst, so the real cost is lower).

usage: python shortlist_coverage.py vocab.npy sessions_seq.jsonl [--ids test.txt] [--k 1024,2048,4096,8192,16384]

Needs no numpy: vocab.npy is read as a plain int64 array.
"""
import argparse
import ast
import json
import struct


def read_npy_i64(path):
    with open(path, "rb") as f:
        assert f.read(6) == b"\x93NUMPY"
        major = f.read(2)[0]
        hlen = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(f.read(hlen).decode("latin1"))
        assert header["descr"] in ("<i8", "<i4") and not header["fortran_order"]
        fmt = "q" if header["descr"] == "<i8" else "i"
        data = f.read()
        return list(struct.unpack("<%d%s" % (len(data) // struct.calcsize(fmt), fmt), data))


def generated(r):
    if "gen_ranges" not in r:
        return r["gen_tokens"]
    seq = r["prompt_tokens"] + r["gen_tokens"]
    return [t for s, e in r["gen_ranges"] for t in seq[s:e]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vocab")
    ap.add_argument("sessions")
    ap.add_argument("--ids", help="file of session ids (one per line), e.g. the held-out list")
    ap.add_argument("--k", default="1024,2048,4096,8192,12288,16384")
    args = ap.parse_args()
    rank = {t: i for i, t in enumerate(read_npy_i64(args.vocab))}
    keep = {x.strip() for x in open(args.ids) if x.strip()} if args.ids else None
    ks = [int(x) for x in args.k.split(",")]
    n, n_in, hits = 0, 0, [0] * len(ks)
    n_rows = 0
    for line in open(args.sessions):
        r = json.loads(line)
        if keep is not None and str(r.get("id")) not in keep:
            continue
        n_rows += 1
        for t in generated(r):
            n += 1
            d = rank.get(t)
            if d is None:
                continue
            n_in += 1
            for j, k in enumerate(ks):
                hits[j] += d < k
    print(f"{n_rows} sessions, {n} generated tokens; in the draft vocabulary ({len(rank)} ids): {n_in / max(n, 1):.4f}")
    for k, h in zip(ks, hits):
        print(f"first {k:6d} draft ids: {h / max(n, 1):.4f} of generated tokens")


main()
