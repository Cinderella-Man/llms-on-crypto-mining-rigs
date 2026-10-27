#!/usr/bin/env python3
"""Byte and bit-width tallies behind docs/research/notes/slq.md. Standard library only; writes TSV files.

  svg OUT LABEL=FILE...      The precision maps of kaitchup/Qwen3.8-27B-SLQ-GGUF (assets/precision-*.svg): every
                             tensor's GGML type, donor and payload bytes (gguf_tensors.tsv), bytes per type
                             (gguf_formats.tsv), and per file the embedding, output head and decoder bytes
                             (gguf_summary.tsv).
  safetensors OUT LABEL=DIR...  Humming-packed SLQ checkpoints (kaitchup/Qwen3.8-27B-SLQ-EAR*): bytes per tensor
                             class (vllm_classes.tsv), every decoder projection with its stored bit width, recovered from
                             the packed weight's size (vllm_modules.tsv), and per checkpoint the projection bytes, the
                             parameter-weighted bit width and the parameters kept in BF16 (vllm_summary.tsv). Only the
                             safetensors headers are read.
"""
import glob
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict

CELL = re.compile(
    r'<g data-tensor="([^"]+)" data-origin="([^"]*)" data-ggml-type="([^"]+)"[^>]*>\s*<title\s*>'
    r'.*?Payload: ([0-9,]+) bytes; shape \[([0-9, ]+)\]', re.S)


def labelled(args):
    return [a.split("=", 1) for a in args]


def write(path, header, rows):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def svg(out, items):
    tensors, formats, summary = [], [], []
    for label, path in items:
        text = open(path, encoding="utf-8").read()
        cells = CELL.findall(text)
        by_type = Counter()
        count = Counter()
        special = {}
        for name, origin, gtype, payload, shape in cells:
            nbytes = int(payload.replace(",", ""))
            tensors.append((label, name, origin, gtype, nbytes, shape.replace(" ", "")))
            by_type[gtype] += nbytes
            count[gtype] += 1
            if name in ("token_embd.weight", "output.weight"):
                special[name] = (gtype, nbytes)
        total = sum(by_type.values())
        for gtype, nbytes in by_type.most_common():
            formats.append((label, gtype, count[gtype], nbytes, f"{100 * nbytes / total:.1f}"))
        embd, head = special["token_embd.weight"], special["output.weight"]
        summary.append((label, len(cells), total, embd[0], embd[1], head[0], head[1], total - embd[1] - head[1]))
    write(os.path.join(out, "gguf_tensors.tsv"), ("file", "tensor", "donor", "ggml_type", "bytes", "shape"), tensors)
    write(os.path.join(out, "gguf_formats.tsv"), ("file", "ggml_type", "tensors", "bytes", "share_pct"), formats)
    write(os.path.join(out, "gguf_summary.tsv"),
          ("file", "tensors", "bytes", "embd_type", "embd_bytes", "head_type", "head_bytes", "decoder_bytes"), summary)


def classify(name):
    if "visual" in name:
        return "vision"
    if name.startswith("mtp.") or ".mtp." in name:
        return "mtp"
    if "embed_tokens" in name:
        return "embed"
    if "lm_head" in name:
        return "head"
    if ".layers." in name:
        return "decoder"
    return "other"


def read_tensors(ckpt):
    tensors = {}
    for path in sorted(glob.glob(os.path.join(ckpt, "model-*.safetensors"))):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name, v in header.items():
            if name != "__metadata__":
                tensors[name] = (v["dtype"], v["shape"], v["data_offsets"][1] - v["data_offsets"][0])
    return tensors


def safetensors(out, items):
    classes, modules, summary = [], [], []
    for label, ckpt in items:
        tensors = read_tensors(ckpt)
        with open(os.path.join(ckpt, "config.json")) as f:
            qc = json.load(f).get("quantization_config", {})
        group = int(qc.get("group_size", 128))
        per_class = Counter()
        for name, (_, _, nbytes) in tensors.items():
            per_class[classify(name)] += nbytes
        for c, nbytes in sorted(per_class.items()):
            classes.append((label, c, nbytes))

        quantized = sorted(n[: -len(".weight_scale")] for n in tensors
                           if n.endswith(".weight_scale") and classify(n) == "decoder")
        proj_bytes = 0
        bits_params = defaultdict(int)
        qparams = 0
        qbits = 0
        for m in quantized:
            dtype, shape, wbytes = tensors[m + ".weight"]
            _, sshape, sbytes = tensors[m + ".weight_scale"]
            out_f, n_groups = sshape
            in_f = n_groups * group
            bits = wbytes * 8 / (out_f * in_f)
            zero = tensors.get(m + ".zero_point")
            zero_kind = {"I32": "int", "BF16": "float"}.get(zero[0], zero[0]) if zero else "none"
            zbytes = zero[2] if zero else 0
            proj_bytes += wbytes + sbytes + zbytes
            qparams += out_f * in_f
            qbits += out_f * in_f * bits
            bits_params[round(bits, 3)] += out_f * in_f
            modules.append((label, m, out_f, in_f, f"{bits:g}", zero_kind, wbytes + sbytes + zbytes))
        # every 2-D decoder weight that is not packed is a projection kept in BF16 (norms are 1-D, conv1d 3-D)
        packed = set(quantized)
        kept = [(n, s, b) for n, (d, s, b) in tensors.items()
                if classify(n) == "decoder" and n.endswith(".weight") and d == "BF16" and len(s) == 2
                and n[: -len(".weight")] not in packed]
        kept_params = sum(s[0] * s[1] for _, s, _ in kept)
        for n, s, b in sorted(kept):
            modules.append((label, n[: -len(".weight")], s[0], s[1], "16", "bf16", b))
        proj_bytes += sum(b for _, _, b in kept)
        if kept_params:
            bits_params[16] += kept_params
        hist = ";".join(f"{b:g}:{p / (qparams + kept_params):.3f}" for b, p in sorted(bits_params.items()))
        summary.append((label, len(quantized), len(kept), qparams, f"{qbits / qparams:.3f}", kept_params,
                        f"{(qbits + 16 * kept_params) / (qparams + kept_params):.3f}", proj_bytes,
                        f"{proj_bytes * 8 / (qparams + kept_params):.3f}", per_class["decoder"], hist))
    write(os.path.join(out, "vllm_classes.tsv"), ("checkpoint", "class", "bytes"), classes)
    write(os.path.join(out, "vllm_modules.tsv"),
          ("checkpoint", "module", "out", "in", "bits", "zero_point", "bytes"), modules)
    write(os.path.join(out, "vllm_summary.tsv"),
          ("checkpoint", "packed_modules", "bf16_modules", "packed_params", "packed_mean_bits", "bf16_params",
           "nominal_mean_bits_all_projections", "projection_bytes", "stored_bits_per_weight", "decoder_class_bytes",
           "params_share_by_bits"), summary)


def main():
    if len(sys.argv) < 4 or sys.argv[1] not in ("svg", "safetensors"):
        sys.exit(__doc__)
    mode, out, items = sys.argv[1], sys.argv[2], labelled(sys.argv[3:])
    os.makedirs(out, exist_ok=True)
    (svg if mode == "svg" else safetensors)(out, items)


if __name__ == "__main__":
    main()
