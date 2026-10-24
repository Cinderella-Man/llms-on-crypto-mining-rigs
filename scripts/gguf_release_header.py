#!/usr/bin/env python3
# Rewrites the metadata of a GGUF before it is uploaded (docs/research/notes/release_v2.md, section 1.5): every tensor
# is copied byte for byte, every metadata key is kept unless an option changes it. Files composed from Unsloth's carry
# Unsloth's header (general.license = other, general.license.name = flint-preview, general.name = Flint Preview,
# general.quantized_by = Unsloth), which is wrong for a file of ours; quantize.imatrix.* stays, since it records whose
# importance matrix was used. The GGUF format keeps the tensor data after the metadata, so the whole file is rewritten.
#
#   gguf_release_header.py IN.gguf                        print the metadata (strings, numbers, array lengths)
#   gguf_release_header.py IN.gguf OUT.gguf [options]     write OUT with the changes, print what changed, then verify
#                                                         that OUT's tensors equal IN's (names, types, shapes, bytes)
# Options (repeatable where it makes sense):
#   --preset flint          general.license = apache-2.0 (general.license.name and .link removed), base model Flint
#                           by RecursiveSI (https://huggingface.co/RecursiveSI/flint), and the makers' sampling:
#                           general.sampling.temp 1.0, top_p 0.95, top_k 20, min_p 0 (llama-server's own min-p default
#                           is 0.05, so without the key stock users would not get the makers' sampling)
#   --name NAME             general.name                  --quantized-by WHO   general.quantized_by
#   --set KEY=TEXT          a string key                  --set-f32 KEY=X      a float32 key
#   --set-i32 KEY=N         an int32 key                  --set-u32 KEY=N      a uint32 key
#   --delete KEY            remove a key
#   --dry-run               print the changes, write nothing
#   --no-verify             skip the tensor comparison (it reads both files once more)
# Needs numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py). Exit status: 0 done, 1 verification failed, 2 usage.
import argparse
import hashlib
import sys

import gguf

VT = gguf.GGUFValueType
PRESETS = {
    "flint": {
        "set": {
            "general.license": (VT.STRING, "apache-2.0"),
            "general.base_model.count": (VT.UINT32, 1),
            "general.base_model.0.name": (VT.STRING, "Flint"),
            "general.base_model.0.organization": (VT.STRING, "RecursiveSI"),
            "general.base_model.0.repo_url": (VT.STRING, "https://huggingface.co/RecursiveSI/flint"),
            "general.sampling.temp": (VT.FLOAT32, 1.0),
            "general.sampling.top_p": (VT.FLOAT32, 0.95),
            "general.sampling.top_k": (VT.INT32, 20),
            "general.sampling.min_p": (VT.FLOAT32, 0.0),
        },
        "delete": ["general.license.name", "general.license.link"],
    },
}


def shown(field):
    """a field's value for printing: strings and numbers as they are, arrays by length and first items"""
    if field.types[0] == VT.ARRAY:
        n = len(field.data)
        if field.types[-1] == VT.STRING or n > 8:
            return f"[{n} × {field.types[-1].name}]"
        return str(field.contents())
    return repr(field.contents())


def parse_kv(text, opt):
    if "=" not in text:
        sys.exit(f"{opt} needs KEY=VALUE, got {text!r}")
    return text.split("=", 1)


def main():
    ap = argparse.ArgumentParser(add_help=True, description="Rewrite a GGUF's metadata for a release (see the header).")
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--name")
    ap.add_argument("--quantized-by")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--set-f32", action="append", default=[])
    ap.add_argument("--set-i32", action="append", default=[])
    ap.add_argument("--set-u32", action="append", default=[])
    ap.add_argument("--delete", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    r = gguf.GGUFReader(a.src)
    if a.dst is None:
        for f in r.fields.values():
            if not f.name.startswith("GGUF."):
                print(f"{f.name} = {shown(f)}")
        print(f"{len(r.tensors)} tensors")
        return 0

    changes, deletes = {}, set(a.delete)
    if a.preset:
        changes.update(PRESETS[a.preset]["set"])
        deletes.update(PRESETS[a.preset]["delete"])
    if a.name:
        changes["general.name"] = (VT.STRING, a.name)
    if a.quantized_by:
        changes["general.quantized_by"] = (VT.STRING, a.quantized_by)
    for opt, vt, conv in (("set", VT.STRING, str), ("set_f32", VT.FLOAT32, float), ("set_i32", VT.INT32, int),
                          ("set_u32", VT.UINT32, int)):
        for kv in getattr(a, opt):
            k, v = parse_kv(kv, "--" + opt.replace("_", "-"))
            changes[k] = (vt, conv(v))
    for k in changes:
        deletes.discard(k)
    if gguf.Keys.General.ARCHITECTURE in changes or gguf.Keys.General.ARCHITECTURE in deletes:
        sys.exit("general.architecture cannot be changed here")

    old = {f.name: f for f in r.fields.values() if not f.name.startswith("GGUF.")}
    for k in sorted(deletes):
        if k in old:
            print(f"- {k} = {shown(old[k])}")
    for k, (vt, v) in changes.items():
        if k in old:
            ov = old[k].contents()
            same_value = ov == v or (vt == VT.FLOAT32 and isinstance(ov, float) and abs(ov - v) < 1e-6)
            if old[k].types[0] == vt and same_value:
                continue
            print(f"~ {k} = {shown(old[k])} -> {v!r} ({vt.name})")
        else:
            print(f"+ {k} = {v!r} ({vt.name})")
    if a.dry_run:
        return 0

    arch = r.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    w = gguf.GGUFWriter(a.dst, arch=arch, endianess=r.endianess)
    for f in old.values():  # the source's order, changed values in place
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name in deletes:
            continue
        if f.name in changes:
            vt, v = changes[f.name]
            w.add_key_value(f.name, v, vt)
        else:
            vt = f.types[0]
            w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == VT.ARRAY else None)
    for k, (vt, v) in changes.items():  # new keys at the end
        if k not in old:
            w.add_key_value(k, v, vt)
    for t in r.tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in r.tensors:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()
    print(f"wrote {a.dst}: {len(r.tensors)} tensors")
    if a.no_verify:
        return 0

    o = gguf.GGUFReader(a.dst)
    if len(o.tensors) != len(r.tensors):
        print(f"VERIFY FAILED: {len(o.tensors)} tensors against {len(r.tensors)}")
        return 1
    for t, u in zip(r.tensors, o.tensors):
        same = (t.name == u.name and t.tensor_type == u.tensor_type and list(t.shape) == list(u.shape)
                and t.n_bytes == u.n_bytes
                and hashlib.sha256(t.data.tobytes()).digest() == hashlib.sha256(u.data.tobytes()).digest())
        if not same:
            print(f"VERIFY FAILED at tensor {t.name}")
            return 1
    print("verified: every tensor's name, type, shape and bytes equal the source's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
