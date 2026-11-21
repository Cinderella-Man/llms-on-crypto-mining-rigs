# C910: a copy of a GGUF without the tensors whose names contain a given part. Here ".nextn.shared_head_": the
# random qwen35moe MTP fixture's own MTP output head (with its scales) and head norm, so that the MTP graph reads the
# main output head and norm, as real Qwen MTP files do, and --mtp-device's copies of them. From pull request 7's review
# (2026-11-20), unchanged below this header.
# The copy loop follows gguf-py's gguf_new_metadata.py.
# Usage: python3 -I strip.py <gguf-py dir> <in.gguf> <out.gguf> <name part>
import sys

sys.path.insert(0, sys.argv[1])
import gguf  # noqa: E402

src, dst, part = sys.argv[2], sys.argv[3], sys.argv[4]

# the fixture holds empty arrays, which the GGUF format allows and the loader reads, but gguf-py refuses to write:
# write them as the format does (value type, element type, count 0)
_pack_val = gguf.GGUFWriter._pack_val


def _pack_val_empty_ok(self, val, vtype, add_vtype, sub_type=None):
    if vtype == gguf.GGUFValueType.ARRAY and sub_type is not None and len(val) == 0:
        out = bytearray()
        if add_vtype:
            out += self._pack("I", vtype)
        out += self._pack("I", sub_type)
        out += self._pack("Q", 0)
        return bytes(out)
    return _pack_val(self, val, vtype, add_vtype, sub_type)


gguf.GGUFWriter._pack_val = _pack_val_empty_ok
reader = gguf.GGUFReader(src, "r")
arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    val = field.contents()
    if val_type == gguf.GGUFValueType.ARRAY and len(val) == 0:
        # gguf-py's reader keeps no element type for an empty array, so the key cannot be copied as it was; the
        # fixture's empty arrays are optional keys (classifier labels), which the loader reads only when present
        print(f"empty array dropped: {field.name}")
        continue
    writer.add_key_value(field.name, val, val_type, sub_type=sub_type)
kept = [t for t in reader.tensors if part not in t.name]
dropped = [t.name for t in reader.tensors if part in t.name]
for t in kept:
    writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for t in kept:
    writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
writer.close()
print(f"{src} -> {dst}: kept {len(kept)} tensors, dropped {dropped}")
