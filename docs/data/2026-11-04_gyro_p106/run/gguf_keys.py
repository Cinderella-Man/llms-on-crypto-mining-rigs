#!/usr/bin/env python3
"""A GGUF's metadata keys and values from its header (standard library; reads only the key/value section).

  python3 -I gguf_keys.py FILE [PREFIX]     every key starting with PREFIX (default: all), one per line;
                                            arrays print their length, element type and first items
"""
import struct
import sys

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
NAMES = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32", 7: "bool", 8: "str", 9: "arr",
         10: "u64", 11: "i64", 12: "f64"}


def read(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def string(f):
    return f.read(read(f, "<Q")).decode("utf-8", "replace")


def value(f, t):
    if t == 8:
        return string(f)
    if t == 9:
        et, n = read(f, "<I"), read(f, "<Q")
        return (NAMES[et], [value(f, et) for _ in range(n)])
    return read(f, SCALAR[t])


def main():
    path, prefix = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", "not a GGUF file"
        read(f, "<I")
        read(f, "<Q")
        for _ in range(read(f, "<Q")):
            k, t = string(f), read(f, "<I")
            v = value(f, t)
            if not k.startswith(prefix):
                continue
            if isinstance(v, tuple):
                et, items = v
                shown = ", ".join(map(str, items[:12]))
                print(f"{k} = [{len(items)} x {et}] {shown}{', ...' if len(items) > 12 else ''}")
            else:
                print(f"{k} = {v}")


if __name__ == "__main__":
    main()
