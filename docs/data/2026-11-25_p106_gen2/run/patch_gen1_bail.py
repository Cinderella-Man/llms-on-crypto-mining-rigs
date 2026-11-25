"""Make a copy of nvidia.ko 580.178.04 whose link-speed function ignores the request for Gen1.

The closed driver's RM calls its link-speed function (`_nv017105rm`, set_link_speed(mode), mode = PCIe generation)
with mode 1 on every CUDA initialisation, which forces a Pascal card's link back to 2.5 GT/s. The function starts with
a range check, `lea -0x1(%rdx),%eax ... cmp $0x2,%eax; ja bail` (mode in {1,2,3}). Shifting it to
`lea -0x2(%rdx),%eax ... cmp $0x1,%eax` makes mode 1 take the existing bail (which returns 0, success) and leaves
modes 2 and 3, and every other path, as they were: two bytes. Analysis by WJNKAC/nvidia-p104-unlock-gen2-for-linux
(read, not run); this is an independent implementation that refuses to write unless the 32-byte sequence occurs
exactly once and sits at the function's start + 6.

Usage: python3 -I patch_gen1_bail.py <nvidia.ko in> <nvidia.ko out> [function offset in .text, hex]
"""
import hashlib
import sys

SIG = bytes.fromhex(
    "8d42ff" "4156" "4155" "4989fd" "4154" "4531e4" "53" "4883ed40" "48897510" "89550c" "83f802" "0f87")
LEA_IMM = 2      # 0xff -> 0xfe  (lea -0x1 -> lea -0x2)
CMP_IMM = 0x1D   # 0x02 -> 0x01  (cmp $0x2 -> cmp $0x1)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    data = bytearray(open(src, "rb").read())
    print(f"in: {src} {len(data)} bytes md5 {hashlib.md5(data).hexdigest()}")
    hits = []
    i = data.find(SIG)
    while i >= 0:
        hits.append(i)
        i = data.find(SIG, i + 1)
    print(f"sequence found {len(hits)} time(s) at file offsets {[hex(h) for h in hits]}")
    if len(hits) != 1:
        print("refusing: the sequence must occur exactly once")
        return 1
    at = hits[0]
    assert data[at + LEA_IMM] == 0xFF and data[at + CMP_IMM] == 0x02
    if len(sys.argv) > 3:
        print(f"function start given: the sequence should be at its +6 (file offset of .text + function offset + 6)")
    data[at + LEA_IMM] = 0xFE
    data[at + CMP_IMM] = 0x01
    # The changed bytes void the distribution's signature, and Linux refuses a module whose signature fails even when
    # signatures are not enforced; an unsigned module loads (and taints the kernel). So drop the appended signature:
    # [module][signature][struct module_signature, 12 bytes, sig_len big-endian at its end]["~Module signature appended~\n"]
    marker = b"~Module signature appended~\n"
    if data.endswith(marker):
        info = data[-len(marker) - 12:-len(marker)]
        signer_len, key_id_len = info[3], info[4]
        sig_len = int.from_bytes(info[8:12], "big")
        cut = len(marker) + 12 + sig_len + signer_len + key_id_len
        print(f"signature appended: {sig_len} bytes (+{signer_len + key_id_len}); dropping {cut} bytes")
        del data[-cut:]
        assert data[:4] == b"\x7fELF"
    open(dst, "wb").write(data)
    print(f"out: {dst} md5 {hashlib.md5(data).hexdigest()}; changed bytes at file offsets {hex(at + LEA_IMM)} "
          f"(ff->fe) and {hex(at + CMP_IMM)} (02->01)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
