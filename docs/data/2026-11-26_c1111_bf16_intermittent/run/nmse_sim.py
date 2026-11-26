# C1111 (Q1844): the error test-backend-ops would report for
#   MUL_MAT(type_a=bf16,type_b=f32,m=1057,n=1,k=128,bs=[8,1],nr=[4,1],per=[0,2,1,3])
# under a few ways of going wrong, to compare with the one failure's ERR = 0.499536458. The error is test-backend-ops'
# normalized mean squared error, sum((out - ref)^2) / sum(out^2): normalized by the tested backend's output, not by the
# reference (its nmse(a, b) = mse(a, b) / mse(a, 0), called with the card's values first). Inputs as the test draws
# them: uniform in [-1, 1], A rounded to BF16 (round to nearest even), 8 channels of A shared by 32 channels of B
# (channel c of the output uses A's channel c / 4).
# Plain Python (no packages), seeded, about a minute: python3 -I nmse_sim.py
import random
import struct

random.seed(1)
K, M, CA, CB = 128, 1057, 8, 32


def bf16(x):
    u = struct.unpack('<I', struct.pack('<f', x))[0]
    return (u + 0x7FFF + ((u >> 16) & 1)) >> 16


def bf16_val(h):
    return struct.unpack('<f', struct.pack('<I', h << 16))[0]


def f16_of_bits(h):
    return struct.unpack('<e', struct.pack('<H', h))[0]


A = [[[bf16(random.uniform(-1, 1)) for _ in range(K)] for _ in range(M)] for _ in range(CA)]
B = [[random.uniform(-1, 1) for _ in range(K)] for _ in range(CB)]


def dot(arow, brow, conv):
    return sum(conv(a) * b for a, b in zip(arow, brow))


ref = [[dot(A[c // 4][r], B[c], bf16_val) for r in range(M)] for c in range(CB)]


def nmse(out):
    num = sum((o - r) ** 2 for oc, rc in zip(out, ref) for o, r in zip(oc, rc))
    return num / sum(o * o for oc in out for o in oc)


# the F16 sibling's kernels run on the BF16 case's data (a CUDA graph captured for the F16 case replayed)
f16 = [[dot(A[c // 4][r], B[c], f16_of_bits) for r in range(M)] for c in range(CB)]
print('bf16 bits read as f16:', round(nmse(f16), 4))
# half the output never written: zeros, or the replay's restored output tensor (uniform in [-1, 1])
half0 = [ch if c < 16 else [0.0] * M for c, ch in enumerate(ref)]
print('channels 16-31 zero:', round(nmse(half0), 4))
rows0 = [[v if r < M // 2 else 0.0 for r, v in enumerate(ch)] for ch in ref]
print('rows M/2.. zero:', round(nmse(rows0), 4))
stale = [ch if c < 16 else [random.uniform(-1, 1) for _ in range(M)] for c, ch in enumerate(ref)]
print('channels 16-31 stale uniform:', round(nmse(stale), 4))
