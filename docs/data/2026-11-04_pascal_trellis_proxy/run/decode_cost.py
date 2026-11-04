#!/usr/bin/env python3
# Estimated one-token decode cost on a P106-090 (Pascal sm_61, 5 SMs, ~1.8 GHz under load) of every code in the
# Pascal trellis proxy (registry Q929), in picoseconds per weight of a routed-expert matrix-vector product.
#
# Per 4 weights and one activation column, a kernel spends:
#   alu   issue slots on the 128-lane pipes: shifts and funnel shifts (SHF), logic (LOP/LOP3), byte permutes (PRMT),
#         bit-field extracts (BFE), 16-bit multiply-adds (XMAD; a 32-bit multiply is 3 of them on Pascal), dp4a,
#         plus the per-block scale work spread over its weights
#   lds   random 32-bit (or 16-bit) shared-memory reads: 32 lanes at random addresses hit ~3.5 lanes in the busiest of
#         the 32 banks, so one warp read takes ~3.5 cycles of the SM's one shared-memory port
#   ldsf  conflict-free shared-memory reads (a table copied once per lane, or per bank group)
# and the model is
#   ps per weight = F + A * alu + B * lds + B * ldsf / 3.5,   A = 1 / (4 * 5 SMs * 128 lanes * 1.8 GHz) = 0.217 ps,
#   B = 3.5 cycles / (128 weights * 5 SMs * 1.8 GHz) = 3.04 ps,
# with F (activations, weight loads, the reduction and launch overhead of a 1-token product on Flash-Next's expert
# shapes) fitted to Miner Llama's Q2_0 kernel. The other measured kernels check it (`check`): N2P, IQ2_S, IQ3_XXS with
# shared-memory tables and agentionai's TQ2_T kernel, all on a P106 at one token.
#
#   decode_cost.py              the anchors, the fit and every code's estimate
import re, sys

A = 1 / (4 * 5 * 128 * 1.8e9) * 1e12
B = 3.5 / (128 * 5 * 1.8e9) * 1e12

# measured on a P106 at one token (ps per weight): Q2_0 docs/data/2026-11-04_gyro_p106 (Miner Llama's Q2_0 gate/up),
# N2P and IQ (Pascal kernels with shared-memory tables) docs/data/2026-10-28_lowbit_rig and registry Q164,
# TQ2_T docs/data/2026-11-04_gyro_p106 (gate/up with the one-row-per-warp layout, GGML_CUDA_TQ_MMVQ_ROWS=1)
ANCHORS = {
    # name: (alu, lds, ldsf, measured, what the count holds)
    'Q2_0':    (3.0, 0, 0, 3.3, 'PRMT lookup 1, shift 0.75, dp4a 1, scale and sum per 64 weights 0.25'),
    'N2P':     (4.75, 0, 0, 4.18, 'PRMT lookup 1, table select 0.5, shift 0.75, dp4a 1, 4-bit sub-scale per 8 (BFE, XMAD, add) 1.5'),
    'IQ2_S':   (7.0, 1, 0, 6.67, 'grid index (BFE, LOP, SHF) 1.5, one 8-byte grid read per 8 weights = 1 random 4-byte read, signs 4, dp4a 1, scale 0.5'),
    'IQ3_XXS': (6.25, 1, 0, 7.07, 'grid index 2, one 4-byte grid read, signs (parity, expand, apply) 3, dp4a 1, scale 0.25'),
    'TQ2_T':   (11.5, 2, 0, 10.0, 'state (PRMT, AND) 2.5, 32-bit multiply = 3 XMAD, two 11-bit fields 2, two 2-byte reads (+ addresses 2), pack 1, dp4a 1'),
}
F = ANCHORS['Q2_0'][3] - A * ANCHORS['Q2_0'][0]

# the codes (per 4 weights, one column); keys match ptq_proxy.py's names without the .i8 / +r128 suffixes
CODES = {
    'tq2t':        (11.6, 2, 0, "agentionai's kernel as it is (the TQ2_T anchor)"),
    'lut4':        (3.1, 1, 0, '11-bit window: funnel shift + mask into a word address 2, one 4-byte read, dp4a 1'),
    'lut4.k25':    (3.1, 1, 0, 'as lut4 (the window is not byte-aligned; the funnel shift takes any offset)'),
    'hash4':       (5.1, 1, 0, 'state PRMT 1, 16-bit XMAD hash 1, field to address (SHF, LOP) 2, one 4-byte read, dp4a 1'),
    'hash4.k25':   (6.1, 1, 0, 'as hash4, the 16-bit state from a 10-bit-step stream: funnel shift + mask 2'),
    'hash4.n128':  (6.1, 0, 1 * 1.5, 'as hash4 + lane offset 1; 16 copies (8 KB): two lanes per copy, ~1.5-way conflicts'),
    'hash4.n64':   (6.1, 0, 1, 'as hash4 + lane offset 1; 32 lane copies of 64 entries (8 KB): conflict-free'),
    'lut2':        (6.1, 2, 0, 'two 12-bit windows to addresses 4, two 2-byte reads, pack PRMT 1, dp4a 1'),
    'lut2.k25':    (6.1, 2, 0, 'as lut2'),
    'hp4':         (6.1, 0, 0, 'state PRMT 1, 16-bit XMAD 1, fold (SHF 1, LOP3 xor-and-mask 1), PRMT lookup of 8 int8 levels in two registers 1, dp4a 1'),
    'hp4.k25':     (7.1, 0, 0, 'as hp4, the state from a 10-bit-step stream: funnel shift + mask 2'),
    'hp4.n16':     (10.1, 0, 0, 'as hp4 with 4-bit fields: two PRMT lookups (16 levels in four registers), mask PRMT 1, select LOP3 1, the fold mask 1'),
    'hp4m':        (8.1, 0, 0, "as hp4 with Gyro's 32-bit multiply (3 XMAD) and no fold"),
    'hp4s':        (9.1, 0, 0, 'as hp4 + a table pair chosen by a hash bit: sign-extend to a mask 1, two LOP3 blends of the table registers 2'),
    'tcq8u':       (7.1, 0, 0, 'codes to bytes (SHF, LOP) 2, parity of 2 state bits (2 funnel SHF, LOP3) 3, OR 1, dp4a 1; uniform levels, the offset from the activation sums per block'),
    'tcq8':        (8.1, 0, 0, 'as tcq8u in nibble layout + PRMT lookup of 8 int8 levels held in two registers 1'),
    'tcq8.3':      (10.1, 0, 0, 'as tcq8 with a 3-term parity (one more SHF and LOP)'),
    'tcq16':       (14.1, 0, 0, 'two parities 6, codes 2, OR 1, two PRMT lookups (16 levels in four registers) 2, mask PRMT 1, select LOP3 1, dp4a 1'),
}


def est(alu, lds, ldsf):
    return F + A * alu + B * lds + B / 3.5 * ldsf


def code_key(fmt):
    """ptq_proxy.py format name -> CODES key."""
    base = fmt.replace('+r128', '')
    base = re.sub(r'\.i8$', '', base)
    parts = base.split('.')
    fam = parts[0]
    if fam == 'hash4' and parts[1] in ('n128', 'n64'):
        return 'hash4.' + parts[1]
    if fam == 'hp4' and parts[1] == 'n16':
        return 'hp4.n16'
    if parts[-1] == 'k25':
        return fam + '.k25'
    return fam


def estimate(fmt, three_term=False):
    k = code_key(fmt)
    if k == 'tcq8' and three_term:
        k = 'tcq8.3'
    if k not in CODES:
        return None
    alu, lds, ldsf, _ = CODES[k]
    return est(alu, lds, ldsf)


def main():
    print('A = %.3f ps per ALU slot per 4 weights, B = %.2f ps per random shared-memory read per 4 weights, F = %.2f ps' % (A, B, F))
    print('\n%-8s %6s %4s %5s %9s %9s  %s' % ('kernel', 'alu', 'lds', 'ldsf', 'measured', 'model', 'count'))
    for n, (alu, lds, ldsf, meas, what) in ANCHORS.items():
        m = est(alu, lds, ldsf)
        print('%-8s %6.2f %4g %5g %9.2f %9.2f  %+.0f %%  %s' % (n, alu, lds, ldsf, meas, m, 100 * (m / meas - 1), what))
    print('\n%-11s %6s %4s %5s %9s  %s' % ('code', 'alu', 'lds', 'ldsf', 'est. ps', 'count'))
    for n, (alu, lds, ldsf, what) in CODES.items():
        print('%-11s %6.2f %4g %5g %9.2f  %s' % (n, alu, lds, ldsf, est(alu, lds, ldsf), what))


if __name__ == '__main__':
    main()
