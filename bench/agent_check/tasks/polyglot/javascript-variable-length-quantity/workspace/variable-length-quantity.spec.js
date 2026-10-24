import { describe, expect, test, xtest } from '@jest/globals';
import { decode, encode } from './variable-length-quantity';

describe('VariableLengthQuantity', () => {
  describe('Encode a series of integers, producing a series of bytes.', () => {
    test('zero', () => {
      expect(encode([0])).toEqual([0]);
    });

    test('largest single byte', () => {
      expect(encode([0x7f])).toEqual([0x7f]);
    });

    test('asymmetric double byte', () => {
      expect(encode([0xad])).toEqual([0x81, 0x2d]);
    });

    test('arbitrary triple byte', () => {
      expect(encode([0x100000])).toEqual([0xc0, 0x80, 0]);
    });

    test('smallest quadruple byte', () => {
      expect(encode([0x200000])).toEqual([0x81, 0x80, 0x80, 0]);
    });

    test('largest quadruple byte', () => {
      expect(encode([0xfffffff])).toEqual([0xff, 0xff, 0xff, 0x7f]);
    });

    test('asymmetric quintuple byte', () => {
      expect(encode([0x86656105])).toEqual([0x88, 0xb3, 0x95, 0xc2, 0x05]);
    });

    test('two multi-byte values', () => {
      expect(encode([0x4000, 0x123456])).toEqual([
        0x81, 0x80, 0, 0xc8, 0xe8, 0x56,
      ]);
    });
  });

  describe('Decode a series of bytes, producing a series of integers.', () => {
    test('one byte', () => {
      expect(decode([0x7f])).toEqual([0x7f]);
    });

    test('four bytes', () => {
      expect(decode([0x81, 0x80, 0x80, 0])).toEqual([0x200000]);
    });

    test('incomplete sequence causes error, even if value is zero', () => {
      expect(() => {
        decode([0x80]);
      }).toThrow(new Error('Incomplete sequence'));
    });
  });
});
