import { describe, expect, test, xtest } from '@jest/globals';
import { decode, encode } from './affine-cipher';

describe('Affine cipher', () => {
  describe('encode', () => {
    test('encode yes', () => {
      expect(encode('yes', { a: 5, b: 7 })).toBe('xbt');
    });

    test('encode O M G', () => {
      expect(encode('O M G', { a: 25, b: 47 })).toBe('hjp');
    });

    test('encode deep thought', () => {
      expect(encode('Truth is fiction.', { a: 5, b: 17 })).toBe(
        'iynia fdqfb ifje',
      );
    });
  });
  describe('decode', () => {
    test('decode exercism', () => {
      expect(decode('tytgn fjr', { a: 3, b: 7 })).toBe('exercism');
    });

    test('decode all the letters', () => {
      expect(
        decode('swxtj npvyk lruol iejdc blaxk swxmh qzglf', { a: 17, b: 33 }),
      ).toBe('thequickbrownfoxjumpsoverthelazydog');
    });

    test('decode with a not coprime to m', () => {
      expect(() => {
        decode('Test', { a: 13, b: 5 });
      }).toThrow('a and m must be coprime.');
    });
  });
});
