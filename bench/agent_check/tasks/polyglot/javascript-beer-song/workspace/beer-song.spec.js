import { describe, expect, test, xtest } from '@jest/globals';
import { recite } from './beer-song';

describe('Beer Song', () => {
  describe('verse', () => {
    describe('single verse', () => {
      test('first generic verse', () => {
        expect(recite(99, 1)).toEqual([
          '99 bottles of beer on the wall, 99 bottles of beer.',
          'Take one down and pass it around, 98 bottles of beer on the wall.',
        ]);
      });

      test('verse with 1 bottle', () => {
        expect(recite(1, 1)).toEqual([
          '1 bottle of beer on the wall, 1 bottle of beer.',
          'Take it down and pass it around, no more bottles of beer on the wall.',
        ]);
      });
    });
  });

  describe('lyrics', () => {
    describe('multiple verses', () => {
      test('first two verses', () => {
        expect(recite(99, 2)).toEqual([
          '99 bottles of beer on the wall, 99 bottles of beer.',
          'Take one down and pass it around, 98 bottles of beer on the wall.',
          '',
          '98 bottles of beer on the wall, 98 bottles of beer.',
          'Take one down and pass it around, 97 bottles of beer on the wall.',
        ]);
      });
    });
  });
});
