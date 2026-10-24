import { describe, expect, test, xtest } from '@jest/globals';
import { recite } from './bottle-song';

describe('Bottle Song', () => {
  describe('verse', () => {
    describe('single verse', () => {
      test('first generic verse', () => {
        let expected = [
          `Ten green bottles hanging on the wall,`,
          `Ten green bottles hanging on the wall,`,
          `And if one green bottle should accidentally fall,`,
          `There'll be nine green bottles hanging on the wall.`,
        ];
        expect(recite(10, 1)).toEqual(expected);
      });

      test('verse with 1 bottle', () => {
        let expected = [
          `One green bottle hanging on the wall,`,
          `One green bottle hanging on the wall,`,
          `And if one green bottle should accidentally fall,`,
          `There'll be no green bottles hanging on the wall.`,
        ];
        expect(recite(1, 1)).toEqual(expected);
      });
    });
  });

  describe('lyrics', () => {
    describe('multiple verses', () => {
      test('first two verses', () => {
        let expected = [
          `Ten green bottles hanging on the wall,`,
          `Ten green bottles hanging on the wall,`,
          `And if one green bottle should accidentally fall,`,
          `There'll be nine green bottles hanging on the wall.`,
          ``,
          `Nine green bottles hanging on the wall,`,
          `Nine green bottles hanging on the wall,`,
          `And if one green bottle should accidentally fall,`,
          `There'll be eight green bottles hanging on the wall.`,
        ];
        expect(recite(10, 2)).toEqual(expected);
      });
    });
  });
});
