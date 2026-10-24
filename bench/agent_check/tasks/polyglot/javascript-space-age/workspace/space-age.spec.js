import { describe, expect, test, xtest } from '@jest/globals';
import { age } from './space-age';

describe('Space Age', () => {
  test('age on Earth', () => {
    expect(age('earth', 1000000000)).toEqual(31.69);
  });

  test('age on Mars', () => {
    expect(age('mars', 2129871239)).toEqual(35.88);
  });

  test('age on Uranus', () => {
    expect(age('uranus', 1210123456)).toEqual(0.46);
  });
});
