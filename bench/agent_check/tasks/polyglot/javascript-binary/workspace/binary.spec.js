import { describe, test, expect, xtest } from '@jest/globals';
import { Binary } from './binary';

describe('binary', () => {
  test('0 is decimal 0', () => expect(new Binary('0').toDecimal()).toEqual(0));

  test('11 is decimal 3', () =>
    expect(new Binary('11').toDecimal()).toEqual(3));

  test('11010 is decimal 26', () =>
    expect(new Binary('11010').toDecimal()).toEqual(26));

  test('invalid inputs are null', () => {
    // "2 is not a valid binary digit
    expect(new Binary('2').toDecimal()).toEqual(null);

    // a number containing a non-binary digit is invalid
    expect(new Binary('01201').toDecimal()).toEqual(null);

    // a number with trailing non-binary characters is invalid
    expect(new Binary('10nope').toDecimal()).toEqual(null);

    // a number with leading non-binary characters is invalid
    expect(new Binary('nope10').toDecimal()).toEqual(null);

    // a number with internal non-binary characters is invalid
    expect(new Binary('10nope10').toDecimal()).toEqual(null);

    // a number and a word whitespace separated is invalid
    expect(new Binary('001nope').toDecimal()).toEqual(null);
  });
});
