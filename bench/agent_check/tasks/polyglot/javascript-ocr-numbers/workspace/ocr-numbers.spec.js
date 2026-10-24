import { describe, expect, test, xtest } from '@jest/globals';
import { convert } from './ocr-numbers';

describe('ocr', () => {
  test('recognizes zero', () => {
    expect(
      // prettier-ignore
      convert(
        ' _ \n' +
        '| |\n' +
        '|_|\n' +
        '   '
      ),
    ).toBe('0');
  });

  test('recognizes three', () => {
    expect(
      // prettier-ignore
      convert(
        ' _ \n' +
        ' _|\n' +
        ' _|\n' +
        '   '
      ),
    ).toBe('3');
  });

  test('recognizes six', () => {
    expect(
      // prettier-ignore
      convert(
        ' _ \n' +
        '|_ \n' +
        '|_|\n' +
        '   '
      ),
    ).toBe('6');
  });

  test('recognizes nine', () => {
    expect(
      // prettier-ignore
      convert(
        ' _ \n' +
        '|_|\n' +
        ' _|\n' +
        '   '
      ),
    ).toBe('9');
  });

  test('converts 110101100', () => {
    expect(
      // prettier-ignore
      convert(
        '       _     _        _  _ \n' +
        '  |  || |  || |  |  || || |\n' +
        '  |  ||_|  ||_|  |  ||_||_|\n' +
        '                           '
      ),
    ).toBe('110101100');
  });

  test('converts 123 456 789', () => {
    expect(
      // prettier-ignore
      convert(
        '    _  _ \n' +
        '  | _| _|\n' +
        '  ||_  _|\n' +
        '         \n' +
        '    _  _ \n' +
        '|_||_ |_ \n' +
        '  | _||_|\n' +
        '         \n' +
        ' _  _  _ \n' +
        '  ||_||_|\n' +
        '  ||_| _|\n' +
        '         '
      ),
    ).toBe('123,456,789');
  });
});
