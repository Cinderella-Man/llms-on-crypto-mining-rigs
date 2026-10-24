import { describe, expect, test, xtest } from '@jest/globals';
import { say } from './say';

describe('say', () => {
  test('zero', () => {
    expect(say(0)).toBe('zero');
  });

  test('twenty', () => {
    expect(say(20)).toBe('twenty');
  });

  test('ninety-nine', () => {
    expect(say(99)).toBe('ninety-nine');
  });

  test('two hundred', () => {
    expect(say(200)).toBe('two hundred');
  });

  test('one thousand two hundred thirty-four', () => {
    expect(say(1234)).toBe('one thousand two hundred thirty-four');
  });

  test('one million two thousand three hundred forty-five', () => {
    expect(say(1002345)).toBe(
      'one million two thousand three hundred forty-five',
    );
  });

  test('raises an error below zero', () => {
    expect(() => {
      say(-1);
    }).toThrow(new Error('Number must be between 0 and 999,999,999,999.'));
  });
});
