import { describe, expect, test, xtest } from '@jest/globals';
import { clean } from './phone-number';

describe('Phone Number', () => {
  describe('Cleanup user-entered phone numbers', () => {
    test('cleans the number', () => {
      expect(clean('(223) 456-7890')).toEqual('2234567890');
    });

    test('invalid when 9 digits', () => {
      expect(() => clean('123456789')).toThrow(
        new Error('Must not be fewer than 10 digits'),
      );
    });

    test('valid when 11 digits and starting with 1 even with punctuation', () => {
      expect(clean('+1 (223) 456-7890')).toEqual('2234567890');
    });

    test('invalid with punctuations', () => {
      expect(() => clean('523-@:!-7890')).toThrow(
        new Error('Punctuations not permitted'),
      );
    });

    test('invalid if exchange code starts with 0', () => {
      expect(() => clean('(223) 056-7890')).toThrow(
        new Error('Exchange code cannot start with zero'),
      );
    });

    test('invalid if area code starts with 1 on valid 11-digit number', () => {
      expect(() => clean('1 (123) 456-7890')).toThrow(
        new Error('Area code cannot start with one'),
      );
    });
  });
});
