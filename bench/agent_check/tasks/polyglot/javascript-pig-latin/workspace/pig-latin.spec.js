import { describe, expect, test, xtest } from '@jest/globals';
import { translate } from './pig-latin';

describe('Pig Latin', () => {
  describe('ay is added to words that start with vowels', () => {
    test('word beginning with a', () => {
      expect(translate('apple')).toEqual('appleay');
    });

    test('word beginning with o', () => {
      expect(translate('object')).toEqual('objectay');
    });
  });

  describe('first letter and ay are moved to the end of words that start with consonants', () => {
    test('word beginning with p', () => {
      expect(translate('pig')).toEqual('igpay');
    });

    test('word beginning with q without a following u', () => {
      expect(translate('qat')).toEqual('atqay');
    });
  });

  describe('some letter clusters are treated like a single consonant', () => {
    test('word beginning with ch', () => {
      expect(translate('chair')).toEqual('airchay');
    });

    test('word beginning with th', () => {
      expect(translate('therapy')).toEqual('erapythay');
    });
  });

  describe('some letter clusters are treated like a single vowel', () => {
    test('word beginning with yt', () => {
      expect(translate('yttria')).toEqual('yttriaay');
    });
  });

  describe('position of y in a word determines if it is a consonant or a vowel', () => {
    test('y is treated like a consonant at the beginning of a word', () => {
      expect(translate('yellow')).toEqual('ellowyay');
    });
  });

  describe('phrases are translated', () => {
    test('a whole phrase', () => {
      expect(translate('quick fast run')).toEqual('ickquay astfay unray');
    });
  });
});
