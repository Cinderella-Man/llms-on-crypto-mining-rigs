import { describe, expect, test, xtest } from '@jest/globals';
import { Scale } from './scale-generator';

describe('ScaleGenerator', () => {
  describe('Chromatic scales', () => {
    test('Chromatic scale with sharps', () => {
      const expected = [
        'C',
        'C#',
        'D',
        'D#',
        'E',
        'F',
        'F#',
        'G',
        'G#',
        'A',
        'A#',
        'B',
      ];
      expect(new Scale('C').chromatic()).toEqual(expected);
    });

    test('Chromatic scale with flats from D', () => {
      const expected = [
        'D',
        'Eb',
        'E',
        'F',
        'Gb',
        'G',
        'Ab',
        'A',
        'Bb',
        'B',
        'C',
        'Db',
      ];
      expect(new Scale('d').chromatic()).toEqual(expected);
    });
  });

  describe('Scales with specified intervals', () => {
    test('Simple major scale', () => {
      const expected = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C'];
      expect(new Scale('C').interval('MMmMMMm')).toEqual(expected);
    });

    test('Minor scale with sharps', () => {
      const expected = ['F#', 'G#', 'A', 'B', 'C#', 'D', 'E', 'F#'];
      expect(new Scale('f#').interval('MmMMmMM')).toEqual(expected);
    });

    test('Phrygian mode', () => {
      const expected = ['E', 'F', 'G', 'A', 'B', 'C', 'D', 'E'];
      expect(new Scale('e').interval('mMMMmMM')).toEqual(expected);
    });

    test('Locrian mode', () => {
      const expected = ['G', 'Ab', 'Bb', 'C', 'Db', 'Eb', 'F', 'G'];
      expect(new Scale('g').interval('mMMmMMM')).toEqual(expected);
    });

    test('Hexatonic', () => {
      const expected = ['Db', 'Eb', 'F', 'G', 'A', 'B', 'Db'];
      expect(new Scale('Db').interval('MMMMMM')).toEqual(expected);
    });
  });
});
