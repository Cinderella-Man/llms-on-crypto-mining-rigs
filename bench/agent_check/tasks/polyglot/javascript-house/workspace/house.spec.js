import { describe, expect, test, xtest } from '@jest/globals';
import { House } from './house';

describe('House', () => {
  test('verse one - the house that jack built', () => {
    const lyrics = ['This is the house that Jack built.'];
    expect(House.verse(1)).toEqual(lyrics);
  });

  test('verse four - the cat that killed', () => {
    const lyrics = [
      'This is the cat',
      'that killed the rat',
      'that ate the malt',
      'that lay in the house that Jack built.',
    ];
    expect(House.verse(4)).toEqual(lyrics);
  });

  test('verse seven - the maiden all forlorn', () => {
    const lyrics = [
      'This is the maiden all forlorn',
      'that milked the cow with the crumpled horn',
      'that tossed the dog',
      'that worried the cat',
      'that killed the rat',
      'that ate the malt',
      'that lay in the house that Jack built.',
    ];
    expect(House.verse(7)).toEqual(lyrics);
  });

  test('verse ten - the rooster that crowed in the morn', () => {
    const lyrics = [
      'This is the rooster that crowed in the morn',
      'that woke the priest all shaven and shorn',
      'that married the man all tattered and torn',
      'that kissed the maiden all forlorn',
      'that milked the cow with the crumpled horn',
      'that tossed the dog',
      'that worried the cat',
      'that killed the rat',
      'that ate the malt',
      'that lay in the house that Jack built.',
    ];
    expect(House.verse(10)).toEqual(lyrics);
  });

  test('multiple verses', () => {
    const startVerse = 4;
    const endVerse = 8;
    const lyrics = [
      'This is the cat',
      'that killed the rat',
      'that ate the malt',
      'that lay in the house that Jack built.',
      '',
      'This is the dog',
      'that worried the cat',
      'that killed the rat',
      'that ate the malt',
      'that lay in the house that Jack built.',
      '',
      'This is the cow with the crumpled horn',
      'that tossed the dog',
      'that worried the cat',
      'that killed the rat',
      'that ate the malt',
      'that lay in the house that Jack built.',
      '',
      'This is the maiden all forlorn',
      'that milked the cow with the crumpled horn',
      'that tossed the dog',
      'that worried the cat',
      'that killed the rat',
      'that ate the malt',
      'that lay in the house that Jack built.',
      '',
      'This is the man all tattered and torn',
      'that kissed the maiden all forlorn',
      'that milked the cow with the crumpled horn',
      'that tossed the dog',
      'that worried the cat',
      'that killed the rat',
      'that ate the malt',
      'that lay in the house that Jack built.',
    ];
    expect(House.verses(startVerse, endVerse)).toEqual(lyrics);
  });
});
