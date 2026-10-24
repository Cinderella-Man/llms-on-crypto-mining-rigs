import { describe, expect, test, xtest } from '@jest/globals';
import { recite } from './twelve-days';

describe('TwelveDays', () => {
  test('verse first day a partridge in a pear tree', () => {
    const expectedVerseOne =
      'On the first day of Christmas my true love gave to me: a Partridge in a Pear Tree.\n';
    expect(recite(1)).toEqual(expectedVerseOne);
  });

  test('verse fourth day four calling birds', () => {
    const expectedVerseFour =
      'On the fourth day of Christmas my true love gave to me: four Calling Birds, three French Hens, two Turtle Doves, and a Partridge in a Pear Tree.\n';
    expect(recite(4)).toEqual(expectedVerseFour);
  });

  test('verse seventh day seven swans-a-swimming', () => {
    const expectedVerseSeven =
      'On the seventh day of Christmas my true love gave to me: seven Swans-a-Swimming, six Geese-a-Laying, five Gold Rings, four Calling Birds, three French Hens, two Turtle Doves, and a Partridge in a Pear Tree.\n';
    expect(recite(7)).toEqual(expectedVerseSeven);
  });

  test('verse tenth day ten lords-a-leaping', () => {
    const expectedVerseTen =
      'On the tenth day of Christmas my true love gave to me: ten Lords-a-Leaping, nine Ladies Dancing, eight Maids-a-Milking, seven Swans-a-Swimming, six Geese-a-Laying, five Gold Rings, four Calling Birds, three French Hens, two Turtle Doves, and a Partridge in a Pear Tree.\n';
    expect(recite(10)).toEqual(expectedVerseTen);
  });

  test('lyrics recites first three verses of the song', () => {
    const expectedVerseOneToThree =
      'On the first day of Christmas my true love gave to me: a Partridge in a Pear Tree.\n\n' +
      'On the second day of Christmas my true love gave to me: two Turtle Doves, and a Partridge in a Pear Tree.\n\n' +
      'On the third day of Christmas my true love gave to me: three French Hens, two Turtle Doves, and a Partridge in a Pear Tree.\n';
    expect(recite(1, 3)).toEqual(expectedVerseOneToThree);
  });
});
