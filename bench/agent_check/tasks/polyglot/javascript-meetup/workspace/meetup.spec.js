import { describe, expect, test, xtest } from '@jest/globals';
import { meetup } from './meetup';

describe('Meetup', () => {
  test('monteenth of May 2013', () => {
    expect(meetup(2013, 5, 'teenth', 'Monday')).toEqual(new Date(2013, 4, 13));
  });

  test('tuesteenth of March 2013', () => {
    expect(meetup(2013, 3, 'teenth', 'Tuesday')).toEqual(new Date(2013, 2, 19));
  });

  test('wednesteenth of January 2013', () => {
    expect(meetup(2013, 1, 'teenth', 'Wednesday')).toEqual(
      new Date(2013, 0, 16),
    );
  });

  test('thursteenth of May 2013', () => {
    expect(meetup(2013, 5, 'teenth', 'Thursday')).toEqual(
      new Date(2013, 4, 16),
    );
  });

  test('friteenth of April 2013', () => {
    expect(meetup(2013, 4, 'teenth', 'Friday')).toEqual(new Date(2013, 3, 19));
  });

  test('saturteenth of February 2013', () => {
    expect(meetup(2013, 2, 'teenth', 'Saturday')).toEqual(
      new Date(2013, 1, 16),
    );
  });

  test('sunteenth of May 2013', () => {
    expect(meetup(2013, 5, 'teenth', 'Sunday')).toEqual(new Date(2013, 4, 19));
  });

  test('first Monday of March 2013', () => {
    expect(meetup(2013, 3, 'first', 'Monday')).toEqual(new Date(2013, 2, 4));
  });

  test('first Tuesday of June 2013', () => {
    expect(meetup(2013, 6, 'first', 'Tuesday')).toEqual(new Date(2013, 5, 4));
  });

  test('first Thursday of September 2013', () => {
    expect(meetup(2013, 9, 'first', 'Thursday')).toEqual(new Date(2013, 8, 5));
  });

  test('first Friday of December 2013', () => {
    expect(meetup(2013, 12, 'first', 'Friday')).toEqual(new Date(2013, 11, 6));
  });

  test('first Sunday of March 2013', () => {
    expect(meetup(2013, 3, 'first', 'Sunday')).toEqual(new Date(2013, 2, 3));
  });

  test('second Monday of April 2013', () => {
    expect(meetup(2013, 4, 'second', 'Monday')).toEqual(new Date(2013, 3, 8));
  });

  test('second Wednesday of July 2013', () => {
    expect(meetup(2013, 7, 'second', 'Wednesday')).toEqual(
      new Date(2013, 6, 10),
    );
  });

  test('second Thursday of October 2013', () => {
    expect(meetup(2013, 10, 'second', 'Thursday')).toEqual(
      new Date(2013, 9, 10),
    );
  });

  test('second Saturday of January 2013', () => {
    expect(meetup(2013, 1, 'second', 'Saturday')).toEqual(
      new Date(2013, 0, 12),
    );
  });

  test('second Sunday of April 2013', () => {
    expect(meetup(2013, 4, 'second', 'Sunday')).toEqual(new Date(2013, 3, 14));
  });

  test('third Tuesday of May 2013', () => {
    expect(meetup(2013, 5, 'third', 'Tuesday')).toEqual(new Date(2013, 4, 21));
  });

  test('third Wednesday of August 2013', () => {
    expect(meetup(2013, 8, 'third', 'Wednesday')).toEqual(
      new Date(2013, 7, 21),
    );
  });

  test('third Friday of November 2013', () => {
    expect(meetup(2013, 11, 'third', 'Friday')).toEqual(new Date(2013, 10, 15));
  });

  test('third Saturday of February 2013', () => {
    expect(meetup(2013, 2, 'third', 'Saturday')).toEqual(new Date(2013, 1, 16));
  });

  test('fourth Monday of March 2013', () => {
    expect(meetup(2013, 3, 'fourth', 'Monday')).toEqual(new Date(2013, 2, 25));
  });

  test('fourth Tuesday of June 2013', () => {
    expect(meetup(2013, 6, 'fourth', 'Tuesday')).toEqual(new Date(2013, 5, 25));
  });

  test('fourth Thursday of September 2013', () => {
    expect(meetup(2013, 9, 'fourth', 'Thursday')).toEqual(
      new Date(2013, 8, 26),
    );
  });

  test('fourth Friday of December 2013', () => {
    expect(meetup(2013, 12, 'fourth', 'Friday')).toEqual(
      new Date(2013, 11, 27),
    );
  });

  test('fourth Sunday of March 2013', () => {
    expect(meetup(2013, 3, 'fourth', 'Sunday')).toEqual(new Date(2013, 2, 24));
  });

  test('last Monday of April 2013', () => {
    expect(meetup(2013, 4, 'last', 'Monday')).toEqual(new Date(2013, 3, 29));
  });

  test('last Wednesday of July 2013', () => {
    expect(meetup(2013, 7, 'last', 'Wednesday')).toEqual(new Date(2013, 6, 31));
  });

  test('last Thursday of October 2013', () => {
    expect(meetup(2013, 10, 'last', 'Thursday')).toEqual(new Date(2013, 9, 31));
  });

  test('last Saturday of January 2013', () => {
    expect(meetup(2013, 1, 'last', 'Saturday')).toEqual(new Date(2013, 0, 26));
  });

  test('last Sunday of April 2013', () => {
    expect(meetup(2013, 4, 'last', 'Sunday')).toEqual(new Date(2013, 3, 28));
  });

  test('last Sunday of February 2015', () => {
    expect(meetup(2015, 2, 'last', 'Sunday')).toEqual(new Date(2015, 1, 22));
  });
});
