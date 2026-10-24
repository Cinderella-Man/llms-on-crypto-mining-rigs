import { describe, expect, test, xtest } from '@jest/globals';
import { bestHands } from './poker';

describe('Poker', () => {
  test('single hand always wins', () => {
    const hands = ['4S 5S 7H 8D JC'];
    const expected = ['4S 5S 7H 8D JC'];
    expect(bestHands(hands)).toEqual(expected);
  });

  test('multiple hands with the same high cards, tie compares next highest ranked, down to last card', () => {
    const hands = ['3S 5H 6S 8D 7H', '2S 5D 6D 8C 7S'];
    const expected = ['3S 5H 6S 8D 7H'];
    expect(bestHands(hands)).toEqual(expected);
  });

  test('highest pair wins', () => {
    const hands = ['4S 2H 6S 2D JH', '2S 4H 6C 4D JD'];
    const expected = ['2S 4H 6C 4D JD'];
    expect(bestHands(hands)).toEqual(expected);
  });

  test('both hands have two pairs, highest ranked pair wins', () => {
    const hands = ['2S 8H 2D 8D 3H', '4S 5H 4C 8S 5D'];
    const expected = ['2S 8H 2D 8D 3H'];
    expect(bestHands(hands)).toEqual(expected);
  });

  test('both hands have two pairs that add to the same value, win goes to highest pair', () => {
    const hands = ['6S 6H 3S 3H AS', '7H 7S 2H 2S AC'];
    const expected = ['7H 7S 2H 2S AC'];
    expect(bestHands(hands)).toEqual(expected);
  });

  test('both hands have three of a kind, tie goes to highest ranked triplet', () => {
    const hands = ['2S 2H 2C 8D JH', '4S AH AS 8C AD'];
    const expected = ['4S AH AS 8C AD'];
    expect(bestHands(hands)).toEqual(expected);
  });

  test('aces can end a straight (10 J Q K A)', () => {
    const hands = ['4S 5H 4C 8D 4H', '10D JH QS KD AC'];
    const expected = ['10D JH QS KD AC'];
    expect(bestHands(hands)).toEqual(expected);
  });

  test('both hands with a straight, tie goes to highest ranked card', () => {
    const hands = ['4S 6C 7S 8D 5H', '5S 7H 8S 9D 6H'];
    const expected = ['5S 7H 8S 9D 6H'];
    expect(bestHands(hands)).toEqual(expected);
  });

  test('both hands have a flush, tie goes to high card, down to the last one if necessary', () => {
    const hands = ['2H 7H 8H 9H 6H', '3S 5S 6S 7S 8S'];
    const expected = ['2H 7H 8H 9H 6H'];
    expect(bestHands(hands)).toEqual(expected);
  });

  test('with multiple decks, both hands have a full house with the same triplet, tie goes to the pair', () => {
    const hands = ['5H 5S 5D 9S 9D', '5H 5S 5D 8S 8D'];
    const expected = ['5H 5S 5D 9S 9D'];
    expect(bestHands(hands)).toEqual(expected);
  });

  test('with multiple decks, both hands with identical four of a kind, tie determined by kicker', () => {
    const hands = ['3S 3H 2S 3D 3C', '3S 3H 4S 3D 3C'];
    const expected = ['3S 3H 4S 3D 3C'];
    expect(bestHands(hands)).toEqual(expected);
  });

  test('aces can start a straight flush (A 2 3 4 5)', () => {
    const hands = ['KS AH AS AD AC', '4H AH 3H 2H 5H'];
    const expected = ['4H AH 3H 2H 5H'];
    expect(bestHands(hands)).toEqual(expected);
  });

  test('even though an ace is usually high, a 5-high straight flush is the lowest-scoring straight flush', () => {
    const hands = ['2H 3H 4H 5H 6H', '4D AD 3D 2D 5D'];
    const expected = ['2H 3H 4H 5H 6H'];
    expect(bestHands(hands)).toEqual(expected);
  });
});
