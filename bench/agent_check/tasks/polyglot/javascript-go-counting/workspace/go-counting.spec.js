import { describe, expect, test, xtest } from '@jest/globals';
import { GoCounting } from './go-counting';

describe('Go Counting', () => {
  describe('getTerritory', () => {
    test('Black corner territory on 5x5 board', () => {
      const board = ['  B  ', ' B B ', 'B W B', ' W W ', '  W  '];
      const goCounting = new GoCounting(board);
      const expectedTerritory = {
        owner: 'BLACK',
        territory: [
          [0, 0],
          [0, 1],
          [1, 0],
        ],
      };
      expect(goCounting.getTerritory(0, 1)).toEqual(expectedTerritory);
    });

    test('A stone and not a territory on 5x5 board', () => {
      const board = ['  B  ', ' B B ', 'B W B', ' W W ', '  W  '];
      const goCounting = new GoCounting(board);
      const expectedTerritory = { owner: 'NONE', territory: [] };
      expect(goCounting.getTerritory(1, 1)).toEqual(expectedTerritory);
    });

    test('Invalid because Y is too low for 5x5 board', () => {
      const board = ['  B  ', ' B B ', 'B W B', ' W W ', '  W  '];
      const goCounting = new GoCounting(board);
      const expectedTerritory = { error: 'Invalid coordinate' };
      expect(goCounting.getTerritory(1, -1)).toEqual(expectedTerritory);
    });
  });

  describe('getTerritories', () => {
    test('One territory is the whole board', () => {
      const board = [' '];
      const goCounting = new GoCounting(board);
      const expectedTerritories = {
        territoryBlack: [],
        territoryWhite: [],
        territoryNone: [[0, 0]],
      };
      expect(goCounting.getTerritories()).toEqual(expectedTerritories);
    });
  });
});
