import { describe, expect, test, xtest } from '@jest/globals';
import { RestAPI } from './rest-api';

describe('Rest API', () => {
  describe('user management', () => {
    test('no users', () => {
      const restAPI = new RestAPI({ users: [] });
      expect(restAPI.get('/users')).toEqual({ users: [] });
    });
  });

  describe('iou', () => {
    test('both users have 0 balance', () => {
      const seedUsers = [
        { name: 'Adam', owes: {}, owed_by: {}, balance: 0 },
        { name: 'Bob', owes: {}, owed_by: {}, balance: 0 },
      ];
      const restAPI = new RestAPI({ users: seedUsers });
      const payload = { lender: 'Adam', borrower: 'Bob', amount: 3 };
      const expectedUsers = [
        { name: 'Adam', owes: {}, owed_by: { Bob: 3 }, balance: 3 },
        { name: 'Bob', owes: { Adam: 3 }, owed_by: {}, balance: -3 },
      ];
      expect(restAPI.post('/iou', payload)).toEqual({ users: expectedUsers });
    });

    test('lender owes borrower', () => {
      const seedUsers = [
        { name: 'Adam', owes: { Bob: 3 }, owed_by: {}, balance: -3 },
        { name: 'Bob', owes: {}, owed_by: { Adam: 3 }, balance: 3 },
      ];
      const restAPI = new RestAPI({ users: seedUsers });
      const payload = { lender: 'Adam', borrower: 'Bob', amount: 2 };
      const expectedUsers = [
        { name: 'Adam', owes: { Bob: 1 }, owed_by: {}, balance: -1 },
        { name: 'Bob', owes: {}, owed_by: { Adam: 1 }, balance: 1 },
      ];
      expect(restAPI.post('/iou', payload)).toEqual({ users: expectedUsers });
    });
  });
});
