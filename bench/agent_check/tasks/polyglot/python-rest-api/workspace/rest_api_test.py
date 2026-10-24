# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/rest-api/canonical-data.json
# File last updated on 2023-07-19

import json
import unittest

from rest_api import (
    RestAPI,
)


class RestApiTest(unittest.TestCase):
    def test_no_users(self):
        database = {"users": []}
        api = RestAPI(database)

        response = api.get("/users")
        expected = {"users": []}
        self.assertDictEqual(json.loads(response), expected)

    def test_both_users_have_0_balance(self):
        database = {
            "users": [
                {"name": "Adam", "owes": {}, "owed_by": {}, "balance": 0.0},
                {"name": "Bob", "owes": {}, "owed_by": {}, "balance": 0.0},
            ]
        }
        api = RestAPI(database)
        payload = json.dumps({"lender": "Adam", "borrower": "Bob", "amount": 3.0})
        response = api.post("/iou", payload)
        expected = {
            "users": [
                {"name": "Adam", "owes": {}, "owed_by": {"Bob": 3.0}, "balance": 3.0},
                {"name": "Bob", "owes": {"Adam": 3.0}, "owed_by": {}, "balance": -3.0},
            ]
        }
        self.assertDictEqual(json.loads(response), expected)

    def test_lender_owes_borrower(self):
        database = {
            "users": [
                {"name": "Adam", "owes": {"Bob": 3.0}, "owed_by": {}, "balance": -3.0},
                {"name": "Bob", "owes": {}, "owed_by": {"Adam": 3.0}, "balance": 3.0},
            ]
        }
        api = RestAPI(database)
        payload = json.dumps({"lender": "Adam", "borrower": "Bob", "amount": 2.0})
        response = api.post("/iou", payload)
        expected = {
            "users": [
                {"name": "Adam", "owes": {"Bob": 1.0}, "owed_by": {}, "balance": -1.0},
                {"name": "Bob", "owes": {}, "owed_by": {"Adam": 1.0}, "balance": 1.0},
            ]
        }
        self.assertDictEqual(json.loads(response), expected)
